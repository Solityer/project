from __future__ import annotations

import csv
import pathlib
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .backend import backend_state
from .config import generated_config, load_app_config
from .constants import DATASET_ORDER, PROOF_BACKEND_ID, PROTOCOL_ID
from .data import load_dataset, normalize_graph
from .forward import forward_full
from .model import load_model
from .proof import artifact_ids, build_proof
from .util import digest_json, ensure_dir, write_json


def selected_datasets(value: str) -> List[str]:
    if value == "all":
        return DATASET_ORDER[:]
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(out) - set(DATASET_ORDER))
    if bad:
        raise ValueError(f"unsupported datasets: {bad}")
    return out


def selected_backends(value: str) -> List[str]:
    if value == "all":
        return ["cpu", "gpu"]
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(out) - {"cpu", "gpu"})
    if bad:
        raise ValueError(f"unsupported backends: {bad}")
    return out


def run_one(
    dataset_name: str,
    backend: str,
    project_root: pathlib.Path,
    zk_root: pathlib.Path,
    results_dir: pathlib.Path,
    force: bool,
) -> Dict[str, Any]:
    config_path = generated_config(dataset_name, backend, project_root, zk_root)
    config = load_app_config(config_path, project_root)
    artifact_dir = results_dir / backend / dataset_name
    if artifact_dir.exists() and force:
        shutil.rmtree(artifact_dir)
    ensure_dir(artifact_dir)
    started = time.perf_counter()
    dataset = load_dataset(config)
    graph = normalize_graph(dataset, config)
    if graph.num_nodes != dataset.num_nodes:
        raise RuntimeError(f"{dataset_name}/{backend} is not full dataset")
    model = load_model(config.checkpoint_bundle)
    gpu_seed = {
        "dataset": dataset_name,
        "backend": backend,
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "checkpoint": str(config.checkpoint_bundle),
    }
    gpu_info = backend_state(backend, zk_root, gpu_seed)
    forward_started = time.perf_counter()
    trace = forward_full(graph, model)
    forward_ms = (time.perf_counter() - forward_started) * 1000.0
    ids = artifact_ids(config, dataset)
    paths = {"artifact_dir": artifact_dir}
    proof = build_proof(config, dataset, graph, model, trace, backend, gpu_info, paths)
    if proof.verification["status"] != "VERIFY_OK":
        raise RuntimeError(f"{dataset_name}/{backend} verifier rejected: {proof.verification}")

    write_json(artifact_dir / "metadata.json", proof.metadata)
    write_json(artifact_dir / "commitments.json", proof.commitments)
    write_json(artifact_dir / "transcript.json", proof.transcript)
    write_json(artifact_dir / "challenges.json", proof.transcript)
    write_json(artifact_dir / "proof_payload.json", proof.proof_payload)
    write_json(artifact_dir / "proof_object.json", proof.proof_payload)
    write_json(artifact_dir / "verification.json", proof.verification)
    write_json(artifact_dir / "verifier_output.json", proof.verification)
    write_json(artifact_dir / "witness_manifest.json", proof.witness_manifest)
    write_json(artifact_dir / "gpu_backend.json", gpu_info)
    write_json(artifact_dir / "data_id.json", ids["data_id"])
    write_json(artifact_dir / "model_id.json", ids["model_id"])
    write_json(artifact_dir / "config_id.json", ids["config_id"])
    benchmark = dict(proof.benchmark)
    benchmark["forward_time_ms"] = forward_ms
    benchmark["wall_time_ms"] = (time.perf_counter() - started) * 1000.0
    write_json(artifact_dir / "benchmark_summary.json", benchmark)
    manifest = {
        "status": "VERIFY_OK",
        "dataset": dataset_name,
        "dataset_display_name": dataset.name,
        "backend": backend,
        "protocol_id": PROTOCOL_ID,
        "proof_backend_id": PROOF_BACKEND_ID,
        "config": str(config_path),
        "artifact_path": str(artifact_dir),
        "metadata_path": str(artifact_dir / "metadata.json"),
        "transcript_path": str(artifact_dir / "transcript.json"),
        "proof_payload_path": str(artifact_dir / "proof_payload.json"),
        "verification_path": str(artifact_dir / "verification.json"),
        "witness_manifest_path": str(artifact_dir / "witness_manifest.json"),
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "G_batch": graph.graph_count,
        "node_ptr": graph.node_ptr.tolist(),
        "edge_ptr": graph.edge_ptr.tolist(),
        "full_dataset": True,
        "gpu_backend": gpu_info,
        "proof_size_bytes_accounting": "len(canonical_json(proof_payload)); witness materialization and audit JSON are excluded",
        "run_digest": digest_json({"metadata": proof.metadata, "commitments": proof.commitments, "proof": proof.proof_payload}),
    }
    write_json(artifact_dir / "run_manifest.json", manifest)
    row = {
        "dataset": dataset_name,
        "backend": backend,
        "status": "VERIFY_OK",
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "G_batch": graph.graph_count,
        "L": len(model.hidden_layers) + 1,
        "hidden_profile": ",".join(f"{layer.head_count}x{layer.head_dim}" for layer in model.hidden_layers),
        "d_in_profile": ",".join(str(item) for item in model.manifest["d_in_profile"]),
        "K_out": model.k_out,
        "C": model.class_count,
        "commitment_time_ms": benchmark["commitment_time_ms"],
        "prove_time_ms": benchmark["prove_time_ms"],
        "verify_time_ms": benchmark["verify_time_ms"],
        "proof_size_bytes": benchmark["proof_size_bytes"],
        "config_id": ids["config_id"]["id"],
        "model_id": ids["model_id"]["id"],
        "data_id": ids["data_id"]["id"],
        "artifact_path": str(artifact_dir),
    }
    return row


def write_summary(results_dir: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "backend",
        "status",
        "N",
        "E",
        "G_batch",
        "L",
        "hidden_profile",
        "d_in_profile",
        "K_out",
        "C",
        "commitment_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "proof_size_bytes",
        "config_id",
        "model_id",
        "data_id",
        "artifact_path",
    ]
    rows = sorted(rows, key=lambda row: (0 if row["backend"] == "cpu" else 1, DATASET_ORDER.index(row["dataset"])))
    ensure_dir(results_dir)
    for backend in ["cpu", "gpu"]:
        with (results_dir / f"{backend}_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                if row["backend"] == backend:
                    writer.writerow({name: row.get(name, "") for name in columns})
    with (results_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    write_json(
        results_dir / "summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_id": PROTOCOL_ID,
            "proof_backend_id": PROOF_BACKEND_ID,
            "rows": rows,
            "invalid_archive": "archive/invalid_adapter_20260525",
        },
    )
    readme = [
        "# GAT-ZKML-zkGPT Results",
        "",
        "This directory contains only the rerun independent zkGPT-style baseline results.",
        "The previous adapter artifacts were moved to `archive/invalid_adapter_20260525` and are excluded from these tables.",
        "",
        "Official tables:",
        "",
        "- `summary.csv` / `summary.json`",
        "- `cpu_results.csv`",
        "- `gpu_results.csv`",
        "",
        "`proof_size_bytes` is the canonical serialized `proof_payload.json` payload size. Witness manifests and audit metadata are not counted.",
        "",
    ]
    (results_dir / "README_REPRODUCIBILITY.md").write_text("\n".join(readme), encoding="utf-8")


def run_many(
    datasets: List[str],
    backends: List[str],
    project_root: pathlib.Path,
    zk_root: pathlib.Path,
    results_dir: pathlib.Path,
    force: bool,
) -> List[Dict[str, Any]]:
    ensure_dir(zk_root / "runs_tmp")
    rows: List[Dict[str, Any]] = []
    for backend in backends:
        for dataset in datasets:
            print(f"[zkGPT] running {dataset}/{backend}", flush=True)
            row = run_one(dataset, backend, project_root, zk_root, results_dir, force)
            rows.append(row)
            print(
                f"[zkGPT] {dataset}/{backend} VERIFY_OK "
                f"prove_ms={row['prove_time_ms']:.3f} verify_ms={row['verify_time_ms']:.3f} "
                f"proof_size_bytes={row['proof_size_bytes']}",
                flush=True,
            )
    write_summary(results_dir, rows)
    return rows

