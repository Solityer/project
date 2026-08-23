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
from .formal_gate import require_formal_experiment
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


def _hardware_and_device(backend: str, gpu_info: Dict[str, Any]) -> str:
    if backend == "gpu":
        return f"gpu:{gpu_info.get('device_name', 'unknown')}"
    return "cpu"


def _gpu_hot_path_status(backend: str, gpu_info: Dict[str, Any]) -> str:
    if backend != "gpu":
        return ""
    if (
        gpu_info.get("proof_hot_path") is True
        and gpu_info.get("cuda_kernel_executed") is True
        and gpu_info.get("cpu_fallback") is False
    ):
        return "GPU_HOT_PATH_BOUND"
    return "GPU_HOT_PATH_INVALID"


def _proof_scope_fields(
    config: Any,
    graph: Any,
    model: Any,
    backend: str,
    gpu_info: Dict[str, Any],
    ids: Dict[str, Any],
) -> Dict[str, Any]:
    model_arch = {
        "model_arch_id": model.model_arch_id,
        "layers": len(model.hidden_layers) + 1,
        "hidden_profile": [
            {"head_count": layer.head_count, "head_dim": layer.head_dim}
            for layer in model.hidden_layers
        ],
        "K_out": model.k_out,
        "C": model.class_count,
        "output_average_rule": model.output_average_rule,
    }
    model_params = {
        "model_param_id": model.model_param_id,
        "checkpoint_bundle": str(config.checkpoint_bundle),
        "model_id": ids["model_id"]["id"],
        "quant_cfg_id": model.quant_cfg_id,
        "static_table_id": model.static_table_id,
    }
    return {
        "baseline_identity": "zkGPT",
        "proof_scope": "full_gat_zkgpt_segment_commitments_lookup_matmul_gkr_openings",
        "scope_mismatch": False,
        "full_gat_forward_proved": True,
        "final_logits_bound": True,
        "setup_assumption": "bn254_hyrax_mle_openmasked_plus_lasso_grand_product_lookup_plus_sumcheck_gkr",
        "proof_size_definition": "canonical serialized proof_payload.json including commitments, lookup proof, matrix sumcheck proof, GKR proof, combine proof, and masked openings; excludes raw witness arrays, datasets, checkpoints, logs, and audit metadata",
        "verifier_reads_witness": False,
        "witness_included_in_proof": False,
        "public_input_layout": "public graph, public features, public metadata, public logits",
        "task_type": graph.task_type,
        "model_arch": model_arch,
        "model_params": model_params,
        "same_checkpoint": True,
        "same_quantization": True,
        "same_graph_preprocessing": True,
        "same_self_loop_rule": True,
        "same_edge_sort_rule": True,
        "same_public_output_policy": True,
        "self_loop_rule": graph.self_loop_rule,
        "edge_sort_rule": graph.edge_sort_rule,
        "batching_rule": graph.batching_rule,
        "hardware_and_device": _hardware_and_device(backend, gpu_info),
        "gpu_hot_path_status": _gpu_hot_path_status(backend, gpu_info),
        "cpu_fallback": "" if backend != "gpu" else bool(gpu_info.get("cpu_fallback", True)),
        "run_status": "VERIFY_OK_FULL_GAT_ZKGPT",
        "failure_reason_if_not_comparable": "",
    }


def run_one(
    dataset_name: str,
    backend: str,
    project_root: pathlib.Path,
    zk_root: pathlib.Path,
    results_dir: pathlib.Path,
    force: bool,
) -> Dict[str, Any]:
    require_formal_experiment()
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
    paths = {"artifact_dir": artifact_dir, "zk_root": zk_root}
    proof = build_proof(config, dataset, graph, model, trace, backend, gpu_info, paths)
    if proof.verification["status"] != "VERIFY_OK":
        raise RuntimeError(f"{dataset_name}/{backend} verifier rejected: {proof.verification}")
    scope_fields = _proof_scope_fields(config, graph, model, backend, gpu_info, ids)

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
        **scope_fields,
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
        "baseline_identity": scope_fields["baseline_identity"],
        "proof_scope": scope_fields["proof_scope"],
        "scope_mismatch": scope_fields["scope_mismatch"],
        "full_gat_forward_proved": scope_fields["full_gat_forward_proved"],
        "final_logits_bound": scope_fields["final_logits_bound"],
        "setup_assumption": scope_fields["setup_assumption"],
        "proof_size_definition": scope_fields["proof_size_definition"],
        "verifier_reads_witness": scope_fields["verifier_reads_witness"],
        "witness_included_in_proof": scope_fields["witness_included_in_proof"],
        "public_input_layout": scope_fields["public_input_layout"],
        "task_type": scope_fields["task_type"],
        "model_arch": digest_json(scope_fields["model_arch"]),
        "model_params": digest_json(scope_fields["model_params"]),
        "same_checkpoint": scope_fields["same_checkpoint"],
        "same_quantization": scope_fields["same_quantization"],
        "same_graph_preprocessing": scope_fields["same_graph_preprocessing"],
        "same_self_loop_rule": scope_fields["same_self_loop_rule"],
        "same_edge_sort_rule": scope_fields["same_edge_sort_rule"],
        "same_public_output_policy": scope_fields["same_public_output_policy"],
        "self_loop_rule": scope_fields["self_loop_rule"],
        "edge_sort_rule": scope_fields["edge_sort_rule"],
        "batching_rule": scope_fields["batching_rule"],
        "hardware_and_device": scope_fields["hardware_and_device"],
        "gpu_hot_path_status": scope_fields["gpu_hot_path_status"],
        "cpu_fallback": scope_fields["cpu_fallback"],
        "run_status": scope_fields["run_status"],
        "failure_reason_if_not_comparable": scope_fields["failure_reason_if_not_comparable"],
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
        "baseline_identity",
        "proof_scope",
        "scope_mismatch",
        "full_gat_forward_proved",
        "final_logits_bound",
        "setup_assumption",
        "proof_size_definition",
        "verifier_reads_witness",
        "witness_included_in_proof",
        "public_input_layout",
        "task_type",
        "model_arch",
        "model_params",
        "same_checkpoint",
        "same_quantization",
        "same_graph_preprocessing",
        "same_self_loop_rule",
        "same_edge_sort_rule",
        "same_public_output_policy",
        "self_loop_rule",
        "edge_sort_rule",
        "batching_rule",
        "hardware_and_device",
        "gpu_hot_path_status",
        "cpu_fallback",
        "run_status",
        "failure_reason_if_not_comparable",
    ]
    rows = sorted(rows, key=lambda row: (0 if row["backend"] == "cpu" else 1, DATASET_ORDER.index(row["dataset"])))
    ensure_dir(results_dir)
    for backend in ["cpu", "gpu"]:
        with (results_dir / f"{backend}_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                if row["backend"] == backend:
                    writer.writerow({name: row.get(name, "") for name in columns})
    with (results_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
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
            "invalid_archives": [
                "archive/invalid_adapter_20260525",
                "archive/invalid_digest_replay_20260529",
            ],
        },
    )
    readme = [
        "# GAT-ZKML-zkGPT Results",
        "",
        "This directory contains the official rerun independent zkGPT-style baseline results.",
        "The previous adapter and digest/replay artifacts live under `archive/` and are excluded from these tables.",
        "",
        "Official tables:",
        "",
        "- `summary.csv` / `summary.json`",
        "- `cpu_results.csv`",
        "- `gpu_results.csv`",
        "",
        "Each dataset/backend directory includes metadata, full chunked segment commitments, transcript, lookup proof, matrix sumcheck proof, GKR proof, Hyrax/MLE openings, masked openings, final proof payload, verifier output, witness manifest, benchmark summary, and run manifest.",
        "",
        "`proof_size_bytes` is the canonical serialized `proof_payload.json` payload size, including verifier-facing proof messages. Witness manifests, raw witness materialization, datasets, checkpoints, logs, and audit metadata are not counted.",
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
    require_formal_experiment()
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
