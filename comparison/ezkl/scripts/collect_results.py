#!/usr/bin/env python3
"""Collect native-verifier-backed ezkl cells into the canonical matrix summary.

The default request is the strict 5x2 matrix.  Explicit dataset/backend
selectors produce a clearly marked incomplete report and never make it
formal-experiment eligible.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
from pathlib import Path

from resource_policy import (
    CEREMONY_DESCRIPTOR,
    CEREMONY_SRS,
    DATASET_ORDER,
    FORMAL_ROUTE,
    ROOT,
    atomic_write_json,
    atomic_write_text,
    canonical_cell_dir,
    resolve_local,
    safe_mkdir,
    validate_ceremony_material,
    validate_cuda_backend_material,
    validate_cell_inventory,
    validate_results_tree,
)
from build_identity_contract import identity_digest, validate_build_identity  # noqa: E402
from formal_source_manifest import formal_source_id  # noqa: E402


RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _manifest(
    cell: Path,
    *,
    ceremony: dict[str, object] | None = None,
) -> dict[str, object]:
    ceremony = ceremony or validate_ceremony_material(ROOT)
    validate_cell_inventory(cell)
    path = cell / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {path}")
    if payload.get("status") != "verified" or payload.get("verify_passed") is not True:
        raise ValueError(f"cell is not independently verified: {path}")
    if payload.get("route_id") != FORMAL_ROUTE:
        raise ValueError(f"formal route mismatch: {path}")
    if payload.get("manifest_path") != str(path.relative_to(ROOT)):
        raise ValueError(f"manifest path provenance mismatch: {path}")
    expected_prover = "cuda_icicle" if payload.get("device") == "cuda" else "cpu"
    if payload.get("prover_backend") != expected_prover:
        raise ValueError(f"prover backend identity mismatch: {path}")
    if payload.get("verifier_backend") != "cpu_independent":
        raise ValueError(f"verifier backend is not independent CPU: {path}")
    expected_paths = {
        "config": resolve_local(ROOT, Path("configs") / f"{payload.get('dataset', '')}.cfg", "config")
        if payload.get("dataset") not in DATASET_ORDER
        else resolve_local(
            ROOT,
            Path("configs") / {
                "cora_full": "cora_full.cfg",
                "citeseer_full": "citeseer_full.cfg",
                "pubmed_full": "pubmed_full.cfg",
                "ppi_full_formal": "ppi_full.cfg",
                "ogbn_arxiv_full": "ogbn_arxiv_full.cfg",
            }[str(payload["dataset"])],
            "config",
        ),
        "dataset_root": resolve_local(
            ROOT,
            Path("data/cache") / {
                "cora_full": "cora",
                "citeseer_full": "citeseer",
                "pubmed_full": "pubmed",
                "ppi_full_formal": "ppi",
                "ogbn_arxiv_full": "ogbn_arxiv",
            }.get(str(payload.get("dataset")), ""),
            "dataset_root",
        ),
        "checkpoint": resolve_local(
            ROOT,
            Path("artifacts/checkpoints") / {
                "cora_full": "cora_gat",
                "citeseer_full": "citeseer_gat",
                "pubmed_full": "pubmed_gat",
                "ppi_full_formal": "ppi_gat",
                "ogbn_arxiv_full": "ogbn_arxiv_gat",
            }.get(str(payload.get("dataset")), ""),
            "checkpoint",
        ),
        "srs_path": resolve_local(ROOT, CEREMONY_SRS, "ceremony SRS"),
        "ceremony_descriptor": resolve_local(
            ROOT, CEREMONY_DESCRIPTOR, "ceremony descriptor"
        ),
    }
    for field, expected_path in expected_paths.items():
        raw = payload.get(field)
        if not isinstance(raw, str):
            raise ValueError(f"{field} provenance is missing: {path}")
        try:
            actual_path = resolve_local(ROOT, raw, field)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{field} provenance escapes baseline: {path}") from exc
        if actual_path != expected_path:
            raise ValueError(f"{field} provenance is not canonical: {path}")
    if payload.get("source_srs_sha256") not in (None, ceremony["source_srs_sha256"]):
        raise ValueError(f"source SRS digest is not the staged ceremony digest: {path}")
    if payload.get("source_srs_k") not in (None, ceremony["source_srs_max_k"]):
        raise ValueError(f"source SRS k is not the formal ceremony k=28: {path}")
    try:
        security = json.loads((cell / "security_profile.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cell security profile: {path}") from exc
    if not isinstance(security, dict) or any(
        security.get(field) != value
        for field, value in {
            "ceremony_descriptor_kind": ceremony["descriptor_kind"],
            "ceremony_id": ceremony["ceremony_id"],
            "source_srs_sha256": ceremony["source_srs_sha256"],
            "source_srs_max_k": ceremony["source_srs_max_k"],
        }.items()
    ):
        raise ValueError(f"security profile ceremony identity mismatch: {path}")
    device = payload.get("device")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"manifest device is invalid: {path}")
    expected_cuda_backend = (
        validate_cuda_backend_material(ROOT)
        if device == "cuda"
        else {"status": "not_applicable"}
    )
    if payload.get("cuda_backend_provenance") != expected_cuda_backend:
        raise ValueError(f"CUDA backend provenance is stale or missing: {path}")
    expected_source_id = formal_source_id(ROOT)
    if payload.get("formal_semantic_source_id") != expected_source_id:
        raise ValueError(f"formal source identity is stale: {path}")
    expected_prover_binary = ROOT / (
        "target-gpu/release/ezkl_formal" if device == "cuda" else "target/release/ezkl_formal"
    )
    verifier_binary = ROOT / "target/release/ezkl_formal"
    adapter = ROOT / "build_gat_adapter" / "gat_ezkl_export"
    prover_identity_path = ROOT / "build_identity" / f"{device}.json"
    verifier_identity_path = ROOT / "build_identity" / "cpu.json"
    prover_identity = validate_build_identity(
        ROOT, prover_identity_path, backend=str(device),
        binary=expected_prover_binary, adapter=adapter,
    )
    verifier_identity = validate_build_identity(
        ROOT, verifier_identity_path, backend="cpu",
        binary=verifier_binary, adapter=adapter,
    )
    if payload.get("formal_source_manifest") != prover_identity["formal_source_manifest"]:
        raise ValueError(f"formal source inventory mismatch: {path}")
    if payload.get("prover_build_identity_sha256") != identity_digest(prover_identity_path):
        raise ValueError(f"prover build identity digest mismatch: {path}")
    if payload.get("verifier_build_identity_sha256") != identity_digest(verifier_identity_path):
        raise ValueError(f"verifier build identity digest mismatch: {path}")
    for prefix, identity in (("prover", prover_identity), ("verifier", verifier_identity)):
        if payload.get(f"{prefix}_formal_semantic_source_id") != identity["formal_semantic_source_id"]:
            raise ValueError(f"{prefix} semantic source binding mismatch: {path}")
        if payload.get(f"{prefix}_source_tree_digest") != identity["source_tree_digest"]:
            raise ValueError(f"{prefix} source tree binding mismatch: {path}")
        if payload.get(f"{prefix}_dependency_lock_digest") != identity["dependency_lock_digest"]:
            raise ValueError(f"{prefix} dependency lock binding mismatch: {path}")
    for field, name in (
        ("proof_sha256", "proof.json"),
        ("public_statement_sha256", "public_statement.json"),
        ("compiled_circuit_sha256", "network.compiled"),
        ("vk_sha256", "vk.key"),
    ):
        if payload.get(field) != sha256(cell / name):
            raise ValueError(f"{field} mismatch: {path}")
    return payload


def selected_cells(
    datasets: tuple[str, ...] | list[str] = DATASET_ORDER,
    devices: tuple[str, ...] | list[str] = ("cpu", "cuda"),
) -> set[tuple[str, str]]:
    """Return the explicitly requested matrix cells.

    The collector always reports the canonical 5x2 inventory, but only the
    selected cells are required for a partial/resume report.  This keeps a
    one-backend run from being mistaken for a complete experiment while still
    preserving already verified cells in the matrix summary.
    """

    unknown_datasets = sorted(set(datasets) - set(DATASET_ORDER))
    unknown_devices = sorted(set(devices) - {"cpu", "cuda"})
    if unknown_datasets:
        raise ValueError(f"unsupported formal dataset(s): {unknown_datasets}")
    if unknown_devices:
        raise ValueError(f"unsupported formal backend(s): {unknown_devices}")
    return {(dataset, device) for dataset in datasets for device in devices}


def _collect(
    *,
    allow_incomplete: bool,
    datasets: tuple[str, ...] = DATASET_ORDER,
    devices: tuple[str, ...] = ("cpu", "cuda"),
) -> tuple[list[dict[str, object]], list[str]]:
    # Validate the namespace before inspecting any selected manifest.  This
    # rejects unknown/extra result directories and unsafe entries even for a
    # partial collection, while ``require_complete`` remains a matrix-level
    # concern.
    validate_results_tree(ROOT, require_complete=not allow_incomplete)
    ceremony = validate_ceremony_material(ROOT)
    required_cells = selected_cells(datasets, devices)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for backend in ("cpu", "cuda"):
        for dataset in DATASET_ORDER:
            cell = canonical_cell_dir(ROOT, backend, dataset)
            try:
                payload = _manifest(cell, ceremony=ceremony)
                if payload.get("dataset") != dataset or payload.get("device") != backend:
                    raise ValueError("manifest dataset/backend does not match canonical cell")
                timings = payload.get("timings_ms", {})
                if not isinstance(timings, dict):
                    raise ValueError("manifest timings_ms is not an object")
                rows.append(
                    {
                        "dataset": dataset,
                        "backend": backend,
                        "route_id": FORMAL_ROUTE,
                        "status": "verified",
                        "proof_generated": True,
                        "verified": True,
                        "formal_experiment_eligible": True,
                        "setup_ms": timings.get("setup"),
                        "gen_witness_ms": timings.get("gen_witness"),
                        "prove_ms": timings.get("prove"),
                        "verify_ms": timings.get("verify"),
                        "proof_size_bytes": payload.get("proof_size_bytes"),
                        "proof_sha256": payload.get("proof_sha256"),
                        "public_statement_sha256": payload.get("public_statement_sha256"),
                        "compiled_circuit_sha256": payload.get("compiled_circuit_sha256"),
                        "vk_sha256": payload.get("vk_sha256"),
                        "runtime_device_marker": payload.get("runtime_device_marker"),
                        "prover_backend": payload.get("prover_backend"),
                        "verifier_backend": payload.get("verifier_backend"),
                        "formal_semantic_source_id": payload.get("formal_semantic_source_id"),
                        "formal_source_manifest_count": payload.get("formal_source_manifest_count"),
                        "prover_build_identity_sha256": payload.get("prover_build_identity_sha256"),
                        "verifier_build_identity_sha256": payload.get("verifier_build_identity_sha256"),
                        "prover_source_tree_digest": payload.get("prover_source_tree_digest"),
                        "verifier_source_tree_digest": payload.get("verifier_source_tree_digest"),
                        "verification_runtime_marker": payload.get("verification_runtime_marker"),
                        "gpu_backend_execution": payload.get("gpu_backend_execution"),
                        "gpu_prove_telemetry": payload.get("gpu_prove_telemetry"),
                        "cuda_backend_provenance": payload.get("cuda_backend_provenance"),
                        "source_srs_sha256": payload.get("source_srs_sha256"),
                        "source_srs_k": payload.get("source_srs_k"),
                        "downsized_srs_k": payload.get("downsized_srs_k"),
                        "ceremony_id": ceremony["ceremony_id"],
                        "ceremony_descriptor_sha256": ceremony["ceremony_descriptor_sha256"],
                        "proving_srs_sha256": payload.get("proving_srs_sha256"),
                        "verification_srs_sha256": payload.get("verification_srs_sha256"),
                        "manifest": str((cell / "run_manifest.json").relative_to(ROOT)),
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "backend": backend,
                        "status": "not_run" if not cell.exists() else "invalid",
                        "proof_generated": False,
                        "verified": False,
                        "formal_experiment_eligible": False,
                        "manifest": str((cell / "run_manifest.json").relative_to(ROOT)),
                    }
                )
                if (dataset, backend) in required_cells:
                    errors.append(f"{backend}/{dataset}: {exc}")
    if errors and not allow_incomplete:
        raise RuntimeError("formal 5x2 matrix is incomplete or invalid:\n" + "\n".join(errors))
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER,
                        default=list(DATASET_ORDER),
                        help="cells to require; omitted means all five datasets")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"),
                        default=["cpu", "cuda"],
                        help="backends to require; omitted means CPU and CUDA")
    args = parser.parse_args()
    datasets = tuple(args.datasets)
    devices = tuple(args.devices)
    full_matrix = selected_cells(datasets, devices) == selected_cells()
    # A selected subset is intentionally a partial report.  Strict collection
    # is reserved for an explicitly complete 5x2 request.
    allow_partial = args.allow_incomplete or not full_matrix
    try:
        rows, errors = _collect(
            allow_incomplete=allow_partial,
            datasets=datasets,
            devices=devices,
        )
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2
    document = {
        "result_kind": "ezkl_formal_gat_results_v2",
        "baseline": "ezkl",
        "route_id": FORMAL_ROUTE,
        "status": "complete" if full_matrix and not errors else "incomplete",
        "formal_experiment_eligible": full_matrix and not errors,
        "errors": errors,
        "matrix": {"datasets": list(DATASET_ORDER), "backends": ["cpu", "cuda"]},
        "requested_matrix": {"datasets": list(datasets), "backends": list(devices)},
        "verified_runs": rows,
    }
    safe_mkdir(ROOT, RESULTS, "formal results root")
    atomic_write_json(RESULTS / "formal_results.json", document)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["dataset", "backend", "status"]
    csv_path = RESULTS / "formal_results.csv"
    import io

    rendered = io.StringIO(newline="")
    writer = csv.DictWriter(
        rendered, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(csv_path, rendered.getvalue())
    print(json.dumps({"status": document["status"], "verified_runs": sum(row.get("verified") is True for row in rows), "errors": len(errors)}, sort_keys=True))
    return 0 if allow_partial or not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
