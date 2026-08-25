#!/usr/bin/env python3
"""Collect native-verifier-backed ezkl cells into the canonical matrix summary.

The default request is the strict 5x2 matrix.  Explicit dataset/backend
selectors produce a clearly marked incomplete report and never make it
formal-experiment eligible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from resource_policy import (
    DATASET_ORDER,
    FORMAL_ROUTE,
    ROOT,
    atomic_write_json,
    canonical_cell_dir,
    validate_cell_inventory,
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


def _manifest(cell: Path) -> dict[str, object]:
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
    device = payload.get("device")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"manifest device is invalid: {path}")
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
    required_cells = selected_cells(datasets, devices)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for backend in ("cpu", "cuda"):
        for dataset in DATASET_ORDER:
            cell = canonical_cell_dir(ROOT, backend, dataset)
            try:
                payload = _manifest(cell)
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
                        "source_srs_sha256": payload.get("source_srs_sha256"),
                        "source_srs_k": payload.get("source_srs_k"),
                        "downsized_srs_k": payload.get("downsized_srs_k"),
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
    except RuntimeError as exc:
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
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS / "formal_results.json", document)
    fields = list(rows[0]) if rows else ["dataset", "backend", "status"]
    csv_path = RESULTS / "formal_results.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(csv_path)
    print(json.dumps({"status": document["status"], "verified_runs": sum(row.get("verified") is True for row in rows), "errors": len(errors)}, sort_keys=True))
    return 0 if allow_partial or not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
