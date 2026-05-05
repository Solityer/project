#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_KEYS = {
    "dataset",
    "route_id",
    "backend",
    "status",
    "strict_public_input_equivalent",
    "full_gat_forward_in_zkdl_proof",
    "final_logits_derivation_complete",
    "no_precomputed_logits_bypass",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
    "block_reason",
    "failure_step",
}

GPU_KEYS = {
    "gpu_backend_active",
    "gpu_device_name",
    "gpu_device_count",
    "gpu_backend_name",
    "cpu_fallback_used",
}

METRICS = [
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]


def read_dataset_order(root: Path) -> list[str]:
    return [
        line.strip()
        for line in (root / "configs" / "dataset_order.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_manifest(root: Path, dataset: str, route: str, errors: list[str]) -> None:
    manifest_path = root / "runs" / dataset / route / "run_manifest.json"
    require(manifest_path.exists(), f"missing manifest {manifest_path}", errors)
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = REQUIRED_KEYS | (GPU_KEYS if route == "gpu" else set())
    missing = sorted(key for key in keys if key not in manifest)
    require(not missing, f"{manifest_path}: missing keys {missing}", errors)
    require(manifest.get("dataset") == dataset, f"{manifest_path}: dataset mismatch", errors)
    require(manifest.get("backend") == route, f"{manifest_path}: backend mismatch", errors)
    require(
        manifest.get("route_id") == f"zkdl_gat_full_forward_{route}",
        f"{manifest_path}: route_id mismatch",
        errors,
    )
    require(manifest.get("no_precomputed_logits_bypass") is True, f"{manifest_path}: bypass flag is not true", errors)

    status = manifest.get("status")
    require(status in {"verified", "blocked", "failed"}, f"{manifest_path}: invalid status {status}", errors)

    if status == "verified":
        require(manifest.get("strict_public_input_equivalent") is True, f"{manifest_path}: verified without public equivalence", errors)
        require(manifest.get("full_gat_forward_in_zkdl_proof") is True, f"{manifest_path}: verified without full GAT proof", errors)
        require(manifest.get("final_logits_derivation_complete") is True, f"{manifest_path}: verified without final logits derivation", errors)
        require(not manifest.get("block_reason"), f"{manifest_path}: verified row has block_reason", errors)
        for metric in METRICS:
            require(isinstance(manifest.get(metric), (int, float)), f"{manifest_path}: verified metric {metric} is not numeric", errors)
        if route == "gpu":
            require(manifest.get("cpu_fallback_used") is False, f"{manifest_path}: verified GPU used CPU fallback", errors)
            require(manifest.get("gpu_backend_active") is True, f"{manifest_path}: verified GPU backend inactive", errors)
    else:
        require(bool(manifest.get("block_reason")), f"{manifest_path}: blocked/failed row lacks block_reason", errors)
        require(bool(manifest.get("failure_step")), f"{manifest_path}: blocked/failed row lacks failure_step", errors)
        if route == "gpu":
            require(manifest.get("cpu_fallback_used") is False, f"{manifest_path}: GPU route used CPU fallback", errors)

    source_files = manifest.get("source_files", {})
    source_sha256 = manifest.get("source_sha256", {})
    for source_key, source_path in source_files.items():
        path = Path(source_path)
        require(path.exists(), f"{manifest_path}: source file missing {source_path}", errors)
        hash_key = f"{source_key}_sha256"
        expected = source_sha256.get(hash_key)
        require(bool(expected), f"{manifest_path}: missing sha256 for {source_key}", errors)
        if path.exists() and expected:
            actual = sha256_file(path)
            require(actual == expected, f"{manifest_path}: sha256 mismatch for {source_key}", errors)


def validate_results_files(root: Path, route: str, errors: list[str]) -> None:
    expected = [
        root / "results" / f"zkdl_{route}_full_results.json",
        root / "results" / f"zkdl_{route}_full_results.csv",
        root / "results" / f"main_vs_zkdl_{route}_full_comparison.json",
        root / "results" / f"main_vs_zkdl_{route}_full_comparison.csv",
    ]
    for path in expected:
        require(path.exists(), f"missing result file {path}", errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=["cpu", "gpu", "all"], default="all")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    routes = ["cpu", "gpu"] if args.route == "all" else [args.route]
    datasets = read_dataset_order(root)
    errors: list[str] = []
    for route in routes:
        for dataset in datasets:
            validate_manifest(root, dataset, route, errors)
        validate_results_files(root, route, errors)

    if errors:
        for error in errors:
            print(error)
        return 1
    print("validation_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
