#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DATASETS = [
    ("cora_full", "cora"),
    ("citeseer_full", "citeseer"),
    ("pubmed_full", "pubmed"),
    ("ppi_full_formal", "ppi"),
    ("ogbn_arxiv_full", "ogbn-arxiv"),
]


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main_rows_by_dataset(path):
    payload = load_json(path)
    return {row["dataset"]: row for row in payload["rows"]}


def route_result_row(manifest):
    return {
        "dataset": manifest["dataset"],
        "route_id": manifest["route_id"],
        "backend": manifest["backend"],
        "status": manifest["status"],
        "commitment_time_ms": manifest["commitment_time_ms"],
        "prove_time_ms": manifest["prove_time_ms"],
        "verify_time_ms": manifest["verify_time_ms"],
        "proof_size_bytes": manifest["proof_size_bytes"],
        "strict_public_input_equivalent": manifest["strict_public_input_equivalent"],
        "full_gat_forward_in_zkdl_proof": manifest["full_gat_forward_in_zkdl_proof"],
        "final_logits_derivation_complete": manifest["final_logits_derivation_complete"],
        "no_precomputed_logits_bypass": manifest["no_precomputed_logits_bypass"],
        "failure_step": manifest["failure_step"],
        "block_reason": manifest["block_reason"],
        "gpu_backend_active": manifest.get("gpu_backend_active", False),
        "gpu_device_name": manifest.get("gpu_device_name", ""),
        "gpu_device_count": manifest.get("gpu_device_count", 0),
        "gpu_backend_name": manifest.get("gpu_backend_name", ""),
        "cpu_fallback_used": manifest.get("cpu_fallback_used", False),
    }


def comparison_row(manifest, main_row, route):
    return {
        "dataset": manifest["dataset"],
        "main_route": route,
        "zkdl_route_id": manifest["route_id"],
        "zkdl_status": manifest["status"],
        "comparison_status": "verified" if manifest["status"] == "verified" else "blocked",
        "main_commitment_time_ms": main_row["commitment_time_ms"],
        "main_prove_time_ms": main_row["prove_time_ms"],
        "main_verify_time_ms": main_row["verify_time_ms"],
        "main_proof_size_bytes": main_row["proof_size_bytes"],
        "zkdl_commitment_time_ms": manifest["commitment_time_ms"],
        "zkdl_prove_time_ms": manifest["prove_time_ms"],
        "zkdl_verify_time_ms": manifest["verify_time_ms"],
        "zkdl_proof_size_bytes": manifest["proof_size_bytes"],
        "failure_step": manifest["failure_step"],
        "block_reason": manifest["block_reason"],
        "gpu_backend_active": manifest.get("gpu_backend_active", False),
        "cpu_fallback_used": manifest.get("cpu_fallback_used", False),
        "gpu_device_name": manifest.get("gpu_device_name", ""),
    }


def run_route(root, build_dir, route):
    runner = root / build_dir / "zkdl_gat_runner"
    if not runner.exists():
        raise SystemExit(f"runner not found: {runner}")

    if route == "gpu":
        probe = root / build_dir / "zkdl_gat_gpu_probe"
        if probe.exists():
            run([str(probe), "--out", "runs/gpu_probe.json"], root)

    manifests = []
    for dataset_id, _ in DATASETS:
        cfg = root / "configs" / f"{dataset_id}.cfg"
        out = root / "runs" / dataset_id / route / "run_manifest.json"
        run([
            str(runner),
            "--config", str(cfg),
            "--route", route,
            "--out", str(out),
            "--gpu-probe", "runs/gpu_probe.json",
        ], root)
        manifests.append(load_json(out))
    return manifests


def collect(root, cpu_manifests, gpu_manifests):
    results = root / "results"
    generated_at = datetime.now(timezone.utc).isoformat()
    route_fields = [
        "dataset", "route_id", "backend", "status",
        "commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes",
        "strict_public_input_equivalent", "full_gat_forward_in_zkdl_proof",
        "final_logits_derivation_complete", "no_precomputed_logits_bypass",
        "failure_step", "block_reason",
    ]
    gpu_extra = ["gpu_backend_active", "gpu_device_name", "gpu_device_count", "gpu_backend_name", "cpu_fallback_used"]

    cpu_rows = [route_result_row(item) for item in cpu_manifests]
    gpu_rows = [route_result_row(item) for item in gpu_manifests]
    write_json(results / "zkdl_cpu_full_results.json", {
        "generated_at": generated_at,
        "backend": "cpu",
        "route_id": "zkdl_gat_full_forward_cpu",
        "rows": cpu_rows,
    })
    write_json(results / "zkdl_gpu_full_results.json", {
        "generated_at": generated_at,
        "backend": "gpu",
        "route_id": "zkdl_gat_full_forward_gpu",
        "rows": gpu_rows,
    })
    write_csv(results / "zkdl_cpu_full_results.csv", cpu_rows, route_fields)
    write_csv(results / "zkdl_gpu_full_results.csv", gpu_rows, route_fields + gpu_extra)

    main_cpu = main_rows_by_dataset(Path("/home/pzh/project/runs/benchmarks/latest.json"))
    main_gpu = main_rows_by_dataset(Path("/home/pzh/project/runs/benchmarks/cuda_formal/latest.json"))
    name_map = dict(DATASETS)
    cpu_cmp = [comparison_row(item, main_cpu[name_map[item["dataset"]]], "cpu") for item in cpu_manifests]
    gpu_cmp = [comparison_row(item, main_gpu[name_map[item["dataset"]]], "gpu") for item in gpu_manifests]
    cmp_fields = [
        "dataset", "main_route", "zkdl_route_id", "zkdl_status", "comparison_status",
        "main_commitment_time_ms", "main_prove_time_ms", "main_verify_time_ms", "main_proof_size_bytes",
        "zkdl_commitment_time_ms", "zkdl_prove_time_ms", "zkdl_verify_time_ms", "zkdl_proof_size_bytes",
        "failure_step", "block_reason",
    ]
    write_json(results / "main_vs_zkdl_cpu_full_comparison.json", {
        "generated_at": generated_at,
        "route": "cpu",
        "rows": cpu_cmp,
    })
    write_json(results / "main_vs_zkdl_gpu_full_comparison.json", {
        "generated_at": generated_at,
        "route": "gpu",
        "rows": gpu_cmp,
    })
    write_csv(results / "main_vs_zkdl_cpu_full_comparison.csv", cpu_cmp, cmp_fields)
    write_csv(results / "main_vs_zkdl_gpu_full_comparison.csv", gpu_cmp, cmp_fields + ["gpu_backend_active", "cpu_fallback_used", "gpu_device_name"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--cpu-build-dir", default="build")
    parser.add_argument("--gpu-build-dir", default="build_cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cpu_manifests = []
    gpu_manifests = []
    if args.route in ("cpu", "both"):
        cpu_manifests = run_route(root, args.cpu_build_dir, "cpu")
    if args.route in ("gpu", "both"):
        gpu_manifests = run_route(root, args.gpu_build_dir, "gpu")
    if args.route == "cpu":
        gpu_manifests = [load_json(root / "runs" / dataset_id / "gpu" / "run_manifest.json") for dataset_id, _ in DATASETS]
    if args.route == "gpu":
        cpu_manifests = [load_json(root / "runs" / dataset_id / "cpu" / "run_manifest.json") for dataset_id, _ in DATASETS]
    collect(root, cpu_manifests, gpu_manifests)


if __name__ == "__main__":
    main()
