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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metric(value):
    return "" if value is None else value


def md_table(rows, include_gpu=False):
    headers = ["dataset", "status", "commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"]
    if include_gpu:
        headers.extend(["gpu_backend_active", "cpu_fallback_used"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(metric(row.get(key))) for key in headers) + " |")
    return "\n".join(lines)


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
    write_markdown_reports(root, results, cpu_rows, gpu_rows, cpu_cmp, gpu_cmp, generated_at)


def write_markdown_reports(root, results, cpu_rows, gpu_rows, cpu_cmp, gpu_cmp, generated_at):
    blocked = [
        row for row in cpu_rows + gpu_rows
        if row["status"] != "verified"
    ]
    summary = [
        "# zkDL Full GAT Baseline Summary",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "Correct baseline directory: `comparison/baselines/zkdl`.",
        "",
        "The full-dataset statement is attempted for `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. No row is marked verified unless a real zkDL proof and verifier acceptance path succeeds.",
        "",
        "## CPU",
        "",
        md_table(cpu_rows),
        "",
        "## GPU",
        "",
        md_table(gpu_rows, include_gpu=True),
        "",
        "## Blocked Items",
        "",
        "| dataset | route_id | failure_step | block_reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in blocked:
        summary.append(f"| {row['dataset']} | {row['route_id']} | {row['failure_step']} | {row['block_reason']} |")
    summary.append("")
    (results / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    control = [
        "# Control Variable Audit",
        "",
        "Both CPU and GPU routes use the same five full dataset configs under `comparison/baselines/zkdl/configs` and read the same main-repository data/checkpoint sources in read-only mode.",
        "",
        "| dataset | cpu_status | gpu_status | cpu_public_equivalent | gpu_public_equivalent | cpu_route | gpu_route |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    gpu_by_dataset = {row["dataset"]: row for row in gpu_rows}
    for cpu in cpu_rows:
        gpu = gpu_by_dataset[cpu["dataset"]]
        control.append(
            f"| {cpu['dataset']} | {cpu['status']} | {gpu['status']} | "
            f"{cpu['strict_public_input_equivalent']} | {gpu['strict_public_input_equivalent']} | "
            f"{cpu['route_id']} | {gpu['route_id']} |"
        )
    control.extend([
        "",
        "The GPU primitive probe is recorded only as backend availability evidence. It is not copied into full GAT proof metrics and cannot make a full GAT row verified.",
    ])
    (results / "control_variable_audit.md").write_text("\n".join(control), encoding="utf-8")

    scanned_roots = [root / "include", root / "src", root / "scripts", root / "configs", root / "manifests"]
    forbidden_refs = []
    for scanned_root in scanned_roots:
        for path in scanned_root.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                needles = [
                    str(root.parents[2] / "src"),
                    str(root.parents[2] / "include"),
                    str(root.parents[2] / "scripts"),
                    str(root.parents[1] / "zkdl"),
                ]
                for needle in needles:
                    if needle in text:
                        forbidden_refs.append(f"{path.relative_to(root)} -> {needle}")
    code = [
        "# Code Cleanliness Audit",
        "",
        "Scanned baseline-owned `include`, `src`, `scripts`, `configs`, and `manifests` paths.",
        "",
        f"- forbidden dependency references: `{len(forbidden_refs)}`",
        "- C++ runner owns manifest construction, full-dataset public-input equivalence checks, source hashing, proof-statement domain construction, and zkDL proof-stage gating.",
        "- Python is limited to route orchestration, collection, validation, and report formatting.",
        "",
    ]
    if forbidden_refs:
        code.append("## Findings")
        code.extend(f"- {item}" for item in forbidden_refs)
    else:
        code.append("No forbidden main `src`, `include`, or `scripts` dependency references were found.")
    (results / "code_cleanliness_audit.md").write_text("\n".join(code), encoding="utf-8")

    wrong_dir = root.parents[1] / "zkdl"
    cleanup = [
        "# Cleanup Report",
        "",
        f"- wrong baseline directory exists at report generation: `{wrong_dir.exists()}`",
        "- formal baseline directory: `comparison/baselines/zkdl`",
        "- result and run paths are written only under `comparison/baselines/zkdl`.",
        "- `comparison/zkdl` is not used as a build, run, or result directory by these scripts.",
    ]
    (results / "cleanup_report.md").write_text("\n".join(cleanup), encoding="utf-8")


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
