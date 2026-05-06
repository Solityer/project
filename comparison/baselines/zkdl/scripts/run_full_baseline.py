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

MAIN_REPO_METHOD_ROWS = [
    ("config parsing", "src/util/config.cpp::load_config", "src/zkdl_gat/baseline.cpp::load_dataset_config", "rewritten", "same formal keys are required by the zkDL cfg files; no main src/include is linked"),
    ("dataset cache reading", "src/data/loader.cpp::load_dataset/load_cached_dataset", "src/zkdl_gat/baseline.cpp::load_dataset_meta/load_feature_matrix/load_edges", "rewritten", "reads /home/pzh/project/data/cache/* directly in the same cache layout"),
    ("feature/edge/label", "src/data/loader.cpp::load_features_matrix/load_labels/edges parser", "src/zkdl_gat/baseline.cpp::validate_feature_source/validate_label_source/load_edges", "ported", "feature shape, label count, and 2/3-column edge formats are checked before proving"),
    ("PPI batch", "src/data/loader.cpp::selected_graph_ids/normalize_graph_input", "src/zkdl_gat/baseline.cpp::normalize_public_input", "ported", "batch_graphs=295 and node_ptr from cache_node_ptr are consumed for ppi_full_formal"),
    ("ogbn_arxiv", "scripts/prepare_ogbn_arxiv.py output + src/data/loader.cpp npy path", "src/zkdl_gat/baseline.cpp::load_feature_matrix", "ported", "features.npy float32 path is supported without calling main scripts"),
    ("self-loop", "src/data/loader.cpp::normalize_graph_input self-loop insertion", "src/zkdl_gat/baseline.cpp::normalize_public_input", "ported", "adds one per-node self-loop after local remapping with deduplication"),
    ("edge sort", "src/data/loader.cpp stable graph_id,dst sort", "src/zkdl_gat/baseline.cpp::normalize_public_input", "ported", "stable_sort by graph_id then dst then stable_index"),
    ("node_ptr/edge_ptr", "src/data/loader.cpp::normalize_graph_input", "src/zkdl_gat/baseline.cpp::normalize_public_input", "ported", "manifest gate compares node_ptr and edge_ptr against main run_manifest"),
    ("checkpoint", "src/model/gat.cpp::inspect/load_checkpoint_bundle_parameters", "src/zkdl_gat/baseline.cpp::load_model_params/load_attention_head", "ported", "manifest.json and tensors.txt are parsed by tensor name and shape"),
    ("quant config", "src/util/config.cpp + checkpoint manifest quant_cfg_id", "src/zkdl_gat/baseline.cpp::fixed_point_policy/build_manifest", "ported", "range_bits/quant_cfg_id are gate-checked; fixed-point scale and signed encoding are recorded"),
    ("GAT forward", "src/model/gat.cpp::forward_note_style", "src/zkdl_gat/baseline.cpp::compute_full_forward", "ported", "hidden heads, concat, output attention, and output head average are computed locally"),
    ("output bias/head average", "src/model/gat.cpp::attention_head_forward_impl/average_matrices", "src/zkdl_gat/baseline.cpp::compute_head_forward/compute_full_forward", "ported", "output heads add bias before arithmetic average; hidden heads do not add output bias"),
    ("public input", "src/protocol/challenges.cpp and run_manifest public topology fields", "src/zkdl_gat/baseline.cpp::hash_normalized_public_input/build_manifest", "rewritten", "public topology hash is derived after gate-equivalent node/edge normalization"),
    ("final logits", "src/model/gat.cpp::forward_note_style", "src/zkdl_gat/baseline.cpp::hash_final_logits/run_full_gat_semantic_precheck", "rewritten", "semantic precheck hashes locally derived final logits; not accepted as proof"),
    ("metric scope", "src/main.cpp metrics and src/protocol/prover.cpp::proof_size_bytes", "scripts/run_full_baseline.py + src/zkdl_gat/baseline.cpp metrics gating", "rewritten", "formal four metrics remain null unless status=verified with real zkDL proof/verifier"),
]

ZKDL_PROOF_ROWS = [
    ("proof object", "src/zkdl_gat/baseline.cpp::build_zkdl_native_proof_object", "full_gat_forward.zkdl_proof.json", "native proof artifact is generated separately from relation diagnostics", "implemented"),
    ("commitment", "commitment.cuh::Commitment::commit + portable adapter", "tensor commitments", "verifier checks committed labels and opening claims", "implemented"),
    ("opening", "commitment.cuh::Commitment::open/me_open + portable opening records", "MLE opening claims", "verifier checks opening digest consistency for transported proof bytes", "implemented"),
    ("transcript", "proof.cuh sumcheck challenges + native transcript root", "Fiat-Shamir transcript values", "verifier recomputes transcript root and hashes proof artifact bytes", "implemented"),
    ("sumcheck", "proof.cuh::inner_product_sumcheck/hadamard_product_sumcheck/binary_sumcheck; zkmatmul.cuh; zkprod.cuh; zkrelu.cuh", "dense/product/binary/ReLU-style primitive sections", "composed verifier walks every full-GAT section", "implemented"),
    ("lookup/range", "zkrelu.cuh binary auxiliary + native lookup/range sections", "LeakyReLU/ELU/Exp and rescale range requirements", "verifier fails at the concrete native section if a lookup/range claim is invalid", "implemented"),
    ("inverse/binding", "native scaled inverse and final logits binding sections", "softmax denominator inverse and public output binding", "verifier checks section residuals and final logits public binding", "implemented"),
    ("verifier acceptance", "src/zkdl_gat/baseline.cpp::verify_zkdl_native_full_gat_proof", "composed full GAT statement", "native zkDL verifier is called before status is decided", "implemented"),
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


def md_rows_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def equivalence_rows(cpu_rows):
    rows = []
    for row in cpu_rows:
        stats = row.get("dataset_stats", {})
        layer_head = f"L={stats.get('layer_count', '')}; hidden={stats.get('hidden_heads_per_layer', '')}; K_out={stats.get('output_heads', '')}"
        rows.append((
            row["dataset"],
            stats.get("node_count", ""),
            stats.get("edge_count", ""),
            stats.get("feature_dim", ""),
            stats.get("class_count", ""),
            layer_head,
            stats.get("checkpoint_tensors_sha256", ""),
            stats.get("quant_cfg_id", ""),
            stats.get("public_input_hash", ""),
            stats.get("final_logits_shape", ""),
            stats.get("final_logits_hash", ""),
            stats.get("main_manifest_match", str(row.get("full_dataset_gate_passed", ""))),
        ))
    return rows


def main_rows_by_dataset(path):
    payload = load_json(path)
    return {row["dataset"]: row for row in payload["rows"]}


def route_result_row(manifest):
    return {
        "dataset": manifest["dataset"],
        "route_id": manifest["route_id"],
        "backend": manifest["backend"],
        "status": manifest["status"],
        "proof_backend": manifest["proof_backend"],
        "verifier_backend": manifest["verifier_backend"],
        "full_dataset": manifest["full_dataset"],
        "full_dataset_gate_passed": manifest["full_dataset_gate_passed"],
        "semantic_precheck_passed": manifest["semantic_precheck_passed"],
        "zkdl_proof_verified": manifest["zkdl_proof_verified"],
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
        "zkdl_proof_backend": manifest["proof_backend"],
        "zkdl_verifier_backend": manifest["verifier_backend"],
        "full_dataset_gate_passed": manifest["full_dataset_gate_passed"],
        "semantic_precheck_passed": manifest["semantic_precheck_passed"],
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
        legacy = root / "runs" / dataset_id / route / "zkdl_gat_composed_proof.json"
        if legacy.exists():
            legacy.unlink()
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
        "dataset", "route_id", "backend", "status", "proof_backend", "verifier_backend",
        "full_dataset", "full_dataset_gate_passed", "semantic_precheck_passed", "zkdl_proof_verified",
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
        "zkdl_proof_backend", "zkdl_verifier_backend", "full_dataset_gate_passed", "semantic_precheck_passed",
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
    write_markdown_reports(root, results, cpu_rows, gpu_rows, cpu_cmp, gpu_cmp, generated_at, cpu_manifests)


def write_markdown_reports(root, results, cpu_rows, gpu_rows, cpu_cmp, gpu_cmp, generated_at, cpu_manifests):
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
        "`semantic_precheck_passed=true` only means the baseline recomputed the full-dataset forward and local semantic checker accepted it. It is not a zkDL proof and never contributes to `proof_size_bytes`.",
        "",
        "CPU runs also emit `runs/<dataset>/cpu/debug_all_relations.json`, an all-relations diagnostic covering the full GAT relation payload in one pass. That diagnostic is not a proof and is not used for `status=verified`.",
        "",
        "Relation payload files named `full_gat_forward.zkdl_relation_payload.json` are deliberately typed as diagnostics. CPU proof attempts now emit a separate `full_gat_forward.zkdl_proof.json` artifact and call the native zkDL verifier before status is decided.",
        "",
        "Formal commitment/prove/verify/proof-size metrics are populated only for `status=verified`. Failed proof-attempt timings are debug fields in each run manifest and are excluded from the comparison tables.",
        "",
        "## Main Repo Method Porting Map",
        "",
        md_rows_table(
            ["item", "main_repo_reference", "zkdl_baseline_file", "status", "evidence"],
            MAIN_REPO_METHOD_ROWS,
        ),
        "",
        "## zkDL Proof Retention Map",
        "",
        md_rows_table(
            ["item", "zkdl_source_file_or_api", "proof_section", "verifier_check", "status"],
            ZKDL_PROOF_ROWS,
        ),
        "",
        "## Equivalence Map",
        "",
        md_rows_table(
            [
                "dataset", "N", "E", "feature_dim", "class_count", "layer/head config",
                "checkpoint hash", "quant_cfg_id", "public_input_hash",
                "final_logits_shape", "final_logits_hash", "main_manifest_match",
            ],
            equivalence_rows(cpu_manifests),
        ),
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
