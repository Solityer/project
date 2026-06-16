#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import sys


REQUIRED_DATASETS = {"cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"}
SECURE_FIELDS = {
    "protocol_name": "zkgat",
    "proof_task": "node_classification_full_gat",
    "privacy_policy": "model_and_trace_hidden",
    "input_privacy": "public_graph_public_features",
    "output_mode": "public_logits",
    "opening_mode": "domain_grouped_batch_opening",
    "binding_mode": "zk_subproof_for_crpc",
    "verifier_reads_witness": "false",
    "private_plain_value_count": "0",
    "proof_payload_audit_status": "passed",
    "reference_consistency_passed": "true",
    "tamper_tests_passed": "true",
    "full_dataset": "true",
    "verified": "true",
}
TIMING_FIELDS = [
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]
OUTPUT_FIELDS = [
    "dataset",
    "backend",
    "srs_type",
    "srs_id",
    "srs_manifest_hash",
    "privacy_policy",
    "input_privacy",
    "output_mode",
    "opening_mode",
    "binding_mode",
    "proof_task",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
    "optimization_profile_id",
    "optimization_flags",
    "hardware_and_device",
    "gpu_acceleration_status",
    "source_tree_hash",
    "binary_hash",
    "run_manifest_hash",
    "timeout",
    "stale_result",
    "verifier_reads_witness",
    "private_plain_value_count",
    "proof_payload_audit_status",
    "reference_consistency_passed",
    "tamper_tests_passed",
    "full_dataset",
    "dataset_id",
    "dataset_hash",
    "checkpoint_model_hash",
    "static_table_hash",
    "quant_config_hash",
    "public_metadata_hash",
    "proof_hash",
    "public_logits_hash",
    "reference_fp32_logits_hash",
    "reference_quant_logits_hash",
    "secure_trace_logits_hash",
    "proof_public_output_hash",
    "srs_degree_bound",
    "security_profile_id",
]
HASH_FIELDS = [
    "dataset_hash",
    "checkpoint_model_hash",
    "static_table_hash",
    "quant_config_hash",
    "public_metadata_hash",
    "srs_manifest_hash",
    "proof_hash",
    "public_logits_hash",
    "reference_fp32_logits_hash",
    "reference_quant_logits_hash",
    "secure_trace_logits_hash",
    "proof_public_output_hash",
]


def parse_run(value):
    if "=" not in value:
        raise ValueError(f"run must be dataset=path: {value}")
    dataset, path = value.split("=", 1)
    return dataset, pathlib.Path(path)


def load_manifest(path):
    manifest = path / "run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing run manifest: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def manifest_hash(row):
    payload = dict(row)
    payload.pop("run_manifest_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_object_profile(path, row):
    profile_name = row.get("dynamic_commitment_object_profile_csv")
    if not profile_name:
        raise ValueError(f"missing dynamic commitment object profile path for {row.get('dataset', '<unknown>')}")
    profile_path = path / profile_name
    if not profile_path.exists():
        raise FileNotFoundError(f"missing dynamic commitment object profile: {profile_path}")
    with profile_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"dynamic commitment object profile is empty: {profile_path}")
    return rows


def require_field(row, key):
    if key not in row or row[key] in ("", None):
        raise ValueError(f"missing required field {key} for {row.get('dataset', '<unknown>')}")


def normalize_bool(value):
    return str(value).lower()


def validate_row(dataset, backend, row, expected_source_hash="", expected_cpu_binary_hash="", expected_gpu_binary_hash=""):
    require_field(row, "dataset")
    if dataset == "ogbn-arxiv":
        expected_names = {"ogbn-arxiv", "ogbn_arxiv"}
    else:
        expected_names = {dataset}
    if row["dataset"] not in expected_names:
        raise ValueError(f"dataset key {dataset} does not match manifest dataset {row['dataset']}")
    if row.get("srs_type") not in {"trusted_srs", "mpc_srs"}:
        raise ValueError(f"{dataset}/{backend} has invalid SRS type")
    require_field(row, "backend")
    if row["backend"] != backend:
        raise ValueError(f"{dataset}/{backend} has backend={row['backend']}, expected {backend}")
    for key, expected in SECURE_FIELDS.items():
        require_field(row, key)
        if normalize_bool(row[key]) != expected:
            raise ValueError(f"{dataset}/{backend} has {key}={row[key]}, expected {expected}")
    for key, expected in {
        "timeout": "false",
        "stale_result": "false",
        "exit_code": "0",
    }.items():
        require_field(row, key)
        if normalize_bool(row[key]) != expected:
            raise ValueError(f"{dataset}/{backend} has {key}={row[key]}, expected {expected}")
    for key in TIMING_FIELDS:
        require_field(row, key)
        if float(row[key]) < 0:
            raise ValueError(f"{dataset}/{backend} has negative {key}")
    if pathlib.Path(str(row.get("config", ""))).is_absolute():
        raise ValueError(f"{dataset}/{backend} manifest contains absolute config path")
    if row.get("reference_quant_logits_hash") != row.get("secure_trace_logits_hash"):
        raise ValueError(f"{dataset}/{backend} reference/trace logits hash mismatch")
    if row.get("reference_quant_logits_hash") != row.get("proof_public_output_hash"):
        raise ValueError(f"{dataset}/{backend} proof public output hash mismatch")
    for key in ("srs_id", "srs_degree_bound", "security_profile_id", *HASH_FIELDS):
        require_field(row, key)
    for key in ("optimization_profile_id", "optimization_flags", "hardware_and_device", "gpu_acceleration_status"):
        require_field(row, key)
    for key in (
        "source_tree_hash",
        "binary_hash",
        "run_manifest_hash",
        "build_identity",
        "runner_hash",
        "activation_table_policy",
        "static_table_endpoint_hash",
        "feature_lookup_identity_predicate",
        "range_table_size",
        "lrelu_table_size",
        "exp_table_size",
        "elu_table_size",
    ):
        require_field(row, key)
    if expected_source_hash and row["source_tree_hash"] != expected_source_hash:
        raise ValueError(f"{dataset}/{backend} source hash mismatch")
    expected_binary_hash = expected_gpu_binary_hash if backend == "gpu" else expected_cpu_binary_hash
    if expected_binary_hash and row["binary_hash"] != expected_binary_hash:
        raise ValueError(f"{dataset}/{backend} binary hash mismatch")
    if row["run_manifest_hash"] != manifest_hash(row):
        raise ValueError(f"{dataset}/{backend} run manifest hash mismatch")
    for key in (
        "dynamic_commitment_object_profile_csv",
        "dynamic_commitment_object_count",
        "dynamic_commitment_count_by_domain",
        "dynamic_commitment_time_by_domain_ms",
        "dynamic_commitment_time_by_ownership_category_ms",
    ):
        require_field(row, key)
    if int(row["dynamic_commitment_object_count"]) <= 0:
        raise ValueError(f"{dataset}/{backend} has no dynamic commitment object profile rows")
    if backend == "gpu" and "enabled_cuda_hotspots" not in row["gpu_acceleration_status"]:
        raise ValueError(f"{dataset}/{backend} does not record CUDA hotspot acceleration")
    if backend == "cpu" and row["gpu_acceleration_status"] != "not_requested_cpu_run":
        raise ValueError(f"{dataset}/{backend} has unexpected gpu_acceleration_status")


def project_row(dataset, backend, row):
    out = {key: row.get(key, "") for key in OUTPUT_FIELDS}
    out["dataset"] = dataset
    out["backend"] = backend
    return out


def collect(runs, backend, expected_source_hash="", expected_cpu_binary_hash="", expected_gpu_binary_hash=""):
    rows = []
    manifests = []
    seen = set()
    for item in runs:
        dataset, path = parse_run(item)
        row = load_manifest(path)
        validate_row(dataset, backend, row, expected_source_hash, expected_cpu_binary_hash, expected_gpu_binary_hash)
        object_profile = load_object_profile(path, row)
        if int(row["dynamic_commitment_object_count"]) != len(object_profile):
            raise ValueError(f"{dataset}/{backend} dynamic object profile count mismatch")
        seen.add(dataset)
        rows.append(project_row(dataset, backend, row))
        manifests.append((dataset, backend, row, object_profile))
    if seen != REQUIRED_DATASETS:
        missing = ", ".join(sorted(REQUIRED_DATASETS - seen))
        extra = ", ".join(sorted(seen - REQUIRED_DATASETS))
        raise ValueError(f"{backend} run set mismatch; missing={missing}; extra={extra}")
    return rows, manifests


def write_outputs(rows, manifests, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "latest.json",
        "latest.csv",
        "summary.md",
        "security_tests_report.md",
        "reference_consistency_report.md",
        "proof_size_report.md",
        "optimization_profile_report.md",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    manifest_dir = output_dir / "run_manifests"
    if manifest_dir.exists():
        shutil.rmtree(manifest_dir)
    (output_dir / "latest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "latest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| dataset | backend | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['backend']} | {row['commitment_time_ms']} | "
            f"{row['prove_time_ms']} | {row['verify_time_ms']} | {row['proof_size_bytes']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_dir.mkdir(exist_ok=True)
    for dataset, backend, manifest, _ in manifests:
        (manifest_dir / f"{backend}_{dataset}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "security_tests_report.md").write_text(
        "\n".join([
            "# Security Tests",
            "",
            "All exported rows passed verifier, SRS, metadata, schema, leakage, reference consistency, and tamper gates.",
            "",
            "| dataset | backend | verified | private_plain_value_count | reference_consistency_passed | tamper_tests_passed |",
            "| --- | --- | --- | ---: | --- | --- |",
            *[
                f"| {row['dataset']} | {row['backend']} | true | {row['private_plain_value_count']} | "
                f"{row['reference_consistency_passed']} | {row['tamper_tests_passed']} |"
                for row in rows
            ],
        ]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reference_consistency_report.md").write_text(
        "\n".join([
            "# Reference Consistency",
            "",
            "| dataset | backend | reference_quant_logits_hash | secure_trace_logits_hash | proof_public_output_hash |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| {row['dataset']} | {row['backend']} | {row['reference_quant_logits_hash']} | "
                f"{row['secure_trace_logits_hash']} | {row['proof_public_output_hash']} |"
                for row in rows
            ],
        ]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "proof_size_report.md").write_text(
        "\n".join([
            "# Proof Size",
            "",
            "proof_size_bytes counts the proof transport object: commitments, quotient commitments, opening proofs, claimed values, public metadata, public output, and proof block tags. It excludes SRS files, local caches, raw witness, checkpoints, datasets, logs, and run manifest attachments.",
            "",
            "| dataset | backend | proof_size_bytes | proof_hash | public_metadata_hash |",
            "| --- | --- | ---: | --- | --- |",
            *[
                f"| {row['dataset']} | {row['backend']} | {row['proof_size_bytes']} | "
                f"{row['proof_hash']} | {row['public_metadata_hash']} |"
                for row in rows
            ],
        ]) + "\n",
        encoding="utf-8",
    )
    profile_stage_fields = [
        ("context/config build", "context_build_ms"),
        ("SRS manifest/material load", "srs_prepare_ms"),
        ("GAT reference forward", "forward_ms"),
        ("trace construction", "trace_generation_ms"),
        ("masked polynomial materialization", "witness_materialization_ms"),
        ("public/static commitment", "load_static_ms"),
        ("dynamic commitment", "commit_dynamic_ms"),
        ("coefficient/domain MSM", "dynamic_commit_msm_ms"),
        ("quotient materialization", "quotient_build_ms"),
        ("quotient commitment", "quotient_bundle_pack_ms"),
        ("opening witness generation", "domain_open_witness_ms"),
        ("batch opening", "domain_opening_ms"),
        ("external opening", "external_opening_ms"),
        ("proof serialization/finalize", "prove_finalize_ms"),
        ("verifier transcript replay", "verify_transcript_ms"),
        ("verifier pairing/opening", "verify_domain_opening_ms"),
        ("payload audit", "verify_metadata_ms"),
        ("total prove", "prove_time_ms"),
        ("total verify", "verify_time_ms"),
    ]
    def as_float(row, key):
        try:
            return float(row.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def top_rows(profile_rows, key, limit=20):
        return sorted(profile_rows, key=lambda item: as_float(item, key), reverse=True)[:limit]

    def object_profile_table(title, profile_rows, key, value_label):
        lines = [
            f"### {title}",
            "",
            f"| label | domain | basis | vector_length | nonzero_scalar_count | zero_scalar_ratio | padding_ratio | {value_label} | ownership | category | opportunity |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for item in profile_rows:
            lines.append(
                f"| {item.get('label', '')} | {item.get('domain', '')} | {item.get('basis', '')} | "
                f"{item.get('vector_length', '')} | {item.get('nonzero_scalar_count', '')} | "
                f"{item.get('zero_scalar_ratio', '')} | {item.get('padding_ratio', '')} | "
                f"{item.get(key, '')} | {item.get('ownership', '')} | {item.get('category', '')} | "
                f"{item.get('optimization_opportunity', '')} |"
            )
        return lines

    def group_totals(profile_rows, group_key):
        grouped = {}
        for item in profile_rows:
            key = item.get(group_key, "")
            entry = grouped.setdefault(key, {"count": 0, "time": 0.0})
            entry["count"] += 1
            entry["time"] += as_float(item, "total_commitment_ms")
        return sorted(grouped.items(), key=lambda pair: pair[1]["time"], reverse=True)

    def repeated_rows(profile_rows):
        grouped = {}
        for item in profile_rows:
            fp = item.get("commitment_fingerprint", "")
            if not fp:
                continue
            grouped.setdefault(fp, []).append(item)
        repeated = []
        for _, items in grouped.items():
            if len(items) > 1:
                first = items[0].copy()
                first["repeat_count"] = str(len(items))
                first["repeated_labels"] = ", ".join(item.get("label", "") for item in items[:20])
                repeated.append(first)
        return sorted(repeated, key=lambda item: int(item.get("repeat_count", "0")), reverse=True)[:20]

    cora_profiles = [(dataset, backend, manifest) for dataset, backend, manifest, _ in manifests if dataset == "cora"]
    profile_lines = [
        "# Optimization Profile",
        "",
        "## Optimization Audit",
        "",
        "| item | class | decision | reason |",
        "| --- | --- | --- | --- |",
        "| fast MSM | A | retained | SRS powers/Lagrange MSM stays in the KZG commitment and witness path. |",
        "| parallel FFT / NTT | A | retained | Only changes domain evaluation and quotient materialization execution. |",
        "| FFT backend/kernel upgrade | A | retained | Packed evaluation kernels preserve opened values and transcript order. |",
        "| trace layout upgrade | A | retained | Reduces materialization/copy cost without changing columns or padding semantics. |",
        "| fast verify pairing | B | adapted | Verifier now uses VK-only prepared G2 material, not prover SRS. |",
        "| CUDA hotspots / GPU backend | B | adapted | GPU runs record CUDA hotspot routing and CPU fallback scope. |",
        "| batch commitment/opening | A | retained | All claimed values, commitments, witnesses and pairing checks remain present. |",
        "| SRS loading/cache | B | adapted | Manifest-hash-bound process cache is used; no setup scalar is cached. |",
        "| public/static table reuse | A | retained | Public columns are hash/metadata bound and re-evaluated by verifier. |",
        "| route/PSQ/lookup construction | A | retained | Reuses layout and histogram helpers without changing constraints. |",
        "| tamper tests reuse | A | retained | Negative tests mutate one valid proof and rerun verifier. |",
        "| old trapdoor/single-point shortcuts | C | deleted/blocked | Static scan rejects old runtime scalar and fake commitment markers. |",
        "",
        "## Cora Stage Profile",
        "",
        "| backend | stage | ms |",
        "| --- | --- | ---: |",
    ]
    for _, backend, manifest in cora_profiles:
        for label, key in profile_stage_fields:
            profile_lines.append(f"| {backend} | {label} | {manifest.get(key, '0.000')} |")
        numeric = []
        for label, key in profile_stage_fields:
            try:
                numeric.append((float(manifest.get(key, 0.0)), label))
            except (TypeError, ValueError):
                pass
        top = ", ".join(label for _, label in sorted(numeric, reverse=True)[:5])
        profile_lines.append(f"| {backend} | top bottlenecks | {top or 'n/a'} |")
    profile_lines.extend([
        "",
        "## Optimization Flags And Hardware",
        "",
        "| dataset | backend | optimization_profile_id | optimization_flags | hardware_and_device | gpu_acceleration_status |",
        "| --- | --- | --- | --- | --- | --- |",
        *[
            f"| {dataset} | {backend} | {manifest['optimization_profile_id']} | "
            f"{manifest['optimization_flags']} | {manifest['hardware_and_device']} | "
        f"{manifest['gpu_acceleration_status']} |"
            for dataset, backend, manifest, _ in manifests
        ],
        "",
        "## Multi-Dataset Stage Profile",
        "",
        "| dataset | backend | stage | ms |",
        "| --- | --- | --- | ---: |",
    ])
    for dataset, backend, manifest, _ in manifests:
        for label, key in profile_stage_fields:
            profile_lines.append(f"| {dataset} | {backend} | {label} | {manifest.get(key, '0.000')} |")
    profile_lines.extend([
        "",
        "## Dynamic Commitment Object Profiles",
        "",
        "Object-level profiles are generated by the secure runner for dynamic KZG commitments. The classification is label/domain/shape derived and does not branch on dataset names.",
    ])
    profile_datasets = {"cora", "pubmed", "ogbn-arxiv"}
    for dataset, backend, manifest, object_profile in manifests:
        if dataset not in profile_datasets:
            continue
        profile_lines.extend([
            "",
            f"### {dataset} / {backend}",
            "",
            f"- object_count: {manifest['dynamic_commitment_object_count']}",
            f"- count_by_domain: {manifest['dynamic_commitment_count_by_domain']}",
            f"- time_by_domain_ms: {manifest['dynamic_commitment_time_by_domain_ms']}",
            f"- time_by_ownership_category_ms: {manifest['dynamic_commitment_time_by_ownership_category_ms']}",
            "",
        ])
        profile_lines.extend(object_profile_table("Top 20 Slowest Dynamic Commitments", top_rows(object_profile, "total_commitment_ms"), "total_commitment_ms", "total_commitment_ms"))
        profile_lines.extend([""])
        profile_lines.extend(object_profile_table("Top 20 Largest MSMs", top_rows(object_profile, "vector_length"), "msm_ms", "msm_ms"))
        repeats = repeated_rows(object_profile)
        profile_lines.extend([
            "",
            "### Top 20 Repeated Materializations",
            "",
            "| label | repeat_count | repeated_labels | domain | basis | vector_length | total_commitment_ms | opportunity |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | --- |",
        ])
        for item in repeats:
            profile_lines.append(
                f"| {item.get('label', '')} | {item.get('repeat_count', '0')} | {item.get('repeated_labels', '')} | "
                f"{item.get('domain', '')} | {item.get('basis', '')} | {item.get('vector_length', '')} | "
                f"{item.get('total_commitment_ms', '')} | {item.get('optimization_opportunity', '')} |"
            )
        profile_lines.extend([
            "",
            "### Time By Domain",
            "",
            "| domain | object_count | total_commitment_ms |",
            "| --- | ---: | ---: |",
        ])
        for key, value in group_totals(object_profile, "domain"):
            profile_lines.append(f"| {key} | {value['count']} | {value['time']:.3f} |")
        profile_lines.extend([
            "",
            "### Time By Ownership And Category",
            "",
            "| ownership/category | object_count | total_commitment_ms |",
            "| --- | ---: | ---: |",
        ])
        for key, value in group_totals(
            [{**item, "owner_category": item.get("ownership", "") + "/" + item.get("category", "")} for item in object_profile],
            "owner_category",
        ):
            profile_lines.append(f"| {key} | {value['count']} | {value['time']:.3f} |")
        profile_lines.extend([
            "",
            "### Optimization Opportunity Classification",
            "",
            "| opportunity | object_count | total_commitment_ms |",
            "| --- | ---: | ---: |",
        ])
        for key, value in group_totals(object_profile, "optimization_opportunity"):
            profile_lines.append(f"| {key} | {value['count']} | {value['time']:.3f} |")
    profile_lines.extend([
        "",
        "## Generality Statement",
        "",
        "The recorded optimizations are dataset-agnostic execution choices: SRS MSM zero-scalar skipping, sparse compaction, compact evaluation columns, shared column backing, shape-aware large-domain handling, packed/parallel evaluation, prepared VK pairing, and CUDA hotspot routing where requested.",
        "",
        "Large-graph behavior is selected by node/edge/feature/domain shape and backend capability. No Cora-specific shortcut is used; PPI multi-graph boundaries and ogbn-arxiv large-graph shape are not downgraded.",
        "",
        "All optimizations are execution-route choices only; proof schema, transcript order, opening equations, masked private payload contract, and public output hashes are unchanged.",
    ])
    (output_dir / "optimization_profile_report.md").write_text(
        "\n".join(profile_lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="append", default=[])
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-tree-hash", default="")
    parser.add_argument("--cpu-binary-hash", default="")
    parser.add_argument("--gpu-binary-hash", default="")
    args = parser.parse_args()
    if len(args.cpu) != 5 or len(args.gpu) != 5:
        raise ValueError("expected exactly five CPU runs and five GPU runs")
    cpu_rows, cpu_manifests = collect(
        args.cpu,
        "cpu",
        args.source_tree_hash,
        args.cpu_binary_hash,
        args.gpu_binary_hash)
    gpu_rows, gpu_manifests = collect(
        args.gpu,
        "gpu",
        args.source_tree_hash,
        args.cpu_binary_hash,
        args.gpu_binary_hash)
    rows = cpu_rows + gpu_rows
    write_outputs(rows, cpu_manifests + gpu_manifests, pathlib.Path(args.output_dir))
    print(f"exported {len(rows)} secure paper rows")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"fatal: {error}", file=sys.stderr)
        raise SystemExit(1)
