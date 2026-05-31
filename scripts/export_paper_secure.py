#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
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
    "verifier_reads_witness",
    "private_plain_value_count",
    "reference_consistency_passed",
    "tamper_tests_passed",
    "full_dataset",
    "dataset_id",
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
    "srs_degree_bound",
    "proof_payload_audit_status",
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


def require_field(row, key):
    if key not in row or row[key] in ("", None):
        raise ValueError(f"missing required field {key} for {row.get('dataset', '<unknown>')}")


def normalize_bool(value):
    return str(value).lower()


def validate_row(dataset, backend, row):
    require_field(row, "dataset")
    if dataset == "ogbn-arxiv":
        expected_names = {"ogbn-arxiv", "ogbn_arxiv"}
    else:
        expected_names = {dataset}
    if row["dataset"] not in expected_names:
        raise ValueError(f"dataset key {dataset} does not match manifest dataset {row['dataset']}")
    if row.get("srs_type") not in {"trusted_srs", "mpc_srs"}:
        raise ValueError(f"{dataset}/{backend} has invalid SRS type")
    for key, expected in SECURE_FIELDS.items():
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


def project_row(dataset, backend, row):
    out = {key: row.get(key, "") for key in OUTPUT_FIELDS}
    out["dataset"] = dataset
    out["backend"] = backend
    return out


def collect(runs, backend):
    rows = []
    manifests = []
    seen = set()
    for item in runs:
        dataset, path = parse_run(item)
        row = load_manifest(path)
        validate_row(dataset, backend, row)
        seen.add(dataset)
        rows.append(project_row(dataset, backend, row))
        manifests.append((dataset, backend, row))
    if seen != REQUIRED_DATASETS:
        missing = ", ".join(sorted(REQUIRED_DATASETS - seen))
        extra = ", ".join(sorted(seen - REQUIRED_DATASETS))
        raise ValueError(f"{backend} run set mismatch; missing={missing}; extra={extra}")
    return rows, manifests


def write_outputs(rows, manifests, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
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
    manifest_dir = output_dir / "run_manifests"
    manifest_dir.mkdir(exist_ok=True)
    for dataset, backend, manifest in manifests:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="append", default=[])
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.cpu) != 5 or len(args.gpu) != 5:
        raise ValueError("expected exactly five CPU runs and five GPU runs")
    cpu_rows, cpu_manifests = collect(args.cpu, "cpu")
    gpu_rows, gpu_manifests = collect(args.gpu, "gpu")
    rows = cpu_rows + gpu_rows
    write_outputs(rows, cpu_manifests + gpu_manifests, pathlib.Path(args.output_dir))
    print(f"exported {len(rows)} secure paper rows")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"fatal: {error}", file=sys.stderr)
        raise SystemExit(1)
