#!/usr/bin/env python3
"""Rebuild the real-graph mechanism tables from the 66 initial proof receipts.

This standard-library-only report is intentionally separate from the repository
runner's gated summarize/publication-integrity commands. It does not verify
proofs. It requires a passing receipt/hash audit and records that the user's
full independent re-verification was cancelled after 13 of 66 cells.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from pathlib import Path
from typing import Any


SESSION = Path(__file__).resolve().parent
FINAL_ROOT = SESSION.parent
PROJECT_ROOT = SESSION.parents[2]
CANCELLED_EVIDENCE = (
    FINAL_ROOT / "incidents" / "reverify_cancelled_20261010T085441Z_c57bc5f351"
)
METRICS = (
    ("commitment_time_ms", "Commitment Time (ms)"),
    ("prove_time_ms", "Prove Time (ms)"),
    ("verify_time_ms", "Verify Time (ms)"),
    ("proof_size_bytes", "Proof Size (Bytes)"),
)
EXPECTED_CONFIGS = (
    ("ff_cora_full_d1_g1", "feature_folding", "cora", 1, 1, None),
    ("ff_cora_full_d2_g1", "feature_folding", "cora", 2, 1, None),
    ("ff_cora_full_d4_g1", "feature_folding", "cora", 4, 1, None),
    ("ff_cora_full_d8_g1", "feature_folding", "cora", 8, 1, None),
    ("ff_citeseer_full_d8_g1", "feature_folding", "citeseer", 8, 1, None),
    ("ff_ppi_components-0_d8_g1", "feature_folding", "ppi", 8, 1, [0]),
    ("ff_ppi_components-0-9_d8_g2", "feature_folding", "ppi", 8, 2, [0, 9]),
    ("sn_cora_full_d8_g1", "shared_neighborhood", "cora", 8, 1, None),
    ("sn_citeseer_full_d8_g1", "shared_neighborhood", "citeseer", 8, 1, None),
    ("sn_ppi_components-0_d8_g1", "shared_neighborhood", "ppi", 8, 1, [0]),
    ("sn_ppi_components-0-9_d8_g2", "shared_neighborhood", "ppi", 8, 2, [0, 9]),
)
PAIR_ID_FIELDS = (
    "mechanism", "dataset", "N", "E", "D", "G", "component_ids",
    "node_ptr", "edge_ptr", "nonself_neighbor_stats", "input_bundle_sha256",
    "input_metadata_sha256", "input_witness_sha256", "data_identity_sha256",
    "pair_input_commitment_sha256", "srs_id", "srs_manifest_sha256",
    "protocol_schema_id", "security_target_bits", "security_target_numerator_threshold",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def atomic_write(path: Path, raw: bytes) -> None:
    require(not path.is_symlink(), f"refusing symlink output: {path}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.is_symlink(), f"refusing symlink temporary output: {temporary}")
    with temporary.open("wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def parse_json_cell(row: dict[str, str], field: str) -> Any:
    try:
        return json.loads(row[field])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in cells.csv field {field}") from exc


def safe_cell_file(path_text: str, expected_sha: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = SESSION / path
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(SESSION)
    except ValueError as exc:
        raise ValueError(f"{label} escapes immutable session: {path}") from exc
    current = SESSION
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"{label} traverses a symlink: {current}")
    require(sha256_file(path) == expected_sha, f"{label} SHA-256 mismatch: {path}")
    return path


def as_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def sample_stats(values: list[float]) -> dict[str, Any]:
    require(len(values) == 3, f"expected exactly 3 observations, found {len(values)}")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": len(values)}


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8)


def validate_session() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    summary_path = SESSION / "summary.json"
    cells_path = SESSION / "cells.csv"
    audit_path = SESSION / "audit.json"
    results_manifest_path = SESSION / "results_manifest.json"
    summary = load_json(summary_path, "session summary")
    audit = load_json(audit_path, "receipt audit")
    cancellation = load_json(CANCELLED_EVIDENCE / "cancellation_record.json", "cancellation record")
    cancellation_manifest = load_json(CANCELLED_EVIDENCE / "evidence_manifest.json", "cancellation evidence manifest")

    require(summary.get("session_id") == SESSION.name and summary.get("session_status") == "complete",
            "session identity/status is not final")
    require(summary.get("expected_cell_count") == 66 and summary.get("cell_count") == 66 and
            summary.get("verified_cells") == 66 and summary.get("expected_config_count") == 11 and
            summary.get("expected_pair_count") == 33 and summary.get("repetitions") == 3,
            "formal summary does not record the required 11x3x2 completed experiment")
    require(audit.get("format") == "zkgat_real_mechanism_result_audit_v1" and
            audit.get("status") == "PASS" and audit.get("session_id") == summary["session_id"] and
            audit.get("summary_sha256") == sha256_file(summary_path) and
            audit.get("cell_count") == 66 and audit.get("verified_cells") == 66 and
            audit.get("reverified_independently") is False,
            "ordinary receipt/hash audit is not a passing initial-only audit")
    require(not (SESSION / "audit_reverify.json").exists() and
            not (SESSION / "audit_reverify.sha256").exists(),
            "unexpected full-reverification archive; refusing ambiguous initial-only report")
    require(cancellation.get("status") == "CANCELLED_BY_EXPLICIT_USER_INSTRUCTION" and
            cancellation.get("post_session_independent_reverification", {}).get("completed_verify_ok") == 13 and
            cancellation.get("post_session_independent_reverification", {}).get("expected_cells") == 66 and
            cancellation.get("post_session_independent_reverification", {}).get("full_reverify_archive_committed") is False,
            "cancellation evidence does not match the recorded 13/66 cancelled attempt")
    expected_manifest_sha = (CANCELLED_EVIDENCE / "evidence_manifest.sha256").read_text().split()[0]
    require(expected_manifest_sha == sha256_file(CANCELLED_EVIDENCE / "evidence_manifest.json"),
            "cancellation evidence manifest hash mismatch")
    require(cancellation_manifest.get("session_id") == summary["session_id"],
            "cancellation evidence belongs to a different session")
    expected_configs = [
        {"config_id": cid, "mechanism": mechanism, "dataset": dataset, "width": d,
         "graphs": g, "components": components}
        for cid, mechanism, dataset, d, g, components in EXPECTED_CONFIGS
    ]
    require(summary.get("formal_configs") == expected_configs,
            "formal configuration matrix differs from the fixed experiment")
    require(summary.get("selected_layer") == 0 and summary.get("selected_head") == 0 and
            summary.get("formal_head_width") == 8,
            "formal hidden layer/head identity differs from the fixed experiment")
    rows: list[dict[str, str]] = []
    with cells_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames is not None, "cells.csv has no header")
        rows = list(reader)
    require(len(rows) == 66, f"cells.csv has {len(rows)} rows, expected 66")
    result_manifest = load_json(results_manifest_path, "formal results manifest")
    require(result_manifest.get("session_id") == summary["session_id"] and
            result_manifest.get("session_status") == "complete",
            "formal results manifest does not bind this completed session")
    return summary, audit, rows, {
        "cancellation": cancellation,
        "cancellation_manifest_sha256": expected_manifest_sha,
        "results_manifest": result_manifest,
    }


def build_pairs(
    summary: dict[str, Any], audit: dict[str, Any], rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {item[0]: item for item in EXPECTED_CONFIGS}
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        cid = row["config_id"]
        require(cid in expected, f"unregistered config {cid}")
        repetition = int(row["repetition"])
        variant = row["variant"]
        key = (cid, repetition, variant)
        require(key not in seen, f"duplicate cell {key}")
        seen.add(key)
        require(repetition in (1, 2, 3) and variant in ("original", "control"),
                f"invalid repetition/variant for {key}")
        require(row["cell_status"] == "complete" and row["pair_status"] == "paired" and
                row["verification_status"] == "VERIFY_OK" and row["cell_status"] != "failed" and
                row["prover_exit_code"] == "0" and row["verifier_exit_code"] == "0",
                f"initial proof receipt is not successful for {key}")
        require(row["backend"] == "cuda" and as_bool(row["cuda_actual"]) and
                as_bool(row["cuda_ntt_forced"]) and int(row["cuda_kernel_invocations_delta"]) > 0,
                f"CUDA execution receipt is incomplete for {key}")
        require(as_bool(row["security_budget_valid"]) and int(row["security_target_bits"]) == 120,
                f"security policy receipt is invalid or below the fixed target for {key}")
        stats = parse_json_cell(row, "cuda_stats_delta")
        require(int(stats.get("ntt_calls", 0)) > 0 and int(stats.get("intt_calls", 0)) > 0 and
                int(stats.get("msm_single_calls", 0)) + int(stats.get("msm_batch_calls", 0)) > 0,
                f"CUDA NTT/INTT/MSM evidence missing for {key}")
        proof = safe_cell_file(row["proof_path"], row["proof_sha256"], f"proof {key}")
        require(proof.stat().st_size == int(row["proof_size_bytes"]), f"proof size mismatch for {key}")
        safe_cell_file(row["statement_path"], row["statement_sha256"], f"statement {key}")
        safe_cell_file(row["prover_result_path"], row["prover_result_sha256"], f"prover receipt {key}")
        verifier_path = safe_cell_file(row["verifier_result_path"], row["verifier_result_sha256"],
                                       f"independent verifier receipt {key}")
        verifier_receipt = load_json(verifier_path, f"independent verifier receipt {key}")
        require(verifier_receipt.get("status") == "VERIFY_OK", f"verifier receipt status mismatch for {key}")
        grouped.setdefault((cid, repetition), {})[variant] = row

    require(len(seen) == 66, f"unique cells count {len(seen)} != 66")
    audit_cells = audit.get("cells")
    require(isinstance(audit_cells, list) and len(audit_cells) == 66,
            "passing receipt audit does not enumerate all 66 cells")
    require(all(isinstance(item, dict) and item.get("status") == "RECEIPT_VERIFIED"
                for item in audit_cells),
            "ordinary audit did not accept every initial verifier receipt")
    audit_keys = {
        (str(item.get("config_id")), int(item.get("repetition", -1)), str(item.get("variant")))
        for item in audit_cells if isinstance(item, dict)
    }
    row_keys = {(r["config_id"], int(r["repetition"]), r["variant"]) for r in rows}
    require(audit_keys == row_keys, "receipt audit cells do not exactly match the 66 cells.csv identities")

    pairs: list[dict[str, Any]] = []
    for cid, mechanism, dataset, d, g, components in EXPECTED_CONFIGS:
        for repetition in (1, 2, 3):
            variants = grouped.get((cid, repetition), {})
            original, control = variants.get("original"), variants.get("control")
            require(original is not None and control is not None, f"incomplete pair {(cid, repetition)}")
            for field in PAIR_ID_FIELDS:
                require(original[field] == control[field],
                        f"Original/Control input or policy mismatch for {(cid, repetition)}: {field}")
            require(original["mechanism"] == mechanism and original["dataset"] == dataset and
                    int(original["D"]) == d and int(original["G"]) == g,
                    f"configuration metadata mismatch for {(cid, repetition)}")
            component_ids = json.loads(original["component_ids"])
            if components is not None:
                require(component_ids == components, f"PPI component selection mismatch for {cid}")
            else:
                require(component_ids in ([], [0]), f"unexpected component IDs for full graph {cid}")
            pair: dict[str, Any] = {
                "config_id": cid, "mechanism": mechanism, "dataset": dataset,
                "N": int(original["N"]), "E": int(original["E"]), "D": int(original["D"]),
                "G": int(original["G"]), "component_ids": component_ids,
                "repetition": repetition,
                "pair_input_commitment_sha256": original["pair_input_commitment_sha256"],
                "original": {name: float(original[name]) for name, _ in METRICS},
                "control": {name: float(control[name]) for name, _ in METRICS},
                "proofs": {
                    "original": {k: original[k] for k in (
                        "proof_sha256", "statement_sha256", "prover_result_sha256",
                        "verifier_result_sha256", "proof_path")},
                    "control": {k: control[k] for k in (
                        "proof_sha256", "statement_sha256", "prover_result_sha256",
                        "verifier_result_sha256", "proof_path")},
                },
                "metrics": {},
            }
            for name, _ in METRICS:
                o = float(original[name])
                c = float(control[name])
                pair["metrics"][name] = {
                    "original_minus_control": o - c,
                    "saving_percent": None if c == 0 else (c - o) / c * 100.0,
                    "improvement_factor": None if o == 0 else c / o,
                }
            pairs.append(pair)

    pair_rows = []
    for pair in pairs:
        row = {k: pair[k] for k in (
            "mechanism", "config_id", "dataset", "N", "E", "D", "G",
            "component_ids", "repetition", "pair_input_commitment_sha256")}
        row["component_ids"] = json.dumps(pair["component_ids"], separators=(",", ":"))
        for name, _ in METRICS:
            row[f"original_{name}"] = pair["original"][name]
            row[f"control_{name}"] = pair["control"][name]
            for effect, value in pair["metrics"][name].items():
                row[f"{name}_{effect}"] = value
        pair_rows.append(row)
    return pairs, pair_rows


def crosscheck_official_summary(summary: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    official = {item["config_id"]: item for item in summary.get("groups", [])}
    require(set(official) == {g["config_id"] for g in groups},
            "summary.json official group list does not match the fixed configurations")
    for group in groups:
        expected = official[group["config_id"]]
        for metric, stats in group["metrics"].items():
            for side in ("original", "control", "paired_original_minus_control", "saving_percent",
                         "improvement_factor"):
                for field in ("mean", "sample_sd", "n"):
                    a = stats[side][field]
                    b = expected["metrics"][metric][side][field]
                    require((a is None and b is None) or
                            (a is not None and b is not None and
                             (a == b if field == "n" else close(float(a), float(b)))),
                            f"initial-only recomputation differs from formal summary: "
                            f"{group['config_id']} {metric} {side}.{field}: {a} vs {b}")


def summarize_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for cid, mechanism, dataset, d, g, components in EXPECTED_CONFIGS:
        selected = [p for p in pairs if p["config_id"] == cid]
        require(len(selected) == 3, f"{cid} has {len(selected)} paired repetitions")
        first = selected[0]
        group: dict[str, Any] = {
            "config_id": cid, "mechanism": mechanism, "dataset": dataset,
            "N": first["N"], "E": first["E"], "D": first["D"], "G": first["G"],
            "component_ids": first["component_ids"], "requested_repetitions": 3,
            "complete_repetitions": len(selected), "metrics": {},
        }
        for name, _ in METRICS:
            original = [p["original"][name] for p in selected]
            control = [p["control"][name] for p in selected]
            diff = [p["metrics"][name]["original_minus_control"] for p in selected]
            saving = [p["metrics"][name]["saving_percent"] for p in selected
                      if p["metrics"][name]["saving_percent"] is not None]
            factor = [p["metrics"][name]["improvement_factor"] for p in selected
                      if p["metrics"][name]["improvement_factor"] is not None]
            group["metrics"][name] = {
                "original": sample_stats(original), "control": sample_stats(control),
                "paired_original_minus_control": sample_stats(diff),
                "saving_percent": sample_stats(saving),
                "improvement_factor": sample_stats(factor),
            }
        groups.append(group)
    return groups


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode())


def csv_summary(groups: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = ["mechanism", "config_id", "dataset", "N", "E", "D", "G", "component_ids",
              "metric", "original_mean", "original_sample_sd", "control_mean", "control_sample_sd",
              "paired_original_minus_control_mean", "paired_original_minus_control_sample_sd",
              "saving_percent_mean", "saving_percent_sample_sd",
              "improvement_factor_mean", "improvement_factor_sample_sd", "repetitions"]
    rows: list[dict[str, Any]] = []
    for group in groups:
        for metric, label in METRICS:
            s = group["metrics"][metric]
            rows.append({
                "mechanism": group["mechanism"], "config_id": group["config_id"],
                "dataset": group["dataset"], "N": group["N"], "E": group["E"],
                "D": group["D"], "G": group["G"],
                "component_ids": json.dumps(group["component_ids"], separators=(",", ":")),
                "metric": label,
                "original_mean": s["original"]["mean"], "original_sample_sd": s["original"]["sample_sd"],
                "control_mean": s["control"]["mean"], "control_sample_sd": s["control"]["sample_sd"],
                "paired_original_minus_control_mean": s["paired_original_minus_control"]["mean"],
                "paired_original_minus_control_sample_sd": s["paired_original_minus_control"]["sample_sd"],
                "saving_percent_mean": s["saving_percent"]["mean"],
                "saving_percent_sample_sd": s["saving_percent"]["sample_sd"],
                "improvement_factor_mean": s["improvement_factor"]["mean"],
                "improvement_factor_sample_sd": s["improvement_factor"]["sample_sd"],
                "repetitions": "3/3",
            })
    return fields, rows


def f(value: float, metric: str) -> str:
    return f"{value:.2f}" if metric == "proof_size_bytes" else f"{value:.6f}"


def markdown_table(mechanism: str, groups: list[dict[str, Any]], scope_text: str) -> str:
    title = ("Sparse Node-to-Edge Feature Verification" if mechanism == "feature_folding"
             else "Shared Neighborhood Proof Reuse")
    lines = [
        f"# {title} — initial-verification-only real-data CUDA results",
        "",
        f"Session: {SESSION.name}; status: complete; backend: CUDA; accepted initial proof receipts: 66/66.",
        "Measurements: three paired repetitions per configuration. CPU receipt/hash audit passed.",
        "Full independent re-verification was explicitly cancelled by the user after 13/66 cells; "
        "this report does not claim a full re-verification archive or publication-integrity PASS.",
        scope_text,
        "Layer/head source: formal hidden layer 0, attention head 0, width 8. Feature Folding D<8 "
        "uses the first D coordinates of that same real width-8 hidden feature as a controlled mechanism width.",
        "Commitment and Prove are disjoint native proof partitions; Verify is the recorded independent-verifier "
        "measurement; Proof Size is the serialized mechanism proof file length.",
        "Per-pair effects are computed first: Saving (%) = (Control - Original) / Control × 100; "
        "Improvement Factor = Control / Original. Means and sample SDs summarize the three values; "
        "no confidence interval or significance claim is made.",
        "",
        "| Dataset | N | E | D | G | PPI components | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired O−C mean | Paired O−C sample SD | Saving % mean | Saving % sample SD | Control/Original mean | Control/Original sample SD | Repetitions |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        if group["mechanism"] != mechanism:
            continue
        components = ",".join(map(str, group["component_ids"])) or "—"
        for metric, label in METRICS:
            s = group["metrics"][metric]
            row = [
                group["dataset"], group["N"], group["E"], group["D"], group["G"], components, label,
                f(s["original"]["mean"], metric), f(s["original"]["sample_sd"], metric),
                f(s["control"]["mean"], metric), f(s["control"]["sample_sd"], metric),
                f(s["paired_original_minus_control"]["mean"], metric),
                f(s["paired_original_minus_control"]["sample_sd"], metric),
                f(s["saving_percent"]["mean"], "saving_percent"),
                f(s["saving_percent"]["sample_sd"], "saving_percent"),
                f(s["improvement_factor"]["mean"], "ratio"),
                f(s["improvement_factor"]["sample_sd"], "ratio"), "3/3",
            ]
            lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    summary, audit, rows, extra = validate_session()
    pairs, pair_rows = build_pairs(summary, audit, rows)
    groups = summarize_pairs(pairs)
    crosscheck_official_summary(summary, groups)
    policy = {
        "validation_scope": "initial-verification-receipt-and-hash-audit-only",
        "initial_cells": 66, "initial_verify_ok": 66,
        "post_session_full_reverification": "CANCELLED_BY_EXPLICIT_USER_INSTRUCTION",
        "post_session_reverification_verify_ok": 13, "post_session_reverification_expected": 66,
        "full_reverification_archive": None,
        "publication_integrity_pass": False,
        "ordinary_audit": {"status": audit["status"], "reverified_independently": False,
                           "verified_cells": audit["verified_cells"]},
        "limitations": [
            "No claim is made that all 66 proofs passed a second independent verification.",
            "Three repetitions are descriptive; no statistical significance is claimed.",
            "The measurements are mechanism-level proof costs, not end-to-end GAT proof performance.",
        ],
    }
    statistics_document = {
        "format": "zkgat_real_mechanism_initial_only_statistics_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": summary["session_id"], "session_root": str(SESSION),
        "experiment": {
            "mechanisms": ["feature_folding", "shared_neighborhood"],
            "configuration_count": 11, "paired_repetitions_per_configuration": 3,
            "pair_count": 33, "proof_count": 66,
            "selected_layer": summary["selected_layer"], "selected_head": summary["selected_head"],
            "formal_head_width": summary["formal_head_width"],
            "backend": summary.get("backend"), "gpu_sharing_mode": summary.get("gpu_sharing_mode"),
            "provenance": summary.get("provenance"),
            "timing_semantics": summary.get("timing_semantics"),
        },
        "verification_policy": policy,
        "formulae": {
            "paired_original_minus_control": "Original - Control",
            "saving_percent": "(Control - Original) / Control * 100",
            "improvement_factor": "Control / Original",
            "aggregate": "compute each pair first, then report mean and sample standard deviation over n=3",
        },
        "groups": groups, "pairs": pairs,
    }
    outputs: dict[str, Path] = {
        "raw_csv": SESSION / "initial_only_cells.csv",
        "raw_json": SESSION / "initial_only_cells.json",
        "pairs_csv": SESSION / "initial_only_pairs.csv",
        "pairs_json": SESSION / "initial_only_pairs.json",
        "statistics_json": SESSION / "initial_only_statistics.json",
        "feature_folding_csv": SESSION / "feature_folding_summary_initial_only.csv",
        "feature_folding_markdown": SESSION / "feature_folding_summary_initial_only.md",
        "shared_neighborhood_csv": SESSION / "shared_neighborhood_summary_initial_only.csv",
        "shared_neighborhood_markdown": SESSION / "shared_neighborhood_summary_initial_only.md",
    }
    source_cells = SESSION / "cells.csv"
    atomic_write(outputs["raw_csv"], source_cells.read_bytes())
    atomic_write(outputs["raw_json"], json_bytes(rows))
    pair_fields = list(pair_rows[0].keys())
    write_csv(outputs["pairs_csv"], pair_fields, pair_rows)
    atomic_write(outputs["pairs_json"], json_bytes(pairs))
    atomic_write(outputs["statistics_json"], json_bytes(statistics_document))
    fields, summary_rows = csv_summary(groups)
    for mechanism, key in (
        ("feature_folding", "feature_folding"),
        ("shared_neighborhood", "shared_neighborhood"),
    ):
        selected = [row for row in summary_rows if row["mechanism"] == mechanism]
        write_csv(outputs[f"{key}_csv"], fields, selected)
        scope_text = (
            "Feature Folding D<8 is a controlled-width experiment on the first D coordinates "
            "of a real 8-wide hidden head."
            if mechanism == "feature_folding" else
            "Shared Neighborhood compares the shared PSQ/neighborhood proof with separate denominator "
            "and weighted-aggregate proofs."
        )
        atomic_write(outputs[f"{key}_markdown"],
                     markdown_table(mechanism, groups, scope_text).encode())

    input_names = (
        "summary.json", "cells.csv", "audit.json", "results_manifest.json",
        "source_manifest.json", "build_source_manifest.json",
        "configuration_and_srs_manifest.json", "cuda_activity_samples.jsonl",
    )
    input_hashes = {
        name: {"path": str(SESSION / name), "sha256": sha256_file(SESSION / name)}
        for name in input_names if (SESSION / name).is_file()
    }
    document_hashes = {}
    for relative in ("README.md", "docs/mechanism_ablation.md"):
        document = PROJECT_ROOT / relative
        document_hashes[relative] = {
            "path": str(document), "sha256": sha256_file(document),
            "size_bytes": document.stat().st_size,
        }
    script_hash = sha256_file(Path(__file__).resolve())
    report_files = {
        name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in outputs.items()
    }
    integrity = {
        "format": "zkgat_mechanism_initial_only_report_integrity_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": summary["session_id"], "validation_scope": policy["validation_scope"],
        "full_reverification_status": "CANCELLED_BY_USER_AFTER_13_OF_66",
        "report_script": {"path": str(Path(__file__).resolve()), "sha256": script_hash},
        "cancellation_evidence": {
            "path": str(CANCELLED_EVIDENCE / "evidence_manifest.json"),
            "sha256": extra["cancellation_manifest_sha256"],
        },
        "receipt_audit": {"path": str(SESSION / "audit.json"), "status": audit["status"],
                          "sha256": input_hashes["audit.json"]["sha256"]},
        "source_inputs": input_hashes, "report_files": report_files,
        "publication_documents_initial_only": {
            "files": document_hashes,
            "strict_runner_receipt": "NOT_CREATED",
            "reason": "record-publication-docs is gated by a full passing audit_reverify archive, "
                      "which was not committed because the user cancelled full re-verification.",
        },
        "python": sys.version, "platform": platform.platform(),
        "generator_command": f"{sys.executable} {Path(__file__).resolve()}",
    }
    integrity_path = SESSION / "initial_only_report_integrity.json"
    integrity_bytes = json_bytes(integrity)
    atomic_write(integrity_path, integrity_bytes)
    atomic_write(SESSION / "initial_only_report_integrity.sha256",
                 f"{sha256_bytes(integrity_bytes)}  initial_only_report_integrity.json\n".encode())
    print(f"INITIAL_ONLY_REPORT_OK session={SESSION} cells=66 pairs=33 configs=11")
    print(f"INTEGRITY={integrity_path} sha256={sha256_bytes(integrity_bytes)}")
    for key in ("feature_folding_markdown", "feature_folding_csv",
                "shared_neighborhood_markdown", "shared_neighborhood_csv",
                "statistics_json", "raw_csv", "raw_json", "pairs_csv", "pairs_json"):
        print(f"{key}={outputs[key]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"initial-only-report: {exc}", file=sys.stderr)
        raise SystemExit(2)
