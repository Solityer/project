#!/usr/bin/env python3
"""Collect only complete, verifier-backed zkGPT result artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.constants import DATASET_ORDER, PROOF_BACKEND_ID, PROTOCOL_ID  # noqa: E402
from zkgpt_gat.config import load_app_config  # noqa: E402
from zkgpt_gat.gpu_execution import validate_gpu_execution_evidence  # noqa: E402
from zkgpt_gat.privacy import check_multi_session_profiles, registration_identity_fingerprint  # noqa: E402
from zkgpt_gat.registry import canonical_registry_public_path  # noqa: E402
from zkgpt_gat.resources import canonical_dataset_name, required_resource_paths, resolve_local  # noqa: E402
from zkgpt_gat.result_validation import (  # noqa: E402
    ResultValidationError,
    canonical_artifact_dir,
    canonical_manifest_path,
    load_canonical_manifest,
    required_artifact_file,
    resolve_results_dir,
    validate_artifact,
    validate_matrix_rows,
)
from zkgpt_gat.util import digest_json, ensure_dir, write_json, write_text_atomic  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_value(entries: dict[str, str], key: str) -> str:
    value = entries.get(key, "")
    if not value:
        raise ValueError(f"config is missing {key}")
    return value


def _fresh_independent_replay(manifest_path: pathlib.Path) -> None:
    """Replay one cell in a fresh verifier process before collecting it.

    The existing report is intentionally captured before replay.  A replay
    that silently rewrites a tampered/stale report is not provenance
    continuity, so collection requires byte-for-byte equality as well as the
    digest fields recorded in the run manifest.
    """
    artifact_dir, manifest = load_canonical_manifest(ROOT, manifest_path)
    dataset = canonical_dataset_name(str(manifest.get("dataset", "")))
    backend = str(manifest.get("backend", ""))
    expected_artifact_dir = canonical_artifact_dir(ROOT, backend, dataset)
    if artifact_dir != expected_artifact_dir:
        raise ValueError("production artifact is outside canonical results/runs containment")
    schema = manifest.get("schema")
    if (isinstance(schema, str) and schema.startswith("zkGPT_test_")) or manifest.get("status") == "FAILED_DIAGNOSTIC":
        raise ValueError("test/FAILED_DIAGNOSTIC artifacts are never collected as formal results")
    for duplicate in ("proof_object.json", "verifier_output.json"):
        if os.path.lexists(artifact_dir / duplicate):
            raise ValueError(f"duplicate non-canonical artifact is forbidden: {duplicate}")
    metadata = json.loads(
        required_artifact_file(artifact_dir, "metadata.json").read_text(encoding="utf-8")
    )
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    config = resolve_local(ROOT, manifest.get("config", ""), "config")
    entries = load_app_config(config, ROOT).entries
    expected_y = required_resource_paths(ROOT, dataset)["public_y_path"]
    configured_y = resolve_local(ROOT, _config_value(entries, "public_y_path"), "public_y_path")
    if configured_y != expected_y or not expected_y.is_file():
        raise ValueError("config/public_y is not the canonical baseline-local public Y")
    registry = canonical_registry_public_path(
        ROOT, _config_value(entries, "model_param_id")
    )
    if not registry.is_file():
        raise ValueError("trusted model registry is missing")
    before = required_artifact_file(artifact_dir, "independent_verification.json").read_bytes()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "verify_proof.py"),
        "--project-root", str(ROOT),
        "--config", str(config),
        "--trusted-registry-public", str(registry),
        "--artifact-dir", str(artifact_dir),
        "--public-y", str(expected_y),
    ]
    replay = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if replay.returncode != 0:
        raise ValueError("fresh independent verifier replay failed: " + (replay.stdout + "\n" + replay.stderr).strip())
    after = required_artifact_file(artifact_dir, "independent_verification.json").read_bytes()
    if after != before:
        raise ValueError("fresh replay report bytes differ from the recorded report")
    report = json.loads(after.decode("utf-8"))
    if not isinstance(report, dict) or report.get("status") != "VERIFY_OK":
        raise ValueError("fresh replay report is not VERIFY_OK")
    if manifest.get("independent_verification_sha256") != _sha256_bytes(after):
        raise ValueError("fresh replay report sha256 does not match run manifest")
    if manifest.get("independent_report_digest") != digest_json(report):
        raise ValueError("fresh replay report provenance digest does not match run manifest")
    if manifest.get("independent_verification_status") != report.get("status"):
        raise ValueError("fresh replay report status provenance mismatch")
    if manifest.get("backend") == "gpu":
        transcript = json.loads(
            required_artifact_file(artifact_dir, "transcript.json").read_text(encoding="utf-8")
        )
        gpu = json.loads(
            required_artifact_file(artifact_dir, "gpu_proof_hot_path.json").read_text(encoding="utf-8")
        )
        valid, errors = validate_gpu_execution_evidence(
            gpu,
            session_nonce=str(metadata.get("session_nonce", "")),
            final_transcript_digest=str(transcript.get("final_digest", "")),
        )
        if not valid:
            raise ValueError("fresh replay GPU nonce/transcript evidence invalid: " + ";".join(errors))


def _validate_single_session_privacy(manifest_path: pathlib.Path) -> dict[str, Any]:
    """Validate one real artifact bundle's privacy profile and provenance."""

    artifact_dir, manifest = load_canonical_manifest(ROOT, manifest_path)
    expected_artifact_dir = canonical_artifact_dir(
        ROOT, str(manifest.get("backend", "")), str(manifest.get("dataset", ""))
    )
    if artifact_dir != expected_artifact_dir:
        raise ValueError("production artifact is outside canonical results/runs containment")
    metadata = json.loads(
        required_artifact_file(artifact_dir, "metadata.json").read_text(encoding="utf-8")
    )
    commitments = json.loads(
        required_artifact_file(artifact_dir, "commitments.json").read_text(encoding="utf-8")
    )
    proof_payload = json.loads(
        required_artifact_file(artifact_dir, "proof_payload.json").read_text(encoding="utf-8")
    )
    transcript = json.loads(
        required_artifact_file(artifact_dir, "transcript.json").read_text(encoding="utf-8")
    )
    session = {
        "metadata": metadata,
        "commitments": commitments,
        "proof_payload": proof_payload,
        "transcript": transcript,
        "pre_profile_transcript_digest": transcript.get("pre_profile_transcript_digest"),
    }
    privacy = check_multi_session_profiles([session])
    if privacy.get("status") != "VERIFY_OK":
        raise ValueError("single-session privacy profile failed: " + str(privacy.get("reason", "unknown")))
    profile = proof_payload.get("PrivacyProfile")
    if not isinstance(profile, dict):
        raise ValueError("proof payload PrivacyProfile is missing")
    items = commitments.get("items")
    if not isinstance(items, dict) or not isinstance(items.get("com_W"), dict):
        raise ValueError("commitments com_W registration is missing")
    return {
        "privacy_session_status": privacy["status"],
        "privacy_profile_schema": profile.get("schema"),
        "privacy_profile_id": profile.get("profile_id"),
        "privacy_profile_digest": profile.get("profile_digest"),
        "privacy_profile_status": profile.get("status"),
        "privacy_session_nonce": profile.get("instance_nonce"),
        "privacy_com_W_fingerprint": registration_identity_fingerprint(items["com_W"], segment="com_W"),
    }


def _collect(results_dir: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    results_dir = resolve_results_dir(ROOT, results_dir)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for backend in ("cpu", "gpu"):
        for dataset in DATASET_ORDER:
            manifest_path = canonical_manifest_path(ROOT, backend, dataset)
            if not manifest_path.is_file():
                rows.append({
                    "dataset": dataset,
                    "backend": backend,
                    "status": "not_run",
                    "proof_generated": False,
                    "verified": False,
                    "formal_experiment_eligible": False,
                    "artifact_path": str(manifest_path.parent.relative_to(ROOT)),
                })
                errors.append(f"manifest_missing:{dataset}/{backend}")
                continue
            try:
                _fresh_independent_replay(manifest_path)
                privacy_provenance = _validate_single_session_privacy(manifest_path)
                row = validate_artifact(ROOT, manifest_path, dataset, backend)
                row.update(privacy_provenance)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rows.append({
                    "dataset": dataset,
                    "backend": backend,
                    "status": "invalid",
                    "proof_generated": False,
                    "verified": False,
                    "formal_experiment_eligible": False,
                    "artifact_path": str(manifest_path.parent.relative_to(ROOT)),
                })
                errors.append(f"artifact_invalid:{dataset}/{backend}:{exc}")
                continue
            rows.append({key: value for key, value in row.items() if key != "manifest"})
    return rows, errors


def _write_outputs(results_dir: pathlib.Path, rows: list[dict[str, Any]], errors: list[str]) -> bool:
    results_dir = resolve_results_dir(ROOT, results_dir)
    eligible = not errors
    try:
        validate_matrix_rows(rows)
    except ResultValidationError as exc:
        eligible = False
        errors.append(str(exc))
    payload = {
        "schema": "zkGPT_gat_formal_result_matrix_v4",
        "baseline": "zkGPT",
        "protocol_id": PROTOCOL_ID,
        "proof_backend_id": PROOF_BACKEND_ID,
        "status": "complete" if eligible else "not_complete",
        "formal_experiment_eligible": eligible,
        "errors": errors,
        "rows": rows,
    }
    write_json(results_dir / "final_zkgpt_results.json", payload)
    fields = ["dataset", "backend", "status", "proof_generated", "verified", "formal_experiment_eligible", "artifact_path", "run_digest", "proof_digest", "proof_size_bytes", "gpu_contract_errors", "privacy_session_status", "privacy_profile_schema", "privacy_profile_id", "privacy_profile_digest", "privacy_profile_status", "privacy_session_nonce", "privacy_com_W_fingerprint"]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    write_text_atomic(
        results_dir / "final_zkgpt_results.csv",
        csv_buffer.getvalue(),
        label="final result CSV",
    )
    lines = [
        "# zkGPT formal result matrix",
        "",
        f"formal_experiment_eligible = {str(eligible).lower()}",
        "",
        "| backend | dataset | status | proof generated | verified |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['backend']} | {row['dataset']} | {row['status']} | {str(row.get('proof_generated', False)).lower()} | {str(row.get('verified', False)).lower()} |"
        for row in rows
    )
    write_text_atomic(
        results_dir / "final_zkgpt_report.md",
        "\n".join(lines) + "\n",
        label="final result Markdown",
    )
    return eligible


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect verifier-backed zkGPT artifacts")
    parser.add_argument("--results-dir", type=pathlib.Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        results_dir = resolve_results_dir(ROOT, args.results_dir)
    except ResultValidationError as exc:
        raise SystemExit(str(exc)) from exc
    ensure_dir(results_dir, root=ROOT, label="results root")
    rows, errors = _collect(results_dir)
    return 0 if _write_outputs(results_dir, rows, errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
