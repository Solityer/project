#!/usr/bin/env python3
"""Independent matrix-level validator for the formal zkGPT results."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.constants import DATASET_ORDER, PROOF_BACKEND_ID, PROTOCOL_ID  # noqa: E402
from zkgpt_gat.config import load_app_config  # noqa: E402
from zkgpt_gat.privacy import check_multi_session_profiles, registration_identity_fingerprint  # noqa: E402
from zkgpt_gat.registry import canonical_registry_public_path  # noqa: E402
from zkgpt_gat.result_validation import (  # noqa: E402
    ResultValidationError,
    canonical_artifact_dir,
    canonical_manifest_path,
    required_artifact_file,
    resolve_inside,
    resolve_results_dir,
    validate_canonical_cell_dir_from_path,
    validate_artifact,
    validate_matrix_rows,
)
from zkgpt_gat.util import ensure_dir, write_json  # noqa: E402


def _config_value(path: pathlib.Path, key: str, default: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip()
    return default


def _replay(root: pathlib.Path, artifact_dir: pathlib.Path, manifest: dict[str, Any]) -> str | None:
    expected_artifact_dir = canonical_artifact_dir(
        root, str(manifest.get("backend", "")), str(manifest.get("dataset", ""))
    )
    try:
        artifact_dir = validate_canonical_cell_dir_from_path(root, artifact_dir)
    except ResultValidationError as exc:
        return str(exc)
    if artifact_dir != expected_artifact_dir:
        return "artifact directory is not canonical results/runs/backend/dataset"
    config = resolve_inside(root, manifest.get("config", ""), "config")
    config_entries = load_app_config(config, root).entries
    registry = canonical_registry_public_path(
        root, config_entries.get("model_param_id", "")
    )
    public_y = resolve_inside(root, _config_value(config, "public_y_path", ""), "canonical public Y")
    command = [
        sys.executable,
        str(root / "scripts" / "verify_proof.py"),
        "--project-root", str(root),
        "--config", str(config),
        "--trusted-registry-public", str(registry),
        "--artifact-dir", str(artifact_dir),
        "--public-y", str(public_y),
    ]
    completed = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return (completed.stdout + "\n" + completed.stderr).strip()
    return None


def _read_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultValidationError(f"{label} must be a JSON object: {path}")
    return value


def _load_privacy_session(artifact_dir: pathlib.Path) -> tuple[str, dict[str, Any]]:
    """Load the authenticated public bundle used for multi-session checks.

    The caller invokes this only after strict artifact validation and a fresh
    independent verifier replay.  This function therefore only assembles the
    already validated wire objects; it never accepts a naked profile or a
    profile copied out of a different proof payload.
    """

    metadata = _read_object(required_artifact_file(artifact_dir, "metadata.json"), "metadata")
    commitments = _read_object(required_artifact_file(artifact_dir, "commitments.json"), "commitments")
    proof_payload = _read_object(required_artifact_file(artifact_dir, "proof_payload.json"), "proof payload")
    transcript = _read_object(required_artifact_file(artifact_dir, "transcript.json"), "transcript")
    pre_digest = transcript.get("pre_profile_transcript_digest")
    if not isinstance(pre_digest, str) or len(pre_digest) != 64:
        raise ResultValidationError("privacy session transcript pre-profile digest is missing")
    items = commitments.get("items")
    if not isinstance(items, dict) or not isinstance(items.get("com_W"), dict):
        raise ResultValidationError("privacy session com_W registration is missing")
    com_w = registration_identity_fingerprint(items["com_W"], segment="com_W")
    return com_w, {
        "metadata": metadata,
        "commitments": commitments,
        "proof_payload": proof_payload,
        "transcript": transcript,
        "pre_profile_transcript_digest": pre_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the complete zkGPT 5x2 result matrix")
    parser.add_argument("--results-dir", type=pathlib.Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        results_dir = resolve_results_dir(ROOT, args.results_dir)
    except ResultValidationError as exc:
        raise SystemExit(str(exc)) from exc
    result_path = results_dir / "final_zkgpt_results.json"
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    validated_artifacts: list[pathlib.Path] = []
    if not result_path.is_file():
        errors.append("result_matrix_missing")
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or result.get("schema") != "zkGPT_gat_formal_result_matrix_v4":
                errors.append("result_matrix_schema_invalid")
            if result.get("baseline") != "zkGPT" or result.get("protocol_id") != PROTOCOL_ID or result.get("proof_backend_id") != PROOF_BACKEND_ID:
                errors.append("result_matrix_identity_invalid")
            if result.get("status") != "complete" or result.get("formal_experiment_eligible") is not True:
                errors.append("result_matrix_not_complete")
            rows = result.get("rows", []) if isinstance(result, dict) else []
            validate_matrix_rows(rows)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, ResultValidationError) as exc:
            errors.append(f"result_matrix_invalid:{exc}")
    row_map = {(str(row.get("backend")), str(row.get("dataset"))): row for row in rows}
    for backend in ("cpu", "gpu"):
        for dataset in DATASET_ORDER:
            key = (backend, dataset)
            manifest_path = canonical_manifest_path(ROOT, backend, dataset)
            if key not in row_map:
                errors.append(f"matrix_row_missing:{dataset}/{backend}")
                continue
            if not manifest_path.is_file():
                errors.append(f"manifest_missing:{dataset}/{backend}")
                continue
            try:
                row = validate_artifact(ROOT, manifest_path, dataset, backend)
                expected = row_map[key]
                for field in ("status", "verified", "proof_generated", "run_digest", "proof_digest", "proof_size_bytes"):
                    if expected.get(field) != row.get(field):
                        raise ResultValidationError(f"matrix row {field} mismatch")
                replay_error = _replay(ROOT, manifest_path.parent, row["manifest"])
                if replay_error:
                    errors.append(f"independent_replay_failed:{dataset}/{backend}:{replay_error}")
                else:
                    # Replay is fresh-process evidence; re-check all recorded
                    # digests and report-derived fields after it completes.
                    validate_artifact(ROOT, manifest_path, dataset, backend)
                    validated_artifacts.append(manifest_path.parent)
            except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError, ResultValidationError) as exc:
                errors.append(f"artifact_validation_failed:{dataset}/{backend}:{exc}")

    # All ten cells must pass their individual checks and fresh verifier replay
    # before their real proof bundles are composed.  Grouping by the
    # authenticated com_W fingerprint permits independent model registrations
    # while still enforcing multi-session freshness for every reused model.
    privacy_groups: dict[str, list[dict[str, Any]]] = {}
    for artifact_dir in validated_artifacts:
        try:
            com_w, session = _load_privacy_session(artifact_dir)
            privacy_groups.setdefault(com_w, []).append(session)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError, ResultValidationError) as exc:
            errors.append(f"privacy_session_load_failed:{artifact_dir.relative_to(ROOT)}:{exc}")
    privacy_group_audit: list[dict[str, Any]] = []
    for com_w, sessions in sorted(privacy_groups.items()):
        privacy_check = check_multi_session_profiles(sessions)
        if privacy_check.get("status") != "VERIFY_OK":
            errors.append(
                f"multi_session_privacy_failed:{com_w}:{privacy_check.get('reason', 'unknown')}"
            )
        privacy_group_audit.append({
            "com_W_fingerprint": com_w,
            "session_count": len(sessions),
            "status": privacy_check.get("status"),
            "reason": privacy_check.get("reason", ""),
        })
    audit = {
        "schema": "zkGPT_gat_formal_validation_v4",
        "baseline": "zkGPT",
        "status": "verified" if not errors else "not_verified",
        "formal_experiment_eligible": not errors,
        "errors": errors,
        "rows_checked": len(rows),
        "replay_required": True,
        "privacy_groups": privacy_group_audit,
    }
    ensure_dir(results_dir, root=ROOT, label="results root")
    write_json(results_dir / "local_validation.json", audit)
    print(f"zkgpt_formal_validation={audit['status']} errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
