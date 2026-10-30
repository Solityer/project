#!/usr/bin/env python3
"""Narrow, receipt-backed compatibility for a proved-only continuation.

This module deliberately does not relax the normal resume identity.  It only
provides helpers for the explicit Setup-compatibility path and for preserving
the original, small Setup/producer receipts without rescanning large keys.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import copy
from pathlib import Path
from typing import Any

from formal_common import ROOT, ContractError, abort, canonical_json_digest, load_json, require_digest


BASELINE_SESSION_ID = "d3dc1ccb1cfa4a78829c306b835f300e"
BASELINE_MANIFEST_SHA256 = "02d43ff2859ebb6ef4e5f58ebe09eb3582b5a2ba2a06603ac59ccf12a4b7d547"
BASELINE_RELATIVE = Path("runs/check_logs/ezkl_resume_compat_d3dc1ccb1cfa4a78829c306b835f300e/baseline.json")
COMPATIBILITY_SCHEMA = "ezkl.setup-compatible-prove-only.v1"
SOURCE_REVIEW_SCHEMA = "ezkl.compatible-resume-source-review.v1"
REQUEST_SCHEMA = "ezkl.compatible-resume-request.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_COMPAT_OUTPUT_ROOT = "runs/formal_ppi_ogbn_20261029_001"
_COMPAT_MATRIX = f"{_COMPAT_OUTPUT_ROOT}/formal_matrix.json"
_COMPAT_DATASETS = ["ppi", "ogbn_arxiv"]
_REQUIRED_SNAPSHOT_INDEXES = {
    "supplemental_snapshot_index.json",
    "supplemental_snapshot_index_002.json",
    "supplemental_snapshot_index_003.json",
    "supplemental_snapshot_index_004.json",
    "supplemental_snapshot_index_005.json",
}
_SETUP_REUSE_FILES = {
    "setup_timing.json", "admission.json", "settings.json", "compiled.circuit", "pk.key", "vk.key",
    "native_keygen_budget.json", "native_keygen_receipt.json", "gat_static_tables.json",
    "gat_integer_ledger.json", "circuit_ref.json", "producer_context.json",
}
_LARGE_SETUP_FILES = {"compiled.circuit", "pk.key", "vk.key"}


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        abort(f"compatible resume {label} is not a canonical SHA-256")
    return value


def _require_path_ref(value: object, expected_path: str | None, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        abort(f"compatible resume {label} path/digest reference is malformed")
    path = _safe_relative_path(value.get("path"), label)
    if expected_path is not None and path != expected_path:
        abort(f"compatible resume {label} path is not the exact session-bound path")
    _require_sha(value.get("sha256"), f"{label} digest")
    return {"path": path, "sha256": value["sha256"]}


def _require_admission_ref(value: object, expected_path: str, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "settings_sha256"}:
        abort(f"compatible resume {label} reference is malformed")
    result = _require_path_ref({"path": value.get("path"), "sha256": value.get("sha256")}, expected_path, label)
    _require_sha(value.get("settings_sha256"), f"{label} settings digest")
    return {**result, "settings_sha256": value["settings_sha256"]}


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        abort(f"compatible resume {label} must be a regular file: {path}")


def _source_files() -> list[Path]:
    from formal_common import _source_files as enumerate_source_files

    return enumerate_source_files()


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        abort(f"compatible resume {label} path is malformed")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        abort(f"compatible resume {label} path is unsafe: {value}")
    return value


def _snapshot_file(path: Path, *, relative: str, expected_digest: str, expected_size: int | None = None) -> bytes:
    _regular_file(path, f"source snapshot {relative}")
    payload = path.read_bytes()
    if _digest_bytes(payload) != expected_digest or (expected_size is not None and len(payload) != expected_size):
        abort(f"compatible resume source snapshot fails its frozen digest/size: {relative}")
    return payload


def _patch_digest(old_bytes: bytes, new_bytes: bytes, relative: str) -> tuple[str, int, int]:
    old_text = old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    new_text = new_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    patch = "".join(
        difflib.unified_diff(
            old_text,
            new_text,
            fromfile=f"baseline/{relative}" if old_bytes else "/dev/null",
            tofile=f"current/{relative}",
        )
    )
    return (
        _digest_bytes(patch.encode()),
        sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")),
        sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")),
    )


def _bind_snapshot_index(
    *,
    index_path: Path,
    index: dict[str, Any],
    base_dir: Path,
    baseline_by_path: dict[str, dict[str, Any]],
    snapshot_paths: dict[str, list[Path]],
    review_only_snapshots: dict[str, tuple[Path, str, int]],
) -> None:
    """Validate one of the two frozen supplemental index.v1 encodings."""
    plural_schema = "ezkl.formal-resume-supplemental-snapshot-index.v1"
    singular_schema = "ezkl.formal-resume-supplemental-snapshot.v1"
    schema = index.get("schema")
    if schema == plural_schema:
        if set(index) not in (
            {"schema", "session_id", "baseline_json_sha256", "snapshots"},
            {"schema", "session_id", "baseline_json_sha256", "snapshots", "unmanifested_review_snapshots"},
        ):
            abort(f"compatible resume supplemental index has unknown fields: {index_path}")
        if index.get("session_id") != BASELINE_SESSION_ID or index.get("baseline_json_sha256") != BASELINE_MANIFEST_SHA256:
            abort(f"compatible resume supplemental index is not bound to the frozen baseline/session: {index_path}")
        entries = index.get("snapshots")
        if not isinstance(entries, list):
            abort(f"compatible resume supplemental index snapshots are malformed: {index_path}")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "baseline_sha256", "snapshot_sha256", "size_bytes"}:
                abort(f"compatible resume supplemental snapshot entry is malformed: {index_path}")
            relative = _safe_relative_path(entry.get("path"), "supplemental snapshot")
            expected = entry.get("baseline_sha256")
            snapshot_digest = entry.get("snapshot_sha256")
            size = entry.get("size_bytes")
            if (
                not isinstance(expected, str) or not _HEX64.fullmatch(expected)
                or snapshot_digest != expected
                or isinstance(size, bool) or not isinstance(size, int) or size < 0
            ):
                abort(f"compatible resume supplemental snapshot digest/size is malformed: {relative}")
            frozen = baseline_by_path.get(relative)
            if frozen is None or frozen.get("sha256") != expected or frozen.get("size_bytes") != size:
                abort(f"supplemental source snapshot does not match the 874-file baseline: {relative}")
            snapshot = base_dir / "snapshot" / relative
            _snapshot_file(snapshot, relative=relative, expected_digest=expected, expected_size=size)
            snapshot_paths.setdefault(relative, []).append(snapshot)

        review_entries = index.get("unmanifested_review_snapshots", [])
        if not isinstance(review_entries, list):
            abort(f"compatible resume unmanifested review snapshots are malformed: {index_path}")
        for entry in review_entries:
            expected_keys = {"path", "preimage_sha256", "snapshot", "snapshot_sha256", "size_bytes"}
            if not isinstance(entry, dict) or set(entry) != expected_keys:
                abort(f"compatible resume unmanifested review snapshot entry is malformed: {index_path}")
            relative = _safe_relative_path(entry.get("path"), "unmanifested review snapshot")
            if relative in baseline_by_path:
                abort(f"review-only snapshot unexpectedly overlaps the official 874-file census: {relative}")
            digest = entry.get("preimage_sha256")
            snapshot_digest = entry.get("snapshot_sha256")
            size = entry.get("size_bytes")
            snapshot_relative = _safe_relative_path(entry.get("snapshot"), "unmanifested snapshot")
            expected_snapshot = (base_dir / "snapshot_unmanifested" / relative).relative_to(ROOT).as_posix()
            if (
                not isinstance(digest, str) or not _HEX64.fullmatch(digest)
                or snapshot_digest != digest
                or isinstance(size, bool) or not isinstance(size, int) or size < 0
                or snapshot_relative != expected_snapshot
            ):
                abort(f"compatible resume unmanifested review snapshot binding is malformed: {relative}")
            snapshot = ROOT / snapshot_relative
            _snapshot_file(snapshot, relative=relative, expected_digest=digest, expected_size=size)
            prior = review_only_snapshots.get(relative)
            if prior is not None and prior != (snapshot, digest, size):
                abort(f"compatible resume review-only snapshots conflict: {relative}")
            review_only_snapshots[relative] = (snapshot, digest, size)
        return

    if schema == singular_schema:
        if set(index) not in (
            {"schema", "baseline_sha256", "path", "sha256", "snapshot"},
            {"schema", "baseline_json_sha256", "path", "sha256", "snapshot"},
        ):
            abort(f"compatible resume supplemental snapshot index has unknown fields: {index_path}")
        baseline_key = "baseline_sha256" if "baseline_sha256" in index else "baseline_json_sha256"
        if index.get(baseline_key) != BASELINE_MANIFEST_SHA256:
            abort(f"compatible resume supplemental source snapshot index is not bound to the frozen baseline: {index_path}")
        relative = _safe_relative_path(index.get("path"), "supplemental snapshot")
        digest = index.get("sha256")
        snapshot_relative = _safe_relative_path(index.get("snapshot"), "supplemental snapshot")
        frozen = baseline_by_path.get(relative)
        if (
            frozen is None or not isinstance(digest, str) or not _HEX64.fullmatch(digest)
            or frozen.get("sha256") != digest
            or snapshot_relative != (base_dir / "snapshot" / relative).relative_to(ROOT).as_posix()
        ):
            abort(f"supplemental source snapshot does not match baseline file {relative}")
        snapshot = ROOT / snapshot_relative
        _snapshot_file(snapshot, relative=relative, expected_digest=digest, expected_size=frozen.get("size_bytes"))
        snapshot_paths.setdefault(relative, []).append(snapshot)
        return

    abort(f"compatible resume supplemental snapshot index schema is unsupported: {index_path}")


def reviewed_source_delta() -> dict[str, Any]:
    """Return source changes against the frozen 874-file baseline.

    Any changed baseline file must have an immutable pre-edit snapshot.  The
    function never reads or hashes experiment PK/SRS artifacts.
    """
    baseline_path = ROOT / BASELINE_RELATIVE
    _regular_file(baseline_path, "frozen source baseline")
    baseline_raw = baseline_path.read_bytes()
    if _digest_bytes(baseline_raw) != BASELINE_MANIFEST_SHA256:
        abort("compatible resume frozen baseline manifest digest changed")
    baseline = json.loads(baseline_raw)
    if (
        baseline.get("schema") != "ezkl.formal-resume-source-baseline.v1"
        or baseline.get("session_id") != BASELINE_SESSION_ID
        or baseline.get("official_source_file_count") != 874
        or baseline.get("source_tree_sha256") not in (None, baseline.get("session_source_tree_sha256"))
        or baseline.get("session_source_tree_sha256")
        != "e797b2ec8ddab401e5704e9d5dee39824364094bd32a1845ad1612530a249c8e"
    ):
        abort("compatible resume frozen baseline metadata is unsupported")
    records = baseline.get("source_files")
    if not isinstance(records, list) or len(records) != 874:
        abort("compatible resume frozen source file census is malformed")
    baseline_by_path: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not _HEX64.fullmatch(str(item.get("sha256", ""))):
            abort("compatible resume frozen source file entry is malformed")
        relative = item["path"]
        if relative in baseline_by_path or Path(relative).is_absolute() or ".." in Path(relative).parts:
            abort("compatible resume frozen source paths are duplicated or unsafe")
        baseline_by_path[relative] = item

    base_dir = baseline_path.parent
    snapshot_paths: dict[str, list[Path]] = {}
    review_only_snapshots: dict[str, tuple[Path, str, int]] = {}
    builtin_snapshots = baseline.get("snapshots")
    if not isinstance(builtin_snapshots, list):
        abort("compatible resume baseline snapshot list is malformed")
    for value in builtin_snapshots:
        relative = _safe_relative_path(value, "baseline snapshot")
        frozen = baseline_by_path.get(relative)
        if frozen is None:
            abort(f"compatible resume baseline snapshot is not part of its source census: {relative}")
        snapshot = base_dir / "snapshot" / relative
        _snapshot_file(snapshot, relative=relative, expected_digest=frozen["sha256"], expected_size=frozen.get("size_bytes"))
        snapshot_paths.setdefault(relative, []).append(snapshot)

    index_paths = sorted(base_dir.glob("supplemental_snapshot_index*.json"))
    index_relatives = {path.name for path in index_paths}
    if index_relatives != _REQUIRED_SNAPSHOT_INDEXES:
        missing = sorted(_REQUIRED_SNAPSHOT_INDEXES - index_relatives)
        unexpected = sorted(index_relatives - _REQUIRED_SNAPSHOT_INDEXES)
        detail = ", ".join([*(f"missing:{name}" for name in missing), *(f"unexpected:{name}" for name in unexpected)])
        abort(f"compatible resume frozen snapshot index inventory differs from the exact five-index set: {detail}")
    snapshot_index_files: list[dict[str, str]] = []
    for index_path in index_paths:
        _regular_file(index_path, "supplemental snapshot index")
        snapshot_index_files.append({
            "path": index_path.relative_to(ROOT).as_posix(),
            "sha256": _digest_bytes(index_path.read_bytes()),
        })
        index = load_json(index_path, "supplemental source snapshot index")
        if not isinstance(index, dict):
            abort("compatible resume supplemental source snapshot index is malformed")
        _bind_snapshot_index(
            index_path=index_path,
            index=index,
            base_dir=base_dir,
            baseline_by_path=baseline_by_path,
            snapshot_paths=snapshot_paths,
            review_only_snapshots=review_only_snapshots,
        )

    current_paths: dict[str, Path] = {}
    for path in _source_files():
        if path.is_symlink() or not path.is_file():
            abort(f"compatible resume source census contains a non-regular file: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if relative in current_paths:
            abort(f"compatible resume source census contains a duplicate path: {relative}")
        current_paths[relative] = path

    changes: list[dict[str, Any]] = []
    diff_texts: list[str] = []
    for relative, frozen in baseline_by_path.items():
        current = current_paths.pop(relative, None)
        if current is None:
            abort(f"compatible resume source file was removed since the frozen baseline: {relative}")
        current_bytes = current.read_bytes()
        current_hash = _digest_bytes(current_bytes)
        if current_hash == frozen["sha256"]:
            continue
        if relative not in snapshot_paths:
            abort(f"changed compatible-resume source file has no frozen pre-edit snapshot: {relative}")
        old_path = snapshot_paths[relative][0]
        old_bytes = _snapshot_file(old_path, relative=relative, expected_digest=frozen["sha256"], expected_size=frozen.get("size_bytes"))
        diff_sha, added, removed = _patch_digest(old_bytes, current_bytes, relative)
        patch = "".join(difflib.unified_diff(old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True), current_bytes.decode("utf-8", errors="replace").splitlines(keepends=True), fromfile=f"baseline/{relative}", tofile=f"current/{relative}"))
        diff_texts.append(patch)
        changes.append({
            "path": relative,
            "change": "modified",
            "baseline_sha256": frozen["sha256"],
            "current_sha256": current_hash,
            "baseline_size_bytes": frozen.get("size_bytes"),
            "current_size_bytes": len(current_bytes),
            "diff_sha256": diff_sha,
            "diff_added_lines": added,
            "diff_removed_lines": removed,
        })

    for relative, path in sorted(current_paths.items()):
        data = path.read_bytes()
        digest = _digest_bytes(data)
        diff_sha, added, removed = _patch_digest(b"", data, relative)
        patch = "".join(difflib.unified_diff([], data.decode("utf-8", errors="replace").splitlines(keepends=True), fromfile="/dev/null", tofile=f"current/{relative}"))
        diff_texts.append(patch)
        changes.append({
            "path": relative,
            "change": "added",
            "baseline_sha256": None,
            "current_sha256": digest,
            "baseline_size_bytes": 0,
            "current_size_bytes": len(data),
            "diff_sha256": diff_sha,
            "diff_added_lines": added,
            "diff_removed_lines": removed,
        })

    review_only_changes: list[dict[str, Any]] = []
    for relative, (snapshot, old_digest, old_size) in sorted(review_only_snapshots.items()):
        current = ROOT / relative
        if current.is_symlink() or not current.is_file():
            abort(f"compatible resume review-only source is missing or unsafe: {relative}")
        old_bytes = _snapshot_file(snapshot, relative=relative, expected_digest=old_digest, expected_size=old_size)
        current_bytes = current.read_bytes()
        current_digest = _digest_bytes(current_bytes)
        if current_digest == old_digest:
            continue
        diff_sha, added, removed = _patch_digest(old_bytes, current_bytes, relative)
        patch = "".join(difflib.unified_diff(old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True), current_bytes.decode("utf-8", errors="replace").splitlines(keepends=True), fromfile=f"baseline-review/{relative}", tofile=f"current/{relative}"))
        diff_texts.append(patch)
        review_only_changes.append({
            "path": relative,
            "change": "modified",
            "baseline_sha256": old_digest,
            "current_sha256": current_digest,
            "baseline_size_bytes": old_size,
            "current_size_bytes": len(current_bytes),
            "diff_sha256": diff_sha,
            "diff_added_lines": added,
            "diff_removed_lines": removed,
        })
    combined = "\n".join(diff_texts).encode()
    return {
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_source_tree_sha256": baseline["session_source_tree_sha256"],
        "baseline_source_file_count": len(records),
        "current_source_file_count": len(baseline_by_path) + len(current_paths),
        "snapshot_index_files": snapshot_index_files,
        "changed_files": changes,
        "review_only_changed_files": review_only_changes,
        "diff_summary": {
            "modified_file_count": sum(item["change"] == "modified" for item in changes),
            "added_file_count": sum(item["change"] == "added" for item in changes),
            "review_only_changed_file_count": len(review_only_changes),
            "removed_file_count": 0,
            "added_lines": sum(item["diff_added_lines"] for item in changes + review_only_changes),
            "removed_lines": sum(item["diff_removed_lines"] for item in changes + review_only_changes),
            "combined_diff_sha256": _digest_bytes(combined),
        },
    }


def validate_source_review(path: Path, *, expected_session_id: str = BASELINE_SESSION_ID) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Require a separately authored approval for this exact current source delta."""
    _regular_file(path, "source review")
    try:
        path.resolve(strict=True).relative_to((ROOT / "runs" / "check_logs").resolve(strict=True))
    except (OSError, ValueError):
        abort("compatible source review must be stored below runs/check_logs outside the source census")
    document = load_json(path, "compatible resume source review")
    required = {
        "schema", "session_id", "baseline_manifest_sha256", "decision", "reviewer",
        "reviewed_changed_files", "reviewed_review_only_files", "reviewed_snapshot_indexes",
    }
    if not isinstance(document, dict) or set(document) != required:
        abort("compatible source review has unknown, missing, or malformed fields")
    if (
        document.get("schema") != SOURCE_REVIEW_SCHEMA
        or document.get("session_id") != expected_session_id
        or document.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or document.get("decision") != "APPROVED"
        or not isinstance(document.get("reviewer"), str)
        or not document["reviewer"].strip()
    ):
        abort("compatible source review is not an approval bound to the exact baseline/session")
    delta = reviewed_source_delta()

    def approved_projection(items: object, label: str) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            abort(f"compatible source review {label} is malformed")
        projected: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {"path", "change", "baseline_sha256", "current_sha256", "diff_sha256"}:
                abort(f"compatible source review {label} entry is malformed")
            if (
                not isinstance(item.get("path"), str)
                or item.get("change") not in {"modified", "added"}
                or (item.get("baseline_sha256") is not None and not _HEX64.fullmatch(str(item.get("baseline_sha256"))))
                or not isinstance(item.get("current_sha256"), str) or not _HEX64.fullmatch(item["current_sha256"])
                or not isinstance(item.get("diff_sha256"), str) or not _HEX64.fullmatch(item["diff_sha256"])
            ):
                abort(f"compatible source review {label} entry has an invalid identity")
            projected.append({key: item[key] for key in ("path", "change", "baseline_sha256", "current_sha256", "diff_sha256")})
        if projected != sorted(projected, key=lambda item: (item["path"], item["change"])):
            abort(f"compatible source review {label} entries are not in canonical path order")
        return projected

    actual_changed = [
        {key: item[key] for key in ("path", "change", "baseline_sha256", "current_sha256", "diff_sha256")}
        for item in delta["changed_files"]
    ]
    actual_review_only = [
        {key: item[key] for key in ("path", "change", "baseline_sha256", "current_sha256", "diff_sha256")}
        for item in delta["review_only_changed_files"]
    ]
    approved_indexes = document["reviewed_snapshot_indexes"]
    if (
        not isinstance(approved_indexes, list)
        or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in approved_indexes)
        or approved_indexes != sorted(approved_indexes, key=lambda item: item["path"] if isinstance(item.get("path"), str) else "")
        or approved_indexes != delta["snapshot_index_files"]
    ):
        abort("compatible source review does not bind the exact frozen snapshot-index path/digest inventory")
    if approved_projection(document["reviewed_changed_files"], "changed files") != actual_changed:
        abort("compatible source review does not approve the exact current 874-baseline delta")
    if approved_projection(document["reviewed_review_only_files"], "review-only files") != actual_review_only:
        abort("compatible source review does not approve the exact current review-only delta")
    digest = _digest_bytes(path.read_bytes())
    return document, digest, delta


def _snapshot_path(session_id: str) -> str:
    return f"{_COMPAT_OUTPUT_ROOT}/.sessions/{session_id}/compatibility/original_session.json"


def validate_original_session_snapshot(
    path: Path, *, session_id: str = BASELINE_SESSION_ID, expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and authenticate the pre-resume session bytes needed to recover old identity."""
    expected_relative = _snapshot_path(session_id)
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != (ROOT / expected_relative).resolve(strict=True):
        abort("compatible resume original-session snapshot path is missing, unsafe, or not session-bound")
    raw = path.read_bytes()
    digest = _digest_bytes(raw)
    if expected_sha256 is not None and digest != _require_sha(expected_sha256, "original-session snapshot digest"):
        abort("compatible resume original-session snapshot bytes differ from the immutable request")
    try:
        original = json.loads(raw)
    except json.JSONDecodeError as exc:
        abort(f"compatible resume original-session snapshot is invalid JSON: {exc}")
    if not isinstance(original, dict):
        abort("compatible resume original-session snapshot is not an object")
    baseline = load_json(ROOT / BASELINE_RELATIVE, "compatible resume frozen baseline")
    if (
        original.get("schema_version") != 1
        or original.get("baseline") != "ezkl"
        or original.get("session_id") != session_id
        or original.get("status") != "INTERRUPTED"
        or original.get("output_root") != _COMPAT_OUTPUT_ROOT
        or original.get("matrix") != _COMPAT_MATRIX
        or original.get("selected_datasets") != _COMPAT_DATASETS
        or original.get("completed_cells") not in (None, {})
        or original.get("compatible_resume_request") is not None
        or baseline.get("session_id") != session_id
        or original.get("source_tree_sha256") != baseline.get("session_source_tree_sha256")
        or not isinstance(original.get("resume_identity"), dict)
        or original["resume_identity"].get("source_tree_sha256") != baseline.get("session_source_tree_sha256")
    ):
        abort("compatible resume original session does not match the exact interrupted baseline session")
    if set(original.get("setup_reuse_receipts", {})) != {"ppi"}:
        abort("compatible resume original session does not have the exact retained PPI Setup selection")
    admission_receipts = original.get("native_admissions")
    if not isinstance(admission_receipts, dict) or set(admission_receipts) != set(_COMPAT_DATASETS):
        abort("compatible resume original session does not have the exact two-dataset Admission selection")
    for dataset in _COMPAT_DATASETS:
        expected_dir = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{session_id}/admission/{dataset}"
        receipt = admission_receipts[dataset]
        if (
            not isinstance(receipt, dict)
            or receipt.get("prepared_dir") != expected_dir
            or receipt.get("path") != f"{expected_dir}/admission.json"
            or _HEX64.fullmatch(str(receipt.get("sha256", ""))) is None
            or _HEX64.fullmatch(str(receipt.get("settings_sha256", ""))) is None
        ):
            abort(f"compatible resume original {dataset} Admission anchor is malformed")
        admission_path = ROOT / expected_dir / "admission.json"
        settings_path = ROOT / expected_dir / "settings.json"
        _regular_file(admission_path, f"original {dataset} Admission")
        _regular_file(settings_path, f"original {dataset} settings")
        if _digest_bytes(admission_path.read_bytes()) != receipt["sha256"] or _digest_bytes(settings_path.read_bytes()) != receipt["settings_sha256"]:
            abort(f"compatible resume original {dataset} Admission/settings bytes differ from the frozen session anchors")
    setup_anchor = original["setup_reuse_receipts"].get("ppi")
    if not isinstance(setup_anchor, str) or _HEX64.fullmatch(setup_anchor) is None:
        abort("compatible resume original PPI setup receipt anchor is malformed")
    return original, digest


def freeze_original_session_snapshot(session_file: Path, *, session_id: str = BASELINE_SESSION_ID) -> tuple[Path, str, dict[str, Any]]:
    """Freeze the old session JSON before the formal runner mutates resume state."""
    expected_session_file = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{session_id}.json"
    if session_file.is_symlink() or not session_file.is_file() or session_file.resolve(strict=True) != expected_session_file.resolve(strict=True):
        abort("compatible resume must use the exact original session file")
    existing = load_json(session_file, "original EZKL compatible resume session")
    if not isinstance(existing, dict):
        abort("compatible resume original session is malformed")
    compatibility_dir = session_file.parent / session_id / "compatibility"
    if compatibility_dir.is_symlink() or (compatibility_dir.exists() and not compatibility_dir.is_dir()):
        abort("compatible resume session sidecar path is unsafe")
    compatibility_dir.mkdir(parents=True, exist_ok=True)
    snapshot = compatibility_dir / "original_session.json"
    if snapshot.exists() or snapshot.is_symlink():
        original, digest = validate_original_session_snapshot(snapshot, session_id=session_id)
        if existing.get("compatible_resume_request") is None and existing != original:
            abort("compatible resume existing snapshot conflicts with the unmodified original session")
        return snapshot, digest, original
    if existing.get("compatible_resume_request") is not None:
        abort("compatible resume session request exists without its original immutable snapshot")
    original, _ = validate_original_session_snapshot_bytes(session_file.read_bytes(), session_id=session_id)
    raw = session_file.read_bytes()
    digest = _digest_bytes(raw)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(snapshot, flags, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_parent(snapshot.parent)
    validate_original_session_snapshot(snapshot, session_id=session_id, expected_sha256=digest)
    return snapshot, digest, original


def validate_original_session_snapshot_bytes(raw: bytes, *, session_id: str = BASELINE_SESSION_ID) -> tuple[dict[str, Any], str]:
    """Validate original session bytes before they are copied into the sidecar."""
    try:
        original = json.loads(raw)
    except json.JSONDecodeError as exc:
        abort(f"compatible resume original session is invalid JSON: {exc}")
    if not isinstance(original, dict):
        abort("compatible resume original session is not an object")
    # Reuse the same path-independent checks by validating required identity and anchors here.
    baseline = load_json(ROOT / BASELINE_RELATIVE, "compatible resume frozen baseline")
    if (
        original.get("schema_version") != 1 or original.get("baseline") != "ezkl"
        or original.get("session_id") != session_id or original.get("status") != "INTERRUPTED"
        or original.get("output_root") != _COMPAT_OUTPUT_ROOT or original.get("matrix") != _COMPAT_MATRIX
        or original.get("selected_datasets") != _COMPAT_DATASETS or original.get("completed_cells") not in (None, {})
        or original.get("compatible_resume_request") is not None
        or original.get("source_tree_sha256") != baseline.get("session_source_tree_sha256")
        or not isinstance(original.get("resume_identity"), dict)
        or original["resume_identity"].get("source_tree_sha256") != baseline.get("session_source_tree_sha256")
        or not isinstance(original.get("setup_reuse_receipts"), dict)
        or set(original["setup_reuse_receipts"]) != {"ppi"}
        or not isinstance(original.get("native_admissions"), dict)
        or set(original["native_admissions"]) != set(_COMPAT_DATASETS)
    ):
        abort("compatible resume original session does not match the exact interrupted baseline session")
    return original, _digest_bytes(raw)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def admission_semantics_digest(admission: dict[str, Any], *, dataset: str) -> str:
    if not isinstance(admission, dict):
        abort(f"{dataset} compatible Admission is not an object")
    memory = admission.get("native_memory_model")
    if not isinstance(memory, dict) or "cuda_free_bytes_at_admission" not in memory:
        abort(f"{dataset} compatible Admission is missing its CUDA free-memory sample")
    free = memory["cuda_free_bytes_at_admission"]
    total = memory.get("cuda_total_memory_bytes")
    if (
        isinstance(free, bool) or not isinstance(free, int) or free < 0
        or isinstance(total, bool) or not isinstance(total, int) or total < 0 or free > total
    ):
        abort(f"{dataset} compatible Admission CUDA capacity sample is invalid")
    projection = copy.deepcopy(admission)
    projection["native_memory_model"].pop("cuda_free_bytes_at_admission")
    return canonical_json_digest(projection)


def compatible_source_review_index_digest(delta: dict[str, Any]) -> str:
    indexes = delta.get("snapshot_index_files")
    if not isinstance(indexes, list) or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in indexes):
        abort("compatible source delta lacks the exact snapshot-index inventory")
    return canonical_json_digest(indexes)


def validate_compatible_resume_request(
    session_document: dict[str, Any], *, expected_session_id: str = BASELINE_SESSION_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable compatible-resume request and return its frozen old session."""
    request = session_document.get("compatible_resume_request") if isinstance(session_document, dict) else None
    required = {
        "schema", "session_id", "output_root", "matrix", "selected_datasets", "retained_setup_datasets",
        "source_review_path", "source_review_sha256", "baseline_manifest_sha256", "baseline_source_tree_sha256",
        "source_delta_sha256", "snapshot_index_inventory_sha256", "old_session_snapshot_path", "old_session_snapshot_sha256",
    }
    if not isinstance(request, dict) or set(request) != required:
        abort("compatible resume session request has unknown, missing, or malformed fields")
    expected_snapshot_path = _snapshot_path(expected_session_id)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("session_id") != expected_session_id
        or session_document.get("session_id") != expected_session_id
        or request.get("output_root") != _COMPAT_OUTPUT_ROOT
        or request.get("matrix") != _COMPAT_MATRIX
        or request.get("selected_datasets") != _COMPAT_DATASETS
        or request.get("retained_setup_datasets") != ["ppi"]
        or request.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or request.get("old_session_snapshot_path") != expected_snapshot_path
        or _HEX64.fullmatch(str(request.get("baseline_source_tree_sha256", ""))) is None
        or _HEX64.fullmatch(str(request.get("source_delta_sha256", ""))) is None
        or _HEX64.fullmatch(str(request.get("snapshot_index_inventory_sha256", ""))) is None
    ):
        abort("compatible resume request is not bound to the exact session, output, selection, and baseline")
    baseline = load_json(ROOT / BASELINE_RELATIVE, "compatible resume frozen baseline")
    if request.get("baseline_source_tree_sha256") != baseline.get("session_source_tree_sha256"):
        abort("compatible resume request baseline source tree differs from the frozen 874-file baseline")
    snapshot_path = ROOT / expected_snapshot_path
    old_session, old_session_sha256 = validate_original_session_snapshot(
        snapshot_path,
        session_id=expected_session_id,
        expected_sha256=request.get("old_session_snapshot_sha256"),
    )
    source_review_path = ROOT / _safe_relative_path(request.get("source_review_path"), "source review")
    try:
        source_review_path.resolve(strict=True).relative_to((ROOT / "runs" / "check_logs").resolve(strict=True))
    except (OSError, ValueError):
        abort("compatible resume request source review is outside runs/check_logs")
    review, review_sha256, delta = validate_source_review(source_review_path, expected_session_id=expected_session_id)
    if (
        review_sha256 != _require_sha(request.get("source_review_sha256"), "source-review digest")
        or review.get("decision") != "APPROVED"
        or delta["diff_summary"]["combined_diff_sha256"] != request.get("source_delta_sha256")
        or compatible_source_review_index_digest(delta) != request.get("snapshot_index_inventory_sha256")
    ):
        abort("compatible resume request source review/delta/index inventory changed")
    if session_document.get("selected_datasets") != _COMPAT_DATASETS:
        abort("compatible resume active session selection differs from the immutable request")
    old_setups = old_session.get("setup_reuse_receipts")
    if not isinstance(old_setups, dict) or set(old_setups) != {"ppi"}:
        abort("compatible resume original setup anchor selection changed")
    return request, old_session


def compare_admission_semantics(old_admission: dict[str, Any], current_admission: dict[str, Any], *, dataset: str) -> None:
    """Compare full admitted identity, allowing only the approved GPU free-memory sample to vary."""
    if not isinstance(old_admission, dict) or not isinstance(current_admission, dict):
        abort(f"{dataset} compatible Admission is not a JSON object")
    old_memory = old_admission.get("native_memory_model")
    current_memory = current_admission.get("native_memory_model")
    if not isinstance(old_memory, dict) or not isinstance(current_memory, dict):
        abort(f"{dataset} compatible Admission has no exact native memory model")
    old_free = old_memory.get("cuda_free_bytes_at_admission")
    current_free = current_memory.get("cuda_free_bytes_at_admission")
    old_total = old_memory.get("cuda_total_memory_bytes")
    current_total = current_memory.get("cuda_total_memory_bytes")
    for label, value in (("old free", old_free), ("current free", current_free), ("old total", old_total), ("current total", current_total)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            abort(f"{dataset} compatible Admission {label} CUDA capacity sample is invalid")
    if old_total != current_total or old_free > old_total or current_free > current_total:
        abort(f"{dataset} compatible Admission CUDA capacity samples do not bind the same physical memory size")
    old_projection = copy.deepcopy(old_admission)
    current_projection = copy.deepcopy(current_admission)
    old_projection["native_memory_model"].pop("cuda_free_bytes_at_admission")
    current_projection["native_memory_model"].pop("cuda_free_bytes_at_admission")
    if canonical_json_digest(old_projection) != canonical_json_digest(current_projection):
        old_keys = set(old_projection)
        current_keys = set(current_projection)
        changed = sorted(key for key in old_keys | current_keys if old_projection.get(key) != current_projection.get(key))
        abort(f"{dataset} compatible Admission semantic projection differs outside the approved free-memory sample: {', '.join(changed)}")


def validate_retained_setup_chain(
    original_session: dict[str, Any], *, session_id: str = BASELINE_SESSION_ID,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Authenticate the original PPI Setup chain without rehashing multi-GB keys."""
    snapshot_path = ROOT / _snapshot_path(session_id)
    frozen_session, snapshot_sha256 = validate_original_session_snapshot(snapshot_path, session_id=session_id)
    if original_session != frozen_session:
        abort("compatible retained Setup source is not the exact frozen original session")
    if set(frozen_session.get("setup_reuse_receipts", {})) != {"ppi"}:
        abort("compatible retained Setup selection is not exactly PPI")

    output = ROOT / _COMPAT_OUTPUT_ROOT / "ppi"
    session_base = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{session_id}"
    setup_receipt_path = output / "setup_receipt.json"
    budget_path = output / "native_keygen_budget.json"
    keygen_receipt_path = output / "native_keygen_receipt.json"
    for path, label in (
        (setup_receipt_path, "retained PPI Setup receipt"),
        (budget_path, "retained PPI native keygen budget"),
        (keygen_receipt_path, "retained PPI native keygen receipt"),
    ):
        _regular_file(path, label)

    setup_receipt_sha256 = _digest_bytes(setup_receipt_path.read_bytes())
    if setup_receipt_sha256 != frozen_session["setup_reuse_receipts"]["ppi"]:
        abort("retained PPI Setup receipt bytes differ from the immutable original session anchor")
    setup_receipt = load_json(setup_receipt_path, "retained PPI Setup receipt")
    if (
        not isinstance(setup_receipt, dict)
        or set(setup_receipt) != {
            "schema", "session_id", "dataset", "resume_identity_sha256", "admission_sha256",
            "native_keygen_receipt_sha256", "setup_files",
        }
        or setup_receipt.get("schema") != "ezkl.setup-reuse-receipt.v1"
        or setup_receipt.get("session_id") != session_id
        or setup_receipt.get("dataset") != "ppi"
        or setup_receipt.get("resume_identity_sha256") != canonical_json_digest(frozen_session["resume_identity"])
    ):
        abort("retained PPI Setup receipt is not bound to the frozen old Setup identity")

    old_admission_anchor = frozen_session["native_admissions"]["ppi"]
    old_admission_path = ROOT / old_admission_anchor["path"]
    old_admission = load_json(old_admission_path, "retained PPI Setup Admission")
    old_admission_sha256 = _digest_bytes(old_admission_path.read_bytes())
    if setup_receipt.get("admission_sha256") != old_admission_sha256 or old_admission_sha256 != old_admission_anchor["sha256"]:
        abort("retained PPI Setup receipt is not bound to the frozen old PPI Admission")

    setup_files = setup_receipt.get("setup_files")
    if not isinstance(setup_files, dict) or set(setup_files) != _SETUP_REUSE_FILES:
        abort("retained PPI Setup receipt file census is unknown, incomplete, or malformed")
    for name, digest in setup_files.items():
        _require_sha(digest, f"retained PPI {name} digest")
        path = output / name
        _regular_file(path, f"retained PPI Setup artifact {name}")
        if path.stat().st_size <= 0:
            abort(f"retained PPI Setup artifact is empty: {name}")
        if name not in _LARGE_SETUP_FILES and _digest_bytes(path.read_bytes()) != digest:
            abort(f"retained PPI Setup artifact bytes differ from the immutable Setup receipt: {name}")
    if setup_files["admission.json"] != old_admission_sha256 or setup_files["settings.json"] != old_admission_anchor["settings_sha256"]:
        abort("retained PPI output Admission/settings digests differ from the original Admission anchors")

    budget_sha256 = _digest_bytes(budget_path.read_bytes())
    keygen_receipt_sha256 = _digest_bytes(keygen_receipt_path.read_bytes())
    if (
        setup_files["native_keygen_budget.json"] != budget_sha256
        or setup_files["native_keygen_receipt.json"] != keygen_receipt_sha256
        or setup_receipt.get("native_keygen_receipt_sha256") != keygen_receipt_sha256
    ):
        abort("retained PPI keygen receipt/budget bytes differ from the original Setup census")
    budget = load_json(budget_path, "retained PPI native keygen budget")
    keygen_receipt = load_json(keygen_receipt_path, "retained PPI native keygen receipt")
    memory = old_admission.get("native_memory_model") if isinstance(old_admission, dict) else None
    relation = old_admission.get("relation_identity") if isinstance(old_admission, dict) else None
    if (
        not isinstance(budget, dict)
        or budget.get("schema") != "ezkl.native-keygen-budget.v3"
        or budget.get("session_id") != session_id
        or budget.get("dataset") != "ppi"
        or budget.get("admission_sha256") != old_admission_sha256
        or budget.get("settings_sha256") != old_admission_anchor["settings_sha256"]
        or budget.get("binary_sha256") != frozen_session["resume_identity"].get("binary_sha256")
        or budget.get("build_receipt_sha256") != frozen_session["resume_identity"].get("build_receipt_sha256")
        or budget.get("build_environment_sha256") != frozen_session["resume_identity"].get("build_environment_sha256")
        or not isinstance(keygen_receipt, dict)
        or keygen_receipt.get("schema") != "ezkl.native-keygen-receipt.v4"
        or keygen_receipt.get("session_id") != session_id
        or keygen_receipt.get("dataset") != "ppi"
        or keygen_receipt.get("budget_receipt_sha256") != budget_sha256
        or keygen_receipt.get("settings_sha256") != old_admission_anchor["settings_sha256"]
        or keygen_receipt.get("compiled_circuit_sha256") != setup_files["compiled.circuit"]
        or keygen_receipt.get("vk_sha256") != setup_files["vk.key"]
        or keygen_receipt.get("pk_sha256") != setup_files["pk.key"]
        or keygen_receipt.get("relation_identity") != relation
        or not isinstance(memory, dict)
    ):
        abort("retained PPI native keygen budget/census is not bound to the original binary, Admission, and Setup artifacts")

    tables_path = output / "gat_static_tables.json"
    ledger_path = output / "gat_integer_ledger.json"
    tables = load_json(tables_path, "retained PPI static tables")
    ledger = load_json(ledger_path, "retained PPI integer ledger")
    table_identity = relation.get("tables") if isinstance(relation, dict) else None
    ledger_identity = relation.get("ledger_identity") if isinstance(relation, dict) else None
    if (
        not isinstance(table_identity, dict)
        or tables.get("static_tables_id") != table_identity.get("static_tables_id")
        or tables.get("static_tables_digest") != table_identity.get("static_tables_digest")
        or not isinstance(ledger_identity, dict)
        or ledger.get("ledger_id") != ledger_identity.get("id")
        or ledger.get("ledger_digest") != ledger_identity.get("digest")
    ):
        abort("retained PPI Setup tables or integer ledger differ from the original Admission identity")

    setup_identity = {
        "schema": "ezkl.retained-setup-identity.v1",
        "session_id": session_id,
        "dataset": "ppi",
        "old_resume_identity_sha256": canonical_json_digest(frozen_session["resume_identity"]),
        "admission_sha256": old_admission_sha256,
        "settings_sha256": old_admission_anchor["settings_sha256"],
        "setup_receipt_sha256": setup_receipt_sha256,
        "native_keygen_budget_sha256": budget_sha256,
        "native_keygen_receipt_sha256": keygen_receipt_sha256,
        "setup_files": {name: setup_files[name] for name in sorted(setup_files)},
    }
    identity_sha256 = canonical_json_digest(setup_identity)
    refs = {
        "origin": "retained_old",
        "setup_identity_sha256": identity_sha256,
        "setup_receipt": {"path": f"{_COMPAT_OUTPUT_ROOT}/ppi/setup_receipt.json", "sha256": setup_receipt_sha256},
        "native_keygen_budget": {"path": f"{_COMPAT_OUTPUT_ROOT}/ppi/native_keygen_budget.json", "sha256": budget_sha256},
        "native_keygen_receipt": {"path": f"{_COMPAT_OUTPUT_ROOT}/ppi/native_keygen_receipt.json", "sha256": keygen_receipt_sha256},
        "setup_files": {
            "settings_sha256": setup_files["settings.json"],
            "compiled_circuit_sha256": setup_files["compiled.circuit"],
            "vk_sha256": setup_files["vk.key"],
            "pk_sha256": setup_files["pk.key"],
            "static_tables_sha256": setup_files["gat_static_tables.json"],
            "integer_ledger_sha256": setup_files["gat_integer_ledger.json"],
        },
        "bound_setup_admission": {
            "path": f"{session_base}/admission/ppi/admission.json",
            "sha256": old_admission_sha256,
            "settings_sha256": old_admission_anchor["settings_sha256"],
        },
    }
    return setup_identity, identity_sha256, refs


def validate_original_artifact_manifest(
    original_session: dict[str, Any], *, session_id: str = BASELINE_SESSION_ID,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Authenticate the original PPI artifact-manifest snapshot from its old final-stage receipt."""
    frozen_session, _session_snapshot_sha256 = validate_original_session_snapshot(
        ROOT / _snapshot_path(session_id), session_id=session_id,
    )
    if original_session != frozen_session:
        abort("original artifact manifest source is not the exact frozen session snapshot")
    stage_receipts = frozen_session.get("stage_receipts")
    final_anchor = stage_receipts.get("ppi", {}).get("final") if isinstance(stage_receipts, dict) else None
    if (
        not isinstance(final_anchor, dict)
        or set(final_anchor) != {"receipt", "sha256"}
        or not isinstance(final_anchor.get("receipt"), dict)
        or final_anchor.get("sha256") != canonical_json_digest(final_anchor["receipt"])
    ):
        abort("original PPI final-stage receipt is absent or not digest-anchored")
    final_receipt = final_anchor["receipt"]
    expected_final_keys = {
        "schema", "session_id", "dataset", "stage", "resume_identity_sha256", "admission_sha256",
        "setup_receipt_sha256", "inputs_sha256", "outputs_sha256", "elapsed_ms", "recorded_at_utc",
        "artifact_manifest_snapshot_sha256",
    }
    if (
        set(final_receipt) != expected_final_keys
        or final_receipt.get("schema") != "ezkl.formal-stage-receipt.v1"
        or final_receipt.get("session_id") != session_id
        or final_receipt.get("dataset") != "ppi"
        or final_receipt.get("stage") != "final"
        or final_receipt.get("resume_identity_sha256") != canonical_json_digest(frozen_session["resume_identity"])
    ):
        abort("original PPI final-stage receipt has an unsupported or mismatched identity")
    snapshot_sha256 = final_receipt.get("artifact_manifest_snapshot_sha256")
    _require_sha(snapshot_sha256, "original artifact manifest snapshot digest")
    relative = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{session_id}/stage-artifacts/ppi/artifact_manifest.final.json"
    snapshot_path = ROOT / relative
    _regular_file(snapshot_path, "original artifact manifest snapshot")
    raw = snapshot_path.read_bytes()
    if _digest_bytes(raw) != snapshot_sha256:
        abort("original PPI artifact-manifest snapshot differs from the original final-stage anchor")
    original_manifest = load_json(snapshot_path, "original PPI artifact manifest snapshot")
    required_manifest_keys = {
        "schema_version", "baseline", "relation_id", "dataset", "backend", "proof_scope", "privacy_policy",
        "public_statement_mode", "output_mode", "quant_cfg_id", "static_tables_id", "circuit_ref",
        "compiled_circuit_id", "settings_id", "vk_id", "binary_sha256", "source_tree_sha256",
        "resource_manifest_sha256", "srs_sha256", "settings_sha256", "compiled_circuit_sha256", "vk_sha256",
        "pk_sha256", "srs_provenance", "files", "source_snapshot", "statement_sha256",
        "producer_manifest_sha256", "ezkl_security_profile_id",
    }
    if (
        not isinstance(original_manifest, dict)
        or set(original_manifest) != required_manifest_keys
        or original_manifest.get("schema_version") != 1
        or original_manifest.get("dataset") != "ppi"
        or original_manifest.get("backend") != "gpu"
    ):
        abort("original PPI artifact-manifest snapshot has an unsupported header")
    files = original_manifest.get("files")
    if not isinstance(files, list) or not files:
        abort("original PPI artifact-manifest snapshot has no file census")
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            abort(f"original PPI artifact-manifest file record {index} is malformed")
        name, size, digest = record["path"], record["size"], record["sha256"]
        if (
            not isinstance(name, str) or not name or Path(name).is_absolute()
            or ".." in Path(name).parts or name in records
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0
        ):
            abort(f"original PPI artifact-manifest file record {index} has an unsafe path or size")
        _require_sha(digest, f"original PPI artifact-manifest {name} digest")
        records[name] = record
    expected_files = {
        "settings.json", "compiled.circuit", "pk.key", "vk.key", "circuit_ref.json", "statement.json",
        "public_output.json", "srs.kzg", "producer_context.json", "producer_manifest.json",
        "gat_static_tables.json", "gat_integer_ledger.json", "admission.json", "setup_receipt.json",
        "native_keygen_budget.json", "native_keygen_receipt.json", "native_prove_budget.json",
    }
    if set(records) != expected_files:
        abort("original PPI artifact-manifest file census differs from the exact pre-proof membership")
    projection = dict(original_manifest)
    projection["files"] = [
        record for record in files
        if not isinstance(record, dict) or record.get("path") not in {"proof.json", "native_pk_read_timing.json"}
    ]
    outputs = final_receipt.get("outputs_sha256")
    if not isinstance(outputs, dict) or set(outputs) != {"artifact_manifest", "srs", "final_log"}:
        abort("original PPI final-stage receipt has an unsupported output identity map")
    if outputs.get("artifact_manifest") != canonical_json_digest(projection):
        abort("original PPI artifact manifest does not match its old final-stage output digest")
    ref = {"path": relative, "sha256": snapshot_sha256}
    return original_manifest, snapshot_sha256, ref


def validate_retained_witness_chain(
    original_session: dict[str, Any], *, session_id: str = BASELINE_SESSION_ID,
) -> dict[str, Any]:
    """Authenticate the original PPI Witness stage and retained witness bytes."""
    frozen_session, session_snapshot_sha256 = validate_original_session_snapshot(
        ROOT / _snapshot_path(session_id), session_id=session_id,
    )
    if original_session != frozen_session:
        abort("retained Witness source is not the exact frozen original session")
    if set(frozen_session.get("setup_reuse_receipts", {})) != {"ppi"}:
        abort("retained Witness Setup selection is not exactly PPI")
    original_manifest, _manifest_sha256, _manifest_ref = validate_original_artifact_manifest(
        frozen_session, session_id=session_id,
    )
    records = {record["path"]: record for record in original_manifest["files"]}
    receipt_anchor = frozen_session.get("stage_receipts", {}).get("ppi", {}).get("witness")
    if (
        not isinstance(receipt_anchor, dict)
        or set(receipt_anchor) != {"receipt", "sha256"}
        or not isinstance(receipt_anchor.get("receipt"), dict)
        or receipt_anchor.get("sha256") != canonical_json_digest(receipt_anchor["receipt"])
    ):
        abort("original PPI Witness stage receipt is absent or not digest-anchored")
    receipt = receipt_anchor["receipt"]
    expected_receipt_keys = {
        "schema", "session_id", "dataset", "stage", "resume_identity_sha256", "admission_sha256",
        "setup_receipt_sha256", "native_prove_budget_sha256", "inputs_sha256", "outputs_sha256",
        "elapsed_ms", "recorded_at_utc",
    }
    old_admission = frozen_session.get("native_admissions", {}).get("ppi")
    old_setup_sha256 = frozen_session.get("setup_reuse_receipts", {}).get("ppi")
    old_identity = frozen_session.get("resume_identity")
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("schema") != "ezkl.formal-stage-receipt.v1"
        or receipt.get("session_id") != session_id
        or receipt.get("dataset") != "ppi"
        or receipt.get("stage") != "witness"
        or not isinstance(old_admission, dict)
        or receipt.get("admission_sha256") != old_admission.get("sha256")
        or not isinstance(old_setup_sha256, str)
        or receipt.get("setup_receipt_sha256") != old_setup_sha256
        or receipt.get("resume_identity_sha256") != canonical_json_digest(old_identity)
    ):
        abort("original PPI Witness stage receipt has a mismatched old identity")
    elapsed = receipt.get("elapsed_ms")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed <= 0:
        abort("original PPI Witness stage receipt has an invalid timing")

    compatibility_dir = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility"
    budget_relative = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{session_id}/compatibility/legacy/native_prove_budget.json"
    old_budget_path = ROOT / budget_relative
    _regular_file(old_budget_path, "frozen old native prove budget")
    old_budget_sha256 = _digest_bytes(old_budget_path.read_bytes())
    old_budget = load_json(old_budget_path, "frozen old native prove budget")
    if (
        not isinstance(old_budget, dict)
        or old_budget.get("schema") != "ezkl.native-prove-budget.v8"
        or old_budget.get("session_id") != session_id
        or old_budget.get("dataset") != "ppi"
        or old_budget_sha256 != receipt.get("native_prove_budget_sha256")
        or old_budget_sha256 != records.get("native_prove_budget.json", {}).get("sha256")
        or old_budget.get("admission_sha256") != old_admission.get("sha256")
        or old_budget.get("setup_receipt_sha256") != old_setup_sha256
        or old_budget.get("binary_sha256") != old_identity.get("binary_sha256")
        or old_budget.get("build_receipt_sha256") != old_identity.get("build_receipt_sha256")
        or old_budget.get("build_environment_sha256") != old_identity.get("build_environment_sha256")
    ):
        abort("retained PPI Witness is not bound to the immutable old prove budget")

    old_admission_path = ROOT / old_admission["path"]
    old_admission_bytes = _snapshot_file(
        old_admission_path,
        relative=str(old_admission["path"]),
        expected_digest=_require_sha(old_admission.get("sha256"), "old PPI Admission"),
    )
    old_admission_doc = json.loads(old_admission_bytes)
    old_identity_configs = old_identity.get("config_sha256s") if isinstance(old_identity, dict) else None
    config_sha = None
    if isinstance(old_identity_configs, list):
        for entry in old_identity_configs:
            if isinstance(entry, list) and len(entry) == 2 and entry[0] == "ppi":
                config_sha = entry[1]
                break
    if not isinstance(config_sha, str):
        abort("old PPI Witness identity has no exact secure-config digest")
    expected_inputs = {
        "compiled_circuit": records.get("compiled.circuit", {}).get("sha256"),
        "native_prove_budget": old_budget_sha256,
        "public_output": records.get("public_output.json", {}).get("sha256"),
        "resource_manifest": original_manifest.get("resource_manifest_sha256"),
        "secure_config": config_sha,
        "settings": records.get("settings.json", {}).get("sha256"),
        "srs": records.get("srs.kzg", {}).get("sha256"),
        "statement": records.get("statement.json", {}).get("sha256"),
    }
    inputs = receipt.get("inputs_sha256")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs) or inputs != expected_inputs:
        abort("original PPI Witness receipt inputs differ from the frozen source/prove/statement/settings artifacts")
    witness_relative = f"{_COMPAT_OUTPUT_ROOT}/ppi/witness.json"
    witness_path = ROOT / witness_relative
    _regular_file(witness_path, "retained PPI witness")
    witness_sha256 = _digest_bytes(witness_path.read_bytes())
    log_relative = f"{_COMPAT_OUTPUT_ROOT}/ppi/logs/witness.log"
    witness_log_path = ROOT / log_relative
    _regular_file(witness_log_path, "retained PPI witness log")
    witness_log_sha256 = _digest_bytes(witness_log_path.read_bytes())
    expected_outputs = {"witness": witness_sha256, "witness_log": witness_log_sha256}
    if receipt.get("outputs_sha256") != expected_outputs or witness_path.stat().st_size <= 0:
        abort("retained PPI witness bytes/log differ from the frozen Witness stage receipt")
    witness_record = {
        "origin": "retained_old",
        "session_snapshot": {
            "path": _snapshot_path(session_id),
            "sha256": session_snapshot_sha256,
        },
        "stage_receipt_sha256": receipt_anchor["sha256"],
        "witness": {
            "path": witness_relative,
            "sha256": witness_sha256,
            "size_bytes": witness_path.stat().st_size,
        },
        "old_native_prove_budget": {
            "path": budget_relative,
            "sha256": old_budget_sha256,
        },
        "old_admission_sha256": old_admission["sha256"],
        "old_setup_receipt_sha256": old_setup_sha256,
    }
    return witness_record


def _compatible_artifact_manifest_projection(
    original_manifest: dict[str, Any],
    *,
    current_identity: dict[str, Any],
    prove_budget_path: Path,
    receipt_copy_path: Path,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Build the only permitted pre-Prove current-manifest transform."""
    _require_sha(current_identity.get("binary_sha256"), "current manifest binary identity")
    _require_sha(current_identity.get("source_tree_sha256"), "current manifest source identity")
    _require_sha(receipt_sha256, "local compatibility receipt digest")
    if prove_budget_path.is_symlink() or not prove_budget_path.is_file():
        abort("compatible current native prove budget must be a regular file")
    if receipt_copy_path.is_symlink() or not receipt_copy_path.is_file():
        abort("compatible local receipt copy must be a regular file")
    budget_sha256 = _digest_bytes(prove_budget_path.read_bytes())
    receipt_bytes = receipt_copy_path.read_bytes()
    if _digest_bytes(receipt_bytes) != receipt_sha256:
        abort("compatible local receipt copy does not match its authoritative receipt digest")
    authoritative_receipt_path = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID / "compatibility" / "compatible_resume_receipt.json"
    _regular_file(authoritative_receipt_path, "authoritative compatible resume receipt")
    authoritative_receipt, authoritative_receipt_sha256 = validate_compatibility_receipt(
        authoritative_receipt_path, session_id=BASELINE_SESSION_ID,
    )
    if (
        authoritative_receipt_sha256 != receipt_sha256
        or authoritative_receipt_path.read_bytes() != receipt_bytes
        or authoritative_receipt.get("current_prover_identity_sha256") != canonical_json_digest(current_identity)
    ):
        abort("compatible receipt copy, authoritative sidecar, and current Prover identity differ")

    expected = copy.deepcopy(original_manifest)
    expected["binary_sha256"] = current_identity["binary_sha256"]
    expected["source_tree_sha256"] = current_identity["source_tree_sha256"]
    files = expected.get("files")
    if not isinstance(files, list):
        abort("original compatible artifact manifest file census is malformed")
    budget_matches = [record for record in files if isinstance(record, dict) and record.get("path") == "native_prove_budget.json"]
    receipt_matches = [record for record in files if isinstance(record, dict) and record.get("path") == "compatible_resume_receipt.json"]
    if len(budget_matches) != 1 or receipt_matches:
        abort("original artifact manifest has an unexpected prove-budget or compatibility-receipt census")
    budget_matches[0] = {
        "path": "native_prove_budget.json",
        "size": prove_budget_path.stat().st_size,
        "sha256": budget_sha256,
    }
    files.append({
        "path": "compatible_resume_receipt.json",
        "size": receipt_copy_path.stat().st_size,
        "sha256": receipt_sha256,
    })
    return expected


def validate_compatible_artifact_manifest(
    artifact_manifest_path: Path,
    *,
    original_session: dict[str, Any],
    session_id: str = BASELINE_SESSION_ID,
    current_identity: dict[str, Any],
    proof_complete: bool = False,
) -> tuple[dict[str, Any], str]:
    """Validate the controlled old-to-current manifest transform without rehashing PK/SRS."""
    original_manifest, _old_manifest_sha256, _old_manifest_ref = validate_original_artifact_manifest(
        original_session, session_id=session_id,
    )
    _setup_identity, _setup_identity_sha256, setup_refs = validate_retained_setup_chain(
        original_session, session_id=session_id,
    )
    output = ROOT / _COMPAT_OUTPUT_ROOT / "ppi"
    if artifact_manifest_path.resolve(strict=False) != (output / "artifact_manifest.json").resolve(strict=False):
        abort("compatible current artifact manifest is not at the exact PPI result path")
    expected = _compatible_artifact_manifest_projection(
        original_manifest,
        current_identity=current_identity,
        prove_budget_path=output / "native_prove_budget.json",
        receipt_copy_path=output / "compatible_resume_receipt.json",
        receipt_sha256=_digest_bytes((output / "compatible_resume_receipt.json").read_bytes())
        if (output / "compatible_resume_receipt.json").is_file() and not (output / "compatible_resume_receipt.json").is_symlink()
        else "",
    )
    current = load_json(artifact_manifest_path, "compatible current artifact manifest")
    current_files = current.get("files") if isinstance(current, dict) else None
    if not isinstance(current_files, list):
        abort("compatible current artifact manifest has no file census")
    expected_records = expected["files"]
    if proof_complete:
        proof_path = output / "proof.json"
        timing_path = output / "native_pk_read_timing.json"
        for path, name in ((proof_path, "proof.json"), (timing_path, "native_pk_read_timing.json")):
            _regular_file(path, f"current native {name}")
            expected_records.append({
                "path": name,
                "size": path.stat().st_size,
                "sha256": _digest_bytes(path.read_bytes()),
            })
    expected["files"] = expected_records
    if current != expected:
        abort("current PPI artifact manifest contains changes outside the approved compatibility transform")

    # The old native manifest and retained Setup chain authorize the three
    # large inputs by their old digests plus current regular-file sizes. The
    # native Prove parser will read the PK; avoid a redundant Python rehash.
    records = {record["path"]: record for record in current_files if isinstance(record, dict)}
    unchanged_large = set(_LARGE_SETUP_FILES) | {"srs.kzg"}
    for name, record in records.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or Path(name).parts[0] == ".":
            abort(f"compatible current artifact has an unsafe file path: {name}")
        target = output / name
        cursor = output
        for component in Path(name).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                abort(f"compatible current artifact path contains a symlink: {name}")
        _regular_file(target, f"current artifact {name}")
        if target.stat().st_size != record["size"]:
            abort(f"compatible current artifact size differs from the manifest: {name}")
        if name not in unchanged_large and _digest_bytes(target.read_bytes()) != record["sha256"]:
            abort(f"compatible current artifact bytes differ from the manifest: {name}")
    setup_files = setup_refs["setup_files"]
    for name, field in (
        ("settings.json", "settings_sha256"),
        ("compiled.circuit", "compiled_circuit_sha256"),
        ("vk.key", "vk_sha256"),
        ("pk.key", "pk_sha256"),
        ("gat_static_tables.json", "static_tables_sha256"),
        ("gat_integer_ledger.json", "integer_ledger_sha256"),
    ):
        if records.get(name, {}).get("sha256") != setup_files.get(field):
            abort(f"compatible current artifact {name} is not bound to the retained Setup chain")
    return current, _digest_bytes(artifact_manifest_path.read_bytes())


def install_compatible_artifact_manifest(
    artifact_manifest_path: Path,
    *,
    original_session: dict[str, Any],
    session_id: str = BASELINE_SESSION_ID,
    current_identity: dict[str, Any],
    receipt_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Install the approved pre-Prove manifest over only the authenticated original manifest."""
    original_manifest, old_manifest_sha256, _old_ref = validate_original_artifact_manifest(
        original_session, session_id=session_id,
    )
    output = ROOT / _COMPAT_OUTPUT_ROOT / "ppi"
    if artifact_manifest_path.resolve(strict=False) != (output / "artifact_manifest.json").resolve(strict=False):
        abort("compatible manifest destination is not the exact PPI output manifest")
    receipt_copy = output / "compatible_resume_receipt.json"
    expected = _compatible_artifact_manifest_projection(
        original_manifest,
        current_identity=current_identity,
        prove_budget_path=output / "native_prove_budget.json",
        receipt_copy_path=receipt_copy,
        receipt_sha256=receipt_sha256,
    )
    payload = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
    if artifact_manifest_path.is_symlink() or not artifact_manifest_path.is_file():
        abort("compatible manifest installation requires the existing original PPI manifest")
    current_raw = artifact_manifest_path.read_bytes()
    if _digest_bytes(current_raw) == _digest_bytes(payload):
        return expected, _digest_bytes(current_raw)
    if _digest_bytes(current_raw) != old_manifest_sha256:
        abort("compatible manifest installation refuses to replace a non-original or already-changed manifest")
    temporary = artifact_manifest_path.with_name(f".{artifact_manifest_path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, artifact_manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(artifact_manifest_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return expected, _digest_bytes(payload)


def write_exclusive_json(path: Path, value: dict[str, Any], label: str) -> str:
    """Atomically create a JSON receipt and refuse all overwrites."""
    if path.is_symlink() or path.exists():
        abort(f"compatible resume {label} refuses to overwrite an existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        abort(f"compatible resume {label} parent directory is symlinked: {path.parent}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        abort(f"compatible resume {label} refuses to overwrite an existing path: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _digest_bytes(payload)


def write_immutable_json(path: Path, value: dict[str, Any], label: str) -> str:
    """Create immutable JSON once, or authenticate an identical prior creation."""
    if path.is_symlink():
        abort(f"compatible resume {label} path is symlinked: {path}")
    if path.exists():
        _regular_file(path, label)
        existing_raw = path.read_bytes()
        existing = load_json(path, label)
        if existing != value:
            abort(f"compatible resume {label} conflicts with its immutable prior value")
        return _digest_bytes(existing_raw)
    return write_exclusive_json(path, value, label)


def snapshot_immutable_file(source: Path, destination: Path, label: str) -> str:
    """Preserve exact source bytes at a write-once path, or verify an identical snapshot."""
    _regular_file(source, f"{label} source")
    payload = source.read_bytes()
    if destination.is_symlink():
        abort(f"compatible resume {label} snapshot path is symlinked")
    if destination.exists():
        _regular_file(destination, f"{label} snapshot")
        if destination.read_bytes() != payload:
            abort(f"compatible resume {label} snapshot conflicts with its immutable prior value")
        return _digest_bytes(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cursor = destination.parent
    while cursor != ROOT and cursor != cursor.parent:
        if cursor.is_symlink():
            abort(f"compatible resume {label} snapshot parent is symlinked: {cursor}")
        cursor = cursor.parent
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o400)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError:
        abort(f"compatible resume {label} snapshot appeared during exclusive creation")
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _digest_bytes(payload)


def snapshot_legacy_json_files(output: Path, compatibility_dir: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Preserve small old JSON receipts byte-for-byte, without hashing key/SRS files."""
    snapshots: dict[str, dict[str, Any]] = {}
    legacy_dir = compatibility_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    if legacy_dir.is_symlink():
        abort("compatible resume legacy snapshot directory is symlinked")
    for name in names:
        source = output / name
        _regular_file(source, f"legacy {name}")
        data = source.read_bytes()
        digest = _digest_bytes(data)
        destination = legacy_dir / name
        if destination.exists() or destination.is_symlink():
            _regular_file(destination, f"legacy snapshot {name}")
            if destination.read_bytes() != data:
                abort(f"compatible resume legacy snapshot conflicts with the existing {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o400)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        snapshots[name] = {
            "legacy_path": source.relative_to(ROOT).as_posix(),
            "snapshot_path": destination.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "size_bytes": len(data),
        }
    return snapshots


def validate_compatibility_receipt(path: Path, *, session_id: str | None = None) -> tuple[dict[str, Any], str]:
    _regular_file(path, "compatibility receipt")
    value = load_json(path, "Setup compatibility receipt")
    digest = _digest_bytes(path.read_bytes())
    expected_session = session_id or BASELINE_SESSION_ID
    output = f"{_COMPAT_OUTPUT_ROOT}/ppi"
    session_base = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{expected_session}"
    expected_keys = {
        "schema", "session_id", "dataset", "status", "baseline_manifest_sha256", "old_session_snapshot",
        "source_review", "old_setup_identity_sha256", "current_prover_identity_sha256",
        "admission_semantics_sha256", "bound_setup_admission", "current_capacity_admission",
        "capacity_precheck", "setup_chain", "current_prover_receipts", "shared_artifacts",
    }
    if (
        not isinstance(value, dict) or set(value) != expected_keys
        or value.get("schema") != COMPATIBILITY_SCHEMA
        or value.get("session_id") != expected_session
        or value.get("dataset") != "ppi"
        or value.get("status") != "SEALED_FOR_PROVE_ONLY"
        or value.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
    ):
        abort("Setup compatibility receipt is incomplete or has an unsupported identity boundary")
    _require_sha(digest, "Setup compatibility receipt digest")
    _require_sha(value["old_setup_identity_sha256"], "old Setup identity digest")
    _require_sha(value["current_prover_identity_sha256"], "current Prover identity digest")
    _require_sha(value["admission_semantics_sha256"], "Admission semantic digest")
    snapshot = _require_path_ref(value["old_session_snapshot"], _snapshot_path(expected_session), "old session snapshot")
    source_review = value["source_review"]
    if not isinstance(source_review, dict) or set(source_review) != {"path", "sha256", "source_delta_sha256", "snapshot_index_inventory_sha256"}:
        abort("compatible resume source-review reference has an unsupported shape")
    source_path = _safe_relative_path(source_review.get("path"), "source review")
    if not source_path.startswith("runs/check_logs/"):
        abort("compatible resume source-review reference must be under runs/check_logs")
    for name in ("sha256", "source_delta_sha256", "snapshot_index_inventory_sha256"):
        _require_sha(source_review.get(name), f"source review {name}")
    _require_admission_ref(
        value["bound_setup_admission"],
        f"{session_base}/admission/ppi/admission.json",
        "bound Setup Admission",
    )
    _require_admission_ref(
        value["current_capacity_admission"],
        f"{session_base}/compatibility/admission/ppi/admission.json",
        "current capacity Admission",
    )
    capacity = value["capacity_precheck"]
    if not isinstance(capacity, dict) or set(capacity) != {"path", "sha256", "capacity_source_identity_sha256", "recheck_path", "recheck_sha256"}:
        abort("compatible resume capacity precheck binding is malformed")
    _require_path_ref(
        {"path": capacity.get("path"), "sha256": capacity.get("sha256")},
        f"{session_base}/compatibility/capacity_precheck_ppi.json",
        "current capacity precheck",
    )
    _require_sha(capacity.get("capacity_source_identity_sha256"), "capacity source identity digest")
    _require_path_ref(
        {"path": capacity.get("recheck_path"), "sha256": capacity.get("recheck_sha256")},
        f"{session_base}/compatibility/capacity_rechecks.json",
        "current capacity recheck",
    )
    setup_chain = value["setup_chain"]
    setup_keys = {"origin", "setup_identity_sha256", "setup_receipt", "native_keygen_budget", "native_keygen_receipt", "setup_files"}
    if not isinstance(setup_chain, dict) or set(setup_chain) != setup_keys or setup_chain.get("origin") != "retained_old":
        abort("compatible resume retained Setup chain is malformed")
    _require_sha(setup_chain.get("setup_identity_sha256"), "retained Setup identity digest")
    _require_path_ref(setup_chain["setup_receipt"], f"{output}/setup_receipt.json", "retained Setup receipt")
    _require_path_ref(setup_chain["native_keygen_budget"], f"{output}/native_keygen_budget.json", "retained keygen budget")
    _require_path_ref(setup_chain["native_keygen_receipt"], f"{output}/native_keygen_receipt.json", "retained keygen receipt")
    setup_files = setup_chain["setup_files"]
    exact_setup_file_keys = {
        "settings_sha256", "compiled_circuit_sha256", "vk_sha256", "pk_sha256",
        "static_tables_sha256", "integer_ledger_sha256",
    }
    if not isinstance(setup_files, dict) or set(setup_files) != exact_setup_file_keys:
        abort("compatible resume retained Setup artifact digests are malformed")
    for name, file_digest in setup_files.items():
        _require_sha(file_digest, f"retained Setup file {name}")
    prover = value["current_prover_receipts"]
    if not isinstance(prover, dict) or set(prover) != {
        "native_prove_budget", "binary_sha256", "build_receipt_sha256", "build_environment_sha256",
        "retained_witness",
    }:
        abort("compatible resume current Prover receipt binding is malformed")
    _require_path_ref(prover["native_prove_budget"], f"{output}/native_prove_budget.json", "current native prove budget")
    for name in ("binary_sha256", "build_receipt_sha256", "build_environment_sha256"):
        _require_sha(prover.get(name), f"current Prover {name}")
    retained_witness = prover["retained_witness"]
    witness_keys = {
        "origin", "session_snapshot", "stage_receipt_sha256", "witness", "old_native_prove_budget",
        "old_admission_sha256", "old_setup_receipt_sha256",
    }
    if not isinstance(retained_witness, dict) or set(retained_witness) != witness_keys or retained_witness.get("origin") != "retained_old":
        abort("compatible resume retained Witness binding is malformed")
    _require_path_ref(retained_witness["session_snapshot"], _snapshot_path(expected_session), "retained Witness old session")
    _require_sha(retained_witness.get("stage_receipt_sha256"), "retained Witness stage receipt")
    witness_ref = retained_witness["witness"]
    if (
        not isinstance(witness_ref, dict)
        or set(witness_ref) != {"path", "sha256", "size_bytes"}
        or witness_ref.get("path") != f"{output}/witness.json"
        or isinstance(witness_ref.get("size_bytes"), bool)
        or not isinstance(witness_ref.get("size_bytes"), int)
        or witness_ref["size_bytes"] <= 0
    ):
        abort("compatible resume retained Witness artifact binding is malformed")
    _require_sha(witness_ref.get("sha256"), "retained Witness artifact")
    _require_path_ref(
        retained_witness["old_native_prove_budget"],
        f"{session_base}/compatibility/legacy/native_prove_budget.json",
        "retained Witness old prove budget",
    )
    _require_sha(retained_witness.get("old_admission_sha256"), "retained Witness old Admission")
    _require_sha(retained_witness.get("old_setup_receipt_sha256"), "retained Witness old Setup receipt")
    if (
        retained_witness.get("old_admission_sha256") != value["bound_setup_admission"].get("sha256")
        or retained_witness.get("old_setup_receipt_sha256") != setup_chain["setup_receipt"].get("sha256")
    ):
        abort("compatible resume retained Witness does not bind the original PPI Setup Admission")
    shared = value["shared_artifacts"]
    shared_keys = {
        "statement_sha256", "public_output_sha256", "settings_sha256", "compiled_circuit_sha256", "vk_sha256",
        "pk_sha256", "static_tables_sha256", "integer_ledger_sha256", "original_artifact_manifest",
    }
    if not isinstance(shared, dict) or set(shared) != shared_keys:
        abort("compatible resume shared artifact binding is malformed")
    for name in shared_keys - {"original_artifact_manifest"}:
        _require_sha(shared.get(name), f"shared artifact {name}")
    _require_path_ref(
        shared["original_artifact_manifest"],
        f"{_COMPAT_OUTPUT_ROOT}/.sessions/{expected_session}/stage-artifacts/ppi/artifact_manifest.final.json",
        "original artifact manifest",
    )
    if (
        setup_chain["setup_files"].get("settings_sha256") != value["bound_setup_admission"].get("settings_sha256")
        or shared.get("settings_sha256") != value["bound_setup_admission"].get("settings_sha256")
        or value["current_capacity_admission"].get("settings_sha256") != value["bound_setup_admission"].get("settings_sha256")
        or setup_chain["setup_files"].get("compiled_circuit_sha256") != shared.get("compiled_circuit_sha256")
        or setup_chain["setup_files"].get("vk_sha256") != shared.get("vk_sha256")
        or setup_chain["setup_files"].get("pk_sha256") != shared.get("pk_sha256")
        or setup_chain["setup_files"].get("static_tables_sha256") != shared.get("static_tables_sha256")
        or setup_chain["setup_files"].get("integer_ledger_sha256") != shared.get("integer_ledger_sha256")
    ):
        abort("compatible resume receipt does not bind its old Setup and current artifact identities consistently")
    if path.resolve(strict=True) != (ROOT / f"{session_base}/compatibility/compatible_resume_receipt.json").resolve(strict=True):
        abort("compatible resume receipt is not at the exact session sidecar path")
    return value, digest


def canonical_digest(value: Any) -> str:
    return canonical_json_digest(value)
