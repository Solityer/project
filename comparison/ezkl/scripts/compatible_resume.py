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
import stat
import ctypes
import errno
from pathlib import Path
from typing import Any

from formal_common import ROOT, ContractError, abort, canonical_json_digest, load_json, require_digest


BASELINE_SESSION_ID = "d3dc1ccb1cfa4a78829c306b835f300e"
BASELINE_MANIFEST_SHA256 = "02d43ff2859ebb6ef4e5f58ebe09eb3582b5a2ba2a06603ac59ccf12a4b7d547"
BASELINE_RELATIVE = Path("runs/check_logs/ezkl_resume_compat_d3dc1ccb1cfa4a78829c306b835f300e/baseline.json")
COMPATIBILITY_SCHEMA = "ezkl.setup-compatible-prove-only.v1"
SOURCE_REVIEW_SCHEMA = "ezkl.compatible-resume-source-review.v1"
REQUEST_SCHEMA = "ezkl.compatible-resume-request.v1"
PREFLIGHT_RETRY_SCHEMA = "ezkl.compatible-resume-preflight-retry.v1"
ADMISSION_PREFLIGHT_RETRY_SCHEMA = "ezkl.compatible-admission-preflight-retry.v1"
INTERRUPTED_BEFORE_PROVE_RECOVERY_SCHEMA = "ezkl.compatible-interrupted-before-prove-recovery.v1"
CONFIGURED_ADMISSION_RETRY_SCHEMA = "ezkl.compatible-configured-admission-retry.v1"
CAPACITY_ADMISSION_RETRY_SCHEMA = "ezkl.compatible-capacity-admission-retry.v1"
CAPACITY_ADMISSION_RETRY_V2_SCHEMA = "ezkl.compatible-capacity-admission-retry.v2"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_COMPAT_OUTPUT_ROOT = "runs/formal_ppi_ogbn_20261029_001"
_COMPAT_MATRIX = f"{_COMPAT_OUTPUT_ROOT}/formal_matrix.json"
_COMPAT_DATASETS = ["ppi", "ogbn_arxiv"]
_PRE_RETRY_REVIEW_PATH = "runs/check_logs/ezkl_compatible_resume_source_review_20261030_v3.json"
_PRE_RETRY_REVIEW_SHA256 = "f3478e8b4391787f3969f7cd37cc176fbf02f054f04de21f539d2033b613562e"
_PRE_RETRY_REQUEST_SHA256 = "e164a8c2d63d205acc28c44c76d268404287894200594fd007e3fdd210cdcc17"
_V5_POSTSTOP_SESSION_SHA256 = "38b16e98a93b506827e094b555240fdfa53d6e74d743db56d88026cbf0d4a4ab"
_V5_REQUEST_CANONICAL_SHA256 = "0d51a1b6c0d03deb8d01dee5d8b3986d0b69629738afc9ea460ed0ce5098ec69"
_V5_V1_RETRY_CANONICAL_SHA256 = "e3c4bfcf1bd47f7f43479c99c4f0c8fa4d98220f94a4b08f83588933b94663e5"
_V5_V2_RETRY_CANONICAL_SHA256 = "d26ce5199e444efa984e2691d286c289dcb9e8312f9e952d29dde08fd04c4608"
_V5_SOURCE_REVIEW_PATH = "runs/check_logs/ezkl_compatible_resume_source_review_20261101_v5.json"
_V5_SOURCE_REVIEW_SHA256 = "c9ae8e49af267e4a43c1206fd270c360a88fa9cf55aef04bd58a04db8628470f"
_ORIGINAL_SESSION_SHA256 = "5b277cc02da471e94a76b47627a7d5b35c36defb73510777f20429ec16c37093"
_V5_PROVER_IDENTITY_SHA256 = "c1fe92e41047cab7397bab9caf9220d1d3393d439b5a7982715614dca57c0c6e"
_V5_CAPACITY_IDENTITY_SHA256 = "403dfe76f7e185847555b342151f24e22b268676a82c0c1cdd7adcdffefff42d"
_V5_CAPACITY_REPORT_SHA256 = "f240a31b8d631e6e1844d7d2b28947637d8237e633d3422438a75c5f505b0420"
_V6_CONFIGURED_ADMISSION_SESSION_SHA256 = "b0596d30385dc39ea035435862084bc3f7c555e13055375592baa0d6698786ff"
_V6_CONFIGURED_ADMISSION_REQUEST_SHA256 = "31ae327a61b70d017f46a3a25221d8b9b175fdc50fd961644ff6a30662137d7d"
_V6_CONFIGURED_ADMISSION_REVIEW_PATH = "runs/check_logs/ezkl_compatible_resume_source_review_20261102_v6.json"
_V6_CONFIGURED_ADMISSION_REVIEW_SHA256 = "3955177c933f9129200d27309f94eeb82a5e12822d2eab8c9f980d5e3493ef28"
_V7_CAPACITY_ADMISSION_SESSION_SHA256 = "b9c888ef0200a05e3a6dab2832e2f88ad0776805aa189f4e5dcba92fd6be55c3"
_V7_CAPACITY_ADMISSION_REQUEST_SHA256 = "2d5bd80c2005ece883971e8fe25dcd662138c66017451248d656315f4ea36c5f"
_V7_CONFIGURED_ADMISSION_RETRY_CANONICAL_SHA256 = "53d8115eac80d61f34ba3d419e63e4ae4e00c00e84c8f272fa3c3338bb063d40"
_V7_CAPACITY_ADMISSION_REVIEW_PATH = "runs/check_logs/ezkl_compatible_resume_source_review_20261102_v7b.json"
_V7_CAPACITY_ADMISSION_REVIEW_SHA256 = "f5ebdd3bf29c0e93b4a61f45931458535a70388d5679ed1b68913062b00a99ee"
_V7_CAPACITY_ADMISSION_PROVER_IDENTITY_SHA256 = "b88c1e56ab8c1bec3eb7336082db340bbb89ea7dcfbc0d8055026173e0585d13"
_V7_CAPACITY_ADMISSION_CAPACITY_IDENTITY_SHA256 = "02d1da1db9ff7cc6aa287a5c17a7799844e731ba01f388620a5c6de9ae4db989"
_V7_CAPACITY_ADMISSION_REPORT_SHA256 = "f67e5470864df2624c355870a1418f22bef15126b4a76f84035cb6ac9e05c57e"
_V7_CAPACITY_ADMISSION_MAP_CANONICAL_SHA256 = "7026cd8ad057a46d2591300e969acf6df992fbfe7c9ef905e382c909ff88c404"
_V7_NATIVE_PROVE_BUDGET_SHA256 = "0bdc46d4706c2e01f161ccabef265da60b845ca277dfa0cc3125272b327482f8"
_V8_POSTFAILURE_SESSION_SHA256 = "2c7cb544d2b12221bf12183b8929a62209fc6378835d1c79633900a68f3325d1"
_V8_POSTFAILURE_REQUEST_SHA256 = "6102a10bfef00cb7fb58e30cae9f5f9b4866a43069084bf3f344d7a23c120634"
_V8_POSTFAILURE_HISTORY_CANONICAL_SHA256 = "12cb8293eb2463c31dc2700c5bcd92805590db6299d40df9c6767ea3d80b82ee"
_V8_SOURCE_REVIEW_SHA256 = "830d273923e2eaaea1a3fcc18291f63f672742cbb787f825754242dd17338778"
_V8_RETRY_RECORD_SHA256 = "e101e47967c3ff21aefff1c44d1af423b8f4c15fa8629ac0bd7eff42d1296d15"
_V8_CAPACITY_CHECKS_CANONICAL_SHA256 = "3cc3f65987389c0e6abf888c822936d717cb8253bd3fbdb29675ec2c8a727688"
_V8_PPI_PROVE_BUDGET_SHA256 = "3c826097071b7860398a5bde55e1b4c969264d0e251fd86d83f60b73d36af13e"
_V8_BAD_COMPATIBILITY_RECEIPT_SHA256 = "0319ae2c7d40b94ac3cba3addabab22a97641045fdebc6caf4de8f06a62257f2"
_V8_PPI_CAPACITY_REPORT_SHA256 = "1b682567e0c4f0c7e0475ee7a314307ae098c22220a08ff35235ec1d1e07c3a5"
_V8_PPI_CAPACITY_RECHECKS_SHA256 = "f919657601800e6a5f98d7e512eb1199445f461c895398e9ad44d089c9174c5c"
_V8_LEGACY_NATIVE_PROVE_BUDGET_SHA256 = "e83974d372c217204145bc722d81d1797b0bedd889e80ee0d70127d78b0add2a"
_ORIGINAL_PPI_SETUP_RECEIPT_SHA256 = "16dd568b18db7a7601866cf01fd23921af2edcc8abbb186a507fa23d3e963b70"
_ORIGINAL_PPI_WITNESS_STAGE_SHA256 = "e7bc1ce024bcbf35b907f5eb114e26fff103f52a438e059b3834a21efcc8e9e9"
_REQUIRED_SNAPSHOT_INDEXES = {
    "supplemental_snapshot_index.json",
    "supplemental_snapshot_index_002.json",
    "supplemental_snapshot_index_003.json",
    "supplemental_snapshot_index_004.json",
    "supplemental_snapshot_index_005.json",
    "supplemental_snapshot_index_006.json",
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


def _artifact_manifest_file_records(files: object, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(files, list) or not files:
        abort(f"{label} has no file census")
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            abort(f"{label} file record {index} is malformed")
        name, size, digest = record["path"], record["size"], record["sha256"]
        if (
            not isinstance(name, str) or not name or Path(name).is_absolute()
            or ".." in Path(name).parts or name in records
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0
        ):
            abort(f"{label} file record {index} has an unsafe path or size")
        _require_sha(digest, f"{label} {name} digest")
        records[name] = record
    return records


def artifact_manifest_file_sha256(manifest: object, path: str) -> str:
    """Return one unique file digest from an already-authenticated artifact manifest."""
    if not isinstance(manifest, dict):
        abort("compatible resume artifact manifest is not an object")
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        abort("compatible resume artifact-manifest file path is unsafe")
    records = _artifact_manifest_file_records(manifest.get("files"), "compatible resume artifact manifest")
    record = records.get(path)
    if record is None:
        abort(f"compatible resume artifact manifest is missing {path}")
    return _require_sha(record.get("sha256"), f"artifact manifest {path} digest")


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
        abort(f"compatible resume frozen snapshot index inventory differs from the exact six-index set: {detail}")
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
    actual_changed.sort(key=lambda item: (item["path"], item["change"]))
    actual_review_only = [
        {key: item[key] for key in ("path", "change", "baseline_sha256", "current_sha256", "diff_sha256")}
        for item in delta["review_only_changed_files"]
    ]
    actual_review_only.sort(key=lambda item: (item["path"], item["change"]))
    approved_indexes = document["reviewed_snapshot_indexes"]
    actual_indexes = sorted(delta["snapshot_index_files"], key=lambda item: item["path"])
    if (
        not isinstance(approved_indexes, list)
        or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in approved_indexes)
        or approved_indexes != sorted(approved_indexes, key=lambda item: item["path"] if isinstance(item.get("path"), str) else "")
        or approved_indexes != actual_indexes
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
    if (
        not isinstance(indexes, list)
        or any(
            not isinstance(item, dict) or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str) or not _HEX64.fullmatch(item["sha256"])
            for item in indexes
        )
    ):
        abort("compatible source delta lacks the exact snapshot-index inventory")
    return canonical_json_digest(sorted(indexes, key=lambda item: item["path"]))


def validate_historical_preflight_request(request: object) -> dict[str, Any]:
    """Authenticate only the exact v3 request that preceded the approved retry.

    This intentionally does not run the old review against the current source
    tree. Its raw digest is pinned by the old request and by the retry audit.
    """
    required = {
        "schema", "session_id", "output_root", "matrix", "selected_datasets", "retained_setup_datasets",
        "source_review_path", "source_review_sha256", "baseline_manifest_sha256", "baseline_source_tree_sha256",
        "source_delta_sha256", "snapshot_index_inventory_sha256", "old_session_snapshot_path", "old_session_snapshot_sha256",
    }
    if not isinstance(request, dict) or set(request) != required:
        abort("preflight retry old request has unknown, missing, or malformed fields")
    if canonical_json_digest(request) != _PRE_RETRY_REQUEST_SHA256:
        abort("preflight retry old request is not the exact failed v3 request")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("session_id") != BASELINE_SESSION_ID
        or request.get("output_root") != _COMPAT_OUTPUT_ROOT
        or request.get("matrix") != _COMPAT_MATRIX
        or request.get("selected_datasets") != _COMPAT_DATASETS
        or request.get("retained_setup_datasets") != ["ppi"]
        or request.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or request.get("source_review_path") != _PRE_RETRY_REVIEW_PATH
        or request.get("source_review_sha256") != _PRE_RETRY_REVIEW_SHA256
        or request.get("old_session_snapshot_path") != _snapshot_path(BASELINE_SESSION_ID)
    ):
        abort("preflight retry old request is not bound to the one original session and v3 approval")
    review_path = ROOT / _PRE_RETRY_REVIEW_PATH
    _regular_file(review_path, "historical v3 source review")
    review_bytes = review_path.read_bytes()
    if _digest_bytes(review_bytes) != _PRE_RETRY_REVIEW_SHA256:
        abort("historical v3 source review bytes differ from the old request anchor")
    try:
        review = json.loads(review_bytes)
    except json.JSONDecodeError as exc:
        abort(f"historical v3 source review is invalid JSON: {exc}")
    if (
        not isinstance(review, dict)
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("session_id") != BASELINE_SESSION_ID
        or review.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or review.get("decision") != "APPROVED"
        or review.get("reviewer") != "root"
    ):
        abort("historical v3 source review is not the exact approved session review")
    return request


def _preflight_retry_directory(session_id: str) -> Path:
    if session_id != BASELINE_SESSION_ID:
        abort("preflight retry audit is bound to the one original session")
    return ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "preflight_retry"


def _admission_preflight_retry_directory(session_id: str) -> Path:
    if session_id != BASELINE_SESSION_ID:
        abort("Admission preflight retry audit is bound to the one original session")
    return ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "admission_preflight_retry"


def _json_snapshot_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _load_retry_snapshot(ref: object, expected_path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    expected_relative = expected_path.relative_to(ROOT).as_posix()
    bound = _require_path_ref(ref, expected_relative, label)
    raw = _snapshot_file(ROOT / bound["path"], relative=bound["path"], expected_digest=bound["sha256"])
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"compatible Admission preflight {label} snapshot is invalid JSON: {exc}")
    if not isinstance(value, dict):
        abort(f"compatible Admission preflight {label} snapshot is not an object")
    return value, raw


def _historical_approved_review(request: dict[str, Any], label: str) -> dict[str, Any]:
    relative = _safe_relative_path(request.get("source_review_path"), f"{label} source review")
    path = ROOT / relative
    try:
        path.resolve(strict=True).relative_to((ROOT / "runs" / "check_logs").resolve(strict=True))
    except (OSError, ValueError):
        abort(f"compatible Admission preflight {label} source review is outside runs/check_logs")
    _regular_file(path, f"{label} historical source review")
    raw = path.read_bytes()
    if _digest_bytes(raw) != request.get("source_review_sha256"):
        abort(f"compatible Admission preflight {label} source review bytes differ from the bound request")
    try:
        review = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"compatible Admission preflight {label} source review is invalid JSON: {exc}")
    if (
        not isinstance(review, dict)
        or set(review) != {
            "schema", "session_id", "baseline_manifest_sha256", "decision", "reviewer",
            "reviewed_changed_files", "reviewed_review_only_files", "reviewed_snapshot_indexes",
        }
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("session_id") != BASELINE_SESSION_ID
        or review.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or review.get("decision") != "APPROVED"
        or review.get("reviewer") != "root"
    ):
        abort(f"compatible Admission preflight {label} source review is not the exact approved historical schema")
    return review


def _validate_admission_preflight_state(
    document: object,
    original: dict[str, Any],
    old_request: dict[str, Any],
    *,
    check_outputs: bool,
    current_files: dict[str, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    """Authenticate only the failed selected-capacity preflight state and its two current Admissions."""
    if not isinstance(document, dict):
        abort("compatible Admission preflight failed-session snapshot is malformed")
    expected_compatible = {
        "compatible_resume_request", "compatible_preflight_retry", "compatible_prover_identity",
        "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "compatible_native_admissions",
    }
    if "compatible_admission_preflight_retry" in document:
        expected_compatible.add("compatible_admission_preflight_retry")
    actual_compatible = {key for key in document if key.startswith("compatible_")}
    if actual_compatible != expected_compatible:
        abort("compatible Admission preflight has unknown, missing, or extra compatible session fields")
    if (
        document.get("schema_version") != 1
        or document.get("baseline") != "ezkl"
        or document.get("session_id") != BASELINE_SESSION_ID
        or document.get("status") != "FAILED"
        or document.get("stage") != "selected_dataset_native_capacity_preflight"
        or document.get("output_root") != _COMPAT_OUTPUT_ROOT
        or document.get("matrix") != _COMPAT_MATRIX
        or document.get("selected_datasets") != _COMPAT_DATASETS
        or document.get("completed_cells") not in (None, {})
        or document.get("compatible_resume_request") != old_request
        or document.get("native_admissions") != original.get("native_admissions")
        or document.get("setup_reuse_receipts") != original.get("setup_reuse_receipts")
        or document.get("stage_receipts") != original.get("stage_receipts")
    ):
        abort("compatible Admission preflight is not the exact selected-session failure before proof progress")
    checks = document.get("capacity_checks")
    original_checks = original.get("capacity_checks")
    if (
        not isinstance(checks, list)
        or not isinstance(original_checks, list)
        or len(original_checks) != 4
        or canonical_json_digest(checks) != canonical_json_digest(original_checks)
    ):
        abort("compatible Admission preflight capacity history differs from the original four-check prefix")

    current_identity = document.get("compatible_prover_identity")
    current_identity_sha = document.get("compatible_prover_identity_sha256")
    old_resume_identity = document.get("resume_identity")
    capacity_identity = document.get("compatible_capacity_source_identity")
    capacity_identity_sha = document.get("compatible_capacity_source_identity_sha256")
    if (
        not isinstance(current_identity, dict)
        or current_identity.get("schema") != "ezkl-formal-resume-identity-v1"
        or current_identity_sha != canonical_json_digest(current_identity)
        or not isinstance(old_resume_identity, dict)
        or canonical_json_digest(old_resume_identity) != canonical_json_digest(original.get("resume_identity"))
        or not isinstance(capacity_identity, dict)
        or capacity_identity_sha != canonical_json_digest(capacity_identity)
        or capacity_identity.get("source_tree_sha256") != current_identity.get("source_tree_sha256")
    ):
        abort("compatible Admission preflight current prover/capacity identity pair is not exactly session-bound")
    retry_record = document.get("compatible_preflight_retry")
    if not isinstance(retry_record, dict):
        abort("compatible Admission preflight lacks its first approved source-review migration record")

    original_admissions = original.get("native_admissions")
    current_admissions = document.get("compatible_native_admissions")
    expected_keys = {
        "path", "sha256", "prepared_dir", "settings_sha256", "scalar_domain_vector_bytes",
        "total_assignments", "logrows", "domain_size", "constraint_system", "native_memory_model",
        "relation_identity",
    }
    if (
        not isinstance(original_admissions, dict)
        or set(original_admissions) != set(_COMPAT_DATASETS)
        or not isinstance(current_admissions, dict)
        or set(current_admissions) != set(_COMPAT_DATASETS)
    ):
        abort("compatible Admission preflight maps must contain exactly the original PPI and OGBN-Arxiv selections")

    authenticated: dict[str, dict[str, Any]] = {}
    output_session_dir = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID
    for dataset in _COMPAT_DATASETS:
        current_ref = current_admissions[dataset]
        old_ref = original_admissions[dataset]
        current_rel = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/admission/{dataset}/admission.json"
        old_rel = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/admission/{dataset}/admission.json"
        if (
            not isinstance(current_ref, dict) or set(current_ref) != expected_keys
            or current_ref.get("path") != current_rel
            or current_ref.get("prepared_dir") != str(Path(current_rel).parent)
            or not isinstance(old_ref, dict) or set(old_ref) != expected_keys
            or old_ref.get("path") != old_rel
            or old_ref.get("prepared_dir") != str(Path(old_rel).parent)
        ):
            abort(f"compatible Admission preflight {dataset} map path/schema is not exact")
        _require_sha(current_ref.get("sha256"), f"{dataset} current Admission digest")
        _require_sha(current_ref.get("settings_sha256"), f"{dataset} current settings digest")
        _require_sha(old_ref.get("sha256"), f"{dataset} original Admission digest")
        _require_sha(old_ref.get("settings_sha256"), f"{dataset} original settings digest")
        source_admission = ROOT / current_rel
        source_settings = source_admission.parent / "settings.json"
        if current_files is not None:
            if dataset not in current_files:
                abort(f"compatible Admission preflight {dataset} immutable snapshot is missing")
            admission_path, settings_path = current_files[dataset]
        else:
            admission_path, settings_path = source_admission, source_settings
        for file_path, expected_digest, label in (
            (admission_path, current_ref["sha256"], f"{dataset} current Admission"),
            (settings_path, current_ref["settings_sha256"], f"{dataset} current settings"),
        ):
            _regular_file(file_path, label)
            if _digest_bytes(file_path.read_bytes()) != expected_digest:
                abort(f"compatible Admission preflight {label} bytes differ from the session map")
        current_admission = load_json(admission_path, f"compatible preflight current {dataset} Admission")
        old_admission_path = ROOT / old_rel
        old_settings_path = old_admission_path.parent / "settings.json"
        _regular_file(old_admission_path, f"original {dataset} Admission")
        _regular_file(old_settings_path, f"original {dataset} settings")
        if (
            _digest_bytes(old_admission_path.read_bytes()) != old_ref.get("sha256")
            or _digest_bytes(old_settings_path.read_bytes()) != old_ref.get("settings_sha256")
            or not isinstance(current_admission, dict)
            or current_admission.get("dataset") != dataset
            or current_admission.get("settings_sha256") != current_ref["settings_sha256"]
            or any(current_ref.get(key) != current_admission.get(key) for key in (
                "scalar_domain_vector_bytes", "total_assignments", "logrows", "domain_size",
                "constraint_system", "native_memory_model", "relation_identity",
            ))
        ):
            abort(f"compatible Admission preflight {dataset} map is not bound to its Admission bytes")
        old_admission = load_json(old_admission_path, f"original {dataset} Admission")
        if (
            not isinstance(old_admission, dict)
            or old_admission.get("dataset") != dataset
            or old_admission.get("settings_sha256") != old_ref.get("settings_sha256")
            or any(old_ref.get(key) != old_admission.get(key) for key in (
                "scalar_domain_vector_bytes", "total_assignments", "logrows", "domain_size",
                "constraint_system", "native_memory_model", "relation_identity",
            ))
        ):
            abort(f"original {dataset} Admission anchor is malformed")
        compare_admission_semantics(
            old_admission,
            current_admission,
            dataset=dataset,
            old_settings_path=old_settings_path,
            current_settings_path=settings_path,
        )
        authenticated[dataset] = current_admission

    if check_outputs:
        output_root = ROOT / _COMPAT_OUTPUT_ROOT
        for dataset in _COMPAT_DATASETS:
            output = output_root / dataset
            if output.is_symlink():
                abort(f"compatible Admission preflight {dataset} output is a symlink")
            for name in ("proof.json", "result.json", "native_verify_metrics.json", "core_run_manifest.json"):
                if (output / name).exists() or (output / name).is_symlink():
                    abort(f"compatible Admission preflight refuses existing {dataset} proof/result artifact: {name}")
        compatibility = output_session_dir / "compatibility"
        for name in ("capacity_precheck_ppi.json", "capacity_rechecks.json", "compatible_resume_receipt.json", "compatible_stage_receipts.json"):
            path = compatibility / name
            if path.exists() or path.is_symlink():
                abort(f"compatible Admission preflight refuses existing downstream stage artifact: {name}")
    return authenticated


def _validate_admission_retry_snapshots(
    record: dict[str, Any], archived: dict[str, Any], original: dict[str, Any], old_request: dict[str, Any],
) -> None:
    retry_dir = _admission_preflight_retry_directory(BASELINE_SESSION_ID)
    map_path = retry_dir / "compatible_native_admissions.json"
    map_ref = _require_path_ref(record.get("admissions_map_snapshot"), map_path.relative_to(ROOT).as_posix(), "Admission map snapshot")
    raw_map = _snapshot_file(ROOT / map_ref["path"], relative=map_ref["path"], expected_digest=map_ref["sha256"])
    try:
        snap_map = json.loads(raw_map)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"compatible Admission preflight Admission map snapshot is invalid JSON: {exc}")
    archived_map = archived.get("compatible_native_admissions")
    if not isinstance(snap_map, dict) or snap_map != archived_map:
        abort("compatible Admission preflight Admission map snapshot differs from the archived failed session")
    entries = record.get("admission_file_snapshots")
    if not isinstance(entries, list) or len(entries) != len(_COMPAT_DATASETS):
        abort("compatible Admission preflight file-snapshot inventory is malformed")
    expected_datasets = list(_COMPAT_DATASETS)
    if [item.get("dataset") if isinstance(item, dict) else None for item in entries] != expected_datasets:
        abort("compatible Admission preflight file-snapshot inventory is not in exact dataset order")
    snapshot_files: dict[str, tuple[Path, Path]] = {}
    for dataset, item in zip(_COMPAT_DATASETS, entries):
        if not isinstance(item, dict) or set(item) != {
            "dataset", "source_admission", "source_settings", "snapshot_admission", "snapshot_settings",
        }:
            abort(f"compatible Admission preflight {dataset} snapshot entry has unknown or missing fields")
        current_ref = archived_map.get(dataset) if isinstance(archived_map, dict) else None
        if not isinstance(current_ref, dict):
            abort(f"compatible Admission preflight archived {dataset} map is missing")
        source_admission_rel = current_ref.get("path")
        source_settings_rel = f"{Path(str(source_admission_rel)).parent.as_posix()}/settings.json"
        source_admission = _require_path_ref(
            item.get("source_admission"), source_admission_rel, f"{dataset} source Admission",
        )
        source_settings = _require_path_ref(
            item.get("source_settings"), source_settings_rel, f"{dataset} source settings",
        )
        if source_admission["sha256"] != current_ref.get("sha256") or source_settings["sha256"] != current_ref.get("settings_sha256"):
            abort(f"compatible Admission preflight {dataset} source snapshots differ from the archived map")
        admission_snapshot_path = retry_dir / "admissions" / dataset / "admission.json"
        settings_snapshot_path = retry_dir / "admissions" / dataset / "settings.json"
        admission_snapshot = _require_path_ref(
            item.get("snapshot_admission"), admission_snapshot_path.relative_to(ROOT).as_posix(),
            f"{dataset} Admission immutable copy",
        )
        settings_snapshot = _require_path_ref(
            item.get("snapshot_settings"), settings_snapshot_path.relative_to(ROOT).as_posix(),
            f"{dataset} settings immutable copy",
        )
        _snapshot_file(ROOT / admission_snapshot["path"], relative=admission_snapshot["path"], expected_digest=admission_snapshot["sha256"])
        _snapshot_file(ROOT / settings_snapshot["path"], relative=settings_snapshot["path"], expected_digest=settings_snapshot["sha256"])
        if (
            admission_snapshot["sha256"] != source_admission["sha256"]
            or settings_snapshot["sha256"] != source_settings["sha256"]
        ):
            abort(f"compatible Admission preflight {dataset} immutable copy differs from the failed-session bytes")
        snapshot_files[dataset] = (ROOT / admission_snapshot["path"], ROOT / settings_snapshot["path"])
    _validate_admission_preflight_state(
        archived, original, old_request, check_outputs=False, current_files=snapshot_files,
    )

    report = record.get("capacity_report_snapshot")
    report_snapshot_path = retry_dir / "capacity_precheck_current.json"
    if report is None:
        pass
    else:
        if not isinstance(report, dict) or set(report) != {"source", "snapshot"}:
            abort("compatible Admission preflight capacity report snapshot record is malformed")
        report_path = (
            ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID
            / "compatibility" / "capacity_precheck_current.json"
        )
        source_ref = _require_path_ref(
            report.get("source"), report_path.relative_to(ROOT).as_posix(), "current capacity report",
        )
        snapshot_ref = _require_path_ref(report.get("snapshot"), report_snapshot_path.relative_to(ROOT).as_posix(), "capacity report immutable copy")
        _snapshot_file(ROOT / snapshot_ref["path"], relative=snapshot_ref["path"], expected_digest=snapshot_ref["sha256"])
        if snapshot_ref["sha256"] != source_ref["sha256"]:
            abort("compatible Admission preflight capacity report immutable copy digest differs from its source anchor")


def _validate_admission_preflight_retry_record(
    session_document: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = session_document.get("compatible_admission_preflight_retry")
    expected_keys = {
        "schema", "session_id", "failed_session_snapshot", "old_request_snapshot", "old_request_canonical_sha256",
        "old_preflight_retry_snapshot", "original_session_snapshot", "old_source_review", "new_source_review",
        "new_request_canonical_sha256", "prior_prover_identity_sha256", "prior_capacity_source_identity_sha256",
        "admissions_map_snapshot", "admission_file_snapshots", "capacity_report_snapshot",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        abort("compatible Admission preflight retry audit has unknown, missing, or malformed fields")
    if (
        record.get("schema") != ADMISSION_PREFLIGHT_RETRY_SCHEMA
        or record.get("session_id") != BASELINE_SESSION_ID
        or record.get("new_request_canonical_sha256") != canonical_json_digest(request)
    ):
        abort("compatible Admission preflight retry audit is not bound to the exact current request/session")
    _require_sha(record.get("prior_prover_identity_sha256"), "prior prover identity digest")
    _require_sha(record.get("prior_capacity_source_identity_sha256"), "prior capacity-source identity digest")

    retry_dir = _admission_preflight_retry_directory(BASELINE_SESSION_ID)
    archived_path = retry_dir / "failed_session.json"
    old_request_path = retry_dir / "old_request.json"
    old_retry_path = retry_dir / "old_preflight_retry.json"
    archived, _ = _load_retry_snapshot(record.get("failed_session_snapshot"), archived_path, "failed-session")
    old_request_value, _ = _load_retry_snapshot(record.get("old_request_snapshot"), old_request_path, "old-request")
    old_request = old_request_value
    _require_sha(record.get("old_request_canonical_sha256"), "old request canonical digest")
    if (
        canonical_json_digest(old_request) != record.get("old_request_canonical_sha256")
        or archived.get("compatible_resume_request") != old_request
        or canonical_json_digest(old_request) == canonical_json_digest(request)
        or archived.get("compatible_prover_identity_sha256") != record.get("prior_prover_identity_sha256")
        or archived.get("compatible_capacity_source_identity_sha256") != record.get("prior_capacity_source_identity_sha256")
    ):
        abort("compatible Admission preflight retry old request or current-identity pair does not match the archived failed session")
    # The v4 request itself is authenticated by its immutable archived session and the prior v3->v4 audit.
    if not isinstance(old_request, dict) or set(old_request) != {
        "schema", "session_id", "output_root", "matrix", "selected_datasets", "retained_setup_datasets",
        "source_review_path", "source_review_sha256", "baseline_manifest_sha256", "baseline_source_tree_sha256",
        "source_delta_sha256", "snapshot_index_inventory_sha256", "old_session_snapshot_path", "old_session_snapshot_sha256",
    }:
        abort("compatible Admission preflight old request has an unknown schema")
    if (
        old_request.get("schema") != REQUEST_SCHEMA
        or old_request.get("session_id") != BASELINE_SESSION_ID
        or old_request.get("output_root") != _COMPAT_OUTPUT_ROOT
        or old_request.get("matrix") != _COMPAT_MATRIX
        or old_request.get("selected_datasets") != _COMPAT_DATASETS
        or old_request.get("retained_setup_datasets") != ["ppi"]
        or old_request.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
    ):
        abort("compatible Admission preflight old request is not the exact selected-session request")
    old_retry = archived.get("compatible_preflight_retry")
    old_retry_ref_path = old_retry_path.relative_to(ROOT).as_posix()
    old_retry_ref = _require_path_ref(
        {"path": record.get("old_preflight_retry_snapshot", {}).get("path") if isinstance(record.get("old_preflight_retry_snapshot"), dict) else None,
         "sha256": record.get("old_preflight_retry_snapshot", {}).get("sha256") if isinstance(record.get("old_preflight_retry_snapshot"), dict) else None},
        old_retry_ref_path,
        "first retry audit snapshot",
    )
    old_retry_snapshot = _snapshot_file(ROOT / old_retry_ref["path"], relative=old_retry_ref["path"], expected_digest=old_retry_ref["sha256"])
    try:
        old_retry_value = json.loads(old_retry_snapshot)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"compatible Admission preflight first retry audit snapshot is invalid: {exc}")
    if not isinstance(old_retry, dict) or old_retry_value != old_retry or canonical_json_digest(old_retry) != canonical_json_digest(old_retry_value):
        abort("compatible Admission preflight first retry record differs from its immutable snapshot")
    original_ref = _require_path_ref(
        record.get("original_session_snapshot"), request.get("old_session_snapshot_path"), "original-session snapshot",
    )
    if (
        original_ref["path"] != old_request.get("old_session_snapshot_path")
        or original_ref["sha256"] != old_request.get("old_session_snapshot_sha256")
        or original_ref["sha256"] != request.get("old_session_snapshot_sha256")
    ):
        abort("compatible Admission preflight retry changed the original-session snapshot anchor")
    frozen_original, original_digest = validate_original_session_snapshot(
        ROOT / original_ref["path"], session_id=BASELINE_SESSION_ID, expected_sha256=original_ref["sha256"],
    )
    if original_digest != original_ref["sha256"] or frozen_original != original:
        abort("compatible Admission preflight retry is not bound to the immutable original session")

    old_source_ref = _require_path_ref(
        record.get("old_source_review"), old_request.get("source_review_path"), "old source review",
    )
    new_source_ref = _require_path_ref(
        record.get("new_source_review"), request.get("source_review_path"), "new source review",
    )
    if old_source_ref.get("sha256") != old_request.get("source_review_sha256") or new_source_ref.get("sha256") != request.get("source_review_sha256"):
        abort("compatible Admission preflight retry changed an approved source-review digest")
    _historical_approved_review(old_request, "v4 old")

    _validate_preflight_retry_record(archived, old_request, original)
    v1_digest = canonical_json_digest(old_retry)
    first_ref = record.get("old_preflight_retry_snapshot")
    if not isinstance(first_ref, dict) or set(first_ref) != {"path", "sha256", "canonical_sha256"} or first_ref.get("canonical_sha256") != v1_digest:
        abort("compatible Admission preflight retry does not bind the original v3-to-v4 audit digest")
    _validate_admission_retry_snapshots(record, archived, original, old_request)
    return old_request, archived


def validate_failed_admission_preflight_resume_gate(session_document: dict[str, Any]) -> None:
    """Allow only the exact selected-capacity failure to enter the audited v4->v5 transition."""
    if (
        not isinstance(session_document, dict)
        or session_document.get("status") != "FAILED"
        or session_document.get("stage") != "selected_dataset_native_capacity_preflight"
    ):
        abort("compatible Admission retry requires the exact FAILED/selected_dataset_native_capacity_preflight stage")
    request = session_document.get("compatible_resume_request")
    second_retry = session_document.get("compatible_admission_preflight_retry")
    if second_retry is not None:
        current_request, original = validate_compatible_resume_request(
            session_document, expected_session_id=BASELINE_SESSION_ID,
        )
        if current_request != request:
            abort("compatible Admission retry request changed after its audited transition")
        checks = session_document.get("capacity_checks")
        original_checks = original.get("capacity_checks")
        if not isinstance(checks, list) or not isinstance(original_checks, list) or checks[:4] != original_checks:
            abort("compatible Admission retry stage has an invalid capacity history prefix")
        _validate_no_proof_outputs()
        return
    old_request = request
    if not isinstance(old_request, dict):
        abort("compatible Admission retry has no old v4 request")
    if canonical_json_digest(old_request) == _PRE_RETRY_REQUEST_SHA256:
        abort("compatible Admission retry requires the already approved v4 source-review request")
    old_snapshot_rel = _safe_relative_path(old_request.get("old_session_snapshot_path"), "original-session snapshot")
    original, digest = validate_original_session_snapshot(
        ROOT / old_snapshot_rel,
        session_id=BASELINE_SESSION_ID,
        expected_sha256=old_request.get("old_session_snapshot_sha256"),
    )
    if digest != old_request.get("old_session_snapshot_sha256"):
        abort("compatible Admission retry original-session snapshot changed")
    _validate_preflight_retry_record(session_document, old_request, original)
    _validate_admission_preflight_state(session_document, original, old_request, check_outputs=True)
    _historical_approved_review(old_request, "v4 old")


def _validate_no_proof_outputs() -> None:
    output_root = ROOT / _COMPAT_OUTPUT_ROOT
    for dataset in _COMPAT_DATASETS:
        output = output_root / dataset
        if output.is_symlink():
            abort(f"compatible Admission retry {dataset} output is a symlink")
        for name in ("proof.json", "result.json", "native_verify_metrics.json", "core_run_manifest.json"):
            if (output / name).exists() or (output / name).is_symlink():
                abort(f"compatible Admission retry refuses existing {dataset} proof/result artifact: {name}")


def prepare_compatible_admission_preflight_retry(
    session_file: Path, request: dict[str, Any],
) -> dict[str, Any] | None:
    """Freeze and audit the one permitted v4->v5 selected-capacity retry."""
    expected_session = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if (
        session_file.is_symlink() or not session_file.is_file()
        or session_file.resolve(strict=True) != expected_session.resolve(strict=True)
    ):
        abort("compatible Admission preflight retry must use the exact original session record")
    try:
        live_raw = session_file.read_bytes()
        live_document = json.loads(live_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"cannot read compatible Admission preflight retry session: {exc}")
    if not isinstance(live_document, dict):
        abort("compatible Admission preflight retry session is malformed")
    existing = live_document.get("compatible_resume_request")
    second_retry = live_document.get("compatible_admission_preflight_retry")
    if existing == request:
        if second_retry is not None:
            validate_compatible_resume_request(live_document, expected_session_id=BASELINE_SESSION_ID)
        return None
    if live_document.get("compatible_preflight_retry") is None:
        # Initial original->v3/v4 binding and the first native-build retry are
        # owned by prepare_compatible_preflight_retry, not this second link.
        return None
    if second_retry is not None:
        # A narrowly pinned interrupted-before-Prove recovery is a separate
        # transition. Its eligibility and archive are handled by the dedicated
        # V6 helper below; this V4->V5 helper must not reinterpret it.
        if live_document.get("compatible_interrupted_before_prove_recovery") is not None:
            abort("compatible interrupted recovery request cannot be replaced by another source migration")
        return None
    if not isinstance(existing, dict):
        abort("compatible Admission preflight retry requires the exact migrated v4 request and audit")
    old_request = existing
    validate_failed_admission_preflight_resume_gate(live_document)
    old_snapshot_rel = _safe_relative_path(old_request.get("old_session_snapshot_path"), "original-session snapshot")
    original, original_sha = validate_original_session_snapshot(
        ROOT / old_snapshot_rel,
        session_id=BASELINE_SESSION_ID,
        expected_sha256=old_request.get("old_session_snapshot_sha256"),
    )
    if request.get("old_session_snapshot_path") != old_snapshot_rel or request.get("old_session_snapshot_sha256") != original_sha:
        abort("compatible Admission preflight new request changed the original-session anchor")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("session_id") != BASELINE_SESSION_ID
        or request.get("output_root") != _COMPAT_OUTPUT_ROOT
        or request.get("matrix") != _COMPAT_MATRIX
        or request.get("selected_datasets") != _COMPAT_DATASETS
        or request.get("retained_setup_datasets") != ["ppi"]
        or canonical_json_digest(request) == canonical_json_digest(old_request)
    ):
        abort("compatible Admission preflight new request is not a distinct request for the same session")

    review_path = ROOT / _safe_relative_path(request.get("source_review_path"), "new source review")
    review, review_sha, delta = validate_source_review(review_path, expected_session_id=BASELINE_SESSION_ID)
    if (
        review_sha != request.get("source_review_sha256")
        or review.get("decision") != "APPROVED"
        or delta["diff_summary"]["combined_diff_sha256"] != request.get("source_delta_sha256")
        or compatible_source_review_index_digest(delta) != request.get("snapshot_index_inventory_sha256")
    ):
        abort("compatible Admission preflight new request does not bind the exact approved source review")

    _validate_admission_preflight_state(live_document, original, old_request, check_outputs=True)
    _historical_approved_review(old_request, "v4 old")

    retry_dir = _admission_preflight_retry_directory(BASELINE_SESSION_ID)
    failed_path = retry_dir / "failed_session.json"
    old_request_path = retry_dir / "old_request.json"
    old_v1_path = retry_dir / "old_preflight_retry.json"
    map_path = retry_dir / "compatible_native_admissions.json"
    failed_sha = _write_immutable_snapshot(failed_path, live_raw, "admission failed-session")
    old_request_raw = _json_snapshot_bytes(old_request)
    old_request_sha = _write_immutable_snapshot(old_request_path, old_request_raw, "admission old-request")
    old_v1 = live_document.get("compatible_preflight_retry")
    if not isinstance(old_v1, dict):
        abort("compatible Admission preflight retry lacks the original v3-to-v4 audit record")
    old_v1_raw = _json_snapshot_bytes(old_v1)
    old_v1_sha = _write_immutable_snapshot(old_v1_path, old_v1_raw, "first retry audit")
    current_map = live_document.get("compatible_native_admissions")
    if not isinstance(current_map, dict):
        abort("compatible Admission preflight retry has no exact two-dataset Admission map")
    map_raw = _json_snapshot_bytes(current_map)
    map_sha = _write_immutable_snapshot(map_path, map_raw, "Admission map")

    file_entries: list[dict[str, Any]] = []
    for dataset in _COMPAT_DATASETS:
        current_ref = current_map[dataset]
        source_admission = ROOT / current_ref["path"]
        source_settings = source_admission.parent / "settings.json"
        _regular_file(source_admission, f"{dataset} failed Admission")
        _regular_file(source_settings, f"{dataset} failed settings")
        admission_raw = source_admission.read_bytes()
        settings_raw = source_settings.read_bytes()
        if _digest_bytes(admission_raw) != current_ref["sha256"] or _digest_bytes(settings_raw) != current_ref["settings_sha256"]:
            abort(f"compatible Admission preflight {dataset} source bytes changed before snapshot")
        admission_snapshot_path = retry_dir / "admissions" / dataset / "admission.json"
        settings_snapshot_path = retry_dir / "admissions" / dataset / "settings.json"
        admission_snapshot_sha = _write_immutable_snapshot(admission_snapshot_path, admission_raw, f"{dataset} Admission")
        settings_snapshot_sha = _write_immutable_snapshot(settings_snapshot_path, settings_raw, f"{dataset} settings")
        file_entries.append({
            "dataset": dataset,
            "source_admission": {"path": current_ref["path"], "sha256": current_ref["sha256"]},
            "source_settings": {"path": source_settings.relative_to(ROOT).as_posix(), "sha256": current_ref["settings_sha256"]},
            "snapshot_admission": {"path": admission_snapshot_path.relative_to(ROOT).as_posix(), "sha256": admission_snapshot_sha},
            "snapshot_settings": {"path": settings_snapshot_path.relative_to(ROOT).as_posix(), "sha256": settings_snapshot_sha},
        })

    compatibility_dir = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID / "compatibility"
    report_path = compatibility_dir / "capacity_precheck_current.json"
    report_record: dict[str, Any] | None = None
    if report_path.exists() or report_path.is_symlink():
        _regular_file(report_path, "failed current capacity report")
        report_raw = report_path.read_bytes()
        report_snapshot_path = retry_dir / "capacity_precheck_current.json"
        report_snapshot_sha = _write_immutable_snapshot(report_snapshot_path, report_raw, "current capacity report")
        report_record = {
            "source": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": _digest_bytes(report_raw)},
            "snapshot": {"path": report_snapshot_path.relative_to(ROOT).as_posix(), "sha256": report_snapshot_sha},
        }

    record: dict[str, Any] = {
        "schema": ADMISSION_PREFLIGHT_RETRY_SCHEMA,
        "session_id": BASELINE_SESSION_ID,
        "failed_session_snapshot": {"path": failed_path.relative_to(ROOT).as_posix(), "sha256": failed_sha},
        "old_request_snapshot": {"path": old_request_path.relative_to(ROOT).as_posix(), "sha256": old_request_sha},
        "old_request_canonical_sha256": canonical_json_digest(old_request),
        "old_preflight_retry_snapshot": {
            "path": old_v1_path.relative_to(ROOT).as_posix(),
            "sha256": old_v1_sha,
            "canonical_sha256": canonical_json_digest(old_v1),
        },
        "original_session_snapshot": {"path": old_snapshot_rel, "sha256": original_sha},
        "old_source_review": {"path": old_request["source_review_path"], "sha256": old_request["source_review_sha256"]},
        "new_source_review": {"path": request["source_review_path"], "sha256": request["source_review_sha256"]},
        "new_request_canonical_sha256": canonical_json_digest(request),
        "prior_prover_identity_sha256": live_document["compatible_prover_identity_sha256"],
        "prior_capacity_source_identity_sha256": live_document["compatible_capacity_source_identity_sha256"],
        "admissions_map_snapshot": {"path": map_path.relative_to(ROOT).as_posix(), "sha256": map_sha},
        "admission_file_snapshots": file_entries,
        "capacity_report_snapshot": report_record,
    }
    candidate = copy.deepcopy(live_document)
    candidate["compatible_resume_request"] = request
    candidate["compatible_admission_preflight_retry"] = record
    for name in (
        "compatible_native_admissions", "compatible_prover_identity", "compatible_prover_identity_sha256",
        "compatible_capacity_source_identity", "compatible_capacity_source_identity_sha256",
    ):
        candidate.pop(name, None)
    validate_compatible_resume_request(candidate, expected_session_id=BASELINE_SESSION_ID)
    return record


def _interrupted_before_prove_recovery_directory(session_id: str) -> Path:
    if session_id != BASELINE_SESSION_ID:
        abort("interrupted-before-Prove recovery is bound to the one original session")
    return ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "interrupted_before_prove_v6"


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _tree_inventory(path: Path, label: str) -> dict[str, Any]:
    """Record metadata for an untrusted partial tree without hashing large files."""
    try:
        root_stat = path.lstat()
    except OSError as exc:
        abort(f"interrupted-before-Prove {label} is unavailable: {exc}")
    if not stat.S_ISDIR(root_stat.st_mode):
        abort(f"interrupted-before-Prove {label} must be a real directory")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            abort(f"cannot inventory interrupted-before-Prove {label}: {exc}")
        for child in children:
            child_path = Path(child.path)
            try:
                info = child_path.lstat()
            except OSError as exc:
                abort(f"interrupted-before-Prove {label} changed during inventory: {exc}")
            if stat.S_ISLNK(info.st_mode):
                abort(f"interrupted-before-Prove {label} contains a symlink: {child_path}")
            if stat.S_ISDIR(info.st_mode):
                kind = "directory"
                size = 0
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
                size = info.st_size
            else:
                abort(f"interrupted-before-Prove {label} contains a non-regular entry: {child_path}")
            child_relative = relative / child.name
            entries.append({
                "path": child_relative.as_posix(),
                "kind": kind,
                "size_bytes": size,
                "mode": stat.S_IMODE(info.st_mode),
                "device": info.st_dev,
                "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns,
            })
            if kind == "directory":
                visit(child_path, child_relative)

    visit(path, Path())
    return {
        "schema": "ezkl.compatible-interrupted-partial-tree-inventory.v1",
        "root": {
            "device": root_stat.st_dev,
            "inode": root_stat.st_ino,
            "mode": stat.S_IMODE(root_stat.st_mode),
            "mtime_ns": root_stat.st_mtime_ns,
        },
        "entries": entries,
    }


def _read_inventory_snapshot(ref: object, expected_path: Path, label: str) -> dict[str, Any]:
    bound = _require_path_ref(ref, expected_path.relative_to(ROOT).as_posix(), label)
    raw = _snapshot_file(ROOT / bound["path"], relative=bound["path"], expected_digest=bound["sha256"])
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"interrupted-before-Prove {label} is invalid JSON: {exc}")
    if not isinstance(value, dict) or value.get("schema") != "ezkl.compatible-interrupted-partial-tree-inventory.v1":
        abort(f"interrupted-before-Prove {label} has an unsupported inventory schema")
    return value


def _rename_noreplace(source: Path, destination: Path, label: str) -> None:
    """Use Linux renameat2(RENAME_NOREPLACE); never emulate with an overwriting rename."""
    if not _path_exists(source) or _path_exists(destination):
        abort(f"interrupted-before-Prove {label} rename preconditions changed")
    try:
        source_stat = source.lstat()
        parent_stat = destination.parent.stat()
    except OSError as exc:
        abort(f"interrupted-before-Prove {label} rename path is unavailable: {exc}")
    if stat.S_ISLNK(source_stat.st_mode) or source_stat.st_dev != parent_stat.st_dev:
        abort(f"interrupted-before-Prove {label} rename is symlinked or crosses filesystems")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        abort("interrupted-before-Prove recovery requires renameat2(RENAME_NOREPLACE)")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            abort("filesystem does not support no-replace recovery rename")
        abort(f"interrupted-before-Prove {label} no-replace rename failed: {os.strerror(error)}")
    _fsync_parent(source.parent)
    if destination.parent != source.parent:
        _fsync_parent(destination.parent)


def _reconcile_recovery_move(
    *, source: Path, destination: Path, label: str, inventory: dict[str, Any] | None = None,
    sha256: str | None = None,
) -> None:
    source_exists = _path_exists(source)
    destination_exists = _path_exists(destination)
    if source_exists == destination_exists:
        abort(f"interrupted-before-Prove {label} has conflicting source/archive state")
    chosen = source if source_exists else destination
    try:
        info = chosen.lstat()
    except OSError as exc:
        abort(f"interrupted-before-Prove {label} path is unsafe: {exc}")
    if stat.S_ISDIR(info.st_mode):
        if inventory is None:
            abort(f"interrupted-before-Prove {label} lacks a bound directory inventory")
        actual = _tree_inventory(chosen, label)
        if canonical_json_digest(actual) != canonical_json_digest(inventory):
            abort(f"interrupted-before-Prove {label} inventory differs from its immutable plan")
    elif stat.S_ISREG(info.st_mode):
        if sha256 is None or _digest_bytes(chosen.read_bytes()) != sha256:
            abort(f"interrupted-before-Prove {label} bytes differ from their immutable plan")
    else:
        abort(f"interrupted-before-Prove {label} is not a regular file or directory")
    if source_exists:
        _rename_noreplace(source, destination, label)
        if stat.S_ISDIR(info.st_mode):
            actual = _tree_inventory(destination, label)
            if canonical_json_digest(actual) != canonical_json_digest(inventory):
                abort(f"interrupted-before-Prove {label} changed during no-replace rename")


def _v5_interrupted_archive_paths() -> dict[str, Path]:
    archive = _interrupted_before_prove_recovery_directory(BASELINE_SESSION_ID)
    return {
        "archive": archive,
        "failed_session": archive / "failed_session.json",
        "old_request": archive / "old_request.json",
        "old_v1": archive / "old_preflight_retry.json",
        "old_v2": archive / "old_admission_preflight_retry.json",
        "old_review": archive / "old_source_review_v5.json",
        "admissions_map": archive / "compatible_native_admissions.json",
        "capacity_attempts": archive / "capacity_admission_attempts.json",
        "capacity_report_snapshot": archive / "capacity_precheck_current.json",
        "capacity_report_archived": archive / "capacity_precheck_current_source.json",
        "admission_sidecar": archive / "admission_sidecar",
        "admission_inventory": archive / "admission_sidecar_inventory.json",
        "admission_files": archive / "admissions",
        "partial_output": archive / "ogbn_arxiv_partial",
        "partial_inventory": archive / "ogbn_arxiv_partial_inventory.json",
        "plan": archive / "recovery_plan.json",
    }


def _validate_v5_interrupted_state(
    document: dict[str, Any], *, request: dict[str, Any], original: dict[str, Any],
    current_files: dict[str, tuple[Path, Path]], report_path: Path,
) -> dict[str, Any]:
    expected_compatible = {
        "compatible_resume_request", "compatible_preflight_retry", "compatible_admission_preflight_retry",
        "compatible_prover_identity", "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "compatible_native_admissions",
    }
    if (
        document.get("schema_version") != 1 or document.get("baseline") != "ezkl"
        or document.get("session_id") != BASELINE_SESSION_ID or document.get("status") != "INTERRUPTED"
        or document.get("stage") != "ogbn_arxiv_setup" or document.get("output_root") != _COMPAT_OUTPUT_ROOT
        or document.get("matrix") != _COMPAT_MATRIX or document.get("selected_datasets") != _COMPAT_DATASETS
        or document.get("completed_cells") not in (None, {})
        or document.get("compatible_resume_request") is None
        or canonical_json_digest(document["compatible_resume_request"]) != _V5_REQUEST_CANONICAL_SHA256
        or {key for key in document if key.startswith("compatible_")} != expected_compatible
        or document.get("native_admissions") != original.get("native_admissions")
        or document.get("setup_reuse_receipts") != original.get("setup_reuse_receipts")
        or document.get("stage_receipts") != original.get("stage_receipts")
        or document.get("compatible_stage_receipts") not in (None, {})
        or document.get("results") not in (None, {})
        or document.get("verified_results") not in (None, {})
    ):
        abort("interrupted-before-Prove recovery is not the exact poststop PPI/OGBN session before proof progress")
    if (
        set(document.get("setup_reuse_receipts", {})) != {"ppi"}
        or set(document.get("stage_receipts", {})) != {"ppi"}
        or document.get("capacity_checks") is None
        or len(document["capacity_checks"]) != 7
        or canonical_json_digest(document["capacity_checks"][:4]) != canonical_json_digest(original.get("capacity_checks"))
        or [(item.get("dataset"), item.get("phase")) for item in document["capacity_checks"][4:]]
        != [("ogbn_arxiv", "setup"), ("ppi", "witness_prove_verify"), ("ogbn_arxiv", "setup")]
    ):
        abort("interrupted-before-Prove recovery capacity history is not the exact four-prefix plus three failed-attempt gates")
    identity = document.get("compatible_prover_identity")
    capacity_identity = document.get("compatible_capacity_source_identity")
    if (
        not isinstance(identity, dict) or canonical_json_digest(identity) != _V5_PROVER_IDENTITY_SHA256
        or document.get("compatible_prover_identity_sha256") != _V5_PROVER_IDENTITY_SHA256
        or not isinstance(capacity_identity, dict) or canonical_json_digest(capacity_identity) != _V5_CAPACITY_IDENTITY_SHA256
        or document.get("compatible_capacity_source_identity_sha256") != _V5_CAPACITY_IDENTITY_SHA256
    ):
        abort("interrupted-before-Prove recovery current identity pair differs from the pinned v5 attempt")
    failed = document.get("dataset_failures")
    ppi_failure = failed.get("ppi") if isinstance(failed, dict) else None
    if (
        not isinstance(failed, dict) or set(failed) != {"ppi"} or not isinstance(ppi_failure, dict)
        or ppi_failure.get("stage") != "ppi_configured_admission"
        or ppi_failure.get("exception_type") != "NameError"
        or ppi_failure.get("error") != "name 'compatible_current_capacity_source_identity' is not defined"
        or "compatible_current_capacity_source_identity" not in str(ppi_failure.get("traceback", ""))
    ):
        abort("interrupted-before-Prove recovery active failure is not the exact PPI configured-admission NameError")
    attempts = document.get("capacity_admission_attempts")
    report_rel = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/capacity_precheck_current.json"
    if (
        not isinstance(attempts, dict)
        or set(attempts) != {"datasets", "readiness", "gate_status", "report_path", "report_sha256", "purpose", "failures", "audit_failures"}
        or attempts.get("datasets") != _COMPAT_DATASETS
        or attempts.get("readiness") != "NOT_ESTABLISHED"
        or attempts.get("gate_status") != "current_v4_pk_payload_lower_bound_unavailable"
        or attempts.get("report_path") != report_rel
        or attempts.get("report_sha256") != _V5_CAPACITY_REPORT_SHA256
        or attempts.get("purpose") != "selected-native-admission-and-v4-PK-lower-bound-report-only; no setup or proof success implied"
        or attempts.get("failures") != {} or attempts.get("audit_failures") != {}
    ):
        abort("interrupted-before-Prove recovery active capacity report reference is not the exact pinned v5 attempt")
    _regular_file(report_path, "V5 current capacity report")
    if _digest_bytes(report_path.read_bytes()) != _V5_CAPACITY_REPORT_SHA256:
        abort("interrupted-before-Prove recovery current capacity report bytes differ from the pinned v5 report")

    old_request = document["compatible_resume_request"]
    _historical_approved_review(old_request, "v5 old")
    if old_request.get("source_review_path") != _V5_SOURCE_REVIEW_PATH or old_request.get("source_review_sha256") != _V5_SOURCE_REVIEW_SHA256:
        abort("interrupted-before-Prove recovery v5 request is not bound to the approved historical review")
    old_retry = document.get("compatible_preflight_retry")
    admission_retry = document.get("compatible_admission_preflight_retry")
    if (
        not isinstance(old_retry, dict) or canonical_json_digest(old_retry) != _V5_V1_RETRY_CANONICAL_SHA256
        or not isinstance(admission_retry, dict) or canonical_json_digest(admission_retry) != _V5_V2_RETRY_CANONICAL_SHA256
    ):
        abort("interrupted-before-Prove recovery v1/v2 migration audit chain is not the pinned v5 chain")
    # The V2 admission-retry validator authenticates the archived V4 request,
    # then validates V1 against that request before binding V2 to V5.  Calling
    # V1 directly with the current V5 request would conflate the two epochs.
    _validate_admission_preflight_retry_record(document, old_request, original)

    state_for_map = copy.deepcopy(document)
    state_for_map["status"] = "FAILED"
    state_for_map["stage"] = "selected_dataset_native_capacity_preflight"
    state_for_map["capacity_checks"] = original.get("capacity_checks")
    _validate_admission_preflight_state(
        state_for_map, original, old_request, check_outputs=False, current_files=current_files,
    )
    retained = {
        "setup_reuse_receipts": original.get("setup_reuse_receipts"),
        "stage_receipts": original.get("stage_receipts", {}).get("ppi"),
    }
    retained_digest = canonical_json_digest(retained)
    return {"request": old_request, "v1": old_retry, "v2": admission_retry, "retained_ppi_chain_sha256": retained_digest}


def validate_interrupted_before_prove_resume_candidate(
    session_document: dict[str, Any], request: dict[str, Any],
) -> None:
    """Validate the one pinned V5 interrupted-before-Prove state before a V6 transition."""
    session_file = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file():
        abort("interrupted-before-Prove recovery exact session file is missing or unsafe")
    raw = session_file.read_bytes()
    if _digest_bytes(raw) != _V5_POSTSTOP_SESSION_SHA256:
        abort("interrupted-before-Prove recovery failed-session bytes differ from the pinned poststop state")
    try:
        live = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"interrupted-before-Prove recovery failed-session JSON is invalid: {exc}")
    if not isinstance(live, dict) or canonical_json_digest(live) != canonical_json_digest(session_document):
        abort("interrupted-before-Prove recovery in-memory session differs from the pinned live bytes")
    old_request = live.get("compatible_resume_request")
    original_path = _snapshot_path(BASELINE_SESSION_ID)
    if (
        not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA
        or request.get("session_id") != BASELINE_SESSION_ID or request.get("output_root") != _COMPAT_OUTPUT_ROOT
        or request.get("matrix") != _COMPAT_MATRIX or request.get("selected_datasets") != _COMPAT_DATASETS
        or request.get("retained_setup_datasets") != ["ppi"]
        or request.get("old_session_snapshot_path") != original_path
        or request.get("old_session_snapshot_sha256") != _ORIGINAL_SESSION_SHA256
        or canonical_json_digest(request) == _V5_REQUEST_CANONICAL_SHA256
        or live.get("compatible_interrupted_before_prove_recovery") is not None
    ):
        abort("interrupted-before-Prove recovery request is not a distinct exact V6 request for the original session")
    original, original_sha = validate_original_session_snapshot(
        ROOT / original_path, session_id=BASELINE_SESSION_ID, expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if original_sha != _ORIGINAL_SESSION_SHA256:
        abort("interrupted-before-Prove recovery original session snapshot changed")
    archive = _interrupted_before_prove_recovery_directory(BASELINE_SESSION_ID)
    paths = _v5_interrupted_archive_paths()
    plan_path = paths["plan"]
    if _path_exists(plan_path):
        _regular_file(plan_path, "recovery plan")
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            abort(f"interrupted-before-Prove recovery plan is invalid: {exc}")
        _validate_recovery_snapshots(plan, live, request, original)
        current_files = _recovery_current_files(plan)
        report_path = ROOT / plan["capacity_report"]["snapshot"]["path"]
        _validate_v5_interrupted_state(live, request=request, original=original, current_files=current_files, report_path=report_path)
        _validate_uncommitted_recovery_plan(plan, live, request, original)
        _validate_recovery_output_absence(allow_archived_ogbn_partial=True)
        _validate_v6_request_review(request)
        return
    current_files = {
        dataset: (
            ROOT / live["compatible_native_admissions"][dataset]["path"],
            ROOT / Path(live["compatible_native_admissions"][dataset]["path"]).parent / "settings.json",
        )
        for dataset in _COMPAT_DATASETS
    }
    report_ref = live["capacity_admission_attempts"]["report_path"]
    _validate_v5_interrupted_state(
        live, request=request, original=original, current_files=current_files, report_path=ROOT / report_ref,
    )
    _validate_v6_request_review(request)
    _validate_recovery_output_absence()


def _validate_recovery_output_absence(*, allow_archived_ogbn_partial: bool = False) -> None:
    """Recheck the narrow pre-Prove boundary immediately before committing V6."""
    _validate_no_proof_outputs()
    output = ROOT / _COMPAT_OUTPUT_ROOT / "ogbn_arxiv"
    if output.is_symlink() or (not output.is_dir() and not (allow_archived_ogbn_partial and not _path_exists(output))):
        abort("interrupted-before-Prove recovery OGBN partial output is missing or unsafe")
    for forbidden in ("pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json"):
        if (output / forbidden).exists() or (output / forbidden).is_symlink():
            abort(f"interrupted-before-Prove recovery refuses an unreceipted OGBN setup artifact: {forbidden}")


def _validate_v6_request_review(request: dict[str, Any]) -> None:
    review_path = ROOT / _safe_relative_path(request.get("source_review_path"), "new source review")
    review, review_sha, delta = validate_source_review(review_path, expected_session_id=BASELINE_SESSION_ID)
    if (
        review_sha != request.get("source_review_sha256") or review.get("decision") != "APPROVED"
        or delta["diff_summary"]["combined_diff_sha256"] != request.get("source_delta_sha256")
        or compatible_source_review_index_digest(delta) != request.get("snapshot_index_inventory_sha256")
    ):
        abort("interrupted-before-Prove recovery V6 request is not bound to the exact current approved source review")


def _recovery_current_files(plan: dict[str, Any]) -> dict[str, tuple[Path, Path]]:
    entries = plan.get("admission_file_snapshots")
    if not isinstance(entries, list) or [item.get("dataset") if isinstance(item, dict) else None for item in entries] != _COMPAT_DATASETS:
        abort("interrupted-before-Prove recovery Admission snapshot inventory is malformed")
    result: dict[str, tuple[Path, Path]] = {}
    archive = _interrupted_before_prove_recovery_directory(BASELINE_SESSION_ID)
    for dataset, item in zip(_COMPAT_DATASETS, entries):
        if not isinstance(item, dict) or set(item) != {"dataset", "admission", "settings"}:
            abort(f"interrupted-before-Prove recovery {dataset} Admission snapshots are malformed")
        admission_ref = _require_path_ref(item.get("admission"), (archive / "admissions" / dataset / "admission.json").relative_to(ROOT).as_posix(), f"{dataset} archived Admission")
        settings_ref = _require_path_ref(item.get("settings"), (archive / "admissions" / dataset / "settings.json").relative_to(ROOT).as_posix(), f"{dataset} archived settings")
        _snapshot_file(ROOT / admission_ref["path"], relative=admission_ref["path"], expected_digest=admission_ref["sha256"])
        _snapshot_file(ROOT / settings_ref["path"], relative=settings_ref["path"], expected_digest=settings_ref["sha256"])
        result[dataset] = (ROOT / admission_ref["path"], ROOT / settings_ref["path"])
    return result


def _validate_recovery_snapshots(plan: dict[str, Any], failed: dict[str, Any], request: dict[str, Any], original: dict[str, Any]) -> None:
    paths = _v5_interrupted_archive_paths()
    exact = {
        "failed_session_snapshot", "old_request_snapshot", "old_preflight_retry_snapshot",
        "old_admission_preflight_retry_snapshot", "old_source_review_snapshot", "original_session_snapshot",
        "new_source_review", "new_request_canonical_sha256", "old_request_canonical_sha256",
        "v1_retry_canonical_sha256", "v2_retry_canonical_sha256", "capacity_checks_canonical_sha256",
        "prior_prover_identity_sha256", "prior_capacity_source_identity_sha256", "retained_ppi_chain_sha256",
        "admissions_map_snapshot", "admission_file_snapshots", "capacity_admission_attempts_snapshot",
        "capacity_report", "admission_sidecar", "admission_sidecar_inventory", "ogbn_partial",
        "ogbn_partial_inventory", "no_proof_outputs", "no_ogbn_setup_artifacts", "schema", "session_id",
    }
    if not isinstance(plan, dict) or set(plan) != exact:
        abort("interrupted-before-Prove recovery plan has unknown, missing, or malformed fields")
    marker_path = paths["plan"]
    _regular_file(marker_path, "immutable recovery plan")
    marker_raw = marker_path.read_bytes()
    if _digest_bytes(marker_raw) != _digest_bytes(_json_snapshot_bytes(plan)):
        abort("interrupted-before-Prove recovery plan file differs from its session-bound record")
    if (
        plan.get("schema") != INTERRUPTED_BEFORE_PROVE_RECOVERY_SCHEMA
        or plan.get("session_id") != BASELINE_SESSION_ID
        or plan.get("new_request_canonical_sha256") != canonical_json_digest(request)
        or plan.get("old_request_canonical_sha256") != _V5_REQUEST_CANONICAL_SHA256
        or plan.get("v1_retry_canonical_sha256") != _V5_V1_RETRY_CANONICAL_SHA256
        or plan.get("v2_retry_canonical_sha256") != _V5_V2_RETRY_CANONICAL_SHA256
        or plan.get("prior_prover_identity_sha256") != _V5_PROVER_IDENTITY_SHA256
        or plan.get("prior_capacity_source_identity_sha256") != _V5_CAPACITY_IDENTITY_SHA256
        or plan.get("capacity_checks_canonical_sha256") != canonical_json_digest(failed.get("capacity_checks"))
        or plan.get("retained_ppi_chain_sha256") != canonical_json_digest({
            "setup_reuse_receipts": original.get("setup_reuse_receipts"),
            "stage_receipts": original.get("stage_receipts", {}).get("ppi"),
        })
        or plan.get("original_session_snapshot") != {"path": _snapshot_path(BASELINE_SESSION_ID), "sha256": _ORIGINAL_SESSION_SHA256}
        or plan.get("new_source_review") != {"path": request.get("source_review_path"), "sha256": request.get("source_review_sha256")}
        or plan.get("no_proof_outputs") != [
            f"{_COMPAT_OUTPUT_ROOT}/{dataset}/{name}"
            for dataset in _COMPAT_DATASETS
            for name in ("proof.json", "result.json", "native_verify_metrics.json", "core_run_manifest.json")
        ]
        or plan.get("no_ogbn_setup_artifacts") != [
            f"{_COMPAT_OUTPUT_ROOT}/ogbn_arxiv/{name}"
            for name in ("pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json")
        ]
    ):
        abort("interrupted-before-Prove recovery plan is not bound to the pinned V5 session/audit/request chain")
    for key, expected in (
        ("failed_session_snapshot", paths["failed_session"]),
        ("old_request_snapshot", paths["old_request"]),
        ("old_preflight_retry_snapshot", paths["old_v1"]),
        ("old_admission_preflight_retry_snapshot", paths["old_v2"]),
        ("old_source_review_snapshot", paths["old_review"]),
        ("admissions_map_snapshot", paths["admissions_map"]),
        ("capacity_admission_attempts_snapshot", paths["capacity_attempts"]),
    ):
        ref = _require_path_ref(plan.get(key), expected.relative_to(ROOT).as_posix(), f"{key}")
        _snapshot_file(ROOT / ref["path"], relative=ref["path"], expected_digest=ref["sha256"])
    refs = {
        "failed_session_snapshot": paths["failed_session"],
        "old_request_snapshot": paths["old_request"],
        "old_preflight_retry_snapshot": paths["old_v1"],
        "old_admission_preflight_retry_snapshot": paths["old_v2"],
        "old_source_review_snapshot": paths["old_review"],
        "admissions_map_snapshot": paths["admissions_map"],
        "capacity_admission_attempts_snapshot": paths["capacity_attempts"],
    }
    for key, expected in refs.items():
        ref = plan[key]
        if key == "failed_session_snapshot" and ref["sha256"] != _V5_POSTSTOP_SESSION_SHA256:
            abort("interrupted-before-Prove archived V5 session digest changed")
    old_request_raw = _snapshot_file(ROOT / plan["old_request_snapshot"]["path"], relative=plan["old_request_snapshot"]["path"], expected_digest=plan["old_request_snapshot"]["sha256"])
    try:
        old_request = json.loads(old_request_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"interrupted-before-Prove archived V5 request is invalid: {exc}")
    if not isinstance(old_request, dict) or canonical_json_digest(old_request) != _V5_REQUEST_CANONICAL_SHA256:
        abort("interrupted-before-Prove archived V5 request differs from its fixed canonical digest")
    if json.loads(_snapshot_file(ROOT / plan["failed_session_snapshot"]["path"], relative=plan["failed_session_snapshot"]["path"], expected_digest=plan["failed_session_snapshot"]["sha256"])) != failed:
        abort("interrupted-before-Prove archived failed session differs from its immutable plan")
    for key, field in (("old_preflight_retry_snapshot", "compatible_preflight_retry"), ("old_admission_preflight_retry_snapshot", "compatible_admission_preflight_retry")):
        raw = _snapshot_file(ROOT / plan[key]["path"], relative=plan[key]["path"], expected_digest=plan[key]["sha256"])
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            abort(f"interrupted-before-Prove archived {field} record is invalid: {exc}")
        if canonical_json_digest(value) != canonical_json_digest(failed.get(field)):
            abort(f"interrupted-before-Prove archived {field} differs from the V5 session")
    review_ref = _require_path_ref(plan.get("old_source_review_snapshot"), paths["old_review"].relative_to(ROOT).as_posix(), "V5 source-review snapshot")
    review_raw = _snapshot_file(ROOT / review_ref["path"], relative=review_ref["path"], expected_digest=review_ref["sha256"])
    if review_ref["sha256"] != _V5_SOURCE_REVIEW_SHA256 or _digest_bytes(review_raw) != old_request.get("source_review_sha256"):
        abort("interrupted-before-Prove archived V5 review differs from its approved request")
    try:
        review = json.loads(review_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"interrupted-before-Prove archived V5 review is invalid: {exc}")
    if not isinstance(review, dict) or review.get("decision") != "APPROVED" or review.get("reviewer") != "root":
        abort("interrupted-before-Prove archived V5 review is not approved")
    if (
        review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("session_id") != BASELINE_SESSION_ID
        or review.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
    ):
        abort("interrupted-before-Prove archived V5 review schema/session binding changed")
    if plan.get("old_source_review_snapshot") != {
        "path": paths["old_review"].relative_to(ROOT).as_posix(),
        "sha256": _V5_SOURCE_REVIEW_SHA256,
    }:
        abort("interrupted-before-Prove archived V5 review path changed")
    if plan.get("capacity_report") != {
        "source_path": f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/capacity_precheck_current.json",
        "archive_path": paths["capacity_report_archived"].relative_to(ROOT).as_posix(),
        "snapshot": {"path": paths["capacity_report_snapshot"].relative_to(ROOT).as_posix(), "sha256": _V5_CAPACITY_REPORT_SHA256},
    }:
        abort("interrupted-before-Prove archived capacity-report reference is malformed")
    _snapshot_file(ROOT / plan["capacity_report"]["snapshot"]["path"], relative=plan["capacity_report"]["snapshot"]["path"], expected_digest=_V5_CAPACITY_REPORT_SHA256)
    attempts_raw = _snapshot_file(ROOT / plan["capacity_admission_attempts_snapshot"]["path"], relative=plan["capacity_admission_attempts_snapshot"]["path"], expected_digest=plan["capacity_admission_attempts_snapshot"]["sha256"])
    try:
        attempts = json.loads(attempts_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"interrupted-before-Prove archived capacity attempts are invalid: {exc}")
    if attempts != failed.get("capacity_admission_attempts"):
        abort("interrupted-before-Prove archived capacity attempts differ from the failed V5 session")
    map_ref = _require_path_ref(
        plan.get("admissions_map_snapshot"), paths["admissions_map"].relative_to(ROOT).as_posix(), "V5 Admission map snapshot",
    )
    map_raw = _snapshot_file(ROOT / map_ref["path"], relative=map_ref["path"], expected_digest=map_ref["sha256"])
    try:
        archived_map = json.loads(map_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"interrupted-before-Prove archived Admission map is invalid: {exc}")
    if archived_map != failed.get("compatible_native_admissions"):
        abort("interrupted-before-Prove archived Admission map differs from the failed V5 session")
    for dataset, item in zip(_COMPAT_DATASETS, plan["admission_file_snapshots"]):
        old_ref = archived_map.get(dataset) if isinstance(archived_map, dict) else None
        if (
            not isinstance(item, dict) or item.get("dataset") != dataset or not isinstance(old_ref, dict)
            or item.get("admission", {}).get("sha256") != old_ref.get("sha256")
            or item.get("settings", {}).get("sha256") != old_ref.get("settings_sha256")
        ):
            abort(f"interrupted-before-Prove archived {dataset} Admission snapshot differs from its V5 map")
    frozen_original, frozen_digest = validate_original_session_snapshot(
        ROOT / plan["original_session_snapshot"]["path"], session_id=BASELINE_SESSION_ID,
        expected_sha256=plan["original_session_snapshot"]["sha256"],
    )
    if frozen_digest != _ORIGINAL_SESSION_SHA256 or frozen_original != original:
        abort("interrupted-before-Prove archive does not preserve the original PPI Setup/Witness session anchor")
    sidecar = plan.get("admission_sidecar")
    if not isinstance(sidecar, dict) or set(sidecar) != {"source_path", "archive_path"}:
        abort("interrupted-before-Prove archived Admission sidecar reference is malformed")
    if sidecar != {
        "source_path": f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/admission",
        "archive_path": paths["admission_sidecar"].relative_to(ROOT).as_posix(),
    }:
        abort("interrupted-before-Prove Admission sidecar path changed")
    admission_inventory = _read_inventory_snapshot(plan.get("admission_sidecar_inventory"), paths["admission_inventory"], "Admission sidecar inventory")
    partial = plan.get("ogbn_partial")
    if not isinstance(partial, dict) or set(partial) != {"source_path", "archive_path"} or partial != {
        "source_path": f"{_COMPAT_OUTPUT_ROOT}/ogbn_arxiv",
        "archive_path": paths["partial_output"].relative_to(ROOT).as_posix(),
    }:
        abort("interrupted-before-Prove OGBN partial-output reference changed")
    partial_inventory = _read_inventory_snapshot(plan.get("ogbn_partial_inventory"), paths["partial_inventory"], "OGBN partial inventory")
    if plan.get("admission_sidecar_inventory") is None or plan.get("ogbn_partial_inventory") is None:
        abort("interrupted-before-Prove recovery is missing a closed filesystem inventory")
    return None


def _validate_uncommitted_recovery_plan(
    plan: dict[str, Any], failed: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
) -> None:
    _validate_recovery_snapshots(plan, failed, request, original)
    paths = _v5_interrupted_archive_paths()
    admission_inventory = _read_inventory_snapshot(plan.get("admission_sidecar_inventory"), paths["admission_inventory"], "Admission sidecar inventory")
    partial_inventory = _read_inventory_snapshot(plan.get("ogbn_partial_inventory"), paths["partial_inventory"], "OGBN partial inventory")
    _reconcile_recovery_move(
        source=ROOT / plan["admission_sidecar"]["source_path"],
        destination=ROOT / plan["admission_sidecar"]["archive_path"], label="Admission sidecar", inventory=admission_inventory,
    )
    report = plan["capacity_report"]
    _reconcile_recovery_move(
        source=ROOT / report["source_path"], destination=ROOT / report["archive_path"],
        label="current capacity report", sha256=report["snapshot"]["sha256"],
    )
    partial = plan["ogbn_partial"]
    _reconcile_recovery_move(
        source=ROOT / partial["source_path"], destination=ROOT / partial["archive_path"],
        label="OGBN partial output", inventory=partial_inventory,
    )


def _validate_committed_recovery_archive(plan: dict[str, Any]) -> None:
    paths = _v5_interrupted_archive_paths()
    admission_inventory = _read_inventory_snapshot(plan.get("admission_sidecar_inventory"), paths["admission_inventory"], "Admission sidecar inventory")
    partial_inventory = _read_inventory_snapshot(plan.get("ogbn_partial_inventory"), paths["partial_inventory"], "OGBN partial inventory")
    for path, label, expected_inventory in (
        (ROOT / plan["admission_sidecar"]["archive_path"], "archived Admission sidecar", admission_inventory),
        (ROOT / plan["ogbn_partial"]["archive_path"], "archived OGBN partial output", partial_inventory),
    ):
        actual = _tree_inventory(path, label)
        if canonical_json_digest(actual) != canonical_json_digest(expected_inventory):
            abort(f"interrupted-before-Prove {label} differs from its committed inventory")
    report = plan.get("capacity_report")
    if not isinstance(report, dict) or set(report) != {"source_path", "archive_path", "snapshot"}:
        abort("interrupted-before-Prove committed capacity report record is malformed")
    archived_report = ROOT / report["archive_path"]
    _regular_file(archived_report, "archived current capacity report")
    if _digest_bytes(archived_report.read_bytes()) != report["snapshot"].get("sha256"):
        abort("interrupted-before-Prove committed capacity report bytes differ from its archive anchor")


def _validate_interrupted_before_prove_recovery_record(
    session_document: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
) -> None:
    plan = session_document.get("compatible_interrupted_before_prove_recovery")
    if not isinstance(plan, dict):
        abort("V6 compatible request is missing its interrupted-before-Prove recovery audit")
    paths = _v5_interrupted_archive_paths()
    failed, _ = _load_retry_snapshot(plan.get("failed_session_snapshot"), paths["failed_session"], "V5 failed-session")
    old_request, _ = _load_retry_snapshot(plan.get("old_request_snapshot"), paths["old_request"], "V5 old-request")
    if (
        canonical_json_digest(old_request) != _V5_REQUEST_CANONICAL_SHA256
        or failed.get("compatible_resume_request") != old_request
        or plan.get("new_request_canonical_sha256") != canonical_json_digest(request)
        or session_document.get("compatible_resume_request") != request
        or session_document.get("compatible_preflight_retry") != failed.get("compatible_preflight_retry")
        or session_document.get("compatible_admission_preflight_retry") != failed.get("compatible_admission_preflight_retry")
        or request.get("old_session_snapshot_path") != _snapshot_path(BASELINE_SESSION_ID)
        or request.get("old_session_snapshot_sha256") != _ORIGINAL_SESSION_SHA256
    ):
        abort("V6 compatible request/recovery audit is not an exact continuation of the pinned V5 session")
    _validate_recovery_snapshots(plan, failed, request, original)
    current_files = _recovery_current_files(plan)
    capacity_report = ROOT / plan["capacity_report"]["snapshot"]["path"]
    _validate_v5_interrupted_state(
        failed, request=request, original=original, current_files=current_files, report_path=capacity_report,
    )
    _validate_committed_recovery_archive(plan)


def _build_interrupted_recovery_plan(
    live_raw: bytes, live: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
) -> dict[str, Any]:
    paths = _v5_interrupted_archive_paths()
    archive = paths["archive"]
    old_request = live["compatible_resume_request"]
    old_v1 = live["compatible_preflight_retry"]
    old_v2 = live["compatible_admission_preflight_retry"]
    map_value = live["compatible_native_admissions"]
    report_rel = live["capacity_admission_attempts"]["report_path"]
    report_source = ROOT / report_rel
    admission_source = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID / "compatibility" / "admission"
    partial_source = ROOT / _COMPAT_OUTPUT_ROOT / "ogbn_arxiv"

    failed_sha = _write_immutable_snapshot(paths["failed_session"], live_raw, "poststop failed-session")
    old_request_sha = _write_immutable_snapshot(paths["old_request"], _json_snapshot_bytes(old_request), "V5 request")
    old_v1_sha = _write_immutable_snapshot(paths["old_v1"], _json_snapshot_bytes(old_v1), "V1 retry record")
    old_v2_sha = _write_immutable_snapshot(paths["old_v2"], _json_snapshot_bytes(old_v2), "V2 retry record")
    old_review = ROOT / _V5_SOURCE_REVIEW_PATH
    _regular_file(old_review, "V5 approved source review")
    old_review_raw = old_review.read_bytes()
    if _digest_bytes(old_review_raw) != _V5_SOURCE_REVIEW_SHA256:
        abort("interrupted-before-Prove V5 source-review bytes differ from the pinned approval")
    old_review_sha = _write_immutable_snapshot(paths["old_review"], old_review_raw, "V5 source review")
    map_sha = _write_immutable_snapshot(paths["admissions_map"], _json_snapshot_bytes(map_value), "V5 Admission map")
    attempts_sha = _write_immutable_snapshot(paths["capacity_attempts"], _json_snapshot_bytes(live["capacity_admission_attempts"]), "V5 capacity attempts")
    report_raw = report_source.read_bytes()
    report_snapshot_sha = _write_immutable_snapshot(paths["capacity_report_snapshot"], report_raw, "V5 current capacity report")
    if report_snapshot_sha != _V5_CAPACITY_REPORT_SHA256:
        abort("interrupted-before-Prove V5 capacity report differs from its pinned digest")

    admission_file_snapshots: list[dict[str, Any]] = []
    for dataset in _COMPAT_DATASETS:
        current_ref = map_value[dataset]
        source_admission = ROOT / current_ref["path"]
        source_settings = source_admission.parent / "settings.json"
        _regular_file(source_admission, f"V5 {dataset} Admission")
        _regular_file(source_settings, f"V5 {dataset} settings")
        admission_raw = source_admission.read_bytes()
        settings_raw = source_settings.read_bytes()
        admission_path = paths["admission_files"] / dataset / "admission.json"
        settings_path = paths["admission_files"] / dataset / "settings.json"
        admission_sha = _write_immutable_snapshot(admission_path, admission_raw, f"V5 {dataset} Admission")
        settings_sha = _write_immutable_snapshot(settings_path, settings_raw, f"V5 {dataset} settings")
        admission_file_snapshots.append({
            "dataset": dataset,
            "admission": {"path": admission_path.relative_to(ROOT).as_posix(), "sha256": admission_sha},
            "settings": {"path": settings_path.relative_to(ROOT).as_posix(), "sha256": settings_sha},
        })
    admission_inventory_value = _tree_inventory(admission_source, "V5 Admission sidecar")
    admission_inventory_sha = _write_immutable_snapshot(
        paths["admission_inventory"], _json_snapshot_bytes(admission_inventory_value), "V5 Admission sidecar inventory",
    )
    partial_inventory_value = _tree_inventory(partial_source, "OGBN partial output")
    partial_inventory_sha = _write_immutable_snapshot(
        paths["partial_inventory"], _json_snapshot_bytes(partial_inventory_value), "OGBN partial-output inventory",
    )
    retained = {
        "setup_reuse_receipts": original.get("setup_reuse_receipts"),
        "stage_receipts": original.get("stage_receipts", {}).get("ppi"),
    }
    original_ref = {"path": _snapshot_path(BASELINE_SESSION_ID), "sha256": _ORIGINAL_SESSION_SHA256}
    plan: dict[str, Any] = {
        "schema": INTERRUPTED_BEFORE_PROVE_RECOVERY_SCHEMA,
        "session_id": BASELINE_SESSION_ID,
        "failed_session_snapshot": {"path": paths["failed_session"].relative_to(ROOT).as_posix(), "sha256": failed_sha},
        "old_request_snapshot": {"path": paths["old_request"].relative_to(ROOT).as_posix(), "sha256": old_request_sha},
        "old_preflight_retry_snapshot": {"path": paths["old_v1"].relative_to(ROOT).as_posix(), "sha256": old_v1_sha},
        "old_admission_preflight_retry_snapshot": {"path": paths["old_v2"].relative_to(ROOT).as_posix(), "sha256": old_v2_sha},
        "old_source_review_snapshot": {"path": paths["old_review"].relative_to(ROOT).as_posix(), "sha256": old_review_sha},
        "original_session_snapshot": original_ref,
        "new_source_review": {"path": request["source_review_path"], "sha256": request["source_review_sha256"]},
        "new_request_canonical_sha256": canonical_json_digest(request),
        "old_request_canonical_sha256": canonical_json_digest(old_request),
        "v1_retry_canonical_sha256": canonical_json_digest(old_v1),
        "v2_retry_canonical_sha256": canonical_json_digest(old_v2),
        "capacity_checks_canonical_sha256": canonical_json_digest(live.get("capacity_checks")),
        "prior_prover_identity_sha256": live["compatible_prover_identity_sha256"],
        "prior_capacity_source_identity_sha256": live["compatible_capacity_source_identity_sha256"],
        "retained_ppi_chain_sha256": canonical_json_digest(retained),
        "admissions_map_snapshot": {"path": paths["admissions_map"].relative_to(ROOT).as_posix(), "sha256": map_sha},
        "admission_file_snapshots": admission_file_snapshots,
        "capacity_admission_attempts_snapshot": {"path": paths["capacity_attempts"].relative_to(ROOT).as_posix(), "sha256": attempts_sha},
        "capacity_report": {
            "source_path": report_rel,
            "archive_path": paths["capacity_report_archived"].relative_to(ROOT).as_posix(),
            "snapshot": {"path": paths["capacity_report_snapshot"].relative_to(ROOT).as_posix(), "sha256": report_snapshot_sha},
        },
        "admission_sidecar": {
            "source_path": admission_source.relative_to(ROOT).as_posix(),
            "archive_path": paths["admission_sidecar"].relative_to(ROOT).as_posix(),
        },
        "admission_sidecar_inventory": {"path": paths["admission_inventory"].relative_to(ROOT).as_posix(), "sha256": admission_inventory_sha},
        "ogbn_partial": {
            "source_path": partial_source.relative_to(ROOT).as_posix(),
            "archive_path": paths["partial_output"].relative_to(ROOT).as_posix(),
        },
        "ogbn_partial_inventory": {"path": paths["partial_inventory"].relative_to(ROOT).as_posix(), "sha256": partial_inventory_sha},
        "no_proof_outputs": [
            f"{_COMPAT_OUTPUT_ROOT}/{dataset}/{name}"
            for dataset in _COMPAT_DATASETS
            for name in ("proof.json", "result.json", "native_verify_metrics.json", "core_run_manifest.json")
        ],
        "no_ogbn_setup_artifacts": [
            f"{_COMPAT_OUTPUT_ROOT}/ogbn_arxiv/{name}"
            for name in ("pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json")
        ],
    }
    _write_immutable_snapshot(paths["plan"], _json_snapshot_bytes(plan), "recovery plan")
    _reconcile_recovery_move(
        source=admission_source, destination=paths["admission_sidecar"], label="Admission sidecar", inventory=admission_inventory_value,
    )
    _reconcile_recovery_move(
        source=report_source, destination=paths["capacity_report_archived"], label="current capacity report", sha256=report_snapshot_sha,
    )
    _reconcile_recovery_move(
        source=partial_source, destination=paths["partial_output"], label="OGBN partial output", inventory=partial_inventory_value,
    )
    return plan


def prepare_interrupted_before_prove_recovery(
    session_file: Path, request: dict[str, Any],
) -> dict[str, Any] | None:
    """Archive and stage only the pinned V5 interrupted-before-Prove to V6 transition."""
    expected_session = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file() or session_file.resolve(strict=True) != expected_session.resolve(strict=True):
        abort("interrupted-before-Prove recovery must use the exact original session record")
    try:
        live_raw = session_file.read_bytes()
        live = json.loads(live_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"cannot read interrupted-before-Prove recovery session: {exc}")
    if not isinstance(live, dict):
        abort("interrupted-before-Prove recovery session is malformed")
    if live.get("compatible_resume_request") == request:
        if live.get("compatible_interrupted_before_prove_recovery") is not None:
            validate_compatible_resume_request(live, expected_session_id=BASELINE_SESSION_ID)
        return None
    validate_interrupted_before_prove_resume_candidate(live, request)
    paths = _v5_interrupted_archive_paths()
    if _path_exists(paths["plan"]):
        _regular_file(paths["plan"], "recovery plan")
        try:
            plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            abort(f"interrupted-before-Prove recovery plan is invalid: {exc}")
        original, _ = validate_original_session_snapshot(
            ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
            expected_sha256=_ORIGINAL_SESSION_SHA256,
        )
        current_files = _recovery_current_files(plan)
        report_path = ROOT / plan["capacity_report"]["snapshot"]["path"]
        _validate_v5_interrupted_state(live, request=request, original=original, current_files=current_files, report_path=report_path)
        _validate_uncommitted_recovery_plan(plan, live, request, original)
        return plan
    original, _ = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    return _build_interrupted_recovery_plan(live_raw, live, request, original)


def apply_interrupted_before_prove_recovery_commit(
    session_document: dict[str, Any], request: dict[str, Any], plan: dict[str, Any],
) -> None:
    """Apply the already-archived V5->V6 transition to the in-lock session copy."""
    paths = _v5_interrupted_archive_paths()
    failed, _ = _load_retry_snapshot(plan.get("failed_session_snapshot"), paths["failed_session"], "V5 failed-session")
    if canonical_json_digest(failed) != canonical_json_digest(session_document):
        abort("interrupted-before-Prove session changed before the atomic V6 commit")
    original, _ = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    current_files = _recovery_current_files(plan)
    report_path = ROOT / plan["capacity_report"]["snapshot"]["path"]
    _validate_v5_interrupted_state(failed, request=request, original=original, current_files=current_files, report_path=report_path)
    _validate_uncommitted_recovery_plan(plan, failed, request, original)
    _validate_recovery_output_absence(allow_archived_ogbn_partial=True)
    if plan.get("new_request_canonical_sha256") != canonical_json_digest(request):
        abort("interrupted-before-Prove recovery plan changed before the atomic V6 commit")
    session_document["compatible_resume_request"] = request
    session_document["compatible_interrupted_before_prove_recovery"] = plan
    session_document["capacity_checks"] = copy.deepcopy(original.get("capacity_checks"))
    for key in (
        "capacity_admission_attempts", "compatible_native_admissions", "compatible_prover_identity",
        "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "dataset_failures", "active_dataset",
    ):
        session_document.pop(key, None)


def _configured_admission_retry_directory(session_id: str) -> Path:
    if session_id != BASELINE_SESSION_ID:
        abort("configured-Admission retry audit is bound to the one original session")
    return ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "configured_admission_retry_v7"


def _configured_admission_retry_paths(session_id: str) -> dict[str, Path]:
    archive = _configured_admission_retry_directory(session_id)
    return {
        "archive": archive,
        "failed_session": archive / "failed_session.json",
        "old_request": archive / "old_request.json",
        "new_request": archive / "new_request.json",
        "old_recovery": archive / "old_interrupted_recovery.json",
        "old_review": archive / "old_source_review_v6.json",
        "old_admissions_map": archive / "compatible_native_admissions.json",
        "capacity_attempts": archive / "capacity_admission_attempts.json",
        "capacity_report": archive / "capacity_precheck_current.json",
        "admission_files": archive / "admissions",
        "record": archive / "retry_record.json",
    }


def _validate_approved_historical_review_bytes(raw: bytes, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if _digest_bytes(raw) != expected_sha256:
        abort(f"{label} historical source-review bytes differ from their fixed digest")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"{label} historical source review is invalid JSON: {exc}")
    if (
        not isinstance(value, dict)
        or set(value) != {
            "schema", "session_id", "baseline_manifest_sha256", "decision", "reviewer",
            "reviewed_changed_files", "reviewed_review_only_files", "reviewed_snapshot_indexes",
        }
        or value.get("schema") != SOURCE_REVIEW_SCHEMA
        or value.get("session_id") != BASELINE_SESSION_ID
        or value.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or value.get("decision") != "APPROVED"
        or value.get("reviewer") != "root"
    ):
        abort(f"{label} historical source review is not the exact approved schema")
    return value


def _validate_v6_archived_context(
    failed: dict[str, Any], old_request: dict[str, Any], original: dict[str, Any],
    *, old_review_raw: bytes, old_recovery: dict[str, Any],
) -> None:
    """Authenticate the fixed V6 context without comparing its review to the V7 source tree."""
    if (
        canonical_json_digest(old_request) != _V6_CONFIGURED_ADMISSION_REQUEST_SHA256
        or failed.get("compatible_resume_request") != old_request
        or old_request.get("source_review_path") != _V6_CONFIGURED_ADMISSION_REVIEW_PATH
        or old_request.get("source_review_sha256") != _V6_CONFIGURED_ADMISSION_REVIEW_SHA256
        or failed.get("compatible_interrupted_before_prove_recovery") != old_recovery
        or canonical_json_digest(old_recovery) != canonical_json_digest(failed.get("compatible_interrupted_before_prove_recovery"))
    ):
        abort("configured-Admission retry V6 request/recovery context differs from its pinned predecessor")
    _validate_approved_historical_review_bytes(
        old_review_raw, expected_sha256=_V6_CONFIGURED_ADMISSION_REVIEW_SHA256, label="V6",
    )
    if (
        canonical_json_digest(failed.get("compatible_preflight_retry")) != _V5_V1_RETRY_CANONICAL_SHA256
        or canonical_json_digest(failed.get("compatible_admission_preflight_retry")) != _V5_V2_RETRY_CANONICAL_SHA256
    ):
        abort("configured-Admission retry V6 session does not preserve the pinned V1/V2 audit chain")
    frozen_original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if original_sha != _ORIGINAL_SESSION_SHA256 or frozen_original != original:
        abort("configured-Admission retry changed the original PPI Setup/Witness anchor")
    _validate_interrupted_before_prove_recovery_record(failed, old_request, original)


def _validate_configured_admission_failure_document(
    document: dict[str, Any], original: dict[str, Any], *,
    current_files: dict[str, tuple[Path, Path]] | None = None,
    report_path: Path | None = None,
    check_outputs: bool,
) -> None:
    expected_compatible = {
        "compatible_resume_request", "compatible_preflight_retry", "compatible_admission_preflight_retry",
        "compatible_interrupted_before_prove_recovery", "compatible_prover_identity",
        "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "compatible_native_admissions",
    }
    if (
        document.get("schema_version") != 1 or document.get("baseline") != "ezkl"
        or document.get("session_id") != BASELINE_SESSION_ID or document.get("status") != "FAILED"
        or document.get("stage") != "ppi_configured_admission" or document.get("active_dataset") != "ppi"
        or document.get("output_root") != _COMPAT_OUTPUT_ROOT or document.get("matrix") != _COMPAT_MATRIX
        or document.get("selected_datasets") != _COMPAT_DATASETS
        or document.get("completed_cells") not in (None, {})
        or {key for key in document if key.startswith("compatible_")} != expected_compatible
        or document.get("native_admissions") != original.get("native_admissions")
        or document.get("setup_reuse_receipts") != original.get("setup_reuse_receipts")
        or document.get("stage_receipts") != original.get("stage_receipts")
        or document.get("compatible_stage_receipts") not in (None, {})
        or document.get("results") not in (None, {})
        or document.get("verified_results") not in (None, {})
        or canonical_json_digest(document.get("compatible_resume_request")) != _V6_CONFIGURED_ADMISSION_REQUEST_SHA256
    ):
        abort("configured-Admission retry is not the exact pinned V6 failure before proof progress")
    failure_map = document.get("dataset_failures")
    failure = failure_map.get("ppi") if isinstance(failure_map, dict) else None
    if (
        not isinstance(failure_map, dict) or set(failure_map) != {"ppi"}
        or not isinstance(failure, dict) or failure.get("stage") != "ppi_configured_admission"
        or failure.get("exception_type") != "ContractError"
        or failure.get("error") != "ppi compatible Admission semantic projection differs outside the approved free-memory sample: path, prepared_dir, sha256"
    ):
        abort("configured-Admission retry failure is not the exact pinned PPI Admission projection error")
    if (
        canonical_json_digest(document.get("compatible_preflight_retry")) != _V5_V1_RETRY_CANONICAL_SHA256
        or canonical_json_digest(document.get("compatible_admission_preflight_retry")) != _V5_V2_RETRY_CANONICAL_SHA256
        or not isinstance(document.get("compatible_interrupted_before_prove_recovery"), dict)
    ):
        abort("configured-Admission retry V1/V2/V6 historical audit chain is missing or changed")
    if set(document.get("setup_reuse_receipts", {})) != {"ppi"} or set(document.get("stage_receipts", {})) != {"ppi"}:
        abort("configured-Admission retry must retain only the authenticated original PPI Setup/Witness stage")

    checks = document.get("capacity_checks")
    original_checks = original.get("capacity_checks")
    if (
        not isinstance(checks, list) or len(checks) != 6 or not isinstance(original_checks, list)
        or len(original_checks) != 4
        or canonical_json_digest(checks[:4]) != canonical_json_digest(original_checks)
        or [(item.get("dataset"), item.get("phase")) for item in checks[4:]]
        != [("ogbn_arxiv", "setup"), ("ppi", "witness_prove_verify")]
    ):
        abort("configured-Admission retry capacity history is not the exact pinned six-check V6 attempt")

    identity = document.get("compatible_prover_identity")
    capacity_identity = document.get("compatible_capacity_source_identity")
    if (
        not isinstance(identity, dict) or identity.get("schema") != "ezkl-formal-resume-identity-v1"
        or document.get("compatible_prover_identity_sha256") != canonical_json_digest(identity)
        or not isinstance(capacity_identity, dict)
        or document.get("compatible_capacity_source_identity_sha256") != canonical_json_digest(capacity_identity)
        or capacity_identity.get("source_tree_sha256") != identity.get("source_tree_sha256")
        or canonical_json_digest(document.get("resume_identity")) != canonical_json_digest(original.get("resume_identity"))
    ):
        abort("configured-Admission retry V6 current prover/capacity identity pair is not session-bound")

    attempts = document.get("capacity_admission_attempts")
    expected_report = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/capacity_precheck_current.json"
    expected_attempt_keys = {
        "datasets", "readiness", "gate_status", "report_path", "report_sha256", "purpose", "failures", "audit_failures",
    }
    if (
        not isinstance(attempts, dict) or set(attempts) != expected_attempt_keys
        or attempts.get("datasets") != _COMPAT_DATASETS
        or attempts.get("readiness") != "NOT_ESTABLISHED"
        or attempts.get("gate_status") != "current_v4_pk_payload_lower_bound_unavailable"
        or attempts.get("report_path") != expected_report
        or attempts.get("purpose") != "selected-native-admission-and-v4-PK-lower-bound-report-only; no setup or proof success implied"
        or attempts.get("failures") != {} or attempts.get("audit_failures") != {}
    ):
        abort("configured-Admission retry capacity-attempt metadata is not the exact archived V6 report-only gate")
    _require_sha(attempts.get("report_sha256"), "V6 current capacity report digest")
    if report_path is None:
        report_path = ROOT / expected_report
    _regular_file(report_path, "configured-Admission V6 capacity report")
    if _digest_bytes(report_path.read_bytes()) != attempts["report_sha256"]:
        abort("configured-Admission retry capacity report differs from its V6 session anchor")

    projection = copy.deepcopy(document)
    projection.pop("compatible_interrupted_before_prove_recovery", None)
    projection["status"] = "FAILED"
    projection["stage"] = "selected_dataset_native_capacity_preflight"
    projection["capacity_checks"] = copy.deepcopy(original_checks)
    _validate_admission_preflight_state(
        projection, original, document["compatible_resume_request"], check_outputs=False, current_files=current_files,
    )
    if check_outputs:
        _validate_no_proof_outputs()
        output = ROOT / _COMPAT_OUTPUT_ROOT / "ogbn_arxiv"
        if output.is_symlink():
            abort("configured-Admission retry refuses a symlinked OGBN output directory")
        for name in ("pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json"):
            if (output / name).exists() or (output / name).is_symlink():
                abort(f"configured-Admission retry refuses unverified OGBN setup output: {name}")
        compatibility = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID / "compatibility"
        for name in ("capacity_precheck_ppi.json", "capacity_rechecks.json", "compatible_resume_receipt.json", "compatible_stage_receipts.json"):
            path = compatibility / name
            if path.exists() or path.is_symlink():
                abort(f"configured-Admission retry refuses downstream stage output: {name}")


def validate_failed_configured_admission_resume_gate(session_document: dict[str, Any]) -> None:
    """Admit only the single pinned V6 PPI configured-Admission failure for V7 migration."""
    session_file = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file():
        abort("configured-Admission retry exact session file is missing or unsafe")
    raw = session_file.read_bytes()
    if _digest_bytes(raw) != _V6_CONFIGURED_ADMISSION_SESSION_SHA256:
        abort("configured-Admission retry failed-session bytes differ from the pinned V6 state")
    try:
        live = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"configured-Admission retry failed-session JSON is invalid: {exc}")
    if not isinstance(live, dict) or canonical_json_digest(live) != canonical_json_digest(session_document):
        abort("configured-Admission retry in-memory session differs from the pinned V6 bytes")
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if original_sha != _ORIGINAL_SESSION_SHA256:
        abort("configured-Admission retry original session snapshot changed")
    _validate_configured_admission_failure_document(live, original, check_outputs=True)
    request = live["compatible_resume_request"]
    recovery = live["compatible_interrupted_before_prove_recovery"]
    review_path = ROOT / request["source_review_path"]
    _regular_file(review_path, "pinned V6 source review")
    review_raw = review_path.read_bytes()
    if _digest_bytes(review_raw) != _V6_CONFIGURED_ADMISSION_REVIEW_SHA256:
        abort("configured-Admission retry V6 source-review bytes differ from the historical pin")
    _validate_approved_historical_review_bytes(
        review_raw, expected_sha256=_V6_CONFIGURED_ADMISSION_REVIEW_SHA256, label="V6",
    )
    _validate_v6_archived_context(live, request, original, old_review_raw=review_raw, old_recovery=recovery)


def _configured_admission_retry_snapshot_files(
    record: dict[str, Any], failed: dict[str, Any], original: dict[str, Any], old_request: dict[str, Any],
) -> tuple[dict[str, tuple[Path, Path]], Path]:
    paths = _configured_admission_retry_paths(BASELINE_SESSION_ID)
    map_ref = _require_path_ref(
        record.get("admissions_map_snapshot"), paths["old_admissions_map"].relative_to(ROOT).as_posix(),
        "V6 Admission map snapshot",
    )
    map_raw = _snapshot_file(ROOT / map_ref["path"], relative=map_ref["path"], expected_digest=map_ref["sha256"])
    try:
        map_value = json.loads(map_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"configured-Admission retry archived Admission map is invalid: {exc}")
    if map_value != failed.get("compatible_native_admissions"):
        abort("configured-Admission retry archived Admission map differs from the V6 failed session")
    entries = record.get("admission_file_snapshots")
    if not isinstance(entries, list) or [item.get("dataset") if isinstance(item, dict) else None for item in entries] != _COMPAT_DATASETS:
        abort("configured-Admission retry Admission-file snapshot inventory is not the exact dataset pair")
    result: dict[str, tuple[Path, Path]] = {}
    for dataset, entry in zip(_COMPAT_DATASETS, entries):
        if not isinstance(entry, dict) or set(entry) != {
            "dataset", "source_admission", "source_settings", "snapshot_admission", "snapshot_settings",
        }:
            abort(f"configured-Admission retry {dataset} Admission snapshot entry is malformed")
        source_ref = map_value.get(dataset) if isinstance(map_value, dict) else None
        if not isinstance(source_ref, dict):
            abort(f"configured-Admission retry {dataset} old map entry is missing")
        source_admission_path = source_ref.get("path")
        source_settings_path = f"{Path(str(source_admission_path)).parent.as_posix()}/settings.json"
        source_admission = _require_path_ref(entry.get("source_admission"), source_admission_path, f"{dataset} old Admission")
        source_settings = _require_path_ref(entry.get("source_settings"), source_settings_path, f"{dataset} old settings")
        if source_admission["sha256"] != source_ref.get("sha256") or source_settings["sha256"] != source_ref.get("settings_sha256"):
            abort(f"configured-Admission retry {dataset} Admission/settings source refs differ from the session map")
        admission_snapshot_path = paths["admission_files"] / dataset / "admission.json"
        settings_snapshot_path = paths["admission_files"] / dataset / "settings.json"
        admission_ref = _require_path_ref(
            entry.get("snapshot_admission"), admission_snapshot_path.relative_to(ROOT).as_posix(), f"{dataset} archived Admission",
        )
        settings_ref = _require_path_ref(
            entry.get("snapshot_settings"), settings_snapshot_path.relative_to(ROOT).as_posix(), f"{dataset} archived settings",
        )
        for ref, source_sha, label in (
            (admission_ref, source_ref.get("sha256"), f"{dataset} Admission"),
            (settings_ref, source_ref.get("settings_sha256"), f"{dataset} settings"),
        ):
            _snapshot_file(ROOT / ref["path"], relative=ref["path"], expected_digest=ref["sha256"])
            if ref["sha256"] != source_sha:
                abort(f"configured-Admission retry archived {label} digest differs from the V6 source map")
        result[dataset] = (ROOT / admission_ref["path"], ROOT / settings_ref["path"])

    attempts_path = paths["capacity_attempts"]
    attempts_ref = _require_path_ref(
        record.get("capacity_attempts_snapshot"), attempts_path.relative_to(ROOT).as_posix(), "V6 capacity-attempt snapshot",
    )
    attempts_raw = _snapshot_file(ROOT / attempts_ref["path"], relative=attempts_ref["path"], expected_digest=attempts_ref["sha256"])
    try:
        attempts = json.loads(attempts_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"configured-Admission retry archived capacity attempts are invalid: {exc}")
    if attempts != failed.get("capacity_admission_attempts"):
        abort("configured-Admission retry archived capacity attempts differ from the V6 failed session")

    expected_report_path = attempts.get("report_path") if isinstance(attempts, dict) else None
    report_record = record.get("capacity_report_snapshot")
    if not isinstance(report_record, dict) or set(report_record) != {"source", "snapshot"}:
        abort("configured-Admission retry capacity-report archive binding is malformed")
    source_report = _require_path_ref(
        report_record.get("source"), expected_report_path, "V6 source capacity report",
    )
    expected_snapshot_path = paths["capacity_report"]
    snapshot_ref = _require_path_ref(
        report_record.get("snapshot"), expected_snapshot_path.relative_to(ROOT).as_posix(), "V6 archived capacity report",
    )
    report_raw = _snapshot_file(ROOT / snapshot_ref["path"], relative=snapshot_ref["path"], expected_digest=snapshot_ref["sha256"])
    if source_report["sha256"] != attempts.get("report_sha256") or snapshot_ref["sha256"] != source_report["sha256"]:
        abort("configured-Admission retry capacity report differs from its authenticated V6 report reference")
    try:
        report = json.loads(report_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"configured-Admission retry archived capacity report is invalid JSON: {exc}")
    if not isinstance(report, dict):
        abort("configured-Admission retry archived capacity report is not an object")
    return result, ROOT / snapshot_ref["path"]


def _validate_configured_admission_retry_record(
    session_document: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = session_document.get("compatible_configured_admission_retry")
    expected_keys = {
        "schema", "session_id", "failed_session_snapshot", "old_request_snapshot", "old_request_canonical_sha256",
        "new_request_snapshot", "new_request_canonical_sha256", "old_interrupted_recovery_snapshot",
        "old_interrupted_recovery_canonical_sha256", "old_source_review", "old_source_review_snapshot",
        "new_source_review", "old_v1_retry_canonical_sha256", "old_v2_retry_canonical_sha256",
        "original_session_snapshot", "capacity_checks_canonical_sha256", "prior_prover_identity_sha256",
        "prior_capacity_source_identity_sha256", "admissions_map_snapshot", "admission_file_snapshots",
        "capacity_attempts_snapshot", "capacity_report_snapshot",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        abort("configured-Admission retry audit has unknown, missing, or malformed fields")
    paths = _configured_admission_retry_paths(BASELINE_SESSION_ID)
    record_snapshot = paths["record"]
    _regular_file(record_snapshot, "V7 retry-record snapshot")
    if record_snapshot.read_bytes() != _json_snapshot_bytes(record):
        abort("configured-Admission retry audit differs from its immutable V7 record snapshot")
    if (
        record.get("schema") != CONFIGURED_ADMISSION_RETRY_SCHEMA
        or record.get("session_id") != BASELINE_SESSION_ID
        or record.get("new_request_canonical_sha256") != canonical_json_digest(request)
        or session_document.get("compatible_resume_request") != request
        or session_document.get("compatible_preflight_retry") is None
        or session_document.get("compatible_admission_preflight_retry") is None
        or session_document.get("compatible_interrupted_before_prove_recovery") is None
    ):
        abort("configured-Admission retry audit is not bound to the exact current V7 request/session")

    failed, failed_raw = _load_retry_snapshot(record.get("failed_session_snapshot"), paths["failed_session"], "V6 failed-session")
    old_request, _ = _load_retry_snapshot(record.get("old_request_snapshot"), paths["old_request"], "V6 old request")
    new_request_snapshot, _ = _load_retry_snapshot(record.get("new_request_snapshot"), paths["new_request"], "V7 new request")
    old_recovery, _ = _load_retry_snapshot(
        record.get("old_interrupted_recovery_snapshot"), paths["old_recovery"], "V6 interrupted-recovery record",
    )
    if (
        _digest_bytes(failed_raw) != _V6_CONFIGURED_ADMISSION_SESSION_SHA256
        or canonical_json_digest(old_request) != _V6_CONFIGURED_ADMISSION_REQUEST_SHA256
        or canonical_json_digest(old_request) != record.get("old_request_canonical_sha256")
        or new_request_snapshot != request
        or canonical_json_digest(new_request_snapshot) != record.get("new_request_canonical_sha256")
        or record.get("old_interrupted_recovery_canonical_sha256") != canonical_json_digest(old_recovery)
        or record.get("old_v1_retry_canonical_sha256") != _V5_V1_RETRY_CANONICAL_SHA256
        or record.get("old_v2_retry_canonical_sha256") != _V5_V2_RETRY_CANONICAL_SHA256
        or record.get("capacity_checks_canonical_sha256") != canonical_json_digest(failed.get("capacity_checks"))
        or record.get("prior_prover_identity_sha256") != failed.get("compatible_prover_identity_sha256")
        or record.get("prior_capacity_source_identity_sha256") != failed.get("compatible_capacity_source_identity_sha256")
    ):
        abort("configured-Admission retry archived V6 request/session/identity chain changed")
    current_setup_receipts = session_document.get("setup_reuse_receipts")
    current_stage_receipts = session_document.get("stage_receipts")
    original_setup_receipts = original.get("setup_reuse_receipts")
    original_stage_receipts = original.get("stage_receipts")
    if (
        canonical_json_digest(session_document.get("compatible_preflight_retry"))
        != canonical_json_digest(failed.get("compatible_preflight_retry"))
        or canonical_json_digest(session_document.get("compatible_admission_preflight_retry"))
        != canonical_json_digest(failed.get("compatible_admission_preflight_retry"))
        or canonical_json_digest(session_document.get("compatible_interrupted_before_prove_recovery"))
        != canonical_json_digest(old_recovery)
        or canonical_json_digest(session_document.get("resume_identity"))
        != canonical_json_digest(original.get("resume_identity"))
        or session_document.get("native_admissions") != original.get("native_admissions")
        or not isinstance(current_setup_receipts, dict) or not isinstance(original_setup_receipts, dict)
        or current_setup_receipts.get("ppi") != original_setup_receipts.get("ppi")
        or not isinstance(current_stage_receipts, dict) or not isinstance(original_stage_receipts, dict)
        or current_stage_receipts.get("ppi") != original_stage_receipts.get("ppi")
    ):
        abort("configured-Admission retry changed the preserved V1/V2/V6 or original PPI identity/receipt chain")

    original_ref = _require_path_ref(
        record.get("original_session_snapshot"), _snapshot_path(BASELINE_SESSION_ID), "original PPI session snapshot",
    )
    if original_ref["sha256"] != _ORIGINAL_SESSION_SHA256 or original != validate_original_session_snapshot(
        ROOT / original_ref["path"], session_id=BASELINE_SESSION_ID, expected_sha256=original_ref["sha256"],
    )[0]:
        abort("configured-Admission retry changed the original PPI Setup/Witness anchor")

    old_review_ref = record.get("old_source_review")
    old_review_snapshot_ref = _require_path_ref(
        record.get("old_source_review_snapshot"), paths["old_review"].relative_to(ROOT).as_posix(), "V6 historical source review",
    )
    if not isinstance(old_review_ref, dict) or set(old_review_ref) != {"path", "sha256"}:
        abort("configured-Admission retry V6 review binding is malformed")
    if (
        old_review_ref.get("path") != _V6_CONFIGURED_ADMISSION_REVIEW_PATH
        or old_review_ref.get("sha256") != _V6_CONFIGURED_ADMISSION_REVIEW_SHA256
        or old_review_ref.get("path") != old_request.get("source_review_path")
        or old_review_ref.get("sha256") != old_request.get("source_review_sha256")
    ):
        abort("configured-Admission retry changed the pinned V6 source-review reference")
    old_review_raw = _snapshot_file(
        ROOT / old_review_snapshot_ref["path"], relative=old_review_snapshot_ref["path"],
        expected_digest=old_review_snapshot_ref["sha256"],
    )
    _validate_approved_historical_review_bytes(
        old_review_raw, expected_sha256=_V6_CONFIGURED_ADMISSION_REVIEW_SHA256, label="V6",
    )
    _validate_v6_archived_context(
        failed, old_request, original, old_review_raw=old_review_raw, old_recovery=old_recovery,
    )
    new_review = record.get("new_source_review")
    if (
        not isinstance(new_review, dict) or set(new_review) != {"path", "sha256"}
        or new_review != {"path": request.get("source_review_path"), "sha256": request.get("source_review_sha256")}
    ):
        abort("configured-Admission retry V7 source review differs from the current request")

    snapshot_files, report_snapshot_path = _configured_admission_retry_snapshot_files(
        record, failed, original, old_request,
    )
    _validate_configured_admission_failure_document(
        failed, original, current_files=snapshot_files, report_path=report_snapshot_path, check_outputs=False,
    )
    return record, failed


def prepare_compatible_configured_admission_retry(
    session_file: Path, request: dict[str, Any],
) -> dict[str, Any] | None:
    """Freeze and audit only the pinned V6 PPI configured-Admission failure before V7."""
    expected_session = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if (
        session_file.is_symlink() or not session_file.is_file()
        or session_file.resolve(strict=True) != expected_session.resolve(strict=True)
    ):
        abort("configured-Admission retry must use the exact original session record")
    live_raw = session_file.read_bytes()
    if _digest_bytes(live_raw) != _V6_CONFIGURED_ADMISSION_SESSION_SHA256:
        try:
            live = json.loads(live_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            abort(f"configured-Admission retry session JSON is invalid: {exc}")
        if isinstance(live, dict) and live.get("compatible_configured_admission_retry") is not None:
            if live.get("compatible_resume_request") == request:
                validate_compatible_resume_request(live, expected_session_id=BASELINE_SESSION_ID)
                return None
        abort("configured-Admission retry failed-session bytes differ from the one pinned V6 candidate")
    try:
        live = json.loads(live_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"configured-Admission retry session JSON is invalid: {exc}")
    if not isinstance(live, dict):
        abort("configured-Admission retry session is malformed")
    if live.get("compatible_resume_request") == request:
        return None
    if live.get("compatible_configured_admission_retry") is not None:
        abort("configured-Admission retry source migration is one-time and already recorded")
    validate_failed_configured_admission_resume_gate(live)
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if (
        request.get("schema") != REQUEST_SCHEMA or request.get("session_id") != BASELINE_SESSION_ID
        or request.get("output_root") != _COMPAT_OUTPUT_ROOT or request.get("matrix") != _COMPAT_MATRIX
        or request.get("selected_datasets") != _COMPAT_DATASETS or request.get("retained_setup_datasets") != ["ppi"]
        or request.get("old_session_snapshot_path") != _snapshot_path(BASELINE_SESSION_ID)
        or request.get("old_session_snapshot_sha256") != original_sha
        or canonical_json_digest(request) == _V6_CONFIGURED_ADMISSION_REQUEST_SHA256
    ):
        abort("configured-Admission retry V7 request is not a distinct exact request for the pinned session")
    _validate_v6_request_review(request)

    paths = _configured_admission_retry_paths(BASELINE_SESSION_ID)
    # Pin the one chosen V7 request before writing the remaining snapshots. A
    # crash can repeat this exact request, but cannot silently switch reviews.
    new_request_sha = _write_immutable_snapshot(paths["new_request"], _json_snapshot_bytes(request), "V7 request")
    if canonical_json_digest(request) != canonical_json_digest(json.loads((ROOT / paths["new_request"].relative_to(ROOT)).read_text(encoding="utf-8"))):
        abort("configured-Admission retry V7 request snapshot changed during preparation")

    failed_sha = _write_immutable_snapshot(paths["failed_session"], live_raw, "V6 failed-session")
    old_request = live["compatible_resume_request"]
    old_request_sha = _write_immutable_snapshot(paths["old_request"], _json_snapshot_bytes(old_request), "V6 request")
    old_recovery = live["compatible_interrupted_before_prove_recovery"]
    old_recovery_sha = _write_immutable_snapshot(paths["old_recovery"], _json_snapshot_bytes(old_recovery), "V6 recovery record")
    old_review_path = ROOT / _V6_CONFIGURED_ADMISSION_REVIEW_PATH
    _regular_file(old_review_path, "V6 historical source review")
    old_review_raw = old_review_path.read_bytes()
    _validate_approved_historical_review_bytes(
        old_review_raw, expected_sha256=_V6_CONFIGURED_ADMISSION_REVIEW_SHA256, label="V6",
    )
    old_review_sha = _write_immutable_snapshot(paths["old_review"], old_review_raw, "V6 historical source review")

    map_value = live.get("compatible_native_admissions")
    if not isinstance(map_value, dict) or set(map_value) != set(_COMPAT_DATASETS):
        abort("configured-Admission retry V6 map must contain exactly the selected PPI and OGBN entries")
    map_sha = _write_immutable_snapshot(paths["old_admissions_map"], _json_snapshot_bytes(map_value), "V6 Admission map")
    attempts = live.get("capacity_admission_attempts")
    if not isinstance(attempts, dict):
        abort("configured-Admission retry V6 capacity-attempt record is missing")
    attempts_sha = _write_immutable_snapshot(paths["capacity_attempts"], _json_snapshot_bytes(attempts), "V6 capacity attempts")
    report_path = ROOT / attempts["report_path"]
    _regular_file(report_path, "V6 capacity report")
    report_raw = report_path.read_bytes()
    report_sha = _digest_bytes(report_raw)
    if report_sha != attempts.get("report_sha256"):
        abort("configured-Admission retry V6 capacity report bytes differ from the active report anchor")
    report_snapshot_sha = _write_immutable_snapshot(paths["capacity_report"], report_raw, "V6 capacity report")

    admission_entries: list[dict[str, Any]] = []
    for dataset in _COMPAT_DATASETS:
        current_ref = map_value[dataset]
        admission_path = ROOT / current_ref["path"]
        settings_path = admission_path.parent / "settings.json"
        _regular_file(admission_path, f"V6 {dataset} Admission")
        _regular_file(settings_path, f"V6 {dataset} settings")
        admission_raw = admission_path.read_bytes()
        settings_raw = settings_path.read_bytes()
        if _digest_bytes(admission_raw) != current_ref.get("sha256") or _digest_bytes(settings_raw) != current_ref.get("settings_sha256"):
            abort(f"configured-Admission retry V6 {dataset} sidecar bytes differ from the session map")
        admission_snapshot_path = paths["admission_files"] / dataset / "admission.json"
        settings_snapshot_path = paths["admission_files"] / dataset / "settings.json"
        admission_snapshot_sha = _write_immutable_snapshot(admission_snapshot_path, admission_raw, f"V6 {dataset} Admission")
        settings_snapshot_sha = _write_immutable_snapshot(settings_snapshot_path, settings_raw, f"V6 {dataset} settings")
        admission_entries.append({
            "dataset": dataset,
            "source_admission": {"path": current_ref["path"], "sha256": current_ref["sha256"]},
            "source_settings": {"path": settings_path.relative_to(ROOT).as_posix(), "sha256": current_ref["settings_sha256"]},
            "snapshot_admission": {"path": admission_snapshot_path.relative_to(ROOT).as_posix(), "sha256": admission_snapshot_sha},
            "snapshot_settings": {"path": settings_snapshot_path.relative_to(ROOT).as_posix(), "sha256": settings_snapshot_sha},
        })
    record: dict[str, Any] = {
        "schema": CONFIGURED_ADMISSION_RETRY_SCHEMA,
        "session_id": BASELINE_SESSION_ID,
        "failed_session_snapshot": {"path": paths["failed_session"].relative_to(ROOT).as_posix(), "sha256": failed_sha},
        "old_request_snapshot": {"path": paths["old_request"].relative_to(ROOT).as_posix(), "sha256": old_request_sha},
        "old_request_canonical_sha256": canonical_json_digest(old_request),
        "new_request_snapshot": {"path": paths["new_request"].relative_to(ROOT).as_posix(), "sha256": new_request_sha},
        "new_request_canonical_sha256": canonical_json_digest(request),
        "old_interrupted_recovery_snapshot": {"path": paths["old_recovery"].relative_to(ROOT).as_posix(), "sha256": old_recovery_sha},
        "old_interrupted_recovery_canonical_sha256": canonical_json_digest(old_recovery),
        "old_source_review": {"path": _V6_CONFIGURED_ADMISSION_REVIEW_PATH, "sha256": _V6_CONFIGURED_ADMISSION_REVIEW_SHA256},
        "old_source_review_snapshot": {"path": paths["old_review"].relative_to(ROOT).as_posix(), "sha256": old_review_sha},
        "new_source_review": {"path": request["source_review_path"], "sha256": request["source_review_sha256"]},
        "old_v1_retry_canonical_sha256": canonical_json_digest(live["compatible_preflight_retry"]),
        "old_v2_retry_canonical_sha256": canonical_json_digest(live["compatible_admission_preflight_retry"]),
        "original_session_snapshot": {"path": _snapshot_path(BASELINE_SESSION_ID), "sha256": original_sha},
        "capacity_checks_canonical_sha256": canonical_json_digest(live["capacity_checks"]),
        "prior_prover_identity_sha256": live["compatible_prover_identity_sha256"],
        "prior_capacity_source_identity_sha256": live["compatible_capacity_source_identity_sha256"],
        "admissions_map_snapshot": {"path": paths["old_admissions_map"].relative_to(ROOT).as_posix(), "sha256": map_sha},
        "admission_file_snapshots": admission_entries,
        "capacity_attempts_snapshot": {"path": paths["capacity_attempts"].relative_to(ROOT).as_posix(), "sha256": attempts_sha},
        "capacity_report_snapshot": {
            "source": {"path": attempts["report_path"], "sha256": report_sha},
            "snapshot": {"path": paths["capacity_report"].relative_to(ROOT).as_posix(), "sha256": report_snapshot_sha},
        },
    }
    record_sha = _write_immutable_snapshot(paths["record"], _json_snapshot_bytes(record), "V7 retry record")
    if record_sha != _digest_bytes(_json_snapshot_bytes(record)):
        abort("configured-Admission retry V7 audit snapshot digest changed")
    candidate = copy.deepcopy(live)
    apply_compatible_configured_admission_retry_commit(candidate, request, record)
    validate_compatible_resume_request(candidate, expected_session_id=BASELINE_SESSION_ID)
    return record


def apply_compatible_configured_admission_retry_commit(
    session_document: dict[str, Any], request: dict[str, Any], record: dict[str, Any],
) -> None:
    """Atomically rebind only the pinned PPI configured-Admission failure to V7."""
    paths = _configured_admission_retry_paths(BASELINE_SESSION_ID)
    failed, failed_raw = _load_retry_snapshot(record.get("failed_session_snapshot"), paths["failed_session"], "V6 failed-session")
    if _digest_bytes(failed_raw) != _V6_CONFIGURED_ADMISSION_SESSION_SHA256:
        abort("configured-Admission retry V6 failed-session snapshot changed before commit")
    if session_document.get("compatible_configured_admission_retry") == record and session_document.get("compatible_resume_request") == request:
        _validate_configured_admission_retry_record(session_document, request, validate_original_session_snapshot(
            ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
            expected_sha256=_ORIGINAL_SESSION_SHA256,
        )[0])
        return
    if canonical_json_digest(session_document) != canonical_json_digest(failed):
        abort("configured-Admission retry session changed before the atomic V7 commit")
    validate_failed_configured_admission_resume_gate(session_document)
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if request.get("old_session_snapshot_sha256") != original_sha:
        abort("configured-Admission retry V7 request changed the original session anchor before commit")
    candidate = copy.deepcopy(session_document)
    candidate["compatible_resume_request"] = request
    candidate["compatible_configured_admission_retry"] = record
    _validate_configured_admission_retry_record(candidate, request, original)
    _validate_configured_admission_outputs()
    session_document["compatible_resume_request"] = request
    session_document["compatible_configured_admission_retry"] = record
    session_document["capacity_checks"] = copy.deepcopy(original.get("capacity_checks"))
    for name in (
        "capacity_admission_attempts", "compatible_native_admissions", "compatible_prover_identity",
        "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "active_dataset",
    ):
        session_document.pop(name, None)
    failures = session_document.get("dataset_failures")
    if not isinstance(failures, dict) or set(failures) != {"ppi"}:
        abort("configured-Admission retry active PPI failure marker changed before commit")
    session_document.pop("dataset_failures", None)


def _validate_configured_admission_outputs() -> None:
    _validate_no_proof_outputs()
    output = ROOT / _COMPAT_OUTPUT_ROOT / "ogbn_arxiv"
    if output.is_symlink():
        abort("configured-Admission retry refuses a symlinked OGBN output directory")
    for name in ("pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json"):
        if (output / name).exists() or (output / name).is_symlink():
            abort(f"configured-Admission retry refuses unverified OGBN setup output: {name}")
    compatibility = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / BASELINE_SESSION_ID / "compatibility"
    for name in ("capacity_precheck_ppi.json", "capacity_rechecks.json", "compatible_resume_receipt.json", "compatible_stage_receipts.json"):
        path = compatibility / name
        if path.exists() or path.is_symlink():
            abort(f"configured-Admission retry refuses downstream stage output: {name}")


def _capacity_admission_retry_paths(session_id: str) -> dict[str, Path]:
    if session_id != BASELINE_SESSION_ID:
        abort("capacity-Admission retry audit is bound to the one original session")
    archive = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "capacity_admission_retry_v8"
    return {
        "archive": archive,
        "failed_session": archive / "failed_session_v7.json",
        "old_request": archive / "old_request_v7.json",
        "old_retry_record": archive / "configured_admission_retry_v7.json",
        "old_review": archive / "source_review_v7b.json",
        "new_request": archive / "new_request_v8.json",
        "new_review": archive / "source_review_v8.json",
        "admissions_map": archive / "compatible_native_admissions_v7.json",
        "admissions": archive / "admissions_v7",
        "capacity_attempts": archive / "capacity_admission_attempts_v7.json",
        "capacity_report": archive / "capacity_precheck_current_v7.json",
        "native_prove_budget": archive / "ppi_native_prove_budget_v7.json",
        "record": archive / "retry_record.json",
    }


def _capacity_admission_retry_v2_paths(session_id: str) -> dict[str, Path]:
    if session_id != BASELINE_SESSION_ID:
        abort("capacity-Admission retry V2 audit is bound to the one original session")
    archive = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / session_id / "compatibility" / "capacity_admission_retry_v9"
    return {
        "archive": archive,
        "parent_v1_record": archive / "parent_v1_retry_record.json",
        "failed_session": archive / "failed_session_v8.json",
        "old_request": archive / "old_request_v8.json",
        "new_request": archive / "new_request_v9.json",
        "new_review": archive / "source_review_v9.json",
        "bad_receipts": archive / "bad_receipts",
        "capacity_metadata": archive / "capacity_metadata",
        "native_prove_budget": archive / "ppi_native_prove_budget_v8.json",
        "record": archive / "retry_record.json",
    }


def _validate_v8_capacity_request(request: object, original: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "session_id", "output_root", "matrix", "selected_datasets", "retained_setup_datasets",
        "source_review_path", "source_review_sha256", "baseline_manifest_sha256", "baseline_source_tree_sha256",
        "source_delta_sha256", "snapshot_index_inventory_sha256", "old_session_snapshot_path", "old_session_snapshot_sha256",
    }
    if not isinstance(request, dict) or set(request) != required:
        abort("capacity-Admission retry V8 request has unknown, missing, or malformed fields")
    if (
        request.get("schema") != REQUEST_SCHEMA or request.get("session_id") != BASELINE_SESSION_ID
        or request.get("output_root") != _COMPAT_OUTPUT_ROOT or request.get("matrix") != _COMPAT_MATRIX
        or request.get("selected_datasets") != _COMPAT_DATASETS or request.get("retained_setup_datasets") != ["ppi"]
        or request.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or request.get("old_session_snapshot_path") != _snapshot_path(BASELINE_SESSION_ID)
        or request.get("old_session_snapshot_sha256") != _ORIGINAL_SESSION_SHA256
        or request.get("baseline_source_tree_sha256") != load_json(ROOT / BASELINE_RELATIVE, "compatible resume frozen baseline").get("session_source_tree_sha256")
        or canonical_json_digest(request) == _V7_CAPACITY_ADMISSION_REQUEST_SHA256
    ):
        abort("capacity-Admission retry V8 request is not the exact new request for the pinned session")
    review_relative = _safe_relative_path(request.get("source_review_path"), "V8 source review")
    review_path = ROOT / review_relative
    try:
        review_path.resolve(strict=True).relative_to((ROOT / "runs" / "check_logs").resolve(strict=True))
    except (OSError, ValueError):
        abort("capacity-Admission retry V8 source review is outside runs/check_logs")
    review, review_sha, delta = validate_source_review(review_path, expected_session_id=BASELINE_SESSION_ID)
    if (
        review_sha != _require_sha(request.get("source_review_sha256"), "V8 source-review digest")
        or review.get("decision") != "APPROVED"
        or delta.get("diff_summary", {}).get("combined_diff_sha256") != request.get("source_delta_sha256")
        or compatible_source_review_index_digest(delta) != request.get("snapshot_index_inventory_sha256")
    ):
        abort("capacity-Admission retry V8 request/source review is not an exact current closed-world approval")
    return request


def _validate_v7_capacity_budget(
    budget_path: Path, failed: dict[str, Any], original: dict[str, Any],
) -> None:
    _regular_file(budget_path, "V7 PPI prove budget")
    raw = budget_path.read_bytes()
    if _digest_bytes(raw) != _V7_NATIVE_PROVE_BUDGET_SHA256:
        abort("capacity-Admission retry V7 prove-budget bytes differ from the pinned current budget")
    try:
        budget = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry V7 prove budget is invalid JSON: {exc}")
    identity = failed.get("compatible_prover_identity")
    identity_sha = failed.get("compatible_prover_identity_sha256")
    if (
        not isinstance(identity, dict) or identity_sha != _V7_CAPACITY_ADMISSION_PROVER_IDENTITY_SHA256
        or canonical_json_digest(identity) != identity_sha
    ):
        abort("capacity-Admission retry V7 prove budget has no pinned current prover identity")
    _setup_identity, _setup_identity_sha, setup_refs = validate_retained_setup_chain(original, session_id=BASELINE_SESSION_ID)
    manifest, _manifest_sha, _manifest_ref = validate_original_artifact_manifest(original, session_id=BASELINE_SESSION_ID)
    old_admission = original.get("native_admissions", {}).get("ppi")
    expected = {
        "schema": "ezkl.native-prove-budget.v8",
        "session_id": BASELINE_SESSION_ID,
        "dataset": "ppi",
        "admission_sha256": old_admission.get("sha256") if isinstance(old_admission, dict) else None,
        "settings_sha256": old_admission.get("settings_sha256") if isinstance(old_admission, dict) else None,
        "compiled_circuit_sha256": setup_refs.get("setup_files", {}).get("compiled_circuit_sha256"),
        "pk_sha256": setup_refs.get("setup_files", {}).get("pk_sha256"),
        "vk_sha256": setup_refs.get("setup_files", {}).get("vk_sha256"),
        "setup_receipt_sha256": setup_refs.get("setup_receipt", {}).get("sha256"),
        "native_keygen_budget_sha256": setup_refs.get("native_keygen_budget", {}).get("sha256"),
        "native_keygen_receipt_sha256": setup_refs.get("native_keygen_receipt", {}).get("sha256"),
        "resource_manifest_sha256": manifest.get("resource_manifest_sha256"),
        "binary_sha256": identity.get("binary_sha256"),
        "build_receipt_sha256": identity.get("build_receipt_sha256"),
        "build_environment_sha256": identity.get("build_environment_sha256"),
    }
    if not isinstance(budget, dict) or any(budget.get(key) != value for key, value in expected.items()):
        abort("capacity-Admission retry V7 prove budget is not bound to the retained PPI Setup and current V7 identity")


def _validate_v7_capacity_failure(
    document: dict[str, Any], original: dict[str, Any], *,
    admission_files: dict[str, tuple[Path, Path]] | None = None,
    report_path: Path | None = None, budget_path: Path | None = None,
    check_outputs: bool,
) -> None:
    expected_compatible = {
        "compatible_resume_request", "compatible_preflight_retry", "compatible_admission_preflight_retry",
        "compatible_interrupted_before_prove_recovery", "compatible_configured_admission_retry",
        "compatible_prover_identity", "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "compatible_native_admissions",
    }
    if (
        document.get("schema_version") != 1 or document.get("baseline") != "ezkl"
        or document.get("session_id") != BASELINE_SESSION_ID or document.get("status") != "FAILED"
        or document.get("stage") != "ppi_capacity_admission" or document.get("active_dataset") != "ppi"
        or document.get("output_root") != _COMPAT_OUTPUT_ROOT or document.get("matrix") != _COMPAT_MATRIX
        or document.get("selected_datasets") != _COMPAT_DATASETS
        or document.get("completed_cells") not in (None, {})
        or {key for key in document if key.startswith("compatible_")} != expected_compatible
        or document.get("compatible_resume_request") is None
        or canonical_json_digest(document["compatible_resume_request"]) != _V7_CAPACITY_ADMISSION_REQUEST_SHA256
        or canonical_json_digest(document.get("compatible_configured_admission_retry")) != _V7_CONFIGURED_ADMISSION_RETRY_CANONICAL_SHA256
        or document.get("native_admissions") != original.get("native_admissions")
        or document.get("setup_reuse_receipts") != original.get("setup_reuse_receipts")
        or document.get("stage_receipts") != original.get("stage_receipts")
        or document.get("compatible_stage_receipts") not in (None, {})
        or document.get("results") not in (None, {}) or document.get("verified_results") not in (None, {})
    ):
        abort("capacity-Admission retry is not the exact pinned V7 pre-Prove PPI failure")
    request = document["compatible_resume_request"]
    if (
        request.get("source_review_path") != _V7_CAPACITY_ADMISSION_REVIEW_PATH
        or request.get("source_review_sha256") != _V7_CAPACITY_ADMISSION_REVIEW_SHA256
        or request.get("old_session_snapshot_path") != _snapshot_path(BASELINE_SESSION_ID)
        or request.get("old_session_snapshot_sha256") != _ORIGINAL_SESSION_SHA256
        or canonical_json_digest(document.get("compatible_preflight_retry")) != _V5_V1_RETRY_CANONICAL_SHA256
        or canonical_json_digest(document.get("compatible_admission_preflight_retry")) != _V5_V2_RETRY_CANONICAL_SHA256
        or set(document.get("setup_reuse_receipts", {})) != {"ppi"}
        or set(document.get("stage_receipts", {})) != {"ppi"}
        or canonical_json_digest(document.get("resume_identity")) != canonical_json_digest(original.get("resume_identity"))
    ):
        abort("capacity-Admission retry V7 request or retained PPI Setup/Witness history changed")
    old_review_path = ROOT / _V7_CAPACITY_ADMISSION_REVIEW_PATH
    _regular_file(old_review_path, "pinned V7 source review")
    old_review_raw = old_review_path.read_bytes()
    _validate_approved_historical_review_bytes(
        old_review_raw, expected_sha256=_V7_CAPACITY_ADMISSION_REVIEW_SHA256, label="V7",
    )
    checks = document.get("capacity_checks")
    original_checks = original.get("capacity_checks")
    if (
        not isinstance(checks, list) or len(checks) != 6 or not isinstance(original_checks, list)
        or len(original_checks) != 4
        or canonical_json_digest(checks[:4]) != canonical_json_digest(original_checks)
        or [(item.get("dataset"), item.get("phase")) for item in checks[4:]]
        != [("ogbn_arxiv", "setup"), ("ppi", "witness_prove_verify")]
    ):
        abort("capacity-Admission retry V7 check ledger is not the exact original four plus two current checks")
    identity = document.get("compatible_prover_identity")
    capacity_identity = document.get("compatible_capacity_source_identity")
    if (
        not isinstance(identity, dict) or canonical_json_digest(identity) != _V7_CAPACITY_ADMISSION_PROVER_IDENTITY_SHA256
        or document.get("compatible_prover_identity_sha256") != _V7_CAPACITY_ADMISSION_PROVER_IDENTITY_SHA256
        or not isinstance(capacity_identity, dict)
        or canonical_json_digest(capacity_identity) != _V7_CAPACITY_ADMISSION_CAPACITY_IDENTITY_SHA256
        or document.get("compatible_capacity_source_identity_sha256") != _V7_CAPACITY_ADMISSION_CAPACITY_IDENTITY_SHA256
        or capacity_identity.get("source_tree_sha256") != identity.get("source_tree_sha256")
    ):
        abort("capacity-Admission retry V7 current prover/capacity identity pair changed")
    failure_map = document.get("dataset_failures")
    failure = failure_map.get("ppi") if isinstance(failure_map, dict) else None
    if (
        not isinstance(failure_map, dict) or set(failure_map) != {"ppi"}
        or not isinstance(failure, dict) or failure.get("stage") != "ppi_capacity_admission"
        or failure.get("exception_type") != "ContractError"
        or failure.get("error") != "compatible PPI capacity gate cannot be frozen over a conflicting prior gate/report"
    ):
        abort("capacity-Admission retry failure marker is not the exact pinned PPI gate conflict")

    current_map = document.get("compatible_native_admissions")
    if not isinstance(current_map, dict) or set(current_map) != set(_COMPAT_DATASETS):
        abort("capacity-Admission retry current Admission map must contain exactly both selected datasets")
    if canonical_json_digest(current_map) != _V7_CAPACITY_ADMISSION_MAP_CANONICAL_SHA256:
        abort("capacity-Admission retry current Admission map differs from the pinned V7 map")
    entry_keys = {
        "path", "sha256", "prepared_dir", "settings_sha256", "scalar_domain_vector_bytes",
        "total_assignments", "logrows", "domain_size", "constraint_system", "native_memory_model",
        "relation_identity",
    }
    authenticated_files: dict[str, tuple[Path, Path]] = {}
    original_map = original.get("native_admissions")
    for dataset in _COMPAT_DATASETS:
        ref = current_map[dataset]
        old_ref = original_map.get(dataset) if isinstance(original_map, dict) else None
        expected_dir = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/admission/{dataset}"
        old_admission_path = ROOT / (old_ref.get("path") if isinstance(old_ref, dict) else "")
        old_settings_path = old_admission_path.parent / "settings.json"
        if (
            not isinstance(ref, dict) or set(ref) != entry_keys
            or ref.get("path") != f"{expected_dir}/admission.json"
            or ref.get("prepared_dir") != expected_dir
            or not isinstance(old_ref, dict)
        ):
            abort(f"capacity-Admission retry {dataset} map path/schema differs from the pinned sidecar contract")
        _require_sha(ref.get("sha256"), f"V7 {dataset} Admission")
        _require_sha(ref.get("settings_sha256"), f"V7 {dataset} settings")
        old_admission = load_json(old_admission_path, f"original {dataset} Admission")
        if _digest_bytes(old_admission_path.read_bytes()) != old_ref.get("sha256"):
            abort(f"capacity-Admission retry original {dataset} Admission changed")
        if _digest_bytes(old_settings_path.read_bytes()) != old_ref.get("settings_sha256"):
            abort(f"capacity-Admission retry original {dataset} settings changed")
        if admission_files is None:
            admission_path = ROOT / ref["path"]
            settings_path = admission_path.parent / "settings.json"
        else:
            try:
                admission_path, settings_path = admission_files[dataset]
            except KeyError:
                abort(f"capacity-Admission retry {dataset} Admission archive is missing")
        _regular_file(admission_path, f"V7 {dataset} Admission")
        _regular_file(settings_path, f"V7 {dataset} settings")
        if (
            _digest_bytes(admission_path.read_bytes()) != ref["sha256"]
            or _digest_bytes(settings_path.read_bytes()) != ref["settings_sha256"]
        ):
            abort(f"capacity-Admission retry {dataset} Admission/settings bytes differ from their map")
        current_admission = load_json(admission_path, f"V7 {dataset} Admission")
        if (
            not isinstance(current_admission, dict) or current_admission.get("dataset") != dataset
            or current_admission.get("settings_sha256") != ref["settings_sha256"]
            or any(ref.get(key) != current_admission.get(key) for key in (
                "scalar_domain_vector_bytes", "total_assignments", "logrows", "domain_size",
                "constraint_system", "native_memory_model", "relation_identity",
            ))
        ):
            abort(f"capacity-Admission retry {dataset} map differs from its authenticated Admission JSON")
        compare_admission_semantics(
            old_admission, current_admission, dataset=dataset,
            old_settings_path=old_settings_path, current_settings_path=settings_path,
        )
        authenticated_files[dataset] = admission_path, settings_path

    attempts = document.get("capacity_admission_attempts")
    expected_report = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}/compatibility/capacity_precheck_current.json"
    if (
        not isinstance(attempts, dict)
        or set(attempts) != {"datasets", "readiness", "gate_status", "report_path", "report_sha256", "purpose", "failures", "audit_failures"}
        or attempts.get("datasets") != _COMPAT_DATASETS or attempts.get("readiness") != "NOT_ESTABLISHED"
        or attempts.get("gate_status") != "current_v4_pk_payload_lower_bound_unavailable"
        or attempts.get("report_path") != expected_report or attempts.get("report_sha256") != _V7_CAPACITY_ADMISSION_REPORT_SHA256
        or attempts.get("purpose") != "selected-native-admission-and-v4-PK-lower-bound-report-only; no setup or proof success implied"
        or attempts.get("failures") != {} or attempts.get("audit_failures") != {}
    ):
        abort("capacity-Admission retry report metadata is not the pinned V7 admission-only report")
    if report_path is None:
        report_path = ROOT / expected_report
    _regular_file(report_path, "V7 capacity report")
    if _digest_bytes(report_path.read_bytes()) != _V7_CAPACITY_ADMISSION_REPORT_SHA256:
        abort("capacity-Admission retry V7 report bytes differ from their pinned digest")
    report = load_json(report_path, "V7 capacity report")
    if not isinstance(report, dict):
        abort("capacity-Admission retry V7 report is malformed")

    if budget_path is None:
        budget_path = ROOT / _COMPAT_OUTPUT_ROOT / "ppi" / "native_prove_budget.json"
    _validate_v7_capacity_budget(budget_path, document, original)
    if check_outputs:
        _validate_configured_admission_outputs()


def _capacity_admission_snapshot_files(
    record: dict[str, Any], failed: dict[str, Any],
) -> tuple[dict[str, tuple[Path, Path]], Path, Path]:
    paths = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    map_ref = _require_path_ref(
        record.get("admissions_map_snapshot"), paths["admissions_map"].relative_to(ROOT).as_posix(), "V7 Admission-map snapshot",
    )
    raw_map = _snapshot_file(ROOT / map_ref["path"], relative=map_ref["path"], expected_digest=map_ref["sha256"])
    try:
        archived_map = json.loads(raw_map)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry V7 Admission map snapshot is invalid: {exc}")
    if archived_map != failed.get("compatible_native_admissions"):
        abort("capacity-Admission retry V7 Admission map snapshot differs from its failed-session bytes")
    entries = record.get("admission_file_snapshots")
    if not isinstance(entries, list) or [entry.get("dataset") if isinstance(entry, dict) else None for entry in entries] != _COMPAT_DATASETS:
        abort("capacity-Admission retry V7 Admission-file inventory is not the exact two-dataset pair")
    result: dict[str, tuple[Path, Path]] = {}
    for dataset, entry in zip(_COMPAT_DATASETS, entries):
        if not isinstance(entry, dict) or set(entry) != {"dataset", "source_admission", "source_settings", "snapshot_admission", "snapshot_settings"}:
            abort(f"capacity-Admission retry V7 {dataset} snapshot entry is malformed")
        source = archived_map.get(dataset)
        if not isinstance(source, dict):
            abort(f"capacity-Admission retry V7 {dataset} map entry is missing")
        settings_source_path = f"{Path(source['path']).parent.as_posix()}/settings.json"
        source_admission_ref = _require_path_ref(entry.get("source_admission"), source.get("path"), f"V7 {dataset} source Admission")
        source_settings_ref = _require_path_ref(entry.get("source_settings"), settings_source_path, f"V7 {dataset} source settings")
        if source_admission_ref["sha256"] != source.get("sha256") or source_settings_ref["sha256"] != source.get("settings_sha256"):
            abort(f"capacity-Admission retry V7 {dataset} source refs differ from its Admission map")
        expected_admission = paths["admissions"] / dataset / "admission.json"
        expected_settings = paths["admissions"] / dataset / "settings.json"
        admission_ref = _require_path_ref(entry.get("snapshot_admission"), expected_admission.relative_to(ROOT).as_posix(), f"V7 {dataset} archived Admission")
        settings_ref = _require_path_ref(entry.get("snapshot_settings"), expected_settings.relative_to(ROOT).as_posix(), f"V7 {dataset} archived settings")
        for ref, source_sha, label in (
            (admission_ref, source["sha256"], f"V7 {dataset} Admission"),
            (settings_ref, source["settings_sha256"], f"V7 {dataset} settings"),
        ):
            _snapshot_file(ROOT / ref["path"], relative=ref["path"], expected_digest=ref["sha256"])
            if ref["sha256"] != source_sha:
                abort(f"capacity-Admission retry archived {label} bytes differ from the V7 map")
        result[dataset] = ROOT / admission_ref["path"], ROOT / settings_ref["path"]

    attempts_path = paths["capacity_attempts"]
    attempts_ref = _require_path_ref(
        record.get("capacity_attempts_snapshot"), attempts_path.relative_to(ROOT).as_posix(), "V7 capacity-attempt snapshot",
    )
    attempts_raw = _snapshot_file(ROOT / attempts_ref["path"], relative=attempts_ref["path"], expected_digest=attempts_ref["sha256"])
    try:
        attempts = json.loads(attempts_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry V7 attempts snapshot is invalid: {exc}")
    if attempts != failed.get("capacity_admission_attempts"):
        abort("capacity-Admission retry V7 attempts differ from failed-session bytes")
    report_record = record.get("capacity_report_snapshot")
    if not isinstance(report_record, dict) or set(report_record) != {"source", "snapshot"}:
        abort("capacity-Admission retry V7 report snapshot binding is malformed")
    source_report = _require_path_ref(report_record.get("source"), attempts.get("report_path"), "V7 capacity report source")
    expected_report_snapshot = paths["capacity_report"]
    report_snapshot_ref = _require_path_ref(
        report_record.get("snapshot"), expected_report_snapshot.relative_to(ROOT).as_posix(), "V7 capacity report snapshot",
    )
    report_raw = _snapshot_file(
        ROOT / report_snapshot_ref["path"], relative=report_snapshot_ref["path"], expected_digest=report_snapshot_ref["sha256"],
    )
    if (
        source_report["sha256"] != _V7_CAPACITY_ADMISSION_REPORT_SHA256
        or report_snapshot_ref["sha256"] != source_report["sha256"]
        or attempts.get("report_sha256") != source_report["sha256"]
    ):
        abort("capacity-Admission retry V7 capacity report differs from its session/map anchor")
    try:
        report = json.loads(report_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry archived V7 report is invalid: {exc}")
    if not isinstance(report, dict):
        abort("capacity-Admission retry archived V7 report is malformed")

    budget_record = record.get("ppi_v7_native_prove_budget")
    if not isinstance(budget_record, dict) or set(budget_record) != {"source", "snapshot"}:
        abort("capacity-Admission retry V7 prove-budget snapshot binding is malformed")
    source_budget = _require_path_ref(
        budget_record.get("source"), f"{_COMPAT_OUTPUT_ROOT}/ppi/native_prove_budget.json", "V7 prove-budget source",
    )
    expected_budget_snapshot = paths["native_prove_budget"]
    snapshot_budget = _require_path_ref(
        budget_record.get("snapshot"), expected_budget_snapshot.relative_to(ROOT).as_posix(), "V7 prove-budget snapshot",
    )
    budget_raw = _snapshot_file(
        ROOT / snapshot_budget["path"], relative=snapshot_budget["path"], expected_digest=snapshot_budget["sha256"],
    )
    if (
        source_budget["sha256"] != _V7_NATIVE_PROVE_BUDGET_SHA256
        or snapshot_budget["sha256"] != source_budget["sha256"]
        or _digest_bytes(budget_raw) != _V7_NATIVE_PROVE_BUDGET_SHA256
    ):
        abort("capacity-Admission retry V7 prove budget differs from its immutable source digest")
    return result, ROOT / report_snapshot_ref["path"], ROOT / snapshot_budget["path"]


def _validate_capacity_admission_retry_record(
    session_document: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
    *, authenticated_v8_failure_raw: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = session_document.get("compatible_capacity_admission_retry")
    expected_keys = {
        "schema", "session_id", "failed_session_snapshot", "old_request_snapshot", "old_request_canonical_sha256",
        "old_configured_admission_retry_snapshot", "old_configured_admission_retry_canonical_sha256",
        "old_source_review", "old_source_review_snapshot", "new_request_snapshot", "new_request_canonical_sha256",
        "new_source_review", "new_source_review_snapshot", "original_session_snapshot", "capacity_checks_canonical_sha256",
        "prior_prover_identity_sha256", "prior_capacity_source_identity_sha256", "admissions_map_snapshot",
        "admission_file_snapshots", "capacity_attempts_snapshot", "capacity_report_snapshot", "ppi_v7_native_prove_budget",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        abort("capacity-Admission retry V8 audit has unknown, missing, or malformed fields")
    paths = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    _regular_file(paths["record"], "V8 retry-record snapshot")
    if paths["record"].read_bytes() != _json_snapshot_bytes(record):
        abort("capacity-Admission retry V8 audit differs from its immutable record snapshot")
    if (
        record.get("schema") != CAPACITY_ADMISSION_RETRY_SCHEMA or record.get("session_id") != BASELINE_SESSION_ID
        or session_document.get("compatible_resume_request") != request
        or record.get("new_request_canonical_sha256") != canonical_json_digest(request)
    ):
        abort("capacity-Admission retry V8 audit is not bound to the current request/session")
    failed, failed_raw = _load_retry_snapshot(record.get("failed_session_snapshot"), paths["failed_session"], "V7 failed-session")
    old_request, _ = _load_retry_snapshot(record.get("old_request_snapshot"), paths["old_request"], "V7 request")
    old_v7_record, _ = _load_retry_snapshot(
        record.get("old_configured_admission_retry_snapshot"), paths["old_retry_record"], "V7 configured-Admission record",
    )
    new_request, _ = _load_retry_snapshot(record.get("new_request_snapshot"), paths["new_request"], "V8 request")
    if (
        _digest_bytes(failed_raw) != _V7_CAPACITY_ADMISSION_SESSION_SHA256
        or canonical_json_digest(old_request) != _V7_CAPACITY_ADMISSION_REQUEST_SHA256
        or canonical_json_digest(old_request) != record.get("old_request_canonical_sha256")
        or failed.get("compatible_resume_request") != old_request
        or canonical_json_digest(old_v7_record) != _V7_CONFIGURED_ADMISSION_RETRY_CANONICAL_SHA256
        or canonical_json_digest(old_v7_record) != record.get("old_configured_admission_retry_canonical_sha256")
        or failed.get("compatible_configured_admission_retry") != old_v7_record
        or new_request != request or canonical_json_digest(new_request) != record.get("new_request_canonical_sha256")
        or canonical_json_digest(failed.get("capacity_checks")) != record.get("capacity_checks_canonical_sha256")
        or failed.get("compatible_prover_identity_sha256") != record.get("prior_prover_identity_sha256")
        or failed.get("compatible_capacity_source_identity_sha256") != record.get("prior_capacity_source_identity_sha256")
    ):
        abort("capacity-Admission retry archived V7 request/session/identity chain changed")

    old_review_ref = record.get("old_source_review")
    if not isinstance(old_review_ref, dict) or set(old_review_ref) != {"path", "sha256"} or old_review_ref != {
        "path": _V7_CAPACITY_ADMISSION_REVIEW_PATH, "sha256": _V7_CAPACITY_ADMISSION_REVIEW_SHA256,
    } or old_review_ref != {
        "path": old_request.get("source_review_path"), "sha256": old_request.get("source_review_sha256"),
    }:
        abort("capacity-Admission retry historical V7 source-review reference changed")
    old_review_snapshot = _require_path_ref(
        record.get("old_source_review_snapshot"), paths["old_review"].relative_to(ROOT).as_posix(), "V7 source-review snapshot",
    )
    old_review_raw = _snapshot_file(
        ROOT / old_review_snapshot["path"], relative=old_review_snapshot["path"], expected_digest=old_review_snapshot["sha256"],
    )
    _validate_approved_historical_review_bytes(old_review_raw, expected_sha256=_V7_CAPACITY_ADMISSION_REVIEW_SHA256, label="V7")
    if old_review_snapshot["sha256"] != _V7_CAPACITY_ADMISSION_REVIEW_SHA256:
        abort("capacity-Admission retry V7 source-review snapshot digest changed")

    new_review_ref = record.get("new_source_review")
    if not isinstance(new_review_ref, dict) or set(new_review_ref) != {"path", "sha256"} or new_review_ref != {
        "path": request.get("source_review_path"), "sha256": request.get("source_review_sha256"),
    }:
        abort("capacity-Admission retry V8 source-review reference differs from the request")
    new_review_snapshot = _require_path_ref(
        record.get("new_source_review_snapshot"), paths["new_review"].relative_to(ROOT).as_posix(), "V8 source-review snapshot",
    )
    if new_review_snapshot["sha256"] != request.get("source_review_sha256"):
        abort("capacity-Admission retry V8 source-review snapshot digest differs from the request")
    _snapshot_file(
        ROOT / new_review_snapshot["path"], relative=new_review_snapshot["path"], expected_digest=new_review_snapshot["sha256"],
    )

    original_ref = _require_path_ref(
        record.get("original_session_snapshot"), _snapshot_path(BASELINE_SESSION_ID), "original-session snapshot",
    )
    if original_ref["sha256"] != _ORIGINAL_SESSION_SHA256 or original != validate_original_session_snapshot(
        ROOT / original_ref["path"], session_id=BASELINE_SESSION_ID, expected_sha256=original_ref["sha256"],
    )[0]:
        abort("capacity-Admission retry changed the original PPI Setup/Witness anchor")
    v1_historical_context = session_document
    if authenticated_v8_failure_raw is not None:
        if (
            _digest_bytes(authenticated_v8_failure_raw) != _V8_POSTFAILURE_SESSION_SHA256
            or json.loads(authenticated_v8_failure_raw) != session_document
        ):
            abort("capacity-Admission retry V1 historical projection is not bound to the exact V8 failure bytes")
        current_history = session_document.get("dataset_failure_history")
        prior_history = failed.get("dataset_failure_history")
        if (
            not isinstance(current_history, dict)
            or not isinstance(prior_history, dict)
            or set(current_history) != set(prior_history)
            or canonical_json_digest(current_history) != _V8_POSTFAILURE_HISTORY_CANONICAL_SHA256
            or not isinstance(prior_history.get("ppi"), list)
            or not isinstance(current_history.get("ppi"), list)
            or len(current_history["ppi"]) != len(prior_history["ppi"]) + 1
            or current_history["ppi"][:-1] != prior_history["ppi"]
            or any(
                current_history.get(dataset) != prior_history.get(dataset)
                for dataset in set(current_history) - {"ppi"}
            )
        ):
            abort("capacity-Admission retry V1 history is not the exact pinned V7 prefix plus V8 failure")
        known_failure = current_history["ppi"][-1]
        if (
            not isinstance(known_failure, dict)
            or known_failure.get("timestamp") != "2026-11-01T18:07:02.415665+00:00"
            or known_failure.get("stage") != "ppi_capacity_admission"
            or known_failure.get("exception_type") != "ContractError"
            or known_failure.get("error") != "compatible resume shared artifact public_output_sha256 is not a canonical SHA-256"
        ):
            abort("capacity-Admission retry V1 history has an unknown V8 failure suffix")
        v1_historical_context = copy.deepcopy(session_document)
        v1_historical_context["dataset_failure_history"] = copy.deepcopy(prior_history)

    if (
        session_document.get("compatible_preflight_retry") != failed.get("compatible_preflight_retry")
        or session_document.get("compatible_admission_preflight_retry") != failed.get("compatible_admission_preflight_retry")
        or session_document.get("compatible_interrupted_before_prove_recovery") != failed.get("compatible_interrupted_before_prove_recovery")
        or session_document.get("compatible_configured_admission_retry") != failed.get("compatible_configured_admission_retry")
        or session_document.get("native_admissions") != original.get("native_admissions")
        or session_document.get("setup_reuse_receipts") != original.get("setup_reuse_receipts")
        or session_document.get("stage_receipts") != original.get("stage_receipts")
        or v1_historical_context.get("dataset_failure_history") != failed.get("dataset_failure_history")
        or canonical_json_digest(session_document.get("resume_identity")) != canonical_json_digest(original.get("resume_identity"))
    ):
        abort("capacity-Admission retry changed a preserved V1/V2/V6/V7 or original Setup/Witness chain")
    current_checks = session_document.get("capacity_checks")
    original_checks = original.get("capacity_checks")
    if (
        not isinstance(current_checks, list) or not isinstance(original_checks, list)
        or len(original_checks) != 4
        or canonical_json_digest(current_checks[:4]) != canonical_json_digest(original_checks)
    ):
        abort("capacity-Admission retry current capacity ledger lost its immutable original four-check prefix")

    # Authenticate V7 only in its immutable failed-session context. Its review is
    # historical here; the V8 request above is validated against the live census.
    _validate_configured_admission_retry_record(failed, old_request, original)
    archived_admissions, archived_report, archived_budget = _capacity_admission_snapshot_files(record, failed)
    _validate_v7_capacity_failure(
        failed, original, admission_files=archived_admissions, report_path=archived_report,
        budget_path=archived_budget, check_outputs=False,
    )

    uncommitted = copy.deepcopy(failed)
    uncommitted["compatible_resume_request"] = request
    uncommitted["compatible_capacity_admission_retry"] = record
    if canonical_json_digest(session_document) == canonical_json_digest(uncommitted):
        _validate_v7_capacity_failure(failed, original, check_outputs=True)
    return record, failed


def _capacity_admission_v2_file_specs() -> tuple[tuple[str, str, str, str], ...]:
    session_base = f"{_COMPAT_OUTPUT_ROOT}/.sessions/{BASELINE_SESSION_ID}"
    return (
        ("bad_receipts", f"{session_base}/compatibility/compatible_resume_receipt.json", "sidecar.json", _V8_BAD_COMPATIBILITY_RECEIPT_SHA256),
        ("bad_receipts", f"{_COMPAT_OUTPUT_ROOT}/ppi/compatible_resume_receipt.json", "ppi_output.json", _V8_BAD_COMPATIBILITY_RECEIPT_SHA256),
        ("capacity_metadata", f"{session_base}/compatibility/capacity_precheck_current.json", "capacity_precheck_current.json", _V8_PPI_CAPACITY_REPORT_SHA256),
        ("capacity_metadata", f"{session_base}/compatibility/capacity_precheck_ppi.json", "capacity_precheck_ppi.json", _V8_PPI_CAPACITY_REPORT_SHA256),
        ("capacity_metadata", f"{session_base}/compatibility/capacity_rechecks.json", "capacity_rechecks.json", _V8_PPI_CAPACITY_RECHECKS_SHA256),
    )


def _validate_capacity_admission_v2_failed_session(
    failed: dict[str, Any], failed_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the exact V8 failure and its V1 parent in historical context."""
    if _digest_bytes(failed_raw) != _V8_POSTFAILURE_SESSION_SHA256:
        abort("capacity-Admission retry V2 failed-session bytes differ from the one pinned V8 receipt failure")
    try:
        parsed = json.loads(failed_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry V2 failed-session snapshot is invalid JSON: {exc}")
    if not isinstance(parsed, dict) or parsed != failed:
        abort("capacity-Admission retry V2 failed-session object differs from its authenticated bytes")
    old_request = failed.get("compatible_resume_request")
    if (
        not isinstance(old_request, dict)
        or canonical_json_digest(old_request) != _V8_POSTFAILURE_REQUEST_SHA256
        or failed.get("status") != "FAILED"
        or failed.get("stage") != "ppi_capacity_admission"
        or failed.get("active_dataset") != "ppi"
        or failed.get("capacity_checks") is None
        or canonical_json_digest(failed.get("capacity_checks")) != _V8_CAPACITY_CHECKS_CANONICAL_SHA256
        or failed.get("compatible_stage_receipts") not in (None, {})
        or failed.get("results") not in (None, {})
        or failed.get("verified_results") not in (None, {})
    ):
        abort("capacity-Admission retry V2 is not the exact pinned pre-Prove PPI failure")
    previous_record = failed.get("compatible_capacity_admission_retry")
    if not isinstance(previous_record, dict) or previous_record.get("schema") != CAPACITY_ADMISSION_RETRY_SCHEMA:
        abort("capacity-Admission retry V2 parent is not the complete V1 audit")
    v1_paths = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    _regular_file(v1_paths["record"], "immutable V1 capacity retry audit")
    v1_raw = v1_paths["record"].read_bytes()
    if _digest_bytes(v1_raw) != _V8_RETRY_RECORD_SHA256 or v1_raw != _json_snapshot_bytes(previous_record):
        abort("capacity-Admission retry V2 parent V1 archive differs from the exact pinned record")
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if original_sha != _ORIGINAL_SESSION_SHA256:
        abort("capacity-Admission retry V2 original session snapshot changed")
    _validate_capacity_admission_retry_record(
        failed, old_request, original, authenticated_v8_failure_raw=failed_raw,
    )
    return original, previous_record


def _validate_capacity_admission_v2_candidate(
    failed: dict[str, Any], failed_raw: bytes, request: dict[str, Any], *,
    allow_archived_sources: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one pinned V8-receipt-failure to current-source V2 rebind."""
    paths = _capacity_admission_retry_v2_paths(BASELINE_SESSION_ID)
    original, previous_record = _validate_capacity_admission_v2_failed_session(failed, failed_raw)
    old_request = failed.get("compatible_resume_request")
    if (
        request.get("old_session_snapshot_sha256") != _ORIGINAL_SESSION_SHA256
        or request == old_request
        or request.get("source_review_sha256") == _V8_SOURCE_REVIEW_SHA256
    ):
        abort("capacity-Admission retry V2 request is not a distinct current reviewed request")
    _validate_v8_capacity_request(request, original)

    # Validate V1 only against its immutable V8 failed-session context. In
    # particular, do not re-run its old source review against today's source.
    manifest, manifest_sha, _manifest_ref = validate_original_artifact_manifest(original, session_id=BASELINE_SESSION_ID)
    if manifest_sha != "a4641a5d99803ef1779ff8fea4b3a158039101f0d579b93d77bab943983ef7d3":
        abort("capacity-Admission retry V2 original PPI artifact manifest changed")
    _setup_identity, _setup_identity_sha, _setup_refs = validate_retained_setup_chain(original, session_id=BASELINE_SESSION_ID)
    witness = validate_retained_witness_chain(original, session_id=BASELINE_SESSION_ID)
    if (
        _setup_refs.get("setup_receipt", {}).get("sha256") != _ORIGINAL_PPI_SETUP_RECEIPT_SHA256
        or witness.get("stage_receipt_sha256") != _ORIGINAL_PPI_WITNESS_STAGE_SHA256
        or witness.get("old_native_prove_budget", {}).get("sha256") != _V8_LEGACY_NATIVE_PROVE_BUDGET_SHA256
    ):
        abort("capacity-Admission retry V2 original PPI Setup/Witness/legacy-budget anchors changed")

    for category, source_rel, leaf, expected_sha in _capacity_admission_v2_file_specs():
        snapshot = paths[category] / leaf
        if snapshot.is_symlink():
            abort(f"capacity-Admission retry V2 {leaf} archive is symlinked")
        source = ROOT / source_rel
        if source.exists() or source.is_symlink():
            _regular_file(source, f"current V8 {leaf}")
            if _digest_bytes(source.read_bytes()) != expected_sha:
                abort(f"capacity-Admission retry V2 current {leaf} differs from the pinned failure")
        elif not allow_archived_sources or not snapshot.is_file() or _digest_bytes(snapshot.read_bytes()) != expected_sha:
            abort(f"capacity-Admission retry V2 current {leaf} is absent without its immutable recovery copy")

    budget_path = ROOT / _COMPAT_OUTPUT_ROOT / "ppi" / "native_prove_budget.json"
    _regular_file(budget_path, "current V8 PPI prove budget")
    if _digest_bytes(budget_path.read_bytes()) != _V8_PPI_PROVE_BUDGET_SHA256:
        abort("capacity-Admission retry V2 current prove budget differs from the exact pre-Prove CAS source")
    _validate_no_proof_outputs()
    ogbn = ROOT / _COMPAT_OUTPUT_ROOT / "ogbn_arxiv"
    if ogbn.is_symlink() or any((ogbn / name).exists() or (ogbn / name).is_symlink() for name in (
        "pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json",
    )):
        abort("capacity-Admission retry V2 refuses any OGBN Setup output")
    return original, previous_record


def validate_capacity_admission_v2_controller_candidate(
    session_file: Path, session_document: dict[str, Any], source_review_path: Path,
) -> bool:
    """Validate the exact V8-to-V2 controller entry with the CLI's current review."""
    expected = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file() or session_file.resolve(strict=True) != expected.resolve(strict=True):
        abort("capacity-Admission retry V2 controller gate must use the exact baseline session")
    raw = session_file.read_bytes()
    if _digest_bytes(raw) != _V8_POSTFAILURE_SESSION_SHA256:
        return False
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry V2 controller session is invalid JSON: {exc}")
    if not isinstance(parsed, dict) or parsed != session_document:
        abort("capacity-Admission retry V2 controller session differs from its authenticated bytes")
    record = parsed.get("compatible_capacity_admission_retry")
    if not isinstance(record, dict) or record.get("schema") != CAPACITY_ADMISSION_RETRY_SCHEMA:
        return False
    _validate_capacity_admission_v2_failed_session(parsed, raw)
    validate_source_review(source_review_path, expected_session_id=BASELINE_SESSION_ID)
    _validate_no_proof_outputs()
    ogbn = ROOT / _COMPAT_OUTPUT_ROOT / "ogbn_arxiv"
    if ogbn.is_symlink() or any((ogbn / name).exists() or (ogbn / name).is_symlink() for name in (
        "pk.key", "vk.key", "native_keygen_receipt.json", "setup_receipt.json", "native_setup_receipt.json",
    )):
        abort("capacity-Admission retry V2 refuses any OGBN Setup output")
    return True


def _validate_capacity_admission_v2_record(
    session_document: dict[str, Any], request: dict[str, Any], original: dict[str, Any],
) -> dict[str, Any]:
    record = session_document.get("compatible_capacity_admission_retry")
    expected_keys = {
        "schema", "session_id", "parent_v1_archive", "parent_v1_record", "parent_v1_record_snapshot", "failed_session_snapshot",
        "old_request_snapshot", "old_request_canonical_sha256", "new_request_snapshot",
        "new_request_canonical_sha256", "new_source_review", "new_source_review_snapshot",
        "original_session_snapshot", "capacity_checks_canonical_sha256", "bad_receipts",
        "capacity_metadata", "ppi_v8_native_prove_budget",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        abort("capacity-Admission retry V2 audit has unknown, missing, or malformed fields")
    paths = _capacity_admission_retry_v2_paths(BASELINE_SESSION_ID)
    _regular_file(paths["record"], "V2 retry-record snapshot")
    record_raw = paths["record"].read_bytes()
    if record_raw != _json_snapshot_bytes(record):
        abort("capacity-Admission retry V2 audit differs from its immutable record snapshot")
    if (
        record.get("schema") != CAPACITY_ADMISSION_RETRY_V2_SCHEMA
        or record.get("session_id") != BASELINE_SESSION_ID
        or session_document.get("compatible_resume_request") != request
        or record.get("new_request_canonical_sha256") != canonical_json_digest(request)
        or record.get("capacity_checks_canonical_sha256") != _V8_CAPACITY_CHECKS_CANONICAL_SHA256
    ):
        abort("capacity-Admission retry V2 audit is not bound to the current request/session/ledger")
    parent_archive = record.get("parent_v1_archive")
    expected_v1 = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    if parent_archive != expected_v1["archive"].relative_to(ROOT).as_posix():
        abort("capacity-Admission retry V2 parent archive path changed")
    parent_ref = _require_path_ref(record.get("parent_v1_record"), expected_v1["record"].relative_to(ROOT).as_posix(), "V1 parent record")
    parent_snapshot = _require_path_ref(
        record.get("parent_v1_record_snapshot"), paths["parent_v1_record"].relative_to(ROOT).as_posix(), "V1 parent-record snapshot",
    )
    if parent_ref["sha256"] != _V8_RETRY_RECORD_SHA256 or parent_snapshot["sha256"] != _V8_RETRY_RECORD_SHA256:
        abort("capacity-Admission retry V2 parent V1 record reference digest changed")
    parent_raw = _snapshot_file(ROOT / parent_ref["path"], relative=parent_ref["path"], expected_digest=_V8_RETRY_RECORD_SHA256)
    copied_parent_raw = _snapshot_file(ROOT / parent_snapshot["path"], relative=parent_snapshot["path"], expected_digest=_V8_RETRY_RECORD_SHA256)
    if parent_raw != copied_parent_raw:
        abort("capacity-Admission retry V2 parent V1 record copy differs from its original archive")

    failed_ref = _require_path_ref(record.get("failed_session_snapshot"), paths["failed_session"].relative_to(ROOT).as_posix(), "V8 failed-session snapshot")
    if failed_ref["sha256"] != _V8_POSTFAILURE_SESSION_SHA256:
        abort("capacity-Admission retry V2 failed-session reference digest changed")
    failed_raw = _snapshot_file(ROOT / failed_ref["path"], relative=failed_ref["path"], expected_digest=_V8_POSTFAILURE_SESSION_SHA256)
    failed = json.loads(failed_raw)
    old_ref = _require_path_ref(record.get("old_request_snapshot"), paths["old_request"].relative_to(ROOT).as_posix(), "V8 request snapshot")
    expected_old_sha = _digest_bytes(_json_snapshot_bytes(failed.get("compatible_resume_request")))
    if old_ref["sha256"] != expected_old_sha:
        abort("capacity-Admission retry V2 old-request reference digest changed")
    old_raw = _snapshot_file(ROOT / old_ref["path"], relative=old_ref["path"], expected_digest=expected_old_sha)
    old_request = json.loads(old_raw)
    if (
        canonical_json_digest(old_request) != _V8_POSTFAILURE_REQUEST_SHA256
        or record.get("old_request_canonical_sha256") != _V8_POSTFAILURE_REQUEST_SHA256
        or failed.get("compatible_capacity_admission_retry") != json.loads(parent_raw)
    ):
        abort("capacity-Admission retry V2 archived V8 session does not contain the parent V1 audit")
    archived_original, _previous_record = _validate_capacity_admission_v2_failed_session(failed, failed_raw)
    if archived_original != original:
        abort("capacity-Admission retry V2 archived V8 session changed its original session anchor")

    new_ref = _require_path_ref(record.get("new_request_snapshot"), paths["new_request"].relative_to(ROOT).as_posix(), "V9 request snapshot")
    expected_new_sha = _digest_bytes(_json_snapshot_bytes(request))
    if new_ref["sha256"] != expected_new_sha:
        abort("capacity-Admission retry V2 new-request reference digest changed")
    new_raw = _snapshot_file(ROOT / new_ref["path"], relative=new_ref["path"], expected_digest=expected_new_sha)
    if json.loads(new_raw) != request:
        abort("capacity-Admission retry V2 request snapshot differs from the active request")
    _validate_v8_capacity_request(request, original)
    review_ref = record.get("new_source_review")
    if not isinstance(review_ref, dict) or set(review_ref) != {"path", "sha256"} or review_ref != {
        "path": request.get("source_review_path"), "sha256": request.get("source_review_sha256"),
    }:
        abort("capacity-Admission retry V2 source review differs from the current request")
    review_snapshot = _require_path_ref(
        record.get("new_source_review_snapshot"), paths["new_review"].relative_to(ROOT).as_posix(), "V9 source review snapshot",
    )
    if review_snapshot["sha256"] != request.get("source_review_sha256"):
        abort("capacity-Admission retry V2 source-review snapshot digest differs from the request")
    _snapshot_file(ROOT / review_snapshot["path"], relative=review_snapshot["path"], expected_digest=review_snapshot["sha256"])

    original_ref = _require_path_ref(record.get("original_session_snapshot"), _snapshot_path(BASELINE_SESSION_ID), "original session snapshot")
    if original_ref["sha256"] != _ORIGINAL_SESSION_SHA256 or original != validate_original_session_snapshot(
        ROOT / original_ref["path"], session_id=BASELINE_SESSION_ID, expected_sha256=original_ref["sha256"],
    )[0]:
        abort("capacity-Admission retry V2 changed the original PPI Setup/Witness anchor")

    for key, category, specs in (
        ("bad_receipts", "bad_receipts", _capacity_admission_v2_file_specs()[:2]),
        ("capacity_metadata", "capacity_metadata", _capacity_admission_v2_file_specs()[2:]),
    ):
        entries = record.get(key)
        if not isinstance(entries, list) or len(entries) != len(specs):
            abort(f"capacity-Admission retry V2 {key} inventory is malformed")
        for entry, (expected_category, source_rel, leaf, expected_sha) in zip(entries, specs):
            target = paths[expected_category] / leaf
            if not isinstance(entry, dict) or set(entry) != {"source", "snapshot"}:
                abort(f"capacity-Admission retry V2 {leaf} archive entry is malformed")
            source_ref = _require_path_ref(entry["source"], source_rel, f"V2 {leaf} source")
            snapshot_ref = _require_path_ref(entry["snapshot"], target.relative_to(ROOT).as_posix(), f"V2 {leaf} snapshot")
            if source_ref["sha256"] != expected_sha or snapshot_ref["sha256"] != expected_sha:
                abort(f"capacity-Admission retry V2 {leaf} digest differs from its pinned failure bytes")
            _snapshot_file(ROOT / snapshot_ref["path"], relative=snapshot_ref["path"], expected_digest=expected_sha)

    budget_ref = record.get("ppi_v8_native_prove_budget")
    if not isinstance(budget_ref, dict) or set(budget_ref) != {"source", "snapshot"}:
        abort("capacity-Admission retry V2 prove-budget archive binding is malformed")
    source_budget = _require_path_ref(budget_ref["source"], f"{_COMPAT_OUTPUT_ROOT}/ppi/native_prove_budget.json", "V8 prove-budget source")
    snapshot_budget = _require_path_ref(
        budget_ref["snapshot"], paths["native_prove_budget"].relative_to(ROOT).as_posix(), "V8 prove-budget snapshot",
    )
    budget_raw = _snapshot_file(ROOT / snapshot_budget["path"], relative=snapshot_budget["path"], expected_digest=_V8_PPI_PROVE_BUDGET_SHA256)
    if source_budget["sha256"] != _V8_PPI_PROVE_BUDGET_SHA256 or snapshot_budget["sha256"] != source_budget["sha256"] or _digest_bytes(budget_raw) != _V8_PPI_PROVE_BUDGET_SHA256:
        abort("capacity-Admission retry V2 prove-budget archive differs from the exact V8 source")
    return record


def validate_capacity_admission_v2_rebind_candidate(session_file: Path, request: dict[str, Any]) -> None:
    """Early one-time guard for the exact current-review successor to the V8 failure."""
    expected = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file() or session_file.resolve(strict=True) != expected.resolve(strict=True):
        abort("capacity-Admission retry V2 guard must use the exact baseline session")
    raw = session_file.read_bytes()
    try:
        failed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry V2 failed session is invalid JSON: {exc}")
    if not isinstance(failed, dict):
        abort("capacity-Admission retry V2 failed session is malformed")
    record = failed.get("compatible_capacity_admission_retry")
    if isinstance(record, dict) and record.get("schema") == CAPACITY_ADMISSION_RETRY_V2_SCHEMA:
        if failed.get("compatible_resume_request") != request:
            abort("capacity-Admission retry V2 is one-time and its current request cannot be changed")
        original, _ = validate_original_session_snapshot(
            ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
            expected_sha256=_ORIGINAL_SESSION_SHA256,
        )
        _validate_capacity_admission_v2_record(failed, request, original)
        return
    if not isinstance(record, dict) or record.get("schema") != CAPACITY_ADMISSION_RETRY_SCHEMA:
        abort("capacity-Admission retry V2 requires the exact V1 parent audit")
    _validate_capacity_admission_v2_candidate(failed, raw, request, allow_archived_sources=True)


def _prepare_capacity_admission_v2_retry(
    session_file: Path, failed: dict[str, Any], failed_raw: bytes, request: dict[str, Any],
) -> dict[str, Any]:
    paths = _capacity_admission_retry_v2_paths(BASELINE_SESSION_ID)
    original, _previous_record = _validate_capacity_admission_v2_candidate(
        failed, failed_raw, request, allow_archived_sources=True,
    )
    if paths["record"].exists() or paths["record"].is_symlink():
        _regular_file(paths["record"], "prepared V2 retry record")
        existing_record = load_json(paths["record"], "prepared V2 retry record")
        if not isinstance(existing_record, dict) or existing_record.get("new_request_canonical_sha256") != canonical_json_digest(request):
            abort("capacity-Admission retry V2 archive already binds a different one-time request")
        candidate = copy.deepcopy(failed)
        _apply_capacity_admission_v2_candidate(candidate, request, existing_record, original)
        _validate_capacity_admission_v2_record(candidate, request, original)
        return existing_record

    v1_paths = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    v1_raw = v1_paths["record"].read_bytes()
    v1_record_snapshot_sha = _write_immutable_snapshot(paths["parent_v1_record"], v1_raw, "V1 parent retry record")
    failed_sha = _write_immutable_snapshot(paths["failed_session"], failed_raw, "V8 failed-session")
    old_request = failed["compatible_resume_request"]
    old_request_sha = _write_immutable_snapshot(paths["old_request"], _json_snapshot_bytes(old_request), "V8 request")
    new_request_sha = _write_immutable_snapshot(paths["new_request"], _json_snapshot_bytes(request), "V9 request")
    review_path = ROOT / request["source_review_path"]
    _regular_file(review_path, "V9 current source review")
    review_raw = review_path.read_bytes()
    if _digest_bytes(review_raw) != request["source_review_sha256"]:
        abort("capacity-Admission retry V2 current source review changed while snapshotting")
    review_sha = _write_immutable_snapshot(paths["new_review"], review_raw, "V9 source review")

    bad_receipt_entries: list[dict[str, Any]] = []
    metadata_entries: list[dict[str, Any]] = []
    for category, source_rel, leaf, expected_sha in _capacity_admission_v2_file_specs():
        source = ROOT / source_rel
        snapshot_path = paths[category] / leaf
        if source.exists() or source.is_symlink():
            _regular_file(source, f"V8 {leaf} source")
            raw = source.read_bytes()
        elif snapshot_path.is_file() and not snapshot_path.is_symlink():
            raw = snapshot_path.read_bytes()
        else:
            abort(f"capacity-Admission retry V2 cannot archive absent {leaf}")
        if _digest_bytes(raw) != expected_sha:
            abort(f"capacity-Admission retry V2 {leaf} differs from its pinned source bytes")
        snapshot_sha = _write_immutable_snapshot(snapshot_path, raw, f"V8 {leaf}")
        if category == "bad_receipts":
            bad_receipt_entries.append({
                "source": {"path": source_rel, "sha256": expected_sha},
                "snapshot": {"path": snapshot_path.relative_to(ROOT).as_posix(), "sha256": snapshot_sha},
            })
        else:
            metadata_entries.append({
                "source": {"path": source_rel, "sha256": expected_sha},
                "snapshot": {"path": snapshot_path.relative_to(ROOT).as_posix(), "sha256": snapshot_sha},
            })
    budget_source_rel = f"{_COMPAT_OUTPUT_ROOT}/ppi/native_prove_budget.json"
    budget_raw = (ROOT / budget_source_rel).read_bytes()
    budget_sha = _write_immutable_snapshot(paths["native_prove_budget"], budget_raw, "V8 PPI native prove budget")
    if budget_sha != _V8_PPI_PROVE_BUDGET_SHA256:
        abort("capacity-Admission retry V2 prove-budget snapshot differs from the exact V8 CAS source")

    record: dict[str, Any] = {
        "schema": CAPACITY_ADMISSION_RETRY_V2_SCHEMA,
        "session_id": BASELINE_SESSION_ID,
        "parent_v1_archive": v1_paths["archive"].relative_to(ROOT).as_posix(),
        "parent_v1_record": {"path": v1_paths["record"].relative_to(ROOT).as_posix(), "sha256": _V8_RETRY_RECORD_SHA256},
        "parent_v1_record_snapshot": {"path": paths["parent_v1_record"].relative_to(ROOT).as_posix(), "sha256": v1_record_snapshot_sha},
        "failed_session_snapshot": {"path": paths["failed_session"].relative_to(ROOT).as_posix(), "sha256": failed_sha},
        "old_request_snapshot": {"path": paths["old_request"].relative_to(ROOT).as_posix(), "sha256": old_request_sha},
        "old_request_canonical_sha256": canonical_json_digest(old_request),
        "new_request_snapshot": {"path": paths["new_request"].relative_to(ROOT).as_posix(), "sha256": new_request_sha},
        "new_request_canonical_sha256": canonical_json_digest(request),
        "new_source_review": {"path": request["source_review_path"], "sha256": request["source_review_sha256"]},
        "new_source_review_snapshot": {"path": paths["new_review"].relative_to(ROOT).as_posix(), "sha256": review_sha},
        "original_session_snapshot": {"path": _snapshot_path(BASELINE_SESSION_ID), "sha256": _ORIGINAL_SESSION_SHA256},
        "capacity_checks_canonical_sha256": canonical_json_digest(failed["capacity_checks"]),
        "bad_receipts": bad_receipt_entries,
        "capacity_metadata": metadata_entries,
        "ppi_v8_native_prove_budget": {
            "source": {"path": budget_source_rel, "sha256": budget_sha},
            "snapshot": {"path": paths["native_prove_budget"].relative_to(ROOT).as_posix(), "sha256": budget_sha},
        },
    }
    record_sha = _write_immutable_snapshot(paths["record"], _json_snapshot_bytes(record), "V2 retry record")
    if record_sha != _digest_bytes(_json_snapshot_bytes(record)):
        abort("capacity-Admission retry V2 record snapshot digest changed")
    candidate = copy.deepcopy(failed)
    _apply_capacity_admission_v2_candidate(candidate, request, record, original)
    _validate_capacity_admission_v2_record(candidate, request, original)
    return record


def _apply_capacity_admission_v2_candidate(
    session_document: dict[str, Any], request: dict[str, Any], record: dict[str, Any], original: dict[str, Any],
) -> None:
    session_document["compatible_resume_request"] = request
    session_document["compatible_capacity_admission_retry"] = record
    session_document["capacity_checks"] = copy.deepcopy(original.get("capacity_checks"))
    for name in (
        "capacity_admission_attempts", "compatible_native_admissions", "compatible_prover_identity",
        "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "capacity_source_identity",
        "capacity_source_identity_sha256", "active_dataset", "dataset_failures",
    ):
        session_document.pop(name, None)


def _migrate_capacity_admission_v2_files(record: dict[str, Any]) -> None:
    """Remove only exact V8 bad files after durable no-overwrite archive copies exist."""
    paths = _capacity_admission_retry_v2_paths(BASELINE_SESSION_ID)
    entries = record.get("bad_receipts", []) + record.get("capacity_metadata", [])
    _validate_no_proof_outputs()
    for entry in entries:
        source_ref = entry.get("source") if isinstance(entry, dict) else None
        snapshot_ref = entry.get("snapshot") if isinstance(entry, dict) else None
        if not isinstance(source_ref, dict) or not isinstance(snapshot_ref, dict):
            abort("capacity-Admission retry V2 migration file references are malformed")
        source = ROOT / _safe_relative_path(source_ref.get("path"), "V8 migration source")
        archived = ROOT / _safe_relative_path(snapshot_ref.get("path"), "V8 migration archive")
        expected_sha = _require_sha(source_ref.get("sha256"), "V8 migration source digest")
        _regular_file(archived, "immutable V8 migration archive")
        if _digest_bytes(archived.read_bytes()) != expected_sha or snapshot_ref.get("sha256") != expected_sha:
            abort("capacity-Admission retry V2 migration archive copy is not exact")
        if source.exists() or source.is_symlink():
            _regular_file(source, "V8 migration source")
            if _digest_bytes(source.read_bytes()) != expected_sha:
                abort("capacity-Admission retry V2 refuses to remove changed V8 source bytes")
            source.unlink()
            _fsync_parent(source.parent)
    _validate_no_proof_outputs()


def validate_failed_capacity_admission_resume_gate(session_document: dict[str, Any]) -> None:
    """Admit only the exact pinned V7 PPI capacity-gate failure for V8 recovery."""
    if not isinstance(session_document, dict) or session_document.get("status") != "FAILED" or session_document.get("stage") != "ppi_capacity_admission":
        abort("capacity-Admission retry requires the exact FAILED/ppi_capacity_admission boundary")
    if session_document.get("compatible_capacity_admission_retry") is not None:
        validate_compatible_resume_request(session_document, expected_session_id=BASELINE_SESSION_ID)
        _validate_no_proof_outputs()
        return
    session_file = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file() or _digest_bytes(session_file.read_bytes()) != _V7_CAPACITY_ADMISSION_SESSION_SHA256:
        abort("capacity-Admission retry failed-session bytes differ from the one pinned V7 candidate")
    live = load_json(session_file, "capacity-Admission retry V7 failed session")
    if canonical_json_digest(live) != canonical_json_digest(session_document):
        abort("capacity-Admission retry in-memory session differs from the pinned V7 bytes")
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if original_sha != _ORIGINAL_SESSION_SHA256:
        abort("capacity-Admission retry original session snapshot changed")
    _validate_v7_capacity_failure(live, original, check_outputs=True)
    v7_record_path = _configured_admission_retry_paths(BASELINE_SESSION_ID)["record"]
    _regular_file(v7_record_path, "pinned V7 configured-Admission record")
    v7_record = load_json(v7_record_path, "pinned V7 configured-Admission record")
    if canonical_json_digest(v7_record) != _V7_CONFIGURED_ADMISSION_RETRY_CANONICAL_SHA256:
        abort("capacity-Admission retry V7 configured-Admission record changed")


def prepare_compatible_capacity_admission_retry(
    session_file: Path, request: dict[str, Any],
) -> dict[str, Any] | None:
    """Archive the pinned V7 capacity-gate failure and prepare its one V8 rebind."""
    expected_session = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if (
        session_file.is_symlink() or not session_file.is_file()
        or session_file.resolve(strict=True) != expected_session.resolve(strict=True)
    ):
        abort("capacity-Admission retry must use the exact original session record")
    live_raw = session_file.read_bytes()
    try:
        live = json.loads(live_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"capacity-Admission retry session JSON is invalid: {exc}")
    if not isinstance(live, dict):
        abort("capacity-Admission retry session is malformed")
    existing_retry = live.get("compatible_capacity_admission_retry")
    if isinstance(existing_retry, dict) and existing_retry.get("schema") == CAPACITY_ADMISSION_RETRY_V2_SCHEMA:
        if live.get("compatible_resume_request") != request:
            abort("capacity-Admission retry V2 request cannot be changed after its one-time audit")
        validate_compatible_resume_request(live, expected_session_id=BASELINE_SESSION_ID)
        return None
    if existing_retry is not None:
        if live.get("compatible_resume_request") == request:
            validate_compatible_resume_request(live, expected_session_id=BASELINE_SESSION_ID)
            return None
        if isinstance(existing_retry, dict) and existing_retry.get("schema") == CAPACITY_ADMISSION_RETRY_SCHEMA:
            return _prepare_capacity_admission_v2_retry(session_file, live, live_raw, request)
        abort("capacity-Admission retry has an unsupported or already-consumed transition")
    # This helper is called first for every compatible resume request. Decline
    # sessions owned by the earlier retry transitions so their existing strict
    # validators can handle them; only the exact known capacity-failure stage
    # is eligible for this one V7-to-V8 migration.
    if live.get("status") != "FAILED" or live.get("stage") != "ppi_capacity_admission":
        return None
    if _digest_bytes(live_raw) != _V7_CAPACITY_ADMISSION_SESSION_SHA256:
        abort("capacity-Admission retry failed-session bytes differ from the one pinned V7 candidate")
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    if original_sha != _ORIGINAL_SESSION_SHA256:
        abort("capacity-Admission retry original session snapshot changed")
    _validate_v8_capacity_request(request, original)
    _validate_v7_capacity_failure(live, original, check_outputs=True)
    paths = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    old_request = live["compatible_resume_request"]
    v7_record = live["compatible_configured_admission_retry"]
    v7_record_path = _configured_admission_retry_paths(BASELINE_SESSION_ID)["record"]
    _regular_file(v7_record_path, "V7 configured-Admission retry record")
    v7_record_raw = v7_record_path.read_bytes()
    if (
        canonical_json_digest(json.loads(v7_record_raw)) != _V7_CONFIGURED_ADMISSION_RETRY_CANONICAL_SHA256
        or json.loads(v7_record_raw) != v7_record
    ):
        abort("capacity-Admission retry immutable V7 record differs from the failed session")

    new_request_sha = _write_immutable_snapshot(paths["new_request"], _json_snapshot_bytes(request), "V8 request")
    failed_sha = _write_immutable_snapshot(paths["failed_session"], live_raw, "V7 failed-session")
    old_request_sha = _write_immutable_snapshot(paths["old_request"], _json_snapshot_bytes(old_request), "V7 request")
    old_v7_record_sha = _write_immutable_snapshot(paths["old_retry_record"], v7_record_raw, "V7 retry record")
    old_review_path = ROOT / _V7_CAPACITY_ADMISSION_REVIEW_PATH
    _regular_file(old_review_path, "V7 historical source review")
    old_review_raw = old_review_path.read_bytes()
    _validate_approved_historical_review_bytes(old_review_raw, expected_sha256=_V7_CAPACITY_ADMISSION_REVIEW_SHA256, label="V7")
    old_review_sha = _write_immutable_snapshot(paths["old_review"], old_review_raw, "V7 source review")
    new_review_path = ROOT / request["source_review_path"]
    _regular_file(new_review_path, "V8 current source review")
    new_review_raw = new_review_path.read_bytes()
    if _digest_bytes(new_review_raw) != request["source_review_sha256"]:
        abort("capacity-Admission retry V8 source review changed while snapshotting")
    new_review_sha = _write_immutable_snapshot(paths["new_review"], new_review_raw, "V8 source review")
    map_value = live["compatible_native_admissions"]
    map_sha = _write_immutable_snapshot(paths["admissions_map"], _json_snapshot_bytes(map_value), "V7 Admission map")
    admission_entries: list[dict[str, Any]] = []
    for dataset in _COMPAT_DATASETS:
        ref = map_value[dataset]
        admission_path = ROOT / ref["path"]
        settings_path = admission_path.parent / "settings.json"
        admission_raw = admission_path.read_bytes()
        settings_raw = settings_path.read_bytes()
        admission_snapshot_path = paths["admissions"] / dataset / "admission.json"
        settings_snapshot_path = paths["admissions"] / dataset / "settings.json"
        admission_sha = _write_immutable_snapshot(admission_snapshot_path, admission_raw, f"V7 {dataset} Admission")
        settings_sha = _write_immutable_snapshot(settings_snapshot_path, settings_raw, f"V7 {dataset} settings")
        admission_entries.append({
            "dataset": dataset,
            "source_admission": {"path": ref["path"], "sha256": ref["sha256"]},
            "source_settings": {"path": settings_path.relative_to(ROOT).as_posix(), "sha256": ref["settings_sha256"]},
            "snapshot_admission": {"path": admission_snapshot_path.relative_to(ROOT).as_posix(), "sha256": admission_sha},
            "snapshot_settings": {"path": settings_snapshot_path.relative_to(ROOT).as_posix(), "sha256": settings_sha},
        })
    attempts = live["capacity_admission_attempts"]
    attempts_sha = _write_immutable_snapshot(paths["capacity_attempts"], _json_snapshot_bytes(attempts), "V7 capacity attempts")
    report_path = ROOT / attempts["report_path"]
    report_raw = report_path.read_bytes()
    report_sha = _write_immutable_snapshot(paths["capacity_report"], report_raw, "V7 current capacity report")
    budget_source_path = ROOT / _COMPAT_OUTPUT_ROOT / "ppi" / "native_prove_budget.json"
    budget_raw = budget_source_path.read_bytes()
    budget_sha = _write_immutable_snapshot(paths["native_prove_budget"], budget_raw, "V7 PPI native prove budget")
    if report_sha != _V7_CAPACITY_ADMISSION_REPORT_SHA256 or budget_sha != _V7_NATIVE_PROVE_BUDGET_SHA256:
        abort("capacity-Admission retry V7 report/prove-budget snapshot differs from its exact source anchor")
    record: dict[str, Any] = {
        "schema": CAPACITY_ADMISSION_RETRY_SCHEMA,
        "session_id": BASELINE_SESSION_ID,
        "failed_session_snapshot": {"path": paths["failed_session"].relative_to(ROOT).as_posix(), "sha256": failed_sha},
        "old_request_snapshot": {"path": paths["old_request"].relative_to(ROOT).as_posix(), "sha256": old_request_sha},
        "old_request_canonical_sha256": canonical_json_digest(old_request),
        "old_configured_admission_retry_snapshot": {"path": paths["old_retry_record"].relative_to(ROOT).as_posix(), "sha256": old_v7_record_sha},
        "old_configured_admission_retry_canonical_sha256": canonical_json_digest(v7_record),
        "old_source_review": {"path": _V7_CAPACITY_ADMISSION_REVIEW_PATH, "sha256": _V7_CAPACITY_ADMISSION_REVIEW_SHA256},
        "old_source_review_snapshot": {"path": paths["old_review"].relative_to(ROOT).as_posix(), "sha256": old_review_sha},
        "new_request_snapshot": {"path": paths["new_request"].relative_to(ROOT).as_posix(), "sha256": new_request_sha},
        "new_request_canonical_sha256": canonical_json_digest(request),
        "new_source_review": {"path": request["source_review_path"], "sha256": request["source_review_sha256"]},
        "new_source_review_snapshot": {"path": paths["new_review"].relative_to(ROOT).as_posix(), "sha256": new_review_sha},
        "original_session_snapshot": {"path": _snapshot_path(BASELINE_SESSION_ID), "sha256": original_sha},
        "capacity_checks_canonical_sha256": canonical_json_digest(live["capacity_checks"]),
        "prior_prover_identity_sha256": live["compatible_prover_identity_sha256"],
        "prior_capacity_source_identity_sha256": live["compatible_capacity_source_identity_sha256"],
        "admissions_map_snapshot": {"path": paths["admissions_map"].relative_to(ROOT).as_posix(), "sha256": map_sha},
        "admission_file_snapshots": admission_entries,
        "capacity_attempts_snapshot": {"path": paths["capacity_attempts"].relative_to(ROOT).as_posix(), "sha256": attempts_sha},
        "capacity_report_snapshot": {
            "source": {"path": attempts["report_path"], "sha256": report_sha},
            "snapshot": {"path": paths["capacity_report"].relative_to(ROOT).as_posix(), "sha256": report_sha},
        },
        "ppi_v7_native_prove_budget": {
            "source": {"path": f"{_COMPAT_OUTPUT_ROOT}/ppi/native_prove_budget.json", "sha256": budget_sha},
            "snapshot": {"path": paths["native_prove_budget"].relative_to(ROOT).as_posix(), "sha256": budget_sha},
        },
    }
    record_sha = _write_immutable_snapshot(paths["record"], _json_snapshot_bytes(record), "V8 retry record")
    if record_sha != _digest_bytes(_json_snapshot_bytes(record)):
        abort("capacity-Admission retry V8 record snapshot digest changed")
    candidate = copy.deepcopy(live)
    apply_compatible_capacity_admission_retry_commit(candidate, request, record)
    validate_compatible_resume_request(candidate, expected_session_id=BASELINE_SESSION_ID)
    return record


def apply_compatible_capacity_admission_retry_commit(
    session_document: dict[str, Any], request: dict[str, Any], record: dict[str, Any],
) -> None:
    """Commit the exact V7-to-V8 or pinned V8-receipt-failure V2 rebind."""
    if isinstance(record, dict) and record.get("schema") == CAPACITY_ADMISSION_RETRY_V2_SCHEMA:
        paths_v2 = _capacity_admission_retry_v2_paths(BASELINE_SESSION_ID)
        if session_document.get("compatible_capacity_admission_retry") == record and session_document.get("compatible_resume_request") == request:
            original, _ = validate_original_session_snapshot(
                ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
                expected_sha256=_ORIGINAL_SESSION_SHA256,
            )
            _validate_capacity_admission_v2_record(session_document, request, original)
            return
        failed_ref = _require_path_ref(
            record.get("failed_session_snapshot"), paths_v2["failed_session"].relative_to(ROOT).as_posix(), "V8 failed-session snapshot",
        )
        failed_raw = _snapshot_file(ROOT / failed_ref["path"], relative=failed_ref["path"], expected_digest=_V8_POSTFAILURE_SESSION_SHA256)
        failed = json.loads(failed_raw)
        if canonical_json_digest(session_document) != canonical_json_digest(failed):
            abort("capacity-Admission retry V2 session changed before its one-time commit")
        original, _previous_record = _validate_capacity_admission_v2_candidate(
            failed, failed_raw, request, allow_archived_sources=True,
        )
        candidate = copy.deepcopy(session_document)
        _apply_capacity_admission_v2_candidate(candidate, request, record, original)
        _validate_capacity_admission_v2_record(candidate, request, original)
        _migrate_capacity_admission_v2_files(record)
        _validate_no_proof_outputs()
        session_document.clear()
        session_document.update(candidate)
        return
    paths = _capacity_admission_retry_paths(BASELINE_SESSION_ID)
    failed, failed_raw = _load_retry_snapshot(record.get("failed_session_snapshot"), paths["failed_session"], "V7 failed-session")
    if _digest_bytes(failed_raw) != _V7_CAPACITY_ADMISSION_SESSION_SHA256:
        abort("capacity-Admission retry V7 failed-session snapshot changed before commit")
    if session_document.get("compatible_capacity_admission_retry") == record and session_document.get("compatible_resume_request") == request:
        original, _ = validate_original_session_snapshot(
            ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
            expected_sha256=_ORIGINAL_SESSION_SHA256,
        )
        _validate_capacity_admission_retry_record(session_document, request, original)
        return
    if canonical_json_digest(session_document) != canonical_json_digest(failed):
        abort("capacity-Admission retry session changed before the atomic V8 commit")
    validate_failed_capacity_admission_resume_gate(session_document)
    original, original_sha = validate_original_session_snapshot(
        ROOT / _snapshot_path(BASELINE_SESSION_ID), session_id=BASELINE_SESSION_ID,
        expected_sha256=_ORIGINAL_SESSION_SHA256,
    )
    _validate_v8_capacity_request(request, original)
    if request.get("old_session_snapshot_sha256") != original_sha:
        abort("capacity-Admission retry V8 request changed the original PPI session anchor")
    candidate = copy.deepcopy(session_document)
    candidate["compatible_resume_request"] = request
    candidate["compatible_capacity_admission_retry"] = record
    _validate_capacity_admission_retry_record(candidate, request, original)
    _validate_v7_capacity_failure(session_document, original, check_outputs=True)
    session_document["compatible_resume_request"] = request
    session_document["compatible_capacity_admission_retry"] = record
    session_document["capacity_checks"] = copy.deepcopy(original.get("capacity_checks"))
    for name in (
        "capacity_admission_attempts", "compatible_native_admissions", "compatible_prover_identity",
        "compatible_prover_identity_sha256", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256", "active_dataset", "dataset_failures",
    ):
        session_document.pop(name, None)


def _write_immutable_snapshot(path: Path, payload: bytes, label: str) -> str:
    try:
        path.relative_to(ROOT)
    except ValueError:
        abort(f"preflight retry {label} snapshot is outside the baseline")
    cursor = path.parent
    missing: list[Path] = []
    while cursor != ROOT:
        if cursor == cursor.parent or not cursor.is_relative_to(ROOT):
            abort(f"preflight retry {label} parent escapes the baseline")
        if cursor.is_symlink() or (cursor.exists() and not cursor.is_dir()):
            abort(f"preflight retry {label} parent is unsafe: {cursor}")
        if not cursor.exists():
            missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir()
    if path.is_symlink():
        abort(f"preflight retry {label} snapshot must not be a symlink")
    if path.exists():
        _regular_file(path, f"preflight retry {label} snapshot")
        existing = path.read_bytes()
        if existing != payload:
            abort(f"preflight retry {label} snapshot conflicts with the immutable prior attempt")
        return _digest_bytes(existing)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError:
        _regular_file(path, f"preflight retry {label} snapshot")
        existing = path.read_bytes()
        if existing != payload:
            abort(f"preflight retry {label} snapshot conflicts with the immutable prior attempt")
        return _digest_bytes(existing)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_parent(path.parent)
    return _digest_bytes(payload)


def _validate_failed_preflight_state(
    document: object,
    original: dict[str, Any],
    old_request: dict[str, Any],
    *,
    check_outputs: bool,
    allow_retry_record: bool,
) -> None:
    if not isinstance(document, dict):
        abort("preflight retry failed-session snapshot is malformed")
    allowed_compatible_fields = {
        "compatible_resume_request", "compatible_capacity_source_identity",
        "compatible_capacity_source_identity_sha256",
    }
    if allow_retry_record:
        allowed_compatible_fields.add("compatible_preflight_retry")
    unexpected_compatible = {
        key for key in document if key.startswith("compatible_") and key not in allowed_compatible_fields
    }
    capacity_identity = document.get("compatible_capacity_source_identity")
    capacity_identity_sha = document.get("compatible_capacity_source_identity_sha256")
    if not allow_retry_record and (
        not isinstance(capacity_identity, dict)
        or capacity_identity_sha != canonical_json_digest(capacity_identity)
    ):
        abort("preflight retry old capacity-source identity is missing or unbound")
    if allow_retry_record and ((capacity_identity is None) != (capacity_identity_sha is None)):
        abort("preflight retry current capacity-source identity is only partially recorded")
    if capacity_identity is not None and (
        not isinstance(capacity_identity, dict)
        or capacity_identity_sha != canonical_json_digest(capacity_identity)
    ):
        abort("preflight retry current capacity-source identity digest is invalid")
    checks = document.get("capacity_checks")
    original_checks = original.get("capacity_checks")
    if (
        document.get("schema_version") != 1
        or document.get("baseline") != "ezkl"
        or document.get("session_id") != BASELINE_SESSION_ID
        or document.get("status") != "FAILED"
        or document.get("stage") != "native_build_identity"
        or document.get("output_root") != _COMPAT_OUTPUT_ROOT
        or document.get("matrix") != _COMPAT_MATRIX
        or document.get("selected_datasets") != _COMPAT_DATASETS
        or document.get("completed_cells") not in (None, {})
        or not isinstance(document.get("resume_identity"), dict)
        or document["resume_identity"].get("schema") != "ezkl-formal-resume-identity-v1"
        or (
            document.get("compatible_resume_request") != old_request
            if not allow_retry_record
            else (
                not isinstance(document.get("compatible_resume_request"), dict)
                or document["compatible_resume_request"].get("session_id") != BASELINE_SESSION_ID
                or document["compatible_resume_request"].get("old_session_snapshot_path") != old_request.get("old_session_snapshot_path")
                or document["compatible_resume_request"].get("old_session_snapshot_sha256") != old_request.get("old_session_snapshot_sha256")
            )
        )
        or "compatible_preflight_retry" in document
        and not allow_retry_record
        or unexpected_compatible
        or not isinstance(checks, list)
        or not isinstance(original_checks, list)
        or len(original_checks) != 4
        or canonical_json_digest(checks) != canonical_json_digest(original_checks)
        or document.get("native_admissions") != original.get("native_admissions")
        or document.get("setup_reuse_receipts") != original.get("setup_reuse_receipts")
        or document.get("stage_receipts") != original.get("stage_receipts")
    ):
        abort("preflight retry is allowed only before any capacity, stage, admission, or result progress")
    if check_outputs:
        output_root = ROOT / _COMPAT_OUTPUT_ROOT
        for dataset in _COMPAT_DATASETS:
            output = output_root / dataset
            if output.is_symlink():
                abort(f"preflight retry {dataset} output is a symlink")
            for artifact in ("proof.json", "result.json", "native_verify_metrics.json", "core_run_manifest.json"):
                path = output / artifact
                if path.exists() or path.is_symlink():
                    abort(f"preflight retry refuses existing {dataset} proof/result artifact: {artifact}")
        compatibility = output_root / ".sessions" / BASELINE_SESSION_ID / "compatibility"
        for artifact in (
            "capacity_precheck_current.json", "capacity_precheck_ppi.json", "capacity_rechecks.json",
            "compatible_resume_receipt.json", "compatible_stage_receipts.json",
        ):
            path = compatibility / artifact
            if path.exists() or path.is_symlink():
                abort(f"preflight retry refuses existing compatible stage output: {artifact}")


def _validate_preflight_retry_record(
    session_document: dict[str, Any],
    request: dict[str, Any],
    original: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = session_document.get("compatible_preflight_retry")
    expected_keys = {
        "schema", "session_id", "old_request", "new_request_sha256", "failed_session_snapshot",
        "original_session_snapshot", "source_review", "prior_capacity_source_identity_sha256",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        abort("compatible preflight retry audit has unknown, missing, or malformed fields")
    if record.get("schema") != PREFLIGHT_RETRY_SCHEMA or record.get("session_id") != BASELINE_SESSION_ID:
        abort("compatible preflight retry audit is not bound to the exact session/schema")
    if record.get("new_request_sha256") != canonical_json_digest(request):
        abort("compatible preflight retry audit does not bind the current request")
    _require_sha(
        record.get("prior_capacity_source_identity_sha256"),
        "prior capacity-source identity digest",
    )
    retry_dir = _preflight_retry_directory(BASELINE_SESSION_ID)
    failed_path = retry_dir / "failed_session.json"
    old_request_path = retry_dir / "old_request.json"
    failed_ref = _require_path_ref(record.get("failed_session_snapshot"), failed_path.relative_to(ROOT).as_posix(), "failed-session snapshot")
    old_ref = record.get("old_request")
    if not isinstance(old_ref, dict) or set(old_ref) != {"path", "sha256", "canonical_sha256"}:
        abort("compatible preflight retry old-request snapshot reference is malformed")
    old_request_ref = _require_path_ref(
        {"path": old_ref.get("path"), "sha256": old_ref.get("sha256")},
        old_request_path.relative_to(ROOT).as_posix(),
        "old-request snapshot",
    )
    _require_sha(old_ref.get("canonical_sha256"), "old-request canonical digest")
    original_ref = _require_path_ref(
        record.get("original_session_snapshot"),
        request.get("old_session_snapshot_path"),
        "original-session snapshot",
    )
    if (
        original_ref["sha256"] != request.get("old_session_snapshot_sha256")
        or original_ref["path"] != _snapshot_path(BASELINE_SESSION_ID)
    ):
        abort("compatible preflight retry audit changed the shared original-session snapshot")
    source_ref = _require_path_ref(
        record.get("source_review"), request.get("source_review_path"), "current source review",
    )
    if source_ref["sha256"] != request.get("source_review_sha256"):
        abort("compatible preflight retry audit changed the approved current source review")

    old_raw = _snapshot_file(ROOT / old_request_ref["path"], relative=old_request_ref["path"], expected_digest=old_request_ref["sha256"])
    try:
        old_request_value = json.loads(old_raw)
    except json.JSONDecodeError as exc:
        abort(f"compatible preflight retry old request snapshot is invalid JSON: {exc}")
    old_request = validate_historical_preflight_request(old_request_value)
    if old_ref.get("canonical_sha256") != canonical_json_digest(old_request):
        abort("compatible preflight retry old request snapshot digest is inconsistent")

    failed_raw = _snapshot_file(ROOT / failed_ref["path"], relative=failed_ref["path"], expected_digest=failed_ref["sha256"])
    try:
        failed_document = json.loads(failed_raw)
    except json.JSONDecodeError as exc:
        abort(f"compatible preflight retry failed-session snapshot is invalid JSON: {exc}")
    if not isinstance(failed_document, dict) or failed_document.get("compatible_resume_request") != old_request:
        abort("compatible preflight retry failed-session snapshot does not preserve the old request")
    prior_capacity_identity = failed_document.get("compatible_capacity_source_identity")
    prior_capacity_identity_sha = failed_document.get("compatible_capacity_source_identity_sha256")
    if (
        not isinstance(prior_capacity_identity, dict)
        or prior_capacity_identity_sha != canonical_json_digest(prior_capacity_identity)
        or record.get("prior_capacity_source_identity_sha256") != prior_capacity_identity_sha
    ):
        abort("compatible preflight retry audit does not preserve the old capacity-source identity")
    current_capacity_identity = session_document.get("compatible_capacity_source_identity")
    current_capacity_identity_sha = session_document.get("compatible_capacity_source_identity_sha256")
    if current_capacity_identity is not None and (
        not isinstance(current_capacity_identity, dict)
        or current_capacity_identity_sha != canonical_json_digest(current_capacity_identity)
        or current_capacity_identity_sha == prior_capacity_identity_sha
    ):
        abort("compatible preflight retry reused or malformed the old capacity-source identity")
    frozen_original, frozen_digest = validate_original_session_snapshot(
        ROOT / original_ref["path"], session_id=BASELINE_SESSION_ID, expected_sha256=original_ref["sha256"],
    )
    if frozen_digest != request.get("old_session_snapshot_sha256") or frozen_original != original:
        abort("compatible preflight retry audit is not bound to the immutable original session")
    _validate_failed_preflight_state(
        failed_document, original, old_request, check_outputs=False, allow_retry_record=False,
    )
    return old_request, failed_document


def validate_failed_preflight_resume_gate(session_document: dict[str, Any]) -> None:
    """Allow detached retry only at the one audited native-build preflight gate."""
    if not isinstance(session_document, dict) or session_document.get("status") != "FAILED" or session_document.get("stage") != "native_build_identity":
        abort("compatible retry requires the exact FAILED/native_build_identity stage")
    request = session_document.get("compatible_resume_request")
    retry_record = session_document.get("compatible_preflight_retry")
    if retry_record is None:
        old_request = validate_historical_preflight_request(request)
        original, digest = validate_original_session_snapshot(
            ROOT / old_request["old_session_snapshot_path"],
            session_id=BASELINE_SESSION_ID,
            expected_sha256=old_request["old_session_snapshot_sha256"],
        )
        if digest != old_request["old_session_snapshot_sha256"]:
            abort("preflight retry original-session snapshot digest changed")
    else:
        current_request, original = validate_compatible_resume_request(
            session_document, expected_session_id=BASELINE_SESSION_ID,
        )
        old_request, _failed_snapshot = _validate_preflight_retry_record(session_document, current_request, original)
        request = current_request
    _validate_failed_preflight_state(
        session_document, original, old_request, check_outputs=True, allow_retry_record=retry_record is not None,
    )


def prepare_compatible_preflight_retry(session_file: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    """Freeze and audit the one old-v3 to current-review request transition.

    Called only by ``begin_formal_session`` while it owns the baseline flock.
    It creates immutable snapshots but leaves the live session write to that
    caller's atomic session-record commit.
    """
    expected_session = ROOT / _COMPAT_OUTPUT_ROOT / ".sessions" / f"{BASELINE_SESSION_ID}.json"
    if session_file.is_symlink() or not session_file.is_file() or session_file.resolve(strict=True) != expected_session.resolve(strict=True):
        abort("compatible preflight retry must use the exact session record")
    try:
        live_raw = session_file.read_bytes()
        live_document = json.loads(live_raw)
    except (OSError, json.JSONDecodeError) as exc:
        abort(f"cannot read compatible preflight retry session: {exc}")
    if not isinstance(live_document, dict):
        abort("compatible preflight retry session is malformed")
    existing = live_document.get("compatible_resume_request")
    audit = live_document.get("compatible_preflight_retry")
    if existing is None:
        if audit is not None:
            abort("compatible preflight retry audit exists without its current request")
        return None
    if existing == request:
        if audit is not None:
            validate_compatible_resume_request(live_document, expected_session_id=BASELINE_SESSION_ID)
            return None
        return None
    if audit is not None:
        # The sole second transition is handled by the dedicated Admission
        # preflight audit below; this first-transition helper must leave it alone.
        return None
    old_request = validate_historical_preflight_request(existing)
    if live_document.get("status") != "FAILED" or live_document.get("stage") != "native_build_identity":
        abort("compatible preflight request may migrate only from FAILED/native_build_identity")
    original, original_digest = validate_original_session_snapshot(
        ROOT / old_request["old_session_snapshot_path"],
        session_id=BASELINE_SESSION_ID,
        expected_sha256=old_request["old_session_snapshot_sha256"],
    )
    if original_digest != old_request["old_session_snapshot_sha256"]:
        abort("compatible preflight retry original-session snapshot changed")
    _validate_failed_preflight_state(
        live_document, original, old_request, check_outputs=True, allow_retry_record=False,
    )
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("session_id") != BASELINE_SESSION_ID
        or request.get("old_session_snapshot_path") != old_request.get("old_session_snapshot_path")
        or request.get("old_session_snapshot_sha256") != old_request.get("old_session_snapshot_sha256")
        or canonical_json_digest(request) == canonical_json_digest(old_request)
    ):
        abort("compatible preflight retry current request is not a distinct request for the same original session")

    retry_dir = _preflight_retry_directory(BASELINE_SESSION_ID)
    failed_path = retry_dir / "failed_session.json"
    old_request_path = retry_dir / "old_request.json"
    failed_sha = _write_immutable_snapshot(failed_path, live_raw, "failed-session")
    old_request_bytes = (json.dumps(existing, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    old_raw_sha = _write_immutable_snapshot(old_request_path, old_request_bytes, "old-request")
    record: dict[str, Any] = {
        "schema": PREFLIGHT_RETRY_SCHEMA,
        "session_id": BASELINE_SESSION_ID,
        "old_request": {
            "path": old_request_path.relative_to(ROOT).as_posix(),
            "sha256": old_raw_sha,
            "canonical_sha256": canonical_json_digest(existing),
        },
        "new_request_sha256": canonical_json_digest(request),
        "failed_session_snapshot": {
            "path": failed_path.relative_to(ROOT).as_posix(),
            "sha256": failed_sha,
        },
        "original_session_snapshot": {
            "path": old_request["old_session_snapshot_path"],
            "sha256": old_request["old_session_snapshot_sha256"],
        },
        "source_review": {
            "path": request.get("source_review_path"),
            "sha256": request.get("source_review_sha256"),
        },
        "prior_capacity_source_identity_sha256": live_document.get(
            "compatible_capacity_source_identity_sha256"
        ),
    }
    candidate = copy.deepcopy(live_document)
    candidate["compatible_resume_request"] = request
    candidate["compatible_preflight_retry"] = record
    candidate.pop("compatible_capacity_source_identity", None)
    candidate.pop("compatible_capacity_source_identity_sha256", None)
    validate_compatible_resume_request(candidate, expected_session_id=BASELINE_SESSION_ID)
    return record


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
    retry_record = session_document.get("compatible_preflight_retry")
    admission_retry_record = session_document.get("compatible_admission_preflight_retry")
    configured_admission_retry_record = session_document.get("compatible_configured_admission_retry")
    capacity_admission_retry_record = session_document.get("compatible_capacity_admission_retry")
    if (
        canonical_json_digest(request) != _PRE_RETRY_REQUEST_SHA256
        and retry_record is None
    ):
        abort("current compatible resume request lacks its one-time old-request migration audit")
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
    recovery_record = session_document.get("compatible_interrupted_before_prove_recovery")
    if (
        recovery_record is not None
        and canonical_json_digest(request) != _V6_CONFIGURED_ADMISSION_REQUEST_SHA256
        and configured_admission_retry_record is None
    ):
        abort("compatible request advanced beyond the pinned V6 request without the configured-Admission retry audit")
    if capacity_admission_retry_record is not None:
        retry_schema = capacity_admission_retry_record.get("schema") if isinstance(capacity_admission_retry_record, dict) else None
        if retry_schema == CAPACITY_ADMISSION_RETRY_V2_SCHEMA:
            _validate_capacity_admission_v2_record(session_document, request, old_session)
        else:
            _validate_capacity_admission_retry_record(session_document, request, old_session)
    elif configured_admission_retry_record is not None:
        _validate_configured_admission_retry_record(session_document, request, old_session)
    elif recovery_record is not None:
        _validate_interrupted_before_prove_recovery_record(session_document, request, old_session)
    elif admission_retry_record is not None:
        _validate_admission_preflight_retry_record(session_document, request, old_session)
    elif retry_record is not None:
        _validate_preflight_retry_record(session_document, request, old_session)
    return request, old_session


def _settings_semantics_digest(settings_path: Path, *, expected_sha256: object, dataset: str, label: str) -> str:
    """Authenticate a settings file and return its exact semantics excluding GraphSettings' clock field."""
    if not isinstance(settings_path, Path):
        abort(f"{dataset} {label} settings path is malformed")
    try:
        mode = settings_path.lstat().st_mode
    except OSError as exc:
        abort(f"{dataset} {label} settings file cannot be inspected: {exc}")
    if not stat.S_ISREG(mode):
        abort(f"{dataset} {label} settings file must be a regular non-symlink file: {settings_path}")
    expected = _require_sha(expected_sha256, f"{label} Admission settings digest")
    try:
        raw = settings_path.read_bytes()

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON key: {key}")
                value[key] = item
            return value

        settings = json.loads(raw, object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        abort(f"{dataset} {label} settings file is not unambiguous JSON: {exc}")
    if _digest_bytes(raw) != expected:
        abort(f"{dataset} {label} settings bytes differ from their Admission settings_sha256")
    if not isinstance(settings, dict) or "timestamp" not in settings:
        abort(f"{dataset} {label} settings must contain the exact top-level GraphSettings timestamp")
    timestamp = settings.get("timestamp")
    if type(timestamp) is not int or timestamp < 0 or timestamp > (1 << 63) - 1:
        abort(f"{dataset} {label} settings timestamp is not a nonnegative i64")
    settings.pop("timestamp")
    return canonical_json_digest(settings)


def compare_admission_semantics(
    old_admission: dict[str, Any],
    current_admission: dict[str, Any],
    *,
    dataset: str,
    old_settings_path: Path,
    current_settings_path: Path,
) -> str:
    """Compare authenticated Admissions, allowing only the GPU sample and native settings timestamp to vary.

    The current settings digest is normalized to the old Admission's digest only
    after both files have independently matched their Admission anchors and
    their parsed settings differ solely in the top-level GraphSettings timestamp.
    The returned digest therefore preserves the historical old-Admission digest
    contract used by the compatibility receipt.
    """
    if not isinstance(old_admission, dict) or not isinstance(current_admission, dict):
        abort(f"{dataset} compatible Admission is not a JSON object")
    old_settings_semantics = _settings_semantics_digest(
        old_settings_path,
        expected_sha256=old_admission.get("settings_sha256"),
        dataset=dataset,
        label="original",
    )
    current_settings_semantics = _settings_semantics_digest(
        current_settings_path,
        expected_sha256=current_admission.get("settings_sha256"),
        dataset=dataset,
        label="current",
    )
    if old_settings_semantics != current_settings_semantics:
        abort(f"{dataset} compatible settings differ outside the authenticated top-level timestamp")
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
    old_settings_sha256 = _require_sha(old_projection.get("settings_sha256"), f"{dataset} original Admission settings digest")
    _require_sha(current_projection.get("settings_sha256"), f"{dataset} current Admission settings digest")
    # The settings-file authentication and timestamp-only comparison above are
    # the sole authorization for this digest normalization.
    current_projection["settings_sha256"] = old_settings_sha256
    if canonical_json_digest(old_projection) != canonical_json_digest(current_projection):
        old_keys = set(old_projection)
        current_keys = set(current_projection)
        changed = sorted(key for key in old_keys | current_keys if old_projection.get(key) != current_projection.get(key))
        abort(f"{dataset} compatible Admission semantic projection differs outside the approved free-memory sample: {', '.join(changed)}")
    return admission_semantics_digest(old_admission, dataset=dataset)


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
    records = _artifact_manifest_file_records(files, "original PPI artifact manifest")
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

    # Bind the manifest projection to the exact small budget bytes and the
    # current executable/build identity before any native Prove can start.
    prover_receipts = authoritative_receipt.get("current_prover_receipts")
    budget_ref = prover_receipts.get("native_prove_budget") if isinstance(prover_receipts, dict) else None
    if (
        not isinstance(prover_receipts, dict)
        or not isinstance(budget_ref, dict)
        or budget_ref.get("path") != f"{_COMPAT_OUTPUT_ROOT}/ppi/native_prove_budget.json"
        or budget_ref.get("sha256") != budget_sha256
    ):
        abort("compatible receipt prove-budget reference differs from the current native prove budget bytes")
    for receipt_key, identity_key in (
        ("binary_sha256", "binary_sha256"),
        ("build_receipt_sha256", "build_receipt_sha256"),
        ("build_environment_sha256", "build_environment_sha256"),
    ):
        if prover_receipts.get(receipt_key) != current_identity.get(identity_key):
            abort(f"compatible receipt {receipt_key} differs from the current Prover identity")

    expected = copy.deepcopy(original_manifest)
    expected["binary_sha256"] = current_identity["binary_sha256"]
    expected["source_tree_sha256"] = current_identity["source_tree_sha256"]
    files = expected.get("files")
    if not isinstance(files, list):
        abort("original compatible artifact manifest file census is malformed")
    budget_matches = [index for index, record in enumerate(files) if isinstance(record, dict) and record.get("path") == "native_prove_budget.json"]
    receipt_matches = [record for record in files if isinstance(record, dict) and record.get("path") == "compatible_resume_receipt.json"]
    if len(budget_matches) != 1 or receipt_matches:
        abort("original artifact manifest has an unexpected prove-budget or compatibility-receipt census")
    files[budget_matches[0]] = {
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
    original_session, original_session_sha256 = validate_original_session_snapshot(
        ROOT / snapshot["path"],
        session_id=expected_session,
        expected_sha256=snapshot["sha256"],
    )
    original_manifest, _original_manifest_sha256, original_manifest_ref = validate_original_artifact_manifest(
        original_session,
        session_id=expected_session,
    )
    if shared["original_artifact_manifest"] != original_manifest_ref:
        abort("compatible receipt artifact-manifest reference differs from the authenticated original manifest")
    if shared["statement_sha256"] != original_manifest.get("statement_sha256"):
        abort("compatible receipt statement digest differs from the authenticated original artifact manifest")
    if shared["public_output_sha256"] != artifact_manifest_file_sha256(original_manifest, "public_output.json"):
        abort("compatible receipt public-output digest differs from the authenticated original artifact manifest")
    old_admission_ref = original_session.get("native_admissions", {}).get("ppi")
    bound_admission_ref = value["bound_setup_admission"]
    current_admission_ref = value["current_capacity_admission"]
    if (
        not isinstance(old_admission_ref, dict)
        or {name: old_admission_ref.get(name) for name in ("path", "sha256", "settings_sha256")}
        != bound_admission_ref
        or original_session_sha256 != snapshot["sha256"]
    ):
        abort("compatible receipt bound Admission differs from the exact original-session anchor")
    old_admission_path = ROOT / bound_admission_ref["path"]
    current_admission_path = ROOT / current_admission_ref["path"]
    old_settings_path = old_admission_path.parent / "settings.json"
    current_settings_path = current_admission_path.parent / "settings.json"
    for file_path, expected_digest, label in (
        (old_admission_path, bound_admission_ref["sha256"], "original Admission"),
        (current_admission_path, current_admission_ref["sha256"], "current capacity Admission"),
        (old_settings_path, bound_admission_ref["settings_sha256"], "original settings"),
        (current_settings_path, current_admission_ref["settings_sha256"], "current capacity settings"),
    ):
        _regular_file(file_path, f"compatible receipt {label}")
        if _digest_bytes(file_path.read_bytes()) != expected_digest:
            abort(f"compatible receipt {label} bytes differ from their bound digest")
    old_admission = load_json(old_admission_path, "compatible receipt original PPI Admission")
    current_admission = load_json(current_admission_path, "compatible receipt current PPI Admission")
    if (
        not isinstance(old_admission, dict)
        or not isinstance(current_admission, dict)
        or old_admission.get("settings_sha256") != bound_admission_ref["settings_sha256"]
        or current_admission.get("settings_sha256") != current_admission_ref["settings_sha256"]
        or old_admission.get("dataset") != "ppi"
        or current_admission.get("dataset") != "ppi"
    ):
        abort("compatible receipt Admissions do not match their respective settings anchors")
    admission_semantics_sha256 = compare_admission_semantics(
        old_admission,
        current_admission,
        dataset="ppi",
        old_settings_path=old_settings_path,
        current_settings_path=current_settings_path,
    )
    if admission_semantics_sha256 != value["admission_semantics_sha256"]:
        abort("compatible receipt Admission semantic digest is not the old Setup-bound semantic digest")
    if (
        setup_chain["setup_files"].get("settings_sha256") != value["bound_setup_admission"].get("settings_sha256")
        or shared.get("settings_sha256") != value["bound_setup_admission"].get("settings_sha256")
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
