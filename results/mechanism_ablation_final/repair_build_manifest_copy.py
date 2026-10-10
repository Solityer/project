#!/usr/bin/env python3
"""Restore the formal session's byte-identical copy of its declared build manifest."""
import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--build-manifest", required=True, type=Path)
    args = ap.parse_args()
    session = args.session.resolve(strict=True)
    source = args.build_manifest.resolve(strict=True)
    destination = session / "build_source_manifest.json"
    summary_path = session / "summary.json"
    result_manifest_path = session / "results_manifest.json"
    if destination.is_symlink() or source.is_symlink() or not destination.is_file() or not source.is_file():
        raise SystemExit("manifest source/destination must be regular non-symlink files")
    summary = json.loads(summary_path.read_text())
    result_manifest = json.loads(result_manifest_path.read_text())
    before = destination.read_bytes()
    source_bytes = source.read_bytes()
    old_sha, source_sha = sha(before), sha(source_bytes)
    expected_sha = summary["provenance"]["build"]["source_manifest_sha256"]
    before_json, source_json = json.loads(before), json.loads(source_bytes)
    if before_json != source_json:
        raise SystemExit("session/build source manifests are not semantically equal; refusing replacement")
    if source_sha != expected_sha:
        raise SystemExit("build manifest raw SHA does not match the predeclared summary SHA")
    if old_sha == expected_sha:
        raise SystemExit("session manifest already has the declared raw SHA; no repair is needed")
    if old_sha in json.dumps(summary) or old_sha in json.dumps(result_manifest):
        raise SystemExit("old pretty-copy SHA is bound by summary/results manifest; refusing replacement")
    # results_manifest intentionally publishes only the paper tables, cells.csv, summary.json and CUDA trace.
    published_paths = {entry.get("path") for entry in result_manifest.get("files", {}).values()}
    if str(destination) in published_paths or "build_source_manifest.json" in published_paths:
        raise SystemExit("build manifest copy is listed in results_manifest; refusing replacement")
    mode = stat.S_IMODE(destination.stat().st_mode)
    fd, temp_name = tempfile.mkstemp(prefix=".build_source_manifest.json.", dir=session)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(source_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, destination)
        dirfd = os.open(session, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    after = destination.read_bytes()
    new_sha = sha(after)
    if new_sha != expected_sha or after != source_bytes or json.loads(after) != source_json:
        raise SystemExit("post-replacement byte/hash/semantic verification failed")
    out = session.parent
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path = out / f"build_manifest_copy_repair_{timestamp}.json"
    record = {
        "format": "zkgat_mechanism_build_manifest_copy_repair_v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": summary["session_id"],
        "operation": "atomically replace only session build_source_manifest.json with original build manifest bytes",
        "command": f"python3 {Path(__file__).resolve()} --session {session} --build-manifest {source}",
        "session_destination": str(destination),
        "build_source": str(source),
        "old_session_sha256": old_sha,
        "new_session_sha256": new_sha,
        "build_source_sha256": source_sha,
        "summary_predeclared_source_manifest_sha256": expected_sha,
        "old_session_bytes": len(before),
        "new_session_bytes": len(after),
        "json_semantically_equal_before": True,
        "byte_identical_to_build_after": after == source_bytes,
        "json_semantically_equal_after": json.loads(after) == source_json,
        "build_manifest_listed_in_results_manifest": False,
        "old_pretty_sha_referenced_by_summary_or_results_manifest": False,
        "summary_and_results_manifest_unchanged": True,
    }
    atomic_json(record_path, record)
    print(json.dumps({"record": str(record_path), **record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
