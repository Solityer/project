#!/usr/bin/env python3
"""Stage one operator-authenticated official ezkl SRS into baseline-local storage.

This helper only copies an already supplied authentic SRS.  It never invokes an
SRS generator, downloads material, or invents a digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SRS_ROOT = ROOT / "resources" / "native_srs"
MANIFEST = SRS_ROOT / "srs_manifest.json"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"SRS provisioning failed: {message}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reject_source(path: Path) -> Path:
    if path.is_symlink():
        fail(f"source must not be a symlink: {path}")
    lowered = {part.lower() for part in path.parts}
    if "tests" in lowered or "assets" in lowered:
        fail(f"source must not be under tests/assets: {path}")
    if not path.is_file():
        fail(f"source must be a regular file: {path}")
    return path


def reject_destination(path: Path) -> None:
    if path.is_symlink():
        fail(f"destination must not be a symlink: {path}")
    if path.exists():
        fail(f"destination already exists; refusing overwrite: {path}")


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"schema_version": 1, "baseline": "ezkl", "entries": {}}
    if MANIFEST.is_symlink() or not MANIFEST.is_file():
        fail(f"SRS manifest must be a regular non-symlink file: {MANIFEST}")
    try:
        value = json.loads(MANIFEST.read_text())
    except json.JSONDecodeError as exc:
        fail(f"invalid SRS manifest JSON: {exc}")
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("baseline") != "ezkl":
        fail("SRS manifest has an unsupported schema")
    entries = value.get("entries")
    if not isinstance(entries, dict):
        fail("SRS manifest entries must be an object")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--source", required=True, help="operator-supplied authentic official ezkl SRS")
    result.add_argument("--sha256", required=True, help="expected lowercase SHA-256 of --source")
    return result


def main(args: argparse.Namespace) -> int:
    if not HEX64.fullmatch(args.sha256):
        fail("--sha256 must be a lowercase SHA-256 digest")
    source = reject_source(Path(args.source))
    actual = digest(source)
    if actual != args.sha256:
        fail(f"source SHA-256 mismatch: expected {args.sha256}, got {actual}")

    destination = SRS_ROOT / f"{args.dataset}_formal_srs"
    reject_destination(destination)
    SRS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        fail(f"temporary destination already exists: {temporary}")
    try:
        with source.open("rb") as source_stream, temporary.open("xb") as destination_stream:
            while True:
                block = source_stream.read(1024 * 1024)
                if not block:
                    break
                destination_stream.write(block)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = load_manifest()
    entries = manifest["entries"]
    entries[args.dataset] = {
        "dataset": args.dataset,
        "srs_id": f"official_ezkl_native_{args.dataset}_srs",
        "srs_type": "trusted_srs",
        "path": destination.relative_to(ROOT).as_posix(),
        "sha256": actual,
        "size": destination.stat().st_size,
        "source_path": str(source.absolute()),
        "source_sha256": actual,
        "trusted": True,
        "reduced": False,
        "generated": False,
    }
    manifest["entries"] = entries
    atomic_write(MANIFEST, manifest)
    print(f"staged {args.dataset} official ezkl SRS: {destination.relative_to(ROOT)}")
    print(f"sha256={actual}")
    print(f"config: trusted_srs_sha256 = {actual}")
    print(f"config: srs_path = {destination.relative_to(ROOT).as_posix()}")
    print(f"runner: --srs {destination.relative_to(ROOT).as_posix()} --srs-sha256 {actual}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except OSError as exc:
        fail(str(exc))
