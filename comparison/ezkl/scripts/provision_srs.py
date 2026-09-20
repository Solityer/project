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
# The checked-in 516-byte kzg1 test fixture is a k=1 parameter file and cannot
# support any of the full GAT circuits.  Keep a conservative lower bound here
# so a tiny test/placeholder file is rejected before it reaches setup.
MIN_FORMAL_SRS_BYTES = 4096
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
    size = path.stat().st_size
    if size < MIN_FORMAL_SRS_BYTES:
        fail(
            f"source is too small for a full formal Halo2 KZG SRS: {size} bytes "
            f"(minimum {MIN_FORMAL_SRS_BYTES})"
        )
    return path


def reject_destination(path: Path, *, replace: bool) -> None:
    if path.is_symlink():
        fail(f"destination must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        fail(f"destination must be a regular file: {path}")
    if path.exists() and not replace:
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


def update_config_digest(dataset: str, sha256_value: str) -> None:
    """Bind the provisioned SRS digest into the dataset's formal config."""

    config = ROOT / "resources" / "canonical" / "configs" / f"{dataset}_secure.cfg"
    if config.is_symlink() or not config.is_file():
        fail(f"canonical secure config is missing or symlinked: {config}")
    lines = config.read_text().splitlines()
    output: list[str] = []
    seen = False
    for raw in lines:
        key = raw.split("#", 1)[0].strip().split("=", 1)[0].strip() if "=" in raw else ""
        if key == "trusted_srs_sha256":
            output.append(f"trusted_srs_sha256 = {sha256_value}")
            seen = True
        else:
            output.append(raw)
    if not seen:
        fail(f"{config} has no trusted_srs_sha256 field")
    temporary = config.with_name(f".{config.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        fail(f"temporary config path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(output) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config)
    finally:
        if temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--source", required=True, help="operator-supplied authentic official ezkl SRS")
    result.add_argument("--sha256", required=True, help="expected lowercase SHA-256 of --source")
    result.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing baseline-local SRS after authenticating the new source",
    )
    return result


def main(args: argparse.Namespace) -> int:
    if not HEX64.fullmatch(args.sha256):
        fail("--sha256 must be a lowercase SHA-256 digest")
    source = reject_source(Path(args.source))
    actual = digest(source)
    if actual != args.sha256:
        fail(f"source SHA-256 mismatch: expected {args.sha256}, got {actual}")

    destination = SRS_ROOT / f"{args.dataset}_formal_srs"
    if source.resolve() == destination.resolve():
        fail("source must be outside the baseline-local destination")
    reject_destination(destination, replace=args.replace)
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
    update_config_digest(args.dataset, actual)
    print(f"staged {args.dataset} official ezkl SRS: {destination.relative_to(ROOT)}")
    print(f"sha256={actual}")
    print(f"size_bytes={destination.stat().st_size}")
    print(f"config: trusted_srs_sha256 = {actual}")
    print(f"config: srs_path = {destination.relative_to(ROOT).as_posix()}")
    print(f"runner: --srs {destination.relative_to(ROOT).as_posix()} --srs-sha256 {actual}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except OSError as exc:
        fail(str(exc))
