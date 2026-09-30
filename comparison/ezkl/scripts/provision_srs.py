#!/usr/bin/env python3
"""Provision the baseline-local official ezkl KZG SRS through native ``get-srs``.

The production SRS path is deliberately owned by the pinned ezkl binary.  This
script selects the largest public logrows/hash pair exposed by the current
source, asks that binary to download and authenticate it, rechecks the result,
and atomically installs one shared cache for all formal datasets.  The
``gen-srs`` command and operator-supplied files are never accepted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
SRS_ROOT = ROOT / "resources" / "native_srs"
MANIFEST = SRS_ROOT / "srs_manifest.json"
SRS_SOURCE = ROOT / "src" / "pfsys" / "srs.rs"
HASH_SOURCE = ROOT / "src" / "srs_sha.rs"
SCHEMA_VERSION = 2
MIN_FORMAL_SRS_BYTES = 4096
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HASH_ROW = re.compile(r"\(\s*(\d+)\s*,\s*\"([0-9a-f]{64})\"\s*,")
URL_ROW = re.compile(r'pub const PUBLIC_SRS_URL:\s*&str\s*=\s*"([^"]+)"\s*;')


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SRS provisioning failed: {message}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_contract() -> tuple[str, int, str]:
    """Read the URL and maximum authenticated k from the pinned Rust source."""

    if SRS_SOURCE.is_symlink() or not SRS_SOURCE.is_file():
        fail(f"missing native SRS source: {SRS_SOURCE}")
    if HASH_SOURCE.is_symlink() or not HASH_SOURCE.is_file():
        fail(f"missing native SRS hash source: {HASH_SOURCE}")
    url_match = URL_ROW.search(SRS_SOURCE.read_text())
    if url_match is None or not url_match.group(1).startswith("https://"):
        fail("native source has no authenticated HTTPS PUBLIC_SRS_URL")
    hashes = {int(k): value for k, value in HASH_ROW.findall(HASH_SOURCE.read_text())}
    if not hashes or set(hashes) != set(range(1, max(hashes) + 1)):
        fail("native PUBLIC_SRS_SHA256_HASHES is missing a contiguous public range")
    maximum = max(hashes)
    expected = hashes[maximum]
    if not HEX64.fullmatch(expected):
        fail("native maximum SRS hash is not a lowercase SHA-256 digest")
    return url_match.group(1), maximum, expected


def local_binary(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail(f"binary must be baseline-local: {value}")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"binary escapes the ezkl baseline: {value}")
    if resolved.is_symlink() or not resolved.is_file() or not os.access(resolved, os.X_OK):
        fail(f"official ezkl binary is missing or not executable: {resolved}")
    return resolved


def formal_srs_path(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    lowered = "/".join(path.parts).lower()
    if any(marker in lowered for marker in ("dummy", "gen-srs", "reduced", "placeholder", "fake", "mock", "test")):
        fail(f"{label} has a forbidden generated/test path marker: {path}")
    size = path.stat().st_size
    if size < MIN_FORMAL_SRS_BYTES:
        fail(f"{label} is too small ({size} bytes; minimum {MIN_FORMAL_SRS_BYTES})")
    return path


def native_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        fail(f"native SRS path escapes baseline: {path}")


def native_get_srs(binary: Path, target: Path, logrows: int) -> tuple[bool, str]:
    command = [
        str(binary),
        "get-srs",
        "--srs-path",
        native_path(target),
        "--logrows",
        str(logrows),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.strip()
    if output:
        print(output)
    return completed.returncode == 0, output


def validate_native_file(path: Path, expected: str) -> int:
    formal_srs_path(path, "native official SRS")
    actual = digest(path)
    if actual != expected:
        fail(f"native official SRS digest mismatch: expected {expected}, got {actual}")
    return path.stat().st_size


def install_shared_srs(binary: Path, logrows: int, expected: str) -> tuple[Path, int]:
    SRS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = SRS_ROOT / f"kzg{logrows}.srs"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        fail(f"shared native SRS destination is not a regular file: {destination}")

    if destination.exists():
        valid, _ = native_get_srs(binary, destination, logrows)
        if valid:
            size = validate_native_file(destination, expected)
            return destination, size

    fd, temporary_name = tempfile.mkstemp(prefix=f".kzg{logrows}.", suffix=".native-get-srs", dir=SRS_ROOT)
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        valid, _ = native_get_srs(binary, temporary, logrows)
        if not valid:
            fail("native get-srs failed; official SRS was not installed")
        size = validate_native_file(temporary, expected)
        os.replace(temporary, destination)
        size = validate_native_file(destination, expected)
        return destination, size
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def config_path(dataset: str) -> Path:
    path = ROOT / "resources" / "canonical" / "configs" / f"{dataset}_secure.cfg"
    if path.is_symlink() or not path.is_file():
        fail(f"canonical secure config is missing or symlinked: {path}")
    return path


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            fail(f"malformed config line {path}:{number}")
        key, value = (piece.strip() for piece in line.split("=", 1))
        if not key or key in values:
            fail(f"duplicate/empty config key {path}:{number}")
        values[key] = value
    return values


def updated_config(path: Path, *, srs_id: str, srs_path: str, sha256_value: str) -> str:
    updates = {
        "srs_id": srs_id,
        "srs_path": srs_path,
        "trusted_srs_sha256": sha256_value,
    }
    lines = path.read_text().splitlines()
    output: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        key = raw.split("#", 1)[0].strip().split("=", 1)[0].strip() if "=" in raw else ""
        if key in updates:
            output.append(f"{key} = {updates[key]}")
            seen.add(key)
        else:
            output.append(raw)
    if seen != set(updates):
        fail(f"{path} must contain srs_id, srs_path and trusted_srs_sha256")
    return "\n".join(output) + "\n"


def update_configs(*, srs_id: str, srs_path: str, sha256_value: str) -> None:
    paths = [config_path(dataset) for dataset in DATASETS]
    payloads = [updated_config(path, srs_id=srs_id, srs_path=srs_path, sha256_value=sha256_value) for path in paths]
    temporary: list[Path] = []
    try:
        for path, payload in zip(paths, payloads):
            candidate = path.with_name(f".{path.name}.native-srs.tmp")
            if candidate.exists() or candidate.is_symlink():
                fail(f"temporary config path already exists: {candidate}")
            candidate.write_text(payload)
            temporary.append(candidate)
        for path, candidate in zip(paths, temporary):
            os.replace(candidate, path)
    finally:
        for candidate in temporary:
            if candidate.exists():
                candidate.unlink()


def refresh_resource_identity() -> None:
    """Re-authenticate config edits and keep the already-built CPU receipt exact."""

    validator = ROOT / "scripts" / "validate_resources.py"
    completed = subprocess.run(
        [sys.executable, str(validator), "--write-manifest", "--require-complete"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"resource manifest refresh failed after SRS binding: {completed.stdout.strip()}")
    receipt_path = ROOT / "artifacts" / "build" / "cpu" / "build_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        fail(f"CPU build receipt is missing after native SRS provisioning: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"CPU build receipt is invalid JSON: {exc}")
    if not isinstance(receipt, dict):
        fail("CPU build receipt must be a JSON object")
    manifest_path = ROOT / "resources" / "resource_manifest.json"
    receipt["resource_manifest_sha256"] = digest(manifest_path)
    atomic_write(receipt_path, receipt)


def main(args: argparse.Namespace) -> int:
    if any(flag in sys.argv[1:] for flag in ("--source", "--sha256", "--dataset", "--replace", "gen-srs")):
        fail("manual SRS source/hash and dummy gen-srs inputs are forbidden")
    binary = local_binary(args.binary)
    source_url, logrows, expected = source_contract()
    destination, size = install_shared_srs(binary, logrows, expected)
    actual = digest(destination)
    relative = destination.relative_to(ROOT).as_posix()
    srs_id = f"official_ezkl_native_shared_kzg{logrows}_srs"
    source_object = {
        "source": "official_ezkl_ceremony",
        "provisioner": "native_get_srs",
        "source_url": f"{source_url}{logrows}.srs",
        "hash_source": "src/srs_sha.rs",
        "native_hash_verified": True,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "baseline": "ezkl",
        "srs_type": "halo2-kzg-shplonk",
        "shared": True,
        "max_logrows": logrows,
        **source_object,
        "entries": {
            dataset: {
                "dataset": dataset,
                "srs_id": srs_id,
                "srs_type": "trusted_srs",
                "path": relative,
                "sha256": actual,
                "size": size,
                "logrows": logrows,
                "source": source_object["source"],
                "provisioner": source_object["provisioner"],
                "source_url": source_object["source_url"],
                "hash_source": source_object["hash_source"],
                "native_hash_verified": True,
                "trusted": True,
                "reduced": False,
                "generated": False,
            }
            for dataset in DATASETS
        },
    }
    for dataset in DATASETS:
        values = parse_config(config_path(dataset))
        if values.get("srs_type") != "trusted_srs":
            fail(f"{dataset} config is not a trusted_srs config")
    atomic_write(MANIFEST, manifest)
    update_configs(srs_id=srs_id, srs_path=relative, sha256_value=actual)
    refresh_resource_identity()
    print(f"native official SRS ready: {relative}")
    print(f"native get-srs source: {source_object['source_url']}")
    print(f"logrows={logrows} sha256={actual} size_bytes={size}")
    print(f"shared datasets: {','.join(DATASETS)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", default="target/release/ezkl", help="current baseline-local ezkl binary")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        fail(str(exc))
