#!/usr/bin/env python3
"""Provision the baseline-local official ezkl KZG SRS from its source contract.

The pinned ezkl source owns the official URL and SHA-256 pair.  This script
downloads that SRS with resumable standard-library HTTP, authenticates it, and
atomically installs one shared cache for all formal datasets.  The ``gen-srs``
command and operator-supplied files are never accepted here.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
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
CONTENT_RANGE_ROW = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
UNSATISFIED_RANGE_ROW = re.compile(r"^bytes \*/(\d+)$", re.IGNORECASE)
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_BACKOFF_SECONDS = 1.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_USER_AGENT = "ezkl-srs-provisioner/1"


class DownloadError(RuntimeError):
    """A retryable official SRS download failure."""


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


def response_length(response: Any) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as exc:
        raise DownloadError(f"invalid Content-Length: {value}") from exc
    if length < 0:
        raise DownloadError(f"invalid Content-Length: {value}")
    return length


def response_range(response: Any) -> tuple[int, int, int | None]:
    value = response.headers.get("Content-Range", "").strip()
    match = CONTENT_RANGE_ROW.fullmatch(value)
    if match is None:
        raise DownloadError(f"206 response has invalid Content-Range: {value or '<missing>'}")
    start, end = int(match.group(1)), int(match.group(2))
    total = None if match.group(3) == "*" else int(match.group(3))
    if end < start or (total is not None and end >= total):
        raise DownloadError(f"206 response has invalid Content-Range: {value}")
    return start, end, total


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def download_once(url: str, partial: Path) -> bool:
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Accept-Encoding": "identity", "User-Agent": DOWNLOAD_USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            value = exc.headers.get("Content-Range", "").strip()
            match = UNSATISFIED_RANGE_ROW.fullmatch(value)
            if match is not None and int(match.group(1)) == offset:
                return True
        raise DownloadError(f"HTTP {exc.code}: {exc.reason}") from exc
    except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(str(exc)) from exc

    try:
        with response:
            status = response.getcode()
            total = None
            if status == 206:
                start, end, total = response_range(response)
                if start != offset:
                    raise DownloadError(f"206 response starts at {start}, expected {offset}")
                expected_bytes = end - start + 1
                declared_length = response_length(response)
                if declared_length is not None and declared_length != expected_bytes:
                    raise DownloadError(f"206 Content-Length {declared_length} != Content-Range length {expected_bytes}")
                mode = "ab"
            elif status == 200:
                expected_bytes = response_length(response)
                mode = "wb"
                offset = 0
            else:
                raise DownloadError(f"unexpected HTTP status: {status}")

            received = 0
            with partial.open(mode) as stream:
                try:
                    while True:
                        if expected_bytes is None:
                            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                            if not chunk:
                                break
                        else:
                            remaining = expected_bytes - received
                            if remaining == 0:
                                if response.read(1):
                                    raise DownloadError("response body exceeds declared length")
                                break
                            chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, remaining + 1))
                            if not chunk:
                                raise DownloadError(f"short response body: received {received} of {expected_bytes} bytes")
                            if len(chunk) > remaining:
                                stream.write(chunk[:remaining])
                                received += remaining
                                raise DownloadError("response body exceeds declared length")
                        stream.write(chunk)
                        received += len(chunk)
                    if expected_bytes is not None and received != expected_bytes:
                        raise DownloadError(f"short response body: received {received} of {expected_bytes} bytes")
                    if total is not None and offset + received != total:
                        raise DownloadError(f"short range response: reached {offset + received} of {total} bytes")
                finally:
                    stream.flush()
                    os.fsync(stream.fileno())
    except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(str(exc)) from exc
    return True


def install_verified_srs(partial: Path, destination: Path, expected: str, actual: str | None = None) -> tuple[int, str]:
    formal_srs_path(partial, "downloaded official SRS")
    actual = digest(partial) if actual is None else actual
    if actual != expected:
        fail(f"downloaded official SRS digest mismatch: expected {expected}, got {actual}")
    size = partial.stat().st_size
    os.replace(partial, destination)
    sync_directory(destination.parent)
    formal_srs_path(destination, "native official SRS")
    return size, actual


def download_official_srs(url: str, partial: Path, destination: Path, expected: str) -> tuple[int, str]:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            download_once(url, partial)
            return install_verified_srs(partial, destination, expected)
        except DownloadError as exc:
            if attempt == DOWNLOAD_ATTEMPTS:
                fail(f"official SRS download failed after {attempt} attempts: {exc}")
            delay = DOWNLOAD_BACKOFF_SECONDS * attempt
            print(f"official SRS download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed: {exc}; " f"retrying in {delay:g}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def install_shared_srs(source_url: str, logrows: int, expected: str) -> tuple[Path, int, str]:
    SRS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = SRS_ROOT / f"kzg{logrows}.srs"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        fail(f"shared native SRS destination is not a regular file: {destination}")

    if destination.exists():
        if destination.stat().st_size >= MIN_FORMAL_SRS_BYTES:
            actual = digest(destination)
            if actual == expected:
                formal_srs_path(destination, "native official SRS")
                return destination, destination.stat().st_size, actual

    partial = SRS_ROOT / f".kzg{logrows}.srs.partial"
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        fail(f"SRS partial path is not a regular file: {partial}")
    if partial.exists() and partial.stat().st_size >= MIN_FORMAL_SRS_BYTES:
        actual = digest(partial)
        if actual == expected:
            size, actual = install_verified_srs(partial, destination, expected, actual)
            return destination, size, actual
    size, actual = download_official_srs(f"{source_url}{logrows}.srs", partial, destination, expected)
    return destination, size, actual


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
    updates = {"srs_id": srs_id, "srs_path": srs_path, "trusted_srs_sha256": sha256_value}
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
    local_binary(args.binary)
    source_url, logrows, expected = source_contract()
    destination, size, actual = install_shared_srs(source_url, logrows, expected)
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
