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
import signal
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


def _exit_on_sigterm(signum: int, _frame: object) -> NoReturn:
    raise SystemExit(128 + signum)


def digest_with_identity(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    value = hashlib.sha256()
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        named_before = resolved.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if before_identity != (named_before.st_dev, named_before.st_ino, named_before.st_size, named_before.st_mtime_ns, named_before.st_ctime_ns):
            fail(f"SRS file changed before SHA-256 authentication: {path}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        after = os.fstat(stream.fileno())
        named_after = resolved.stat()
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if after_identity != before_identity or after_identity != (named_after.st_dev, named_after.st_ino, named_after.st_size, named_after.st_mtime_ns, named_after.st_ctime_ns) or path.resolve(strict=True) != resolved:
            fail(f"SRS file changed during SHA-256 authentication: {path}")
    return value.hexdigest(), before_identity


def digest(path: Path) -> str:
    return digest_with_identity(path)[0]


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


def install_verified_srs(partial: Path, destination: Path, expected: str, actual: str | None = None, source_identity: tuple[int, int, int, int, int] | None = None) -> tuple[int, str, tuple[int, int, int, int, int]]:
    formal_srs_path(partial, "downloaded official SRS")
    if actual is None:
        actual, source_identity = digest_with_identity(partial)
    elif source_identity is None:
        fail("pre-authenticated SRS digest has no file identity")
    if actual != expected:
        fail(f"downloaded official SRS digest mismatch: expected {expected}, got {actual}")
    size = partial.stat().st_size
    if source_identity != (partial.stat().st_dev, partial.stat().st_ino, partial.stat().st_size, partial.stat().st_mtime_ns, partial.stat().st_ctime_ns):
        fail("downloaded official SRS changed after SHA-256 authentication")
    os.replace(partial, destination)
    sync_directory(destination.parent)
    formal_srs_path(destination, "native official SRS")
    info = destination.stat()
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if (identity[0], identity[1], identity[2], identity[3]) != (source_identity[0], source_identity[1], source_identity[2], source_identity[3]):
        fail("native official SRS identity changed during atomic installation")
    return size, actual, identity


def download_official_srs(url: str, partial: Path, destination: Path, expected: str) -> tuple[int, str, tuple[int, int, int, int, int]]:
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


def install_shared_srs(source_url: str, logrows: int, expected: str) -> tuple[Path, int, str, tuple[int, int, int, int, int]]:
    SRS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = SRS_ROOT / f"kzg{logrows}.srs"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        fail(f"shared native SRS destination is not a regular file: {destination}")

    if destination.exists():
        if destination.stat().st_size >= MIN_FORMAL_SRS_BYTES:
            actual, identity = digest_with_identity(destination)
            if actual == expected:
                formal_srs_path(destination, "native official SRS")
                return destination, destination.stat().st_size, actual, identity

    partial = SRS_ROOT / f".kzg{logrows}.srs.partial"
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        fail(f"SRS partial path is not a regular file: {partial}")
    if partial.exists() and partial.stat().st_size >= MIN_FORMAL_SRS_BYTES:
        actual, identity = digest_with_identity(partial)
        if actual == expected:
            size, actual, identity = install_verified_srs(partial, destination, expected, actual, identity)
            return destination, size, actual, identity
    size, actual, identity = download_official_srs(f"{source_url}{logrows}.srs", partial, destination, expected)
    return destination, size, actual, identity


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
        sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
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
    for path, payload in zip(paths, payloads):
        encoded = payload.encode()
        if path.read_bytes() != encoded:
            atomic_write_bytes(path, encoded)


def rollback_srs_derivation(snapshot: Any, old_srs_manifest: bytes | None) -> None:
    """Restore authenticated resource/config/SRS manifest bytes after a caught failure."""
    for dataset in DATASETS:
        path = config_path(dataset)
        atomic_write_bytes(path, snapshot.identity_files[path])
    atomic_write_bytes(snapshot.manifest_path, snapshot.manifest_bytes)
    if old_srs_manifest is None:
        try:
            MANIFEST.unlink()
            sync_directory(MANIFEST.parent)
        except FileNotFoundError:
            pass
    else:
        atomic_write_bytes(MANIFEST, old_srs_manifest)


def _native_srs_manifest_before_transaction() -> bytes | None:
    if MANIFEST.is_symlink():
        fail(f"native SRS manifest is symlinked: {MANIFEST}")
    if not MANIFEST.exists():
        return None
    if not MANIFEST.is_file():
        fail(f"native SRS manifest is not a regular file: {MANIFEST}")
    try:
        return MANIFEST.read_bytes()
    except OSError as exc:
        fail(f"cannot snapshot native SRS manifest before derivation: {exc}")


def _assert_prestate_unchanged(snapshot: Any, old_srs_manifest: bytes | None) -> None:
    if snapshot.manifest_path.is_symlink() or snapshot.manifest_path.read_bytes() != snapshot.manifest_bytes:
        fail("resource manifest changed while authenticating/provisioning the native SRS")
    for dataset in DATASETS:
        path = config_path(dataset)
        if path.read_bytes() != snapshot.identity_files[path]:
            fail(f"secure config changed while authenticating/provisioning the native SRS: {path}")
    if _native_srs_manifest_before_transaction() != old_srs_manifest:
        fail("native SRS manifest changed while authenticating/provisioning the SRS")


def main(args: argparse.Namespace) -> int:
    if any(flag in sys.argv[1:] for flag in ("--source", "--sha256", "--dataset", "--replace", "gen-srs")):
        fail("manual SRS source/hash and dummy gen-srs inputs are forbidden")
    if args.binary is not None:
        local_binary(args.binary)
    from validate_resources import authenticate_preparation_inputs, refresh_manifest_after_authenticated_srs_derivation

    snapshot = authenticate_preparation_inputs(ROOT)
    old_srs_manifest = _native_srs_manifest_before_transaction()
    source_url, logrows, expected = source_contract()
    destination, size, actual, file_identity = install_shared_srs(source_url, logrows, expected)
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
    previous_sigterm = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        try:
            _assert_prestate_unchanged(snapshot, old_srs_manifest)
            atomic_write(MANIFEST, manifest)
            update_configs(srs_id=srs_id, srs_path=relative, sha256_value=actual)
            refresh_manifest_after_authenticated_srs_derivation(
                snapshot,
                manifest,
                source_url=source_object["source_url"],
                logrows=logrows,
                expected_sha256=expected,
                actual_sha256=actual,
                srs_path=relative,
                srs_size=size,
                srs_file_identity=file_identity,
            )
        except BaseException as failure:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                rollback_srs_derivation(snapshot, old_srs_manifest)
            except BaseException as rollback_failure:
                raise RuntimeError(
                    f"SRS derivation failed ({failure}); authenticated prestate rollback also failed ({rollback_failure})"
                ) from rollback_failure
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(f"native official SRS ready: {relative}")
    print(f"native get-srs source: {source_object['source_url']}")
    print(f"logrows={logrows} sha256={actual} size_bytes={size}")
    print(f"shared datasets: {','.join(DATASETS)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", help="optional current baseline-local ezkl binary; SRS identity comes from pinned source")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (OSError, json.JSONDecodeError) as exc:
        fail(str(exc))
