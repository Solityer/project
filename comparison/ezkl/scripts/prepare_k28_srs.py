#!/usr/bin/env python3
"""Prepare and audit the exact PSE PPOT power-28 source SRS.

The formal runner calls this as a setup stage.  It is intentionally fail
closed: the 288 GiB source is never substituted with a lower-power file or a
locally generated placeholder, and the converter must be a baseline-local,
hash-pinned executable described by ``converter.lock.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CEREMONY = ROOT / "resources" / "ceremony"
PART = CEREMONY / "ppot_0080_final.ptau.part"
SOURCE = CEREMONY / "source.srs"
DESCRIPTOR = CEREMONY / "ceremony.descriptor.json"
AUDIT = CEREMONY / "ppot_0080.audit.json"
TRANSPORT = CEREMONY / "ppot_0080.transport.json"
LOCK = CEREMONY / "converter.lock.json"
CONVERTER_SOURCE = ROOT / "vendor" / "halo2-kzg-srs"
CONVERTER_REPOSITORY = "https://github.com/han0110/halo2-kzg-srs.git"
CONVERTER_COMMIT = "d6929c74916f6be90751405a18df875ffcc0c241"
CONVERTER_GIT_TREE = "77ab460f023c204ba0578484c2d9c45227859d54"
CONVERTER_SOURCE_TREE_SHA256 = "sha256:71f2a38421565c1951e9750b2ac92663a02eda57741375842dc8c40db2a12174"
CONVERTER_PATH = "resources/ceremony/halo2-kzg-srs-converter"
CONVERTER_SHA256 = "sha256:4088ac01ad19adc1d522b5a2c7cf8b74e0a965a26057505be515cf2501fedac3"
CONVERTER_BUILD_TARGET = "aarch64-unknown-linux-gnu"
CONVERTER_BUILD_PROFILE = "release"
CONVERTER_TOOLCHAIN = "nightly-2025-12-01-aarch64-unknown-linux-gnu"
CONVERTER_RUSTC = "rustc 1.93.0-nightly (b84478a1c 2025-11-30)"
CONVERTER_CARGO = "cargo 1.93.0-nightly (2a7c49606 2025-11-25)"
URL = "https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_final.ptau"
SIZE = 309237738642
ETAG = "fd2bd03278b65db562b7ad882d8bc293-9217"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def regular(path: Path, label: str) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must be a single-link regular file: {path}")


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command, cwd=cwd, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"failed to inspect locked converter identity: {' '.join(command)}"
        ) from exc
    return result.stdout.strip()


def _validate_converter_source(lock: dict[str, object]) -> None:
    source = lock.get("source")
    expected_source = {
        "repository": CONVERTER_REPOSITORY,
        "commit": CONVERTER_COMMIT,
        "git_tree": CONVERTER_GIT_TREE,
        "source_tree_sha256": CONVERTER_SOURCE_TREE_SHA256,
        "path": "vendor/halo2-kzg-srs",
    }
    if source != expected_source:
        raise RuntimeError("converter source lock does not match the pinned official checkout")
    if CONVERTER_SOURCE.is_symlink() or not CONVERTER_SOURCE.is_dir():
        raise RuntimeError("locked converter source checkout is not baseline-local")
    if _run_text(["git", "rev-parse", "HEAD"], CONVERTER_SOURCE) != CONVERTER_COMMIT:
        raise RuntimeError("converter source commit does not match lock")
    if _run_text(["git", "rev-parse", "HEAD^{tree}"], CONVERTER_SOURCE) != CONVERTER_GIT_TREE:
        raise RuntimeError("converter source git tree does not match lock")
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"], cwd=CONVERTER_SOURCE,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    actual = "sha256:" + hashlib.sha256(archive.stdout).hexdigest()
    if actual != CONVERTER_SOURCE_TREE_SHA256:
        raise RuntimeError("converter source tree digest does not match lock")


def _validate_converter_build(lock: dict[str, object], converter: Path) -> None:
    build = lock.get("build")
    expected_build = {
        "target": CONVERTER_BUILD_TARGET,
        "profile": CONVERTER_BUILD_PROFILE,
        "toolchain": CONVERTER_TOOLCHAIN,
        "rustc": CONVERTER_RUSTC,
        "cargo": CONVERTER_CARGO,
        "sha256": CONVERTER_SHA256,
    }
    if build != expected_build:
        raise RuntimeError("converter build identity does not match lock")
    if _run_text(["rustc", "-V"], ROOT) != CONVERTER_RUSTC:
        raise RuntimeError("active rustc version does not match converter build lock")
    if _run_text(["cargo", "-V"], ROOT) != CONVERTER_CARGO:
        raise RuntimeError("active cargo version does not match converter build lock")
    active_toolchain = _run_text(["rustup", "show", "active-toolchain"], ROOT).split()[0]
    if active_toolchain != CONVERTER_TOOLCHAIN:
        raise RuntimeError("active rust toolchain does not match converter build lock")
    if "sha256:" + sha256(converter) != CONVERTER_SHA256:
        raise RuntimeError("converter binary digest does not match build lock")


def download() -> None:
    CEREMONY.mkdir(parents=True, exist_ok=True)
    offset = PART.stat().st_size if PART.exists() else 0
    if offset > SIZE:
        raise RuntimeError("partial PPOT file is larger than the pinned transport size")
    headers = {"If-Match": f'"{ETAG}"'}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        response_etag = response.headers.get("ETag", "").strip('"')
        if response_etag != ETAG:
            raise RuntimeError(f"PPOT ETag mismatch: {response_etag!r}")
        if offset and response.status != 206:
            raise RuntimeError("PPOT resume request was not answered with HTTP 206")
        content_length = int(response.headers.get("Content-Length", "0"))
        expected_length = SIZE - offset if offset else SIZE
        if content_length != expected_length:
            raise RuntimeError("PPOT transport Content-Length mismatch")
        mode = "ab" if offset else "wb"
        with PART.open(mode) as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    regular(PART, "PPOT partial")
    if PART.stat().st_size != SIZE:
        raise RuntimeError("PPOT download ended before the pinned size")
    os.replace(PART, CEREMONY / "ppot_0080_final.ptau")
    full = CEREMONY / "ppot_0080_final.ptau"
    regular(full, "PPOT source")
    TRANSPORT.write_text(json.dumps({
        "schema": "ezkl_pse_ppot_transport_v1", "url": URL,
        "filename": full.name, "size": SIZE, "etag": ETAG,
        "sha256": "sha256:" + sha256(full),
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def convert_and_audit() -> None:
    full = CEREMONY / "ppot_0080_final.ptau"
    regular(full, "PPOT source")
    if not LOCK.is_file():
        raise RuntimeError(f"missing baseline-local converter lock: {LOCK}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if set(lock) != {"schema", "source", "build", "converter", "converter_sha256", "command", "audit"}:
        raise RuntimeError("converter lock schema is not exact")
    if lock["schema"] != "ezkl_pinned_halo2_kzg_srs_converter_v1":
        raise RuntimeError("converter lock schema mismatch")
    if lock["converter"] != CONVERTER_PATH or lock["converter_sha256"] != CONVERTER_SHA256:
        raise RuntimeError("converter path/digest is not the pinned baseline-local build")
    _validate_converter_source(lock)
    converter = (ROOT / str(lock["converter"])).resolve()
    if ROOT not in converter.parents or not converter.is_file() or converter.is_symlink():
        raise RuntimeError("converter is not a baseline-local regular file")
    _validate_converter_build(lock, converter)
    command = lock["command"]
    if command != [CONVERTER_PATH, "{input}", "{output_prefix}", "28"]:
        raise RuntimeError("converter command is not pinned to the locked executable")
    stage = Path(tempfile.mkdtemp(prefix=".srs-convert-", dir=CEREMONY))
    try:
        prefix = stage / "source.srs."
        rendered = [
            str(item).replace("{input}", str(full)).replace("{output_prefix}", str(prefix))
            for item in command
        ]
        subprocess.run(rendered, cwd=ROOT, check=True)
        converted_k28 = stage / "source.srs.28"
        regular(converted_k28, "converted k=28 source SRS")
        os.replace(converted_k28, SOURCE)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    regular(SOURCE, "converted source SRS")
    audit = json.loads(json.dumps(lock["audit"]))
    if not isinstance(audit, dict) or set(audit) != {"path", "source_srs_k", "ratio_verified", "transcript_verified", "params_k"}:
        raise RuntimeError("converter audit contract is not exact")
    if audit["path"] != str(SOURCE.relative_to(ROOT)):
        raise RuntimeError("converted source SRS audit path mismatch")
    if audit["source_srs_k"] != 28 or audit["params_k"] != 28 or audit["ratio_verified"] is not True or audit["transcript_verified"] is not True:
        raise RuntimeError("converter did not provide complete k=28 ratio/transcript/params audit")
    output_digest = "sha256:" + sha256(SOURCE)
    AUDIT.write_text(json.dumps({
        "schema": "ezkl_pse_ppot_conversion_audit_v1", "transport_url": URL,
        "transport_size": SIZE, "transport_etag": ETAG,
        "transport_sha256": "sha256:" + sha256(full),
        "converter": lock["converter"], "converter_sha256": lock["converter_sha256"],
        "output": {**audit, "sha256": output_digest},
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    descriptor = {
        "descriptor_kind": "ezkl_trusted_ceremony_srs_v1",
        "ceremony_id": "pse-ppot-0080-beacon-final-power-28",
        "source_srs_sha256": output_digest, "source_srs_max_k": 28,
        "source": {"url": URL, "size": SIZE, "etag": ETAG,
                    "transport_sha256": "sha256:" + sha256(full)},
        "conversion_audit": str(AUDIT.relative_to(ROOT)),
        "converter": lock["converter"], "converter_sha256": lock["converter_sha256"],
    }
    DESCRIPTOR.write_text(json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure", action="store_true", help="download/resume and convert when missing")
    args = parser.parse_args()
    if not args.ensure:
        parser.error("--ensure is required for the formal setup stage")
    full = CEREMONY / "ppot_0080_final.ptau"
    if not LOCK.is_file():
        raise RuntimeError(
            "missing baseline-local converter.lock.json; refusing to download PPOT before converter audit is staged"
        )
    if not full.is_file():
        download()
    convert_and_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
