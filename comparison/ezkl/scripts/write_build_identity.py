#!/usr/bin/env python3
"""Write authenticated build provenance for a formal ezkl backend binary."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from formal_source_manifest import (
    checked_path,
    formal_source_files,
    formal_vendor_files,
    formal_source_id,
    SOURCE_FILTER_CONTRACT,
    lexical_baseline_root,
    tree_digest as manifest_tree_digest,
)
from resource_policy import safe_mkdir, validate_cuda_backend_material


ROOT = lexical_baseline_root(Path(__file__).absolute().parent.parent)
VENDOR_ROOT = ROOT / "vendor"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def tree_digest(paths: list[Path]) -> str:
    return manifest_tree_digest(
        ROOT,
        sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()),
    )


def vendor_tree_digest() -> str:
    try:
        files = formal_vendor_files(ROOT)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not files:
        raise SystemExit("authenticated vendor tree is empty")
    return tree_digest(files)


def validate_dependency_sources() -> None:
    try:
        vendor_root = checked_path(ROOT, "vendor", "authenticated vendor tree")
        vendor_files = formal_vendor_files(ROOT)
        manifests = [checked_path(ROOT, "Cargo.toml", "root Cargo manifest")]
        manifests.extend(path for path in vendor_files if path.name == "Cargo.toml")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8")
        for line in text.splitlines():
            if re.search(r"(?:^|[\s\[])(?:branch|tag)\s*=", line):
                raise SystemExit(f"floating branch/tag selector in {manifest}: {line}")
            match = re.search(r'git\s*=\s*"([^"]+)"', line)
            if match:
                source = match.group(1)
                if "branch=" in source or "tag=" in source or "?branch" in source or "?tag" in source:
                    raise SystemExit(f"floating git selector in {manifest}: {source}")
                if "?" in source and not re.search(r"\?rev=[0-9a-fA-F]{40}(?:#|$)", source):
                    raise SystemExit(f"non-revision git query in {manifest}: {source}")
            match = re.search(r'path\s*=\s*"([^"]+)"', line)
            if match:
                raw_path = Path(match.group(1))
                if raw_path.is_absolute():
                    raise SystemExit(
                        f"absolute path dependency in {manifest}: {match.group(1)}"
                    )
                # Cargo permits a vendored manifest to refer to a sibling
                # package with a relative ``../`` component.  Normalize that
                # lexical spelling before checked_path() inspects every
                # existing component; containment and symlink/hard-link checks
                # still happen on the normalized path, so this does not permit
                # an escape from the authenticated baseline/vendor tree.
                normalized = Path(os.path.abspath(manifest.parent / raw_path))
                try:
                    target = checked_path(ROOT, normalized, "path dependency")
                except (FileNotFoundError, ValueError) as exc:
                    raise SystemExit(str(exc)) from exc
                if not target.is_relative_to(vendor_root):
                    raise SystemExit(f"unverified path dependency in {manifest}: {match.group(1)}")
                if not target.exists():
                    raise SystemExit(f"missing path dependency in {manifest}: {target}")
    lock = ROOT / "Cargo.lock"
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.search(r'source\s*=\s*"([^"]+)"', line)
        if not match:
            continue
        source = match.group(1)
        if source.startswith("git+"):
            if "branch=" in source or "tag=" in source or "?branch" in source or "?tag" in source:
                raise SystemExit(f"floating git selector in Cargo.lock: {source}")
            if "?" in source and not re.search(r"\?rev=[0-9a-fA-F]{40}(?:#|$)", source):
                raise SystemExit(f"non-revision git query in Cargo.lock: {source}")


def rustc_version() -> str:
    try:
        return subprocess.run(
            ["rustc", "-Vv"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs" / "security_profile.json")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_dependency_sources()
    try:
        binary = checked_path(ROOT, args.binary, "formal binary")
        adapter = checked_path(ROOT, args.adapter, "formal adapter")
        profile = checked_path(ROOT, args.profile, "security profile")
        output = checked_path(ROOT, args.output, "build identity output", allow_missing=True)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for label, path in (("binary", binary), ("adapter", adapter), ("security profile", profile)):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    formal_files = formal_source_files(ROOT)
    source_paths = formal_files
    if any(not path.is_file() for path in source_paths):
        raise SystemExit("formal source manifest contains a missing file")
    payload = {
        "build_identity_kind": "ezkl_formal_build_identity_v1",
        "backend": args.backend,
        "features": args.features,
        "source_tree_digest": tree_digest(formal_files),
        "formal_semantic_source_id": formal_source_id(ROOT),
        "source_filter_contract": SOURCE_FILTER_CONTRACT,
        "formal_source_manifest": [path.relative_to(ROOT).as_posix() for path in formal_files],
        "vendor_tree_digest": vendor_tree_digest(),
        "dependency_source_policy": "exact-revision-or-authenticated-vendor-path-v1",
        "dependency_lock_digest": sha256(ROOT / "Cargo.lock"),
        "security_profile_digest": sha256(profile),
        "binary_sha256": sha256(binary),
        "adapter_sha256": sha256(adapter),
        "binary_path": str(binary.relative_to(ROOT)),
        "adapter_path": str(adapter.relative_to(ROOT)),
        "rustc_version": rustc_version(),
        "gpu_telemetry_contract": (
            "cupti_activity_kernel_scoped_v1" if args.backend == "cuda" else "not_applicable"
        ),
        "cuda_backend_provenance": (
            validate_cuda_backend_material(ROOT)
            if args.backend == "cuda"
            else {"status": "not_applicable"}
        ),
    }
    safe_mkdir(ROOT, output.parent, "build identity output parent")
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        checked_path(ROOT, temporary, "build identity temporary output", allow_missing=True)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    encoded = (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"unsafe pre-existing build identity temporary output: {temporary}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
