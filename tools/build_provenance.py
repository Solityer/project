#!/usr/bin/env python3
"""Generate source-locked build provenance for formal run artifacts.

The source identity is independent of working-tree metadata.  It is a
length-delimited SHA-256 digest over a deterministic file inventory, and the
complete inventory is embedded in the executable header so a run can publish
the exact source manifest that was built.  Runtime code adds the hash of the
actual ``/proc/self/exe`` bytes to the build receipt; environment variables
and user supplied binary hashes are never provenance sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any

SOURCE_INVENTORY_ID = "zkgat:source-inventory"
SOURCE_MANIFEST_FORMAT = "zkgat_source_manifest"
SOURCE_MANIFEST_ALGORITHM = "sha256:length-prefixed-path-content"
BUILD_RECEIPT_FORMAT = "zkgat_build_receipt"
BUILD_IDENTITY_ALGORITHM = "zkgat:build-identity"

# Generated/runtime trees are never part of a source identity.  ``build*`` is
# prefix based so local directories such as build_debug are excluded too.
EXCLUDED_COMPONENTS = {"comparison", "runs", "tests", ".git"}


def _excluded(path: pathlib.PurePosixPath) -> bool:
    # ``build*`` denotes generated directories (for example ``build`` and
    # ``build_debug``), not source files whose basename happens to start with
    # ``build``.  The provenance generator itself is source input and must be
    # included in the inventory because changing it changes the generated
    # compiler header and manifest.
    final_index = len(path.parts) - 1
    return any(component in EXCLUDED_COMPONENTS or (index != final_index and component.startswith("build")) for index, component in enumerate(path.parts))


def append_bytes(out: bytearray, value: bytes) -> None:
    out.extend(struct.pack("<Q", len(value)))
    out.extend(value)


def inventory_paths(root: pathlib.Path) -> list[str]:
    """Return canonical relative files, including new non-ignored sources."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        candidates = result.stdout.decode("utf-8").split("\0")
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
        candidates = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()]

    paths: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        relative = pathlib.PurePosixPath(candidate)
        if relative.is_absolute() or _excluded(relative):
            continue
        absolute = root.joinpath(*relative.parts)
        # Do not follow symlinks into source or artifact material.
        if absolute.is_symlink() or not absolute.is_file():
            continue
        paths.append(relative.as_posix())
    return sorted(set(paths))


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: pathlib.Path) -> str:
    canonical = bytearray()
    append_bytes(canonical, SOURCE_INVENTORY_ID.encode("utf-8"))
    for relative in inventory_paths(root):
        append_bytes(canonical, relative.encode("utf-8"))
        append_bytes(canonical, (root / pathlib.Path(*relative.split("/"))).read_bytes())
    return hashlib.sha256(canonical).hexdigest()


def source_manifest_document(root: pathlib.Path) -> dict[str, Any]:
    """Build the complete source manifest used by exporter and aggregator."""
    root = root.resolve()
    files = []
    for relative in inventory_paths(root):
        path = root / pathlib.Path(*relative.split("/"))
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _file_sha256(path)})
    canonical = bytearray()
    append_bytes(canonical, SOURCE_INVENTORY_ID.encode("utf-8"))
    for entry in files:
        append_bytes(canonical, entry["path"].encode("utf-8"))
        append_bytes(canonical, (root / pathlib.Path(*entry["path"].split("/"))).read_bytes())
    return {
        "algorithm": SOURCE_MANIFEST_ALGORITHM,
        "file_count": len(files),
        "files": files,
        "format": SOURCE_MANIFEST_FORMAT,
        "inventory_id": SOURCE_INVENTORY_ID,
        "source_tree_hash": hashlib.sha256(canonical).hexdigest(),
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used for source manifest publication and hashing."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def build_identity(source_hash: str, compiler_id: str, compiler_version: str, build_type: str, flags: str, binary_hash: str) -> str:
    canonical = bytearray()
    append_bytes(canonical, BUILD_IDENTITY_ALGORITHM.encode("utf-8"))
    for value in (source_hash, compiler_id, compiler_version, build_type, flags, binary_hash):
        append_bytes(canonical, value.encode("utf-8"))
    return hashlib.sha256(canonical).hexdigest()


def cxx_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def write_source_manifest(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--source-manifest-output", type=pathlib.Path)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("--build-type", default="")
    parser.add_argument("--flags", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = source_manifest_document(root)
    manifest_json = canonical_json_bytes(manifest).decode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = """#pragma once
// Generated at configure time by tools/build_provenance.py.  Do not edit.
namespace gatzk::build_provenance {
inline constexpr const char kSourceInventoryId[] = %s;
inline constexpr const char kSourceManifestFormat[] = %s;
inline constexpr const char kSourceManifestAlgorithm[] = %s;
inline constexpr const char kSourceTreeHash[] = %s;
inline constexpr const char kSourceManifestJson[] = %s;
inline constexpr const unsigned long long kSourceFileCount = %dULL;
inline constexpr const char kCompilerId[] = %s;
inline constexpr const char kCompilerVersion[] = %s;
inline constexpr const char kBuildType[] = %s;
inline constexpr const char kCompileFlags[] = %s;
inline constexpr const char kBuildIdentityAlgorithm[] = %s;
}
""" % (
        cxx_string(SOURCE_INVENTORY_ID),
        cxx_string(SOURCE_MANIFEST_FORMAT),
        cxx_string(SOURCE_MANIFEST_ALGORITHM),
        cxx_string(manifest["source_tree_hash"]),
        cxx_string(manifest_json),
        manifest["file_count"],
        cxx_string(args.compiler_id),
        cxx_string(args.compiler_version),
        cxx_string(args.build_type),
        cxx_string(args.flags),
        cxx_string(BUILD_IDENTITY_ALGORITHM),
    )
    args.output.write_text(text, encoding="utf-8")
    if args.source_manifest_output is not None:
        write_source_manifest(args.source_manifest_output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
