"""Authenticated identity for the baseline-local CUDA proof backend.

The CUDA executable is a production proof primitive, so an executable that
merely happens to be newer than one source file is not sufficient evidence of
freshness.  This module defines the one manifest consumed by the build script
and by every GPU entry point.  It deliberately contains an explicit source
inventory instead of relying on mtimes or on a repository-wide git checkout.
"""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import stat
import subprocess
from typing import Any, Mapping, Sequence

from .util import canonical_json_file_bytes, digest_file, digest_json, lexical_path, write_json


BUILD_MANIFEST_SCHEMA = "zkGPT_cuda_build_manifest_v1"
BUILD_IDENTITY_SCHEMA = "zkGPT_cuda_build_identity_v1"
BUILD_MANIFEST_RELATIVE_PATH = pathlib.Path("build/cuda_build_manifest.json")
CUDA_BINARY_RELATIVE_PATH = pathlib.Path("build/zkgpt_cuda_native_prover")
CUDA_SOURCE_RELATIVE_PATH = pathlib.Path("backend/cuda/zkgpt_cuda_proof_hot_path.cu")
CUDA_KERNEL_ID = "zkgpt_native_bn254_fr_batch_v3"
CUDA_PROTOCOL_ID = "zkgpt/native-bn254-fr-batch-v1"


def _required_source_paths(root: pathlib.Path) -> list[pathlib.Path]:
    """Return the explicit proof-path source inventory in stable order.

    Python proof modules and formal runner scripts are included because a
    CUDA result is only meaningful together with the consumer that interprets
    its messages.  Tests, generated resources, and documentation are not
    proof-path inputs and are intentionally excluded.
    """
    root = lexical_path(root)
    paths: list[pathlib.Path] = [root / "CMakeLists.txt", root / "requirements.lock"]
    paths.append(root / CUDA_SOURCE_RELATIVE_PATH)
    paths.extend(sorted((root / "src" / "zkgpt_gat").glob("*.py")))
    paths.extend(sorted((root / "scripts").glob("*.py")))
    paths.extend(
        [
            root / "scripts" / "build.sh",
            root / "scripts" / "build_cuda_backend.sh",
        ]
    )
    # De-duplicate while retaining lexical order.  The explicit shell build
    # entries are not Python files, so this also guards future renames.
    unique: dict[str, pathlib.Path] = {}
    for path in paths:
        relative = path.relative_to(root)
        unique[str(relative)] = path
    return [unique[name] for name in sorted(unique)]


def _owned_regular(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    candidate = lexical_path(path)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise OSError(f"cannot inspect baseline root for {label}: {root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"baseline root for {label} must be a real directory: {root}")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes baseline root: {candidate}") from exc
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise OSError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} must not contain symlink: {current}")
        if current != candidate and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} parent must be a directory: {current}")
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise OSError(f"cannot inspect {label}: {candidate}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file: {candidate}")
    if info.st_nlink != 1:
        raise ValueError(f"{label} hardlink count must be one: {candidate}")
    return candidate


def source_inventory(root: pathlib.Path) -> list[dict[str, Any]]:
    """Hash every current proof-path source and return canonical records."""
    root = lexical_path(root)
    entries: list[dict[str, Any]] = []
    for path in _required_source_paths(root):
        owned = _owned_regular(path, root, "CUDA build source")
        entries.append(
            {
                "path": str(owned.relative_to(root)),
                "size_bytes": owned.stat().st_size,
                "sha256": digest_file(owned),
            }
        )
    return entries


def source_manifest_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return digest_json(list(entries))


def _normalise_nvcc_version(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError("nvcc version output is empty")
    return value


def nvcc_version(nvcc: pathlib.Path | str) -> str:
    executable = os.fspath(nvcc)
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvcc --version failed with status {completed.returncode}")
    return _normalise_nvcc_version(completed.stdout)


def build_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select the immutable fields that identify a usable native binary."""
    compiler = manifest.get("compiler")
    target = manifest.get("target")
    compile_info = manifest.get("compile")
    binary = manifest.get("binary")
    source = manifest.get("source")
    if not all(isinstance(value, Mapping) for value in (compiler, target, compile_info, binary, source)):
        raise ValueError("CUDA build manifest identity sections are malformed")
    return {
        "schema": BUILD_IDENTITY_SCHEMA,
        "baseline": manifest.get("baseline"),
        "manifest_schema": manifest.get("schema"),
        "source_manifest_sha256": source.get("manifest_sha256"),
        "binary_sha256": binary.get("sha256"),
        "nvcc_version_sha256": digest_json(compiler.get("nvcc_version")),
        "target": dict(target),
        "compile": dict(compile_info),
        "kernel": manifest.get("kernel"),
        "protocol": manifest.get("protocol"),
    }


def build_identity(manifest: Mapping[str, Any]) -> str:
    return digest_json(build_identity_payload(manifest))


def _canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_file_bytes(dict(payload))


def build_manifest_path(root: pathlib.Path) -> pathlib.Path:
    return lexical_path(root) / BUILD_MANIFEST_RELATIVE_PATH


def binary_path(root: pathlib.Path) -> pathlib.Path:
    return lexical_path(root) / CUDA_BINARY_RELATIVE_PATH


def make_build_manifest(
    root: pathlib.Path,
    *,
    binary: pathlib.Path,
    nvcc: pathlib.Path | str,
    target_compute_capability: str,
    arch_flag: str,
    compile_command: Sequence[str],
) -> dict[str, Any]:
    """Create the complete canonical build receipt after a successful link."""
    root = lexical_path(root)
    binary = _owned_regular(binary, root, "CUDA binary")
    nvcc_path = pathlib.Path(nvcc)
    if not nvcc_path.is_absolute():
        resolved = shutil_which(os.fspath(nvcc_path))
        if resolved is None:
            raise FileNotFoundError(f"nvcc executable is not found: {nvcc}")
        nvcc_path = pathlib.Path(resolved)
    try:
        nvcc_path = nvcc_path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"nvcc executable cannot be resolved: {nvcc_path}") from exc
    if not nvcc_path.is_file():
        raise FileNotFoundError(f"nvcc executable is not a file: {nvcc_path}")
    entries = source_inventory(root)
    source = {
        "inventory_schema": "zkGPT_cuda_source_inventory_v1",
        "files": entries,
        "manifest_sha256": source_manifest_digest(entries),
    }
    version = nvcc_version(nvcc_path)
    command = [str(item) for item in compile_command]
    if not command or command[0] != os.fspath(nvcc_path):
        # The command is part of the identity, so record the resolved compiler
        # spelling rather than allowing two equivalent PATH aliases.
        command = [os.fspath(nvcc_path), *command[1:]] if command else [os.fspath(nvcc_path)]
    manifest: dict[str, Any] = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "baseline": "zkGPT",
        "kernel": CUDA_KERNEL_ID,
        "protocol": CUDA_PROTOCOL_ID,
        "binary": {
            "path": str(binary.relative_to(root)),
            "size_bytes": binary.stat().st_size,
            "sha256": digest_file(binary),
        },
        "source": source,
        "compiler": {
            "nvcc_path": os.fspath(nvcc_path),
            "nvcc_version": version,
            "nvcc_version_sha256": digest_json(version),
        },
        "target": {
            "compute_capability": str(target_compute_capability),
            "arch": f"compute_{arch_flag}",
            "code": f"sm_{arch_flag}",
            "gencode": f"arch=compute_{arch_flag},code=sm_{arch_flag}",
        },
        "compile": {
            # Keep every compiler argument (including source/output operands)
            # in the identity; the complete argv below is the authoritative
            # replay command.
            "flags": command[1:],
            "command": command,
            "command_shell": shlex.join(command),
        },
    }
    manifest["build_identity"] = build_identity(manifest)
    manifest["manifest_digest"] = digest_json(manifest)
    return manifest


def write_build_manifest(
    root: pathlib.Path,
    *,
    binary: pathlib.Path,
    nvcc: pathlib.Path | str,
    target_compute_capability: str,
    arch_flag: str,
    compile_command: Sequence[str],
) -> dict[str, Any]:
    manifest = make_build_manifest(
        root,
        binary=binary,
        nvcc=nvcc,
        target_compute_capability=target_compute_capability,
        arch_flag=arch_flag,
        compile_command=compile_command,
    )
    write_json(build_manifest_path(root), manifest, root=lexical_path(root), label="CUDA build manifest")
    return manifest


def _read_manifest(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"CUDA build manifest is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("CUDA build manifest must be a JSON object")
    if raw != _canonical_manifest_bytes(value):
        raise ValueError("CUDA build manifest is not canonically serialized")
    return value


def verify_build_manifest(root: pathlib.Path, *, require_nvcc: bool = False) -> dict[str, Any]:
    """Verify manifest, current source inventory, and linked binary.

    This function never accepts a stale manifest merely because a binary is
    executable.  It also rejects path aliases and hardlinked artifacts before
    hashing them.
    """
    root = lexical_path(root)
    manifest_path = _owned_regular(build_manifest_path(root), root, "CUDA build manifest")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema") != BUILD_MANIFEST_SCHEMA or manifest.get("baseline") != "zkGPT":
        raise ValueError("CUDA build manifest schema/baseline mismatch")
    claimed_manifest_digest = manifest.get("manifest_digest")
    without_digest = dict(manifest)
    without_digest.pop("manifest_digest", None)
    if claimed_manifest_digest != digest_json(without_digest):
        raise ValueError("CUDA build manifest digest mismatch")
    if manifest.get("build_identity") != build_identity(manifest):
        raise ValueError("CUDA build identity mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("files"), list):
        raise ValueError("CUDA build source inventory is missing")
    current_entries = source_inventory(root)
    if source.get("files") != current_entries:
        raise ValueError("CUDA build manifest is stale for the current proof-path source")
    if source.get("manifest_sha256") != source_manifest_digest(current_entries):
        raise ValueError("CUDA source inventory digest mismatch")
    binary_record = manifest.get("binary")
    if not isinstance(binary_record, Mapping) or binary_record.get("path") != str(CUDA_BINARY_RELATIVE_PATH):
        raise ValueError("CUDA binary path is not canonical")
    binary = _owned_regular(root / CUDA_BINARY_RELATIVE_PATH, root, "CUDA binary")
    binary_sha = digest_file(binary)
    if binary_record.get("sha256") != binary_sha or binary_record.get("size_bytes") != binary.stat().st_size:
        raise ValueError("CUDA binary does not match authenticated build manifest")
    compiler = manifest.get("compiler")
    if not isinstance(compiler, Mapping) or compiler.get("nvcc_version_sha256") != digest_json(compiler.get("nvcc_version")):
        raise ValueError("CUDA compiler identity is malformed")
    nvcc_path = compiler.get("nvcc_path")
    if not isinstance(nvcc_path, str) or not nvcc_path or not pathlib.Path(nvcc_path).is_absolute():
        raise ValueError("CUDA compiler path is not canonical")
    if require_nvcc:
        current_version = nvcc_version(nvcc_path)
        if current_version != compiler.get("nvcc_version"):
            raise ValueError("installed nvcc version differs from CUDA build manifest")
    target = manifest.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("CUDA target identity is missing")
    for key in ("compute_capability", "arch", "code", "gencode"):
        if not isinstance(target.get(key), str) or not target.get(key):
            raise ValueError(f"CUDA target identity is missing {key}")
    capability = str(target["compute_capability"])
    pieces = capability.split(".")
    if len(pieces) != 2 or any(not piece.isdigit() for piece in pieces) or any(not piece for piece in pieces):
        raise ValueError("CUDA compute capability must use major.minor spelling")
    arch_flag = capability.replace(".", "")
    if target.get("arch") != f"compute_{arch_flag}" or target.get("code") != f"sm_{arch_flag}" or target.get("gencode") != f"arch=compute_{arch_flag},code=sm_{arch_flag}":
        raise ValueError("CUDA target identity is internally inconsistent")
    compile_info = manifest.get("compile")
    if not isinstance(compile_info, Mapping):
        raise ValueError("CUDA compile identity is missing")
    command = compile_info.get("command")
    flags = compile_info.get("flags")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("CUDA compile command is malformed")
    if not isinstance(flags, list) or flags != command[1:]:
        raise ValueError("CUDA compile flags do not match the canonical command")
    expected_gencode = f"-gencode=arch=compute_{arch_flag},code=sm_{arch_flag}"
    expected_command = [
        str(nvcc_path),
        "-std=c++17",
        "-O3",
        "-lineinfo",
        expected_gencode,
        str(root / CUDA_SOURCE_RELATIVE_PATH),
        "-o",
        str(root / CUDA_BINARY_RELATIVE_PATH),
    ]
    if command != expected_command:
        raise ValueError("CUDA compile command is not the authenticated native target")
    if compile_info.get("command_shell") != shlex.join(command):
        raise ValueError("CUDA compile command shell identity mismatch")
    return manifest


def shutil_which(program: str) -> str | None:
    """Small stdlib-only ``which`` helper used by the build receipt writer."""
    path_value = os.environ.get("PATH", os.defpath)
    for directory in path_value.split(os.pathsep):
        candidate = pathlib.Path(directory or ".") / program
        try:
            info = candidate.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and info.st_mode & 0o111:
            return os.fspath(candidate)
    return None


__all__ = [
    "BUILD_MANIFEST_SCHEMA",
    "BUILD_IDENTITY_SCHEMA",
    "BUILD_MANIFEST_RELATIVE_PATH",
    "CUDA_BINARY_RELATIVE_PATH",
    "CUDA_KERNEL_ID",
    "CUDA_PROTOCOL_ID",
    "binary_path",
    "build_identity",
    "build_identity_payload",
    "build_manifest_path",
    "make_build_manifest",
    "source_inventory",
    "source_manifest_digest",
    "verify_build_manifest",
    "write_build_manifest",
]
