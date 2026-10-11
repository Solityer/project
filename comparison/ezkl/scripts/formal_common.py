#!/usr/bin/env python3
"""Shared fail-closed checks for the ezkl GAT formal execution scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import time
import csv
import io
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn
from process_guardian import check_guardian_health, finish_guarded_process, run_guarded_capture, spawn_guarded_process, terminate_guarded_process_locally

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "resources" / "resource_manifest.json"
CONTRACT = ROOT / "artifacts" / "gat_adapter" / "formal_cli.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORMAL_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
COMMAND = re.compile(r"^[a-z][a-z0-9-]*$")

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS = ("cpu", "gpu")
FORMAL_BACKEND = "hybrid"
NATIVE_FORMAL_BACKEND = "gpu"
FORMAL_GPU_CARGO_FEATURE = "formal-gpu"
FORMAL_GPU_EXECUTION_POLICY = "halo2-cpu-fft_icicle-bn254-gpu-msm-c16-chunk-1048576-v1"
FORMAL_GPU_FEATURES = (
    "dep:halo2_solidity_verifier",
    "ezkl",
    "gpu-accelerated",
    "no-banner",
    "parallel-poly-read",
    "reusable-verifier",
)
# Setup enables ICICLE but has no persisted per-stage invocation counters.
# A null GPU value preserves that uncertainty; proving has proof-bound evidence.
FORMAL_PHASE_BACKENDS = {
    "prepare_compile": {"cpu": True, "gpu": False},
    "setup": {"cpu": True, "gpu": None, "gpu_enabled": True},
    "public_statement": {"cpu": True, "gpu": False},
    "final_binding": {"cpu": True, "gpu": False},
    "witness_generation": {"cpu": True, "gpu": False},
    "model_commitment": {"cpu": True, "gpu": False},
    "prove": {"cpu": True, "gpu": True},
    "verify": {"cpu": True, "gpu": False},
}
CANONICAL_RELATION = "R_GAT-ezkl"
CANONICAL_PROOF_SCOPE = "full_gat"
CANONICAL_PRIVACY = "model_and_trace_hidden"
CANONICAL_STATEMENT_MODE = "external"
CANONICAL_OUTPUT_MODE = "public_logits"
NATIVE_SECURITY_PROFILE_ID = "halo2-kzg-shplonk"
MIN_FORMAL_SRS_BYTES = 4096
SRS_MANIFEST = ROOT / "resources" / "native_srs" / "srs_manifest.json"
NATIVE_SRS_SCHEMA_VERSION = 2
NATIVE_SRS_SOURCE = "official_ezkl_ceremony"
NATIVE_SRS_PROVISIONER = "native_get_srs"
NATIVE_SRS_URL_ROW = re.compile(r'pub const PUBLIC_SRS_URL:\s*&str\s*=\s*"([^"]+)"\s*;')
NATIVE_SRS_HASH_ROW = re.compile(r"\(\s*(\d+)\s*,\s*\"([0-9a-f]{64})\"\s*,")
CANONICAL_CONTRACT_KEYS = {
    "schema_version",
    "baseline",
    "contract_id",
    "generated_by",
    "adapter_version",
    "ready",
    "relation_id",
    "proof_scope",
    "public_statement_mode",
    "output_mode",
    "privacy_policy",
    "native_verify_required",
    "gpu_evidence_required",
    "proof_instance_binding",
    "official_chain",
    "prepare_command",
    "producer_command",
    "witness_command",
    "prove_command",
    "verify_command",
    "required_flags",
    "native_backend",
    "gpu_backend",
}
CONTRACT_CHAIN = ("gat-prepare", "produce-public-statement", "gat-witness", "prove", "verify")
FORBIDDEN_PATH_MARKERS = (
    "dummy",
    "gen-srs",
    "reduced-srs",
    "reduced_srs",
    "placeholder",
    "fake",
    "mock",
)


class ContractError(RuntimeError):
    """A missing or invalid formal-run prerequisite."""


def abort(message: str) -> NoReturn:
    raise ContractError(message)


_SHA256_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}


def _file_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _assert_hash_file_stable(path: Path, resolved: Path, stream: Any, before: os.stat_result, label: str) -> tuple[str, int, int, int, int, int]:
    try:
        resolved_after = path.resolve(strict=True)
        path_after = resolved_after.stat()
        after = os.fstat(stream.fileno())
    except OSError as exc:
        abort(f"{label} changed while its SHA-256 was being authenticated: {exc}")
    before_identity = _file_stat_identity(before)
    if not stat.S_ISREG(before.st_mode) or resolved_after != resolved or _file_stat_identity(path_after) != before_identity or _file_stat_identity(after) != before_identity:
        abort(f"{label} changed while its SHA-256 was being authenticated")
    return (str(resolved), *before_identity)


def _sha256_stable(path: Path, *, label: str, timeout_seconds: int | None = None, progress: bool = False) -> str:
    resolved = path.resolve(strict=True)
    started = time.monotonic()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _file_stat_identity(resolved.stat()) != _file_stat_identity(before):
            abort(f"{label} changed before its SHA-256 was being authenticated")
        total = before.st_size
        if progress:
            print(f"[ezkl-formal] authenticating {label} SHA-256 ({total:,} bytes; timeout={timeout_seconds}s)", flush=True)
        key = (str(resolved), *_file_stat_identity(before))
        cached = _SHA256_CACHE.get(key)
        if cached is not None:
            _assert_hash_file_stable(path, resolved, stream, before, label)
            if progress:
                print(f"[ezkl-formal] {label} SHA-256 reused stable process-local digest", flush=True)
            return cached
        hasher = hashlib.sha256()
        processed = 0
        next_report = started + 20
        block_size = 8 * 1024 * 1024 if progress else 1024 * 1024
        for block in iter(lambda: stream.read(block_size), b""):
            hasher.update(block)
            processed += len(block)
            now = time.monotonic()
            if timeout_seconds is not None and now - started > timeout_seconds:
                abort(f"{label} SHA-256 authentication exceeded {timeout_seconds}s after {processed:,}/{total:,} bytes")
            if progress and now >= next_report:
                print(f"[ezkl-formal] {label} SHA-256 progress {processed:,}/{total:,} bytes", flush=True)
                next_report = now + 20
        identity = _assert_hash_file_stable(path, resolved, stream, before, label)
        if processed != total:
            abort(f"{label} changed while its SHA-256 was being authenticated")
        value = hasher.hexdigest()
        _SHA256_CACHE[identity] = value
        if progress:
            print(f"[ezkl-formal] {label} SHA-256 authenticated ({processed:,} bytes)", flush=True)
        return value


def sha256(path: Path) -> str:
    return _sha256_stable(path, label=str(path))


def sha256_with_progress(path: Path, label: str, *, timeout_seconds: int = 1800) -> str:
    """Authenticate a large formal prerequisite and reuse only stable local digests."""
    require_file(path, label)
    return _sha256_stable(path, label=label, timeout_seconds=timeout_seconds, progress=True)


_FORMAL_CONTEXT_SEAL = object()


@dataclass(frozen=True, slots=True)
class AuthenticatedFormalContext:
    """In-process evidence created only after full formal preflight succeeds."""

    session_id: str
    resource_manifest_sha256: str
    resource_file_identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...]
    source_tree_sha256: str
    srs_manifest_sha256: str
    srs_path: str
    srs_sha256: str
    srs_file_identity: tuple[int, int, int, int, int]
    config_sha256s: tuple[tuple[str, str], ...]
    _seal: object = field(repr=False, compare=False)


def resource_file_identities() -> tuple[tuple[str, tuple[int, int, int, int, int]], ...]:
    """Snapshot manifest-listed data files without rereading their contents."""
    require_file(MANIFEST, "resource manifest")
    try:
        manifest = json.loads(MANIFEST.read_text())
    except json.JSONDecodeError as exc:
        abort(f"cannot snapshot resource files from invalid manifest: {exc}")
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        abort("resource manifest has no file entries to snapshot")
    paths: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            abort("resource manifest contains a malformed file entry")
        for key in ("path", "source_path"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                abort(f"resource manifest file entry lacks {key}")
            candidate = Path(value)
            path = candidate if candidate.is_absolute() else ROOT / candidate
            if path.is_symlink():
                abort(f"resource manifest {key} must not be a symlink: {path}")
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                abort(f"resource manifest {key} escapes the ezkl baseline: {value}")
            paths[relative] = resolved
    return resource_file_identities_for_paths(paths.values())


def resource_file_identities_for_paths(paths: Any) -> tuple[tuple[str, tuple[int, int, int, int, int]], ...]:
    """Snapshot an explicit set of baseline files using the manifest identity format."""
    by_relative: dict[str, Path] = {}
    for candidate in paths:
        path = Path(candidate)
        if path.is_symlink():
            abort(f"authenticated resource path must not be a symlink: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            abort(f"authenticated resource path escapes the ezkl baseline: {path}")
        by_relative[relative] = resolved
    snapshots: list[tuple[str, tuple[int, int, int, int, int]]] = []
    for relative, path in sorted(by_relative.items()):
        require_file(path, f"manifest-listed resource {relative}")
        info = path.stat()
        snapshots.append((relative, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)))
    return tuple(snapshots)


def create_authenticated_formal_context(
    session_id: str,
    srs_manifest: dict[str, Any],
    *,
    authenticated_resource_file_identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...],
    preflight_config_sha256s: tuple[tuple[str, str], ...],
) -> AuthenticatedFormalContext:
    if resource_file_identities() != authenticated_resource_file_identities:
        abort("manifest-listed resource files changed before the authenticated session context was sealed")
    entries = srs_manifest.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(DATASETS):
        abort("cannot create formal authentication context from an incomplete SRS manifest")
    first = entries[DATASETS[0]]
    srs_path = first.get("path")
    srs_sha256 = first.get("sha256")
    require_digest(srs_sha256, "authenticated SRS digest")
    if any(entry.get("path") != srs_path or entry.get("sha256") != srs_sha256 for entry in entries.values()):
        abort("formal SRS entries do not share one authenticated identity")
    srs_file = local_path(srs_path, "authenticated official SRS")
    require_formal_srs(srs_file)
    srs_stat = srs_file.stat()
    config_sha256s = tuple(preflight_config_sha256s)
    if tuple(dataset for dataset, _ in config_sha256s) != DATASETS:
        abort("formal authentication context has an incomplete preflight config identity")
    if any(sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg") != digest for dataset, digest in config_sha256s):
        abort("formal secure configs changed after the five-dataset prerequisite")
    return AuthenticatedFormalContext(
        session_id=session_id,
        resource_manifest_sha256=resource_manifest_hash(),
        resource_file_identities=authenticated_resource_file_identities,
        source_tree_sha256=source_tree_hash(),
        srs_manifest_sha256=sha256(SRS_MANIFEST),
        srs_path=srs_path,
        srs_sha256=srs_sha256,
        srs_file_identity=(srs_stat.st_dev, srs_stat.st_ino, srs_stat.st_size, srs_stat.st_mtime_ns, srs_stat.st_ctime_ns),
        config_sha256s=config_sha256s,
        _seal=_FORMAL_CONTEXT_SEAL,
    )


def require_authenticated_formal_context(
    context: AuthenticatedFormalContext,
    *,
    session_id: str,
    dataset: str,
    config: Path,
) -> None:
    if not isinstance(context, AuthenticatedFormalContext) or context._seal is not _FORMAL_CONTEXT_SEAL:
        abort("formal shared authentication context was not created by this process after preflight")
    if context.session_id != session_id or dataset not in DATASETS:
        abort("formal shared authentication context belongs to another session/dataset")
    if resource_file_identities() != context.resource_file_identities:
        abort("manifest-listed resource files changed after formal preflight; current session is invalid")
    srs_path = local_path(context.srs_path, "authenticated official SRS")
    srs_stat = require_formal_srs(srs_path).stat()
    if (srs_stat.st_dev, srs_stat.st_ino, srs_stat.st_size, srs_stat.st_mtime_ns, srs_stat.st_ctime_ns) != context.srs_file_identity:
        abort("authenticated official SRS file changed after formal preflight")
    expected_config = dict(context.config_sha256s).get(dataset)
    if expected_config is None or sha256(config) != expected_config:
        abort(f"{dataset} secure config changed after formal preflight")
    if resource_manifest_hash() != context.resource_manifest_sha256:
        abort("resource manifest changed after formal preflight")
    if sha256(SRS_MANIFEST) != context.srs_manifest_sha256:
        abort("native SRS manifest changed after formal preflight")
    if source_tree_hash() != context.source_tree_sha256:
        abort("formal source tree changed after preflight")


def _resolve_executable(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        abort(f"formal GPU toolchain prerequisite is missing from PATH: {name}")
    path = Path(found).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        abort(f"formal tool is not an executable regular file: {path}")
    return str(path)


def _tool_output(path: str, *args: str, timeout_seconds: int = 10) -> str:
    try:
        result = run_guarded_capture(
            [path, *args],
            cwd=ROOT,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        abort(f"cannot identify formal tool {path} within {timeout_seconds}s: {exc}")
    if result.returncode != 0 or not result.stdout.strip():
        abort(f"formal tool identity probe failed for {path}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def _gpu_inventory(nvidia_smi: str) -> list[dict[str, Any]]:
    try:
        probe = run_guarded_capture(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        abort(f"CUDA device probe failed within 10s: {exc}")
    if probe.returncode != 0 or not probe.stdout.strip():
        abort(f"CUDA device probe failed: {probe.stderr.strip() or 'no GPU reported'}")
    devices: list[dict[str, Any]] = []
    try:
        for row in csv.reader(io.StringIO(probe.stdout), skipinitialspace=True):
            fields = [field.strip() for field in row]
            if len(fields) != 5:
                abort(f"CUDA device probe returned an invalid identity row: {row}")
            index_text, uuid, name, driver, capability = fields
            index = int(index_text)
            if index < 0 or not all((uuid, name, driver, capability)):
                abort(f"CUDA device probe returned an incomplete identity row: {row}")
            devices.append(
                {
                    "index": index,
                    "uuid": uuid,
                    "name": name,
                    "driver_version": driver,
                    "compute_capability": capability,
                    "logical_device_id": 0,
                }
            )
    except (ValueError, csv.Error) as exc:
        abort(f"CUDA device probe returned malformed CSV: {exc}")
    indices = [device["index"] for device in devices]
    uuids = [device["uuid"] for device in devices]
    if not devices or len(indices) != len(set(indices)) or len(uuids) != len(set(uuids)):
        abort("CUDA device inventory is empty or contains duplicate physical identities")
    return sorted(devices, key=lambda device: device["index"])


def _selected_cuda_device_from_environment() -> dict[str, Any]:
    raw = os.environ.get("EZKL_FORMAL_SELECTED_CUDA_DEVICE")
    if not raw:
        abort("formal GPU execution has no selected physical CUDA device identity")
    try:
        selected = json.loads(raw)
    except json.JSONDecodeError as exc:
        abort(f"selected CUDA device identity is invalid JSON: {exc}")
    required = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
    if not isinstance(selected, dict) or set(selected) != required:
        abort("selected CUDA device identity has missing or unknown fields")
    if (
        any(not isinstance(selected[key], str) or not selected[key] for key in ("uuid", "name", "driver_version", "compute_capability"))
        or any(isinstance(selected[key], bool) or not isinstance(selected[key], int) or selected[key] < 0 for key in ("index", "logical_device_id"))
        or selected["logical_device_id"] != 0
    ):
        abort("selected CUDA device identity has invalid values")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != selected["uuid"] or os.environ.get("ICICLE_DEVICE_ID") != "0":
        abort("CUDA_VISIBLE_DEVICES/ICICLE_DEVICE_ID do not match the selected physical CUDA device")
    return selected


def select_formal_cuda_device(nvidia_smi: str) -> dict[str, Any]:
    """Choose one physical GPU and map it to native CUDA logical ordinal zero."""
    if os.environ.get("EZKL_FORMAL_SELECTED_CUDA_DEVICE"):
        abort("formal command received a reserved internal selected-device identity from its caller")
    inherited = {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "ICICLE_DEVICE_ID")}
    selected = _gpu_inventory(nvidia_smi)[0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    os.environ["ICICLE_DEVICE_ID"] = "0"
    os.environ["EZKL_FORMAL_SELECTED_CUDA_DEVICE"] = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    normalized = [name for name, value in inherited.items() if value and os.environ.get(name) != value]
    if normalized:
        print(f"[ezkl-formal] normalized inherited GPU selectors to the selected physical device: {', '.join(normalized)}", flush=True)
    print(
        f"[ezkl-formal] selected CUDA device index={selected['index']} uuid={selected['uuid']} "
        f"name={selected['name']} driver={selected['driver_version']} "
        f"compute_capability={selected['compute_capability']} logical_device_id=0",
        flush=True,
    )
    return selected


def _pinned_rust_toolchain() -> str:
    toolchain_path = require_file(ROOT / "rust-toolchain", "pinned Rust toolchain")
    try:
        toolchain_document = tomllib.loads(toolchain_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        abort(f"pinned Rust toolchain file is invalid TOML: {exc}")
    toolchain_settings = toolchain_document.get("toolchain")
    rust_toolchain = toolchain_settings.get("channel") if isinstance(toolchain_settings, dict) else None
    if rust_toolchain != "nightly-2025-12-01":
        abort(f"formal Rust source pin is unsupported: {rust_toolchain!r}")
    return rust_toolchain


def normalize_formal_execution_environment() -> dict[str, str]:
    """Replace ambient toolchain/backend selectors with this baseline's authenticated defaults."""
    rust_toolchain = _pinned_rust_toolchain()
    cargo_home = ROOT / ".cargo_local"
    if cargo_home.is_symlink() or (cargo_home.exists() and not cargo_home.is_dir()) or cargo_home.resolve() != cargo_home:
        abort(f"formal Cargo home must be a baseline-local directory: {cargo_home}")
    cargo_configs = tuple(cargo_home / name for name in ("config", "config.toml"))
    if any(path.exists() or path.is_symlink() for path in cargo_configs):
        abort("canonical formal Cargo home contains an unreviewed config override")
    normalized = []
    expected = {"RUSTUP_TOOLCHAIN": rust_toolchain, "CARGO_HOME": str(cargo_home)}
    for name, value in expected.items():
        if os.environ.get(name) != value:
            normalized.append(name)
        os.environ[name] = value
    cleared = (
        "CUDA_ARCH", "ENABLE_ICICLE_GPU", "EZKL_GAT_BACKEND", "ICICLE_SMALL_K", "ICICLE_SMALL_K_FFT",
        "ICICLE_BACKEND_INSTALL_DIR", "ICICLE_NATIVE_LIB_DIR", "ICICLE_NATIVE_MANIFEST_SHA256",
        "EZKL_NATIVE_PREPARE_BUDGET_BYTES", "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES", "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS",
    )
    for name in cleared:
        if os.environ.pop(name, None):
            normalized.append(name)
    if normalized:
        print(f"[ezkl-formal] normalized inherited formal environment selectors: {', '.join(sorted(set(normalized)))}", flush=True)
    return {"rust_toolchain": rust_toolchain, "cargo_home": str(cargo_home)}


def validate_formal_python_environment() -> dict[str, Any]:
    normalized_environment = normalize_formal_execution_environment()
    python_path = Path(sys.executable).resolve()
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        abort(f"formal entry interpreter is not an executable regular file: {python_path}")
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        abort(f"formal independent producer requires NumPy 1.26.4: {exc}")
    if np.__version__ != "1.26.4":
        abort(f"formal command requires NumPy 1.26.4; current={np.__version__}")
    unsupported = (
        "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "RUSTUP_HOME", "RUSTC",
        "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
        "CUDACXX", "CUDAHOSTCXX", "CUDA_HOME", "CUDA_PATH", "CC", "CXX", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CUDA_COMPILER_LAUNCHER",
        "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES", "CMAKE_GENERATOR", "CMAKE_MAKE_PROGRAM",
        "CMAKE_BUILD_PARALLEL_LEVEL", "EZKL_GAT_REQUIRE_GPU_EVIDENCE", "NVIDIA_VISIBLE_DEVICES",
        "EZKL_FORMAL_SELECTED_CUDA_DEVICE", "EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256", "EZKL_FORMAL_BUILD_RECEIPT_SHA256",
        "EZKL_WORKING_DIR", "EZKL_GAT_RUN_DIR",
    )
    changed = [name for name in unsupported if os.environ.get(name)]
    if changed:
        abort(f"formal command has unsupported environment overrides that can change identity: {', '.join(changed)}")
    rust_toolchain = normalized_environment["rust_toolchain"]
    rustup_path = _resolve_executable("rustup")
    cmake_path = _resolve_executable("cmake")
    nvcc_path = _resolve_executable("nvcc")
    nvidia_smi_path = _resolve_executable("nvidia-smi")
    cc_path = _resolve_executable("cc")
    cxx_path = _resolve_executable("c++")
    make_path = _resolve_executable("make")
    try:
        active = run_guarded_capture(
            [rustup_path, "show", "active-toolchain"],
            cwd=ROOT,
            env=os.environ.copy(),
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        abort(f"cannot identify the active Rust toolchain within 10s: {exc}")
    if active.returncode != 0 or not active.stdout.strip().startswith(rust_toolchain):
        abort(f"active Rust toolchain does not match rust-toolchain pin {rust_toolchain}: {active.stdout.strip() or active.stderr.strip()}")
    rustc_path = _tool_output(rustup_path, "which", "rustc", "--toolchain", rust_toolchain)
    cargo_path = _tool_output(rustup_path, "which", "cargo", "--toolchain", rust_toolchain)
    rustdoc_path = _tool_output(rustup_path, "which", "rustdoc", "--toolchain", rust_toolchain)
    toolchain_roots: set[Path] = set()
    for key, value in (("rustc", rustc_path), ("cargo", cargo_path), ("rustdoc", rustdoc_path)):
        candidate = Path(value).resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            abort(f"pinned {key} toolchain binary is unavailable: {candidate}")
        toolchain_root = candidate.parent.parent
        if not toolchain_root.name.startswith(rust_toolchain + "-"):
            abort(f"pinned {key} toolchain binary is outside the selected rustup toolchain: {value}")
        toolchain_roots.add(toolchain_root)
    if len(toolchain_roots) != 1:
        abort("pinned rustc and cargo binaries come from different rustup toolchains")
    rustc_version = _tool_output(rustc_path, "--version")
    cargo_version = _tool_output(cargo_path, "--version")
    rustdoc_version = _tool_output(rustdoc_path, "--version")
    if any("nightly" not in version for version in (rustc_version, cargo_version, rustdoc_version)):
        abort(
            f"Rust/Cargo/Rustdoc executable versions do not match {rust_toolchain}: "
            f"{rustc_version}; {cargo_version}; {rustdoc_version}"
        )
    cuda_home = str(Path(nvcc_path).resolve().parent.parent)
    identity = {
        "python_executable": str(python_path),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "rust_toolchain": rust_toolchain,
        "rustup_path": rustup_path,
        "rustup": _tool_output(rustup_path, "--version").splitlines()[0],
        "rustc_path": rustc_path,
        "rustc": rustc_version,
        "cargo_path": cargo_path,
        "cargo": cargo_version,
        "rustdoc_path": rustdoc_path,
        "rustdoc": rustdoc_version,
        "cmake_path": cmake_path,
        "cmake": _tool_output(cmake_path, "--version").splitlines()[0],
        "cmake_generator": "Unix Makefiles",
        "make_path": make_path,
        "make": _tool_output(make_path, "--version").splitlines()[0],
        "nvcc_path": nvcc_path,
        "cuda_home": cuda_home,
        "cuda_runtime": _tool_output(nvcc_path, "--version"),
        "nvidia_smi_path": nvidia_smi_path,
        "nvidia_smi": _tool_output(nvidia_smi_path, "--version").splitlines()[0],
        "cc_path": cc_path,
        "cc": _tool_output(cc_path, "--version").splitlines()[0],
        "cxx_path": cxx_path,
        "cxx": _tool_output(cxx_path, "--version").splitlines()[0],
        "path_sha256": sha256_bytes(os.environ.get("PATH", "").encode()),
    }
    print(
        f"[ezkl-formal] runtime identity PASS: interpreter={python_path} Python {identity['python_version']}, "
        f"NumPy {np.__version__}, Rust {rust_toolchain}, CUDA compiler={nvcc_path}",
        flush=True,
    )
    return identity


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def require_file(path: Path, label: str, *, regular: bool = True) -> Path:
    if not path.is_file():
        abort(f"missing {label}: {path}")
    if regular and path.is_symlink():
        abort(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size <= 0:
        abort(f"{label} is empty: {path}")
    return path


def local_path(value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        abort(f"{label} must be a baseline-local relative path: {value}")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        abort(f"{label} escapes the ezkl baseline root: {value}")
    return resolved


def reject_forbidden_path(path: Path, label: str) -> None:
    """Reject test-only, generated, reduced, or placeholder formal artifacts."""

    lowered = "/".join(path.parts).lower()
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        abort(f"{label} uses a forbidden dummy/generated/reduced/placeholder path: {path}")
    if "tests" in {part.lower() for part in path.parts} or "assets" in {part.lower() for part in path.parts}:
        abort(f"{label} must not come from tests/assets: {path}")


def require_formal_srs(path: Path, label: str = "trusted official ezkl SRS") -> Path:
    reject_forbidden_path(path, label)
    if path.name.lower().endswith((".dummy", ".placeholder")):
        abort(f"{label} is not an authentic SRS: {path}")
    result = require_file(path, label)
    if result.stat().st_size < MIN_FORMAL_SRS_BYTES:
        abort(f"{label} is too small for a full formal Halo2 KZG circuit: "
              f"{result.stat().st_size} bytes (minimum {MIN_FORMAL_SRS_BYTES})")
    return result


def resource_manifest_hash() -> str:
    require_file(MANIFEST, "resource manifest")
    return sha256(MANIFEST)


def _source_files() -> list[Path]:
    paths: list[Path] = []
    for relative in ("src", "gat_adapter", ".cargo", "vendor"):
        directory = ROOT / relative
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink() and path.relative_to(ROOT).as_posix() != "vendor/icicle/wrappers/rust/icicle-cuda-runtime/src/bindings.rs" and not {"target", "build", ".git", "__pycache__"}.intersection(path.relative_to(directory).parts))
    for relative in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain"):
        path = ROOT / relative
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    scripts = ROOT / "scripts"
    if scripts.is_dir():
        paths.extend(
            path for path in scripts.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".sh"}
        )
    setup_script = ROOT / "setup-gpu.sh"
    if setup_script.is_file() and not setup_script.is_symlink():
        paths.append(setup_script)
    return sorted(set(paths), key=lambda p: tuple(p.relative_to(ROOT).parts))


def source_tree_hash() -> str:
    files = _source_files()
    if not files:
        abort("no official ezkl source files found")
    h = hashlib.sha256()
    for path in files:
        relative = path.relative_to(ROOT).as_posix().encode()
        file_hash = sha256(path).encode()
        h.update(len(relative).to_bytes(8, "big"))
        h.update(relative)
        h.update(file_hash)
    return h.hexdigest()


def parse_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            abort(f"malformed config line {path}:{line_number}")
        key, value = (piece.strip() for piece in line.split("=", 1))
        if not key or key in values:
            abort(f"duplicate or empty config key {path}:{line_number}")
        values[key] = value
    return values


def validate_resources() -> None:
    validator = ROOT / "scripts" / "validate_resources.py"
    require_file(validator, "resource validator")
    timeout_seconds = 900
    print(f"[ezkl-formal] resource validation started (bounded timeout={timeout_seconds}s)", flush=True)
    process = spawn_guarded_process([sys.executable, "-u", str(validator), "--require-complete"], cwd=ROOT)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                returncode = process.wait(timeout=min(5.0, max(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                check_guardian_health()
    except subprocess.TimeoutExpired:
        _stop_process(process)
        abort(f"resource validation timed out after {timeout_seconds}s; validator process group was terminated")
    except KeyboardInterrupt:
        print("[ezkl-formal] interrupted during resource validation; terminating validator process group", file=sys.stderr, flush=True)
        _stop_process(process)
        raise
    except BaseException:
        _stop_process(process)
        raise
    terminated_leftovers = finish_guarded_process(process)
    if returncode != 0:
        abort(f"resource manifest validation failed with exit code {returncode}")
    if terminated_leftovers:
        abort("resource validator left descendants after exiting; the process group was cleaned")
    print("[ezkl-formal] resource validation completed", flush=True)


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if getattr(process, "_ezkl_process_group", None) is not None:
        terminate_guarded_process_locally(process)
        return
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def load_cli_contract() -> dict[str, Any]:
    require_file(CONTRACT, "GAT adapter CLI contract")
    try:
        contract = json.loads(CONTRACT.read_text())
    except json.JSONDecodeError as exc:
        abort(f"invalid GAT adapter CLI contract: {exc}")
    if not isinstance(contract, dict) or set(contract) != CANONICAL_CONTRACT_KEYS:
        missing = sorted(CANONICAL_CONTRACT_KEYS - set(contract)) if isinstance(contract, dict) else sorted(CANONICAL_CONTRACT_KEYS)
        extra = sorted(set(contract) - CANONICAL_CONTRACT_KEYS) if isinstance(contract, dict) else []
        abort(f"GAT adapter CLI contract schema mismatch; missing={missing}, extra={extra}")
    if contract.get("schema_version") != 1:
        abort("GAT adapter CLI contract has unsupported schema_version")
    if contract.get("baseline") != "ezkl":
        abort("GAT adapter CLI contract baseline is not ezkl")
    if contract.get("contract_id") != "ezkl-gat-native-v1":
        abort("GAT adapter CLI contract has an unknown contract_id")
    if contract.get("generated_by") != "ezkl::gat_adapter":
        abort("GAT adapter CLI contract was not generated by the Rust adapter")
    if not isinstance(contract.get("adapter_version"), str) or not contract["adapter_version"]:
        abort("GAT adapter CLI contract has no adapter_version")
    if contract.get("ready") is not True:
        abort("GAT adapter CLI contract is not marked ready=true")
    if contract.get("relation_id") != CANONICAL_RELATION:
        abort("GAT adapter CLI contract relation_id is not R_GAT-ezkl")
    if contract.get("proof_scope") != CANONICAL_PROOF_SCOPE:
        abort("GAT adapter contract is not full_gat")
    if contract.get("public_statement_mode") != CANONICAL_STATEMENT_MODE:
        abort("GAT adapter must consume an external public statement")
    if contract.get("output_mode") != CANONICAL_OUTPUT_MODE:
        abort("GAT adapter output mode is not public_logits")
    if contract.get("privacy_policy") != CANONICAL_PRIVACY:
        abort("GAT adapter privacy policy is not model_and_trace_hidden")
    if contract.get("native_verify_required") is not True:
        abort("GAT adapter contract does not require native verification")
    if contract.get("gpu_evidence_required") is not True:
        abort("GAT adapter contract does not require GPU evidence")
    stages = contract.get("official_chain")
    if not isinstance(stages, list) or tuple(stages) != CONTRACT_CHAIN:
        abort("GAT adapter contract does not enumerate the complete official ezkl chain")
    commands = {
        key: contract.get(key)
        for key in (
            "prepare_command",
            "producer_command",
            "witness_command",
            "prove_command",
            "verify_command",
        )
    }
    for key, command in commands.items():
        if not isinstance(command, str) or not COMMAND.fullmatch(command):
            abort(f"GAT adapter contract has an invalid {key}")
        if command in {"gen-srs", "get-srs", "mock", "gen-random-data", "calibrate-settings"}:
            abort(f"GAT adapter contract maps {key} to a forbidden non-formal command")
    if (
        commands["prepare_command"] != "gat-prepare" or commands["producer_command"] != "produce-public-statement" or
        commands["witness_command"] != "gat-witness"
    ):
        abort("GAT adapter contract must expose gat-prepare, external producer and gat-witness")
    if commands["prove_command"] != "prove" or commands["verify_command"] != "verify":
        abort("GAT adapter contract must use official prove and verify commands")
    if contract.get("proof_instance_binding") != "native_public_instances":
        abort("GAT adapter contract does not bind native public instances")
    if contract.get("native_backend") != "halo2-kzg-shplonk":
        abort("GAT adapter contract does not identify the native Halo2/KZG/SHPLONK backend")
    if contract.get("gpu_backend") != "icicle-cuda":
        abort("GAT adapter contract does not identify the ICICLE CUDA backend")
    flags = contract.get("required_flags")
    if not isinstance(flags, dict) or set(flags) != {
        "prepare",
        "producer",
        "witness",
        "prove",
        "verify",
    }:
        abort("GAT adapter contract required_flags schema mismatch")
    for stage, values in flags.items():
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.startswith("--") for value in values):
            abort(f"GAT adapter contract required_flags.{stage} is invalid")
    required_flag_sets = {
        "prepare": {
            "--dataset",
            "--phase",
            "--resource-manifest",
            "--output-dir",
            "--srs-path",
            "--public-statement",
            "--public-output",
            "--producer-manifest",
            "--gpu",
        },
        "producer": {
            "--dataset",
            "--resource-manifest",
            "--setup-context",
            "--output-dir",
            "--config",
            "--statement",
            "--public-output",
            "--producer-manifest",
        },
        "witness": {
            "--dataset",
            "--resource-manifest",
            "--compiled-circuit",
            "--statement",
            "--output",
            "--gpu",
        },
        "prove": {"--witness", "--compiled-circuit", "--pk-path", "--proof-path", "--srs-path", "--check-mode"},
        "verify": {
            "--settings-path",
            "--proof-path",
            "--vk-path",
            "--srs-path",
            "--statement",
            "--public-output",
            "--artifact-manifest",
            "--resource-manifest",
        },
    }
    for stage, required in required_flag_sets.items():
        if not required.issubset(set(flags[stage])):
            abort(f"GAT adapter contract required_flags.{stage} omits required official flags")
        if set(flags[stage]) != required:
            abort(f"GAT adapter contract required_flags.{stage} does not match the Rust Clap command exactly")
        if "--reduced-srs" in flags[stage] or "--reduced_srs" in flags[stage]:
            abort("GAT adapter contract enables reduced SRS verification")
    return contract


def validate_config(config: Path, dataset: str) -> dict[str, str]:
    require_file(config, "canonical secure config")
    values = parse_cfg(config)
    if values.get("dataset") != dataset:
        abort(f"config dataset mismatch: expected {dataset}, got {values.get('dataset')}")
    expected_roots = {
        "data_root": "resources/canonical/cache",
        "cache_root": "resources/canonical/cache",
        "checkpoint_bundle": f"resources/canonical/checkpoints/{'ogbn_arxiv_gat' if dataset == 'ogbn_arxiv' else dataset + '_gat'}",
    }
    for key, expected in expected_roots.items():
        if values.get(key) != expected:
            abort(f"{config} has non-canonical {key}: {values.get(key)!r}")
        if key == "checkpoint_bundle":
            require_file(ROOT / expected / "manifest.json", key)
            require_file(ROOT / expected / "tensors.txt", f"{dataset} canonical producer tensors")
        else:
            metadata = ROOT / expected / dataset / "meta.cfg"
            require_file(metadata, f"{dataset} dataset-scoped {key} metadata")
            validate_dataset_metadata(metadata, dataset, values)
    if values.get("auto_prepare_dataset") != "false":
        abort(f"{config} must disable dataset auto-preparation")
    srs_path = values.get("srs_path")
    if not srs_path:
        abort(f"{config} has no srs_path")
    srs = local_path(srs_path, "config srs_path")
    reject_forbidden_path(srs, "config srs_path")
    if not HEX64.fullmatch(values.get("trusted_srs_sha256", "")):
        abort(f"{config} has an invalid trusted_srs_sha256")
    if values.get("ezkl_security_profile_id") != NATIVE_SECURITY_PROFILE_ID:
        abort(f"{config} must pin native ezkl security profile {NATIVE_SECURITY_PROFILE_ID}")
    for key in ("static_table_id", "degree_bound_id"):
        if not values.get(key) or values[key] == "auto":
            abort(f"{config} must bind an explicit {key}")
    for key in ("static_table_digest", "degree_bound_digest"):
        require_digest(values.get(key, ""), f"{config} {key}")
    if "verifier_registry_digest" in values:
        abort(f"{config} still references the main-experiment verifier registry")
    return values


def validate_dataset_metadata(path: Path, dataset: str, config: dict[str, str]) -> dict[str, str]:
    values = parse_cfg(path)
    required = {"name", "num_nodes", "num_features", "num_classes", "input_preprocess_id"}
    missing = required - set(values)
    if missing:
        abort(f"{dataset} canonical metadata is missing required fields: {sorted(missing)}")
    if values["name"] != dataset:
        abort(f"{dataset} canonical metadata name mismatch: {values['name']!r}")
    for key in ("num_nodes", "num_features", "num_classes"):
        if not values[key].isdecimal() or int(values[key]) <= 0:
            abort(f"{dataset} canonical metadata {key} must be a positive integer")
    if not values["input_preprocess_id"]:
        abort(f"{dataset} canonical metadata has an empty input_preprocess_id")
    configured_preprocess = config.get("input_preprocess_id")
    if not configured_preprocess:
        abort(f"{dataset} secure config has no input_preprocess_id")
    if configured_preprocess != values["input_preprocess_id"]:
        abort(f"{dataset} config and canonical metadata input_preprocess_id differ")
    graph_count_text = values.get("graph_count", "1")
    if not graph_count_text.isdecimal() or int(graph_count_text) <= 0:
        abort(f"{dataset} canonical metadata graph_count must be a positive integer")
    expected_graph_count = 295 if dataset == "ppi" else 1
    if int(graph_count_text) != expected_graph_count:
        abort(f"{dataset} canonical metadata graph_count must be {expected_graph_count}")
    feature_keys = {"feature_format", "feature_dtype", "feature_shape", "feature_order", "feature_length", "feature_sha256"}
    present_feature_keys = feature_keys & set(values)
    if dataset in {"ppi", "ogbn_arxiv"}:
        if present_feature_keys != feature_keys or values["feature_format"] != "npy":
            abort(f"{dataset} NPY metadata schema is incomplete")
        if values["feature_dtype"] != "float32" or values["feature_order"] != "C":
            abort(f"{dataset} NPY metadata dtype/order is not canonical")
        shape = values["feature_shape"].split(",")
        if len(shape) != 2 or any(not item.isdecimal() or int(item) <= 0 for item in shape):
            abort(f"{dataset} NPY metadata feature_shape is invalid")
        if tuple(map(int, shape)) != (int(values["num_nodes"]), int(values["num_features"])):
            abort(f"{dataset} NPY metadata feature_shape disagrees with graph dimensions")
        if not values["feature_length"].isdecimal() or int(values["feature_length"]) != int(values["num_nodes"]) * int(values["num_features"]):
            abort(f"{dataset} NPY metadata feature_length disagrees with graph dimensions")
        require_digest(values["feature_sha256"], f"{dataset} NPY feature digest")
    elif present_feature_keys:
        abort(f"{dataset} text-feature metadata unexpectedly declares NPY fields")
    if dataset == "ppi":
        expected = {"task_type": "inductive_multi_graph_node_classification", "report_unit": "graph", "label_format": "multi_hot", "split_format": "node_masks"}
        if any(values.get(key) != expected_value for key, expected_value in expected.items()):
            abort("PPI canonical metadata task/label/split schema mismatch")
    if dataset == "ogbn_arxiv" and (values.get("task_type") != "transductive_node_classification" or values.get("report_unit") != "node"):
        abort("OGBN-Arxiv canonical metadata task/report schema mismatch")
    if "test_count" in values and (not values["test_count"].isdecimal() or int(values["test_count"]) <= 0):
        abort(f"{dataset} canonical metadata test_count must be a positive integer")
    return values


def require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        abort(f"{label} must be a lowercase SHA-256 digest")
    return value


def native_srs_contract() -> tuple[str, int, dict[int, str]]:
    """Read the official URL/hash contract from the current native ezkl source."""

    source = ROOT / "src" / "pfsys" / "srs.rs"
    hashes_source = ROOT / "src" / "srs_sha.rs"
    require_file(source, "native SRS URL source")
    require_file(hashes_source, "native SRS hash source")
    url_match = NATIVE_SRS_URL_ROW.search(source.read_text())
    if url_match is None or not url_match.group(1).startswith("https://"):
        abort("native SRS source has no authenticated HTTPS PUBLIC_SRS_URL")
    hashes = {int(key): value for key, value in NATIVE_SRS_HASH_ROW.findall(hashes_source.read_text())}
    if not hashes or set(hashes) != set(range(1, max(hashes) + 1)):
        abort("native SRS hash source has no contiguous public logrows range")
    return url_match.group(1), max(hashes), hashes


def validate_native_srs_manifest() -> dict[str, Any]:
    """Validate native get-srs provenance and every config/path/digest binding."""

    manifest = load_json(SRS_MANIFEST, "native SRS manifest")
    expected_top = {
        "schema_version",
        "baseline",
        "srs_type",
        "shared",
        "max_logrows",
        "source",
        "provisioner",
        "source_url",
        "hash_source",
        "native_hash_verified",
        "entries",
    }
    require_exact_keys(manifest, expected_top, "native SRS manifest")
    if manifest["schema_version"] != NATIVE_SRS_SCHEMA_VERSION or manifest["baseline"] != "ezkl":
        abort("native SRS manifest schema/baseline mismatch")
    if manifest["srs_type"] != NATIVE_SECURITY_PROFILE_ID or manifest["shared"] is not True:
        abort("native SRS manifest is not shared Halo2 KZG/SHPLONK material")
    if manifest["source"] != NATIVE_SRS_SOURCE or manifest["provisioner"] != NATIVE_SRS_PROVISIONER:
        abort("native SRS manifest is not produced by native get-srs")
    if manifest["hash_source"] != "src/srs_sha.rs" or manifest["native_hash_verified"] is not True:
        abort("native SRS manifest lacks the native embedded-hash attestation")
    source_url, expected_k, hashes = native_srs_contract()
    if manifest["max_logrows"] != expected_k or manifest["source_url"] != f"{source_url}{expected_k}.srs":
        abort("native SRS manifest logrows/source URL is not source-matched")
    entries = manifest["entries"]
    if not isinstance(entries, dict) or set(entries) != set(DATASETS):
        abort(f"native SRS manifest must contain exactly {list(DATASETS)} entries")
    expected_entry_keys = {
        "dataset",
        "srs_id",
        "srs_type",
        "path",
        "sha256",
        "size",
        "logrows",
        "source",
        "provisioner",
        "source_url",
        "hash_source",
        "native_hash_verified",
        "trusted",
        "reduced",
        "generated",
    }
    shared_path: str | None = None
    shared_id: str | None = None
    shared_digest: str | None = None
    file_digests: dict[Path, str] = {}
    for dataset in DATASETS:
        entry = entries[dataset]
        require_exact_keys(entry, expected_entry_keys, f"native SRS entry {dataset}")
        if entry["dataset"] != dataset or entry["srs_type"] != "trusted_srs":
            abort(f"native SRS entry {dataset} identity/type mismatch")
        if entry["source"] != NATIVE_SRS_SOURCE or entry["provisioner"] != NATIVE_SRS_PROVISIONER:
            abort(f"native SRS entry {dataset} is not native get-srs provenance")
        if entry["logrows"] != expected_k or entry["source_url"] != manifest["source_url"]:
            abort(f"native SRS entry {dataset} logrows/source URL mismatch")
        if entry["hash_source"] != "src/srs_sha.rs" or entry["native_hash_verified"] is not True:
            abort(f"native SRS entry {dataset} lacks native hash verification")
        if entry["trusted"] is not True or entry["reduced"] is not False or entry["generated"] is not False:
            abort(f"native SRS entry {dataset} is not trusted full-size material")
        require_digest(entry["sha256"], f"native SRS entry {dataset} sha256")
        if entry["sha256"] != hashes[expected_k]:
            abort(f"native SRS entry {dataset} digest is not the embedded public hash")
        if not isinstance(entry["size"], int) or entry["size"] < MIN_FORMAL_SRS_BYTES:
            abort(f"native SRS entry {dataset} has an invalid/minimal size")
        path = local_path(entry["path"], f"native SRS entry {dataset} path")
        reject_forbidden_path(path, f"native SRS entry {dataset} path")
        require_formal_srs(path, f"native SRS entry {dataset} file")
        if path not in file_digests:
            file_digests[path] = sha256_with_progress(path, f"official shared SRS {path.name}")
        if path.stat().st_size != entry["size"] or file_digests[path] != entry["sha256"]:
            abort(f"native SRS entry {dataset} file digest/size mismatch")
        if shared_path is None:
            shared_path = entry["path"]
            shared_id = entry["srs_id"]
            shared_digest = entry["sha256"]
        elif (entry["path"], entry["srs_id"], entry["sha256"]) != (shared_path, shared_id, shared_digest):
            abort("native SRS entries do not share one authenticated cache")
        config = ROOT / "resources" / "canonical" / "configs" / f"{dataset}_secure.cfg"
        values = parse_cfg(config)
        if values.get("srs_type") != "trusted_srs":
            abort(f"{config} is not a trusted_srs config")
        if values.get("srs_id") != entry["srs_id"] or values.get("srs_path") != entry["path"]:
            abort(f"{config} SRS id/path binding does not match native manifest")
        if values.get("trusted_srs_sha256") != entry["sha256"]:
            abort(f"{config} trusted_srs_sha256 does not match native manifest")
    return manifest


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        abort(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        abort(f"{label} must be a JSON object")
    return value


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        abort(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        abort(f"{label} schema mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return value


def validate_public_statement(
    path: Path,
    *,
    dataset: str,
    circuit_ref: dict[str, str] | None = None,
    public_output_sha256: str | None = None,
    quant_cfg_id: str | None = None,
    static_tables_id: str | None = None,
) -> dict[str, Any]:
    statement = load_json(path, "external public statement")
    required = {
        "schema_version",
        "relation_id",
        "dataset",
        "proof_scope",
        "privacy_policy",
        "public_statement_mode",
        "output_mode",
        "model_arch_id",
        "quant_cfg_id",
        "static_table_id",
        "degree_bound_id",
        "ezkl_security_profile_id",
        "public_ezkl",
        "output_digest",
        "public_output_sha256",
        "statement_digest",
    }
    require_exact_keys(statement, required, "external public statement")
    if statement["schema_version"] != 1 or statement["relation_id"] != CANONICAL_RELATION:
        abort("external public statement has an unsupported schema or relation")
    if statement["dataset"] != dataset or statement["proof_scope"] != CANONICAL_PROOF_SCOPE:
        abort("external public statement dataset/proof scope mismatch")
    if statement["privacy_policy"] != CANONICAL_PRIVACY:
        abort("external public statement privacy policy mismatch")
    if statement["public_statement_mode"] != CANONICAL_STATEMENT_MODE:
        abort("external public statement mode is not external")
    if statement["output_mode"] != CANONICAL_OUTPUT_MODE:
        abort("external public statement output mode is not public_logits")
    for key in ("model_arch_id", "quant_cfg_id", "static_table_id", "degree_bound_id", "ezkl_security_profile_id"):
        if not isinstance(statement[key], str) or not statement[key]:
            abort(f"external public statement has no {key}")
    require_digest(statement["output_digest"], "external public statement output_digest")
    require_digest(statement["public_output_sha256"], "external public statement public_output_sha256")
    require_digest(statement["statement_digest"], "external public statement statement_digest")
    public = require_exact_keys(
        statement["public_ezkl"],
        {
            "topology",
            "static_tables",
            "quant",
            "ledger",
            "circuit",
            "model_hash",
            "model",
            "feature_table",
            "node_ids",
            "src",
            "dst",
            "node_ptr",
            "edge_ptr",
            "y",
        },
        "external public statement public_ezkl",
    )
    for key in ("topology", "static_tables", "quant", "ledger"):
        identity = require_exact_keys(public[key], {"id", "digest"}, f"statement {key}")
        if not isinstance(identity["id"], str) or not identity["id"]:
            abort(f"statement {key}.id is empty")
        require_digest(identity["digest"], f"statement {key}.digest")
    circuit = require_exact_keys(
        public["circuit"],
        {
            "relation_id",
            "compiled_circuit_id",
            "compiled_circuit_digest",
            "settings_id",
            "settings_digest",
            "vk_id",
            "vk_digest",
            "model_graph_id",
            "model_graph_digest",
            "ezkl_security_profile_id",
        },
        "statement circuit_ref",
    )
    parsed_ref = require_circuit_ref(circuit, "external public statement circuit_ref")
    if circuit["ezkl_security_profile_id"] != statement["ezkl_security_profile_id"]:
        abort("external public statement security profile mismatch")
    model = require_exact_keys(public["model"], {"hash", "hash_domain", "canonical_model_digest"}, "statement model")
    if model["hash"] != public["model_hash"] or not isinstance(public["model_hash"], str) or not public["model_hash"]:
        abort("statement native model H(W) binding mismatch")
    if (
        len(public["model_hash"]) != 66 or not public["model_hash"].startswith("0x") or
        any(character not in "0123456789abcdef" for character in public["model_hash"][2:])
    ):
        abort("statement native model H(W) is not a canonical Halo2 field encoding")
    require_digest(model["canonical_model_digest"], "statement canonical model digest")
    if not isinstance(public["feature_table"], list) or not public["feature_table"]:
        abort("statement feature_table is empty")
    if not isinstance(public["node_ids"], list) or not public["node_ids"]:
        abort("statement node_ids is empty")
    if not isinstance(public["y"], list) or not public["y"]:
        abort("statement public Y is empty")
    for key in ("feature_table", "node_ids", "src", "dst", "node_ptr", "edge_ptr"):
        values = public[key]
        if not isinstance(values, list) or not values:
            abort(f"statement {key} is empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            abort(f"statement {key} must contain integers")
        if any(value < 0 for value in values):
            abort(f"statement {key} contains a negative value")
    if len(public["src"]) != len(public["dst"]):
        abort("statement src/dst lengths disagree")
    if len(public["node_ptr"]) < 2 or len(public["edge_ptr"]) < 2:
        abort("statement node_ptr/edge_ptr must contain at least two boundaries")
    if public["node_ptr"][0] != 0 or public["edge_ptr"][0] != 0:
        abort("statement graph pointers must start at zero")
    if public["node_ptr"][-1] != len(public["node_ids"]) or public["edge_ptr"][-1] != len(public["src"]):
        abort("statement graph pointers do not cover the public graph")
    if any(a >= b for a, b in zip(public["node_ptr"], public["node_ptr"][1:])) or any(a >= b for a, b in zip(public["edge_ptr"], public["edge_ptr"][1:])):
        abort("statement graph pointers must be strictly increasing")
    if len(set(public["node_ids"])) != len(public["node_ids"]):
        abort("statement node_ids contain duplicates")
    for node_start, node_end, edge_start, edge_end in zip(public["node_ptr"], public["node_ptr"][1:], public["edge_ptr"], public["edge_ptr"][1:]):
        if any(
            src < node_start or src >= node_end or dst < node_start or dst >= node_end
            for src, dst in zip(public["src"][edge_start:edge_end], public["dst"][edge_start:edge_end])
        ):
            abort("statement edge crosses a graph batch boundary")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in public["y"]):
        abort("statement public Y must contain integers")
    for value in public["y"]:
        if value < -(1 << 127) or value >= (1 << 127):
            abort("statement public Y contains an out-of-range i128")
    output_bytes = bytearray(b"ezkl-gat-public-y-v1")
    for value in public["y"]:
        output_bytes.extend(value.to_bytes(16, "little", signed=True))
    if statement["output_digest"] != sha256_bytes(bytes(output_bytes)):
        abort("statement output_digest does not match public Y")
    if public["quant"]["id"] != statement["quant_cfg_id"] or public["static_tables"]["id"] != statement["static_table_id"] or public["ledger"][
        "id"] != statement["degree_bound_id"]:
        abort("statement registry ids do not match public boundary")
    if quant_cfg_id is not None and statement["quant_cfg_id"] != quant_cfg_id:
        abort("external public statement quant_cfg_id does not match the canonical config")
    if static_tables_id is not None and statement["static_table_id"] != static_tables_id:
        abort("external public statement static_table_id does not match the canonical config")
    require_digest(statement["public_output_sha256"], "external public statement public_output_sha256")
    if public_output_sha256 is not None and statement["public_output_sha256"] != public_output_sha256:
        abort("external public statement does not bind the selected public output Y")
    if circuit_ref is not None and parsed_ref != circuit_ref:
        abort("external public statement circuit_ref does not match the run circuit")
    return statement


def validate_public_output(path: Path, *, dataset: str) -> Any:
    require_file(path, "external public output Y")
    try:
        output = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        abort(f"external public output Y is not valid JSON: {exc}")
    if output in (None, [], {}):
        abort("external public output Y is empty")
    if not isinstance(output, dict) or set(output) != {"dataset", "Y"}:
        abort("external public output must use the canonical {dataset,Y} object")
    if output.get("dataset") != dataset:
        abort("external public output dataset mismatch")
    payload = output.get("Y")
    if payload in (None, [], {}):
        abort("external public output Y payload is empty")
    return output


def public_output_values(path: Path, *, dataset: str) -> list[int]:
    """Return the one-dimensional integer Y vector used by the Rust adapter."""

    output = validate_public_output(path, dataset=dataset)
    payload = output["Y"]
    if not isinstance(payload, list) or not payload:
        abort("external public output Y must be a non-empty one-dimensional array")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in payload):
        abort("external public output Y must contain integers")
    if any(value < -(1 << 127) or value >= (1 << 127) for value in payload):
        abort("external public output Y contains an out-of-range i128")
    return payload


def validate_artifact_identity(
    artifact_manifest: Path,
    *,
    dataset: str,
    backend: str,
    binary: Path,
    srs: Path,
    settings: Path,
    compiled: Path,
    vk: Path,
    pk: Path,
) -> dict[str, Any]:
    manifest = load_json(artifact_manifest, "artifact identity manifest")
    required = {
        "schema_version",
        "baseline",
        "relation_id",
        "dataset",
        "backend",
        "proof_scope",
        "privacy_policy",
        "public_statement_mode",
        "output_mode",
        "quant_cfg_id",
        "static_tables_id",
        "circuit_ref",
        "compiled_circuit_id",
        "settings_id",
        "vk_id",
        "binary_sha256",
        "source_tree_sha256",
        "resource_manifest_sha256",
        "srs_sha256",
        "settings_sha256",
        "compiled_circuit_sha256",
        "vk_sha256",
        "pk_sha256",
        "srs_provenance",
        "files",
        "source_snapshot",
        "statement_sha256",
        "producer_manifest_sha256",
        "ezkl_security_profile_id",
    }
    require_exact_keys(manifest, required, "artifact identity manifest")
    if manifest["schema_version"] != 1 or manifest["dataset"] != dataset or manifest["backend"] != backend:
        abort("artifact identity manifest dataset/backend/schema mismatch")
    if (
        manifest["baseline"] != "ezkl" or manifest["proof_scope"] != CANONICAL_PROOF_SCOPE or manifest["relation_id"] != CANONICAL_RELATION or
        manifest["privacy_policy"] != CANONICAL_PRIVACY or manifest["public_statement_mode"] != CANONICAL_STATEMENT_MODE or
        manifest["output_mode"] != CANONICAL_OUTPUT_MODE
    ):
        abort("artifact identity manifest is not full_gat")
    circuit_ref = require_circuit_ref(manifest["circuit_ref"], "artifact identity circuit_ref")
    if circuit_ref["compiled_circuit_id"] != manifest["compiled_circuit_id"]:
        abort("artifact circuit_ref compiled_circuit_id mismatch")
    if circuit_ref["settings_id"] != manifest["settings_id"]:
        abort("artifact circuit_ref settings_id mismatch")
    if circuit_ref["vk_id"] != manifest["vk_id"]:
        abort("artifact circuit_ref vk_id mismatch")
    for key in ("quant_cfg_id", "static_tables_id", "ezkl_security_profile_id"):
        if not isinstance(manifest[key], str) or not manifest[key] or "pending" in manifest[key].lower():
            abort(f"artifact identity manifest has invalid {key}")
    circuit_document = manifest["circuit_ref"]
    if circuit_document["ezkl_security_profile_id"] != manifest["ezkl_security_profile_id"]:
        abort("artifact circuit_ref security profile mismatch")
    srs_provenance = manifest["srs_provenance"]
    srs_provenance = require_exact_keys(
        srs_provenance,
        {"source", "trusted", "reduced", "generated", "ceremony_id", "sha256"},
        "artifact SRS provenance",
    )
    if srs_provenance.get("source") != "official_ezkl_ceremony":
        abort("artifact SRS is not identified as official ezkl ceremony material")
    if srs_provenance.get("trusted") is not True or srs_provenance.get("reduced") is not False:
        abort("artifact SRS provenance is not trusted full-size material")
    if srs_provenance.get("generated") is not False:
        abort("artifact SRS provenance indicates generated material")
    if not isinstance(srs_provenance.get("ceremony_id"), str) or not srs_provenance["ceremony_id"]:
        abort("artifact SRS provenance has no ceremony_id")
    if srs_provenance.get("sha256") != manifest["srs_sha256"]:
        abort("artifact SRS provenance digest does not match artifact identity")
    require_digest(manifest["srs_sha256"], "artifact srs_sha256")
    require_digest(srs_provenance["sha256"], "artifact SRS provenance sha256")
    source_snapshot = require_exact_keys(manifest["source_snapshot"], {"root", "git_commit", "git_tree"}, "artifact source_snapshot")
    for key in ("root", "git_commit", "git_tree"):
        if not isinstance(source_snapshot[key], str) or not source_snapshot[key]:
            abort(f"artifact source_snapshot.{key} is empty")
    artifact_files = manifest["files"]
    if not isinstance(artifact_files, list) or not artifact_files:
        abort("artifact identity manifest has no authenticated files")
    artifact_root = artifact_manifest.parent
    file_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(artifact_files):
        record = require_exact_keys(record, {"path", "size", "sha256"}, f"artifact files[{index}]")
        relative = record["path"]
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or Path(relative).parts[0] == "."):
            abort(f"artifact files[{index}].path is not baseline-local")
        if relative in file_records:
            abort(f"artifact files contains duplicate path: {relative}")
        require_digest(record["sha256"], f"artifact files[{index}].sha256")
        if isinstance(record["size"], bool) or not isinstance(record["size"], int) or record["size"] <= 0:
            abort(f"artifact files[{index}].size is invalid")
        resolved = artifact_root / relative
        cursor = artifact_root
        for component in Path(relative).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                abort(f"artifact file path contains a symlink: {relative}")
        require_file(resolved, f"artifact file {relative}")
        if resolved.stat().st_size != record["size"] or sha256(resolved) != record["sha256"]:
            abort(f"artifact file digest/size mismatch: {relative}")
        file_records[relative] = record
    files = {
        "binary_sha256": binary,
        "srs_sha256": srs,
        "settings_sha256": settings,
        "compiled_circuit_sha256": compiled,
        "vk_sha256": vk,
        "pk_sha256": pk,
    }
    for key, path in files.items():
        require_digest(manifest[key], f"artifact {key}")
        if manifest[key] != sha256(path):
            abort(f"artifact identity mismatch for {key}: {path}")
    require_digest(manifest["statement_sha256"], "artifact statement_sha256")
    require_digest(manifest["producer_manifest_sha256"], "artifact producer_manifest_sha256")
    for name, digest in (
        ("producer_context.json", sha256(artifact_root / "producer_context.json")),
        ("producer_manifest.json", manifest["producer_manifest_sha256"]),
    ):
        record = file_records.get(name)
        if record is None or record["sha256"] != digest:
            abort(f"artifact file {name} is not bound to the producer chain")
    if manifest["source_tree_sha256"] != source_tree_hash():
        abort("artifact identity source_tree_sha256 does not match current source")
    if manifest["resource_manifest_sha256"] != resource_manifest_hash():
        abort("artifact identity resource_manifest_sha256 does not match current resources")
    for key in ("source_tree_sha256", "resource_manifest_sha256"):
        require_digest(manifest[key], key)
    return manifest


def formal_build_environment_identity(
    runtime_identity: dict[str, Any],
    selected_device: dict[str, Any],
) -> dict[str, Any]:
    required_runtime = {
        "python_executable", "python_version", "numpy_version", "rust_toolchain", "rustup_path", "rustc_path", "rustc",
        "cargo_path", "cargo", "rustdoc_path", "rustdoc", "cmake_path", "cmake", "nvcc_path", "cuda_home", "cuda_runtime", "nvidia_smi_path",
        "rustup", "nvidia_smi", "cc_path", "cc", "cxx_path", "cxx", "make_path", "make", "path_sha256",
    }
    missing = required_runtime - set(runtime_identity)
    if missing:
        abort(f"formal GPU build environment is missing required identity fields: {sorted(missing)}")
    device_required = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
    if not isinstance(selected_device, dict) or set(selected_device) != device_required or selected_device != _selected_cuda_device_from_environment():
        abort("formal GPU build environment does not match the selected CUDA device")
    identity = {
        "schema_version": 1,
        "python_executable": runtime_identity["python_executable"],
        "python_version": runtime_identity["python_version"],
        "numpy_version": runtime_identity["numpy_version"],
        "rust_toolchain": runtime_identity["rust_toolchain"],
        "rustup_path": runtime_identity["rustup_path"],
        "rustup": runtime_identity["rustup"],
        "rustc_path": runtime_identity["rustc_path"],
        "rustc": runtime_identity["rustc"],
        "cargo_path": runtime_identity["cargo_path"],
        "cargo": runtime_identity["cargo"],
        "rustdoc_path": runtime_identity["rustdoc_path"],
        "rustdoc": runtime_identity["rustdoc"],
        "cargo_home": str(ROOT / ".cargo_local"),
        "cargo_target_dir": str(ROOT / "target/gpu"),
        "cargo_mode": "offline-release",
        "cmake_path": runtime_identity["cmake_path"],
        "cmake": runtime_identity["cmake"],
        "cmake_generator": runtime_identity["cmake_generator"],
        "make_path": runtime_identity["make_path"],
        "make": runtime_identity["make"],
        "nvcc_path": runtime_identity["nvcc_path"],
        "cuda_home": runtime_identity["cuda_home"],
        "cuda_runtime": runtime_identity["cuda_runtime"],
        "nvidia_smi_path": runtime_identity["nvidia_smi_path"],
        "nvidia_smi": runtime_identity["nvidia_smi"],
        "cc_path": runtime_identity["cc_path"],
        "cc": runtime_identity["cc"],
        "cxx_path": runtime_identity["cxx_path"],
        "cxx": runtime_identity["cxx"],
        "path_sha256": runtime_identity["path_sha256"],
        "cargo_feature_profile": FORMAL_GPU_CARGO_FEATURE,
        "cuda_architecture": f"sm_{selected_device['compute_capability'].replace('.', '')}",
        "selected_cuda_device": selected_device,
        "features": list(FORMAL_GPU_FEATURES),
    }
    identity["build_environment_sha256"] = canonical_json_digest(identity)
    return identity


def native_backend_identity() -> dict[str, Any]:
    manifest_path = ROOT / "artifacts/build/icicle-cuda/native_backend_manifest.json"
    manifest = load_json(manifest_path, "native ICICLE backend manifest")
    require_exact_keys(
        manifest,
        {
            "schema_version", "halo2_commit", "frontend_commit", "abi", "device", "architecture", "runtime", "libraries",
            "native_source_sha256", "selected_cuda_device", "driver_version", "cuda_compiler_path", "cuda_toolkit_root", "nvidia_smi",
            "cmake_path", "cmake", "cmake_generator", "make_path", "make", "cc_path", "cc", "cxx_path", "cxx", "nvidia_smi_path",
        },
        "native ICICLE backend manifest",
    )
    if manifest["schema_version"] != 2 or manifest["halo2_commit"] != "f441c920be45f8f05d2c06a173d82e8885a5ed4d" or manifest["frontend_commit"] != "5dfe006a0f1bc62ea82ca297709bbf3d22a2ca25":
        abort("native ICICLE backend is not the pinned Halo2/ICICLE source")
    source_root = ROOT / "vendor/icicle/icicle"
    source_hash = hashlib.sha256()
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            abort(f"native ICICLE source must not be symlinked: {path}")
        if path.is_file():
            relative = path.relative_to(source_root).as_posix().encode()
            source_hash.update(len(relative).to_bytes(8, "big"))
            source_hash.update(relative)
            source_hash.update(sha256(path).encode())
    if manifest["native_source_sha256"] != source_hash.hexdigest():
        abort("native ICICLE libraries were not built from the current authenticated CUDA source")
    libraries = manifest["libraries"]
    expected = {f"artifacts/build/icicle-cuda/lib/libingo_{kind}_bn254.a" for kind in ("field", "curve")}
    if not isinstance(libraries, dict) or set(libraries) != expected:
        abort("native ICICLE backend does not identify exactly its BN254 CUDA libraries")
    for name, expected_sha in libraries.items():
        require_digest(expected_sha, "native ICICLE library SHA-256")
        if sha256(require_file(local_path(name, "native ICICLE library"), "native ICICLE library")) != expected_sha:
            abort(f"native ICICLE backend library identity is stale: {name}")
    for field in ("abi", "device", "architecture", "runtime"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            abort(f"native ICICLE backend lacks {field}")
    selected = manifest["selected_cuda_device"]
    required_selected = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
    if not isinstance(selected, dict) or set(selected) != required_selected:
        abort("native ICICLE backend manifest lacks the selected CUDA device identity")
    live_selected = _selected_cuda_device_from_environment()
    if selected != live_selected:
        abort("native ICICLE backend was built for a different selected CUDA device")
    if manifest["device"] != selected["name"] or manifest["driver_version"] != selected["driver_version"]:
        abort("native ICICLE backend device/driver identity does not match its selected device")
    return {
        "native_execution_policy": FORMAL_GPU_EXECUTION_POLICY,
        "native_backend_manifest_sha256": sha256(manifest_path),
        "native_backend_libraries": libraries,
        "cuda_architecture": manifest["architecture"],
        "cuda_runtime": manifest["runtime"],
        "selected_cuda_device": selected,
        "driver_version": manifest["driver_version"],
        "cuda_compiler_path": manifest["cuda_compiler_path"],
        "cuda_toolkit_root": manifest["cuda_toolkit_root"],
        "cmake_path": manifest["cmake_path"],
        "cmake": manifest["cmake"],
        "cmake_generator": manifest["cmake_generator"],
        "make_path": manifest["make_path"],
        "make": manifest["make"],
        "cc_path": manifest["cc_path"],
        "cc": manifest["cc"],
        "cxx_path": manifest["cxx_path"],
        "cxx": manifest["cxx"],
        "nvidia_smi_path": manifest["nvidia_smi_path"],
        "nvidia_smi": manifest["nvidia_smi"],
    }


def validate_gpu_environment() -> dict[str, Any]:
    expected_dir = ROOT / "artifacts/build/icicle-cuda"
    for path in (expected_dir, expected_dir.parent, expected_dir.parent.parent):
        if path.is_symlink():
            abort(f"native ICICLE installation must not be symlinked: {path}")
    install_dir = os.environ.get("ICICLE_BACKEND_INSTALL_DIR", str(expected_dir))
    if not install_dir:
        abort("GPU execution requires the baseline-local ICICLE backend")
    directory = Path(install_dir).resolve()
    if directory != expected_dir.resolve():
        abort("formal GPU backend must use the baseline-local source-matched installation")
    os.environ["ICICLE_BACKEND_INSTALL_DIR"] = str(directory)
    os.environ["ICICLE_NATIVE_LIB_DIR"] = str(directory / "lib")
    if not directory.is_dir() or directory.is_symlink():
        abort(f"ICICLE_BACKEND_INSTALL_DIR is not a regular directory: {install_dir}")
    identity = native_backend_identity()
    os.environ["ICICLE_NATIVE_MANIFEST_SHA256"] = identity["native_backend_manifest_sha256"]
    for name in ("libingo_field_bn254.a", "libingo_curve_bn254.a"):
        require_file(directory / "lib" / name, "native ICICLE CUDA library")
    selected = _selected_cuda_device_from_environment()
    nvcc_path = _resolve_executable("nvcc")
    cmake_path = _resolve_executable("cmake")
    cc_path = _resolve_executable("cc")
    cxx_path = _resolve_executable("c++")
    nvidia_smi_path = _resolve_executable("nvidia-smi")
    make_path = _resolve_executable("make")
    runtime_identity = {
        "nvcc_path": nvcc_path,
        "cuda_home": str(Path(nvcc_path).resolve().parent.parent),
        "cuda_runtime": _tool_output(nvcc_path, "--version"),
        "cmake_path": cmake_path,
        "cmake": _tool_output(cmake_path, "--version").splitlines()[0],
        "cc_path": cc_path,
        "cc": _tool_output(cc_path, "--version").splitlines()[0],
        "cxx_path": cxx_path,
        "cxx": _tool_output(cxx_path, "--version").splitlines()[0],
        "nvidia_smi_path": nvidia_smi_path,
        "nvidia_smi": _tool_output(nvidia_smi_path, "--version").splitlines()[0],
        "make_path": make_path,
        "make": _tool_output(make_path, "--version").splitlines()[0],
    }
    live_devices = _gpu_inventory(runtime_identity["nvidia_smi_path"])
    live_selected = next((device for device in live_devices if device["uuid"] == selected["uuid"]), None)
    if live_selected is None or live_selected != selected:
        abort("selected physical CUDA device no longer matches the live GPU inventory")
    device_name = selected["name"]
    driver_version = selected["driver_version"]
    compute_capability = selected["compute_capability"]
    live_architecture = f"sm_{compute_capability.replace('.', '')}"
    if identity["cuda_architecture"] != live_architecture:
        abort(f"native ICICLE architecture {identity['cuda_architecture']} does not match live GPU {live_architecture}")
    manifest_path = directory / "native_backend_manifest.json"
    manifest = load_json(manifest_path, "native ICICLE backend manifest")
    if manifest.get("device") != device_name or manifest.get("selected_cuda_device") != selected:
        abort("native ICICLE selected-device identity does not match the live physical device")
    if (
        manifest.get("runtime") != runtime_identity["cuda_runtime"]
        or manifest.get("cuda_compiler_path") != runtime_identity["nvcc_path"]
        or manifest.get("cuda_toolkit_root") != runtime_identity["cuda_home"]
    ):
        abort("native ICICLE CUDA runtime/compiler identity does not match current nvcc")
    if manifest.get("cmake_path") != runtime_identity["cmake_path"] or manifest.get("cmake") != runtime_identity["cmake"]:
        abort("native ICICLE CMake identity does not match the current formal build tool")
    if manifest.get("cmake_generator") != "Unix Makefiles" or manifest.get("make_path") != runtime_identity["make_path"] or manifest.get("make") != runtime_identity["make"]:
        abort("native ICICLE CMake generator/build-tool identity does not match the current formal environment")
    if manifest.get("cc_path") != runtime_identity["cc_path"] or manifest.get("cc") != runtime_identity["cc"]:
        abort("native ICICLE C compiler identity does not match the current formal build tool")
    if manifest.get("cxx_path") != runtime_identity["cxx_path"] or manifest.get("cxx") != runtime_identity["cxx"]:
        abort("native ICICLE C++ compiler identity does not match the current formal build tool")
    if manifest.get("nvidia_smi_path") != runtime_identity["nvidia_smi_path"]:
        abort("native ICICLE device-query tool identity does not match the current formal environment")
    if manifest.get("nvidia_smi") != runtime_identity["nvidia_smi"]:
        abort("native ICICLE device-query version does not match the current formal environment")
    return {
        "install_dir": str(directory.resolve()),
        "nvidia_smi": f"index={selected['index']} uuid={selected['uuid']} {device_name}, {driver_version}, {compute_capability}",
        "selected_cuda_device": selected,
        **identity,
    }


def validate_gpu_evidence(path: Path, *, backend: str) -> dict[str, Any]:
    evidence = load_json(path, "GPU/backend evidence")
    return validate_gpu_evidence_document(evidence, backend=backend)


def validate_gpu_evidence_document(evidence: Any, *, backend: str) -> dict[str, Any]:
    """Validate an already-loaded CPU/GPU attestation without weakening it."""

    evidence = require_exact_keys(
        evidence,
        {
            "backend",
            "gpu_backend",
            "backend_loaded",
            "cpu_fallback",
            "device_synchronized",
            "gpu_kernel_invocation_count",
            "provenance",
            "invocation_trace_sha256",
        },
        "GPU/backend evidence",
    )
    if not isinstance(evidence["gpu_kernel_invocation_count"], int) or isinstance(evidence["gpu_kernel_invocation_count"], bool):
        abort("GPU evidence kernel invocation count is not an integer")
    require_digest(evidence["invocation_trace_sha256"], "GPU invocation trace digest")
    if backend == "gpu":
        if evidence.get("backend") != "CUDA":
            abort("GPU evidence does not identify CUDA backend")
        if evidence.get("gpu_backend") != "icicle-cuda":
            abort("GPU evidence does not identify the ICICLE CUDA backend")
        if evidence.get("backend_loaded") is not True:
            abort("GPU evidence does not confirm backend loading")
        if evidence.get("cpu_fallback") is not False:
            abort("GPU evidence indicates CPU fallback or omits the fail-closed flag")
        if evidence.get("device_synchronized") is not True:
            abort("GPU evidence lacks device synchronization")
        count = evidence.get("gpu_kernel_invocation_count")
        if not isinstance(count, int) or count <= 0:
            abort("GPU evidence has no positive GPU kernel invocation count")
        provenance = evidence.get("provenance")
        provenance = require_exact_keys(
            provenance,
            {"frontend_commit", "native_execution_policy", "abi", "device", "architecture", "runtime", "manifest_sha256"},
            "GPU evidence provenance",
        )
        for key in ("frontend_commit", "native_execution_policy", "abi", "device", "architecture", "runtime", "manifest_sha256"):
            value = provenance.get(key)
            if not isinstance(value, str) or not value:
                abort(f"GPU evidence provenance is missing {key}")
        if provenance["native_execution_policy"] != FORMAL_GPU_EXECUTION_POLICY:
            abort("GPU evidence does not identify the fixed CPU-FFT/GPU-MSM execution policy")
        require_digest(provenance["manifest_sha256"], "GPU backend manifest_sha256")
        if provenance["manifest_sha256"] != native_backend_identity()["native_backend_manifest_sha256"]:
            abort("GPU evidence is not bound to the authenticated native CUDA installation")
    elif (
        evidence.get("backend") != "CPU" or evidence.get("gpu_backend") != "none" or evidence.get("backend_loaded") is not True or
        evidence.get("cpu_fallback") is not False or evidence.get("device_synchronized") is not True or evidence["gpu_kernel_invocation_count"] != 0 or
        evidence.get("provenance") is not None
    ):
        abort("CPU run has invalid backend evidence")
    return evidence


def reject_reduced_srs(argv: list[str]) -> None:
    if any("reduced-srs" in arg or "reduced_srs" in arg for arg in argv):
        abort("--reduced-srs is forbidden for formal comparison runs")


def require_positive_timings(timings: Any, label: str = "phase timings") -> dict[str, float]:
    if not isinstance(timings, dict):
        abort(f"{label} must be an object")
    result: dict[str, float] = {}
    for key in ("prepare_compile_setup", "witness_generation", "prove", "verify"):
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            abort(f"{label}.{key} must be a positive finite duration")
        result[key] = float(value)
    return result


def validate_native_proof_metrics(path: Path, *, proof_sha256: str) -> dict[str, Any]:
    metrics = load_json(path, "native proof metrics")
    require_exact_keys(
        metrics,
        {
            "schema_version",
            "operation",
            "included_in",
            "measurement_scope",
            "commitment_time_ms",
            "prove_time_ms",
            "proof_sha256",
            "instance_commitment_count",
            "phase0_advice_commitment_count",
            "affine_normalization_batch_count",
        },
        "native proof metrics",
    )
    if (
        metrics["schema_version"] != 2
        or metrics["operation"] != "native_halo2_initial_instance_and_phase0_advice_commitments"
        or metrics["included_in"] != "native_create_proof"
        or metrics["measurement_scope"] != "pre_challenge_public_instance_and_phase0_advice_kzg_commitments_plus_affine_normalization"
        or metrics["proof_sha256"] != proof_sha256
    ):
        abort("native proof metrics do not match the current proof artifact and timing boundary")
    for name in ("instance_commitment_count", "phase0_advice_commitment_count", "affine_normalization_batch_count"):
        count = metrics[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            abort(f"native proof metrics {name} must be a nonnegative integer diagnostic")
    for name in ("commitment_time_ms", "prove_time_ms"):
        duration = metrics[name]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            abort(f"native proof metrics {name} must be a positive finite measured duration")
    return metrics


def validate_native_verify_metrics(path: Path, *, proof_sha256: str) -> dict[str, Any]:
    metrics = load_json(path, "native verification metrics")
    require_exact_keys(
        metrics,
        {"schema_version", "operation", "included_in", "verify_time_ms", "proof_sha256"},
        "native verification metrics",
    )
    if (
        metrics["schema_version"] != 1
        or metrics["operation"] != "native_halo2_independent_verifier"
        or metrics["included_in"] != "native_verify_proof"
        or metrics["proof_sha256"] != proof_sha256
    ):
        abort("native verification metrics do not match the current proof artifact and timing boundary")
    duration = metrics["verify_time_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        abort("native verification duration_ms must be a positive finite measured duration")
    return metrics


def require_circuit_ref(value: Any, label: str = "circuit_ref") -> dict[str, str]:
    value = require_exact_keys(
        value,
        {
            "relation_id",
            "compiled_circuit_id",
            "compiled_circuit_digest",
            "settings_id",
            "settings_digest",
            "vk_id",
            "vk_digest",
            "model_graph_id",
            "model_graph_digest",
            "ezkl_security_profile_id",
        },
        label,
    )
    required = ("relation_id", "compiled_circuit_id", "settings_id", "vk_id")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        abort(f"{label} is missing a non-empty identity field")
    if value["relation_id"] != CANONICAL_RELATION:
        abort(f"{label}.relation_id is not R_GAT-ezkl")
    if any(any(marker in value[key].lower() for marker in ("unbound", "pending", "placeholder", "dummy", "fake")) for key in required[1:]):
        abort(f"{label} contains a placeholder identity")
    for key in ("compiled_circuit_digest", "settings_digest", "vk_digest", "model_graph_digest"):
        require_digest(value[key], f"{label}.{key}")
    if not isinstance(value["model_graph_id"], str) or not value["model_graph_id"]:
        abort(f"{label}.model_graph_id is empty")
    if not isinstance(value["ezkl_security_profile_id"], str) or not value["ezkl_security_profile_id"]:
        abort(f"{label}.ezkl_security_profile_id is empty")
    return {key: value[key] for key in required}


def print_error(exc: Exception) -> int:
    print(f"formal prerequisite failed: {exc}", file=sys.stderr)
    return 2
