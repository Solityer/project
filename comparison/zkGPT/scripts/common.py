"""Fail-closed helpers shared by the native zkGPT formal runner.

The C++ adapter is the only proof producer and verifier.  This module only
authenticates baseline-local inputs, resolves source-matched binaries,
records freshness/fingerprints, and cross-checks public result metadata.  It
never infers a proof from a digest or manufactures a verifier result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASETS: Tuple[str, ...] = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS: Tuple[str, ...] = ("cpu", "gpu")
FORMAL_BACKENDS: Tuple[str, ...] = ("cpu",)

RESULT_SCHEMA = "zkgpt-gat-result-v1"
REPORT_SCHEMA = "zkgpt-gat-independent-verifier-v3"
EXECUTION_SCHEMA = "zkgpt-gat-execution-v2"
RESOURCE_MANIFEST_SCHEMA = "zkgpt-gat-resource-manifest-v1"
PREFLIGHT_SCHEMA = "zkgpt-gat-formal-preflight-v2"
COLLECTED_SCHEMA = "zkgpt-gat-collected-results-v2"
FINAL_SCHEMA = "zkgpt-gat-final-validation-v2"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Only these files are volatile formal-cell outputs.  Registration and its
# prover secret are intentionally absent and therefore never removed.
VOLATILE_RUN_FILES = frozenset({
    "result.json", "verifier_context.bin", "public_statement.json",
    "public_logits.txt", "proof.artifact", "verifier_report.json",
    "execution.json", "run_provenance.json", "fixture.json",
    "capacity.json", "prover.stdout.log", "prover.stderr.log",
    "verifier.stdout.log", "verifier.stderr.log",
    "registration.stdout.log", "registration.stderr.log",
    "runner-registration.stdout.log", "runner-registration.stderr.log",
    "registration_audit.json", "verifier_context.bin.sha256",
    "public_statement.json.sha256", "public_logits.txt.sha256",
    "proof.artifact.sha256", "verifier_report.json.sha256",
})
PRESERVED_REGISTRATION_FILES = frozenset({
    "model_registration.json", "com_W.prover.secret",
})


class ChainError(RuntimeError):
    """A formal-chain error; callers must fail closed."""


def formal_backend_capability(backend: str) -> Dict[str, Any]:
    if backend == "cpu":
        return {"supported": True, "status": "FORMAL_CPU_MASKED_FULL", "reason": ""}
    if backend == "gpu":
        return {
            "supported": False,
            "status": "UNSUPPORTED_FORMAL_MASKED_FULL_GPU",
            "reason": (
                "formal masked_full GPU path is unsupported; no CUDA evidence "
                "or CPU proof fallback is permitted"
            ),
        }
    raise ChainError(f"unknown backend: {backend}")


def baseline_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reject_path_alias(path: Path, description: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise ChainError(f"{description} must not be a symlink: {path}")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise ChainError(f"{description} must not be a hardlink alias: {path}")
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
        raise ChainError(f"{description} is not a regular file/directory: {path}")


def _root_relative(root: Path, path: Path, description: str) -> str:
    try:
        return str(path.absolute().relative_to(root.absolute()))
    except ValueError as exc:
        raise ChainError(f"{description} is outside baseline root: {path}") from exc


def local_path(root: Path, relative: str, description: str) -> Path:
    """Resolve a baseline-relative path without lexical or alias escapes."""

    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise ChainError(f"{description} must be a relative path")
    candidate = Path(relative)
    if any(part == ".." for part in candidate.parts):
        raise ChainError(f"{description} escapes the baseline root: {relative}")
    root = root.absolute()
    result = root.joinpath(candidate)
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ChainError(f"{description} escapes the baseline root: {relative}") from exc
    current = root
    _reject_path_alias(current, "baseline root")
    for part in candidate.parts:
        current = current / part
        _reject_path_alias(current, description)
    return result


def require_regular(path: Path, description: str) -> Path:
    _reject_path_alias(path, description)
    if not path.is_file():
        raise ChainError(f"missing {description}: {path}")
    return path


def require_directory(path: Path, description: str) -> Path:
    _reject_path_alias(path, description)
    if not path.is_dir():
        raise ChainError(f"missing {description}: {path}")
    return path


def require_prover_secret(path: Path) -> Path:
    require_regular(path, "prover commitment secret")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ChainError(f"prover commitment secret must have mode 0600: {path}")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise ChainError(f"prover commitment secret is not owned by the current user: {path}")
    return path


def ensure_directory(path: Path, description: str, root: Optional[Path] = None) -> Path:
    root = (root or baseline_root()).absolute()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise ChainError(f"{description} is outside baseline root: {path}") from exc
    current = root
    _reject_path_alias(current, "baseline root")
    for part in relative.parts:
        current = current / part
        if current.exists():
            _reject_path_alias(current, description)
            if not current.is_dir():
                raise ChainError(f"{description} is not a directory: {current}")
        else:
            current.mkdir()
    return path


def cleanup_run_outputs(root: Path, run_path: Path) -> Dict[str, Any]:
    """Remove only allowlisted volatile regular files from one run directory.

    Existing registration and secret files are immutable run identity.  Any
    alias, non-regular volatile entry, or path that is not a direct child is a
    hard failure; no recursive deletion is attempted.
    """

    root = root.absolute()
    run_path = run_path.absolute()
    _root_relative(root, run_path, "formal run directory")
    ensure_directory(run_path, "formal run directory", root)
    removed: List[str] = []
    for entry in sorted(run_path.iterdir(), key=lambda item: item.name):
        _reject_path_alias(entry, "formal run entry")
        if entry.name not in VOLATILE_RUN_FILES and not entry.name.endswith(".partial"):
            continue
        if not stat.S_ISREG(entry.lstat().st_mode):
            raise ChainError(f"volatile run output is not a regular file: {entry}")
        entry.unlink()
        removed.append(entry.name)
    for name in PRESERVED_REGISTRATION_FILES:
        preserved = run_path / name
        if preserved.exists():
            _reject_path_alias(preserved, "preserved registration artifact")
    return {"removed": removed, "preserved": sorted(PRESERVED_REGISTRATION_FILES)}


def new_run_nonce() -> str:
    return secrets.token_hex(16)


def assert_fresh_files(paths: Iterable[Path], started_ns: int, description: str = "formal output") -> None:
    for path in paths:
        require_regular(path, description)
        if path.stat().st_mtime_ns < started_ns:
            raise ChainError(f"stale {description} predates this run: {path}")


def sha256_file(path: Path) -> str:
    require_regular(path, "authenticated file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, description: str) -> Dict[str, Any]:
    require_regular(path, description)
    metadata = path.stat()
    return {"sha256": sha256_file(path), "size_bytes": int(metadata.st_size), "mtime_ns": int(metadata.st_mtime_ns)}


def require_fixed_digest(value: Any, key: str, where: Any) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise ChainError(f"{where} requires lowercase fixed-64 {key}")
    return value


def require_optional_fixed_digest(value: Mapping[str, Any], key: str, where: Any) -> Optional[str]:
    if key not in value or value[key] in (None, ""):
        return None
    return require_fixed_digest(value[key], key, where)


def require_circuit_id(value: Any, where: Any) -> str:
    if not isinstance(value, str) or "/" not in value:
        raise ChainError(f"{where} requires a canonical circuit_id suffix")
    suffix = value.rsplit("/", 1)[1]
    require_fixed_digest(suffix, "circuit_id suffix", where)
    return value


def read_key_values(path: Path) -> Dict[str, str]:
    require_regular(path, "configuration")
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ChainError(f"malformed configuration line {path}:{number}")
            key, value = (item.strip() for item in line.split("=", 1))
            if not key or key in values:
                raise ChainError(f"duplicate/empty configuration key {path}:{number}")
            values[key] = value
    return values


def config_path(root: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ChainError(f"unknown dataset: {dataset}")
    return local_path(root, f"baseline-local/configs/{dataset}_secure.cfg", "dataset config")


def config_with_auth(root: Path, dataset: str) -> Tuple[Path, Dict[str, str]]:
    path = config_path(root, dataset)
    verify_sidecar(path.parent, path.name, path.with_name(path.name + ".sha256"))
    return path, read_key_values(path)


def verify_sidecar(directory: Path, required_name: str, sidecar: Path) -> Dict[str, str]:
    """Verify an authenticated one-directory SHA-256 manifest."""

    require_regular(sidecar, "SHA-256 sidecar")
    expected: Dict[str, str] = {}
    with sidecar.open("r", encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ChainError(f"malformed SHA-256 sidecar {sidecar}:{number}")
            digest, name = fields
            if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
                raise ChainError(f"invalid digest in sidecar {sidecar}:{number}")
            if not name or Path(name).name != name or name in expected:
                raise ChainError(f"invalid/duplicate name in sidecar {sidecar}:{number}")
            expected[name] = digest.lower()
    if required_name not in expected:
        raise ChainError(f"sidecar {sidecar} omits required file {required_name}")
    for name, digest in expected.items():
        actual = sha256_file(directory / name)
        if actual != digest:
            raise ChainError(f"SHA-256 mismatch for {directory / name}")
    return expected


def checkpoint_dir(root: Path, dataset: str) -> Path:
    return local_path(root, f"baseline-local/checkpoints/{dataset}_gat", "checkpoint bundle")


def cache_dir(root: Path, dataset: str) -> Path:
    return local_path(root, f"baseline-local/data/cache/{dataset}", "canonical cache")


def verify_checkpoint(root: Path, dataset: str) -> Dict[str, str]:
    directory = checkpoint_dir(root, dataset)
    return verify_sidecar(directory, "manifest.json", directory / "integrity.sha256")


def verify_cache(root: Path, dataset: str) -> Dict[str, str]:
    directory = cache_dir(root, dataset)
    required = ["meta.cfg", "edges.txt"]
    features_txt = directory / "features.txt"
    if features_txt.exists():
        _reject_path_alias(features_txt, "canonical feature cache")
        required.append("features.txt")
    else:
        required.append("features.npy")
    if dataset == "ppi":
        required.extend(["labels_multi.txt", "labels_multi.npy", "node_ptr.txt",
                         "train_mask.npy", "val_mask.npy", "test_mask.npy"])
    elif dataset == "ogbn_arxiv":
        required.append("labels.txt")
    else:
        required.extend(["labels.txt", "test.index"])
    expected = verify_sidecar(directory, required[0], directory / "integrity.sha256")
    for name in required:
        if name not in expected:
            raise ChainError(f"cache sidecar omits required file {directory / name}")
    return expected


def verify_local_resources(root: Path, datasets: Iterable[str] = DATASETS) -> List[str]:
    errors: List[str] = []
    quant = local_path(root, "baseline-local/configs/quantization_scale_chain.cfg", "quantization config")
    try:
        verify_sidecar(quant.parent, quant.name, quant.with_name(quant.name + ".sha256"))
        read_key_values(quant)
    except ChainError as exc:
        errors.append(str(exc))
    for dataset in datasets:
        try:
            _, config = config_with_auth(root, dataset)
            if config.get("dataset") != dataset:
                raise ChainError(f"config dataset identity mismatch: {dataset}")
            for key in ("privacy_policy", "input_privacy", "output_mode", "proof_task"):
                if not config.get(key):
                    raise ChainError(f"config missing {key}: {dataset}")
            if (config["privacy_policy"] != "model_and_trace_hidden" or
                    config["input_privacy"] != "public_graph_public_features" or
                    config["output_mode"] != "public_logits" or
                    config["proof_task"] != "node_classification_full_gat"):
                raise ChainError(f"config does not declare strict full-GAT boundary: {dataset}")
            verify_cache(root, dataset)
            verify_checkpoint(root, dataset)
        except ChainError as exc:
            errors.append(str(exc))
    return errors


def resource_identity(root: Path, dataset: str) -> Dict[str, Any]:
    config_path_value, config = config_with_auth(root, dataset)
    quant = local_path(root, "baseline-local/configs/quantization_scale_chain.cfg", "quantization config")
    verify_sidecar(quant.parent, quant.name, quant.with_name(quant.name + ".sha256"))
    cache = cache_dir(root, dataset)
    checkpoint = checkpoint_dir(root, dataset)
    verify_cache(root, dataset)
    checkpoint_entries = verify_checkpoint(root, dataset)
    return {
        "dataset": dataset,
        "config": config,
        "config_sha256": sha256_file(config_path_value),
        "quantization_sha256": sha256_file(quant),
        "cache_manifest_sha256": sha256_file(cache / "integrity.sha256"),
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "integrity.sha256"),
        "checkpoint_manifest_entry_sha256": checkpoint_entries["manifest.json"],
        "full_dataset_identity": f"canonical:{dataset}",
        "cache_identity": f"baseline-local/data/cache/{dataset}/integrity.sha256",
        "checkpoint_identity": f"baseline-local/checkpoints/{dataset}_gat/integrity.sha256",
    }


def build_binary(root: Path, backend: str) -> Path:
    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    candidates = (
        root / "build" / backend / "gat_adapter" / "zkgpt_gat_prover",
        root / "build" / "gat_adapter" / "zkgpt_gat_prover",
        root / "cmake-build-release" / "gat_adapter" / "zkgpt_gat_prover",
        root / "cmake-build-debug" / "gat_adapter" / "zkgpt_gat_prover",
        root / "cmake-build-relwithdebinfo" / "gat_adapter" / "zkgpt_gat_prover",
        root / "cmake-build-minsizerel" / "gat_adapter" / "zkgpt_gat_prover",
    )
    for candidate in candidates:
        if candidate.exists():
            require_regular(candidate, f"{backend} prover executable")
            if not os.access(candidate, os.X_OK):
                raise ChainError(f"{backend} prover executable is not executable: {candidate}")
            return candidate
    raise ChainError(f"missing {backend} prover executable; run ./build.sh --{backend}")


def verifier_binary(root: Path, backend: str) -> Path:
    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    candidates = (
        root / "build" / backend / "gat_adapter" / "zkgpt_gat_verifier",
        root / "build" / "gat_adapter" / "zkgpt_gat_verifier",
        root / "cmake-build-release" / "gat_adapter" / "zkgpt_gat_verifier",
        root / "cmake-build-debug" / "gat_adapter" / "zkgpt_gat_verifier",
        root / "cmake-build-relwithdebinfo" / "gat_adapter" / "zkgpt_gat_verifier",
        root / "cmake-build-minsizerel" / "gat_adapter" / "zkgpt_gat_verifier",
    )
    for candidate in candidates:
        if candidate.exists():
            require_regular(candidate, f"{backend} verifier executable")
            if not os.access(candidate, os.X_OK):
                raise ChainError(f"{backend} verifier executable is not executable: {candidate}")
            return candidate
    raise ChainError(f"missing {backend} independent verifier executable; run ./build.sh --{backend}")


def smoke_binary(root: Path, backend: str = "cpu") -> Path:
    if backend != "cpu":
        raise ChainError("only the CPU formal smoke is supported")
    candidates = (
        root / "build" / "cpu-tests" / "tests" / "gat_adapter_smoke",
        root / "build" / "cpu-tests" / "gat_adapter_smoke",
        root / "build" / "tests" / "gat_adapter_smoke",
        root / "build" / "gat_adapter_smoke",
        root / "cmake-build-release" / "tests" / "gat_adapter_smoke",
        root / "cmake-build-debug" / "tests" / "gat_adapter_smoke",
    )
    for candidate in candidates:
        if candidate.exists():
            require_regular(candidate, "CPU formal smoke executable")
            if not os.access(candidate, os.X_OK):
                raise ChainError(f"CPU formal smoke executable is not executable: {candidate}")
            return candidate
    raise ChainError("missing CPU formal smoke executable; build the cpu test target")


def binary_fingerprint(path: Path, description: str) -> Dict[str, Any]:
    return file_fingerprint(path, description)


def run_directory(root: Path, dataset: str, backend: str, config: Mapping[str, str]) -> Tuple[str, Path]:
    export = config.get("export_dir") or f"baseline-local/runs/{dataset}_secure"
    run_relative = f"{export}/{backend}"
    run_path = local_path(root, run_relative, "formal run directory")
    ensure_directory(run_path, "formal run directory", root)
    return run_relative, run_path


def atomic_write_json(path: Path, payload: Any, root: Optional[Path] = None) -> None:
    ensure_directory(path.parent, "JSON output directory", root)
    _reject_path_alias(path, "JSON output")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, payload: str, root: Optional[Path] = None) -> None:
    ensure_directory(path.parent, "text output directory", root)
    _reject_path_alias(path, "text output")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_json(path: Path, description: str) -> Any:
    require_regular(path, description)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise ChainError(f"invalid JSON {description}: {path}") from exc


def launch_native(
    root: Path,
    dataset: str,
    backend: str,
    config_path_value: Optional[str] = None,
    operation: str = "prove",
    registration_path_value: Optional[str] = None,
    com_w_secret_path_value: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    if operation not in ("prove", "register", "capacity"):
        raise ChainError(f"unsupported native operation: {operation}")
    executable = build_binary(root, backend)
    args = [str(executable), "--dataset", dataset, "--backend", backend,
            "--baseline-root", str(root)]
    if config_path_value:
        args.extend(["--config", config_path_value])
    if registration_path_value:
        args.extend(["--registration", registration_path_value])
    if com_w_secret_path_value:
        args.extend(["--com-w-secret", com_w_secret_path_value])
    if operation == "register":
        args.append("--register-only")
    elif operation == "capacity":
        args.append("--capacity-only")
    try:
        return subprocess.run(args, cwd=str(root), text=True, capture_output=True,
                              check=False, env=os.environ.copy(), timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ChainError(f"native {operation} timed out for {dataset}/{backend}") from exc


def require_result_shape(value: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ChainError(f"result is not a JSON object: {path}")
    required = ("schema", "dataset", "backend", "status", "verification_status",
                "proof_scope", "privacy_policy", "privacy_profile", "output_mode",
                "capability_status", "formal_backend_supported")
    for key in required:
        if key not in value:
            raise ChainError(f"result missing {key}: {path}")
    if value["schema"] != RESULT_SCHEMA:
        raise ChainError(f"unsupported result schema in {path}")
    if (value["proof_scope"] != "full_gat" or
            value["privacy_policy"] != "model_and_trace_hidden" or
            value["privacy_profile"] != "masked_full" or
            value["output_mode"] != "public_logits"):
        raise ChainError(f"result does not declare strict proof/privacy scope: {path}")
    capability = formal_backend_capability(value["backend"])
    if (value["capability_status"] != capability["status"] or
            value["formal_backend_supported"] is not capability["supported"]):
        raise ChainError(f"result capability does not match backend gate: {path}")
    return value


def artifact_fields(root: Path, result: Mapping[str, Any], run_path: Path) -> Tuple[Path, str]:
    relative = result.get("artifact_path")
    if not isinstance(relative, str) or not relative:
        raise ChainError(f"result has no proof artifact path: {run_path / 'result.json'}")
    artifact = local_path(root, relative, "proof artifact")
    if artifact.parent != run_path:
        raise ChainError(f"proof artifact is outside its result directory: {artifact}")
    digest = result.get("artifact_sha256")
    if digest is not None and digest != "":
        require_fixed_digest(digest, "artifact_sha256", run_path / "result.json")
    return require_regular(artifact, "proof artifact"), str(digest or "")


def require_manifest(root: Path, name: str, expected_schema: str) -> Mapping[str, Any]:
    path = root / "results" / name
    payload = load_json(path, name)
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ChainError(f"unsupported {name} schema: {path}")
    if payload.get("status") not in ("AUTHENTICATED", "READY"):
        raise ChainError(f"{name} is not successful: {path}")
    return payload


def require_bool(value: Mapping[str, Any], key: str, expected: bool, path: Any) -> None:
    if not isinstance(value.get(key), bool) or value[key] is not expected:
        raise ChainError(f"{path} requires {key}={str(expected).lower()}")


def require_string(value: Mapping[str, Any], key: str, path: Any) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ChainError(f"{path} requires non-empty {key}")
    return item


def finite_nonnegative(value: Any, key: str, where: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChainError(f"{where} has invalid {key}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ChainError(f"{where} has invalid {key}")
    return number


def require_timing(result: Mapping[str, Any], path: Any) -> Dict[str, float]:
    timing = result.get("timing_ms")
    if not isinstance(timing, dict):
        raise ChainError(f"{path} is missing timing_ms")
    normalized: Dict[str, float] = {}
    for key in ("load", "registration", "witness", "prove", "verify", "serialize"):
        normalized[key] = finite_nonnegative(timing.get(key), f"timing_ms.{key}", path)
    return normalized


def parse_prover_secret_metadata(path: Path) -> Dict[str, Any]:
    """Validate the public identity prefix without exposing the blinder."""

    require_prover_secret(path)
    try:
        raw = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ChainError(f"cannot read prover secret metadata: {path}") from exc
    fields = raw.split()
    if len(fields) != 8 or fields[0] != "zkgpt-gat-com-w-secret-v1" or fields[1] != "1":
        raise ChainError(f"malformed prover secret metadata: {path}")
    require_fixed_digest(fields[2], "registration_digest", path)
    require_fixed_digest(fields[3], "basis_id", path)
    try:
        logical_size = int(fields[4])
        padded_size = int(fields[5])
    except ValueError as exc:
        raise ChainError(f"malformed prover secret sizes: {path}") from exc
    if logical_size < 0 or padded_size <= 0:
        raise ChainError(f"invalid prover secret sizes: {path}")
    if len(fields[6]) != 64 or any(char not in "0123456789abcdef" for char in fields[6]):
        raise ChainError(f"invalid prover secret commitment point: {path}")
    if len(fields[7]) != 64 or any(char not in "0123456789abcdef" for char in fields[7]):
        raise ChainError(f"invalid prover secret blinder encoding: {path}")
    return {
        "registration_digest": fields[2], "basis_id": fields[3],
        "logical_size": logical_size, "padded_size": padded_size,
        "point_hex": fields[6],
    }


def validate_registration_artifact(
    root: Path,
    dataset: str,
    backend: str,
    config: Mapping[str, str],
    registration_path: Path,
    secret_path: Path,
) -> Dict[str, Any]:
    """Validate a reusable native registration without recomputing W.

    The native prover remains responsible for recomputing model/quantization
    binding.  This Python gate checks every public wire field, canonical four
    segment layout, config identity, and the secret's identity prefix.  Old
    Cora registrations without optional audit flags remain reusable.
    """

    del root  # reserved for future native validation receipt binding
    registration = load_json(registration_path, "model registration")
    if not isinstance(registration, dict):
        raise ChainError(f"model registration is not an object: {registration_path}")
    required = (
        "schema", "dataset", "status", "model_arch_id", "quant_cfg_id",
        "static_tables_id", "field_id", "field_modulus", "manifest_digest",
        "model_digest", "bounds_digest", "quant_digest", "layout_digest",
        "selector", "opening_policy", "binding_digest", "registration_digest",
        "n_raw", "n_in", "input_segments", "commitment_scheme", "basis_id",
        "logical_size", "padded_size", "point_hex",
    )
    for key in required:
        if key not in registration:
            raise ChainError(f"model registration missing {key}: {registration_path}")
    if registration["schema"] != "zkgpt-gat-model-registration-v1":
        raise ChainError(f"unsupported model registration schema: {registration_path}")
    if registration["dataset"] != dataset or registration["status"] != "registered_long_lived_com_W":
        raise ChainError(f"model registration dataset/status mismatch: {registration_path}")
    if registration.get("backend") not in (None, "", backend):
        raise ChainError(f"model registration backend mismatch: {registration_path}")
    if registration["model_arch_id"] != config.get("model_arch_id"):
        raise ChainError(f"model registration architecture mismatch: {registration_path}")
    if registration["quant_cfg_id"] != config.get("quant_cfg_id"):
        raise ChainError(f"model registration quantization identity mismatch: {registration_path}")
    if registration["static_tables_id"] != config.get("static_table_id"):
        raise ChainError(f"model registration static table identity mismatch: {registration_path}")
    if registration["field_id"] != "BN254" or not str(registration["field_modulus"]).isdigit():
        raise ChainError(f"model registration field identity is invalid: {registration_path}")
    for key in ("manifest_digest", "model_digest", "bounds_digest", "quant_digest",
                "layout_digest", "binding_digest", "registration_digest"):
        require_fixed_digest(registration[key], key, registration_path)
    if registration["selector"] != "00:X_pub,01:W,10:Aux,11:L_qry":
        raise ChainError(f"model registration selector is not canonical: {registration_path}")
    if registration["opening_policy"] not in ("same_r_three_openings_v1", "same_r_three_terminal_openings_v1"):
        raise ChainError(f"model registration opening policy is not canonical: {registration_path}")
    try:
        n_raw = int(registration["n_raw"])
        n_in = int(registration["n_in"])
        logical_size = int(registration["logical_size"])
        padded_size = int(registration["padded_size"])
    except (TypeError, ValueError) as exc:
        raise ChainError(f"model registration layout numbers are invalid: {registration_path}") from exc
    if n_raw <= 0 or n_in <= 0 or n_raw > n_in or n_in & (n_in - 1):
        raise ChainError(f"model registration domain is not canonical: {registration_path}")
    if logical_size <= 0 or padded_size != n_in:
        raise ChainError(f"model registration W domain is invalid: {registration_path}")
    segments = registration["input_segments"]
    names = ("X_pub", "W", "Aux", "L_qry")
    if not isinstance(segments, list) or len(segments) != 4:
        raise ChainError(f"model registration must contain four input segments: {registration_path}")
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("name") != names[index]:
            raise ChainError(f"model registration segment {index} is not canonical: {registration_path}")
        try:
            offset = int(segment["offset"])
            logical = int(segment["logical_length"])
            padded = int(segment["padded_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ChainError(f"model registration segment {index} is malformed: {registration_path}") from exc
        if offset != index * n_in or logical < 0 or logical > n_in or padded != n_in:
            raise ChainError(f"model registration segment {index} violates canonical layout: {registration_path}")
    if logical_size != int(segments[1]["logical_length"]):
        raise ChainError(f"model registration W logical size mismatch: {registration_path}")
    if registration["commitment_scheme"] != "hyrax-pedersen-bn254":
        raise ChainError(f"model registration commitment scheme mismatch: {registration_path}")
    require_fixed_digest(registration["basis_id"], "basis_id", registration_path)
    if len(registration["point_hex"]) != 64 or any(char not in "0123456789abcdef" for char in registration["point_hex"]):
        raise ChainError(f"model registration commitment point is not canonical: {registration_path}")
    secret = parse_prover_secret_metadata(secret_path)
    for key in ("registration_digest", "basis_id", "logical_size", "padded_size", "point_hex"):
        if secret[key] != registration[key]:
            raise ChainError(f"prover secret does not match registration {key}: {secret_path}")
    optional_flags = {
        "registration_only": True,
        "forward_executed": False,
        "plan_materialized": False,
        "layout_source": "canonical_metadata",
    }
    for key, expected in optional_flags.items():
        if key in registration and registration[key] != expected:
            raise ChainError(f"model registration audit flag {key} is invalid: {registration_path}")
    return registration


def cross_check_identity(left: Mapping[str, Any], right: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    for key in keys:
        if left.get(key) != right.get(key):
            raise ChainError(f"{where} identity mismatch for {key}")


def relative_path(root: Path, path: Path, description: str) -> str:
    return _root_relative(root, path, description)


def _audit_identity(value: Mapping[str, Any], where: Any) -> Dict[str, Any]:
    identity: Dict[str, Any] = {}
    for key in ("statement_digest", "plan_digest", "proof_digest",
                "registration_digest", "registration_binding_digest",
                "registration_layout_digest", "com_W_point"):
        identity[key] = require_fixed_digest(value.get(key), key, where)
    identity["circuit_id"] = require_circuit_id(value.get("circuit_id"), where)
    for key in ("full_dataset_identity", "checkpoint_identity", "cache_identity"):
        identity[key] = require_string(value, key, where)
    return identity


def validate_formal_cell(
    root: Path,
    dataset: str,
    backend: str,
    preflight: Mapping[str, Any],
    resource_manifest_digest: str,
    seen_artifact_paths: Optional[set] = None,
    seen_artifact_digests: Optional[set] = None,
) -> Dict[str, Any]:
    """Audit one completed cell and all of its cross-boundary identities."""

    capability = formal_backend_capability(backend)
    if not capability["supported"] or backend not in FORMAL_BACKENDS:
        raise ChainError(capability["reason"])
    resource = resource_identity(root, dataset)
    config = resource["config"]
    run_relative = f"{config.get('export_dir') or f'baseline-local/runs/{dataset}_secure'}/{backend}"
    run_path = local_path(root, run_relative, "formal run directory")
    require_directory(run_path, "formal run directory")
    result_path = run_path / "result.json"
    result = require_result_shape(load_json(result_path, "native result"), result_path)
    if (result.get("dataset") != dataset or result.get("backend") != backend or
            result.get("run_dir") != run_relative):
        raise ChainError(f"native result identity mismatch: {result_path}")
    if result.get("status") != "VERIFY_OK" or result.get("verification_status") != "VERIFY_OK":
        raise ChainError(f"native result is not VERIFY_OK: {result_path}")
    for key in ("proof_generated", "verifier_called", "verifier_ok"):
        require_bool(result, key, True, result_path)
    require_bool(result, "verifier_reads_witness", False, result_path)
    require_bool(result, "verifier_reads_checkpoint", False, result_path)
    require_bool(result, "cpu_fallback", False, result_path)
    identity = _audit_identity(result, result_path)
    if result.get("native_backend") != "cpu_native_zkgpt":
        raise ChainError(f"native result reports an unexpected CPU backend: {result_path}")
    for key, expected in (("full_dataset_identity", resource["full_dataset_identity"]),
                          ("checkpoint_identity", resource["checkpoint_identity"]),
                          ("cache_identity", resource["cache_identity"])):
        if result[key] != expected:
            raise ChainError(f"native result resource mismatch for {key}: {result_path}")
    for key in ("config_sha256", "quantization_sha256", "cache_manifest_sha256",
                "checkpoint_manifest_sha256", "checkpoint_manifest_entry_sha256",
                "resource_manifest_sha256"):
        require_fixed_digest(result.get(key), key, result_path)
    if result["config_sha256"] != resource["config_sha256"] or result["quantization_sha256"] != resource["quantization_sha256"] or result["cache_manifest_sha256"] != resource["cache_manifest_sha256"] or result["checkpoint_manifest_sha256"] != resource["checkpoint_manifest_sha256"] or result["checkpoint_manifest_entry_sha256"] != resource["checkpoint_manifest_entry_sha256"] or result["resource_manifest_sha256"] != resource_manifest_digest:
        raise ChainError(f"native result resource digest mismatch: {result_path}")
    timing = require_timing(result, result_path)

    def path_from_result(field: str, description: str, expected_name: str) -> Tuple[str, Path]:
        value = result.get(field)
        if not isinstance(value, str) or not value:
            raise ChainError(f"native result is missing {field}: {result_path}")
        path = local_path(root, value, description)
        if path.parent != run_path:
            raise ChainError(f"{description} escapes its run directory: {path}")
        if path.name != expected_name:
            raise ChainError(f"{description} does not use the canonical filename: {path}")
        require_regular(path, description)
        return value, path

    statement_rel, statement_path = path_from_result("statement_path", "public statement", "public_statement.json")
    context_rel, context_path = path_from_result("verifier_context_path", "verifier context", "verifier_context.bin")
    logits_rel, logits_path = path_from_result("public_logits_path", "public logits", "public_logits.txt")
    artifact_rel, artifact_path = path_from_result("artifact_path", "proof artifact", "proof.artifact")
    registration_rel, registration_path = path_from_result("registration_path", "model registration", "model_registration.json")
    secret_rel, secret_path = path_from_result("prover_secret_path", "prover commitment secret", "com_W.prover.secret")
    require_prover_secret(secret_path)
    registration = validate_registration_artifact(
        root, dataset, backend, config, registration_path, secret_path
    )
    registration_identity = {
        "registration_digest": registration["registration_digest"],
        "registration_binding_digest": registration["binding_digest"],
        "registration_layout_digest": registration["layout_digest"],
        "com_W_point": registration["point_hex"],
    }
    if any(identity[key] != value for key, value in registration_identity.items()):
        raise ChainError(f"native result registration identity mismatch: {result_path}")
    artifact_fp = file_fingerprint(artifact_path, "proof artifact")
    require_fixed_digest(result.get("artifact_sha256"), "artifact_sha256", result_path)
    if result["artifact_sha256"] != artifact_fp["sha256"]:
        raise ChainError(f"native result artifact digest mismatch: {result_path}")
    if result.get("artifact_size_bytes") != artifact_fp["size_bytes"] or result.get("proof_size_bytes") != artifact_fp["size_bytes"]:
        raise ChainError(f"native result artifact size mismatch: {result_path}")
    if seen_artifact_paths is not None and artifact_path in seen_artifact_paths:
        raise ChainError(f"proof artifact path is reused across formal cells: {artifact_path}")
    if seen_artifact_digests is not None and artifact_fp["sha256"] in seen_artifact_digests:
        raise ChainError(f"proof artifact digest is reused across formal cells: {artifact_path}")
    if seen_artifact_paths is not None:
        seen_artifact_paths.add(artifact_path)
    if seen_artifact_digests is not None:
        seen_artifact_digests.add(artifact_fp["sha256"])

    report_path = run_path / "verifier_report.json"
    report = load_json(report_path, "independent verifier report")
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA or report.get("status") != "VERIFY_OK" or report.get("independent") is not True or report.get("masked_full") is not True:
        raise ChainError(f"independent verifier report is not schema-v3 VERIFY_OK: {report_path}")
    if report.get("dataset") != dataset or report.get("backend") != backend:
        raise ChainError(f"independent verifier dataset/backend mismatch: {report_path}")
    require_bool(report, "verifier_reads_witness", False, report_path)
    require_bool(report, "verifier_reads_checkpoint", False, report_path)
    report_identity = _audit_identity(report, report_path)
    if report_identity != identity:
        raise ChainError(f"independent verifier identity tuple mismatch: {report_path}")
    for key, expected in (("full_dataset_identity", resource["full_dataset_identity"]),
                          ("checkpoint_identity", resource["checkpoint_identity"]),
                          ("cache_identity", resource["cache_identity"])):
        if report.get(key) != expected:
            raise ChainError(f"independent verifier resource mismatch for {key}: {report_path}")
    if (report.get("statement_path") != statement_rel or
            report.get("verifier_context_path") != context_rel or
            report.get("public_logits_path") != logits_rel or
            report.get("artifact_path") != artifact_rel):
        raise ChainError(f"independent verifier path tuple mismatch: {report_path}")
    if report.get("artifact_sha256") != artifact_fp["sha256"] or report.get("artifact_size_bytes") != artifact_fp["size_bytes"]:
        raise ChainError(f"independent verifier artifact tuple mismatch: {report_path}")

    execution_path = run_path / "execution.json"
    execution = load_json(execution_path, "execution record")
    if not isinstance(execution, dict) or execution.get("schema") != EXECUTION_SCHEMA or execution.get("status") != "VERIFY_OK":
        raise ChainError(f"execution record is not schema-v2 VERIFY_OK: {execution_path}")
    if execution.get("dataset") != dataset or execution.get("backend") != backend:
        raise ChainError(f"execution dataset/backend mismatch: {execution_path}")
    require_bool(execution, "no_fallback", True, execution_path)
    require_bool(execution, "verifier_reads_witness", False, execution_path)
    require_bool(execution, "verifier_reads_checkpoint", False, execution_path)
    if not isinstance(execution.get("run_nonce"), str) or not re.fullmatch(r"[0-9a-f]{32}", execution["run_nonce"]):
        raise ChainError(f"execution run nonce is not canonical: {execution_path}")
    started_ns = execution.get("run_started_unix_ns")
    if isinstance(started_ns, bool) or not isinstance(started_ns, int) or started_ns <= 0:
        raise ChainError(f"execution run start is invalid: {execution_path}")
    expected_execution = {
        "result_path": str(result_path.relative_to(root)), "verifier_report_path": str(report_path.relative_to(root)),
        "statement_path": statement_rel, "verifier_context_path": context_rel,
        "public_logits_path": logits_rel, "artifact_path": artifact_rel,
        "registration_path": registration_rel, "prover_secret_path": secret_rel,
        "statement_digest": identity["statement_digest"], "plan_digest": identity["plan_digest"],
        "circuit_id": identity["circuit_id"], "proof_digest": identity["proof_digest"],
        "registration_digest": identity["registration_digest"],
        "registration_binding_digest": identity["registration_binding_digest"],
        "registration_layout_digest": identity["registration_layout_digest"],
        "com_W_point": identity["com_W_point"],
        "full_dataset_identity": resource["full_dataset_identity"],
        "checkpoint_identity": resource["checkpoint_identity"], "cache_identity": resource["cache_identity"],
        "artifact_sha256": artifact_fp["sha256"], "artifact_size_bytes": artifact_fp["size_bytes"],
        "artifact_mtime_ns": artifact_fp["mtime_ns"],
        "resource_manifest_sha256": resource_manifest_digest,
        "native_backend": result["native_backend"],
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ChainError(f"execution identity mismatch for {key}: {execution_path}")
    if execution.get("timing_ms") != result.get("timing_ms"):
        raise ChainError(f"execution timing does not match native result: {execution_path}")
    execution_resource_digests = {
        "config_sha256": resource["config_sha256"],
        "quantization_sha256": resource["quantization_sha256"],
        "cache_manifest_sha256": resource["cache_manifest_sha256"],
        "checkpoint_manifest_sha256": resource["checkpoint_manifest_sha256"],
        "checkpoint_manifest_entry_sha256": resource["checkpoint_manifest_entry_sha256"],
        "resource_manifest_sha256": resource_manifest_digest,
    }
    for key, expected in execution_resource_digests.items():
        require_fixed_digest(execution.get(key), key, execution_path)
        if execution[key] != expected:
            raise ChainError(f"execution resource digest mismatch for {key}: {execution_path}")
    prover = build_binary(root, backend)
    verifier = verifier_binary(root, backend)
    for key, executable in (("prover_binary", prover), ("verifier_binary", verifier)):
        fingerprint = execution.get(key)
        if not isinstance(fingerprint, dict) or fingerprint.get("path") != relative_path(root, executable, f"{key} path"):
            raise ChainError(f"execution is missing current {key} fingerprint: {execution_path}")
        current_fp = binary_fingerprint(executable, key)
        if fingerprint != {"path": relative_path(root, executable, f"{key} path"), **current_fp}:
            raise ChainError(f"execution {key} fingerprint is stale: {execution_path}")
    expected_file_fingerprints = {
        "result": file_fingerprint(result_path, "native result"),
        "statement": file_fingerprint(statement_path, "public statement"),
        "verifier_context": file_fingerprint(context_path, "verifier context"),
        "public_logits": file_fingerprint(logits_path, "public logits"),
        "proof_artifact": artifact_fp,
        "verifier_report": file_fingerprint(report_path, "verifier report"),
    }
    if execution.get("file_fingerprints") != expected_file_fingerprints:
        raise ChainError(f"execution file fingerprints do not match current artifacts: {execution_path}")
    assert_fresh_files((result_path, statement_path, context_path, logits_path, artifact_path, report_path, execution_path), started_ns, "formal cell output")
    wall = {key: finite_nonnegative(execution.get(key), key, execution_path) for key in ("prover_wall_ms", "verifier_wall_ms")}
    return {
        "dataset": dataset, "backend": backend,
        "result_path": str(result_path.relative_to(root)), "verifier_report_path": str(report_path.relative_to(root)),
        "statement_path": statement_rel, "verifier_context_path": context_rel, "public_logits_path": logits_rel,
        "registration_path": registration_rel, "prover_secret_path": secret_rel,
        "artifact_path": artifact_rel, "artifact_sha256": artifact_fp["sha256"],
        "artifact_size_bytes": artifact_fp["size_bytes"], "proof_size_bytes": artifact_fp["size_bytes"],
        "proof_digest": identity["proof_digest"], "statement_digest": identity["statement_digest"],
        "circuit_id": identity["circuit_id"], "plan_digest": identity["plan_digest"],
        "registration_digest": identity["registration_digest"],
        "registration_binding_digest": identity["registration_binding_digest"],
        "registration_layout_digest": identity["registration_layout_digest"], "com_W_point": identity["com_W_point"],
        "full_dataset_identity": resource["full_dataset_identity"], "checkpoint_identity": resource["checkpoint_identity"],
        "cache_identity": resource["cache_identity"], "config_sha256": resource["config_sha256"],
        "quantization_sha256": resource["quantization_sha256"], "cache_manifest_sha256": resource["cache_manifest_sha256"],
        "checkpoint_manifest_sha256": resource["checkpoint_manifest_sha256"],
        "timing_ms": require_timing(result, result_path), "wall_ms": wall,
        "native_backend": result["native_backend"], "no_fallback": True, "witness_isolated": True,
        "run_nonce": execution["run_nonce"], "resource_manifest_sha256": resource_manifest_digest,
    }
