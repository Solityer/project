"""Shared, dependency-free helpers for the zkGPT formal execution chain.

The C++ adapter remains the only producer of a proof and the only verifier
implementation.  These helpers only resolve baseline-local paths, authenticate
resource sidecars, launch the native executable, and audit result metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASETS: Tuple[str, ...] = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS: Tuple[str, ...] = ("cpu", "gpu")
RESULT_SCHEMA = "zkgpt-gat-result-v1"


class ChainError(RuntimeError):
    """A fail-closed formal-chain error."""


def baseline_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reject_path_alias(path: Path, description: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise ChainError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
        raise ChainError(f"{description} is not a regular file/directory: {path}")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise ChainError(f"{description} must not be a hardlink alias: {path}")


def local_path(root: Path, relative: str, description: str) -> Path:
    """Resolve a baseline-local lexical path without permitting escapes."""

    if not relative or os.path.isabs(relative):
        raise ChainError(f"{description} must be a relative path")
    candidate = Path(relative)
    if any(part in ("", ".", "..") for part in candidate.parts if part != "."):
        if any(part == ".." for part in candidate.parts):
            raise ChainError(f"{description} escapes the baseline root: {relative}")
    root = root.absolute()
    result = root.joinpath(candidate)
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ChainError(f"{description} escapes the baseline root: {relative}") from exc
    # Check every existing component lexically before any caller publishes or
    # opens it.  A missing final component is reported by the caller.
    current = root
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


def ensure_directory(path: Path, description: str) -> Path:
    """Create only baseline-local output directories with safe components."""

    root = baseline_root()
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ChainError(f"{description} is outside baseline root: {path}") from exc
    current = root
    relative = path.absolute().relative_to(root.absolute())
    for part in relative.parts:
        current = current / part
        if current.exists():
            _reject_path_alias(current, description)
            if not current.is_dir():
                raise ChainError(f"{description} is not a directory: {current}")
        else:
            current.mkdir()
    return path


def sha256_file(path: Path) -> str:
    require_regular(path, "authenticated file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Verify a one-directory SHA-256 manifest and reject unknown files."""

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


def verify_resource_bundle(root: Path, relative_dir: str, required: Sequence[str]) -> Dict[str, str]:
    directory = local_path(root, relative_dir, "resource directory")
    require_directory(directory, "resource directory")
    return verify_sidecar(directory, required[0], directory / "integrity.sha256")


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
    if (directory / "features.txt").is_file():
        required.append("features.txt")
    else:
        required.append("features.npy")
    if dataset == "ppi":
        required.extend(["labels_multi.txt", "labels_multi.npy", "node_ptr.txt",
                         "train_mask.npy", "val_mask.npy", "test_mask.npy"])
    elif dataset == "ogbn_arxiv":
        # The canonical OGBN cache carries labels for every node and does not
        # define a separate test.index file.
        required.append("labels.txt")
    else:
        required.extend(["labels.txt", "test.index"])
    expected = verify_sidecar(directory, required[0], directory / "integrity.sha256")
    for name in required:
        if name not in expected:
            raise ChainError(f"cache sidecar omits required file {directory / name}")
    return expected


def verify_local_resources(root: Path, datasets: Iterable[str] = DATASETS) -> List[str]:
    """Check every staged canonical input without loading a full dataset."""

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
            if config["privacy_policy"] != "model_and_trace_hidden" or config["input_privacy"] != "public_graph_public_features" or config["output_mode"] != "public_logits" or config["proof_task"] != "node_classification_full_gat":
                raise ChainError(f"config does not declare strict full-GAT boundary: {dataset}")
            verify_cache(root, dataset)
            verify_checkpoint(root, dataset)
        except ChainError as exc:
            errors.append(str(exc))
    return errors


def build_binary(root: Path, backend: str) -> Path:
    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    candidates = (
        root / "build" / backend / "gat_adapter" / "zkgpt_gat_prover",
        root / f"cmake-build-{backend}" / "gat_adapter" / "zkgpt_gat_prover",
    )
    for candidate in candidates:
        if candidate.exists():
            require_regular(candidate, f"{backend} prover executable")
            if not os.access(candidate, os.X_OK):
                raise ChainError(f"{backend} prover executable is not executable: {candidate}")
            return candidate
    raise ChainError(f"missing {backend} prover executable; run ./build.sh --{backend}")


def verifier_binary(root: Path, backend: str) -> Path:
    """Locate the explicitly built independent verifier for one backend."""

    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    candidates = (
        root / "build" / backend / "gat_adapter" / "zkgpt_gat_verifier",
        root / f"cmake-build-{backend}" / "gat_adapter" / "zkgpt_gat_verifier",
    )
    for candidate in candidates:
        if candidate.exists():
            require_regular(candidate, f"{backend} verifier executable")
            if not os.access(candidate, os.X_OK):
                raise ChainError(f"{backend} verifier executable is not executable: {candidate}")
            return candidate
    raise ChainError(f"missing {backend} independent verifier executable; run ./build.sh --{backend}")


def run_directory(root: Path, dataset: str, backend: str, config: Mapping[str, str]) -> Tuple[str, Path]:
    export = config.get("export_dir")
    if not export:
        export = f"baseline-local/runs/{dataset}_secure"
    run_relative = f"{export}/{backend}"
    run_path = local_path(root, run_relative, "formal run directory")
    ensure_directory(run_path, "formal run directory")
    return run_relative, run_path


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent, "JSON output directory")
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


def atomic_write_text(path: Path, payload: str) -> None:
    ensure_directory(path.parent, "text output directory")
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
) -> subprocess.CompletedProcess[str]:
    """Launch one explicit native backend; never retry with another backend."""

    if operation not in ("prove", "register"):
        raise ChainError(f"unsupported native operation: {operation}")
    executable = build_binary(root, backend)
    args = [str(executable), "--dataset", dataset, "--backend", backend,
            "--baseline-root", str(root)]
    if config_path_value:
        args.extend(["--config", config_path_value])
    if operation == "register":
        args.append("--register-only")
    environment = os.environ.copy()
    # The native adapter decides whether CUDA is available.  No environment
    # selected backend or CPU fallback is injected by the launcher.
    return subprocess.run(args, cwd=str(root), text=True, capture_output=True,
                          check=False, env=environment)


def require_result_shape(value: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ChainError(f"result is not a JSON object: {path}")
    for key in ("schema", "dataset", "backend", "status", "verification_status",
                "proof_scope", "privacy_policy", "privacy_profile", "output_mode"):
        if key not in value:
            raise ChainError(f"result missing {key}: {path}")
    if value["schema"] != RESULT_SCHEMA:
        raise ChainError(f"unsupported result schema in {path}")
    if value["proof_scope"] != "full_gat" or value["privacy_policy"] != "model_and_trace_hidden" or value["privacy_profile"] != "masked_full" or value["output_mode"] != "public_logits":
        raise ChainError(f"result does not declare strict proof/privacy scope: {path}")
    return value


def artifact_fields(root: Path, result: Mapping[str, Any], run_path: Path) -> Tuple[Path, str]:
    relative = result.get("artifact_path")
    if not isinstance(relative, str) or not relative:
        raise ChainError(f"result has no proof artifact path: {run_path / 'result.json'}")
    artifact = local_path(root, relative, "proof artifact")
    # Result paths are baseline-relative; do not let a caller point at a
    # second root through a relative path alias.
    if artifact.parent != run_path:
        raise ChainError(f"proof artifact is outside its result directory: {artifact}")
    digest = result.get("artifact_sha256", "")
    if digest and (not isinstance(digest, str) or len(digest) != 64 or
                   any(char not in "0123456789abcdefABCDEF" for char in digest)):
        raise ChainError(f"result has malformed artifact_sha256: {run_path / 'result.json'}")
    return require_regular(artifact, "proof artifact"), str(digest).lower()
