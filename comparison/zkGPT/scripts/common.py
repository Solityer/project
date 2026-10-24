"""Shared path, process, resource, and result helpers for the native zkGPT runner."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DATASETS: Tuple[str, ...] = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BACKENDS: Tuple[str, ...] = ("cpu", "gpu")
FORMAL_BACKENDS: Tuple[str, ...] = ("cpu",)

RESULT_SCHEMA = "zkgpt-gat-result-v3"
REPORT_SCHEMA = "zkgpt-gat-independent-verifier-v5"
EXECUTION_SCHEMA = "zkgpt-gat-execution-v3"
RESOURCE_MANIFEST_SCHEMA = "zkgpt-gat-resource-manifest-v1"
SCOPED_RESOURCE_MANIFEST_SCHEMA = "zkgpt-gat-formal-scoped-resource-manifest-v1"
PREFLIGHT_SCHEMA = "zkgpt-gat-formal-preflight-v5"
CAPACITY_PROOF_PATH = "formal_native_masked_lasso_hyrax+native_masked_layered_gkr"
COLLECTED_SCHEMA = "zkgpt-gat-collected-results-v4"
FINAL_SCHEMA = "zkgpt-gat-final-validation-v4"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
BN254_SCALAR_MODULUS = int("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001", 16)
PAPER_TIMING_SEMANTICS = "zkgpt-gat-paper-timing-v5-native-commitment-wire-io-excluded"
FORMAL_SESSION_SCHEMA = "zkgpt-gat-formal-session-v1"

_ACTIVE_OWNED_PROCESSES: Dict[int, subprocess.Popen] = {}
_ACTIVE_PROCESS_LOCK = threading.Lock()
_PROCESS_SIGNAL_HANDLERS_INSTALLED = False

VOLATILE_RUN_FILES = frozenset({
    "result.json", "verifier_context.bin", "public_statement.json",
    "public_logits.txt", "proof.artifact", "verifier_report.json",
    "execution.json", "run_provenance.json", "fixture.json",
    "final_verifier_report.json", "final-verifier.stdout.log", "final-verifier.stderr.log",
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


def canonical_dataset_scope(values: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    if values is None:
        return DATASETS
    if not isinstance(values, (list, tuple)) or not values:
        raise ChainError("requested dataset scope must be a nonempty dataset list")
    if any(type(value) is not str or value not in DATASETS for value in values):
        raise ChainError("requested dataset scope contains an unknown dataset")
    if len(set(values)) != len(values):
        raise ChainError("requested dataset scope contains duplicates")
    selected = set(values)
    return tuple(dataset for dataset in DATASETS if dataset in selected)


def formal_child_environment(extra_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    child_env = os.environ.copy()
    if extra_env:
        child_env.update({str(key): str(value) for key, value in extra_env.items()})
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        if child_env.get(key):
            raise ChainError(f"formal execution refuses loader override environment variable {key}")
        child_env.pop(key, None)
    return child_env


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


def formal_results_directory(root: Path, session_id: str) -> Path:
    if not HEX32_RE.fullmatch(session_id):
        raise ChainError("formal session id must be 32 lowercase hexadecimal characters")
    path = local_path(root, f"results/formal/{session_id}", "formal session result directory")
    ensure_directory(path, "formal session result directory", root)
    return path


def formal_preflight_path(root: Path, session_id: str) -> Path:
    return formal_results_directory(root, session_id) / "formal_preflight.json"


def formal_collection_path(root: Path, session_id: str) -> Path:
    return formal_results_directory(root, session_id) / "collected.json"


def formal_final_path(root: Path, session_id: str) -> Path:
    return formal_results_directory(root, session_id) / "final_validation.json"


def formal_session_path(root: Path, session_id: str) -> Path:
    return formal_results_directory(root, session_id) / "session.json"


def formal_session_status_path(root: Path, session_id: str) -> Path:
    return formal_results_directory(root, session_id) / "session_status.json"


def runtime_source_sha256(root: Path) -> str:
    """Fingerprint formal runtime helpers so session-bound outputs reject stale code."""
    paths = [path for path in (root / "scripts").glob("*.py") if path.is_file()]
    paths.extend(root / name for name in ("build.sh",) if (root / name).is_file())
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda value: value.relative_to(root).as_posix()):
        require_regular(path, "formal runtime source")
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def create_formal_session(root: Path, session_id: Optional[str] = None, dataset_scope: Optional[Sequence[str]] = None) -> Tuple[str, Dict[str, Any]]:
    """Create a fresh run directory and its small progress/status record."""
    session_id = secrets.token_hex(16) if session_id is None else session_id
    if not HEX32_RE.fullmatch(session_id):
        raise ChainError("formal session id must be 32 lowercase hexadecimal characters")
    output_dir = formal_results_directory(root, session_id)
    if any(output_dir.iterdir()):
        raise ChainError(f"formal session directory must be fresh and empty: {output_dir}")
    scope = canonical_dataset_scope(dataset_scope)
    session = {
        "schema": FORMAL_SESSION_SCHEMA,
        "session_id": session_id,
        "datasets": list(DATASETS),
        "dataset_scope": list(scope),
        "started_unix_ns": time.time_ns(),
    }
    path = output_dir / "session.json"
    atomic_write_json(path, session, root)
    atomic_write_json(formal_session_status_path(root, session_id), {
        "schema": FORMAL_SESSION_SCHEMA, "session_id": session_id,
        "status": "RUNNING", "active_stage": "session-created", "completed_stages": [],
        "dataset_scope": list(scope),
        "started_unix_ns": session["started_unix_ns"],
    }, root)
    return session_id, session


@contextmanager
def formal_run_lock(root: Path):
    lock_path = local_path(root, "baseline-local/formal-run.lock", "formal run lock")
    ensure_directory(lock_path.parent, "formal lock directory", root)
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read().strip() or "another official session"
            raise ChainError(f"formal run is already active ({owner}); refusing concurrent execution") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} started_unix_ns={time.time_ns()}\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield stream
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def require_formal_session(root: Path) -> Tuple[str, Dict[str, Any]]:
    session_id = os.environ.get("ZKGPT_FORMAL_SESSION_ID", "")
    if not HEX32_RE.fullmatch(session_id):
        raise ChainError("formal helper requires the session created by the comparison coordinator")
    path = formal_session_path(root, session_id)
    session = load_json(path, "formal session record")
    if not isinstance(session, dict) or session.get("schema") != FORMAL_SESSION_SCHEMA:
        raise ChainError("formal session record is invalid")
    if "dataset_scope" in session:
        try:
            scope = canonical_dataset_scope(session.get("dataset_scope"))
        except ChainError as exc:
            raise ChainError("formal session scope is invalid") from exc
        if session.get("dataset_scope") != list(scope):
            raise ChainError("formal session scope is not canonical")
    else:
        scope = DATASETS
    if session.get("session_id") != session_id or session.get("datasets") != list(DATASETS):
        raise ChainError("formal session record does not identify the canonical dataset universe")
    return session_id, session


def formal_resource_manifest_path(root: Path, session_id: str, session: Mapping[str, Any]) -> Path:
    if (not HEX32_RE.fullmatch(session_id) or session.get("session_id") != session_id or
            session.get("datasets") != list(DATASETS)):
        raise ChainError("resource manifest path requires the canonical immutable formal session")
    if "dataset_scope" in session:
        try:
            scope = canonical_dataset_scope(session.get("dataset_scope"))
        except ChainError as exc:
            raise ChainError("resource manifest path session scope is invalid") from exc
        if session.get("dataset_scope") != list(scope):
            raise ChainError("resource manifest path session scope is not canonical")
    else:
        scope = DATASETS
    if scope == DATASETS:
        return local_path(root, "results/resource_manifest.json", "formal resource manifest")
    return local_path(root, f"results/formal/{session_id}/resource_manifest.json", "scoped formal resource manifest")


def require_formal_resource_manifest(
    root: Path,
    session_id: str,
    session: Mapping[str, Any],
    session_manifest_sha256: str,
) -> Tuple[Mapping[str, Any], Path, str]:
    path = formal_resource_manifest_path(root, session_id, session)
    scope = canonical_dataset_scope(session.get("dataset_scope")) if "dataset_scope" in session else DATASETS
    if scope == DATASETS:
        payload = require_manifest(root, "resource_manifest.json", RESOURCE_MANIFEST_SCHEMA)
    else:
        payload = load_json(path, "scoped formal resource manifest")
        expected_fields = {
            "schema", "status", "baseline", "resource_scope", "setup_mode",
            "static_tables_source", "session_id", "session_manifest_sha256",
            "dataset_scope", "quantization_scale_chain_sha256", "quantization_entries", "datasets",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ChainError("scoped formal resource manifest fields are invalid")
        if (
            payload.get("schema") != SCOPED_RESOURCE_MANIFEST_SCHEMA
            or payload.get("status") != "AUTHENTICATED"
            or payload.get("baseline") != "zkGPT"
            or payload.get("resource_scope") != "baseline-local"
            or payload.get("setup_mode") != "hyrax_native_local"
            or payload.get("static_tables_source") != "authenticated_quantization_scale_chain"
            or payload.get("session_id") != session_id
            or payload.get("session_manifest_sha256") != session_manifest_sha256
            or payload.get("dataset_scope") != list(scope)
        ):
            raise ChainError("scoped formal resource manifest identity differs from its immutable session")
        quant_digest = payload.get("quantization_scale_chain_sha256")
        quant_entries = payload.get("quantization_entries")
        datasets = payload.get("datasets")
        if (not isinstance(quant_digest, str) or not HEX64_RE.fullmatch(quant_digest)
                or not isinstance(quant_entries, dict) or not quant_entries
                or not isinstance(datasets, dict) or set(datasets) != set(scope)
                or any(not isinstance(value, dict) for value in datasets.values())):
            raise ChainError("scoped formal resource manifest selected resources are malformed")
    return payload, path, sha256_file(path)


def validate_scoped_resource_manifest_observation(
    payload: Mapping[str, Any],
    observed_resource_entries: Mapping[str, Mapping[str, Any]],
    observed_quantization_identity: Mapping[str, Any],
) -> None:
    if payload.get("datasets") != {dataset: dict(entry) for dataset, entry in observed_resource_entries.items()}:
        raise ChainError("scoped resource manifest selected entries do not match current authenticated resources")
    expected_quantization = {
        "quantization_scale_chain_sha256": observed_quantization_identity.get("quantization_scale_chain_sha256"),
        "quantization_entries": observed_quantization_identity.get("quantization_entries"),
    }
    if {key: payload.get(key) for key in expected_quantization} != expected_quantization:
        raise ChainError("scoped resource manifest quantization identity is stale")


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


def invalidate_results(root: Path, names: Sequence[str] = ("collected_results.json", "final_validation.json")) -> None:
    for name in names:
        if name not in ("formal_preflight.json", "collected_results.json", "final_validation.json"):
            raise ChainError(f"unsupported aggregate output: {name}")
        path = local_path(root, f"results/{name}", "aggregate output")
        if path.exists():
            require_regular(path, "aggregate output").unlink()


def _stop_child(process: subprocess.Popen, owned_pgid: Optional[int], grace_seconds: float = 8.0) -> None:
    _signal_child_termination(process, owned_pgid)
    deadline = time.monotonic() + grace_seconds
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    if owned_pgid is not None:
        while time.monotonic() < deadline:
            try:
                os.killpg(owned_pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.killpg(owned_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait()


def _signal_child_termination(process: subprocess.Popen, owned_pgid: Optional[int]) -> None:
    try:
        if owned_pgid is not None:
            os.killpg(owned_pgid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def _join_output_pumps(
    threads: Sequence[threading.Thread], process: subprocess.Popen,
    owned_pgid: Optional[int], context: str, drain_seconds: float = 8.0,
) -> None:
    """Drain output after the leader exits, without waiting forever on orphaned pipes.

    A formal child may run for as long as needed; the bounded wait starts only
    after its direct process has exited.  Inherited process groups belong to
    the outer stage, so this helper reports the leak and lets that owner reap
    the whole group rather than killing the parent's group itself.
    """
    deadline = time.monotonic() + drain_seconds
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    alive = [thread for thread in threads if thread.is_alive()]
    if not alive:
        return
    if owned_pgid is not None:
        _stop_child(process, owned_pgid)
        for thread in alive:
            thread.join(2.0)
        alive = [thread for thread in threads if thread.is_alive()]
    if alive:
        raise ChainError(
            f"{context} output pipes remained open after the leader exited; "
            "an owned descendant may still be running"
        )
    raise ChainError(
        f"{context} left descendants holding output pipes after the leader exited; "
        "the owned process group was terminated"
    )


def _process_signal_handler(signum: int, _frame: Any) -> None:
    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled_signal, signal.SIG_IGN)
    raise KeyboardInterrupt(f"formal runner interrupted by signal {signum}")


def install_process_signal_cleanup() -> None:
    global _PROCESS_SIGNAL_HANDLERS_INSTALLED
    if _PROCESS_SIGNAL_HANDLERS_INSTALLED or threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGINT, _process_signal_handler)
    signal.signal(signal.SIGTERM, _process_signal_handler)
    signal.signal(signal.SIGHUP, _process_signal_handler)
    _PROCESS_SIGNAL_HANDLERS_INSTALLED = True


def run_logged(
    command: Sequence[str], root: Path, stdout_path: Path, stderr_path: Path,
    label: str, telemetry: Optional[Dict[str, Any]] = None,
    *, extra_env: Optional[Mapping[str, str]] = None,
    isolate_process_group: bool = False,
    pass_fds: Sequence[int] = (),
) -> int:
    """Stream output and own/reap the command's process group when standalone."""
    install_process_signal_cleanup()
    ensure_directory(stdout_path.parent, "native log directory", root)
    for path in (stdout_path, stderr_path):
        local_path(root, relative_path(root, path, "native log"), "native log")
    child_env = formal_child_environment(extra_env)
    inherited_owned_group = child_env.get("ZKGPT_OWNED_PROCESS_GROUP") == "1"
    owns_group = isolate_process_group or not inherited_owned_group
    if owns_group:
        child_env["ZKGPT_OWNED_PROCESS_GROUP"] = "1"
    with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open("w", encoding="utf-8") as stderr_log:
        process: Optional[subprocess.Popen] = None
        owned_pgid: Optional[int] = None
        failures: List[BaseException] = []
        threads: List[threading.Thread] = []
        try:
            managed_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
            old_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, managed_signals)
            try:
                process = subprocess.Popen(
                    list(command), cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1, env=child_env, start_new_session=owns_group,
                    pass_fds=tuple(pass_fds),
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_signal_mask),
                )
                owned_pgid = process.pid if owns_group else None
                if owned_pgid is not None:
                    with _ACTIVE_PROCESS_LOCK:
                        _ACTIVE_OWNED_PROCESSES[owned_pgid] = process
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_signal_mask)

            def pump(stream: Any, output: Any, terminal: Any) -> None:
                try:
                    for line in iter(stream.readline, ""):
                        output.write(line)
                        output.flush()
                        print(f"[{label}] {line}", end="", file=terminal, flush=True)
                except BaseException as exc:
                    failures.append(exc)
                    _signal_child_termination(process, owned_pgid)
                finally:
                    stream.close()

            assert process is not None
            threads = [threading.Thread(target=pump, args=(process.stdout, stdout_log, sys.stdout), daemon=True), threading.Thread(target=pump, args=(process.stderr, stderr_log, sys.stderr), daemon=True)]
            for thread in threads:
                thread.start()
            if telemetry is not None and hasattr(os, "wait4"):
                _, status, usage = os.wait4(process.pid, 0)
                returncode = os.waitstatus_to_exitcode(status)
                process.returncode = returncode
                telemetry.update({"exit_code": returncode, "peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024), "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime})
            else:
                returncode = process.wait()
                if telemetry is not None:
                    raise ChainError("native per-process resource accounting requires wait4")
        except BaseException:
            if process is not None:
                _stop_child(process, owned_pgid)
            raise
        finally:
            if process is not None:
                started_threads = [thread for thread in threads if thread.ident is not None]
                try:
                    _join_output_pumps(started_threads, process, owned_pgid, label)
                    if owned_pgid is not None:
                        _stop_child(process, owned_pgid, grace_seconds=0.5)
                finally:
                    if owned_pgid is not None:
                        with _ACTIVE_PROCESS_LOCK:
                            _ACTIVE_OWNED_PROCESSES.pop(owned_pgid, None)
        if failures:
            raise ChainError(f"native output logging failed: {failures[0]}")
        return returncode


def run_captured(
    command: Sequence[str], root: Path, timeout_seconds: Optional[float] = None,
    *, extra_env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a small helper command with the same owned-group cleanup as run_logged."""
    install_process_signal_cleanup()
    child_env = formal_child_environment(extra_env)
    inherited_owned_group = child_env.get("ZKGPT_OWNED_PROCESS_GROUP") == "1"
    owns_group = not inherited_owned_group
    if owns_group:
        child_env["ZKGPT_OWNED_PROCESS_GROUP"] = "1"
    process: Optional[subprocess.Popen] = None
    owned_pgid: Optional[int] = None
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    pump_failures: List[BaseException] = []
    threads: List[threading.Thread] = []
    try:
        managed_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        old_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, managed_signals)
        try:
            process = subprocess.Popen(
                list(command), cwd=str(root), text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=child_env, start_new_session=owns_group,
                preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_signal_mask),
            )
            owned_pgid = process.pid if owns_group else None
            if owned_pgid is not None:
                with _ACTIVE_PROCESS_LOCK:
                    _ACTIVE_OWNED_PROCESSES[owned_pgid] = process
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_signal_mask)

        def capture(stream: Any, parts: List[str]) -> None:
            try:
                for line in iter(stream.readline, ""):
                    parts.append(line)
            except BaseException as exc:
                pump_failures.append(exc)
                _signal_child_termination(process, owned_pgid)
            finally:
                stream.close()

        assert process is not None
        threads = [
            threading.Thread(target=capture, args=(process.stdout, stdout_parts), daemon=True),
            threading.Thread(target=capture, args=(process.stderr, stderr_parts), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _stop_child(process, owned_pgid)
            try:
                _join_output_pumps(threads, process, owned_pgid, "captured helper")
            except ChainError:
                raise
            stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)
            raise subprocess.TimeoutExpired(command, timeout_seconds, stdout, stderr) from exc
        _join_output_pumps(threads, process, owned_pgid, "captured helper")
        if owned_pgid is not None:
            _stop_child(process, owned_pgid, grace_seconds=0.5)
        if pump_failures:
            raise ChainError(f"captured helper output collection failed: {pump_failures[0]}")
        return subprocess.CompletedProcess(
            command, process.returncode, "".join(stdout_parts), "".join(stderr_parts)
        )
    except BaseException:
        if process is not None:
            _stop_child(process, owned_pgid)
        raise
    finally:
        if process is not None:
            started_threads = [thread for thread in threads if thread.ident is not None]
            try:
                if started_threads and any(thread.is_alive() for thread in started_threads):
                    _join_output_pumps(started_threads, process, owned_pgid, "captured helper")
            finally:
                if owned_pgid is not None:
                    with _ACTIVE_PROCESS_LOCK:
                        _ACTIVE_OWNED_PROCESSES.pop(owned_pgid, None)


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
    manifest = require_regular(directory / "manifest.json", "checkpoint manifest")
    payload = load_json(manifest, "checkpoint manifest")
    if not isinstance(payload, dict):
        raise ChainError(f"checkpoint manifest is not a JSON object: {manifest}")
    return {"manifest.json": str(manifest)}


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
    for name in required:
        require_regular(directory / name, "canonical cache input")
    return {name: str(directory / name) for name in required}


def resource_manifest_dataset_entry(
    root: Path,
    dataset: str,
    config_path_value: Path,
    cache_entries: Mapping[str, str],
    checkpoint_entries: Mapping[str, str],
) -> Dict[str, Any]:
    cache = cache_dir(root, dataset)
    checkpoint = checkpoint_dir(root, dataset)
    return {
        "config_sha256": sha256_file(config_path_value),
        "cache_sidecar_sha256": sha256_file(cache / "integrity.sha256"),
        "cache_entries": dict(cache_entries),
        "checkpoint_sidecar_sha256": sha256_file(checkpoint / "integrity.sha256"),
        "checkpoint_entries": dict(checkpoint_entries),
        "checkpoint_manifest_sha256": checkpoint_entries["manifest.json"],
    }


def verify_local_resources(
    root: Path,
    datasets: Iterable[str] = DATASETS,
    *,
    resource_entries: Optional[Dict[str, Dict[str, Any]]] = None,
    quantization_identity: Optional[Dict[str, Any]] = None,
) -> List[str]:
    errors: List[str] = []
    quant = local_path(root, "baseline-local/configs/quantization_scale_chain.cfg", "quantization config")
    try:
        require_regular(quant, "quantization config")
        read_key_values(quant)
        if quantization_identity is not None:
            quantization_identity.clear()
    except ChainError as exc:
        errors.append(str(exc))
    for dataset in datasets:
        try:
            _config_path_value, config = config_with_auth(root, dataset)
            if config.get("dataset") != dataset:
                raise ChainError(f"config dataset identity mismatch: {dataset}")
            if config.get("setup_mode") != "hyrax_native_local" or config.get("setup_resource_scope") != "baseline-local":
                raise ChainError(f"{dataset} does not declare baseline-local Hyrax setup")
            if "srs_path" in config or "srs_id" in config or "srs_type" in config:
                raise ChainError(f"{dataset} still declares an external SRS dependency")
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
            if resource_entries is not None:
                resource_entries[dataset] = {}
        except ChainError as exc:
            errors.append(str(exc))
    return errors


def resource_identity(root: Path, dataset: str) -> Dict[str, Any]:
    _config_path_value, config = config_with_auth(root, dataset)
    return {
        "dataset": dataset,
        "config": config,
        "full_dataset_identity": f"canonical:{dataset}",
        "cache_identity": f"baseline-local/data/cache/{dataset}/integrity.sha256",
        "checkpoint_identity": f"baseline-local/checkpoints/{dataset}_gat/integrity.sha256",
    }


def build_binary(root: Path, backend: str) -> Path:
    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    candidate = root / "build" / backend / "gat_adapter" / "zkgpt_gat_prover"
    if candidate.exists():
        require_regular(candidate, f"{backend} prover executable")
        if not os.access(candidate, os.X_OK):
            raise ChainError(f"{backend} prover executable is not executable: {candidate}")
        return candidate
    raise ChainError(f"missing source-matched {backend} prover executable; run ./build.sh --{backend}")


def verifier_binary(root: Path, backend: str) -> Path:
    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    candidate = root / "build" / backend / "gat_adapter" / "zkgpt_gat_verifier"
    if candidate.exists():
        require_regular(candidate, f"{backend} verifier executable")
        if not os.access(candidate, os.X_OK):
            raise ChainError(f"{backend} verifier executable is not executable: {candidate}")
        return candidate
    raise ChainError(f"missing source-matched {backend} independent verifier executable; run ./build.sh --{backend}")


def source_tree_sha256(root: Path) -> str:
    """Hash native build inputs, excluding generated build artifacts."""
    suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".S", ".s", ".asm", ".cmake", ".txt", ".py", ".sh", ".in", ".mak"}
    inputs = [root / "CMakeLists.txt", root / "build.sh"]
    for directory in ("src", "gat_adapter", "3rd"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and not any(part in ("CMakeFiles", "__pycache__", ".git", "build") for part in path.relative_to(root / directory).parts) and (path.suffix in suffixes or path.name in ("Makefile", "CMakeLists.txt")):
                inputs.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(inputs), key=lambda value: value.relative_to(root).as_posix()):
        require_regular(path, "native build source")
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _loaded_mcl_library(root: Path, backend: str, binary: Path, description: str) -> Dict[str, str]:
    ldd = Path("/usr/bin/ldd")
    if not ldd.is_file() or not os.access(ldd, os.X_OK):
        raise ChainError("trusted system ldd is unavailable at /usr/bin/ldd")
    completed = subprocess.run([str(ldd), str(binary)], cwd=str(root), text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=formal_child_environment(), check=False)
    if completed.returncode != 0:
        raise ChainError(f"cannot inspect {description} dynamic loader resolution: {completed.stderr.strip()}")
    matches = [match for line in completed.stdout.splitlines() if
               (match := re.match(r"^libmcl\.so\.3\s+=>\s+(\S+)", line.strip()))]
    if len(matches) != 1 or matches[0].group(1) == "not":
        raise ChainError(f"{description} does not resolve exactly one local libmcl.so.3 dependency")
    loaded_path = Path(matches[0].group(1))
    if not loaded_path.is_absolute():
        raise ChainError(f"{description} MCL loader path is not absolute")
    try:
        resolved_path = loaded_path.resolve(strict=True)
        build_lib_dir = (root / "build" / backend / "lib").resolve(strict=True)
        resolved_path.relative_to(build_lib_dir)
        resolved_path.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ChainError(f"{description} resolves MCL outside its baseline-local build directory") from exc
    require_regular(resolved_path, f"{description} loaded MCL library")
    return {"loader_path": str(loaded_path.relative_to(root)), "resolved_path": str(resolved_path.relative_to(root)),
            "sha256": sha256_file(resolved_path)}


def native_runtime_libraries(root: Path, backend: str) -> Dict[str, Any]:
    if backend not in BACKENDS:
        raise ChainError(f"unknown backend: {backend}")
    prover = _loaded_mcl_library(root, backend, build_binary(root, backend), f"{backend} prover")
    verifier = _loaded_mcl_library(root, backend, verifier_binary(root, backend), f"{backend} verifier")
    if prover["resolved_path"] != verifier["resolved_path"] or prover["sha256"] != verifier["sha256"]:
        raise ChainError(f"{backend} prover and verifier resolve different MCL libraries")
    return {"mcl": {"soname": "libmcl.so.3", "prover": prover, "verifier": verifier}}


def write_build_provenance(root: Path, backend: str) -> None:
    prover = build_binary(root, backend)
    verifier = verifier_binary(root, backend)
    payload = {"schema": "zkgpt-gat-build-v2", "backend": backend, "source_tree_sha256": source_tree_sha256(root), "prover_sha256": sha256_file(prover), "verifier_sha256": sha256_file(verifier), "runtime_libraries": native_runtime_libraries(root, backend)}
    atomic_write_json(root / "build" / backend / "build_provenance.json", payload, root)


def verify_build_provenance(root: Path, backend: str) -> str:
    path = local_path(root, f"build/{backend}/build_provenance.json", "native build provenance")
    payload = load_json(path, "native build provenance")
    source_digest = source_tree_sha256(root)
    expected = {"schema": "zkgpt-gat-build-v2", "backend": backend, "source_tree_sha256": source_digest, "prover_sha256": sha256_file(build_binary(root, backend)), "verifier_sha256": sha256_file(verifier_binary(root, backend)), "runtime_libraries": native_runtime_libraries(root, backend)}
    if payload != expected:
        raise ChainError(f"native build source/binary provenance is stale; run ./build.sh --{backend}")
    return source_digest


def binary_fingerprint(path: Path, description: str) -> Dict[str, Any]:
    return file_fingerprint(path, description)


def run_directory(root: Path, dataset: str, session_id: str, config: Optional[Mapping[str, str]] = None) -> Tuple[str, Path]:
    del config
    if dataset not in DATASETS or not HEX32_RE.fullmatch(session_id):
        raise ChainError("formal run path requires a canonical dataset and session id")
    run_relative = f"baseline-local/runs/formal/{session_id}/{dataset}"
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
    formal_session_id: Optional[str] = None,
    run_directory_relative: Optional[str] = None,
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
    if formal_session_id is not None:
        if not HEX32_RE.fullmatch(formal_session_id):
            raise ChainError("native session label must be a 32-character lowercase hex id")
        args.extend(["--formal-session-id", formal_session_id])
    if run_directory_relative is not None:
        run_path = local_path(root, run_directory_relative, "native run directory")
        if not run_path.is_relative_to(root.absolute()):
            raise ChainError("native run directory escapes the baseline")
        args.extend(["--formal-run-directory", run_directory_relative])
    try:
        return run_captured(args, root, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ChainError(f"native {operation} timed out for {dataset}/{backend}") from exc


def require_result_shape(value: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ChainError(f"result is not a JSON object: {path}")
    required = ("schema", "paper_timing_semantics", "session_id",
                "dataset", "backend", "status", "verification_status",
                "proof_scope", "privacy_policy", "privacy_profile", "output_mode",
                "capability_status", "formal_backend_supported")
    for key in required:
        if key not in value:
            raise ChainError(f"result missing {key}: {path}")
    if value["schema"] != RESULT_SCHEMA:
        raise ChainError(f"unsupported result schema in {path}")
    if value.get("paper_timing_semantics") != PAPER_TIMING_SEMANTICS:
        raise ChainError(f"result uses stale or unsupported paper timing semantics: {path}")
    if not HEX32_RE.fullmatch(str(value.get("session_id", ""))):
        raise ChainError(f"result has invalid formal session identity: {path}")
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


def artifact_fields(root: Path, result: Mapping[str, Any], run_path: Path) -> Path:
    relative = result.get("artifact_path")
    if not isinstance(relative, str) or not relative:
        raise ChainError(f"result has no proof artifact path: {run_path / 'result.json'}")
    artifact = local_path(root, relative, "proof artifact")
    if artifact.parent != run_path:
        raise ChainError(f"proof artifact is outside its result directory: {artifact}")
    return require_regular(artifact, "proof artifact")


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
    if result.get("paper_timing_semantics") != PAPER_TIMING_SEMANTICS:
        raise ChainError(f"{path} uses stale or unsupported paper timing semantics")
    required = (
        "load", "registration", "witness", "preparation", "preparation_net",
        "aux_commitment", "lasso_query_commitment",
        "lasso_family_generator_derive", "lasso_family_commitments",
        "lasso_family_selected_row_mapping",
        "initial_commitment_blinding_prepare", "commitment", "prove_total",
        "self_verify", "prove", "verify", "serialize",
        "artifact_wire_self_check", "artifact_write",
    )
    for key in required:
        normalized[key] = finite_nonnegative(timing.get(key), f"timing_ms.{key}", path)
    for key, value in timing.items():
        if not isinstance(key, str):
            raise ChainError(f"{path} has invalid timing field")
        normalized[key] = finite_nonnegative(value, f"timing_ms.{key}", path)
    def timing_equal(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)

    expected = (normalized["aux_commitment"] + normalized["lasso_query_commitment"] +
                normalized["lasso_family_generator_derive"] +
                normalized["lasso_family_commitments"])
    if not timing_equal(expected, normalized["commitment"]):
        raise ChainError(f"{path} commitment total does not equal its measured native substages")
    if not timing_equal(normalized["preparation"] - normalized["lasso_query_commitment"], normalized["preparation_net"]):
        raise ChainError(f"{path} preparation net does not subtract the overlapping L_qry commitment")
    if normalized["preparation"] + 1e-9 < (normalized["lasso_query_commitment"] + normalized["initial_commitment_blinding_prepare"]):
        raise ChainError(f"{path} preparation total omits initial commitment blinding preparation")
    paper_prove = (normalized["prove_total"] -
                   normalized["initial_commitment_blinding_prepare"] -
                   normalized["commitment"] - normalized["self_verify"] +
                   normalized["serialize"])
    if paper_prove < 0 or not timing_equal(paper_prove, normalized["prove"]):
        raise ChainError(f"{path} Prove Time decomposition is invalid")
    return normalized


def verifier_timing(report: Mapping[str, Any], where: Any) -> Dict[str, float]:
    decode_ms = finite_nonnegative(report.get("decode_crypto_ms"), "decode_crypto_ms", where)
    verify_ms = finite_nonnegative(report.get("verify_crypto_ms"), "verify_crypto_ms", where)
    total_ms = finite_nonnegative(report.get("crypto_verify_ms"), "crypto_verify_ms", where)
    if decode_ms + verify_ms != total_ms:
        raise ChainError(f"{where} Verify Time does not equal its measured decode and verification stages")
    return {"decode_crypto_ms": decode_ms, "verify_crypto_ms": verify_ms, "crypto_verify_ms": total_ms}


def paper_metrics(result: Mapping[str, Any], report: Mapping[str, Any], where: Any) -> Dict[str, Any]:
    timing = require_timing(result, where)
    size = result.get("proof_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ChainError(f"{where} has invalid proof_size_bytes")
    return {
        "commitment_time_ms": timing["commitment"],
        "prove_time_ms": timing["prove"],
        "verify_time_ms": verifier_timing(report, where)["crypto_verify_ms"],
        "proof_size_bytes": size,
        "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
        "timing_scope": "Commitment Time includes native cryptographic computations for initial Aux and L_qry bindings and M_f row bindings, plus the per-proof M_f column-generator derivation; sampling the initial hiding randomness is excluded. Prove Time excludes only that initial randomness preparation, Commitment Time, and prover self-verification; it includes native relation evaluation, borrowed Aux witness construction, Lasso table-value/index and lookup-protocol construction, later protocol work, and canonical in-memory wire serialization. Canonical forward generation, explicit dataset/checkpoint file I/O and decode, and one-time model registration are outside this proof timer; artifact wire self-check and artifact file write are recorded separately and excluded.",
    }


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
    if not 1 <= len(fields[7]) <= 64 or any(char not in "0123456789abcdef" for char in fields[7]):
        raise ChainError(f"invalid prover secret blinder encoding: {path}")
    if fields[7] != "0" and fields[7][0] == "0":
        raise ChainError(f"invalid prover secret blinder encoding: {path}")
    if int(fields[7], 16) >= BN254_SCALAR_MODULUS:
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

    del root
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
    if registration["schema"] != "zkgpt-gat-model-registration-v2":
        raise ChainError(f"unsupported model registration schema: {registration_path}")
    if registration.get("plan_recipe") != "canonical_plan_recipe_v3":
        raise ChainError(f"unsupported model registration plan recipe: {registration_path}")
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
    session_id: str,
    session_manifest_sha256: str,
    native_source_sha256: str,
    preflight_path: Path,
    seen_artifact_paths: Optional[set] = None,
    seen_artifact_digests: Optional[set] = None,
) -> Dict[str, Any]:
    """Audit one completed cell and all of its cross-boundary identities."""

    capability = formal_backend_capability(backend)
    if not capability["supported"] or backend not in FORMAL_BACKENDS:
        raise ChainError(capability["reason"])
    entry = preflight.get("backends", {}).get(backend)
    if not isinstance(entry, dict) or entry.get("ready") is not True or entry.get("source_tree_sha256") != verify_build_provenance(root, backend):
        raise ChainError("formal cell source/preflight binding is stale")
    if (preflight.get("session_id") != session_id or
            preflight.get("session_manifest_sha256") != session_manifest_sha256 or
            preflight.get("runtime_source_sha256") != runtime_source_sha256(root) or
            preflight.get("native_source_sha256") != native_source_sha256):
        raise ChainError("formal cell session/preflight binding is stale")
    resource = resource_identity(root, dataset)
    config = resource["config"]
    run_relative = f"baseline-local/runs/formal/{session_id}/{dataset}"
    run_path = local_path(root, run_relative, "formal run directory")
    require_directory(run_path, "formal run directory")
    result_path = run_path / "result.json"
    result = require_result_shape(load_json(result_path, "native result"), result_path)
    if (result.get("dataset") != dataset or result.get("backend") != backend or
            result.get("run_dir") != run_relative):
        raise ChainError(f"native result identity mismatch: {result_path}")
    expected_nonce = derive_run_nonce(session_id, dataset)
    if (result.get("session_id") != session_id or result.get("run_nonce") != expected_nonce or
            result.get("session_manifest_sha256") != session_manifest_sha256):
        raise ChainError(f"native result does not belong to the current formal session: {result_path}")
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
        raise ChainError(f"independent verifier report is not schema-v4 VERIFY_OK: {report_path}")
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
    if (execution.get("session_id") != session_id or execution.get("run_nonce") != expected_nonce or
            execution.get("session_manifest_sha256") != session_manifest_sha256):
        raise ChainError(f"execution record does not belong to the current formal session: {execution_path}")
    started_ns = execution.get("run_started_unix_ns")
    if isinstance(started_ns, bool) or not isinstance(started_ns, int) or started_ns <= 0:
        raise ChainError(f"execution run start is invalid: {execution_path}")
    expected_execution = {
        "session_id": session_id, "session_manifest_sha256": session_manifest_sha256,
        "preflight_sha256": sha256_file(preflight_path),
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
    if (execution.get("paper_timing_semantics") != PAPER_TIMING_SEMANTICS or
            result.get("paper_timing_semantics") != PAPER_TIMING_SEMANTICS):
        raise ChainError("execution/result use stale paper timing semantics")
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
    metrics = paper_metrics(result, report, report_path)
    if execution.get("paper_metrics") != metrics or execution.get("preflight_sha256") != sha256_file(preflight_path):
        raise ChainError("execution metrics/preflight binding is stale")
    stages = execution.get("process_telemetry")
    if not isinstance(stages, dict) or set(stages) != {"prover", "verifier"}:
        raise ChainError("execution native process telemetry is missing")
    for stage in stages.values():
        if not isinstance(stage, dict) or stage.get("exit_code") != 0 or isinstance(stage.get("peak_rss_bytes"), bool) or not isinstance(stage.get("peak_rss_bytes"), int) or stage["peak_rss_bytes"] <= 0:
            raise ChainError("native process did not exit successfully or has invalid RSS")
        for key in ("cpu_user_seconds", "cpu_system_seconds"):
            finite_nonnegative(stage.get(key), key, execution_path)
    return {
        "dataset": dataset, "backend": backend, "status": "VERIFY_OK", "proof_generated": True, "independent_verifier": True,
        "paper_metrics": metrics, "process_telemetry": stages,
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
        "session_id": session_id, "session_manifest_sha256": session_manifest_sha256,
        "run_nonce": execution["run_nonce"], "resource_manifest_sha256": resource_manifest_digest,
        "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
    }
