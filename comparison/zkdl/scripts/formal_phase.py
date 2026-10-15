#!/usr/bin/env python3
"""Internal phase adapter for the baseline-local zkDL formal runner."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from session_lifecycle import (
    SessionRootError,
    require_bound_session,
    require_session_artifact,
    require_session_root,
)

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
SESSION_RE = re.compile(r"formal_([0-9a-f]{32})\Z")
ACTIVE_PGID: int | None = None
INTERRUPTED: int | None = None
ACTIVE_SESSION_STATE: Path | None = None
ACTIVE_SESSION_ID: str | None = None
ACTIVE_SESSION_STAGE: str | None = None
SIGNAL_GRACE_SECONDS = 2.0


class PhaseError(RuntimeError):
    pass


def _srs_manifest_finalization_capacity(family_count: int, segment_count: int) -> dict[str, int]:
    if sys.implementation.name != "cpython":
        raise PhaseError("zkDL finalization heap bound requires the source-profiled CPython object model")
    if (type(family_count) is not int or family_count < 1 or family_count > (1 << 20) or
            type(segment_count) is not int or segment_count < family_count or segment_count > family_count * 4096):
        raise PhaseError("zkDL SRS manifest family/segment counts are invalid")
    segment_fields = ("segment_index", "logical_offset", "actual_length", "padded_length", "basis_offset", "basis_count")
    # prepare_form_resources.write_srs emits one family chunk and exactly these six integer fields per segment.
    sample = dict.fromkeys(segment_fields, 0)
    pointer_bytes = ctypes.sizeof(ctypes.c_void_p)
    max_u64 = (1 << 64) - 1
    segment_object_bytes = sys.getsizeof(sample) + sum(sys.getsizeof(field) + sys.getsizeof(max_u64) for field in segment_fields) + 2 * pointer_bytes
    # The fixed 512-byte segment and 4096-byte family allowances exceed the indented JSON emitted by write_json.
    raw_json_bytes = 64 * 1024 + family_count * 4096 + segment_count * 512
    # Include each family/chunk's strings and containers, six keys/ints, segment list slots, one hash block and a reserved Python heap.
    python_object_bytes = family_count * 16 * 1024 + segment_count * segment_object_bytes
    hash_buffer_bytes = 1 << 20
    interpreter_reserve_bytes = 256 << 20
    host_peak_bytes = raw_json_bytes + python_object_bytes + hash_buffer_bytes + interpreter_reserve_bytes
    if any(value > (1 << 64) - 1 for value in (segment_object_bytes, raw_json_bytes, python_object_bytes, host_peak_bytes)):
        raise PhaseError("zkDL SRS manifest finalization bound overflows u64")
    return {"family_count": family_count, "segment_count": segment_count,
            "raw_json_bytes_upper_bound": raw_json_bytes,
            "python_object_bytes_upper_bound": python_object_bytes,
            "hash_buffer_bytes": hash_buffer_bytes,
            "interpreter_reserve_bytes": interpreter_reserve_bytes,
            "host_peak_bytes_upper_bound": host_peak_bytes}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _terminate_group(pgid: int, grace_seconds: float = 2.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_handler(signum: int, _frame: object) -> None:
    global INTERRUPTED
    INTERRUPTED = signum
    _mark_active_session_interrupted(signum)
    pgid = ACTIVE_PGID
    if pgid is not None:
        _terminate_group(pgid, SIGNAL_GRACE_SECONDS)
    raise KeyboardInterrupt(f"formal phase interrupted by signal {signum}")


def _mark_active_session_interrupted(signum: int) -> None:
    path = ACTIVE_SESSION_STATE
    session_id = ACTIVE_SESSION_ID
    if path is None or session_id is None:
        return
    try:
        require_session_root(session=path.parent)
        if path.is_symlink() or not path.is_file():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(state, dict) or state.get("schema") != "zkdl_formal_runner_state_v1" or
                state.get("session_id") != session_id or state.get("status") != "RUNNING"):
            return
        state["interrupted_stage"] = state.get("stage", ACTIVE_SESSION_STAGE or "unknown")
        state["status"] = "INTERRUPTED"
        state["stage"] = "interrupted"
        state["failure"] = f"formal phase received termination signal {signum}"
        state["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["updated_at_utc"] = state["finished_at_utc"]
        temporary = path.with_name(f".{path.name}.{os.getpid()}.interrupt.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # The next baseline-local invocation will convert an ownerless RUNNING
        # state to INTERRUPTED; never let signal cleanup mask process teardown.
        return


def _install_parent_death_signal(owner_pid: int, owner_start_ticks: int) -> None:
    """Stop this phase if its local lifecycle owner disappears unexpectedly."""
    if os.getppid() != owner_pid or _proc_start_ticks(owner_pid) != owner_start_ticks:
        raise PhaseError("zkDL formal runner owner changed before parent-death protection was installed")
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux PR_SET_PDEATHSIG: SIGTERM invokes the ordinary local process-group
    # cleanup path below, even if the runner itself was killed with SIGKILL.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise PhaseError(f"cannot install zkDL phase parent-death protection: errno={error}")
    if os.getppid() != owner_pid or _proc_start_ticks(owner_pid) != owner_start_ticks:
        raise PhaseError("zkDL formal runner owner exited while parent-death protection was installed")


def _run(command: list[str], *, log_path: Path | None = None, fresh_log: bool = False,
         env: dict[str, str] | None = None, also_log_paths: tuple[Path, ...] = (), capture_output: bool = False,
         capture_mode: str = "all", pass_fds: tuple[int, ...] = ()) -> str:
    """Run one owned process group; preserve its exit status and drain logs."""
    global ACTIVE_PGID
    identity = require_session_root()
    log_streams = []
    for index, path in enumerate(((log_path,) if log_path is not None else ()) + also_log_paths):
        require_session_artifact(path, ROOT)
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise PhaseError(f"formal stage log directory is missing or unsafe: {path.parent}")
        log_streams.append(path.open("xb" if fresh_log and index == 0 else "ab", buffering=0))
    output: list[str] = []
    output_lock = threading.Lock()
    log_error: list[OSError] = []
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["ZKDL_ALLOW_FULL_DATASET_RUN"] = "1"
    owned_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, owned_signals)
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=0,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        for stream in log_streams:
            stream.close()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise PhaseError(f"cannot start stage {command[0]}: {exc}") from exc
    pgid = process.pid
    ACTIVE_PGID = pgid

    def pump() -> None:
        assert process.stdout is not None
        for raw in iter(process.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            if capture_output and capture_mode == "all":
                with output_lock:
                    output.append(line)
            elif capture_output and capture_mode == "last_line":
                with output_lock:
                    output[:] = [line]
            elif capture_output and capture_mode == "phase_marker":
                try:
                    marker = json.loads(line)
                except json.JSONDecodeError:
                    marker = None
                if ((isinstance(marker, dict) and marker.get("zkdl_phase")) or
                        line.startswith("zkdl-formal-phase=FAILED") or
                        line.startswith("zkdl-formal-phase=INTERRUPTED")):
                    with output_lock:
                        output[:] = [line]
            if log_streams:
                try:
                    for stream in log_streams:
                        stream.write(raw)
                except OSError as exc:
                    if not log_error:
                        log_error.append(exc)
                        _terminate_group(pgid)
            sys.stdout.write(line)
            sys.stdout.flush()

    reader = threading.Thread(target=pump, name="zkdl-formal-log-pump", daemon=True)
    reader.start()
    status: int | None = None
    failed = False
    orphaned_group = False
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        status = process.wait()
        failed = status != 0
        if status == 0 and _group_exists(pgid):
            orphaned_group = True
            failed = True
    except BaseException:
        failed = True
        raise
    finally:
        # Keep the stage PGID owned until all output has drained.  In
        # particular, a failed shell/Python leader may leave a silent child
        # holding the pipe or may have redirected its output elsewhere.
        if failed:
            _terminate_group(pgid)
        reader.join()
        if process.stdout is not None:
            process.stdout.close()
        for stream in log_streams:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        ACTIVE_PGID = None
    with output_lock:
        text = "".join(output)
    require_session_root(identity)
    if log_error:
        raise PhaseError(f"cannot persist complete stage log {log_path}: {log_error[0]}")
    if orphaned_group:
        raise PhaseError(f"stage leader exited successfully but left an owned process group alive: {' '.join(command)}")
    if status != 0:
        detail = text.strip().splitlines()[-1][:1024] if capture_output and text.strip() else ""
        suffix = f"; detail={detail}" if detail else ""
        raise PhaseError(f"stage exited with status {status}: {' '.join(command)}{suffix}")
    return text


def _session_path(raw: str) -> tuple[Path, str]:
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise PhaseError("session directory must be an absolute, non-symlink baseline-local path")
    resolved = path.resolve(strict=False)
    if resolved.parent != ROOT / "runs":
        raise PhaseError("session directory must be directly under zkdl/runs")
    match = SESSION_RE.fullmatch(resolved.name)
    if match is None:
        raise PhaseError("session directory must be named runs/formal_<32 lowercase hex>")
    return resolved, match.group(1)


def _proc_start_ticks(pid: int) -> int:
    """Read field 22 of /proc/<pid>/stat without splitting the parenthesized comm."""
    if pid <= 0:
        raise PhaseError("formal owner lease contains an invalid PID")
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as exc:
        raise PhaseError("formal owner process is not live") from exc
    close = text.rfind(")")
    if close < 0:
        raise PhaseError("formal owner process identity is malformed")
    fields = text[close + 2:].split()
    if len(fields) <= 19:
        raise PhaseError("formal owner process identity is incomplete")
    try:
        return int(fields[19])
    except ValueError as exc:
        raise PhaseError("formal owner process start identity is malformed") from exc


def _proc_parent_pid(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as exc:
        raise PhaseError("formal owner process is not live") from exc
    close = text.rfind(")")
    if close < 0:
        raise PhaseError("formal owner process identity is malformed")
    fields = text[close + 2:].split()
    if not fields:
        raise PhaseError("formal owner process identity is incomplete")
    try:
        return int(fields[1])
    except (IndexError, ValueError) as exc:
        raise PhaseError("formal owner parent identity is malformed") from exc


def _validate_local_owner(session: Path, session_id: str, phase: str, dataset: str | None) -> None:
    """Authorize only a live phase child of the zkDL-local lock-owning runner."""
    global ACTIVE_SESSION_STATE, ACTIVE_SESSION_ID, ACTIVE_SESSION_STAGE
    require_bound_session(session, ROOT)
    token = os.environ.get("ZKDL_FORMAL_OWNER_TOKEN", "")
    lock_fd_text = os.environ.get("ZKDL_FORMAL_LOCK_FD", "")
    if re.fullmatch(r"[0-9a-f]{64}", token) is None or not lock_fd_text.isascii() or not lock_fd_text.isdecimal():
        raise PhaseError("zkDL phases require the live baseline-local formal runner owner lease")
    try:
        lock_fd = int(lock_fd_text)
        if lock_fd < 3:
            raise ValueError
        inherited_lock = os.fstat(lock_fd)
    except (OSError, ValueError) as exc:
        raise PhaseError("zkDL formal runner lock descriptor is missing") from exc

    lock_path = ROOT / "runs" / ".formal.lock"
    try:
        if lock_path.is_symlink():
            raise PhaseError("zkDL formal runner lock path is a symlink")
        expected_lock = lock_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PhaseError("zkDL formal runner lock path is unavailable") from exc
    if not stat.S_ISREG(expected_lock.st_mode) or (inherited_lock.st_dev, inherited_lock.st_ino) != (expected_lock.st_dev, expected_lock.st_ino):
        raise PhaseError("inherited descriptor is not the zkDL-local formal runner lock")

    state_path = session / "state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise PhaseError("zkDL-local formal runner session state is missing")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseError("zkDL-local formal runner session state is unreadable") from exc
    if (not isinstance(state, dict) or state.get("schema") != "zkdl_formal_runner_state_v1" or
            state.get("baseline") != "zkdl" or state.get("session_id") != session_id or state.get("status") != "RUNNING"):
        raise PhaseError("zkDL-local formal runner session identity/status mismatch")

    expected_stage = "prepare" if phase == "prepare" else f"cell:{dataset}" if phase == "cell" else "finalize"
    if state.get("stage") != expected_stage:
        raise PhaseError("zkDL-local formal runner stage does not authorize this phase")
    stored_token_digest = state.get("internal_owner_token_sha256")
    actual_token_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    if not isinstance(stored_token_digest, str) or re.fullmatch(r"[0-9a-f]{64}", stored_token_digest) is None or not hmac.compare_digest(
            stored_token_digest, actual_token_digest):
        raise PhaseError("zkDL-local formal runner owner token does not match the active session")

    owner = state.get("owner")
    lock = state.get("lock")
    if not isinstance(owner, dict) or not isinstance(lock, dict):
        raise PhaseError("zkDL-local formal runner owner or lock lease is missing")
    owner_pid = owner.get("pid")
    owner_boot_id = owner.get("boot_id")
    owner_start_ticks = owner.get("start_ticks")
    if (not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 0 or
            not isinstance(owner_boot_id, str) or re.fullmatch(r"[0-9a-fA-F-]{36}", owner_boot_id) is None or
            not isinstance(owner_start_ticks, str) or re.fullmatch(r"[1-9][0-9]*", owner_start_ticks) is None or
            lock.get("device") != expected_lock.st_dev or lock.get("inode") != expected_lock.st_ino):
        raise PhaseError("zkDL-local formal runner owner or lock lease is malformed")
    try:
        live_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise PhaseError("cannot validate zkDL-local formal runner boot identity") from exc
    if (live_boot_id != owner_boot_id or _proc_start_ticks(owner_pid) != int(owner_start_ticks) or
            _proc_parent_pid(os.getpid()) != owner_pid or os.getppid() != owner_pid):
        raise PhaseError("zkDL phase is not a live child of the recorded baseline-local owner")
    try:
        owner_cmdline = Path(f"/proc/{owner_pid}/cmdline").read_bytes().split(b"\0")
    except OSError as exc:
        raise PhaseError("zkDL-local formal runner command identity is unavailable") from exc
    if not any(b"formal_runner.py" in arg for arg in owner_cmdline):
        raise PhaseError("zkDL-local formal runner command identity mismatch")

    ACTIVE_SESSION_STATE = state_path
    ACTIVE_SESSION_ID = session_id
    ACTIVE_SESSION_STAGE = expected_stage
    _install_parent_death_signal(owner_pid, int(owner_start_ticks))

    # The token only authenticates this adapter entry; never pass it to native descendants.
    os.environ.pop("ZKDL_FORMAL_OWNER_TOKEN", None)
    os.environ.pop("ZKDL_FORMAL_LOCK_FD", None)
    os.environ.pop("COMPARISON_FORMAL_OWNER_TOKEN", None)
    os.environ.pop("COMPARISON_FORMAL_ROOT_LOCK_FD", None)


def _config(session: Path, dataset: str) -> Path:
    return ROOT / "configs" / f"{session.name}_{dataset}.cfg"


def _build_manifest(session: Path) -> Path:
    return session / "build_manifest.json"


def _graph_identity(native: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not set(native) or not set(native).issubset(DATASETS):
        raise PhaseError("native admissions do not contain a canonical admitted dataset subset")
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        if dataset not in native:
            continue
        admission = native[dataset]
        digest = admission.get("canonical_graph_digest")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise PhaseError(f"native admission lacks canonical graph digest for {dataset}")
        counts: dict[str, int] = {}
        for field in ("canonical_graph_nodes", "canonical_graph_edges", "canonical_graph_count"):
            value = admission.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PhaseError(f"native admission has an invalid {field} for {dataset}")
            counts[field] = value
        node_ptr = admission.get("canonical_graph_node_ptr")
        edge_ptr = admission.get("canonical_graph_edge_ptr")
        for field, values, expected_size, terminal in (
            ("node_ptr", node_ptr, counts["canonical_graph_count"] + 1,
             counts["canonical_graph_nodes"]),
            ("edge_ptr", edge_ptr, counts["canonical_graph_count"] + 1,
             counts["canonical_graph_edges"]),
        ):
            if (not isinstance(values, list) or len(values) != expected_size or
                    any(not isinstance(value, int) or isinstance(value, bool) for value in values) or
                    not values or values[0] != 0 or values[-1] != terminal or
                    any(values[index] <= values[index - 1] for index in range(1, len(values)))):
                raise PhaseError(f"native admission has malformed canonical {field} for {dataset}")
        preprocess = admission.get("input_preprocess_id")
        batching = admission.get("batching_rule")
        if not isinstance(preprocess, str) or not preprocess or not isinstance(batching, str) or not batching:
            raise PhaseError(f"native admission lacks preprocessing/batching identity for {dataset}")
        datasets[dataset] = {
            "digest": digest,
            "nodes": counts["canonical_graph_nodes"],
            "edges": counts["canonical_graph_edges"],
            "graphs": counts["canonical_graph_count"],
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "input_preprocess_id": preprocess,
            "batching_rule": batching,
        }
    return {"schema": "zkdl_canonical_graph_set_v1", "complete": set(native) == set(DATASETS),
            "admitted_datasets": [dataset for dataset in DATASETS if dataset in native], "datasets": datasets}


def _checked_capacity_product(left: int, right: int, label: str) -> int:
    maximum = (1 << 64) - 1
    if (type(left) is not int or type(right) is not int or left < 0 or right < 0 or
            (right != 0 and left > maximum // right)):
        raise PhaseError(f"native capacity {label} overflows u64")
    return left * right


def _checked_capacity_sum(*values: int, label: str) -> int:
    maximum = (1 << 64) - 1
    total = 0
    for value in values:
        if type(value) is not int or value < 0 or value > maximum - total:
            raise PhaseError(f"native capacity {label} overflows u64")
        total += value
    return total


def _native_public_payload_bounds(admission: dict[str, Any], nodes: int, classes: int,
                                  proof_bytes: int, operation_descriptor_bytes: int) -> dict[str, int]:
    """Derive resident Y and authenticated static-table payloads from native typed counts."""
    maximum = (1 << 64) - 1
    typed_counts = (
        "native_public_y_padded_domain_elements",
        "native_static_table_raw_i64_capacity_elements",
        "native_static_table_padded_field_elements",
    )
    counts: dict[str, int] = {}
    for field in typed_counts:
        value = admission.get(field)
        if type(value) is not int or value <= 0 or value > maximum:
            raise PhaseError(f"native admission lacks a positive source-typed metric {field}")
        counts[field] = value
    if type(nodes) is not int or type(classes) is not int or nodes <= 0 or classes <= 0:
        raise PhaseError("native capacity public_Y logical shape is invalid")
    if nodes > (1 << 30) or classes > (1 << 30):
        raise PhaseError("native capacity public_Y logical shape exceeds the native domain")
    padded_nodes = 1 << (nodes - 1).bit_length()
    padded_classes = 1 << (classes - 1).bit_length()
    expected_domain = _checked_capacity_product(padded_nodes, padded_classes, "public_Y padded domain")
    public_y_domain = counts["native_public_y_padded_domain_elements"]
    if expected_domain > (1 << 30) or public_y_domain != expected_domain:
        raise PhaseError("native admission public_Y padded domain differs from the authenticated logical shape")
    if (type(proof_bytes) is not int or type(operation_descriptor_bytes) is not int or
            proof_bytes <= 0 or operation_descriptor_bytes < 0 or
            proof_bytes > maximum or operation_descriptor_bytes > maximum):
        raise PhaseError("native capacity proof/descriptor wire counts are invalid")

    public_y_field_bytes = _checked_capacity_product(public_y_domain, 32, "public_Y field bytes")
    # The encoded field vector contributes its u64 element count; append_statement
    # adds a second u64 byte-length frame around that encoded vector.
    public_y_wire_bytes = _checked_capacity_sum(public_y_field_bytes, 16, label="public_Y proof wire bytes")
    non_public_y_wire_and_descriptor = _checked_capacity_sum(
        operation_descriptor_bytes, public_y_wire_bytes, label="public_Y/descriptor proof wire bytes")
    if non_public_y_wire_and_descriptor > proof_bytes:
        raise PhaseError("native proof wire bound is smaller than its descriptor/public_Y sections")
    other_proof_wire_bytes = proof_bytes - non_public_y_wire_and_descriptor
    other_proof_object_and_wire_bytes = _checked_capacity_sum(
        _checked_capacity_product(other_proof_wire_bytes, 4, "non-public_Y proof object/wire bound"),
        operation_descriptor_bytes, label="non-public_Y proof object/wire bound")

    prove_public_y_live = _checked_capacity_sum(
        _checked_capacity_product(public_y_field_bytes, 5, "prover public_Y live copies"),
        32, label="prover public_Y live copies")
    verify_public_y_live = _checked_capacity_sum(
        _checked_capacity_product(public_y_field_bytes, 6, "verifier public_Y live copies"),
        40, label="verifier public_Y live copies")
    logical_output_elements = _checked_capacity_product(nodes, classes, "public_Y logical matrix elements")
    verify_public_y_matrix_bytes = _checked_capacity_product(
        logical_output_elements, 8, "verifier logical public_Y matrix bytes")
    verify_public_y_live = _checked_capacity_sum(
        verify_public_y_live, verify_public_y_matrix_bytes,
        label="verifier public_Y live copies and logical matrix")

    static_table_raw_i64_bytes = _checked_capacity_product(
        counts["native_static_table_raw_i64_capacity_elements"], 8,
        "raw static-table resident bytes")
    static_table_padded_field_bytes = _checked_capacity_product(
        counts["native_static_table_padded_field_elements"], 32,
        "padded static-table resident bytes")
    public_y_witness_block_bytes = _checked_capacity_product(
        min(public_y_domain, 1 << 18), 32, "public_Y witness read block bytes")
    public_y_witness_materialize_live = _checked_capacity_sum(
        public_y_field_bytes, public_y_witness_block_bytes,
        label="public_Y witness materialization")
    # Statement construction pads a temporary Fr vector and then encodes the
    # padded vector while the trace and source Matrix remain resident. The
    # encoder output is another 32P-byte vector plus its u64 element count.
    public_y_digest_live = _checked_capacity_sum(
        _checked_capacity_product(public_y_field_bytes, 2, "public_Y statement digest buffers"),
        8, label="public_Y statement digest buffers")
    public_y_witness_live = max(public_y_witness_materialize_live, public_y_digest_live)

    body_fields = (
        "native_hidden_matmul_body_max_bytes",
        "native_sparse_route_body_max_bytes",
        "native_fac_line8_body_max_bytes",
        "native_constraint_body_max_bytes",
        "native_padding_record_body_max_bytes",
        "native_sumcheck_body_max_bytes",
        "native_terminal_relation_body_max_bytes",
        "native_sorted_lookup_relation_body_max_bytes",
        "native_sorted_lookup_proof_body_max_bytes",
    )
    bodies: dict[str, int] = {}
    for field in body_fields:
        value = admission.get(field)
        if type(value) is not int or value <= 0 or value > maximum:
            raise PhaseError(f"native admission lacks a positive source-typed body bound {field}")
        bodies[field] = value
    hidden_body = bodies["native_hidden_matmul_body_max_bytes"]
    route_body = bodies["native_sparse_route_body_max_bytes"]
    fac_body = bodies["native_fac_line8_body_max_bytes"]
    constraint_body = bodies["native_constraint_body_max_bytes"]
    padding_body = bodies["native_padding_record_body_max_bytes"]
    sumcheck_body = bodies["native_sumcheck_body_max_bytes"]
    terminal_body = bodies["native_terminal_relation_body_max_bytes"]
    lookup_relation_body = bodies["native_sorted_lookup_relation_body_max_bytes"]
    lookup_proof_body = bodies["native_sorted_lookup_proof_body_max_bytes"]
    child_capacity_peak = _checked_capacity_sum(
        _checked_capacity_product(sumcheck_body, 2, "nested sumcheck capacity"),
        _checked_capacity_product(terminal_body, 2, "nested terminal capacity"),
        label="nested sumcheck/terminal capacities")
    constraint_record_body = _checked_capacity_sum(
        constraint_body, 4, label="constraint BCH2 record body")
    padding_serializer_peak = _checked_capacity_sum(
        _checked_capacity_product(padding_body, 2, "padding-record vector capacity"),
        _checked_capacity_product(constraint_record_body, 2, "padding relation vector capacity"),
        child_capacity_peak, label="padding serialization scratch")
    operation_batch_peak = _checked_capacity_sum(
        _checked_capacity_product(constraint_record_body, 2, "operation constraint vector capacity"),
        child_capacity_peak, label="operation constraint serialization scratch")
    lookup_relation_build_peak = _checked_capacity_sum(
        _checked_capacity_product(lookup_proof_body, 2, "sorted-lookup outer vector capacity"),
        _checked_capacity_product(lookup_relation_body, 2, "sorted-lookup relation vector capacity"),
        child_capacity_peak, label="sorted-lookup relation serialization scratch")
    # On a growth-triggering insert, the old and new outer buffers coexist.
    # With a doubling vector growth bound their combined capacities are at
    # most 3x the final body size; the encoded relation remains live too.
    lookup_outer_append_peak = _checked_capacity_sum(
        _checked_capacity_product(lookup_proof_body, 3, "sorted-lookup outer reallocation"),
        _checked_capacity_product(lookup_relation_body, 2, "sorted-lookup encoded relation capacity"),
        label="sorted-lookup outer append scratch")
    proof_serialization_scratch_peak = max(
        padding_serializer_peak, operation_batch_peak,
        lookup_relation_build_peak, lookup_outer_append_peak)
    lookup_parse_peak = _checked_capacity_sum(
        _checked_capacity_product(lookup_proof_body, 2, "sorted-lookup parser body copy"),
        _checked_capacity_product(lookup_relation_body, 2, "sorted-lookup parser relation copy"),
        child_capacity_peak, label="sorted-lookup parse scratch")
    padding_parse_peak = padding_serializer_peak
    relation_transcript_peak = _checked_capacity_sum(
        _checked_capacity_product(lookup_relation_body, 2, "sorted-lookup transcript relation vector"),
        child_capacity_peak, label="sorted-lookup transcript scratch")
    opaque_verify_candidates = (
        _checked_capacity_sum(
            _checked_capacity_product(hidden_body, 8, "hidden opaque typed-proof capacity"),
            _checked_capacity_product(sumcheck_body, 2, "hidden opaque sumcheck capacity"),
            8192, label="hidden opaque verify scratch"),
        _checked_capacity_sum(
            _checked_capacity_product(route_body, 8, "route opaque typed-proof capacity"),
            _checked_capacity_product(sumcheck_body, 2, "route opaque sumcheck capacity"),
            8192, label="route opaque verify scratch"),
        _checked_capacity_sum(
            _checked_capacity_product(fac_body, 8, "FAC opaque typed-proof capacity"),
            _checked_capacity_product(route_body, 2, "FAC route relation capacity"),
            _checked_capacity_product(sumcheck_body, 2, "FAC opaque sumcheck capacity"),
            8192, label="FAC opaque verify scratch"),
        _checked_capacity_sum(
            _checked_capacity_product(constraint_record_body, 8, "constraint opaque typed-proof capacity"),
            _checked_capacity_product(sumcheck_body, 2, "constraint opaque sumcheck capacity"),
            _checked_capacity_product(terminal_body, 2, "constraint opaque terminal capacity"),
            label="constraint opaque verify scratch"),
    )
    opaque_operation_verify_peak = max(opaque_verify_candidates)
    verifier_nested_scratch_peak = max(
        lookup_parse_peak, padding_parse_peak, relation_transcript_peak,
        child_capacity_peak, opaque_operation_verify_peak)
    return {
        "public_y_field_bytes": public_y_field_bytes,
        "public_y_proof_wire_bytes": public_y_wire_bytes,
        "other_proof_object_and_wire_bytes": other_proof_object_and_wire_bytes,
        "public_y_prove_live_bytes": prove_public_y_live,
        "public_y_verify_live_bytes": verify_public_y_live,
        "public_y_witness_live_bytes": public_y_witness_live,
        "public_y_statement_digest_live_bytes": public_y_digest_live,
        "static_table_raw_i64_bytes": static_table_raw_i64_bytes,
        "static_table_padded_field_bytes": static_table_padded_field_bytes,
        "operation_constraint_serialization_scratch_bytes": operation_batch_peak,
        "proof_nested_serialization_scratch_bytes": proof_serialization_scratch_peak,
        "verify_nested_proof_scratch_bytes": verifier_nested_scratch_peak,
        "verify_opaque_typedproof_scratch_bytes": opaque_operation_verify_peak,
    }


def _admission_profile(session: Path, session_id: str, preflight: dict[str, Any],
                       native: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    # Sizes and typed source payloads come from the production native
    # RelationPlan/OperationDescriptor model.  The native producer has already
    # checked every nested body against the exact consumer's parser limit.
    datasets: dict[str, dict[str, Any]] = {}
    native_capacity_diagnostics: dict[str, Any] = {}
    finalization_manifest_capacity: dict[str, dict[str, int]] = {}
    mib = 1 << 20
    cell_log_reserve = 16 * mib
    finalization_bytes = 0
    for dataset in DATASETS:
        if dataset not in native:
            continue
        input_report = preflight["datasets"][dataset]
        audit = input_report["resource_prerequisites"]
        admission = native[dataset]
        srs_bytes = int(audit["srs_plan"]["estimated_materialized_bytes"])
        srs_plan = audit["srs_plan"]
        basis_domain = srs_plan.get("basis_domain")
        frame_bytes = srs_plan.get("frame_bytes")
        if (type(basis_domain) is not int or basis_domain != (1 << 18) or
                type(frame_bytes) is not int or frame_bytes != 40 + basis_domain * 48 or
                srs_bytes < frame_bytes + 2 * (40 + 48)):
            raise PhaseError(f"resource audit has an invalid source-bound SRS setup plan for {dataset}")
        srs_families = srs_plan.get("families")
        family_domains = audit.get("family_domains")
        declared_family_count = audit.get("family_domain_count")
        if (not isinstance(srs_families, dict) or not srs_families or not isinstance(family_domains, dict) or
                set(srs_families) != set(family_domains) or type(declared_family_count) is not int or
                declared_family_count != len(srs_families)):
            raise PhaseError(f"resource audit has malformed exact SRS manifest family counts for {dataset}")
        manifest_segment_count = 0
        for family_id, family_plan in srs_families.items():
            if not isinstance(family_id, str) or not family_id.isascii() or len(family_id.encode("ascii")) > 512 or not isinstance(family_plan, dict):
                raise PhaseError(f"resource audit has an invalid SRS manifest family identity for {dataset}")
            logical_domain = family_plan.get("logical_domain")
            segment_size = family_plan.get("segment_size")
            segment_count = family_plan.get("segment_count")
            if (type(logical_domain) is not int or logical_domain < 1 or logical_domain > (1 << 30) or
                    logical_domain & (logical_domain - 1) or type(segment_size) is not int or
                    segment_size != basis_domain or type(segment_count) is not int or segment_count < 1 or
                    segment_count > 4096 or segment_count != (logical_domain + segment_size - 1) // segment_size or
                    type(family_domains.get(family_id)) is not int or family_domains[family_id] != logical_domain):
                raise PhaseError(f"resource audit has an invalid SRS manifest segment count for {dataset}/{family_id}")
            manifest_segment_count += segment_count
            if manifest_segment_count > (1 << 64) - 1:
                raise PhaseError(f"resource audit SRS manifest segment count overflows u64 for {dataset}")
        finalization_manifest_capacity[dataset] = _srs_manifest_finalization_capacity(len(srs_families), manifest_segment_count)
        family_bytes = int(admission["private_family_storage_upper_bound_bytes"])
        sorted_bytes = int(admission["sorted_lookup_scratch_upper_bound_bytes"])
        tensor_count = int(admission["checkpoint_tensor_count"])
        family_count = int(admission["tensor_family_count"])
        operation_count = int(admission["operation_count"])
        lookup_count = int(admission["sorted_lookup_query_count"])
        typed_fields = (
            "native_operation_descriptor_bytes",
            "native_operation_descriptor_max_bytes",
            "native_sumcheck_body_max_bytes",
            "native_terminal_relation_body_max_bytes",
            "native_constraint_body_max_bytes",
            "native_padding_record_body_max_bytes",
            "native_hidden_matmul_body_max_bytes",
            "native_sparse_route_body_max_bytes",
            "native_fac_line8_body_max_bytes",
            "native_sorted_lookup_relation_body_max_bytes",
            "native_sorted_lookup_proof_body_max_bytes",
            "native_relation_proof_sections_wire_bytes_upper_bound",
            "native_proof_wire_bytes_upper_bound",
            "native_public_y_padded_domain_elements",
            "native_static_table_raw_i64_capacity_elements",
            "native_static_table_padded_field_elements",
            "native_commitment_section_wire_bytes_upper_bound",
            "native_operation_descriptor_resident_bytes_upper_bound",
            "native_trace_i64_payload_bytes",
            "native_trace_field_payload_bytes",
            "native_trace_u8_payload_bytes",
            "native_trace_object_metadata_bytes",
            "native_trace_forward_live_peak_bytes_upper_bound",
            "native_route_sumcheck_live_peak_bytes_upper_bound",
            "native_segmented_sumcheck_live_peak_bytes_upper_bound",
            "native_sumcheck_fallback_call_count",
            "native_sumcheck_cached_call_count",
            "native_sumcheck_spill_call_count",
            "native_sumcheck_spill_provider_value_slots_upper_bound",
            "native_sumcheck_spill_provider_scan_bytes_upper_bound",
            "native_sumcheck_spill_provider_source_bytes_upper_bound",
            "native_sumcheck_spill_storage_io_bytes_upper_bound",
            "native_sumcheck_max_spill_disk_peak_bytes_upper_bound",
            "native_sumcheck_max_spill_ram_peak_bytes_upper_bound",
            "native_maximum_route_source_domain",
            "native_maximum_route_edge_domain",
            "native_maximum_route_selector_factor_elements",
            "relation_plan_text_bytes",
        )
        for field in typed_fields:
            if type(admission.get(field)) is not int or admission[field] < 0:
                raise PhaseError(f"native admission lacks a nonnegative source-typed metric {field} for {dataset}")
        route_source_domain = int(admission["native_maximum_route_source_domain"])
        route_factor_elements = int(admission["native_maximum_route_selector_factor_elements"])
        if (route_source_domain == 0 or route_source_domain > (1 << 30) or
                route_factor_elements == 0 or route_factor_elements > (1 << 64) - 1 or
                route_factor_elements > route_source_domain + 1):
            raise PhaseError(f"native admission returned an invalid route selector factor bound for {dataset}")
        proof_bytes = int(admission["native_proof_wire_bytes_upper_bound"])
        operation_descriptor_bytes = int(admission["native_operation_descriptor_bytes"])
        if proof_bytes == 0 or operation_descriptor_bytes > proof_bytes:
            raise PhaseError(f"native admission returned an invalid proof/descriptor wire bound for {dataset}")
        spill_call_count = int(admission["native_sumcheck_spill_call_count"])
        spill_fallback_count = int(admission["native_sumcheck_fallback_call_count"])
        spill_disk_peak = int(admission["native_sumcheck_max_spill_disk_peak_bytes_upper_bound"])
        spill_ram_peak = int(admission["native_sumcheck_max_spill_ram_peak_bytes_upper_bound"])
        spill_io_bytes = int(admission["native_sumcheck_spill_storage_io_bytes_upper_bound"])
        spill_source_bytes = int(admission["native_sumcheck_spill_provider_source_bytes_upper_bound"])
        if ((spill_call_count == 0) != (spill_disk_peak == 0) or
                (spill_call_count == 0) != (spill_ram_peak == 0) or
                (spill_call_count == 0) != (spill_io_bytes == 0) or
                (spill_call_count == 0) != (spill_source_bytes == 0) or
                spill_disk_peak > (1 << 37) or spill_ram_peak > (1 << 30)):
            raise PhaseError(f"native admission returned an inconsistent Sumcheck spill envelope for {dataset}")
        if spill_fallback_count != 0:
            raise PhaseError(f"native Sumcheck still reports an unbounded in-memory fallback for {dataset}")
        if spill_ram_peak > int(admission["native_segmented_sumcheck_live_peak_bytes_upper_bound"]):
            raise PhaseError(f"native Sumcheck phase RAM bound omits its spill working set for {dataset}")
        nodes = int(admission["nodes"])
        edges = int(admission["edges"])
        graphs = int(admission["graphs"])
        # The authenticated resource-prerequisite record is the authoritative
        # source for input shape.  The per-dataset preflight row has no
        # top-level input_shape field.
        input_shape = audit["input_shape"]
        features = int(input_shape["features"])
        classes = int(input_shape["classes"])
        public_payload = _native_public_payload_bounds(
            admission, nodes, classes, proof_bytes, operation_descriptor_bytes)
        max_merge_rows = int(admission["maximum_sorted_lookup_merge_logical_rows"])
        model_segments = int(admission["model_commitment_segment_count"])
        model_elements = int(admission["model_parameter_elements"])

        # Registry entries are JSON encodings of the native model tensors and
        # their exact segment counts; the per-entry/per-segment allowance
        # covers field names, digests, compressed points and JSON framing.
        registration_bytes = 256 * 1024 + tensor_count * 8192 + model_segments * 2048
        sidecar_bytes = 64 * 1024 + tensor_count * 1024
        input_identity = input_report["input_identity"]
        asset_file_count = (len(input_identity["cache_files"]) + len(input_identity["checkpoint_files"]) +
                            family_count + operation_count + lookup_count + 128)
        manifest_bytes = asset_file_count * 1024 + registration_bytes + 256 * 1024
        relation_plan_bytes = int(admission["relation_plan_text_bytes"])
        public_y_bytes = nodes * classes * 24 + 64
        statement_and_run_bytes = nodes * classes * 32 + public_y_bytes + 2 * mib
        receipt_bytes = 1 * mib
        retained = (srs_bytes + registration_bytes + sidecar_bytes + manifest_bytes +
                    relation_plan_bytes + proof_bytes + statement_and_run_bytes + receipt_bytes)
        # The cell peak includes the source-sized private witness files and
        # sorted T/Q/S scratch while they coexist with retained outputs.
        additional = retained + family_bytes + sorted_bytes + spill_disk_peak + cell_log_reserve

        graph_memory = (nodes * features * 8 + edges * 8 + nodes * 48 +
                        (graphs + 1) * 16 + (nodes + 1) * 8)
        trace_payload = _checked_capacity_sum(
            int(admission["native_trace_i64_payload_bytes"]),
            int(admission["native_trace_field_payload_bytes"]),
            int(admission["native_trace_u8_payload_bytes"]),
            int(admission["native_trace_object_metadata_bytes"]),
            label="native trace payload")
        forward_peak = int(admission["native_trace_forward_live_peak_bytes_upper_bound"])
        segmented_sumcheck_scratch = int(admission["native_segmented_sumcheck_live_peak_bytes_upper_bound"])
        descriptor_resident = int(admission["native_operation_descriptor_resident_bytes_upper_bound"])
        sorting_workspace = _checked_capacity_sum(
            _checked_capacity_product(max_merge_rows, 256, "sorted lookup workspace"),
            8 * mib, label="sorted lookup workspace")
        # CPU G1 MSM uses BLST Pippenger in fixed 16,384-point chunks.  Include
        # the maximum 20-worker caller/pool scratch, the prepared affine basis,
        # per-segment receipts, per-family receipt metadata, and sorted-lookup
        # receipt references.  512 MiB covers 20 x the 16-MiB workspace bound
        # plus the known basis/receipt payloads and allocator headroom.  Count
        # it in each phase that executes G1 arithmetic, including independent
        # verification.
        cpu_g1_msm_workspace = 512 * mib
        # A retained typed Proof and serialization buffer coexist.  For
        # non-descriptor proof payloads, the source-wire payload is multiplied
        # by the native/wire point ratio (Jacobian G1 is at most 3x its
        # 48-byte canonical encoding) plus the serialized byte vector.  Native
        # descriptor resident capacity includes one interned canonical
        # expected tree, a conservative unshared decoded tree for the received
        # proof, both header arrays, and two maximum-wire comparison buffers.
        # Keep the separately retained serialized descriptor bytes in the
        # proof object/wire bound below.
        proof_object_and_wire = public_payload["other_proof_object_and_wire_bytes"]
        srs_memory = _checked_capacity_product(srs_bytes, 4, "materialized SRS memory")
        model_and_graph = _checked_capacity_sum(
            graph_memory,
            _checked_capacity_product(model_elements, 8, "model parameter memory"),
            public_payload["static_table_raw_i64_bytes"],
            label="model/graph/raw static tables")
        prove_and_serialize_public_material = _checked_capacity_sum(
            public_payload["public_y_prove_live_bytes"],
            public_payload["static_table_padded_field_bytes"],
            label="prover/serializer public material")
        verify_public_material = _checked_capacity_sum(
            public_payload["public_y_verify_live_bytes"],
            public_payload["static_table_padded_field_bytes"],
            public_payload["verify_nested_proof_scratch_bytes"],
            label="verifier public material and nested proof buffers")
        prove_public_material = _checked_capacity_sum(
            prove_and_serialize_public_material,
            public_payload["proof_nested_serialization_scratch_bytes"],
            label="prover public material and nested serializer buffers")
        witness_public_materialization_peak = _checked_capacity_sum(
            model_and_graph, trace_payload, public_payload["public_y_witness_live_bytes"],
            label="public_Y witness materialization phase")
        native_phase_peak_bytes = {
            "forward": _checked_capacity_sum(
                model_and_graph, forward_peak, label="forward phase peak"),
            "sorted_lookup": _checked_capacity_sum(
                model_and_graph, trace_payload, sorting_workspace, descriptor_resident,
                label="sorted lookup phase peak"),
            "prove": _checked_capacity_sum(
                model_and_graph, descriptor_resident, segmented_sumcheck_scratch,
                sorting_workspace, proof_object_and_wire, prove_public_material, srs_memory,
                cpu_g1_msm_workspace, 256 * mib, label="prove phase peak"),
            "serialize": _checked_capacity_sum(
                model_and_graph, descriptor_resident, proof_object_and_wire,
                prove_public_material, srs_memory, 256 * mib,
                label="serialization phase peak"),
            "fresh_independent_verify": _checked_capacity_sum(
                model_and_graph, descriptor_resident, proof_object_and_wire,
                verify_public_material, srs_memory, cpu_g1_msm_workspace, 256 * mib,
                label="independent verification phase peak"),
            "public_y_witness_materialization": witness_public_materialization_peak,
        }
        # Native setup materializes one 48-byte G1 wire per G/H/U point in
        # std::set<Wire>, then uses two 1024-point blst batches.  The 512-byte
        # per-point allowance bounds the set node, allocator bookkeeping, and
        # wire; the bounded batch and fixed reserve cover the actual projective,
        # affine, scalar, stream, and process buffers.  The independent native
        # SRS exporter writes two decimal coordinates per point to captured
        # stdout; Python simultaneously holds that text, the parsed JSON tree,
        # and its converted tuple/int list, so charge 2048 bytes per generator.
        # Registry serialization and model-side conversion remain live in this
        # setup phase and are bounded by their authenticated source sizes.
        srs_point_count = basis_domain + 2
        setup_generation_peak = srs_point_count * 512 + 1024 * 8192 + 256 * mib
        setup_decode_registration_peak = (
            basis_domain * 2048 + srs_bytes * 4 + model_elements * 64 +
            registration_bytes + sidecar_bytes + manifest_bytes + 256 * mib
        )
        setup_peak = max(setup_generation_peak, setup_decode_registration_peak)
        witness_peak = max(native_phase_peak_bytes["forward"], native_phase_peak_bytes["sorted_lookup"],
                           native_phase_peak_bytes["public_y_witness_materialization"])
        # Tensor commitments are constructed inside the native relation prove
        # call; its complete live-memory envelope is therefore also the
        # source-derived commitment-phase upper bound.
        commitment_peak = native_phase_peak_bytes["prove"]
        common_phase_peak_bytes = {
            "setup": setup_peak,
            "witness": witness_peak,
            "commitment": commitment_peak,
            "prove": native_phase_peak_bytes["prove"],
            "verify": native_phase_peak_bytes["fresh_independent_verify"],
        }
        host_peak = max(common_phase_peak_bytes.values())
        datasets[dataset] = {
            "additional_disk_peak_bytes": additional,
            "sumcheck_spill_call_count": spill_call_count,
            "sumcheck_fallback_call_count": spill_fallback_count,
            "sumcheck_spill_disk_peak_bytes_upper_bound": spill_disk_peak,
            "sumcheck_spill_ram_peak_bytes_upper_bound": spill_ram_peak,
            "sumcheck_spill_provider_source_bytes_upper_bound": spill_source_bytes,
            "sumcheck_spill_storage_io_bytes_upper_bound": spill_io_bytes,
            "cell_retained_artifact_bytes_upper_bound": retained,
            "host_peak_bytes": host_peak,
            "device_peak_bytes": 0,
            "native_route_source_domain": route_source_domain,
            "native_route_selector_factor_elements": route_factor_elements,
            "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
        }
        native_capacity_diagnostics[dataset] = {
            "srs_basis_domain": basis_domain,
            "srs_point_count": srs_point_count,
            "srs_materialized_bytes_upper_bound": srs_bytes,
            "native_route_source_domain": route_source_domain,
            "native_route_selector_factor_elements": route_factor_elements,
            "native_payload_live_bounds_bytes": public_payload,
            "phase_peak_bytes": {
                **native_phase_peak_bytes,
                "setup_native_srs_generation": setup_generation_peak,
                "setup_srs_decode_and_registry": setup_decode_registration_peak,
                "setup": setup_peak,
                "witness": witness_peak,
                "commitment": commitment_peak,
                "verify": native_phase_peak_bytes["fresh_independent_verify"],
            },
        }
        finalization_bytes += receipt_bytes + 1 * mib
    existing_bytes = 0
    for path in session.rglob("*"):
        if path.is_symlink():
            raise PhaseError(f"session tree contains a symlink during artifact audit: {path}")
        if path.is_file():
            existing_bytes += path.stat().st_size
    device = os.stat(session, follow_symlinks=False).st_dev
    finalization_host_peak = max(item["host_peak_bytes_upper_bound"] for item in finalization_manifest_capacity.values())
    profile = {
        "schema": "zkdl_formal_capacity_profile_v3",
        "baseline": "zkdl",
        "session_id": session_id,
        "fs_device_id": int(device),
        "preexisting_baseline_artifact_bytes": existing_bytes,
        "finalization_disk_bytes_upper_bound": finalization_bytes,
        "finalization_host_peak_bytes_upper_bound": finalization_host_peak,
        "finalization_manifest_capacity": {"schema": "zkdl_srs_manifest_finalization_heap_v1",
                                            "datasets": finalization_manifest_capacity},
        "backend": "cpu",
        "memory_accounting_kind": "host-only",
        "device": None,
        "bound_methods": {
            "additional_disk_peak": "source_derived_upper_bound",
            "host_peak": "source_derived_upper_bound",
            "finalization_host_peak": "srs_manifest_schema_python_heap_upper_bound",
        },
        "datasets": datasets,
    }
    return profile, native_capacity_diagnostics


def _session_artifact_bytes(session: Path) -> int:
    total = 0
    for path in session.rglob("*"):
        if path.is_symlink():
            raise PhaseError(f"session tree contains a symlink during artifact audit: {path}")
        if path.is_file() and "logs" not in path.relative_to(session).parts:
            total += path.stat(follow_symlinks=False).st_size
    return total


def _dataset_status(dataset: str, status: str, stage: str, failure: str = "") -> dict[str, str]:
    return {"dataset": dataset, "status": status, "stage": stage, "failure": failure[:2048]}


def _append_log(path: Path, text: str) -> None:
    require_session_artifact(path, ROOT)
    if path.is_symlink():
        raise PhaseError(f"refusing to append through a symlinked formal log: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PhaseError(f"formal log directory is missing or unsafe: {path.parent}")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.rstrip("\n") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _remove_failed_snapshot(session: Path, dataset: str) -> None:
    identity = require_bound_session(session, ROOT)
    paths = (session / "resources" / "cache" / dataset,
             session / "resources" / "checkpoints" / f"{dataset}_gat",
             session / dataset,
             ROOT / "configs" / f"{session.name}_{dataset}.cfg")
    for path in paths:
        require_session_root(identity, session)
        if path.is_symlink():
            raise PhaseError(f"refusing to remove symlinked partial snapshot path: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    require_session_root(identity, session)
    for path in (session / "resources" / "cache", session / "resources" / "checkpoints",
                 session / "resources", session / "data"):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _write_prepare_outcome(session: Path, session_id: str, outcomes: dict[str, dict[str, str]],
                           preflight: dict[str, Any] | None, error: str,
                           build_manifest: Path | None = None, native: dict[str, dict[str, Any]] | None = None,
                           diagnostics: dict[str, Any] | None = None, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    identity = require_bound_session(session, ROOT)
    passed = sum(row["status"] == "PASS" for row in outcomes.values())
    overall = "PASS" if passed == len(DATASETS) else "PARTIAL" if passed else "FAILED"
    graph = _graph_identity(native or {}) if native else {
        "schema": "zkdl_canonical_graph_set_v1", "complete": False, "admitted_datasets": [], "datasets": {}}
    receipt: dict[str, Any] = {
        "schema": "zkDL_formal_prepare_receipt_v2", "baseline": "zkdl", "session_id": session_id,
        "status": overall, "failure": error[:2048], "dataset_status": outcomes,
        "preflight_report": preflight, "native_admissions": native or {},
        "native_capacity_diagnostics": diagnostics or {}, "graph_identity": graph,
        "capacity_profile": profile,
    }
    if build_manifest is not None and build_manifest.is_file():
        build_doc = json.loads(build_manifest.read_text(encoding="utf-8"))
        receipt.update({"source_digest": build_doc["source_digest"],
                        "build_manifest_path": build_manifest.relative_to(ROOT).as_posix(),
                        "build_manifest_sha256": sha256_file(build_manifest)})
    receipt_path = session / "phase" / "prepare.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_session_root(identity, session)
    marker: dict[str, Any] = {
        "zkdl_phase": "prepare", "status": overall, "session_id": session_id,
        "session_dir": session.relative_to(ROOT).as_posix(),
        "prepare_receipt_path": receipt_path.relative_to(ROOT).as_posix(),
        "prepare_receipt_sha256": sha256_file(receipt_path), "dataset_status": outcomes,
        "failure": error[:2048], "graph_identity": graph,
    }
    if build_manifest is not None and build_manifest.is_file():
        build_log = session / "logs" / "build.log"
        marker.update({"build_log_path": build_log.relative_to(ROOT).as_posix(),
                      "build_log_sha256": sha256_file(build_log),
                      "build_manifest_path": receipt["build_manifest_path"],
                      "build_manifest_sha256": receipt["build_manifest_sha256"],
                      "source_digest": receipt["source_digest"]})
    if profile is not None:
        marker["capacity_profile"] = profile
    print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)
    return marker


def prepare(session: Path, session_id: str) -> None:
    identity = require_bound_session(session, ROOT)
    if session.is_symlink() or not session.is_dir():
        raise PhaseError(f"zkDL session must be created by the baseline-local formal runner: {session}")
    entries = list(session.iterdir())
    if len(entries) != 1 or entries[0].name != "state.json" or entries[0].is_symlink() or not entries[0].is_file():
        raise PhaseError("zkDL formal prepare refuses a session containing artifacts before local ownership is established")
    try:
        state = json.loads(entries[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseError("zkDL formal runner session state is unreadable before prepare") from exc
    if (not isinstance(state, dict) or state.get("schema") != "zkdl_formal_runner_state_v1" or
            state.get("baseline") != "zkdl" or state.get("session_id") != session_id or
            state.get("status") != "RUNNING" or state.get("stage") != "prepare"):
        raise PhaseError("zkDL formal runner session state is not fresh for prepare")
    require_session_root(identity, session)
    (session / "logs").mkdir()
    (session / "phase").mkdir()
    require_session_root(identity, session)
    admission_log = session / "logs" / "admission.log"
    report_path = session / "dataset_prerequisites.json"
    env = dict(os.environ)
    env["ZKDL_PREFLIGHT_REPORT"] = str(report_path)
    _run([sys.executable, str(ROOT / "scripts/formal_preflight.py")],
         log_path=admission_log, fresh_log=True, env=env)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "zkDL_formal_input_preflight_v1" or set(report.get("datasets", {})) != set(DATASETS):
        raise PhaseError("fresh formal input preflight did not produce the complete canonical five-dataset report")
    outcomes = {
        dataset: _dataset_status(dataset,
                                 "PASS" if report["datasets"][dataset].get("status") == "PASS" else "FAILED",
                                 "preflight" if report["datasets"][dataset].get("status") != "PASS" else "pending",
                                 str(report["datasets"][dataset].get("failure", "")))
        for dataset in DATASETS
    }
    dataset_logs = {dataset: session / "logs" / f"{dataset}.log" for dataset in DATASETS}
    for dataset in DATASETS:
        _append_log(dataset_logs[dataset], json.dumps({"stage": "preflight", **report["datasets"][dataset]},
                                                       sort_keys=True, separators=(",", ":")))
    for dataset in DATASETS:
        if outcomes[dataset]["status"] != "PASS":
            continue
        try:
            require_session_root(identity)
            _run([sys.executable, str(ROOT / "scripts/formal_session.py"), "--session", str(session),
                  "--dataset", dataset], log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env)
        except SessionRootError:
            raise
        except Exception as exc:
            _remove_failed_snapshot(session, dataset)
            outcomes[dataset] = _dataset_status(dataset, "FAILED", "snapshot", f"{type(exc).__name__}: {exc}")
            _append_log(dataset_logs[dataset], json.dumps({"stage": "snapshot", "status": "FAILED",
                                                           "error": outcomes[dataset]["failure"]}, separators=(",", ":")))

    if not any(row["status"] == "PASS" for row in outcomes.values()):
        _write_prepare_outcome(session, session_id, outcomes, report,
                               "no dataset passed full-input preflight and snapshot admission")
        return

    build_log = session / "logs" / "build.log"
    build_manifest = _build_manifest(session)
    try:
        require_session_root(identity, session)
        _run([sys.executable, str(ROOT / "scripts/formal_provenance.py"), "--output", str(build_manifest)],
             log_path=build_log, fresh_log=True, env=env)
        env["ZKDL_BUILD_MANIFEST"] = str(build_manifest)
        env["ZKDL_RUNNER_BINARY"] = str(session / "build" / "gat_runner")
        _run([sys.executable, str(ROOT / "scripts/formal_provenance.py"), "--check", str(build_manifest)],
             log_path=admission_log, also_log_paths=tuple(dataset_logs.values()), env=env)
    except SessionRootError:
        raise
    except Exception as exc:
        shared_error = f"shared build/provenance failure: {type(exc).__name__}: {exc}"
        for dataset in DATASETS:
            if outcomes[dataset]["status"] == "PASS":
                outcomes[dataset] = _dataset_status(dataset, "BLOCKED", "build", shared_error)
            _append_log(dataset_logs[dataset], json.dumps({"stage": "build", "status": "BLOCKED",
                                                           "error": shared_error}, separators=(",", ":")))
        _write_prepare_outcome(session, session_id, outcomes, report, shared_error)
        return

    native: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        if outcomes[dataset]["status"] != "PASS":
            continue
        config = _config(session, dataset)
        run_dir = session / dataset
        try:
            require_session_root(identity, session)
            shutil.copy2(build_manifest, run_dir / "build_manifest.json")
            for command in (
                [sys.executable, str(ROOT / "scripts/prepare_form_resources.py"),
                 "--config", str(config), "--full-dataset", "--admission-only"],
                [sys.executable, str(ROOT / "scripts/asset_manifest.py"), "--config", str(config),
                 "--output", str(run_dir / "asset_manifest.json"), "--admission-only"],
            ):
                _run(command, log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env)
            output = _run([str(env["ZKDL_RUNNER_BINARY"]), "admit", "--config", str(config)],
                          log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env,
                          capture_output=True, capture_mode="last_line")
            result = json.loads(output.strip())
            if (result.get("schema") != "zkdl_native_relation_admission_v2" or
                    result.get("status") != "PASS" or result.get("dataset") != dataset):
                raise PhaseError(f"native production relation admission rejected {dataset}")
            native[dataset] = result
        except SessionRootError:
            raise
        except Exception as exc:
            outcomes[dataset] = _dataset_status(dataset, "FAILED", "resource_admission",
                                                f"{type(exc).__name__}: {exc}")
            _append_log(dataset_logs[dataset], json.dumps({"stage": "resource_admission", "status": "FAILED",
                                                           "error": outcomes[dataset]["failure"]}, separators=(",", ":")))

    profiles: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for dataset in DATASETS:
        if dataset not in native:
            continue
        try:
            require_session_root(identity, session)
            one_report = {**report, "datasets": {dataset: report["datasets"][dataset]}}
            one_profile, one_diagnostics = _admission_profile(session, session_id, one_report,
                                                               {dataset: native[dataset]})
            profiles[dataset] = one_profile
            diagnostics[dataset] = one_diagnostics[dataset]
        except SessionRootError:
            raise
        except Exception as exc:
            outcomes[dataset] = _dataset_status(dataset, "FAILED", "capacity_profile",
                                                f"{type(exc).__name__}: {exc}")
            native.pop(dataset, None)
            _append_log(dataset_logs[dataset], json.dumps({"stage": "capacity_profile", "status": "FAILED",
                                                           "error": outcomes[dataset]["failure"]}, separators=(",", ":")))

    if profiles:
        keys = [dataset for dataset in DATASETS if dataset in profiles]
        capacity_profile = dict(profiles[keys[0]])
        capacity_profile["datasets"] = {dataset: profiles[dataset]["datasets"][dataset] for dataset in keys}
        capacity_profile["finalization_manifest_capacity"] = {
            "schema": "zkdl_srs_manifest_finalization_heap_v1",
            "datasets": {dataset: profiles[dataset]["finalization_manifest_capacity"]["datasets"][dataset]
                         for dataset in keys}}
        capacity_profile["finalization_disk_bytes_upper_bound"] = sum(
            profiles[dataset]["finalization_disk_bytes_upper_bound"] for dataset in keys)
        capacity_profile["finalization_host_peak_bytes_upper_bound"] = max(
            profiles[dataset]["finalization_host_peak_bytes_upper_bound"] for dataset in keys)
        capacity_profile["preexisting_baseline_artifact_bytes"] = _session_artifact_bytes(session)
    else:
        capacity_profile = None
    graph_native = {dataset: native[dataset] for dataset in DATASETS if dataset in native}
    _write_prepare_outcome(session, session_id, outcomes, report, "", build_manifest, graph_native,
                           diagnostics, capacity_profile)


def _read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _cell_stage(stage: str, dataset: str, operation: Any, log_path: Path | None = None) -> Any:
    started = time.monotonic()
    start_marker = {"zkdl_cell_stage": stage, "dataset": dataset, "status": "START"}
    print(json.dumps(start_marker, separators=(",", ":")), flush=True)
    if log_path is not None:
        _append_log(log_path, json.dumps(start_marker, separators=(",", ":")))
    try:
        result = operation()
    except SessionRootError:
        raise
    except BaseException as exc:
        marker = {"zkdl_cell_stage": stage, "dataset": dataset, "status": "FAIL",
                  "elapsed_seconds": round(time.monotonic() - started, 6), "error": str(exc)[:512]}
        print(json.dumps(marker, separators=(",", ":")), flush=True)
        if log_path is not None:
            _append_log(log_path, json.dumps(marker, separators=(",", ":")))
        raise
    marker = {"zkdl_cell_stage": stage, "dataset": dataset, "status": "PASS",
              "elapsed_seconds": round(time.monotonic() - started, 6)}
    print(json.dumps(marker, separators=(",", ":")), flush=True)
    if log_path is not None:
        _append_log(log_path, json.dumps(marker, separators=(",", ":")))
    return result


def cell(session: Path, session_id: str, dataset: str) -> None:
    identity = require_bound_session(session, ROOT)
    if dataset not in DATASETS:
        raise PhaseError(f"cell dataset is not canonical: {dataset}")
    receipt_path = session / "phase" / "prepare.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PhaseError("cell phase requires a successful prepare receipt")
    prepared = json.loads(receipt_path.read_text(encoding="utf-8"))
    outcome = prepared.get("dataset_status", {}).get(dataset)
    native_admissions = prepared.get("native_admissions")
    capacity_profile = prepared.get("capacity_profile")
    if (prepared.get("status") not in ("PASS", "PARTIAL") or prepared.get("session_id") != session_id or
            not isinstance(outcome, dict) or outcome.get("status") != "PASS" or
            not isinstance(native_admissions, dict) or dataset not in native_admissions or
            not isinstance(capacity_profile, dict) or dataset not in capacity_profile.get("datasets", {})):
        raise PhaseError("cell phase prepare receipt identity/status mismatch")
    build_manifest = _build_manifest(session)
    env = dict(os.environ)
    env["ZKDL_BUILD_MANIFEST"] = str(build_manifest)
    env["ZKDL_RUNNER_BINARY"] = str(session / "build" / "gat_runner")
    config = _config(session, dataset)
    run_dir = session / dataset
    log_path = session / "logs" / f"{dataset}.log"
    runner = str(session / "build" / "gat_runner")
    def setup() -> None:
        _run([sys.executable, str(ROOT / "scripts/formal_provenance.py"), "--check", str(build_manifest)],
             log_path=log_path, env=env)
        _run([str(ROOT / "scripts/setup_register.sh"), str(config)], log_path=log_path, env=env)
    def cleanup() -> None:
        require_session_root(identity, session)
        for private_path in (run_dir / ".private_native", run_dir / "scratch" / "sorted_lookup"):
            require_session_root(identity, session)
            if private_path.is_symlink():
                raise PhaseError(f"refusing to clean symlinked private prover scratch: {private_path}")
            if private_path.exists():
                if not private_path.is_dir():
                    raise PhaseError(f"private prover scratch path is not a directory: {private_path}")
                shutil.rmtree(private_path)
        scratch_parent = run_dir / "scratch"
        if scratch_parent.is_dir() and not any(scratch_parent.iterdir()):
            scratch_parent.rmdir()
    try:
        _cell_stage("setup", dataset, setup, log_path)
        _cell_stage("prepare", dataset, lambda: _run(["/bin/bash", str(ROOT / "scripts/prepare_local.sh"), str(config), str(run_dir)],
                                                       log_path=log_path, env=env), log_path)
        _cell_stage("prove", dataset, lambda: _run([runner, "prove", "--config", str(config), "--backend", "cpu", "--run-dir", str(run_dir)],
                                                     log_path=log_path, env=env), log_path)
        _cell_stage("cleanup", dataset, cleanup, log_path)
        _cell_stage("verify", dataset, lambda: _run([str(ROOT / "scripts/verify_single.sh"), str(config), str(run_dir)],
                                                      log_path=log_path, env=env), log_path)
    except SessionRootError:
        raise
    except BaseException:
        try:
            cleanup()
        except Exception:
            pass
        raise

    sys.path.insert(0, str(ROOT / "scripts"))
    from result_artifacts import audit_run
    audited = audit_run(ROOT, run_dir, expected_dataset=dataset, expected_backend="cpu")
    proof_path = run_dir / "proof.bin"
    verification_path = run_dir / "verification.result"
    if (not proof_path.is_file() or not verification_path.is_file() or
            _read_kv(verification_path).get("status") != "VERIFIED" or
            audited.get("independent_verification") != "1"):
        raise PhaseError(f"single-cell proof audit/independent verification failed: {dataset}")
    proof_digest = audited["proof_digest"]
    verify_digest = sha256_file(verification_path)
    cell_receipt = {
        "schema": "zkDL_formal_cell_receipt_v1",
        "baseline": "zkdl",
        "session_id": session_id,
        "dataset": dataset,
        "status": "PASS",
        "independent_verification": True,
        "statement_id": audited["statement_id"],
        "proof_id": audited["proof_id"],
        "proof_path": proof_path.relative_to(ROOT).as_posix(),
        "proof_sha256": proof_digest,
        "proof_size_bytes": proof_path.stat().st_size,
        "verification_path": verification_path.relative_to(ROOT).as_posix(),
        "verification_sha256": verify_digest,
        "commitment_time_seconds": float(audited["commitment_time_seconds"]),
        "prove_time_seconds": float(audited["prove_time_seconds"]),
        "verify_time_seconds": float(audited["verify_time_seconds"]),
        "metrics": {
            "commitment_time_seconds": float(audited["commitment_time_seconds"]),
            "prove_time_seconds": float(audited["prove_time_seconds"]),
            "verify_time_seconds": float(audited["verify_time_seconds"]),
            "proof_size_bytes": proof_path.stat().st_size,
        },
        "statement_sha256": audited["statement_digest"],
        "run_record_sha256": sha256_file(run_dir / "run.record"),
        "run_log_path": log_path.relative_to(ROOT).as_posix(),
        "run_log_sha256": sha256_file(log_path),
    }
    cell_receipt_path = session / "phase" / f"{dataset}.json"
    require_session_root(identity, session)
    cell_receipt_path.write_text(json.dumps(cell_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "zkdl_phase": "cell",
        "status": "PASS",
        "dataset": dataset,
        "receipt_path": cell_receipt_path.relative_to(ROOT).as_posix(),
        "receipt_sha256": sha256_file(cell_receipt_path),
        "proof_path": cell_receipt["proof_path"],
        "proof_sha256": proof_digest,
        "proof_size_bytes": proof_path.stat().st_size,
        "verification_path": cell_receipt["verification_path"],
        "verification_sha256": verify_digest,
        "independent_verification": True,
        "commitment_time_seconds": cell_receipt["commitment_time_seconds"],
        "prove_time_seconds": cell_receipt["prove_time_seconds"],
        "verify_time_seconds": cell_receipt["verify_time_seconds"],
        "run_log_path": cell_receipt["run_log_path"],
        "run_log_sha256": cell_receipt["run_log_sha256"],
    }
    print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)


def _partial_result_summary(session: Path, session_id: str, state: dict[str, Any],
                            global_failure: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = require_bound_session(session, ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    from result_artifacts import ArtifactError, audit_run
    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        require_session_root(identity, session)
        state_row = state.get("dataset_status", {}).get(dataset, {})
        cell_receipt_path = session / "phase" / f"{dataset}.json"
        if state_row.get("status") == "PASS" and cell_receipt_path.is_file() and not cell_receipt_path.is_symlink():
            try:
                cell_receipt = json.loads(cell_receipt_path.read_text(encoding="utf-8"))
                if (cell_receipt.get("schema") != "zkDL_formal_cell_receipt_v1" or
                        cell_receipt.get("session_id") != session_id or cell_receipt.get("dataset") != dataset or
                        cell_receipt.get("status") != "PASS" or cell_receipt.get("independent_verification") is not True):
                    raise PhaseError(f"cell receipt identity/status mismatch for {dataset}")
                audited = audit_run(ROOT, session / dataset, expected_dataset=dataset, expected_backend="cpu")
                if (audited.get("proof_digest") != cell_receipt.get("proof_sha256") or
                        audited.get("independent_verification") != "1" or
                        sha256_file(session / dataset / "verification.result") != cell_receipt.get("verification_sha256")):
                    raise PhaseError(f"cell receipt does not match the audited proof/verification for {dataset}")
                dataset_rows.append({"dataset": dataset, "status": "PASS", "stage": "independently_verified",
                                     "result": audited, "cell_receipt_path": cell_receipt_path.relative_to(ROOT).as_posix(),
                                     "cell_receipt_sha256": sha256_file(cell_receipt_path)})
                continue
            except (ArtifactError, OSError, ValueError, TypeError, json.JSONDecodeError, PhaseError) as exc:
                failure = f"final result audit failed: {type(exc).__name__}: {exc}"
                row = {"dataset": dataset, "status": "FAILED", "stage": "final_collection", "failure": failure}
                dataset_rows.append(row)
                state_row.update(row)
                state["dataset_status"][dataset] = state_row
                failure_path = session / "phase" / f"{dataset}_failure.json"
                failure_path.write_text(json.dumps({"schema": "zkdl_formal_dataset_failure_v1", "baseline": "zkdl",
                                                    "session_id": session_id, **row}, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
                continue
        failure_path = session / "phase" / f"{dataset}_failure.json"
        failure_doc: dict[str, Any] = {}
        if failure_path.is_file() and not failure_path.is_symlink():
            try:
                candidate = json.loads(failure_path.read_text(encoding="utf-8"))
                if (candidate.get("schema") == "zkdl_formal_dataset_failure_v1" and
                        candidate.get("session_id") == session_id and candidate.get("dataset") == dataset):
                    failure_doc = candidate
            except (OSError, json.JSONDecodeError):
                pass
        status = state_row.get("status", "FAILED")
        failure = failure_doc.get("failure") or state_row.get("failure") or global_failure or "dataset did not complete a verified proof cell"
        dataset_rows.append({"dataset": dataset, "status": status if status in ("FAILED", "BLOCKED", "INTERRUPTED") else "FAILED",
                             "stage": failure_doc.get("stage") or state_row.get("stage", "unknown"),
                             "failure": failure})
    summary = {
        "schema": "zkDL_gat_formal_partial_result_summary_v1", "mode": "formal_partial",
        "baseline": "zkdl", "session_id": session_id, "runs_path": session.relative_to(ROOT).as_posix(),
        "overall_status": "FAILED", "strict_pass": False,
        "failure": global_failure[:2048], "datasets": dataset_rows,
    }
    return summary, dataset_rows


def finalize(session: Path, session_id: str) -> None:
    identity = require_bound_session(session, ROOT)
    state_path = session / "state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise PhaseError("finalization requires the local formal-runner state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state.get("schema") != "zkdl_formal_runner_state_v1" or state.get("session_id") != session_id or
            state.get("baseline") != "zkdl"):
        raise PhaseError("finalization session state identity mismatch")
    output_dir = ROOT / "results" / session.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PhaseError(f"final result directory is not empty for this formal session: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    require_session_root(identity, session)
    collection = output_dir / "collected.json"
    receipt_path = output_dir / "final_validation.json"
    all_success = (not state.get("global_failure") and
                   all(state.get("dataset_status", {}).get(dataset, {}).get("status") == "PASS"
                       for dataset in DATASETS))
    global_failure = ""
    if all_success:
        try:
            _run([sys.executable, str(ROOT / "scripts/collect_results.py"), "--runs", str(session),
                  "--output", str(collection)])
            validation_output = _run([sys.executable, str(ROOT / "scripts/final_validate.py"), "--runs", str(session),
                                      "--collection", str(collection.relative_to(ROOT))], capture_output=True,
                                    capture_mode="last_line")
            if "final_validation=PASS records=5 datasets=5 independent_verify=5" not in validation_output:
                raise PhaseError("final validator exited successfully without its exact five-cell success marker")
            receipt = {
                "schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
                "session_id": session_id, "status": "PASS", "validator": "scripts/final_validate.py",
                "validator_exit_code": 0, "validator_output": validation_output.strip().splitlines(),
                "collection_path": collection.relative_to(ROOT).as_posix(),
                "collection_sha256": sha256_file(collection), "verified_cells": list(DATASETS),
            }
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            marker = {
                "zkdl_phase": "finalize", "status": "PASS", "session_id": session_id,
                "final_validation_path": receipt_path.relative_to(ROOT).as_posix(),
                "final_validation_sha256": sha256_file(receipt_path),
                "collection_path": collection.relative_to(ROOT).as_posix(),
                "collection_sha256": sha256_file(collection),
            }
            print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)
            return
        except SessionRootError:
            raise
        except Exception as exc:
            global_failure = f"strict final collection/validation failed: {type(exc).__name__}: {exc}"
    else:
        global_failure = "one or more datasets did not complete a proof with independent verification"
    summary, dataset_rows = _partial_result_summary(session, session_id, state, global_failure)
    collection.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
        "session_id": session_id, "status": "FAILED", "strict_pass": False,
        "failure": global_failure[:2048], "collection_path": collection.relative_to(ROOT).as_posix(),
        "collection_sha256": sha256_file(collection), "verified_cells": [
            row["dataset"] for row in dataset_rows if row.get("status") == "PASS"],
        "failed_cells": [row for row in dataset_rows if row.get("status") != "PASS"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "zkdl_phase": "finalize", "status": "FAILED", "session_id": session_id,
        "failure": global_failure[:2048], "final_validation_path": receipt_path.relative_to(ROOT).as_posix(),
        "final_validation_sha256": sha256_file(receipt_path),
        "collection_path": collection.relative_to(ROOT).as_posix(),
        "collection_sha256": sha256_file(collection),
        "dataset_status": {row["dataset"]: row for row in dataset_rows},
    }
    print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-phase", choices=("prepare", "cell", "finalize"))
    parser.add_argument("--session-dir", type=str)
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _signal_handler)
    if args.internal_phase is None or args.session_dir is None:
        parser.error("internal phases require --internal-phase and --session-dir")
    if (args.internal_phase == "cell") != (args.dataset is not None):
        parser.error("--dataset is required only for cell phase")
    session, session_id = _session_path(args.session_dir)
    _validate_local_owner(session, session_id, args.internal_phase, args.dataset)
    if args.internal_phase == "prepare":
        prepare(session, session_id)
    elif args.internal_phase == "cell":
        cell(session, session_id, args.dataset)
    else:
        finalize(session, session_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PhaseError as exc:
        print(f"zkdl-formal-phase=FAILED error={exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt as exc:
        print(f"zkdl-formal-phase=INTERRUPTED error={exc}", file=sys.stderr, flush=True)
        raise SystemExit(130 if INTERRUPTED == signal.SIGINT else 143)
