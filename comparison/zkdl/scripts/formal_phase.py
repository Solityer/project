#!/usr/bin/env python3
"""Internal phase adapter for the baseline-local zkDL formal runner."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
GPU_OFFLOAD_STAGE = "hyrax_ipa_fold"
RESUME_VERIFY_ENV = "ZKDL_RESUME_VERIFY"
SESSION_RE = re.compile(r"formal_([0-9a-f]{32})\Z")
ACTIVE_PGID: int | None = None
INTERRUPTED: int | None = None
ACTIVE_SESSION_STATE: Path | None = None
ACTIVE_SESSION_ID: str | None = None
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
        if path.is_symlink() or not path.is_file():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(state, dict) or state.get("schema") != "zkdl_formal_runner_state_v1" or
                state.get("session_id") != session_id or state.get("status") != "RUNNING"):
            return
        state["interrupted_stage"] = state.get("stage", "unknown")
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
        # Never let signal cleanup mask process teardown.
        return


def _run(command: list[str], *, log_path: Path | None = None, fresh_log: bool = False,
         env: dict[str, str] | None = None, also_log_paths: tuple[Path, ...] = (), capture_output: bool = False,
         capture_mode: str = "all") -> str:
    """Run one owned process group; preserve its exit status and drain logs."""
    global ACTIVE_PGID
    log_streams = []
    for index, path in enumerate(((log_path,) if log_path is not None else ()) + also_log_paths):
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


def _config(session: Path, dataset: str) -> Path:
    return ROOT / "configs" / f"{session.name}_{dataset}.cfg"


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
                       native: dict[str, dict[str, Any]], gpu_plan: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    # Sizes and typed source payloads come from the production native
    # RelationPlan/OperationDescriptor model.  The native producer has already
    # checked every nested body against the exact consumer's parser limit.
    datasets: dict[str, dict[str, Any]] = {}
    native_capacity_diagnostics: dict[str, Any] = {}
    finalization_manifest_capacity: dict[str, dict[str, int]] = {}
    hybrid = gpu_plan is not None
    gpu_device: dict[str, Any] | None = None
    mib = 1 << 20
    cell_log_reserve = 16 * mib
    finalization_bytes = 0
    for dataset in DATASETS:
        if dataset not in native:
            continue
        input_report = preflight["datasets"][dataset]
        audit = input_report["resource_prerequisites"]
        admission = native[dataset]
        gpu = gpu_plan.get(dataset) if gpu_plan is not None else None
        if hybrid and (not isinstance(gpu, dict) or gpu.get("schema") != "zkdl_analysis_only_resource_plan_v1" or
                       gpu.get("status") != "ANALYSIS_ONLY" or gpu.get("dataset") != dataset or
                       gpu.get("gpu_offload_stage") != GPU_OFFLOAD_STAGE):
            raise PhaseError(f"native GPU offload plan identity/schema mismatch for {dataset}")
        srs_bytes = int(audit["srs_plan"]["estimated_materialized_bytes"])
        srs_plan = audit["srs_plan"]
        basis_domain = srs_plan.get("basis_domain")
        frame_bytes = srs_plan.get("frame_bytes")
        if (type(basis_domain) is not int or basis_domain != (1 << 18) or
                type(frame_bytes) is not int or frame_bytes != 40 + basis_domain * 48 or
                srs_bytes < frame_bytes + 2 * (40 + 48)):
            raise PhaseError(f"resource audit has an invalid source-bound SRS setup plan for {dataset}")
        if hybrid:
            gpu_basis = gpu.get("gpu_authenticated_basis_size")
            gpu_padded = gpu.get("gpu_max_opening_padded_generator_size")
            gpu_workspace = gpu.get("gpu_workspace_required_bytes")
            histogram = admission.get("native_hyrax_opening_segment_domain_histogram")
            valid_histogram = (
                isinstance(histogram, list) and bool(histogram) and
                all(isinstance(bucket, dict) and
                    type(bucket.get("segment_size")) is int and 0 < bucket["segment_size"] <= (1 << 64) - 1 and
                    type(bucket.get("padded_generator_size")) is int and
                    0 < bucket["padded_generator_size"] <= (1 << 64) - 1 and
                    bucket["padded_generator_size"] == (1 << bucket["segment_size"].bit_length())
                    for bucket in histogram)
            )
            max_segment_generators = (max(bucket["segment_size"] for bucket in histogram)
                                      if valid_histogram else 0)
            max_histogram_padded = (max(bucket["padded_generator_size"] for bucket in histogram)
                                    if valid_histogram else 0)
            # IPA rounds pad to the next power of two strictly above the real
            # generator count. Identity padding can therefore make P exceed
            # the authenticated real basis B; only each real segment S must
            # fit B, while P must match the admission histogram exactly.
            max_basis_padded = 1 << gpu_basis.bit_length() if type(gpu_basis) is int and gpu_basis > 0 else 0
            if (type(gpu_basis) is not int or gpu_basis != basis_domain or type(gpu_padded) is not int or
                    gpu_padded < 1 or gpu_padded > max_basis_padded or gpu_padded & (gpu_padded - 1) or
                    not valid_histogram or max_segment_generators > gpu_basis or
                    gpu_padded != max_histogram_padded or
                    type(gpu_workspace) is not int or gpu_workspace != 144 * gpu_basis + 704 * gpu_padded + 352 or
                    gpu_workspace > (1 << 64) - 1):
                raise PhaseError(f"native GPU workspace is not bound to the authenticated maximum opening for {dataset}")
            gpu_device_name = gpu.get("gpu_device_name")
            gpu_device_ordinal = gpu.get("gpu_device_ordinal")
            gpu_device_uuid = gpu.get("gpu_device_uuid")
            gpu_free = gpu.get("gpu_cuda_free_bytes")
            gpu_total = gpu.get("gpu_cuda_total_bytes")
            if (not isinstance(gpu_device_name, str) or not gpu_device_name.strip() or
                    type(gpu_device_ordinal) is not int or gpu_device_ordinal < 0 or
                    not isinstance(gpu_device_uuid, str) or not gpu_device_uuid.strip() or
                    type(gpu_free) is not int or type(gpu_total) is not int or
                    gpu_free < gpu_workspace or gpu_total < gpu_free or gpu_total <= 0):
                raise PhaseError(f"native GPU capacity/device identity is invalid for {dataset}")
            gpu_device = {"name": gpu_device_name, "ordinal": gpu_device_ordinal, "uuid": gpu_device_uuid}
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
        if spill_fallback_count != 0:
            raise PhaseError(f"native Sumcheck still reports an unbounded in-memory fallback for {dataset}")
        if spill_ram_peak > int(admission["native_segmented_sumcheck_live_peak_bytes_upper_bound"]):
            raise PhaseError(f"native Sumcheck phase RAM bound omits its spill working set for {dataset}")
        nodes = int(admission["nodes"])
        edges = int(admission["edges"])
        graphs = int(admission["graphs"])
        # The resource-prerequisite record is the authoritative source for
        # input shape. The per-dataset preflight row has no
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
        asset_file_count = family_count + operation_count + lookup_count + 128
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
            "device_peak_bytes": gpu_workspace if hybrid else 0,
            **({"gpu_authenticated_basis_size": gpu_basis, "gpu_max_opening_padded_generator_size": gpu_padded,
                "gpu_workspace_required_bytes": gpu_workspace, "gpu_cuda_free_bytes_at_prepare": gpu_free,
                "gpu_cuda_total_bytes_at_prepare": gpu_total} if hybrid else {}),
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
    device = os.stat(session, follow_symlinks=False).st_dev
    finalization_host_peak = max(item["host_peak_bytes_upper_bound"] for item in finalization_manifest_capacity.values())
    shared_host_cuda_method = (
        "shared_uma_host_available_ge_host_plus_gpu_workspace_cuda_free_ge_workspace"
        if hybrid and gpu_device is not None and gpu_device.get("name") == "NVIDIA GB10" else
        "shared_pool_min_host_cuda_ge_host_plus_gpu_workspace")
    profile = {
        "schema": "zkdl_formal_capacity_profile_v4" if hybrid else "zkdl_formal_capacity_profile_v3",
        "baseline": "zkdl",
        "session_id": session_id,
        "fs_device_id": int(device),
        "finalization_disk_bytes_upper_bound": finalization_bytes,
        "finalization_host_peak_bytes_upper_bound": finalization_host_peak,
        "finalization_manifest_capacity": {"schema": "zkdl_srs_manifest_finalization_heap_v1",
                                            "datasets": finalization_manifest_capacity},
        "backend": "cpu",
        **({"gpu_offload_stage": GPU_OFFLOAD_STAGE} if hybrid else {}),
        "memory_accounting_kind": "shared_host_cuda" if hybrid else "host-only",
        "device": gpu_device,
        "bound_methods": {
            "additional_disk_peak": "source_derived_upper_bound",
            "host_peak": "source_derived_upper_bound",
            **({"gpu_workspace": "authenticated_basis_144B_plus_max_opening_704P_plus_352",
                "shared_host_cuda": shared_host_cuda_method} if hybrid else {}),
            "finalization_host_peak": "srs_manifest_schema_python_heap_upper_bound",
        },
        "datasets": datasets,
    }
    return profile, native_capacity_diagnostics


def _dataset_status(dataset: str, status: str, stage: str, failure: str = "") -> dict[str, str]:
    return {"dataset": dataset, "status": status, "stage": stage, "failure": failure[:2048]}


def _append_log(path: Path, text: str) -> None:
    if path.is_symlink():
        raise PhaseError(f"refusing to append through a symlinked formal log: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PhaseError(f"formal log directory is missing or unsafe: {path.parent}")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.rstrip("\n") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _remove_failed_snapshot(session: Path, dataset: str) -> None:
    paths = (session / "resources" / "cache" / dataset,
             session / "resources" / "checkpoints" / f"{dataset}_gat",
             session / dataset,
             ROOT / "configs" / f"{session.name}_{dataset}.cfg")
    for path in paths:
        if path.is_symlink():
            raise PhaseError(f"refusing to remove symlinked partial snapshot path: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    for path in (session / "resources" / "cache", session / "resources" / "checkpoints",
                 session / "resources", session / "data"):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _write_prepare_outcome(session: Path, session_id: str, outcomes: dict[str, dict[str, str]],
                           preflight: dict[str, Any] | None, error: str,
                           native: dict[str, dict[str, Any]] | None = None,
                           diagnostics: dict[str, Any] | None = None,
                           profile: dict[str, Any] | None = None) -> dict[str, Any]:
    passed = sum(row["status"] == "PASS" for row in outcomes.values())
    overall = "PASS" if passed == len(DATASETS) else "PARTIAL" if passed else "FAILED"
    receipt: dict[str, Any] = {
        "schema": "zkDL_formal_prepare_receipt_v3", "baseline": "zkdl", "session_id": session_id,
        "status": overall, "failure": error[:2048], "dataset_status": outcomes,
        "preflight_report": preflight, "native_admissions": native or {},
        "native_capacity_diagnostics": diagnostics or {}, "capacity_profile": profile,
    }
    receipt_path = session / "phase" / "prepare.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker: dict[str, Any] = {
        "zkdl_phase": "prepare", "status": overall, "session_id": session_id,
        "session_dir": session.relative_to(ROOT).as_posix(),
        "prepare_receipt_path": receipt_path.relative_to(ROOT).as_posix(),
        "dataset_status": outcomes, "failure": error[:2048],
    }
    if profile is not None:
        marker["capacity_profile"] = profile
    print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)
    return marker

def prepare(session: Path, session_id: str) -> None:
    if session.is_symlink() or not session.is_dir():
        raise PhaseError(f"zkDL formal session is missing: {session}")
    state = json.loads((session / "state.json").read_text(encoding="utf-8"))
    if (not isinstance(state, dict) or state.get("schema") != "zkdl_formal_runner_state_v1" or
            state.get("baseline") != "zkdl" or state.get("session_id") != session_id or
            state.get("status") != "RUNNING" or state.get("stage") != "prepare"):
        raise PhaseError("zkDL formal session state is not ready for preparation")
    (session / "phase").mkdir()
    admission_log = session / "logs" / "admission.log"
    report_path = session / "dataset_prerequisites.json"
    env = dict(os.environ)
    env["ZKDL_PREFLIGHT_REPORT"] = str(report_path)
    _run([sys.executable, str(ROOT / "scripts/formal_preflight.py")],
         log_path=admission_log, fresh_log=True, env=env)
    report_doc = json.loads(report_path.read_text(encoding="utf-8"))
    if report_doc.get("schema") != "zkDL_formal_input_preflight_v1" or set(report_doc.get("datasets", {})) != set(DATASETS):
        raise PhaseError("formal resource plan does not contain the five datasets")
    report = report_doc
    outcomes = {
        dataset: _dataset_status(dataset,
                                 "PASS" if report["datasets"][dataset].get("status") == "PASS" else "FAILED",
                                 "resource_plan" if report["datasets"][dataset].get("status") != "PASS" else "pending",
                                 str(report["datasets"][dataset].get("failure", "")))
        for dataset in DATASETS
    }
    dataset_logs = {dataset: session / "logs" / f"{dataset}.log" for dataset in DATASETS}
    for dataset in DATASETS:
        _append_log(dataset_logs[dataset], json.dumps({"stage": "resource_plan", **report["datasets"][dataset]},
                                                       sort_keys=True, separators=(",", ":")))
        if outcomes[dataset]["status"] != "PASS":
            continue
        try:
            _run([sys.executable, str(ROOT / "scripts/formal_session.py"), "--session", str(session),
                  "--dataset", dataset], log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env)
        except Exception as exc:
            _remove_failed_snapshot(session, dataset)
            outcomes[dataset] = _dataset_status(dataset, "FAILED", "config", f"{type(exc).__name__}: {exc}")
            _append_log(dataset_logs[dataset], json.dumps({"stage": "config", "status": "FAILED",
                                                           "error": outcomes[dataset]["failure"]}, separators=(",", ":")))

    if not any(row["status"] == "PASS" for row in outcomes.values()):
        _write_prepare_outcome(session, session_id, outcomes, report, "no dataset passed resource planning")
        return

    native: dict[str, dict[str, Any]] = {}
    env["ZKDL_RUNNER_BINARY"] = str(ROOT / "gat_runner")
    env["ZKDL_RESOURCE_SETUP_BINARY"] = str(ROOT / "gat_resource_setup")
    for dataset in DATASETS:
        if outcomes[dataset]["status"] != "PASS":
            continue
        config = _config(session, dataset)
        try:
            _run(["/bin/bash", str(ROOT / "scripts/setup_register.sh"), str(config)],
                 log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env)
            output = _run([env["ZKDL_RUNNER_BINARY"], "admit", "--config", str(config)],
                          log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env,
                          capture_output=True, capture_mode="last_line")
            result = json.loads(output.strip())
            if (result.get("schema") != "zkdl_native_relation_admission_v2" or
                    result.get("status") != "PASS" or result.get("dataset") != dataset):
                raise PhaseError(f"native production relation admission rejected {dataset}")
            native[dataset] = result
        except Exception as exc:
            outcomes[dataset] = _dataset_status(dataset, "FAILED", "resource_setup",
                                                f"{type(exc).__name__}: {exc}")
            _append_log(dataset_logs[dataset], json.dumps({"stage": "resource_setup", "status": "FAILED",
                                                           "error": outcomes[dataset]["failure"]}, separators=(",", ":")))

    profiles: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for dataset in DATASETS:
        if dataset not in native:
            continue
        try:
            config = _config(session, dataset)
            output = _run([str(ROOT / "gat_runner"), "resource-plan", "--config", str(config),
                           "--gpu-offload-stage", GPU_OFFLOAD_STAGE],
                          log_path=admission_log, also_log_paths=(dataset_logs[dataset],), env=env,
                          capture_output=True, capture_mode="last_line")
            gpu_plan = json.loads(output.strip())
            one_report = {**report, "datasets": {dataset: report["datasets"][dataset]}}
            one_profile, one_diagnostics = _admission_profile(session, session_id, one_report,
                                                               {dataset: native[dataset]}, {dataset: gpu_plan})
            profiles[dataset] = one_profile
            diagnostics[dataset] = one_diagnostics[dataset]
        except Exception as exc:
            outcomes[dataset] = _dataset_status(dataset, "FAILED", "capacity_profile",
                                                f"{type(exc).__name__}: {exc}")
            native.pop(dataset, None)
            _append_log(dataset_logs[dataset], json.dumps({"stage": "capacity_profile", "status": "FAILED",
                                                           "error": outcomes[dataset]["failure"]}, separators=(",", ":")))

    if profiles:
        keys = [dataset for dataset in DATASETS if dataset in profiles]
        capacity_profile = dict(profiles[keys[0]])
        if any(profiles[dataset].get("gpu_offload_stage") != capacity_profile.get("gpu_offload_stage") or
               profiles[dataset].get("device") != capacity_profile.get("device") for dataset in keys[1:]):
            for dataset in keys:
                outcomes[dataset] = _dataset_status(dataset, "BLOCKED", "capacity_profile",
                                                    "GPU offload device changed during preparation")
            capacity_profile = None
            profiles.clear()
        if capacity_profile is None:
            _write_prepare_outcome(session, session_id, outcomes, report,
                                   "GPU offload device changed during preparation", native, diagnostics)
            return
        capacity_profile["datasets"] = {dataset: profiles[dataset]["datasets"][dataset] for dataset in keys}
        capacity_profile["finalization_manifest_capacity"] = {
            "schema": "zkdl_srs_manifest_finalization_heap_v1",
            "datasets": {dataset: profiles[dataset]["finalization_manifest_capacity"]["datasets"][dataset]
                         for dataset in keys}}
        capacity_profile["finalization_disk_bytes_upper_bound"] = sum(
            profiles[dataset]["finalization_disk_bytes_upper_bound"] for dataset in keys)
        capacity_profile["finalization_host_peak_bytes_upper_bound"] = max(
            profiles[dataset]["finalization_host_peak_bytes_upper_bound"] for dataset in keys)
    else:
        capacity_profile = None
    _write_prepare_outcome(session, session_id, outcomes, report, "", native, diagnostics, capacity_profile)

def _read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _read_resume_kv(path: Path, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise PhaseError(f"cannot reuse proved artifact: {label} is missing or not a regular file")
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "=" not in line:
            raise PhaseError(f"cannot reuse proved artifact: malformed {label} line {number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise PhaseError(f"cannot reuse proved artifact: duplicate or empty {label} field {key!r}")
        values[key] = value
    return values


def _validate_reusable_proof(run_dir: Path, dataset: str) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise PhaseError("cannot reuse proved artifact: dataset run directory is missing or not local")
    statement = _read_resume_kv(run_dir / "statement.txt", "public statement")
    envelope = _read_resume_kv(run_dir / "proof.meta", "proof envelope")
    timing = _read_resume_kv(run_dir / "proof.timing", "proof timing")
    statement_id = statement.get("statement_id", "")
    if (statement.get("schema") != "zkDL_gat_statement_v1" or statement.get("dataset") != dataset or
            statement.get("proof_scope") != "full_gat_forward" or
            re.fullmatch(r"[0-9a-f]{64}", statement_id) is None):
        raise PhaseError("cannot reuse proved artifact: public statement identity is incomplete or mismatched")
    if (envelope.get("schema_id") != "zkDL_gat_proof_envelope_v1" or envelope.get("status") != "PROVED" or
            envelope.get("backend") != "cpu" or envelope.get("proof_scope") != "full_gat_forward" or
            envelope.get("statement_id") != statement_id):
        raise PhaseError("cannot reuse proved artifact: proof envelope is not PROVED for this statement")
    if (timing.get("schema_id") != "zkDL_gat_proof_timing_v1" or timing.get("status") != "PROVED" or
            timing.get("statement_id") != statement_id):
        raise PhaseError("cannot reuse proved artifact: proof timing is not PROVED for this statement")
    proof_path = run_dir / "proof.bin"
    if proof_path.is_symlink() or not proof_path.is_file():
        raise PhaseError("cannot reuse proved artifact: proof.bin is missing or not a regular file")
    try:
        declared_size = int(envelope.get("proof_size_bytes", "0"))
    except ValueError as exc:
        raise PhaseError("cannot reuse proved artifact: proof envelope size is invalid") from exc
    if declared_size <= 0 or proof_path.stat().st_size != declared_size:
        raise PhaseError("cannot reuse proved artifact: proof.bin size does not match the PROVED envelope")


def _cell_stage(stage: str, dataset: str, operation: Any, log_path: Path | None = None) -> Any:
    started = time.monotonic()
    start_marker = {"zkdl_cell_stage": stage, "dataset": dataset, "status": "START"}
    print(json.dumps(start_marker, separators=(",", ":")), flush=True)
    if log_path is not None:
        _append_log(log_path, json.dumps(start_marker, separators=(",", ":")))
    try:
        result = operation()
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
    if dataset not in DATASETS:
        raise PhaseError(f"cell dataset is not canonical: {dataset}")
    prepared = json.loads((session / "phase" / "prepare.json").read_text(encoding="utf-8"))
    outcome = prepared.get("dataset_status", {}).get(dataset)
    native_admissions = prepared.get("native_admissions")
    capacity_profile = prepared.get("capacity_profile")
    if (prepared.get("status") not in ("PASS", "PARTIAL") or prepared.get("session_id") != session_id or
            not isinstance(outcome, dict) or outcome.get("status") != "PASS" or
            not isinstance(native_admissions, dict) or dataset not in native_admissions or
            not isinstance(capacity_profile, dict) or dataset not in capacity_profile.get("datasets", {})):
        raise PhaseError("cell phase has no completed resource admission")
    if (capacity_profile.get("schema") != "zkdl_formal_capacity_profile_v4" or
            capacity_profile.get("gpu_offload_stage") != GPU_OFFLOAD_STAGE or
            capacity_profile.get("memory_accounting_kind") != "shared_host_cuda" or
            not isinstance(capacity_profile.get("device"), dict)):
        raise PhaseError("cell phase has no matching GPU capacity profile")
    env = dict(os.environ)
    env["ZKDL_RUNNER_BINARY"] = str(ROOT / "gat_runner")
    env["ZKDL_RESOURCE_SETUP_BINARY"] = str(ROOT / "gat_resource_setup")
    config = _config(session, dataset)
    run_dir = session / dataset
    log_path = session / "logs" / f"{dataset}.log"
    runner = str(ROOT / "gat_runner")
    resume_verify = os.environ.get(RESUME_VERIFY_ENV) or None
    if resume_verify is not None and resume_verify not in DATASETS:
        raise PhaseError(f"unsupported {RESUME_VERIFY_ENV} dataset: {resume_verify}")
    reuse_proved = resume_verify == dataset

    def cleanup() -> None:
        for private_path in (run_dir / ".private_native", run_dir / "scratch" / "sorted_lookup"):
            if private_path.exists():
                if not private_path.is_dir():
                    raise PhaseError(f"private prover scratch path is not a directory: {private_path}")
                shutil.rmtree(private_path)
        scratch_parent = run_dir / "scratch"
        if scratch_parent.is_dir() and not any(scratch_parent.iterdir()):
            scratch_parent.rmdir()

    if reuse_proved:
        # This must precede prove and cleanup: prove overwrites proof metadata, while
        # cleanup removes prover scratch that may still be needed for diagnosis.
        _cell_stage("reuse_proved_preflight", dataset,
                    lambda: _validate_reusable_proof(run_dir, dataset), log_path)

    try:
        if not reuse_proved:
            _cell_stage("prove", dataset,
                        lambda: _run([runner, "prove", "--config", str(config), "--backend", "cpu",
                                      "--gpu-offload-stage", GPU_OFFLOAD_STAGE, "--run-dir", str(run_dir)],
                                     log_path=log_path, env=env), log_path)
        _cell_stage("cleanup", dataset, cleanup, log_path)
        _cell_stage("verify", dataset,
                    lambda: _run([runner, "verify", "--config", str(config), "--run-dir", str(run_dir)],
                                 log_path=log_path, env=env), log_path)
    except BaseException:
        try:
            cleanup()
        except Exception:
            pass
        raise

    sys.path.insert(0, str(ROOT / "scripts"))
    from result_artifacts import audit_run
    audited = audit_run(ROOT, run_dir, expected_dataset=dataset, expected_backend="cpu",
                         expected_gpu_offload_stage=GPU_OFFLOAD_STAGE)
    planned_gpu = capacity_profile["datasets"][dataset]
    device = capacity_profile["device"]
    if (audited.get("gpu_device_name") != device.get("name") or
            audited.get("gpu_device_ordinal") != str(device.get("ordinal")) or
            audited.get("gpu_device_uuid") != device.get("uuid") or
            int(audited.get("gpu_proof_peak_allocated_bytes", "0")) > planned_gpu.get("gpu_workspace_required_bytes", 0)):
        raise PhaseError(f"proof GPU execution or memory exceeds the admitted capacity for {dataset}")
    proof_path = run_dir / "proof.bin"
    verification_path = run_dir / "verification.result"
    if (not proof_path.is_file() or not verification_path.is_file() or
            _read_kv(verification_path).get("status") != "VERIFIED" or
            audited.get("independent_verification") != "1"):
        raise PhaseError(f"proof or independent verification failed: {dataset}")
    proof_size = proof_path.stat().st_size
    cell_receipt = {
        "schema": "zkDL_formal_cell_receipt_v2", "baseline": "zkdl", "session_id": session_id,
        "dataset": dataset, "status": "PASS", "independent_verification": True,
        "statement_id": audited["statement_id"], "proof_path": proof_path.relative_to(ROOT).as_posix(),
        "proof_size_bytes": proof_size,
        "verification_path": verification_path.relative_to(ROOT).as_posix(),
        "commitment_time_seconds": float(audited["commitment_time_seconds"]),
        "prove_time_seconds": float(audited["prove_time_seconds"]),
        "verify_time_seconds": float(audited["verify_time_seconds"]),
        "metrics": {"commitment_time_seconds": float(audited["commitment_time_seconds"]),
                    "prove_time_seconds": float(audited["prove_time_seconds"]),
                    "verify_time_seconds": float(audited["verify_time_seconds"]),
                    "proof_size_bytes": proof_size},
        "run_log_path": log_path.relative_to(ROOT).as_posix(),
        "gpu_offload_stage": audited["gpu_offload_stage"],
        "gpu_device_name": audited["gpu_device_name"],
        "gpu_device_ordinal": audited["gpu_device_ordinal"],
        "gpu_device_uuid": audited["gpu_device_uuid"],
        "gpu_forward_kernel_executed": audited["gpu_forward_kernel_executed"],
        "gpu_proof_kernel_executed": audited["gpu_proof_kernel_executed"],
        "gpu_kernel_executed": audited["gpu_kernel_executed"],
        "gpu_proof_kernel_count": audited["gpu_proof_kernel_count"],
        "gpu_proof_peak_allocated_bytes": audited["gpu_proof_peak_allocated_bytes"],
        "verifier_backend": audited["verifier_backend"],
        "verify_gpu_kernel_executed": audited["verify_gpu_kernel_executed"],
    }
    cell_receipt_path = session / "phase" / f"{dataset}.json"
    cell_receipt_path.write_text(json.dumps(cell_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "zkdl_phase": "cell", "status": "PASS", "dataset": dataset,
        "receipt_path": cell_receipt_path.relative_to(ROOT).as_posix(),
        "proof_path": cell_receipt["proof_path"], "proof_size_bytes": proof_size,
        "verification_path": cell_receipt["verification_path"], "independent_verification": True,
        "commitment_time_seconds": cell_receipt["commitment_time_seconds"],
        "prove_time_seconds": cell_receipt["prove_time_seconds"],
        "verify_time_seconds": cell_receipt["verify_time_seconds"],
        "run_log_path": cell_receipt["run_log_path"],
        "gpu_offload_stage": cell_receipt["gpu_offload_stage"],
        "gpu_device_name": cell_receipt["gpu_device_name"],
        "gpu_device_ordinal": cell_receipt["gpu_device_ordinal"],
        "gpu_device_uuid": cell_receipt["gpu_device_uuid"],
        "gpu_forward_kernel_executed": cell_receipt["gpu_forward_kernel_executed"],
        "gpu_proof_kernel_executed": cell_receipt["gpu_proof_kernel_executed"],
        "gpu_kernel_executed": cell_receipt["gpu_kernel_executed"],
        "gpu_proof_kernel_count": cell_receipt["gpu_proof_kernel_count"],
        "gpu_proof_peak_allocated_bytes": cell_receipt["gpu_proof_peak_allocated_bytes"],
        "verifier_backend": cell_receipt["verifier_backend"],
        "verify_gpu_kernel_executed": cell_receipt["verify_gpu_kernel_executed"],
    }
    print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)

def _partial_result_summary(session: Path, session_id: str, state: dict[str, Any],
                            global_failure: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from result_artifacts import ArtifactError, audit_run
    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        state_row = state.get("dataset_status", {}).get(dataset, {})
        cell_receipt_path = session / "phase" / f"{dataset}.json"
        if state_row.get("status") == "PASS" and cell_receipt_path.is_file() and not cell_receipt_path.is_symlink():
            try:
                cell_receipt = json.loads(cell_receipt_path.read_text(encoding="utf-8"))
                if (cell_receipt.get("schema") != "zkDL_formal_cell_receipt_v2" or
                        cell_receipt.get("session_id") != session_id or cell_receipt.get("dataset") != dataset or
                        cell_receipt.get("status") != "PASS" or cell_receipt.get("independent_verification") is not True):
                    raise PhaseError(f"cell receipt identity/status mismatch for {dataset}")
                audited = audit_run(ROOT, session / dataset, expected_dataset=dataset, expected_backend="cpu",
                                    expected_gpu_offload_stage=GPU_OFFLOAD_STAGE)
                if (audited.get("independent_verification") != "1" or
                        audited.get("statement_id") != cell_receipt.get("statement_id") or
                        audited.get("proof_artifact_path") != cell_receipt.get("proof_path") or
                        any(audited.get(field) != cell_receipt.get(field) for field in (
                            "gpu_offload_stage", "gpu_device_name", "gpu_device_ordinal", "gpu_device_uuid",
                            "gpu_forward_kernel_executed", "gpu_proof_kernel_executed", "gpu_kernel_executed",
                            "gpu_proof_kernel_count", "gpu_proof_peak_allocated_bytes", "verifier_backend",
                            "verify_gpu_kernel_executed"))):
                    raise PhaseError(f"cell receipt does not match the independently verified result for {dataset}")
                dataset_rows.append({"dataset": dataset, "status": "PASS", "stage": "independently_verified",
                                     "result": audited, "cell_receipt_path": cell_receipt_path.relative_to(ROOT).as_posix()})
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
                "verified_cells": list(DATASETS),
            }
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            marker = {
                "zkdl_phase": "finalize", "status": "PASS", "session_id": session_id,
                "final_validation_path": receipt_path.relative_to(ROOT).as_posix(),
                "collection_path": collection.relative_to(ROOT).as_posix(),
            }
            print(json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)
            return
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
        "verified_cells": [
            row["dataset"] for row in dataset_rows if row.get("status") == "PASS"],
        "failed_cells": [row for row in dataset_rows if row.get("status") != "PASS"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "zkdl_phase": "finalize", "status": "FAILED", "session_id": session_id,
        "failure": global_failure[:2048], "final_validation_path": receipt_path.relative_to(ROOT).as_posix(),
        "collection_path": collection.relative_to(ROOT).as_posix(),
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
    if not session.is_dir():
        raise PhaseError(f"formal session directory is missing: {session}")
    global ACTIVE_SESSION_STATE, ACTIVE_SESSION_ID
    ACTIVE_SESSION_STATE = session / "state.json"
    ACTIVE_SESSION_ID = session_id
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
