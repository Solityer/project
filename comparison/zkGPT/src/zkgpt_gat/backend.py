from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import subprocess
import tempfile
from numbers import Integral
from typing import Any, Dict, List, Mapping, Sequence

from .field import BN254_FR_MODULUS
from .build_identity import source_inventory, source_manifest_digest, verify_build_manifest
from .util import digest_json, ensure_dir, read_json, stable_env, write_text_atomic


_MASK64 = (1 << 64) - 1
_GPU_WIRE_VERSION = "ZKGPT_GPU_BATCH_V3"
_GPU_PROTOCOL = "zkgpt/native-bn254-fr-batch-v1"
_GPU_SCHEMA = "zkGPT_gpu_proof_messages_v3"
_HOT_PATH_SEMANTICS = "zkgpt_native_lasso_matmul_gkr_bn254_transcript_primitives_v3"
_GPU_KERNEL = "zkgpt_native_bn254_fr_batch_v3"


def gpu_proof_hot_path_binary(zk_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(zk_root) / "build" / "zkgpt_cuda_native_prover"


def _lexical(path: pathlib.Path) -> pathlib.Path:
    """Normalize a path without resolving symlinks."""
    return pathlib.Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _check_lexical_components(path: pathlib.Path, root: pathlib.Path, *, allow_missing: bool) -> None:
    root = _lexical(root)
    path = _lexical(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"CUDA path escapes baseline root: {path}") from exc
    current = root
    try:
        root_info = current.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect CUDA baseline root: {root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError(f"CUDA baseline root must be a real directory: {root}")
    for component in relative.parts:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                continue
            raise RuntimeError(f"CUDA path component is missing: {current}")
        except OSError as exc:
            raise RuntimeError(f"cannot inspect CUDA path component: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"CUDA path component must not be a symlink: {current}")
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"CUDA path component must be a directory: {current}")


def _authenticated_cuda_binary(zk_root: pathlib.Path) -> tuple[pathlib.Path, str, str, Dict[str, Any]]:
    """Authenticate the baseline-local CUDA executable before invocation."""
    root = _lexical(zk_root)
    manifest = verify_build_manifest(root)
    binary = _lexical(gpu_proof_hot_path_binary(root))
    _check_lexical_components(binary, root, allow_missing=False)
    try:
        info = binary.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect CUDA proof primitive binary: {binary}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"CUDA proof primitive binary must be a regular file: {binary}")
    if info.st_nlink != 1:
        raise RuntimeError(f"CUDA proof primitive binary must have one hardlink: {binary}")
    if not info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        raise RuntimeError(f"CUDA proof primitive binary is not executable: {binary}")
    try:
        if binary.resolve(strict=True) != binary:
            raise RuntimeError(f"CUDA proof primitive binary resolves through an alias: {binary}")
    except OSError as exc:
        raise RuntimeError(f"cannot resolve CUDA proof primitive binary: {binary}") from exc
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if manifest.get("binary", {}).get("sha256") != digest:
        raise RuntimeError("CUDA proof primitive binary digest differs from build manifest")
    identity = str(manifest.get("build_identity", ""))
    if not identity:
        raise RuntimeError("CUDA build manifest omitted build identity")
    return binary, digest, identity, manifest


def _as_field(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer BN254-Fr scalar")
    return int(value) % BN254_FR_MODULUS


def _field_limbs(value: int) -> tuple[int, int, int, int]:
    scalar = _as_field(value, "field value")
    return tuple((scalar >> (64 * limb)) & _MASK64 for limb in range(4))


def _field_hex(value: int) -> str:
    return f"{_as_field(value, 'field value'):064x}"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _log2_power_of_two(value: int) -> int:
    if not _is_power_of_two(value):
        raise ValueError("field-vector length must be a power of two")
    return value.bit_length() - 1


def _field_vector(value: Any, label: str, *, allow_empty: bool = False) -> List[int]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of field scalars")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence of field scalars") from exc
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return [_as_field(item, f"{label}[{index}]") for index, item in enumerate(values)]


def _canonical_operations(operations: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(operations, (str, bytes, bytearray)):
        raise TypeError("GPU operation batch must be a sequence of operation mappings")
    try:
        entries = list(operations)
    except TypeError as exc:
        raise TypeError("GPU operation batch must be a sequence of operation mappings") from exc
    if not entries:
        raise ValueError("GPU operation batch must not be empty")
    canonical: List[Dict[str, Any]] = []
    aliases = {
        "lasso": "lasso_product_tree",
        "lasso_product_tree": "lasso_product_tree",
        "sumcheck": "sumcheck_mle_fold",
        "sumcheck_mle": "sumcheck_mle_fold",
        "sumcheck_mle_fold": "sumcheck_mle_fold",
        "sumcheck_product2": "sumcheck_product2_fold",
        "sumcheck_product2_fold": "sumcheck_product2_fold",
        "sumcheck_product3": "sumcheck_product3_fold",
        "sumcheck_product3_fold": "sumcheck_product3_fold",
        "hyrax": "hyrax_linear_eval_fold",
        "hyrax_linear_eval": "hyrax_linear_eval_fold",
        "hyrax_linear_eval_fold": "hyrax_linear_eval_fold",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"GPU operation {index} must be a mapping")
        raw_kind = str(entry.get("kind", ""))
        kind = aliases.get(raw_kind)
        if kind is None:
            raise ValueError(f"unsupported GPU operation kind: {raw_kind!r}")
        allow_partial = bool(entry.get("_allow_prefix", entry.get("allow_partial", False)))
        if allow_partial and kind == "lasso_product_tree":
            raise ValueError("Lasso does not support a partial round prefix")
        if kind in {"sumcheck_product2_fold", "sumcheck_product3_fold"} and "values" not in entry:
            raw_operands = entry.get("operands")
            if isinstance(raw_operands, Sequence) and not isinstance(raw_operands, (str, bytes, bytearray)) and raw_operands:
                raw_values = raw_operands[0]
            else:
                raw_values = entry.get("values_a")
        else:
            raw_values = entry.get("values")
        values = _field_vector(raw_values, f"operation[{index}].values")
        item: Dict[str, Any] = {"kind": kind, "values": values, "allow_partial": allow_partial}
        if kind == "lasso_product_tree":
            if "beta" not in entry:
                raise ValueError("Lasso operation requires the transcript-derived beta")
            item["beta"] = _as_field(entry["beta"], f"operation[{index}].beta")
        elif kind == "sumcheck_mle_fold" or kind == "hyrax_linear_eval_fold":
            if "challenges" not in entry:
                raise ValueError(f"{kind} requires all transcript-derived round challenges")
            challenges = _field_vector(entry["challenges"], f"operation[{index}].challenges", allow_empty=True)
            expected = _log2_power_of_two(len(values))
            if len(challenges) > expected or (not allow_partial and len(challenges) != expected):
                raise ValueError(
                    f"{kind} requires exactly log2(input_count) challenges "
                    f"({expected}), got {len(challenges)}"
                )
            item["challenges"] = challenges
        else:
            operands = entry.get("operands")
            if operands is None:
                operands = [entry.get("values_a"), entry.get("values_b")]
                if kind == "sumcheck_product3_fold":
                    operands.append(entry.get("values_c"))
            if not isinstance(operands, Sequence) or isinstance(operands, (str, bytes, bytearray)):
                raise TypeError(f"{kind} requires an operands sequence")
            expected_operands = 2 if kind == "sumcheck_product2_fold" else 3
            if len(operands) != expected_operands:
                raise ValueError(f"{kind} requires exactly {expected_operands} operands")
            canonical_operands = []
            for operand_index, operand in enumerate(operands):
                parsed = _field_vector(operand, f"operation[{index}].operands[{operand_index}]")
                if len(parsed) != len(values):
                    raise ValueError(f"{kind} operand lengths must match values")
                canonical_operands.append(parsed)
            if canonical_operands[0] != values:
                raise ValueError(f"{kind} first operand must equal operation values")
            item["operands"] = canonical_operands
            if "challenges" not in entry:
                raise ValueError(f"{kind} requires all transcript-derived round challenges")
            challenges = _field_vector(entry["challenges"], f"operation[{index}].challenges", allow_empty=True)
            expected = _log2_power_of_two(len(values))
            if len(challenges) > expected or (not allow_partial and len(challenges) != expected):
                raise ValueError(
                    f"{kind} requires exactly log2(input_count) challenges "
                    f"({expected}), got {len(challenges)}"
                )
            item["challenges"] = challenges
        canonical.append(item)
    return canonical


def _wire_lines(operations: Sequence[Mapping[str, Any]]) -> List[str]:
    canonical = _canonical_operations(operations)
    lines = [_GPU_WIRE_VERSION]
    opcode = {
        "lasso_product_tree": "LASSO",
        "sumcheck_mle_fold": "SUMCHECK",
        "sumcheck_product2_fold": "SUMCHECK_PRODUCT2",
        "sumcheck_product3_fold": "SUMCHECK_PRODUCT3",
        "hyrax_linear_eval_fold": "HYRAX",
    }
    for operation in canonical:
        values = operation["values"]
        kind = operation["kind"]
        wire_opcode = opcode[kind] + ("_PREFIX" if operation.get("allow_partial") else "")
        if kind == "lasso_product_tree":
            beta = " ".join(str(limb) for limb in _field_limbs(operation["beta"]))
            lines.append(f"{wire_opcode} {len(values)} {beta}")
        else:
            challenges = " ".join(
                " ".join(str(limb) for limb in _field_limbs(challenge))
                for challenge in operation["challenges"]
            )
            lines.append(f"{wire_opcode} {len(values)} {len(operation['challenges'])} {challenges}")
        lines.extend(" ".join(str(limb) for limb in _field_limbs(value)) for value in values)
        if kind in {"sumcheck_product2_fold", "sumcheck_product3_fold"}:
            for operand in operation["operands"][1:]:
                lines.extend(" ".join(str(limb) for limb in _field_limbs(value)) for value in operand)
    lines.append("END")
    return lines


def _run_gpu_primitives_file(
    zk_root: pathlib.Path,
    input_path: pathlib.Path,
    label: str,
    operations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    binary, binary_sha256, build_identity, authenticated_manifest = _authenticated_cuda_binary(zk_root)
    out_path = input_path.with_suffix(".gpu_backend.json")
    env = stable_env()
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("CUDA_CACHE_DISABLE", "1")
    started = __import__("time").perf_counter()
    completed = subprocess.run(
        [str(binary), str(input_path), str(out_path), str(label)],
        cwd=str(zk_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_ms = (__import__("time").perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            "CUDA BN254-Fr proof primitive execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not out_path.is_file():
        raise RuntimeError("CUDA primitive backend exited without a result message")
    state = read_json(out_path)
    if (
        state.get("protocol") != _GPU_PROTOCOL
        or state.get("schema") != _GPU_SCHEMA
        or state.get("wire_version") != _GPU_WIRE_VERSION
    ):
        raise RuntimeError("CUDA backend returned an unknown proof-primitive protocol")
    if state.get("cuda_kernel_executed") is not True or state.get("cpu_fallback") is not False:
        raise RuntimeError("CUDA backend did not provide strict GPU execution evidence")
    if not isinstance(state.get("operations"), list) or not state["operations"]:
        raise RuntimeError("CUDA backend returned no operation proof messages")
    gpu_uuid = state.get("gpu_uuid")
    event_ms = state.get("cuda_event_elapsed_ms")
    if not isinstance(gpu_uuid, str) or len(gpu_uuid) != 32:
        raise RuntimeError("CUDA backend returned no canonical device UUID")
    if isinstance(event_ms, bool) or not isinstance(event_ms, (int, float)) or event_ms < 0:
        raise RuntimeError("CUDA backend returned no CUDA-event execution time")
    expected = recompute_gpu_proof_batch(operations)
    if state.get("operations") != expected.get("operations"):
        raise RuntimeError("CUDA backend output does not match the independent BN254-Fr reference")
    wire_bytes = ("\n".join(_wire_lines(operations)) + "\n").encode("utf-8")
    operation_digest = digest_json(_canonical_operations(operations))
    state["operation_digest"] = operation_digest
    state["wire_input_sha256"] = hashlib.sha256(wire_bytes).hexdigest()
    state["operation_output_digest"] = digest_json(state["operations"])
    state["proof_message_binding_digest"] = digest_json(
        {
            "operation_digest": operation_digest,
            "wire_input_sha256": state["wire_input_sha256"],
            "operation_output_digest": state["operation_output_digest"],
            "operations": state["operations"],
        }
    )
    # Public proof-execution records use one normalized identity/timing name;
    # retain the raw CUDA fields for the low-level message contract as well.
    state["device_uuid"] = gpu_uuid
    state["kernel_time_ms"] = float(event_ms)
    state["binary_sha256"] = binary_sha256
    state["build_identity"] = build_identity
    state["build_manifest_schema"] = authenticated_manifest.get("schema")
    state["build_manifest_digest"] = authenticated_manifest.get("manifest_digest")
    state["source_manifest_sha256"] = authenticated_manifest.get("source", {}).get("manifest_sha256")
    state["cuda_target"] = authenticated_manifest.get("target")
    target = authenticated_manifest.get("target", {})
    if state.get("gpu_compute_capability") != target.get("compute_capability") or state.get("gpu_arch") != target.get("code"):
        raise RuntimeError("CUDA device architecture does not match the authenticated build target")
    state["cuda_stdout"] = completed.stdout.strip()
    for timing_name in ("memory_copy_h2d_ms", "memory_copy_d2h_ms", "synchronization_time_ms"):
        timing_value = state.get(timing_name)
        if isinstance(timing_value, bool) or not isinstance(timing_value, (int, float)) or timing_value < 0:
            raise RuntimeError(f"CUDA backend returned invalid measured timing: {timing_name}")
    state.setdefault("total_time_ms", elapsed_ms)
    kernel_count = state.get("kernel_invocation_count")
    if isinstance(kernel_count, bool) or not isinstance(kernel_count, int) or kernel_count <= 0:
        raise RuntimeError("CUDA backend returned no native kernel invocation count")
    state["kernel_count"] = kernel_count
    return state


def gpu_proof_message_batch(
    zk_root: pathlib.Path,
    operations: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[str, Any]:
    """Compute native BN254-Fr proof messages on CUDA.

    The caller supplies the already-authenticated, transcript-ordered
    messages and challenges for Lasso product-tree, multilinear Sumcheck, and
    Hyrax/linear-evaluation folds.  For sequential Fiat--Shamir rounds use
    :func:`gpu_proof_round` so each next challenge is derived only after the
    previous GPU coefficients are returned.  This function never hashes or
    substitutes those messages, and it raises on a missing binary, missing
    device, wire mismatch, or kernel failure.  ``recompute_gpu_proof_batch``
    is a separate CPU reference for an independent verifier; it is
    deliberately not used as a runtime fallback here.
    """
    canonical = _canonical_operations(operations)
    zk_root = _lexical(pathlib.Path(zk_root))
    if zk_root.is_symlink() or not zk_root.is_dir():
        raise RuntimeError(f"CUDA baseline root must be a real directory: {zk_root}")
    runs_tmp = zk_root / "runs_tmp"
    if os.path.lexists(runs_tmp):
        _check_lexical_components(runs_tmp, zk_root, allow_missing=False)
        info = runs_tmp.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"runs_tmp must be a real directory: {runs_tmp}")
    else:
        _check_lexical_components(runs_tmp, zk_root, allow_missing=True)
        ensure_dir(runs_tmp, root=zk_root, label="CUDA runs_tmp")
    with tempfile.TemporaryDirectory(prefix="zkgpt-gpu-proof-", dir=str(runs_tmp)) as tmp:
        input_path = pathlib.Path(tmp) / "proof_messages.gpuv3"
        write_text_atomic(
            input_path,
            "\n".join(_wire_lines(canonical)) + "\n",
            label="CUDA proof input",
            root=zk_root,
        )
        state = _run_gpu_primitives_file(zk_root, input_path, label, canonical)
    state["proof_hot_path"] = True
    state["hot_path_semantics"] = _HOT_PATH_SEMANTICS
    state["hot_path_components"] = [
        "lasso_product_tree_grand_product_bn254_fr",
        "sumcheck_multilinear_round_coefficients_and_fold_bn254_fr",
        "hyrax_linear_evaluation_fold_bn254_fr",
        "single_gpu_field_transcript_accumulator_bn254_fr",
    ]
    state["cuda_kernel"] = _GPU_KERNEL
    state["input_encoding"] = _GPU_WIRE_VERSION.lower()
    state["operation_digest"] = digest_json(canonical)
    state["input_digest"] = state["wire_input_sha256"]
    state["primitive_input_count"] = int(state.get("field_words_processed", 0))
    state["verifier_binding"] = (
        "operation outputs bind the exact canonical BN254-Fr values, beta/challenges, "
        "round coefficients, folded claims, terminal claims, and one field transcript accumulator"
    )
    return state


def gpu_proof_round(
    zk_root: pathlib.Path,
    operation: Mapping[str, Any],
    challenges: Sequence[int],
    label: str,
) -> Dict[str, Any]:
    """Compute one transcript prefix on CUDA for sequential Fiat--Shamir.

    A Sumcheck/Hyrax caller can derive the next challenge only after seeing
    the preceding GPU coefficients.  This entry point therefore accepts the
    challenges known so far, recomputes that prefix on CUDA, and returns the
    last round's coefficients/folded claim.  The operation is marked partial
    internally; a final full proof should use
    :func:`gpu_proof_message_batch` or call this function through the complete
    challenge prefix and check ``operations[0]["complete"]``.
    """
    challenge_values = list(challenges)
    if not challenge_values:
        raise ValueError("gpu_proof_round requires at least one FS challenge")
    if not isinstance(operation, Mapping):
        raise TypeError("operation must be a mapping")
    entry = dict(operation)
    entry["challenges"] = challenge_values
    entry["_allow_prefix"] = True
    state = gpu_proof_message_batch(zk_root, [entry], label)
    operation_result = state["operations"][0]
    state["round_index"] = len(challenge_values) - 1
    state["round_complete"] = bool(operation_result.get("complete"))
    return state


def _reference_hex(value: int) -> str:
    return f"{int(value) % BN254_FR_MODULUS:064x}"


def recompute_gpu_proof_batch(operations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Independently recompute the CUDA batch messages using Python field math.

    This helper is for a verifier or a short unit test only.  It is not called
    by :func:`gpu_proof_message_batch`, so an unavailable CPU reference cannot
    turn a requested GPU run into a fallback execution.
    """
    canonical = _canonical_operations(operations)
    operation_results: List[Dict[str, Any]] = []
    transcript_stream: List[int] = []
    lasso_aggregate = 1
    sumcheck_aggregate = 0
    gkr_aggregate = 0
    lasso_count = 0
    sumcheck_round_count = 0
    for operation in canonical:
        values = list(operation["values"])
        kind = operation["kind"]
        transcript_stream.extend(values)
        if kind in {"sumcheck_product2_fold", "sumcheck_product3_fold"}:
            for operand in operation["operands"][1:]:
                transcript_stream.extend(operand)
        result: Dict[str, Any] = {
            "kind": kind,
            "input_count": len(values),
            "padded_count": len(values),
            "operand_count": 1 if kind not in {"sumcheck_product2_fold", "sumcheck_product3_fold"} else (2 if kind == "sumcheck_product2_fold" else 3),
            "complete": True,
        }
        if operation["kind"] == "lasso_product_tree":
            beta = operation["beta"]
            padded = 1
            while padded < len(values):
                padded <<= 1
            result["padded_count"] = padded
            acc = 1
            for index in range(padded):
                acc = acc * ((beta + values[index]) % BN254_FR_MODULUS if index < len(values) else 1)
            lasso_aggregate = lasso_aggregate * acc % BN254_FR_MODULUS
            lasso_count += len(values)
            transcript_stream.append(beta)
            result["grand_product_fr_hex"] = _reference_hex(acc)
            result["terminal_value_fr_hex"] = _reference_hex(acc)
        else:
            challenges = operation["challenges"]
            result["complete"] = len(challenges) == _log2_power_of_two(len(values))
            transcript_stream.extend(challenges)
            current_operands = [values]
            if kind in {"sumcheck_product2_fold", "sumcheck_product3_fold"}:
                current_operands.extend(operation["operands"][1:])
            rounds: List[Dict[str, Any]] = []
            for challenge in challenges:
                if kind == "sumcheck_mle_fold" or kind == "hyrax_linear_eval_fold":
                    left = sum(current_operands[0][0::2]) % BN254_FR_MODULUS
                    right = sum(current_operands[0][1::2]) % BN254_FR_MODULUS
                    coefficient_one = (right - left) % BN254_FR_MODULUS
                    coefficients = [left, coefficient_one] if kind == "sumcheck_mle_fold" else []
                elif kind == "sumcheck_product2_fold":
                    left_values, right_values = current_operands
                    coefficients = [0, 0, 0]
                    for a0, a1, b0, b1 in zip(left_values[0::2], left_values[1::2], right_values[0::2], right_values[1::2]):
                        da = (a1 - a0) % BN254_FR_MODULUS
                        db = (b1 - b0) % BN254_FR_MODULUS
                        coefficients[0] = (coefficients[0] + a0 * b0) % BN254_FR_MODULUS
                        coefficients[1] = (coefficients[1] + da * b0 + a0 * db) % BN254_FR_MODULUS
                        coefficients[2] = (coefficients[2] + da * db) % BN254_FR_MODULUS
                else:
                    selector_values, left_values, right_values = current_operands
                    coefficients = [0, 0, 0, 0]
                    for s0, s1, a0, a1, b0, b1 in zip(
                        selector_values[0::2], selector_values[1::2],
                        left_values[0::2], left_values[1::2],
                        right_values[0::2], right_values[1::2],
                    ):
                        ds = (s1 - s0) % BN254_FR_MODULUS
                        da = (a1 - a0) % BN254_FR_MODULUS
                        db = (b1 - b0) % BN254_FR_MODULUS
                        coefficients[0] = (coefficients[0] + ds * 0 + s0 * a0 * b0) % BN254_FR_MODULUS
                        coefficients[1] = (coefficients[1] + ds * a0 * b0 + s0 * da * b0 + s0 * a0 * db) % BN254_FR_MODULUS
                        coefficients[2] = (coefficients[2] + ds * da * b0 + ds * a0 * db + s0 * da * db) % BN254_FR_MODULUS
                        coefficients[3] = (coefficients[3] + ds * da * db) % BN254_FR_MODULUS
                folded_operands = [
                    [
                        (a + challenge * (b - a)) % BN254_FR_MODULUS
                        for a, b in zip(operand[0::2], operand[1::2])
                    ]
                    for operand in current_operands
                ]
                current_operands = folded_operands
                if kind == "sumcheck_product2_fold":
                    folded_claim = current_operands[0][0] * current_operands[1][0] % BN254_FR_MODULUS
                elif kind == "sumcheck_product3_fold":
                    folded_claim = (
                        current_operands[0][0]
                        * current_operands[1][0]
                        * current_operands[2][0]
                    ) % BN254_FR_MODULUS
                else:
                    folded_claim = current_operands[0][0]
                round_result = {
                    "challenge_fr_hex": _reference_hex(challenge),
                    "folded_claim_fr_hex": _reference_hex(folded_claim),
                }
                if coefficients:
                    round_result["coefficients_fr_hex"] = [_reference_hex(value) for value in coefficients]
                rounds.append(round_result)
                if kind in {"sumcheck_mle_fold", "sumcheck_product2_fold", "sumcheck_product3_fold"}:
                    transcript_stream.extend(coefficients)
                    transcript_stream.append(folded_claim)
                else:
                    transcript_stream.append(folded_claim)
            if kind == "sumcheck_product2_fold":
                terminal = current_operands[0][0] * current_operands[1][0] % BN254_FR_MODULUS
            elif kind == "sumcheck_product3_fold":
                terminal = current_operands[0][0] * current_operands[1][0] * current_operands[2][0] % BN254_FR_MODULUS
            else:
                terminal = current_operands[0][0]
            if kind in {"sumcheck_mle_fold", "sumcheck_product2_fold", "sumcheck_product3_fold"}:
                sumcheck_aggregate = (sumcheck_aggregate + terminal) % BN254_FR_MODULUS
                gkr_aggregate = (gkr_aggregate + terminal) % BN254_FR_MODULUS
                sumcheck_round_count += len(rounds)
            result["rounds"] = rounds
            result["terminal_value_fr_hex"] = _reference_hex(terminal)
        transcript_stream.append(int(result["terminal_value_fr_hex"], 16))
        operation_results.append(result)
    transcript_base = 0x9e3779b97f4a7c15
    transcript = 0
    for value in transcript_stream:
        transcript = (transcript * transcript_base + value) % BN254_FR_MODULUS
    return {
        "protocol": _GPU_PROTOCOL,
        "wire_version": _GPU_WIRE_VERSION,
        "operations": operation_results,
        "field_words_processed": sum(
            len(operation["values"])
            + sum(len(operand) for operand in operation.get("operands", [])[1:])
            for operation in canonical
        ),
        "primitive_counts": {
            "lasso": lasso_count,
            "matmul_sumcheck": sumcheck_round_count,
            "gkr_relation": sumcheck_round_count,
            "transcript": len(transcript_stream),
        },
        "lasso_grand_product_fr_hex": _reference_hex(lasso_aggregate),
        "matmul_sumcheck_fr_hex": _reference_hex(sumcheck_aggregate),
        "gkr_relation_fr_hex": _reference_hex(gkr_aggregate),
        "transcript_state_low256_hex": _reference_hex(transcript),
        "transcript_accumulator_fr_hex": _reference_hex(transcript),
    }


def validate_gpu_proof_state(
    state: Mapping[str, Any],
    *,
    require_complete: bool = True,
    expected_operations: Sequence[Mapping[str, Any]] | None = None,
    require_components: bool = True,
) -> tuple[bool, list[str]]:
    """Validate the serialized V3 GPU proof-message contract.

    This is intentionally structural and message-level: collectors must verify
    exact operation kinds, operand arities, round counts, coefficient degrees,
    terminal outputs, and counters instead of accepting a legacy label.  A
    A verifier with the original operation inputs can pass them as
    ``expected_operations`` to compare every output against
    :func:`recompute_gpu_proof_batch` and detect value-preserving tampering.
    """
    errors: list[str] = []

    def hex64(value: Any, label: str) -> None:
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{label}_not_fr_hex64")
            return
        try:
            parsed = int(value, 16)
        except ValueError:
            errors.append(f"{label}_not_hex")
            return
        if parsed >= BN254_FR_MODULUS:
            errors.append(f"{label}_noncanonical")

    if not isinstance(state, Mapping):
        return False, ["gpu_state_not_mapping"]
    if state.get("schema") != _GPU_SCHEMA:
        errors.append("schema_invalid")
    if state.get("protocol") != _GPU_PROTOCOL:
        errors.append("protocol_invalid")
    if state.get("wire_version") != _GPU_WIRE_VERSION:
        errors.append("wire_version_invalid")
    if state.get("cuda_kernel") != _GPU_KERNEL or state.get("cuda_kernel_version") != _GPU_KERNEL:
        errors.append("cuda_kernel_contract_invalid")
    if state.get("hot_path_semantics") != _HOT_PATH_SEMANTICS:
        errors.append("hot_path_semantics_invalid")
    if state.get("cuda_kernel_executed") is not True:
        errors.append("cuda_kernel_not_executed")
    if state.get("cpu_fallback") is not False:
        errors.append("cpu_fallback_present")
    if state.get("proof_hot_path") is not True:
        errors.append("proof_hot_path_not_bound")
    gpu_uuid = state.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or len(gpu_uuid) != 32 or any(character not in "0123456789abcdefABCDEF" for character in gpu_uuid):
        errors.append("gpu_uuid_invalid")
    event_ms = state.get("cuda_event_elapsed_ms")
    if not isinstance(event_ms, (int, float)) or isinstance(event_ms, bool) or event_ms < 0:
        errors.append("cuda_event_elapsed_ms_invalid")
    if state.get("cuda_event_timing") != "cudaEventElapsedTime":
        errors.append("cuda_event_timing_invalid")
    if state.get("transfer_timing") != "host_steady_clock":
        errors.append("transfer_timing_invalid")
    build_identity_value = state.get("build_identity")
    if not isinstance(build_identity_value, str) or len(build_identity_value) != 64 or any(
        char not in "0123456789abcdef" for char in build_identity_value
    ):
        errors.append("build_identity_invalid")
    binary_sha = state.get("binary_sha256")
    if not isinstance(binary_sha, str) or len(binary_sha) != 64 or binary_sha.lower() != binary_sha or any(
        char not in "0123456789abcdef" for char in binary_sha
    ):
        errors.append("binary_sha256_invalid")
    if state.get("build_manifest_schema") != "zkGPT_cuda_build_manifest_v1":
        errors.append("build_manifest_schema_invalid")
    for identity_name in ("build_manifest_digest", "source_manifest_sha256"):
        value = state.get(identity_name)
        if not isinstance(value, str) or len(value) != 64 or value.lower() != value or any(
            char not in "0123456789abcdef" for char in value
        ):
            errors.append(f"{identity_name}_invalid")
    target = state.get("cuda_target")
    if not isinstance(target, Mapping) or any(not isinstance(target.get(key), str) or not target.get(key) for key in ("compute_capability", "arch", "code", "gencode")):
        errors.append("cuda_target_invalid")
    elif (
        str(target.get("compute_capability")).count(".") != 1
        or any(not part.isdigit() for part in str(target.get("compute_capability")).split("."))
    ):
        errors.append("cuda_target_compute_capability_invalid")
    elif (
        target.get("arch") != f"compute_{str(target['compute_capability']).replace('.', '')}"
        or target.get("code") != f"sm_{str(target['compute_capability']).replace('.', '')}"
        or target.get("gencode") != f"arch=compute_{str(target['compute_capability']).replace('.', '')},code=sm_{str(target['compute_capability']).replace('.', '')}"
    ):
        errors.append("cuda_target_internal_mismatch")
    if state.get("gpu_compute_capability") != (target.get("compute_capability") if isinstance(target, Mapping) else None):
        errors.append("gpu_compute_capability_mismatch")
    if state.get("gpu_arch") != (target.get("code") if isinstance(target, Mapping) else None):
        errors.append("gpu_arch_mismatch")

    operations = state.get("operations")
    if not isinstance(operations, list) or not operations:
        errors.append("operations_missing")
        operations = []
    operation_input_fields = 0
    expected_lasso = 0
    expected_sumcheck_rounds = 0
    seen_lasso = False
    seen_sumcheck = False
    seen_hyrax = False
    for index, operation in enumerate(operations):
        prefix = f"operations[{index}]"
        if not isinstance(operation, Mapping):
            errors.append(f"{prefix}_not_mapping")
            continue
        kind = operation.get("kind")
        expected_operand_count = 1
        if kind == "sumcheck_product2_fold":
            expected_operand_count = 2
        elif kind == "sumcheck_product3_fold":
            expected_operand_count = 3
        elif kind in {"sumcheck_mle_fold", "hyrax_linear_eval_fold", "lasso_product_tree"}:
            expected_operand_count = 1
        else:
            errors.append(f"{prefix}_kind_invalid")
        if operation.get("operand_count") != expected_operand_count:
            errors.append(f"{prefix}_operand_count_invalid")
        input_count = operation.get("input_count")
        padded_count = operation.get("padded_count")
        if not isinstance(input_count, int) or isinstance(input_count, bool) or input_count <= 0:
            errors.append(f"{prefix}_input_count_invalid")
            input_count = 0
        if not isinstance(padded_count, int) or isinstance(padded_count, bool) or padded_count < input_count:
            errors.append(f"{prefix}_padded_count_invalid")
            padded_count = 0
        operation_input_fields += input_count * expected_operand_count
        if require_complete and operation.get("complete") is not True:
            errors.append(f"{prefix}_incomplete")
        elif not require_complete and operation.get("complete") not in {True, False}:
            errors.append(f"{prefix}_complete_flag_invalid")
        hex64(operation.get("terminal_value_fr_hex"), f"{prefix}.terminal_value")
        if kind == "lasso_product_tree":
            seen_lasso = True
            expected_lasso += input_count
            hex64(operation.get("grand_product_fr_hex"), f"{prefix}.grand_product")
            if padded_count and not _is_power_of_two(padded_count):
                errors.append(f"{prefix}_padded_count_not_power_two")
            continue
        if kind in {"sumcheck_mle_fold", "sumcheck_product2_fold", "sumcheck_product3_fold"}:
            seen_sumcheck = True
            rounds = operation.get("rounds")
            if not isinstance(rounds, list):
                errors.append(f"{prefix}_rounds_missing")
                rounds = []
            if operation.get("complete") is True and padded_count and _is_power_of_two(padded_count) and len(rounds) != _log2_power_of_two(padded_count):
                errors.append(f"{prefix}_round_count_invalid")
            expected_degree = {
                "sumcheck_mle_fold": 2,
                "sumcheck_product2_fold": 3,
                "sumcheck_product3_fold": 4,
            }[kind]
            expected_sumcheck_rounds += len(rounds)
            for round_index, round_message in enumerate(rounds):
                round_prefix = f"{prefix}.rounds[{round_index}]"
                if not isinstance(round_message, Mapping):
                    errors.append(f"{round_prefix}_not_mapping")
                    continue
                hex64(round_message.get("challenge_fr_hex"), f"{round_prefix}.challenge")
                hex64(round_message.get("folded_claim_fr_hex"), f"{round_prefix}.folded_claim")
                coefficients = round_message.get("coefficients_fr_hex")
                if not isinstance(coefficients, list) or len(coefficients) != expected_degree:
                    errors.append(f"{round_prefix}_coefficient_degree_invalid")
                else:
                    for coefficient_index, coefficient in enumerate(coefficients):
                        hex64(coefficient, f"{round_prefix}.coefficients[{coefficient_index}]")
        elif kind == "hyrax_linear_eval_fold":
            seen_hyrax = True
            rounds = operation.get("rounds")
            if not isinstance(rounds, list):
                errors.append(f"{prefix}_rounds_missing")
                rounds = []
            if operation.get("complete") is True and padded_count and _is_power_of_two(padded_count) and len(rounds) != _log2_power_of_two(padded_count):
                errors.append(f"{prefix}_round_count_invalid")
            for round_index, round_message in enumerate(rounds):
                round_prefix = f"{prefix}.rounds[{round_index}]"
                if not isinstance(round_message, Mapping):
                    errors.append(f"{round_prefix}_not_mapping")
                    continue
                hex64(round_message.get("challenge_fr_hex"), f"{round_prefix}.challenge")
                hex64(round_message.get("folded_claim_fr_hex"), f"{round_prefix}.folded_claim")
                if "coefficients_fr_hex" in round_message:
                    errors.append(f"{round_prefix}_hyrax_coefficients_unexpected")

    if require_components:
        if not seen_lasso:
            errors.append("lasso_operation_missing")
        if not seen_sumcheck:
            errors.append("sumcheck_operation_missing")
        if not seen_hyrax:
            errors.append("hyrax_operation_missing")
    counts = state.get("primitive_counts")
    if not isinstance(counts, Mapping):
        errors.append("primitive_counts_missing")
        counts = {}
    if counts.get("lasso") != expected_lasso:
        errors.append("lasso_count_mismatch")
    if counts.get("matmul_sumcheck") != expected_sumcheck_rounds:
        errors.append("sumcheck_count_mismatch")
    if counts.get("gkr_relation") != expected_sumcheck_rounds:
        errors.append("gkr_count_mismatch")
    transcript_count = counts.get("transcript")
    if not isinstance(transcript_count, int) or isinstance(transcript_count, bool) or transcript_count <= 0:
        errors.append("transcript_count_invalid")
    if state.get("field_words_processed") != operation_input_fields:
        errors.append("field_words_processed_mismatch")
    kernel_count = state.get("kernel_invocation_count")
    if not isinstance(kernel_count, int) or isinstance(kernel_count, bool) or kernel_count <= 0:
        errors.append("kernel_invocation_count_invalid")
    kernel_counts = state.get("kernel_invocation_counts")
    if not isinstance(kernel_counts, Mapping):
        errors.append("kernel_invocation_counts_missing")
    else:
        kernel_names = ("lasso", "matmul_sumcheck", "gkr_relation", "hyrax", "transcript")
        for name in kernel_names:
            value = kernel_counts.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"kernel_invocation_counts_{name}_invalid")
        if all(isinstance(kernel_counts.get(name), int) and not isinstance(kernel_counts.get(name), bool) for name in kernel_names) and isinstance(kernel_count, int):
            if sum(int(kernel_counts[name]) for name in kernel_names) != kernel_count:
                errors.append("kernel_invocation_counts_total_mismatch")
        expected_kernel_counts = {name: 0 for name in kernel_names}
        product_bucket = "matmul_sumcheck" if "matmul" in str(state.get("label", "")).lower() else "gkr_relation"
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            kind = operation.get("kind")
            padded = operation.get("padded_count")
            rounds = operation.get("rounds")
            round_count = len(rounds) if isinstance(rounds, list) else 0
            if kind == "lasso_product_tree" and isinstance(padded, int) and _is_power_of_two(padded):
                expected_kernel_counts["lasso"] += 1 + _log2_power_of_two(padded)
            elif kind == "sumcheck_mle_fold":
                expected_kernel_counts["gkr_relation"] += 2 * round_count
            elif kind == "sumcheck_product2_fold":
                expected_kernel_counts[product_bucket] += 4 * round_count + 1
            elif kind == "sumcheck_product3_fold":
                expected_kernel_counts[product_bucket] += 5 * round_count + 1
            elif kind == "hyrax_linear_eval_fold":
                expected_kernel_counts["hyrax"] += round_count
        expected_kernel_counts["transcript"] = 1
        if all(isinstance(kernel_counts.get(name), int) and not isinstance(kernel_counts.get(name), bool) for name in kernel_names):
            if {name: int(kernel_counts[name]) for name in kernel_names} != expected_kernel_counts:
                errors.append("kernel_invocation_counts_operation_mismatch")
    for timing_name in ("memory_copy_h2d_ms", "memory_copy_d2h_ms", "synchronization_time_ms"):
        timing_value = state.get(timing_name)
        if isinstance(timing_value, bool) or not isinstance(timing_value, (int, float)) or timing_value < 0:
            errors.append(f"{timing_name}_invalid")
    for field_name in (
        "lasso_grand_product_fr_hex",
        "matmul_sumcheck_fr_hex",
        "gkr_relation_fr_hex",
        "transcript_state_low256_hex",
        "transcript_accumulator_fr_hex",
    ):
        hex64(state.get(field_name), field_name)
    if expected_operations is not None:
        try:
            canonical_expected = _canonical_operations(expected_operations)
            expected = recompute_gpu_proof_batch(canonical_expected)
            for field_name in (
                "operations",
                "field_words_processed",
                "primitive_counts",
                "lasso_grand_product_fr_hex",
                "matmul_sumcheck_fr_hex",
                "gkr_relation_fr_hex",
                "transcript_state_low256_hex",
                "transcript_accumulator_fr_hex",
            ):
                if state.get(field_name) != expected.get(field_name):
                    errors.append(f"{field_name}_does_not_match_independent_reference")
            expected_operation_digest = digest_json(canonical_expected)
            expected_wire_digest = hashlib.sha256(("\n".join(_wire_lines(canonical_expected)) + "\n").encode("utf-8")).hexdigest()
            if state.get("operation_digest") != expected_operation_digest:
                errors.append("operation_digest_does_not_match_expected_input")
            if state.get("wire_input_sha256") != expected_wire_digest:
                errors.append("wire_input_sha256_does_not_match_expected_input")
            if state.get("operation_output_digest") != digest_json(expected.get("operations")):
                errors.append("operation_output_digest_does_not_match_expected_output")
            expected_binding = digest_json(
                {
                    "operation_digest": expected_operation_digest,
                    "wire_input_sha256": expected_wire_digest,
                    "operation_output_digest": digest_json(expected.get("operations")),
                    "operations": expected.get("operations"),
                }
            )
            if state.get("proof_message_binding_digest") != expected_binding:
                errors.append("proof_message_binding_digest_invalid")
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"independent_reference_invalid:{exc}")
    return not errors, errors


def run_gpu_primitives_probe(zk_root: pathlib.Path, seed_payload: Dict[str, Any]) -> Dict[str, Any]:
    digest = hashlib.sha256(
        __import__("json").dumps(seed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).digest()
    values = [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 16, 4)]
    operations = [
        {"kind": "lasso_product_tree", "values": values, "beta": 0},
        {"kind": "sumcheck_mle_fold", "values": values, "challenges": [0, 0]},
        {"kind": "hyrax_linear_eval_fold", "values": values, "challenges": [0, 0]},
    ]
    return gpu_proof_message_batch(zk_root, operations, "backend-probe")


def backend_state(backend: str, zk_root: pathlib.Path, seed_payload: Dict[str, Any]) -> Dict[str, Any]:
    if backend == "cpu":
        source_digest = source_manifest_digest(source_inventory(_lexical(pathlib.Path(zk_root))))
        return {
            "backend": "cpu",
            "cuda_kernel_executed": False,
            "gpu_runtime_present": False,
            "build_manifest_schema": "zkGPT_cuda_build_manifest_v1",
            "build_identity": None,
            "build_manifest_digest": None,
            "source_manifest_sha256": source_digest,
            "source_identity_schema": "zkGPT_cuda_source_inventory_v1",
            "note": "CPU proof path",
        }
    if backend == "gpu":
        state = run_gpu_primitives_probe(zk_root, seed_payload)
        if state.get("cuda_kernel_executed") is not True or state.get("cpu_fallback") is not False:
            raise RuntimeError("GPU backend did not execute the strict CUDA primitive batch")
        return state
    raise ValueError(f"unknown backend: {backend}")
