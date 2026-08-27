"""Run-scoped CPU/CUDA dispatch for proof-authoring primitives.

The CUDA path is fail-closed.  Product-Sumcheck coefficients, Lasso grand
products, and MLE terminal evaluations returned here are consumed by the
actual proof messages.  The Python calculations in callers are equality
checks, not runtime fallbacks.
"""
from __future__ import annotations

import contextvars
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .backend import (
    _GPU_KERNEL,
    _GPU_PROTOCOL,
    _GPU_SCHEMA,
    _GPU_WIRE_VERSION,
    gpu_proof_message_batch,
    recompute_gpu_proof_batch,
    validate_gpu_proof_state,
)
from .field import BN254_FR_MODULUS, to_field_int
from .util import digest_json


_ACTIVE: contextvars.ContextVar["ProofExecution | None"] = contextvars.ContextVar(
    "zkgpt_active_proof_execution", default=None
)
GPU_EXECUTION_SCHEMA = "zkGPT_gpu_proof_execution_v3"
GPU_EXECUTION_SEMANTICS = "zkgpt_cuda_authors_bn254_fr_proof_messages_v3"
GPU_COMPONENTS = frozenset({"sumcheck_product", "lasso_grand_product", "hyrax_mle_fold"})
_UUID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_COMPONENT_KINDS = {
    "sumcheck_product": {"sumcheck_mle_fold", "sumcheck_product2_fold", "sumcheck_product3_fold"},
    "lasso_grand_product": {"lasso_product_tree"},
    "hyrax_mle_fold": {"hyrax_linear_eval_fold"},
}


def _scalar(value: str) -> int:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise RuntimeError("CUDA proof primitive returned a non-canonical Fr message")
    return int(value, 16)


def _fr_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        return int(value, 16) < BN254_FR_MODULUS
    except ValueError:
        return False


def _operation_message(state: Mapping[str, Any]) -> dict[str, Any]:
    operations = state.get("operations")
    if not isinstance(operations, list) or len(operations) != 1 or not isinstance(operations[0], Mapping):
        raise RuntimeError("CUDA proof primitive returned an invalid single operation message")
    # The operation contains only proof outputs and public cardinalities.  The
    # private input vectors and their digest are deliberately not copied into
    # the run evidence.
    return dict(operations[0])


def _output_slots(operation: Mapping[str, Any]) -> dict[str, int]:
    kind = operation.get("kind")
    rounds = operation.get("rounds", [])
    round_count = len(rounds) if isinstance(rounds, list) else 0
    if kind == "lasso_product_tree":
        return {"terminal_value_fr": 1, "grand_product_fr": 1, "rounds": 0}
    if kind == "sumcheck_mle_fold":
        return {
            "terminal_value_fr": 1,
            "rounds": round_count,
            "challenge_fr_per_round": 1,
            "folded_claim_fr_per_round": 1,
            "coefficients_fr_per_round": 2,
        }
    if kind == "sumcheck_product2_fold":
        return {
            "terminal_value_fr": 1,
            "rounds": round_count,
            "challenge_fr_per_round": 1,
            "folded_claim_fr_per_round": 1,
            "coefficients_fr_per_round": 3,
        }
    if kind == "sumcheck_product3_fold":
        return {
            "terminal_value_fr": 1,
            "rounds": round_count,
            "challenge_fr_per_round": 1,
            "folded_claim_fr_per_round": 1,
            "coefficients_fr_per_round": 4,
        }
    if kind == "hyrax_linear_eval_fold":
        return {
            "terminal_value_fr": 1,
            "rounds": round_count,
            "challenge_fr_per_round": 1,
            "folded_claim_fr_per_round": 1,
            "coefficients_per_round": 0,
        }
    raise RuntimeError(f"unsupported CUDA operation kind: {kind!r}")


def _public_invocation(state: Mapping[str, Any], label: str, component: str, shape: Sequence[int]) -> dict[str, Any]:
    operation = _operation_message(state)
    primitive_counts = state.get("primitive_counts")
    if not isinstance(primitive_counts, Mapping):
        raise RuntimeError("CUDA proof primitive omitted primitive counts")
    # Raw operation outputs, output digests, and aggregate field values are
    # deliberately not serialized: they are witness-dependent proof messages
    # and must remain in the masked proof payload.  The public record carries
    # their exact slot/cardinality contract; the formal verifier checks the
    # actual values and equations at their native proof positions.
    descriptor = {
        "label": label,
        "component": component,
        "shape": [int(value) for value in shape],
        "operation_kind": operation["kind"],
        "operation_input_count": operation.get("input_count"),
        "operation_padded_count": operation.get("padded_count"),
        "operation_operand_count": operation.get("operand_count"),
        "operation_complete": operation.get("complete"),
        "operation_round_count": len(operation.get("rounds", [])) if isinstance(operation.get("rounds", []), list) else None,
        "operation_output_slots": _output_slots(operation),
        "device_name": state.get("device_name"),
        "gpu_uuid": state.get("gpu_uuid"),
        # Keep the verifier-facing name explicit while retaining the CUDA
        # runtime spelling in ``gpu_uuid``.  They must be identical.
        "device_uuid": state.get("gpu_uuid"),
        "backend": state.get("backend"),
        "protocol": state.get("protocol"),
        "message_schema": state.get("schema"),
        "wire_version": state.get("wire_version"),
        "cuda_kernel": state.get("cuda_kernel"),
        "cuda_kernel_version": state.get("cuda_kernel_version"),
        "binary_sha256": state.get("binary_sha256"),
        "build_identity": state.get("build_identity"),
        "cuda_event_timing": state.get("cuda_event_timing"),
        "cuda_event_elapsed_ms": float(state.get("cuda_event_elapsed_ms", -1.0)),
        "kernel_time_ms": float(state.get("kernel_time_ms", 0.0)),
        "memory_copy_h2d_ms": float(state.get("memory_copy_h2d_ms", 0.0)),
        "memory_copy_d2h_ms": float(state.get("memory_copy_d2h_ms", 0.0)),
        "synchronization_time_ms": float(state.get("synchronization_time_ms", 0.0)),
        "field_words_processed": int(state.get("field_words_processed", 0)),
        "primitive_counts": dict(primitive_counts),
        "cuda_kernel_executed": state.get("cuda_kernel_executed") is True,
        "cpu_fallback": state.get("cpu_fallback") is True,
    }
    descriptor["public_message_digest"] = digest_json(descriptor)
    return descriptor


@dataclass
class ProofExecution:
    backend: str
    zk_root: pathlib.Path
    session_nonce: str
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("proof execution backend must be cpu or gpu")
        self.zk_root = pathlib.Path(self.zk_root).resolve()

    @property
    def gpu(self) -> bool:
        return self.backend == "gpu"

    def _invoke(self, operation: Mapping[str, Any], label: str, component: str, shape: Sequence[int]) -> Mapping[str, Any]:
        state = gpu_proof_message_batch(self.zk_root, [operation], label)
        expected = recompute_gpu_proof_batch([operation])
        valid, errors = validate_gpu_proof_state(
            state,
            require_complete=not bool(operation.get("_allow_prefix", operation.get("allow_partial", False))),
            expected_operations=[operation],
            require_components=False,
        )
        if not valid:
            raise RuntimeError("CUDA proof primitive failed strict message validation: " + ";".join(errors))
        if state.get("operations") != expected.get("operations"):
            raise RuntimeError(f"CUDA {component} output differs from the independent BN254-Fr reference")
        invocation = _public_invocation(state, label, component, shape)
        if not invocation["cuda_kernel_executed"] or invocation["cpu_fallback"]:
            raise RuntimeError("CUDA proof-authoring invocation did not execute fail-closed")
        self.invocations.append(invocation)
        return state["operations"][0]

    def product_coefficients(self, operands: Sequence[Sequence[int]], label: str) -> list[int] | None:
        if not self.gpu:
            return None
        if len(operands) not in {2, 3}:
            raise ValueError("GPU product Sumcheck needs two or three operands")
        vectors = [[to_field_int(value) for value in operand] for operand in operands]
        count = len(vectors[0])
        if count <= 1 or any(len(vector) != count for vector in vectors):
            raise ValueError("GPU product Sumcheck operands have inconsistent domains")
        operation = {
            "kind": f"sumcheck_product{len(vectors)}_fold",
            "operands": vectors,
            # Only round zero is consumed; later formal rounds invoke CUDA
            # again on the inputs folded by the actual transcript challenge.
            "challenges": [0],
            "_allow_prefix": True,
        }
        result = self._invoke(operation, label, "sumcheck_product", [len(vectors), count])
        return [_scalar(value) for value in result["rounds"][0]["coefficients_fr_hex"]]

    def lasso_product(self, factors: Sequence[int], label: str) -> int | None:
        if not self.gpu:
            return None
        values = [to_field_int(value) for value in factors]
        result = self._invoke(
            {"kind": "lasso_product_tree", "values": values, "beta": 0},
            label, "lasso_grand_product", [len(values)],
        )
        return _scalar(result["grand_product_fr_hex"])

    def mle_fold(self, values: Sequence[int], challenges: Sequence[int], label: str) -> int | None:
        if not self.gpu:
            return None
        vector = [to_field_int(value) for value in values]
        point = [to_field_int(value) for value in challenges]
        result = self._invoke(
            {"kind": "hyrax_linear_eval_fold", "values": vector, "challenges": point},
            label, "hyrax_mle_fold", [len(vector), len(point)],
        )
        return _scalar(result["terminal_value_fr_hex"])

    def evidence(self, final_transcript_digest: str) -> dict[str, Any] | None:
        if not self.gpu:
            return None
        if not self.invocations:
            raise RuntimeError("GPU proof path produced no proof-authoring invocations")
        components = {item["component"] for item in self.invocations}
        if components != GPU_COMPONENTS:
            raise RuntimeError(f"GPU proof path has wrong native components: {sorted(components)}")
        payload = {
            "schema": GPU_EXECUTION_SCHEMA,
            "semantic_id": GPU_EXECUTION_SEMANTICS,
            "primitive_contract": _GPU_SCHEMA,
            "wire_version": _GPU_WIRE_VERSION,
            "cuda_kernel": _GPU_KERNEL,
            "backend": "gpu",
            "session_nonce": self.session_nonce,
            "final_transcript_digest": final_transcript_digest,
            "cuda_kernel_executed": all(item["cuda_kernel_executed"] for item in self.invocations),
            "cpu_fallback": any(item["cpu_fallback"] for item in self.invocations),
            "proof_hot_path": True,
            "invocation_count": len(self.invocations),
            "components": sorted(components),
            "invocations": list(self.invocations),
        }
        payload["transcript_binding_digest"] = digest_json(gpu_execution_transcript_binding(payload))
        payload["execution_digest"] = digest_json(payload)
        return payload

    def transcript_binding(self) -> dict[str, Any] | None:
        if not self.gpu:
            return None
        components = {item["component"] for item in self.invocations}
        if not self.invocations or components != GPU_COMPONENTS:
            raise RuntimeError("GPU transcript binding is incomplete")
        return gpu_execution_transcript_binding({
            "schema": GPU_EXECUTION_SCHEMA,
            "semantic_id": GPU_EXECUTION_SEMANTICS,
            "backend": "gpu",
            "session_nonce": self.session_nonce,
            "cuda_kernel_executed": True,
            "cpu_fallback": False,
            "proof_hot_path": True,
            "invocation_count": len(self.invocations),
            "components": sorted(components),
            "invocations": list(self.invocations),
        })


def gpu_execution_transcript_binding(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical public execution record absorbed by global FS."""
    return {
        "schema": evidence.get("schema"),
        "semantic_id": evidence.get("semantic_id"),
        "backend": evidence.get("backend"),
        "session_nonce": evidence.get("session_nonce"),
        "cuda_kernel_executed": evidence.get("cuda_kernel_executed"),
        "cpu_fallback": evidence.get("cpu_fallback"),
        "proof_hot_path": evidence.get("proof_hot_path"),
        "invocation_count": evidence.get("invocation_count"),
        "components": evidence.get("components"),
        "invocations": evidence.get("invocations"),
    }


def validate_gpu_execution_evidence(
    evidence: Mapping[str, Any], *, session_nonce: str | None = None,
    final_transcript_digest: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate public, run-scoped GPU proof-authoring evidence.

    Private operation vectors and deterministic witness digests are not part
    of this record.  The independent verifier checks the CUDA-authored roots,
    coefficients and MLE terminals at their native positions in the proof.
    """
    errors: list[str] = []
    if not isinstance(evidence, Mapping):
        return False, ["gpu_execution_not_mapping"]
    payload = dict(evidence)
    claimed_digest = payload.pop("execution_digest", None)
    if claimed_digest != digest_json(payload):
        errors.append("execution_digest_invalid")
    if evidence.get("schema") != GPU_EXECUTION_SCHEMA:
        errors.append("schema_invalid")
    if evidence.get("semantic_id") != GPU_EXECUTION_SEMANTICS:
        errors.append("semantic_id_invalid")
    if evidence.get("primitive_contract") != _GPU_SCHEMA:
        errors.append("primitive_contract_invalid")
    if evidence.get("wire_version") != _GPU_WIRE_VERSION:
        errors.append("wire_version_invalid")
    if evidence.get("cuda_kernel") != _GPU_KERNEL:
        errors.append("cuda_kernel_invalid")
    if evidence.get("backend") != "gpu":
        errors.append("backend_invalid")
    if session_nonce is not None and evidence.get("session_nonce") != session_nonce:
        errors.append("session_nonce_invalid")
    if final_transcript_digest is not None and evidence.get("final_transcript_digest") != final_transcript_digest:
        errors.append("final_transcript_digest_invalid")
    if evidence.get("cuda_kernel_executed") is not True:
        errors.append("cuda_kernel_not_executed")
    if evidence.get("cpu_fallback") is not False:
        errors.append("cpu_fallback_present")
    if evidence.get("proof_hot_path") is not True:
        errors.append("proof_hot_path_not_bound")
    if evidence.get("transcript_binding_digest") != digest_json(gpu_execution_transcript_binding(evidence)):
        errors.append("transcript_binding_digest_invalid")
    components = evidence.get("components")
    if not isinstance(components, list) or set(components) != GPU_COMPONENTS or len(components) != len(GPU_COMPONENTS):
        errors.append("components_invalid")
    invocations = evidence.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        errors.append("invocations_missing")
        invocations = []
    if evidence.get("invocation_count") != len(invocations):
        errors.append("invocation_count_invalid")
    labels: set[str] = set()
    for index, invocation in enumerate(invocations):
        prefix = f"invocations[{index}]"
        if not isinstance(invocation, Mapping):
            errors.append(f"{prefix}_not_mapping")
            continue
        descriptor = dict(invocation)
        public_digest = descriptor.pop("public_message_digest", None)
        if public_digest != digest_json(descriptor):
            errors.append(f"{prefix}_digest_invalid")
        label = invocation.get("label")
        if not isinstance(label, str) or not label or label in labels:
            errors.append(f"{prefix}_label_invalid")
        else:
            labels.add(label)
        if invocation.get("component") not in GPU_COMPONENTS:
            errors.append(f"{prefix}_component_invalid")
        if invocation.get("backend") != "gpu":
            errors.append(f"{prefix}_backend_invalid")
        for name in ("device_name", "gpu_uuid", "device_uuid", "cuda_kernel", "cuda_kernel_version"):
            if not isinstance(invocation.get(name), str) or not invocation[name]:
                errors.append(f"{prefix}_{name}_invalid")
        binary_sha256 = invocation.get("binary_sha256")
        if not isinstance(binary_sha256, str) or len(binary_sha256) != 64 or binary_sha256.lower() != binary_sha256:
            errors.append(f"{prefix}_binary_sha256_invalid")
        elif any(char not in "0123456789abcdef" for char in binary_sha256):
            errors.append(f"{prefix}_binary_sha256_invalid")
        if invocation.get("build_identity") != digest_json({"kernel": _GPU_KERNEL, "binary_sha256": binary_sha256}):
            errors.append(f"{prefix}_build_identity_invalid")
        if not isinstance(invocation.get("gpu_uuid"), str) or not _UUID_RE.fullmatch(str(invocation.get("gpu_uuid", ""))):
            errors.append(f"{prefix}_gpu_uuid_format_invalid")
        if invocation.get("device_uuid") != invocation.get("gpu_uuid"):
            errors.append(f"{prefix}_device_uuid_mismatch")
        if invocation.get("message_schema") != _GPU_SCHEMA:
            errors.append(f"{prefix}_message_schema_invalid")
        if invocation.get("protocol") != _GPU_PROTOCOL:
            errors.append(f"{prefix}_protocol_invalid")
        if invocation.get("wire_version") != _GPU_WIRE_VERSION:
            errors.append(f"{prefix}_wire_version_invalid")
        if invocation.get("cuda_kernel") != _GPU_KERNEL or invocation.get("cuda_kernel_version") != _GPU_KERNEL:
            errors.append(f"{prefix}_kernel_contract_invalid")
        if invocation.get("cuda_event_timing") != "cudaEventElapsedTime":
            errors.append(f"{prefix}_cuda_event_timing_invalid")
        event_ms = invocation.get("cuda_event_elapsed_ms")
        if isinstance(event_ms, bool) or not isinstance(event_ms, (int, float)) or float(event_ms) < 0.0:
            errors.append(f"{prefix}_cuda_event_elapsed_ms_invalid")
        if invocation.get("cuda_kernel_executed") is not True:
            errors.append(f"{prefix}_kernel_not_executed")
        if invocation.get("cpu_fallback") is not False:
            errors.append(f"{prefix}_cpu_fallback")
        shape = invocation.get("shape")
        if not isinstance(shape, list) or not shape or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
        ):
            errors.append(f"{prefix}_shape_invalid")
        words = invocation.get("field_words_processed")
        if not isinstance(words, int) or isinstance(words, bool) or words <= 0:
            errors.append(f"{prefix}_field_words_invalid")
        kind = invocation.get("operation_kind")
        component = invocation.get("component")
        if kind not in _COMPONENT_KINDS.get(component, set()):
            errors.append(f"{prefix}_operation_kind_component_invalid")
        input_count = invocation.get("operation_input_count")
        padded_count = invocation.get("operation_padded_count")
        operand_count = invocation.get("operation_operand_count")
        if not isinstance(input_count, int) or isinstance(input_count, bool) or input_count <= 0:
            errors.append(f"{prefix}_operation_input_count_invalid")
        if not isinstance(padded_count, int) or isinstance(padded_count, bool) or padded_count < input_count or padded_count & (padded_count - 1):
            errors.append(f"{prefix}_operation_padded_count_invalid")
        expected_operand_count = 1 if kind not in {"sumcheck_product2_fold", "sumcheck_product3_fold"} else (2 if kind == "sumcheck_product2_fold" else 3)
        if operand_count != expected_operand_count:
            errors.append(f"{prefix}_operation_operand_count_invalid")
        round_count = invocation.get("operation_round_count")
        if not isinstance(round_count, int) or isinstance(round_count, bool) or round_count < 0:
            errors.append(f"{prefix}_operation_round_count_invalid")
            round_count = 0
        output_slots = invocation.get("operation_output_slots")
        if not isinstance(output_slots, Mapping):
            errors.append(f"{prefix}_operation_output_slots_missing")
            output_slots = {}
        if kind == "lasso_product_tree":
            expected_slots = {"terminal_value_fr": 1, "grand_product_fr": 1, "rounds": 0}
            if dict(output_slots) != expected_slots:
                errors.append(f"{prefix}_lasso_output_slots_invalid")
            if round_count != 0:
                errors.append(f"{prefix}_lasso_rounds_unexpected")
        elif kind in {"sumcheck_mle_fold", "sumcheck_product2_fold", "sumcheck_product3_fold"}:
            expected_rounds = padded_count.bit_length() - 1 if isinstance(padded_count, int) and padded_count > 0 else -1
            degree = {"sumcheck_mle_fold": 2, "sumcheck_product2_fold": 3, "sumcheck_product3_fold": 4}[kind]
            expected_slots = {
                "terminal_value_fr": 1,
                "rounds": round_count,
                "challenge_fr_per_round": 1,
                "folded_claim_fr_per_round": 1,
                "coefficients_fr_per_round": degree,
            }
            if dict(output_slots) != expected_slots:
                errors.append(f"{prefix}_sumcheck_output_slots_invalid")
            if invocation.get("operation_complete") is True and round_count != expected_rounds:
                errors.append(f"{prefix}_sumcheck_round_count_invalid")
        elif kind == "hyrax_linear_eval_fold":
            expected_rounds = padded_count.bit_length() - 1 if isinstance(padded_count, int) and padded_count > 0 else -1
            expected_slots = {
                "terminal_value_fr": 1,
                "rounds": round_count,
                "challenge_fr_per_round": 1,
                "folded_claim_fr_per_round": 1,
                "coefficients_per_round": 0,
            }
            if dict(output_slots) != expected_slots:
                errors.append(f"{prefix}_hyrax_output_slots_invalid")
            if invocation.get("operation_complete") is True and round_count != expected_rounds:
                errors.append(f"{prefix}_hyrax_round_count_invalid")
        counts = invocation.get("primitive_counts")
        if not isinstance(counts, Mapping):
            errors.append(f"{prefix}_primitive_counts_missing")
        elif any(not isinstance(counts.get(name), int) or isinstance(counts.get(name), bool) or counts.get(name) < 0 for name in ("lasso", "matmul_sumcheck", "gkr_relation", "transcript")):
            errors.append(f"{prefix}_primitive_counts_invalid")
        for name in ("kernel_time_ms", "memory_copy_h2d_ms", "memory_copy_d2h_ms", "synchronization_time_ms"):
            value = invocation.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                errors.append(f"{prefix}_{name}_invalid")
    return not errors, errors


def activate(execution: ProofExecution):
    return _ACTIVE.set(execution)


def deactivate(token: contextvars.Token) -> None:
    _ACTIVE.reset(token)


def active_product_coefficients(operands: Sequence[Sequence[int]], label: str) -> list[int] | None:
    execution = _ACTIVE.get()
    return None if execution is None else execution.product_coefficients(operands, label)


def active_lasso_product(factors: Sequence[int], label: str) -> int | None:
    execution = _ACTIVE.get()
    return None if execution is None else execution.lasso_product(factors, label)


def active_mle_fold(values: Sequence[int], challenges: Sequence[int], label: str) -> int | None:
    execution = _ACTIVE.get()
    return None if execution is None else execution.mle_fold(values, challenges, label)
