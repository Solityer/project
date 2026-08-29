from __future__ import annotations

import hashlib
import unittest

from zkgpt_gat.gpu_execution import (
    GPU_EXECUTION_SCHEMA,
    GPU_EXECUTION_SEMANTICS,
    gpu_execution_transcript_binding,
    validate_gpu_execution_evidence,
)
from zkgpt_gat.backend import _GPU_KERNEL
from zkgpt_gat.util import digest_json


def _invocation(label: str, component: str) -> dict[str, object]:
    fr = "0" * 63 + "1"
    test_binary_sha256 = hashlib.sha256(b"zkGPT-test-native-bn254-fr-batch-v3").hexdigest()
    if component == "sumcheck_product":
        kind = "sumcheck_product2_fold"
        slots = {
            "terminal_value_fr": 1,
            "rounds": 2,
            "challenge_fr_per_round": 1,
            "folded_claim_fr_per_round": 1,
            "coefficients_fr_per_round": 3,
        }
        operand_count = 2
        primitive_counts = {"lasso": 0, "matmul_sumcheck": 2, "gkr_relation": 2, "transcript": 1}
        words = 8
    elif component == "lasso_grand_product":
        kind = "lasso_product_tree"
        slots = {"terminal_value_fr": 1, "grand_product_fr": 1, "rounds": 0}
        operand_count = 1
        primitive_counts = {"lasso": 4, "matmul_sumcheck": 0, "gkr_relation": 0, "transcript": 1}
        words = 4
    else:
        kind = "hyrax_linear_eval_fold"
        slots = {
            "terminal_value_fr": 1,
            "rounds": 2,
            "challenge_fr_per_round": 1,
            "folded_claim_fr_per_round": 1,
            "coefficients_per_round": 0,
        }
        operand_count = 1
        primitive_counts = {"lasso": 0, "matmul_sumcheck": 0, "gkr_relation": 0, "transcript": 1}
        words = 4
    item: dict[str, object] = {
        "label": label,
        "component": component,
        "shape": [2, 8],
        "operation_kind": kind,
        "operation_input_count": 4,
        "operation_padded_count": 4,
        "operation_operand_count": operand_count,
        "operation_complete": True,
        "operation_round_count": slots["rounds"],
        "operation_output_slots": slots,
        "device_name": "test-gpu",
        "gpu_uuid": "00" * 16,
        "device_uuid": "00" * 16,
        "backend": "gpu",
        "protocol": "zkgpt/native-bn254-fr-batch-v1",
        "message_schema": "zkGPT_gpu_proof_messages_v3",
        "wire_version": "ZKGPT_GPU_BATCH_V3",
        "cuda_kernel": _GPU_KERNEL,
        "cuda_kernel_version": _GPU_KERNEL,
        "binary_sha256": test_binary_sha256,
        "build_identity": digest_json({"kernel": _GPU_KERNEL, "binary_sha256": test_binary_sha256}),
        "cuda_event_timing": "cudaEventElapsedTime",
        "cuda_event_elapsed_ms": 1.0,
        "kernel_time_ms": 1.0,
        "memory_copy_h2d_ms": 0.1,
        "memory_copy_d2h_ms": 0.1,
        "synchronization_time_ms": 0.1,
        "field_words_processed": words,
        "primitive_counts": primitive_counts,
        "cuda_kernel_executed": True,
        "cpu_fallback": False,
    }
    item["public_message_digest"] = digest_json(item)
    return item


def _evidence() -> dict[str, object]:
    invocations = [
        _invocation("sumcheck", "sumcheck_product"),
        _invocation("lasso", "lasso_grand_product"),
        _invocation("hyrax", "hyrax_mle_fold"),
    ]
    payload: dict[str, object] = {
        "schema": GPU_EXECUTION_SCHEMA,
        "semantic_id": GPU_EXECUTION_SEMANTICS,
        "primitive_contract": "zkGPT_gpu_proof_messages_v3",
        "wire_version": "ZKGPT_GPU_BATCH_V3",
        "cuda_kernel": "zkgpt_native_bn254_fr_batch_v3",
        "backend": "gpu",
        "session_nonce": "01" * 32,
        "final_transcript_digest": "02" * 32,
        "cuda_kernel_executed": True,
        "cpu_fallback": False,
        "proof_hot_path": True,
        "invocation_count": len(invocations),
        "components": ["hyrax_mle_fold", "lasso_grand_product", "sumcheck_product"],
        "invocations": invocations,
    }
    payload["transcript_binding_digest"] = digest_json(gpu_execution_transcript_binding(payload))
    payload["execution_digest"] = digest_json(payload)
    return payload


class GPUExecutionEvidenceTests(unittest.TestCase):
    def test_strict_evidence_accepts_all_native_components(self) -> None:
        evidence = _evidence()
        valid, errors = validate_gpu_execution_evidence(
            evidence, session_nonce="01" * 32,
            final_transcript_digest="02" * 32,
        )
        self.assertTrue(valid, errors)

    def test_tampered_or_probe_only_evidence_is_rejected(self) -> None:
        evidence = _evidence()
        evidence["invocations"] = list(evidence["invocations"][:-1])
        valid, errors = validate_gpu_execution_evidence(evidence)
        self.assertFalse(valid)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
