"""Lightweight strict checks for the native CUDA proof-message contract."""
from __future__ import annotations

import copy
import pathlib
import unittest

from zkgpt_gat.backend import (
    gpu_proof_message_batch,
    recompute_gpu_proof_batch,
    validate_gpu_proof_state,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _operations() -> list[dict]:
    return [
        {"kind": "lasso_product_tree", "values": [1, 2, 3, 4], "beta": 7},
        {"kind": "sumcheck_product2", "operands": [[1, 2, 3, 4], [5, 6, 7, 8]], "challenges": [5, 6]},
        {
            "kind": "sumcheck_product3",
            "operands": [[1, 0, 1, 0], [2, 3, 4, 5], [6, 7, 8, 9]],
            "challenges": [3, 8],
        },
        {"kind": "hyrax_linear_eval_fold", "values": [9, 10, 11, 12], "challenges": [3, 8]},
    ]


class GPUBackendTests(unittest.TestCase):
    def test_reference_rejects_wrong_round_count(self) -> None:
        with self.assertRaises(ValueError):
            recompute_gpu_proof_batch([
                {"kind": "sumcheck_product2", "operands": [[1, 2], [3, 4]], "challenges": []}
            ])

    def test_cuda_messages_match_independent_reference(self) -> None:
        binary = ROOT / "build" / "zkgpt_cuda_native_prover"
        if not binary.is_file():
            self.skipTest("CUDA proof primitive binary is not built")
        operations = _operations()
        state = gpu_proof_message_batch(ROOT, operations, "unit-gpu-contract")
        expected = recompute_gpu_proof_batch(operations)
        self.assertEqual(state["operations"], expected["operations"])
        for key in (
            "field_words_processed",
            "primitive_counts",
            "lasso_grand_product_fr_hex",
            "matmul_sumcheck_fr_hex",
            "gkr_relation_fr_hex",
            "transcript_state_low256_hex",
            "transcript_accumulator_fr_hex",
        ):
            self.assertEqual(state[key], expected[key])
        valid, errors = validate_gpu_proof_state(state, expected_operations=operations)
        self.assertTrue(valid, errors)

    def test_valid_value_tamper_is_detected_with_reference(self) -> None:
        binary = ROOT / "build" / "zkgpt_cuda_native_prover"
        if not binary.is_file():
            self.skipTest("CUDA proof primitive binary is not built")
        operations = _operations()
        state = gpu_proof_message_batch(ROOT, operations, "unit-gpu-tamper")
        tampered = copy.deepcopy(state)
        value = int(tampered["operations"][1]["rounds"][0]["coefficients_fr_hex"][0], 16)
        tampered["operations"][1]["rounds"][0]["coefficients_fr_hex"][0] = f"{(value + 1) % (2**254):064x}"
        valid, errors = validate_gpu_proof_state(tampered, expected_operations=operations)
        self.assertFalse(valid)
        self.assertTrue(any("independent_reference" in error for error in errors), errors)

    def test_contract_rejects_missing_terminal_message(self) -> None:
        binary = ROOT / "build" / "zkgpt_cuda_native_prover"
        if not binary.is_file():
            self.skipTest("CUDA proof primitive binary is not built")
        state = gpu_proof_message_batch(ROOT, _operations(), "unit-gpu-structure")
        tampered = copy.deepcopy(state)
        del tampered["operations"][0]["grand_product_fr_hex"]
        valid, errors = validate_gpu_proof_state(tampered)
        self.assertFalse(valid)
        self.assertIn("operations[0].grand_product_not_fr_hex64", errors)


if __name__ == "__main__":
    unittest.main()
