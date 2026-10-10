#!/usr/bin/env python3
"""Focused fixture for the native mechanism result's nested backend schema."""

from __future__ import annotations

import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_mechanism_ablation as runner  # noqa: E402


class NativeBackendReceiptTest(unittest.TestCase):
    def test_reads_cuda_attestations_from_native_backend_object(self) -> None:
        native_result = {
            "format": "zkgat_mechanism_result_v1",
            "backend": {
                "prover": "cuda",
                "device": 2,
                "verifier": "cpu",
                "actual_cuda": True,
                "cuda_ntt_forced": True,
                "device_name": "NVIDIA fixture device",
                "cuda_kernel_invocations_delta": 17,
                "cuda_stats_delta": {
                    "ntt_calls": 2,
                    "intt_calls": 2,
                    "msm_single_calls": 1,
                    "msm_batch_calls": 0,
                },
            },
            # The C++ serializer does not emit these top-level aliases. If
            # they reappear in another schema, they must not mask native data.
            "actual_cuda": False,
            "device": 99,
            "device_name": "stale alias",
            "cuda_kernel_invocations_delta": 0,
            "cuda_stats_delta": {},
        }
        normalized = runner.validate_cuda_backend_receipt(native_result, device=2)
        self.assertEqual(normalized["device_name"], "NVIDIA fixture device")
        self.assertEqual(normalized["cuda_kernel_invocations_delta"], 17)
        self.assertEqual(normalized["cuda_stats_delta"]["ntt_calls"], 2)
        self.assertIs(normalized["cuda_ntt_forced"], True)

    def test_rejects_top_level_only_cuda_attestation(self) -> None:
        with self.assertRaisesRegex(ValueError, "native backend receipt"):
            runner.validate_cuda_backend_receipt(
                {"actual_cuda": True, "cuda_ntt_forced": True, "device": 0}, device=0
            )


if __name__ == "__main__":
    unittest.main()
