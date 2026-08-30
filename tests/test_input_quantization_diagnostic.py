#!/usr/bin/env python3
"""Short regression tests for the non-mutating S_input=1 diagnostic."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_planetoid_gat import input_quantization_diagnostic  # noqa: E402


class InputQuantizationDiagnosticTests(unittest.TestCase):
    def test_reports_zero_non_integer_changed_and_range_ratios(self) -> None:
        values = np.asarray([0.0, 0.49, 1.0, -1.0, 2.4], dtype=np.float64)
        report = input_quantization_diagnostic(values, scale=1, range_bits=1)

        self.assertEqual(report["scale"], 1)
        self.assertAlmostEqual(report["zero_ratio"], 0.4)
        self.assertAlmostEqual(report["non_integer_ratio"], 0.4)
        self.assertAlmostEqual(report["changed_ratio"], 0.4)
        self.assertEqual(report["out_of_range_count"], 1)
        self.assertAlmostEqual(report["out_of_range_ratio"], 0.2)
        self.assertAlmostEqual(report["saturation_ratio"], 0.2)
        self.assertEqual(report["abs_input_quantiles"]["1.0"], 2.4)

    def test_diagnostic_does_not_change_scale_or_clip_values(self) -> None:
        values = np.asarray([-8.75, 0.25, 3.75], dtype=np.float64)
        report = input_quantization_diagnostic(values, scale=1, range_bits=2)

        self.assertEqual(report["max_abs_quantized"], 9.0)
        self.assertEqual(report["out_of_range_count"], 2)
        self.assertAlmostEqual(report["saturation_ratio"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
