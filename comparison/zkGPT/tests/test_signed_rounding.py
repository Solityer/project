from __future__ import annotations

import copy
import unittest

import numpy as np

from zkgpt_gat.forward import _signed_round_div_trace
from zkgpt_gat.proof import (
    _integer_bound_ledger,
    _native_division_specs,
    _verify_integer_bound_ledger,
)
from zkgpt_gat.rounding import (
    ROUNDING_MODE_NEAREST,
    ROUNDING_MODE_TOWARD_ZERO,
    division_bound_record,
)
from zkgpt_gat.verifier_context import _native_division_records

from synthetic_e2e_support import synthetic_graph, synthetic_model


class SignedRoundDivTests(unittest.TestCase):
    def test_both_modes_zero_exact_ties_and_non_ties(self) -> None:
        for mode in (ROUNDING_MODE_NEAREST, ROUNDING_MODE_TOWARD_ZERO):
            trace = _signed_round_div_trace(
                np.asarray([0, 8, -8, 2, -2, 1, -1, 3, -3], dtype=object),
                np.asarray([4, 4, 4, 4, 4, 4, 4, 1, 1], dtype=object),
                mode,
            )
            self.assertEqual(trace.rounding_mode, mode)
            self.assertEqual(int(trace.y[0]), 0)
            self.assertEqual(int(trace.y[1]), 2)
            self.assertEqual(int(trace.y[2]), -2)
            self.assertEqual(int(trace.y[3]), 1 if mode == ROUNDING_MODE_NEAREST else 0)
            self.assertEqual(int(trace.y[4]), -1 if mode == ROUNDING_MODE_NEAREST else 0)
            self.assertEqual(int(trace.y[5]), 0)
            self.assertEqual(int(trace.y[6]), 0)
            self.assertEqual(int(trace.y[7]), 3)
            self.assertEqual(int(trace.y[8]), -3)
            for index in range(trace.y.size):
                a, d, q, r, delta = (int(getattr(trace, field).ravel()[index]) for field in ("a", "d", "q", "r", "delta"))
                if mode == ROUNDING_MODE_NEAREST:
                    self.assertEqual(2 * a + d, 2 * d * q + r)
                    self.assertEqual(r + delta, 2 * d - 1)
                else:
                    self.assertEqual(a, d * q + r)
                    self.assertEqual(r + delta, d - 1)
                self.assertGreaterEqual(r, 0)
                self.assertGreaterEqual(delta, 0)

    def test_d_one_and_unknown_mode_fail_closed(self) -> None:
        for mode in (ROUNDING_MODE_NEAREST, ROUNDING_MODE_TOWARD_ZERO):
            trace = _signed_round_div_trace(np.asarray([-9, 0, 9], dtype=object), np.asarray([1, 1, 1], dtype=object), mode)
            self.assertEqual(trace.y.tolist(), [-9, 0, 9])
            for index in range(trace.y.size):
                self.assertEqual(int(trace.d[index]), 1)
                self.assertEqual(int(trace.r[index]), 1 if mode == ROUNDING_MODE_NEAREST else 0)
                self.assertEqual(int(trace.delta[index]), 0)
        with self.assertRaises(ValueError):
            _signed_round_div_trace(np.asarray([1], dtype=object), np.asarray([2], dtype=object), "implicit_default")

    def test_odd_denominator_reachable_remainder_boundaries(self) -> None:
        # For odd d=5, |a|=2 is below the nearest-rounding half threshold;
        # q=0 therefore reaches the certified r=2*d-1 boundary exactly.
        nearest = _signed_round_div_trace(
            np.asarray([-2, 2], dtype=object),
            np.asarray([5, 5], dtype=object),
            ROUNDING_MODE_NEAREST,
        )
        self.assertEqual(nearest.y.tolist(), [0, 0])
        self.assertEqual(nearest.r.tolist(), [9, 9])
        self.assertEqual(nearest.delta.tolist(), [0, 0])
        for index in range(nearest.y.size):
            self.assertEqual(2 * int(nearest.a[index]) + 5, 2 * 5 * int(nearest.q[index]) + int(nearest.r[index]))
            self.assertEqual(int(nearest.r[index]) + int(nearest.delta[index]), 2 * 5 - 1)

        # Toward-zero has the corresponding d-1 boundary; |a|=4 reaches it
        # while q remains zero.
        toward_zero = _signed_round_div_trace(
            np.asarray([-4, 4], dtype=object),
            np.asarray([5, 5], dtype=object),
            ROUNDING_MODE_TOWARD_ZERO,
        )
        self.assertEqual(toward_zero.y.tolist(), [0, 0])
        self.assertEqual(toward_zero.r.tolist(), [4, 4])
        self.assertEqual(toward_zero.delta.tolist(), [0, 0])
        for index in range(toward_zero.y.size):
            self.assertEqual(int(toward_zero.a[index]), 5 * int(toward_zero.q[index]) + int(toward_zero.r[index]))
            self.assertEqual(int(toward_zero.r[index]) + int(toward_zero.delta[index]), 5 - 1)

    def test_native_and_external_gkr_descriptor_parity(self) -> None:
        for mode in (ROUNDING_MODE_NEAREST, ROUNDING_MODE_TOWARD_ZERO):
            native = _native_division_specs("toy.SignedRoundDiv", 2, mode)
            external = _native_division_records("toy.SignedRoundDiv", 2, mode)
            self.assertEqual([(kind, spec.descriptor()) for kind, spec in native], [(item["kind"], item["spec"]) for item in external])

    def test_v2_ledger_covers_scale_alignment_and_rejects_tampering(self) -> None:
        model = synthetic_model()
        graph = synthetic_graph()[1]
        metadata = {
            "rounding_mode": ROUNDING_MODE_NEAREST,
            "quant_cfg_id": model.quant_cfg_id,
            "dim_cfg": {
                "N": graph.num_nodes,
                "C": model.class_count,
                "K_out": model.k_out,
                "B": 8,
                "d_in": [1, 1],
                "hidden_profile": [{"head_count": 1, "head_dim": 1}],
            },
        }
        metadata["integer_bound_ledger"] = _integer_bound_ledger(metadata, model, graph)
        metadata["integer_bound_ledger_digest"] = __import__("zkgpt_gat.util", fromlist=["digest_json"]).digest_json(metadata["integer_bound_ledger"])
        self.assertTrue(_verify_integer_bound_ledger(metadata))
        self.assertIn("output.h0.SignedRoundDiv.scale_alignment", [item["id"] for item in metadata["integer_bound_ledger"]["signed_round_divisions"]])
        for mutate in ("mode", "bound", "cardinality", "missing_scale", "digest"):
            tampered = copy.deepcopy(metadata)
            ledger = tampered["integer_bound_ledger"]
            if mutate == "mode":
                ledger["rounding_mode"] = ROUNDING_MODE_TOWARD_ZERO
            elif mutate == "bound":
                ledger["signed_round_divisions"][0]["B_r"] += 1
            elif mutate == "cardinality":
                ledger["division_cardinality"][ledger["division_instance_ids"][0]] += 1
            elif mutate == "missing_scale":
                ledger["signed_round_divisions"] = [item for item in ledger["signed_round_divisions"] if "scale_alignment" not in item["id"]]
            else:
                tampered["integer_bound_ledger_digest"] = "0" * 64
            self.assertFalse(_verify_integer_bound_ledger(tampered), mutate)

    def test_rounding_bound_actual_sum_is_not_independent_sum(self) -> None:
        record = division_bound_record("x", ROUNDING_MODE_NEAREST, a_bound=8, d_bound=5, d_min=5, count=1, source="x", numerator_source="a", denominator_source="d")
        self.assertEqual(record["B_r"], 9)
        self.assertEqual(record["B_delta"], 9)
        self.assertEqual(record["B_r_plus_delta"], 9)
        self.assertEqual(record["B_r_plus_delta_independent"], 18)


if __name__ == "__main__":
    unittest.main()
