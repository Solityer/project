from __future__ import annotations

import copy
import inspect
import pathlib
import tempfile
import types
import unittest
from dataclasses import replace

import numpy as np

from synthetic_e2e_support import synthetic_graph, synthetic_model  # type: ignore
from zkgpt_gat.bn254 import BN254_FQ_MODULUS, G1Point
from zkgpt_gat.bound_ledger import build_integer_bound_ledger
from zkgpt_gat.constants import ENCODING_ID
from zkgpt_gat.field import BN254_FR_MODULUS, parse_canonical_decimal
from zkgpt_gat.forward import _elu_table_value, _exp_map_fixed, canonical_input_integers, forward_full
from zkgpt_gat.native_hyrax import (
    MASK_DOMAIN,
    MASK_PROTOCOL_ID,
    MASK_VERSION,
    NativeHyrax,
    _point,
    verify_open_payload,
)
from zkgpt_gat.privacy import PrivacyProfileError, _append_evidence
from zkgpt_gat.proof import prepare_input_segments_and_layout
from zkgpt_gat.quantization import (
    CANONICAL_QUANT_CFG_ID,
    canonical_scale_profile,
    scale_chain_from_profile,
    validate_scale_profile,
)
from zkgpt_gat.runner import _same_quantization
from zkgpt_gat.registry_layout import build_model_registry_layout
from zkgpt_gat.transcript import Transcript
from zkgpt_gat.util import digest_json
from zkgpt_gat.resources import validate_static_table_descriptor
from zkgpt_gat.verifier_context import (
    _canonical_static_table_registry,
    build_verifier_context,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProtocolSemanticTests(unittest.TestCase):
    @staticmethod
    def _custom_profile() -> dict[str, int]:
        profile = canonical_scale_profile()
        # This is a complete, internally closed fixture profile.  It exercises
        # non-default S_input while retaining a bounded function-table domain.
        profile.update(
            input=8,
            hidden_weight=8,
            hidden_attention=4,
            output_weight=8,
            output_attention=4,
            output_bias=8,
            exp_output=8,
            elu_input=8,
            elu_output=8,
            aggregate_output=8,
            output_head=8,
        )
        return profile

    @staticmethod
    def _ledger() -> dict[str, object]:
        bounds = {
            name: {"B_W": 2, "B_a_src": 2, "B_a_dst": 2, "B_b": 2}
            for name in ("hidden.l1.h0", "hidden.l2.h0", "output.h0")
        }
        return build_integer_bound_ledger(
            mode="nearest_ties_away",
            quant_cfg_id=CANONICAL_QUANT_CFG_ID,
            authenticated_quant_scales=canonical_scale_profile(),
            registered_model_bounds=bounds,
            hidden_profile=[
                {"layer_index": 0, "head_count": 1, "head_dim": 2},
                {"layer_index": 1, "head_count": 1, "head_dim": 2},
            ],
            d_in=[2, 2, 2],
            classes=1,
            k_out=1,
            max_degree=2,
            feature_bound=8,
            node_count=2,
            range_bits=4,
        )

    def test_scale_chain_derives_h_prime_y_prime_and_rejects_wrong_denominator(self) -> None:
        profile = self._custom_profile()
        self.assertEqual(validate_scale_profile(profile), profile)
        chain = scale_chain_from_profile(
            profile,
            hidden_profile=[
                {"head_count": 8, "head_dim": 8},
                {"head_count": 8, "head_dim": 8},
            ],
            k_out=2,
            quant_cfg_id="test-authenticated-chain",
        )
        self.assertEqual(chain["s_h_prime_hidden"], [[64] * 8, [64] * 8])
        self.assertEqual(chain["s_y_prime"], [64, 64])
        wrong = copy.deepcopy(profile)
        wrong["exp_input"] = 128
        with self.assertRaisesRegex(ValueError, "input scales differ"):
            validate_scale_profile(wrong)
        # The function relation itself consumes the authenticated denominator.
        self.assertNotEqual(
            _exp_map_fixed(128, input_scale=256, output_scale=16),
            _exp_map_fixed(128, input_scale=128, output_scale=16),
        )
        self.assertNotEqual(
            _elu_table_value(-8, input_scale=16, output_scale=16),
            _elu_table_value(-8, input_scale=8, output_scale=16),
        )

    def test_public_input_quantization_uses_nonzero_authenticated_scale(self) -> None:
        _dataset, graph = synthetic_graph()
        graph = replace(graph, features=np.asarray([[0.125]], dtype=np.float64))
        model = synthetic_model()
        model = replace(
            model,
            quant_cfg_id=CANONICAL_QUANT_CFG_ID,
            quant_scales=canonical_scale_profile(),
        )
        np.testing.assert_array_equal(canonical_input_integers(graph, model), [[2]])

    def test_external_static_table_rebuild_resolves_authenticated_profile_first(self) -> None:
        descriptor = validate_static_table_descriptor(ROOT)
        # Keep this direct registry-rebuild regression test genuinely tiny;
        # the formal domains are still derived by the same production ledger
        # path, while no large Decimal table is materialized in unit tests.
        bounds = {
            name: {"B_W": 1, "B_a_src": 1, "B_a_dst": 1, "B_b": 1}
            for name in ("hidden.l1.h0", "hidden.l2.h0", "output.h0")
        }
        ledger = build_integer_bound_ledger(
            mode="nearest_ties_away",
            quant_cfg_id=CANONICAL_QUANT_CFG_ID,
            authenticated_quant_scales=canonical_scale_profile(),
            registered_model_bounds=bounds,
            hidden_profile=[
                {"layer_index": 0, "head_count": 1, "head_dim": 1},
                {"layer_index": 1, "head_count": 1, "head_dim": 1},
            ],
            d_in=[1, 1, 1],
            classes=1,
            k_out=1,
            max_degree=1,
            feature_bound=1,
            node_count=1,
            range_bits=2,
        )
        metadata = {
            "quant_cfg_id": CANONICAL_QUANT_CFG_ID,
            "rounding_mode": "nearest_ties_away",
            "static_table_id": "gat_l3_static_tables_v1",
            "static_table_descriptor_schema": descriptor["schema"],
            "static_table_content_digest": descriptor["content_digest"],
            "static_table_descriptor_digest": descriptor["descriptor_digest"],
            "encoding_id": ENCODING_ID,
            "integer_bound_ledger": ledger,
            "integer_bound_ledger_digest": digest_json(ledger),
            "dim_cfg": {"B": 2},
        }
        registry = _canonical_static_table_registry(metadata)
        self.assertEqual(set(registry), {"T_LReLU", "T_ELU", "T_exp", "T_range"})
        self.assertTrue(all("table_digest" in table for table in registry.values()))

    def test_shape_only_registration_layout_matches_trace_layout(self) -> None:
        dataset, graph = synthetic_graph()
        model = synthetic_model()
        trace = forward_full(graph, model)
        config = types.SimpleNamespace(
            project_root=ROOT,
            range_bits=4,
            rounding_mode=model.rounding_mode,
        )
        _segments, trace_layout, _padded = prepare_input_segments_and_layout(
            config, dataset, graph, model, trace
        )
        registration_layout, _ledger = build_model_registry_layout(
            config, dataset, graph, model
        )
        self.assertEqual(trace_layout, registration_layout)

    def test_ledger_contains_function_ranges_m_range_and_field_fit(self) -> None:
        ledger = self._ledger()
        families = ledger["range_family_policy"]["families"]
        required = {
            "hidden.l1.h0.M_range",
            "hidden.l1.h0.LeakyReLU.x_range",
            "hidden.l1.h0.LeakyReLU.y_range",
            "hidden.l1.h0.ExpMap.x_range",
            "hidden.l1.h0.ExpMap.y_range",
            "hidden.l1.h0.ELU.x_range",
            "hidden.l1.h0.ELU.y_range",
            "output.h0.M_range",
            "output.h0.LeakyReLU.x_range",
            "output.h0.LeakyReLU.y_range",
            "output.h0.ExpMap.x_range",
            "output.h0.ExpMap.y_range",
        }
        self.assertTrue(required <= set(families))
        for policy in families.values():
            self.assertLess(int(policy["encoded_bound"]), BN254_FR_MODULUS // 2)
            self.assertLessEqual(
                int(policy["encoded_bound"]) + 1,
                1 << int(policy["bit_width"]),
            )
        for bounds in ledger["function_bounds"].values():
            self.assertLess(int(bounds["max_packed"]), BN254_FR_MODULUS // 2)

    def test_same_quantization_requires_authenticated_profile_and_chain(self) -> None:
        profile = canonical_scale_profile()
        chain = scale_chain_from_profile(
            profile,
            hidden_profile=[
                {"head_count": 8, "head_dim": 8},
                {"head_count": 8, "head_dim": 8},
            ],
            k_out=2,
            quant_cfg_id=CANONICAL_QUANT_CFG_ID,
        )
        config = types.SimpleNamespace(entries={"quant_cfg_id": CANONICAL_QUANT_CFG_ID})
        binding = {
            "authenticated_quant_scales": profile,
            "quant_scale_chain": chain,
        }
        metadata = {
            "quant_cfg_id": CANONICAL_QUANT_CFG_ID,
            "authenticated_quant_scales": profile,
            "quant_scale_chain": chain,
        }
        self.assertTrue(_same_quantization(config, binding, metadata))
        tampered = dict(metadata, authenticated_quant_scales=dict(profile, input=8))
        self.assertFalse(_same_quantization(config, binding, tampered))
        self.assertFalse(_same_quantization(config, {"quant_scale_chain": chain}, metadata))

    def test_canonical_decimal_aliases_rejected_across_g1_ipa_and_privacy(self) -> None:
        wire = G1Point.generator().to_json()
        self.assertEqual(G1Point.from_json(wire), G1Point.generator())
        for bad in (
            dict(wire, x="01"),
            dict(wire, x="+1"),
            dict(wire, x=str(BN254_FQ_MODULUS + 1)),
            {"x": "1", "y": "2", "infinity": True},
        ):
            with self.assertRaises(ValueError):
                G1Point.from_json(bad)
            with self.assertRaises(ValueError):
                _point(bad)
        self.assertEqual(parse_canonical_decimal("0"), 0)
        for bad in ("01", "+1", "-0", " 1", "１"):
            with self.assertRaises(ValueError):
                parse_canonical_decimal(bad)

        with tempfile.TemporaryDirectory(prefix="zkgpt-decimal-wire-") as tmp:
            root = pathlib.Path(tmp)
            hyrax = NativeHyrax(root)
            registration = hyrax.commit_vector("wire", [3, 5], root)
            state = hyrax.open_registered(
                "wire",
                [3, 5],
                [7],
                registration,
                root,
                transcript=Transcript("decimal-wire"),
            )
            tampered = copy.deepcopy(state.proof)
            tampered["ipa"]["final_a"] = "01"
            result = verify_open_payload(
                "wire",
                tampered,
                [7],
                {key: value for key, value in registration.items() if key != "blind"},
                2,
                transcript=Transcript("decimal-wire"),
            )
            self.assertEqual(result["status"], "VERIFY_FAILED")

        evidence: list[dict[str, object]] = []
        mask_payload = {
            "mask_protocol_id": MASK_PROTOCOL_ID,
            "mask_domain": MASK_DOMAIN,
            "mask_version": MASK_VERSION,
            "mask_commitment": wire,
            "mask_challenge": "1",
        }
        _append_evidence(
            evidence,
            component="lasso",
            path="wire",
            role="opening",
            active_label="wire",
            payload=mask_payload,
            mask=True,
        )
        bad_payload = copy.deepcopy(mask_payload)
        bad_payload["mask_challenge"] = "01"
        with self.assertRaises(PrivacyProfileError):
            _append_evidence(
                [],
                component="lasso",
                path="wire",
                role="opening",
                active_label="wire",
                payload=bad_payload,
                mask=True,
            )

    def test_external_context_source_has_no_checkpoint_loader_call(self) -> None:
        source = inspect.getsource(build_verifier_context)
        self.assertNotIn("_manifest(", source)
        self.assertNotIn("tensors.npz", source)
        self.assertNotIn("tensors.txt", source)


if __name__ == "__main__":
    unittest.main()
