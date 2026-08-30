#!/usr/bin/env python3
"""Short layered-schema regression checks.

The C++ runtime test is the primary behavior check: it calls the production
label, selector, checkpoint-parser, and shape-validation APIs with a small
synthetic manifest.  The source assertions below supplement it by guarding
the formal producer/consumer wiring.  This is not a proof test and does not
load a dataset, canonical checkpoint, or SRS.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def region(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


class LayeredSchemaRegression(unittest.TestCase):
    def test_production_runtime_schema_api(self) -> None:
        runtime = ROOT / "build" / "gatzk_layered_schema_test"
        self.assertTrue(
            runtime.is_file(),
            f"build the test-only production runtime first: {runtime}",
        )
        result = subprocess.run(
            [str(runtime)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )
        self.assertIn("layered schema runtime checks passed", result.stdout)

    def test_labels_keep_layer_and_head_identity(self) -> None:
        labels = read("src/protocol/schema_labels.cpp")
        self.assertRegex(
            labels,
            r'return "P_h_l" \+ std::to_string\(layer_index\)\s*'
            r'\+ "_h" \+ std::to_string\(local_head_index\)',
        )
        self.assertIn('return "P_Q_in_valid_l"', labels)
        self.assertIn('return "P_Q_d_h_valid_l"', labels)
        self.assertIn('return "P_Q_cat_valid_l"', labels)
        self.assertIn('return "P_Q_Nd_h_valid_l"', labels)
        self.assertIn('return "P_Q_cat_valid_out"', labels)
        self.assertIn('return "xi_cat_l"', labels)
        self.assertIn('return "y_cat_l"', labels)

        # A same-local-head fixture in two layers must produce two identities;
        # this guards against silently reintroducing a global-head alias.
        self.assertNotEqual("P_h_l0_h0", "P_h_l1_h0")
        self.assertNotEqual("P_Q_cat_valid_l1", "P_Q_cat_valid_out")

    def test_each_layer_selector_uses_its_real_support(self) -> None:
        prover = read("src/protocol/prover.cpp")
        formal_selector_block = region(
            prover,
            "if (context.model.has_real_multihead)\n        {\n            // Each hidden layer owns its own valid selectors.",
            "        add_public_poly(\n            context,\n            \"P_Q_N\"",
        )
        self.assertIn("hidden_q_in_selector_label(layer_index)", formal_selector_block)
        self.assertIn("hidden_q_d_h_selector_label(layer_index)", formal_selector_block)
        self.assertIn("hidden_q_cat_selector_label(layer_index)", formal_selector_block)
        self.assertIn("hidden_q_nd_h_selector_label(layer_index)", formal_selector_block)
        self.assertNotIn("max_in", formal_selector_block)
        self.assertIn("build_selector(layer.input_dim, in_size)", formal_selector_block)
        self.assertIn("build_selector(layer.shape.head_dim, d_size)", formal_selector_block)
        self.assertIn(
            "build_selector(layer.shape.head_count * layer.shape.head_dim, cat_size)",
            formal_selector_block,
        )
        self.assertIn(
            "build_selector(context.local.num_nodes * layer.shape.head_dim, edge_size)",
            formal_selector_block,
        )

        # The output selector is a distinct formal object and has output width
        # support, rather than reusing the final hidden-layer Q_cat.
        self.assertIn("output_q_cat_selector_label()", prover)
        self.assertIn(
            "build_selector(context.model.output_layer.input_dim, cat_size)",
            prover,
        )

    def test_challenge_and_dynamic_order_are_layered(self) -> None:
        challenges = read("src/protocol/challenges.cpp")
        modern_replay = region(
            challenges,
            "std::map<std::string, algebra::FieldElement> replay_challenges(",
            "\n        crypto::Transcript transcript(\"gatzkml\");",
        )
        # The initial public selector schedule absorbs one Q_Nd_h per layer.
        self.assertEqual(
            len(re.findall(
                r'absorb_public\(\s*transcript,\s*context,\s*'
                r'hidden_q_nd_h_selector_label\(layer_index\)\s*\)',
                modern_replay,
            )),
            1,
        )

        dynamic = region(
            challenges,
            "std::vector<std::string> dynamic_commitment_labels",
            "    std::vector<std::string> quotient_commitment_labels",
        )
        layer_loop = region(
            dynamic,
            "for (std::size_t layer_index = 0;",
            "        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size();",
        )
        self.assertIn("hidden_head_prefix(layer_index, head_index)", layer_loop)
        self.assertIn("append_concat_labels(", layer_loop)
        self.assertLess(
            layer_loop.index("append_concat_labels("),
            dynamic.index("for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size();"),
        )

    def test_formal_witness_and_edge_plan_are_nested(self) -> None:
        trace = read("src/protocol/trace.cpp")
        quotients = read("src/protocol/quotients.cpp")
        self.assertIn(
            "std::vector<std::vector<HiddenQuantizedWitness>>\n                hidden_head_quantized",
            trace,
        )
        self.assertGreaterEqual(
            trace.count("hidden_head_quantized\n                                             .at(layer_index)\n                                             .at(local_head_index)"),
            1,
        )
        self.assertIn(
            "std::vector<std::vector<HiddenHeadEvaluationConstants>> hidden_heads",
            quotients,
        )
        self.assertIn(
            "evaluation_plan->hidden_heads\n                                                 .at(layer_index)\n                                                 .at(local_head_index)",
            quotients,
        )

        formal_sources = [
            read("src/protocol/challenges.cpp"),
            read("src/protocol/proof_codec.cpp"),
            read("src/protocol/prover.cpp"),
            read("src/protocol/quotients.cpp"),
            read("src/protocol/schema.cpp"),
            read("src/protocol/trace.cpp"),
            read("src/protocol/verifier.cpp"),
        ]
        joined = "\n".join(formal_sources)
        for forbidden in (
            "flat_head_index",
            "planned_index",
            "canonical_hidden_head",
            "model.hidden_heads",
        ):
            self.assertNotIn(forbidden, joined)

    def test_codec_and_verifier_reconstruct_pair_objects(self) -> None:
        codec = read("src/protocol/proof_codec.cpp")
        verifier = read("src/protocol/verifier.cpp")
        for source in (codec, verifier):
            self.assertIn("meta.layer_id.value() - 1", source)
            self.assertIn("meta.head_id.value()", source)
            self.assertIn("hidden_weight_column_label(layer, head, column)", source)
        schema = read("src/protocol/schema.cpp")
        self.assertIn('.layer_id = layer + 1', schema)
        self.assertIn('.head_id = head', schema)

    def test_old_aliases_are_not_in_formal_registry(self) -> None:
        challenges = read("src/protocol/challenges.cpp")
        modern_replay = region(
            challenges,
            "std::map<std::string, algebra::FieldElement> replay_challenges(",
            "\n        crypto::Transcript transcript(\"gatzkml\");",
        )
        # Old names may occur in the unreachable compatibility branch, but not
        # in the modern formal replay/registry.
        for alias in (
            '"P_Q_proj_valid"',
            '"P_Q_d_valid"',
            '"P_H_cat"',
            '"P_cat"',
            '"xi_cat"',
            '"y_cat"',
        ):
            self.assertNotIn(alias, modern_replay)

        dynamic = region(
            challenges,
            "std::vector<std::string> dynamic_commitment_labels",
            "    std::vector<std::string> quotient_commitment_labels",
        )
        modern_dynamic = region(
            dynamic,
            'std::vector<std::string> labels = {\n            "P_H"',
            "        return labels;",
        )
        for alias in (
            '"P_Q_proj_valid"',
            '"P_Q_d_valid"',
            '"P_H_cat"',
            '"P_cat"',
            '"xi_cat"',
            '"y_cat"',
        ):
            self.assertNotIn(alias, modern_dynamic)

        model = read("src/model/gat.cpp")
        formal_loader = region(
            model,
            "ModelParameters load_checkpoint_bundle_parameters",
            "        // 输出层同样要求索引连续读取",
        )
        self.assertNotIn("global_head_index", formal_loader)
        self.assertIn("candidate.layer_index == layer.layer_index", formal_loader)
        self.assertIn("candidate.local_head_index == local_head_index", formal_loader)

        parser = region(
            model,
            "CheckpointHeadSpec parse_checkpoint_head_spec",
            "CheckpointOutputHeadSpec parse_checkpoint_output_head_spec",
        )
        self.assertNotIn("parse_required_size_t(entry, \"global_head_index\")", parser)

        for exporter in (
            read("scripts/export_torch_gat_bundle.py"),
            read("scripts/export_gat_checkpoint.py"),
        ):
            self.assertNotIn('"global_head_index"', exporter)

        checker = read("scripts/check_checkpoint_bundle.py")
        self.assertIn('"layer_index" in spec and "local_head_index" in spec', checker)
        self.assertIn("hidden head specs must use canonical layer/local pair order", checker)

    def test_attention_scalar_bias_is_fail_closed_and_output_bias_is_explicit(self) -> None:
        trainer = read("scripts/train_planetoid_gat.py")
        self.assertIn("self.attn_dst = nn.Linear(out_dim, 1, bias=False)", trainer)
        self.assertIn("self.attn_src = nn.Linear(out_dim, 1, bias=False)", trainer)
        self.assertIn("bias-enabled attention checkpoint is not accepted", trainer)

        model = read("src/model/gat.cpp")
        head_parser = region(
            model,
            "CheckpointHeadSpec parse_checkpoint_head_spec",
            "CheckpointOutputHeadSpec parse_checkpoint_output_head_spec",
        )
        output_parser = region(
            model,
            "CheckpointOutputHeadSpec parse_checkpoint_output_head_spec",
            "// 把各个维度乘起来",
        )
        self.assertNotIn("attn_dst_bias", head_parser)
        self.assertNotIn("attn_src_bias", head_parser)
        self.assertIn('parse_required_string(entry, "output_bias")', output_parser)
        forward = region(
            model,
            "HeadForwardTrace attention_head_forward_impl",
            "// 只读取 manifest 元数据",
        )
        self.assertNotIn("attn_dst_bias_fp", forward)
        self.assertNotIn("attn_src_bias_fp", forward)

        torch_exporter = read("scripts/export_torch_gat_bundle.py")
        self.assertIn("attention scalar bias is forbidden", torch_exporter)
        self.assertNotIn('"attn_dst_bias": dst_bias_name', torch_exporter)
        self.assertNotIn('"attn_src_bias": src_bias_name', torch_exporter)
        self.assertIn('result["output_bias"] = output_bias_name', torch_exporter)

        checker = read("scripts/check_checkpoint_bundle.py")
        self.assertIn("hidden head spec must not expose output_bias", checker)
        self.assertIn("legacy attention scalar bias fields are forbidden", checker)

    def test_scale_chain_consumers_are_layer_and_head_aware(self) -> None:
        config = read("src/util/config.cpp")
        self.assertIn('"gat_l3_scale_chain_v1"', config)
        self.assertIn("chain.s_h = {1, 16, 16};", config)
        self.assertIn("chain.s_h[layer + 1] != chain.s_cat[layer]", config)
        self.assertIn("chain.s_h.back() != output_input_scale", config)
        self.assertIn("chain.s_out_pre != chain.s_out_head", config)
        self.assertIn("pre-bias alignment SignedRoundDiv is therefore an identity", config)
        self.assertIn("shared ELU table scale equality", config)
        self.assertIn("chain.s_h_prime = {{16, 16, 16, 16, 16, 16, 16, 16},", config)
        self.assertIn("{256, 256, 256, 256, 256, 256, 256, 256}};", config)
        prover = read("src/protocol/prover.cpp")
        self.assertIn("scales.s_w_hidden.at(layer_index).at(head_index)", prover)
        self.assertIn("scales.s_a_src_hidden.at(layer_index).at(head_index)", prover)
        self.assertIn("scales.s_w_output.at(head_index)", prover)
        trace = read("src/protocol/trace.cpp")
        self.assertIn("aggregate_numerator * aggregate_input_scale", trace)
        self.assertIn("sum_raw[row]) * h_prime_scale", trace)
        quotients = read("src/protocol/quotients.cpp")
        self.assertIn("aggregate_pre_scale * h_agg_num_flat_z", quotients)
        self.assertIn("h_prime_scale * sum_flat_z", quotients)
        self.assertIn("output_pre_scale * y_agg_num_flat", quotients)
        self.assertIn("output_y_prime_scale * sum_flat", quotients)

    def test_quantized_reference_is_independent_and_reported_from_real_match(self) -> None:
        trace = read("src/protocol/trace.cpp")
        reference = region(
            trace,
            "Matrix independent_quantized_reference_forward(",
            "std::shared_ptr<const CachedForwardArtifacts> cached_forward_artifacts(",
        )
        self.assertIn("context.config.quantization", reference)
        self.assertIn("project_raw_matrix(", reference)
        self.assertIn("raw_matvec_no_bias(", reference)
        self.assertIn("quantize_lrelu_raw(", reference)
        self.assertIn("quantize_exp_neg_raw(", reference)
        self.assertIn("signed_round_div_nearest(", reference)
        self.assertIn("quantize_elu_raw(", reference)
        self.assertIn("head.output_bias_fp", reference)
        self.assertNotIn("quantize_hidden_witness_cpu(", reference)
        self.assertNotIn("quantize_output_witness_cpu(", reference)
        self.assertNotIn("forward_artifacts", reference)

        artifacts = region(
            trace,
            "struct CachedForwardArtifacts",
            "std::string trace_cache_prefix(",
        )
        self.assertIn("Matrix reference_quantized_logits", artifacts)
        self.assertIn("independent_quantized_reference_forward(", trace)
        self.assertNotIn(
            "forward_artifacts->output_y_average,\n                forward_artifacts->output_y_average",
            trace,
        )
        self.assertIn(
            "quantized_reference_matches_secure(",
            trace,
        )

        main = read("src/main.cpp")
        self.assertIn("const bool reference_matrix_match = trace.reference_consistency_passed", main)
        self.assertIn("metrics.reference_consistency_passed = reference_consistent", main)
        self.assertIn("reference_matrix_match && reference_payload_audit", main)
        self.assertIn("final_formal_run_accepted(", main)
        self.assertIn("return final_accepted", main)

        exporter = read("src/protocol/artifact_export.cpp")
        self.assertIn(
            '{"reference_consistency_passed", metrics.reference_consistency_passed ? "true" : "false"}',
            exporter,
        )
        self.assertNotIn(
            '{"reference_consistency_passed", verified ? "true" : "false"}',
            exporter,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
