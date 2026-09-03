#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::model::CheckpointBundleInfo;
    using gatzk::model::CheckpointHeadSpec;
    using gatzk::model::CheckpointLayerInfo;
    using gatzk::model::CheckpointOutputHeadSpec;
    using gatzk::model::HiddenLayerShape;
    using gatzk::util::HiddenLayerProfile;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void expect_selector_support(
        std::size_t valid_length,
        std::size_t domain_size,
        const std::string &label)
    {
        const auto selector = gatzk::protocol::build_selector(valid_length, domain_size);
        expect(selector.size() == domain_size, label + " selector domain size mismatch");
        for (std::size_t index = 0; index < domain_size; ++index)
        {
            const auto expected = index < valid_length
                ? FieldElement::one()
                : FieldElement::zero();
            expect(selector[index] == expected, label + " selector support mismatch");
        }
    }

    CheckpointBundleInfo synthetic_info()
    {
        CheckpointBundleInfo info;
        info.family_schema = "multi_layer_multi_head";
        info.output_average_rule = "per_head_bias_then_arithmetic_mean";
        info.dataset = "cora";
        info.input_preprocess_id = "row_normalize";
        info.model_arch_id = "synthetic_layered_arch";
        info.model_param_id = "synthetic_layered_params";
        info.quant_cfg_id = "synthetic_quant";
        info.quantization_scale_chain = "synthetic_chain";
        info.layer_count = 3;
        info.d_in_profile = {5, 6};
        info.hidden_profile = {{2, 3}, {2, 3}};
        info.hidden_layers = {
            CheckpointLayerInfo{.layer_index = 0, .input_dim = 5, .shape = HiddenLayerShape{2, 3}},
            CheckpointLayerInfo{.layer_index = 1, .input_dim = 6, .shape = HiddenLayerShape{2, 3}},
        };
        for (std::size_t layer_index = 0; layer_index < 2; ++layer_index)
        {
            for (std::size_t local_head_index = 0; local_head_index < 2; ++local_head_index)
            {
                CheckpointHeadSpec spec;
                spec.layer_index = layer_index;
                spec.local_head_index = local_head_index;
                info.hidden_head_specs.push_back(spec);
            }
        }
        info.output_head_count = 2;
        info.output_head_specs = {
            CheckpointOutputHeadSpec{.head_index = 0},
            CheckpointOutputHeadSpec{.head_index = 1},
        };
        return info;
    }

    void write_manifest(
        const std::filesystem::path &path,
        bool omit_head_coordinates = false,
        bool forbidden_attention_bias = false,
        bool hidden_output_bias = false,
        const char *unexpected_field = nullptr,
        const char *family_schema = "multi_layer_multi_head",
        const char *omit_field = nullptr,
        const char *duplicate_field = nullptr)
    {
        std::ofstream manifest(path);
        expect(manifest.good(), "unable to create synthetic manifest");
        const auto canonical_chain =
            gatzk::util::canonical_quantization_scale_chain(
                "gat_l3_scale_chain",
                std::vector<HiddenLayerProfile>{{8, 8}, {8, 8}},
                2)
                .canonical_serialization();
        manifest << R"({
  "checkpoint_prefix": "synthetic_checkpoint.pt",
  "family_schema": ")"
                  << family_schema
                  << R"(",
)";
        if (unexpected_field != nullptr)
        {
            manifest << "  \"" << unexpected_field << "\": 1,\n";
        }
        manifest << R"(  "output_average_rule": "per_head_bias_then_arithmetic_mean",
  "dataset": "cora",
  "input_preprocess_id": "row_normalize",
  "model_arch_id": "synthetic_layered_arch",
  "model_param_id": "synthetic_layered_params",
  "quant_cfg_id": "gat_l3_scale_chain",
  "quantization_scale_chain": ")"
                  << canonical_chain
                  << R"(",
  "static_table_id": "synthetic_tables",
  "degree_bound_id": "synthetic_degree",
)"
                  << ((omit_field == nullptr || std::string(omit_field) != "batching_rule")
                          ? "  \"batching_rule\": \"whole_graph_single\",\n"
                          : "")
                  << R"(  "subgraph_rule": "whole_graph",
)"
                  << ((duplicate_field != nullptr)
                          ? std::string("  \"") + duplicate_field + "\": 1,\n"
                          : std::string{})
                  << R"(  "self_loop_rule": "per_node",
  "edge_sort_rule": "edge_gid_then_dst_stable",
  "label_format": "class_index",
  "loss": "CrossEntropyLoss",
  "split_format": "node_masks",
  "task_type": "transductive_node_classification",
  "report_unit": "node",
  "selection_metric": "validation_loss",
  "tensor_count": 0,
  "tensor_index": {},
  "L": 3,
  "K_out": 2,
  "C": 1,
  "d_in_profile": [5, 64],
  "hidden_layers": [
    {"layer_index": 0, "input_dim": 5, "head_count": 8, "head_dim": 8},
    {"layer_index": 1, "input_dim": 64, "head_count": 8, "head_dim": 8}
  ],
  "hidden_profile": [
    {"head_count": 8, "head_dim": 8},
    {"head_count": 8, "head_dim": 8}
  ],
  "hidden_head_specs": [
)";
        for (std::size_t layer_index = 0; layer_index < 2; ++layer_index)
        {
            for (std::size_t local_head_index = 0; local_head_index < 8; ++local_head_index)
            {
                if (layer_index != 0 || local_head_index != 0)
                {
                    manifest << ",\n";
                }
                manifest << "    {";
                if (!omit_head_coordinates)
                {
                    manifest << "\"layer_index\": " << layer_index
                             << ", \"local_head_index\": " << local_head_index << ", ";
                }
                const auto tensor_prefix =
                    "hidden_l" + std::to_string(layer_index) +
                    "_h" + std::to_string(local_head_index);
                manifest << "\"seq_kernel\": \"" << tensor_prefix << "/seq\","
                         << " \"attn_dst_kernel\": \"" << tensor_prefix << "/dst\","
                         << " \"attn_src_kernel\": \"" << tensor_prefix << "/src\"";
                if (forbidden_attention_bias)
                {
                    manifest << ", \"attn_dst_bias\": \"db\"";
                }
                if (hidden_output_bias)
                {
                    manifest << ", \"output_bias\": \"" << tensor_prefix
                             << "/output_bias\"";
                }
                manifest << "}";
            }
        }
        manifest << R"(
  ],
  "output_head_specs": [
    {"head_index": 0, "seq_kernel": "os", "attn_dst_kernel": "odk", "attn_src_kernel": "osk", "output_bias": "oob"},
    {"head_index": 1, "seq_kernel": "os1", "attn_dst_kernel": "odk1", "attn_src_kernel": "osk1", "output_bias": "oob1"}
  ]
}
)";
    }

    void write_tensor(
        std::ofstream &output,
        const std::string &name,
        const std::vector<std::size_t> &shape)
    {
        std::size_t count = 1;
        for (const auto dimension : shape)
        {
            count *= dimension;
        }
        output << "TENSOR " << name << ' ' << shape.size();
        for (const auto dimension : shape)
        {
            output << ' ' << dimension;
        }
        output << ' ' << count << '\n';
        for (std::size_t index = 0; index < count; ++index)
        {
            if (index != 0)
            {
                output << ' ';
            }
            output << '0';
        }
        output << '\n';
    }

    void write_tensor_dump(
        const std::filesystem::path &path,
        bool include_unreferenced_extra)
    {
        std::ofstream output(path);
        expect(output.good(), "unable to create synthetic tensor dump");
        for (std::size_t layer_index = 0; layer_index < 2; ++layer_index)
        {
            const auto input_dim =
                layer_index == 0 ? std::size_t{5} : std::size_t{64};
            for (std::size_t local_head_index = 0; local_head_index < 8; ++local_head_index)
            {
                const auto tensor_prefix =
                    "hidden_l" + std::to_string(layer_index) +
                    "_h" + std::to_string(local_head_index);
                write_tensor(output, tensor_prefix + "/seq", {1, input_dim, 8});
                write_tensor(output, tensor_prefix + "/dst", {1, 8, 1});
                write_tensor(output, tensor_prefix + "/src", {1, 8, 1});
            }
        }
        write_tensor(output, "os", {1, 64, 1});
        write_tensor(output, "odk", {1, 1, 1});
        write_tensor(output, "osk", {1, 1, 1});
        write_tensor(output, "oob", {1});
        write_tensor(output, "os1", {1, 64, 1});
        write_tensor(output, "odk1", {1, 1, 1});
        write_tensor(output, "osk1", {1, 1, 1});
        write_tensor(output, "oob1", {1});
        if (include_unreferenced_extra)
        {
            write_tensor(output, "hidden_l0_h0/output_bias", {8});
        }
    }
}

int main()
{
    try
    {
        const gatzk::model::Matrix secure_logits = {
            {FieldElement::one(), FieldElement(2)}};
        auto matching_reference_logits = secure_logits;
        expect(
            gatzk::protocol::quantized_reference_matches_secure(
                secure_logits,
                matching_reference_logits),
            "identical secure/reference matrices must match");
        matching_reference_logits[0][1] = FieldElement(3);
        expect(
            !gatzk::protocol::quantized_reference_matches_secure(
                secure_logits,
                matching_reference_logits),
            "one-element reference mutation must fail matrix equality");
        expect(
            gatzk::protocol::final_formal_run_accepted(true, true),
            "verifier/reference success must be finally accepted");
        expect(
            !gatzk::protocol::final_formal_run_accepted(true, false),
            "reference mismatch must deny final acceptance");
        expect(
            !gatzk::protocol::final_formal_run_accepted(false, true),
            "cryptographic verifier failure must deny final acceptance");

        const auto h00 = gatzk::protocol::hidden_head_prefix(0, 0);
        const auto h10 = gatzk::protocol::hidden_head_prefix(1, 0);
        expect(h00 == "P_h_l0_h0" && h10 == "P_h_l1_h0" && h00 != h10,
               "hidden head labels must retain layer/local identity");

        const auto qcat_l0 = gatzk::protocol::hidden_q_cat_selector_label(0);
        const auto qcat_l1 = gatzk::protocol::hidden_q_cat_selector_label(1);
        const auto qcat_out = gatzk::protocol::output_q_cat_selector_label();
        expect(qcat_l0 != qcat_l1 && qcat_l1 != qcat_out && qcat_l0 != qcat_out,
               "hidden and output Q_cat labels must be independent");

        for (std::size_t layer_index = 0; layer_index < 2; ++layer_index)
        {
            expect_selector_support(3 + layer_index, 8,
                                    gatzk::protocol::hidden_q_in_selector_label(layer_index));
            expect_selector_support(2 + layer_index, 8,
                                    gatzk::protocol::hidden_q_d_h_selector_label(layer_index));
            expect_selector_support(4 + layer_index, 8,
                                    gatzk::protocol::hidden_q_cat_selector_label(layer_index));
            expect_selector_support(5 + layer_index, 8,
                                    gatzk::protocol::hidden_q_nd_h_selector_label(layer_index));
        }
        expect_selector_support(6, 8, qcat_out);

        auto valid = synthetic_info();
        std::string reason;
        expect(gatzk::model::checkpoint_bundle_matches_formal_proof_shape(valid, &reason),
               "synthetic pair-indexed checkpoint shape must pass: " + reason);

        const auto suffix = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const auto temp_root = std::filesystem::temp_directory_path()
            / ("gatzk_layered_schema_runtime_" + suffix);
        std::filesystem::create_directories(temp_root);
        write_manifest(temp_root / "manifest.json", false);
        std::ifstream canonical_manifest(temp_root / "manifest.json");
        expect(canonical_manifest.good(), "unable to reopen canonical synthetic manifest");
        const std::string canonical_manifest_text(
            std::istreambuf_iterator<char>(canonical_manifest), {});
        expect(canonical_manifest_text.find("local_head_index") != std::string::npos,
               "canonical manifest must carry local head coordinates");
        const auto parsed = gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        expect(parsed.hidden_head_specs.size() == 16,
               "pair-indexed manifest parser count mismatch");
        expect(parsed.hidden_head_specs[0].layer_index == 0
                   && parsed.hidden_head_specs[0].local_head_index == 0
                   && parsed.hidden_head_specs[15].layer_index == 1
                   && parsed.hidden_head_specs[15].local_head_index == 7,
               "pair-indexed manifest parser identity mismatch");
        expect(gatzk::model::checkpoint_bundle_matches_formal_proof_shape(parsed, &reason),
               "parsed canonical pair manifest must pass formal shape validation: " + reason);

        write_manifest(
            temp_root / "manifest.json",
            false,
            false,
            false,
            nullptr,
            "multi_layer_multi_head",
            nullptr,
            "batching_rule");
        bool rejected_duplicate_manifest_field = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_duplicate_manifest_field =
                std::string(error.what()).find("duplicate top-level field")
                != std::string::npos;
        }
        expect(
            rejected_duplicate_manifest_field,
            "duplicate checkpoint manifest field must be rejected");

        write_manifest(
            temp_root / "manifest.json",
            false,
            false,
            false,
            nullptr,
            "multi_layer_multi_head",
            "batching_rule");
        bool rejected_missing_manifest_field = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_missing_manifest_field =
                std::string(error.what()).find("missing canonical field")
                != std::string::npos;
        }
        expect(
            rejected_missing_manifest_field,
            "missing checkpoint manifest field must be rejected");

        const auto scale_chain = gatzk::util::canonical_quantization_scale_chain(
            "gat_l3_scale_chain",
            std::vector<HiddenLayerProfile>{{8, 8}, {8, 8}},
            2);
        expect(scale_chain.s_input == 16 && scale_chain.s_delta == 256
                   && scale_chain.s_exp == 16,
               "canonical input/attention scale values mismatch");
        expect(scale_chain.s_h == std::vector<std::uint64_t>{16, 16, 16},
               "canonical scale chain H values mismatch");
        expect(scale_chain.s_h[1] == scale_chain.s_cat[0]
                   && scale_chain.s_h[2] == scale_chain.s_cat[1]
                   && scale_chain.s_h.back() == scale_chain.s_cat.back(),
               "every hidden concat must feed the next H scale");
        expect(scale_chain.s_w_hidden[0][0] == 4
                   && scale_chain.s_w_hidden[1][0] == 4
                   && scale_chain.s_h_prime[0][0] == 64
                   && scale_chain.s_h_prime[1][0] == 64,
               "canonical layer-aware H_prime scales mismatch");
        expect(scale_chain.s_a_src_hidden[0][0] == 4
                   && scale_chain.s_a_dst_hidden[1][0] == 4,
               "canonical layer-aware attention scales mismatch");
        expect(scale_chain.s_out_pre == 16 && scale_chain.s_out_head == 16
                   && scale_chain.s_out == 16 && scale_chain.s_w_output[0] == 4
                   && scale_chain.s_y_prime[0] == 64
                   && scale_chain.s_a_src_output[0] == 4
                   && scale_chain.s_b_output[1] == 16,
               "canonical output scale chain mismatch");
        expect(scale_chain.canonical_serialization() == scale_chain.canonical_serialization(),
               "canonical scale serialization must be deterministic");
        bool rejected_unknown_scale_id = false;
        try
        {
            (void)gatzk::util::canonical_quantization_scale_chain(
                "unsupported_id", std::vector<HiddenLayerProfile>{{8, 8}, {8, 8}}, 2);
        }
        catch (const std::exception &error)
        {
            rejected_unknown_scale_id =
                std::string(error.what()).find("unsupported quant_cfg_id") != std::string::npos;
        }
        expect(rejected_unknown_scale_id,
               "unknown quantization scale-chain ID must fail closed");

        // SignedRoundDiv output bounds use the same nearest-integer rounding
        // rule as trace.cpp after cancelling the common aggregate sum.  Keep
        // the helper's arithmetic guarded even though the formal datasets use
        // much smaller certified values.
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                16 * 800, 64, "synthetic SignedRoundDiv")
                == 200,
            "canonical SignedRoundDiv quotient bound mismatch");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                32, 64, "synthetic SignedRoundDiv even half case")
                == 1,
            "canonical SignedRoundDiv even half rounding mismatch");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                31, 64, "synthetic SignedRoundDiv below half case")
                == 0,
            "canonical SignedRoundDiv below-half rounding mismatch");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                33, 64, "synthetic SignedRoundDiv above half case")
                == 1,
            "canonical SignedRoundDiv above-half rounding mismatch");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                1, 3, "synthetic SignedRoundDiv odd lower case")
                == 0,
            "canonical SignedRoundDiv odd-denominator lower rounding mismatch");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                2, 3, "synthetic SignedRoundDiv odd upper case")
                == 1,
            "canonical SignedRoundDiv odd-denominator upper rounding mismatch");
        expect(
            std::max<std::uint64_t>(
                1,
                gatzk::protocol::canonical_signed_round_div_bound(
                    0, 64, "synthetic SignedRoundDiv zero compatibility floor"))
                == 1,
            "zero SignedRoundDiv certificate must retain the positive compatibility floor");
        bool rejected_zero_signed_divisor = false;
        try
        {
            (void)gatzk::protocol::canonical_signed_round_div_bound(
                1, 0, "synthetic zero divisor");
        }
        catch (const std::exception &error)
        {
            rejected_zero_signed_divisor =
                std::string(error.what()).find("not positive")
                != std::string::npos;
        }
        expect(
            rejected_zero_signed_divisor,
            "canonical SignedRoundDiv zero divisor must fail closed");
        bool rejected_signed_division_overflow = false;
        try
        {
            (void)gatzk::protocol::canonical_signed_round_div_bound(
                std::numeric_limits<std::uint64_t>::max(),
                3,
                "synthetic rounding overflow");
        }
        catch (const std::exception &error)
        {
            rejected_signed_division_overflow =
                std::string(error.what()).find("addition overflow")
                != std::string::npos;
        }
        expect(
            rejected_signed_division_overflow,
            "canonical SignedRoundDiv rounding overflow must fail closed");

        // PPI's range_bits=14 shape is intentionally small in this test.  It
        // exercises the same canonical static-table intervals without
        // materializing the production tables.
        gatzk::protocol::ProtocolContext ppi_shape_context;
        ppi_shape_context.config.batching_rule = "multi_graph_batch";
        ppi_shape_context.config.range_bits = 14;
        ppi_shape_context.config.quantization.s_delta = 256;
        ppi_shape_context.config.quantization.s_exp = 16;
        ppi_shape_context.dataset.num_nodes = 56944;
        ppi_shape_context.dataset.num_features = 50;
        ppi_shape_context.local.num_nodes = 56944;
        const auto ppi_table_shape =
            gatzk::protocol::plan_canonical_static_table_shape(
                ppi_shape_context);
        expect(
            !ppi_table_shape.large_whole_graph,
            "PPI multi-graph shape must not use the large whole-graph interval");
        expect(
            ppi_table_shape.lrelu_min_raw == -(1LL << 14)
                && ppi_table_shape.lrelu_max_raw == (1LL << 14),
            "PPI LReLU interval does not follow range_bits=14");
        expect(
            ppi_table_shape.lrelu_table_size == (1ULL << 15) + 1U
                && ppi_table_shape.elu_bound == (1ULL << 14)
                && ppi_table_shape.elu_table_size == (1ULL << 15) + 1U,
            "PPI LReLU/ELU table cardinality mismatch");
        expect(
            ppi_table_shape.lrelu_output_bound == (1ULL << 14)
                && ppi_table_shape.exp_input_bound == (1ULL << 15)
                && ppi_table_shape.exp_table_size == (1ULL << 15) + 1U,
            "PPI LReLU/ExpMap bound coverage mismatch");
        expect(
            ppi_table_shape.table_edge_domain_size == (1ULL << 16),
            "PPI static-table edge domain must be the next power of two");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                16 * 800, 64, "PPI ELU coverage")
                <= ppi_table_shape.elu_bound,
            "PPI ELU table interval does not cover canonical division bound");

        // The large whole-graph table planner is intentionally exercised with
        // dimensions only; it must not load OGBN data or allocate the 15M-row
        // ExpMap table.  Its derived delta bound is the authenticated LReLU
        // endpoint bound, while T_range remains exactly 2^B.
        gatzk::protocol::ProtocolContext ogbn_shape_context;
        ogbn_shape_context.config.batching_rule = "whole_graph_single";
        ogbn_shape_context.config.range_bits = 21;
        ogbn_shape_context.config.quantization.s_delta = 256;
        ogbn_shape_context.config.quantization.s_exp = 16;
        ogbn_shape_context.dataset.num_nodes = 169343;
        ogbn_shape_context.dataset.num_features = 128;
        ogbn_shape_context.local.num_nodes = 169343;
        ogbn_shape_context.local.edges.resize(1ULL << 18);
        const auto ogbn_table_shape =
            gatzk::protocol::plan_canonical_static_table_shape(
                ogbn_shape_context);
        expect(ogbn_table_shape.large_whole_graph,
               "OGBN dimension planner did not select whole-graph shape");
        expect(ogbn_table_shape.lrelu_output_bound == 7602173,
               "OGBN LReLU certified output bound changed unexpectedly");
        expect(ogbn_table_shape.exp_input_bound == 15204346,
               "OGBN ExpMap delta bound was not derived from LReLU bound");
        expect(ogbn_table_shape.exp_table_size == 15204347,
               "OGBN ExpMap table must include every delta through the bound");
        expect(ogbn_table_shape.range_table_size == (1ULL << 21),
               "T_range cardinality must remain exactly 2^B");
        expect(ogbn_table_shape.table_edge_domain_size == (1ULL << 24),
               "OGBN derived table domain must be the next power of two");
        expect(
            gatzk::protocol::canonical_exp_table_size_from_delta_bound(
                ogbn_table_shape.exp_input_bound)
                == ogbn_table_shape.exp_table_size,
            "ExpMap row planner and shape planner disagree");
        expect(
            gatzk::protocol::canonical_signed_round_div_bound(
                16 * 800, 64, "OGBN ELU coverage")
                <= (1ULL << 20),
            "OGBN ELU table interval does not cover canonical division bound");
        bool rejected_exp_row_overflow = false;
        try
        {
            (void)gatzk::protocol::canonical_exp_table_size_from_delta_bound(
                std::numeric_limits<std::uint64_t>::max());
        }
        catch (const std::exception &error)
        {
            rejected_exp_row_overflow =
                std::string(error.what()).find("row-count overflow")
                != std::string::npos;
        }
        expect(rejected_exp_row_overflow,
               "ExpMap delta row-count overflow must fail closed");
        bool rejected_table_domain_overflow = false;
        try
        {
            auto oversized = ogbn_shape_context;
            oversized.config.range_bits = 24;
            (void)gatzk::protocol::plan_canonical_static_table_shape(oversized);
        }
        catch (const std::exception &error)
        {
            rejected_table_domain_overflow =
                std::string(error.what()).find("interval is too large")
                != std::string::npos;
        }
        expect(rejected_table_domain_overflow,
               "static-table/domain row overflow must fail closed");

        write_manifest(temp_root / "manifest.json", false);
        write_tensor_dump(temp_root / "tensors.txt", true);
        bool rejected_unreferenced_tensor = false;
        try
        {
            (void)gatzk::model::load_checkpoint_bundle_parameters(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_unreferenced_tensor =
                std::string(error.what()).find("unreferenced tensor") != std::string::npos;
        }
        expect(rejected_unreferenced_tensor,
               "hidden output-bias tensor not referenced by the manifest must fail closed");

        write_manifest(temp_root / "manifest.json", true);
        bool rejected_missing_head_coordinates = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_missing_head_coordinates = std::string(error.what()).find("layer_index") != std::string::npos;
        }
        expect(rejected_missing_head_coordinates,
               "manifest missing head coordinates must be rejected before formal lookup");

        write_manifest(temp_root / "manifest.json", false, true);
        bool rejected_attention_bias = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_attention_bias =
                std::string(error.what()).find("bias-enabled attention") != std::string::npos;
        }
        expect(rejected_attention_bias,
               "attention scalar bias manifest must be rejected before formal lookup");

        write_manifest(temp_root / "manifest.json", false, false, true);
        bool rejected_hidden_output_bias = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_hidden_output_bias =
                std::string(error.what()).find("output_bias manifest field") != std::string::npos;
        }
        expect(rejected_hidden_output_bias,
               "hidden output_bias manifest field must be rejected before formal lookup");

        write_manifest(
            temp_root / "manifest.json",
            false,
            false,
            false,
            "unexpected_manifest_field");
        bool rejected_unexpected_field = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_unexpected_field =
                std::string(error.what()).find(
                    "unknown/non-canonical manifest field")
                != std::string::npos;
        }
        expect(
            rejected_unexpected_field,
            "unexpected checkpoint manifest field must be rejected");

        write_manifest(
            temp_root / "manifest.json",
            false,
            false,
            false,
            nullptr,
            "unsupported_family");
        bool rejected_noncanonical_family = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_noncanonical_family =
                std::string(error.what()).find("family_schema")
                != std::string::npos;
        }
        expect(
            rejected_noncanonical_family,
            "checkpoint manifest with a non-canonical family_schema must be rejected");

        std::error_code cleanup_error;
        std::filesystem::remove_all(temp_root, cleanup_error);

        std::cout << "layered schema runtime checks passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "layered schema runtime check failed: " << error.what() << '\n';
        return 1;
    }
}
