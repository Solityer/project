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
        info.family_schema_version = "multi_layer_multi_head_v2";
        info.output_average_rule = "per_head_bias_then_arithmetic_mean";
        info.model_arch_id = "synthetic_layered_arch";
        info.model_param_id = "synthetic_layered_params";
        info.quant_cfg_id = "synthetic_quant";
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
        bool global_only,
        bool legacy_attention_bias = false,
        bool hidden_output_bias = false)
    {
        std::ofstream manifest(path);
        expect(manifest.good(), "unable to create synthetic manifest");
        manifest << R"({
  "family_schema_version": "multi_layer_multi_head_v2",
  "output_average_rule": "per_head_bias_then_arithmetic_mean",
  "model_arch_id": "synthetic_layered_arch",
  "model_param_id": "synthetic_layered_params",
  "quant_cfg_id": "synthetic_quant",
  "static_table_id": "synthetic_tables",
  "degree_bound_id": "synthetic_degree",
  "L": 3,
  "K_out": 2,
  "C": 1,
  "d_in_profile": [5, 6],
  "hidden_layers": [
    {"layer_index": 0, "input_dim": 5, "head_count": 2, "head_dim": 3},
    {"layer_index": 1, "input_dim": 6, "head_count": 2, "head_dim": 3}
  ],
  "hidden_profile": [
    {"head_count": 2, "head_dim": 3},
    {"head_count": 2, "head_dim": 3}
  ],
  "hidden_head_specs": [
)";
        for (std::size_t layer_index = 0; layer_index < 2; ++layer_index)
        {
            for (std::size_t local_head_index = 0; local_head_index < 2; ++local_head_index)
            {
                if (layer_index != 0 || local_head_index != 0)
                {
                    manifest << ",\n";
                }
                manifest << "    {";
                if (global_only)
                {
                    // This fixture intentionally has only the retired global
                    // identity so the production parser must reject it.
                    manifest << "\"global_head_index\": "
                             << (layer_index * 2 + local_head_index) << ", ";
                }
                else
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
                if (legacy_attention_bias)
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
                layer_index == 0 ? std::size_t{5} : std::size_t{6};
            for (std::size_t local_head_index = 0; local_head_index < 2; ++local_head_index)
            {
                const auto tensor_prefix =
                    "hidden_l" + std::to_string(layer_index) +
                    "_h" + std::to_string(local_head_index);
                write_tensor(output, tensor_prefix + "/seq", {1, input_dim, 3});
                write_tensor(output, tensor_prefix + "/dst", {1, 3, 1});
                write_tensor(output, tensor_prefix + "/src", {1, 3, 1});
            }
        }
        write_tensor(output, "os", {1, 6, 1});
        write_tensor(output, "odk", {1, 1, 1});
        write_tensor(output, "osk", {1, 1, 1});
        write_tensor(output, "oob", {1});
        write_tensor(output, "os1", {1, 6, 1});
        write_tensor(output, "odk1", {1, 1, 1});
        write_tensor(output, "osk1", {1, 1, 1});
        write_tensor(output, "oob1", {1});
        if (include_unreferenced_extra)
        {
            write_tensor(output, "hidden_l0_h0/output_bias", {3});
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

        auto global_only = valid;
        global_only.hidden_head_specs.clear();
        for (std::size_t index = 0; index < 4; ++index)
        {
            CheckpointHeadSpec spec;
            spec.global_head_index = index;
            global_only.hidden_head_specs.push_back(spec);
        }
        expect(!gatzk::model::checkpoint_bundle_matches_formal_proof_shape(global_only, &reason),
               "global-only checkpoint head metadata must fail closed");

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
        expect(canonical_manifest_text.find("global_head_index") == std::string::npos,
               "canonical pair manifest must omit global_head_index");
        const auto parsed = gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        expect(parsed.hidden_head_specs.size() == 4,
               "pair-indexed manifest parser count mismatch");
        expect(parsed.hidden_head_specs[0].layer_index == 0
                   && parsed.hidden_head_specs[0].local_head_index == 0
                   && parsed.hidden_head_specs[3].layer_index == 1
                   && parsed.hidden_head_specs[3].local_head_index == 1,
               "pair-indexed manifest parser identity mismatch");
        expect(gatzk::model::checkpoint_bundle_matches_formal_proof_shape(parsed, &reason),
               "parsed canonical pair manifest must pass formal shape validation: " + reason);

        const auto scale_chain = gatzk::util::canonical_quantization_scale_chain(
            "gat_l3_scale_chain_v1",
            std::vector<HiddenLayerProfile>{{8, 8}, {8, 8}},
            2);
        expect(scale_chain.s_h == std::vector<std::uint64_t>{1, 16, 16},
               "canonical scale chain H values mismatch");
        expect(scale_chain.s_h[1] == scale_chain.s_cat[0]
                   && scale_chain.s_h[2] == scale_chain.s_cat[1]
                   && scale_chain.s_h.back() == scale_chain.s_cat.back(),
               "every hidden concat must feed the next H scale");
        expect(scale_chain.s_h_prime[0][0] == 16 && scale_chain.s_h_prime[1][0] == 256,
               "canonical layer-aware H_prime scales mismatch");
        expect(scale_chain.s_a_src_hidden[0][0] == 16 && scale_chain.s_a_src_hidden[1][0] == 1,
               "canonical layer-aware attention scales mismatch");
        expect(scale_chain.s_out_pre == 16 && scale_chain.s_out_head == 16
                   && scale_chain.s_out == 16 && scale_chain.s_y_prime[0] == 256
                   && scale_chain.s_b_output[1] == 16,
               "canonical output scale chain mismatch");
        expect(scale_chain.canonical_serialization() == scale_chain.canonical_serialization(),
               "canonical scale serialization must be deterministic");
        bool rejected_unknown_scale_id = false;
        try
        {
            (void)gatzk::util::canonical_quantization_scale_chain(
                "legacy_id", std::vector<HiddenLayerProfile>{{8, 8}, {8, 8}}, 2);
        }
        catch (const std::exception &error)
        {
            rejected_unknown_scale_id =
                std::string(error.what()).find("unsupported quant_cfg_id") != std::string::npos;
        }
        expect(rejected_unknown_scale_id,
               "unknown quantization scale-chain ID must fail closed");

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
        bool rejected_global_only = false;
        try
        {
            (void)gatzk::model::inspect_checkpoint_bundle(temp_root.string());
        }
        catch (const std::exception &error)
        {
            rejected_global_only = std::string(error.what()).find("layer_index") != std::string::npos;
        }
        expect(rejected_global_only,
               "global-only manifest must be rejected before formal lookup");

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
