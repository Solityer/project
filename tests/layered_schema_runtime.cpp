#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/schema.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

    void write_manifest(const std::filesystem::path &path, bool global_only)
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
                manifest << "\"seq_kernel\": \"s\","
                         << " \"attn_dst_kernel\": \"dk\","
                         << " \"attn_dst_bias\": \"db\","
                         << " \"attn_src_kernel\": \"sk\","
                         << " \"attn_src_bias\": \"sb\","
                         << " \"output_bias\": \"ob\"}";
            }
        }
        manifest << R"(
  ],
  "output_head_specs": [
    {"head_index": 0, "seq_kernel": "os", "attn_dst_kernel": "odk", "attn_dst_bias": "odb", "attn_src_kernel": "osk", "attn_src_bias": "osb", "output_bias": "oob"},
    {"head_index": 1, "seq_kernel": "os1", "attn_dst_kernel": "odk1", "attn_dst_bias": "odb1", "attn_src_kernel": "osk1", "attn_src_bias": "osb1", "output_bias": "oob1"}
  ]
}
)";
    }
}

int main()
{
    try
    {
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
        std::error_code cleanup_error;
        std::filesystem::remove_all(temp_root, cleanup_error);
        expect(rejected_global_only,
               "global-only manifest must be rejected before formal lookup");

        std::cout << "layered schema runtime checks passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "layered schema runtime check failed: " << error.what() << '\n';
        return 1;
    }
}
