#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

using gatzk::algebra::FieldElement;
using gatzk::model::AttentionHeadParameters;
using gatzk::model::FloatMatrix;
using gatzk::model::ModelParameters;
using gatzk::protocol::Proof;
using gatzk::protocol::ProtocolContext;
using gatzk::protocol::TraceArtifacts;

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::string repo_path(const std::string& relative) {
    return (repo_root() / relative).string();
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string require_throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception but none was thrown");
}

gatzk::util::AppConfig small_debug_config() {
    gatzk::util::AppConfig config;
    config.project_root = repo_root().string();
    config.dataset = "cora";
    config.data_root = "data";
    config.cache_root = "data/cache";
    config.export_dir = "runs/test_small_real";
    config.task_type = "transductive_node_classification";
    config.report_unit = "node";
    config.batching_rule = "whole_graph_single";
    config.subgraph_rule = "sampled_subgraph";
    config.self_loop_rule = "per_node";
    config.edge_sort_rule = "edge_gid_then_dst_stable";
    config.chunking_rule = "none";
    config.hidden_dim = 4;
    config.num_classes = 7;
    config.range_bits = 12;
    config.seed = 11;
    config.local_nodes = 32;
    config.center_node = 0;
    config.layer_count = 2;
    config.hidden_profile = {{1, 4}};
    config.d_in_profile = {1433};
    config.K_out = 1;
    config.batch_graphs = 1;
    config.allow_synthetic_model = true;
    config.dump_trace = false;
    config.auto_prepare_dataset = false;
    config.prove_enabled = true;
    return config;
}

gatzk::util::AppConfig full_graph_formal_config() {
    return gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
}

gatzk::util::AppConfig ppi_batch_config() {
    return gatzk::util::load_config(repo_path("configs/ppi_batch.cfg"));
}

const ProtocolContext& full_graph_formal_context() {
    static const auto context = []() {
        return gatzk::protocol::build_context(full_graph_formal_config());
    }();
    return context;
}

struct ProofFixture {
    ProtocolContext context;
    TraceArtifacts trace;
    Proof proof;
};

const ProofFixture& full_cora_proof_fixture() {
    static const auto fixture = []() {
        auto config = full_graph_formal_config();
        config.export_dir = "runs/test_full_cora";
        const auto context = gatzk::protocol::build_context(config);
        const auto trace = gatzk::protocol::build_trace(context);
        const auto proof = gatzk::protocol::prove(context, trace);
        return ProofFixture{context, trace, proof};
    }();
    return fixture;
}

AttentionHeadParameters make_head(std::size_t input_dim, std::size_t output_dim, double bias_shift) {
    AttentionHeadParameters head;
    head.seq_kernel_fp.assign(input_dim, std::vector<double>(output_dim, 0.0));
    for (std::size_t row = 0; row < input_dim; ++row) {
        head.seq_kernel_fp[row][row % output_dim] = 1.0;
    }
    head.attn_src_kernel_fp.assign(output_dim, 0.25);
    head.attn_dst_kernel_fp.assign(output_dim, 0.5);
    head.output_bias_fp.assign(output_dim, bias_shift);
    return head;
}

ModelParameters make_family_model(std::size_t k_out) {
    ModelParameters model;
    model.has_real_multihead = true;
    model.L = 3;
    model.d_in_profile = {2, 4};
    model.hidden_profile = {{2, 2}, {1, 3}};
    model.K_out = k_out;
    model.C = 2;

    model.hidden_layers = {
        {
            .layer_index = 0,
            .input_dim = 2,
            .shape = {2, 2},
            .heads = {make_head(2, 2, 0.0), make_head(2, 2, 0.0)},
        },
        {
            .layer_index = 1,
            .input_dim = 4,
            .shape = {1, 3},
            .heads = {make_head(4, 3, 0.0)},
        },
    };
    model.hidden_heads = model.hidden_layers.front().heads;

    model.output_layer = {
        .input_dim = 3,
        .output_dim = 2,
        .head_count = k_out,
        .heads = {},
    };
    for (std::size_t head_index = 0; head_index < k_out; ++head_index) {
        model.output_layer.heads.push_back(make_head(3, 2, static_cast<double>(head_index + 1)));
    }
    model.output_head = model.output_layer.heads.front();
    return model;
}

FieldElement quantize_bias(double value) {
    return FieldElement::from_signed(static_cast<std::int64_t>(value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5));
}

FieldElement bias_fold_from_output_head(
    const std::vector<double>& bias,
    std::size_t node_count,
    const FieldElement& point) {
    FieldElement out = FieldElement::zero();
    for (std::size_t i = 0; i < node_count; ++i) {
        for (std::size_t j = 0; j < bias.size(); ++j) {
            out += quantize_bias(bias[j]) * point.pow(static_cast<std::uint64_t>(i * bias.size() + j));
        }
    }
    return out;
}

void test_full_graph_config_parse() {
    const auto config = full_graph_formal_config();
    require(config.dataset == "cora", "expected cora runtime config");
    require(config.layer_count == 2, "expected explicit L=2");
    require(config.hidden_profile.size() == 1, "expected one hidden layer profile");
    require(config.hidden_profile.front().head_count == 8, "expected hidden K=8");
    require(config.hidden_profile.front().head_dim == 8, "expected hidden d_h=8");
    require(config.d_in_profile == std::vector<std::size_t>{1433}, "expected explicit d_in_profile");
    require(config.K_out == 1, "expected explicit K_out");
}

void test_whole_graph_normalization_has_explicit_ptrs_and_sort() {
    const auto context = gatzk::protocol::build_context(full_graph_formal_config());
    require(context.local.public_input.G_batch == 1, "single-graph context must expose G_batch=1");
    require(context.local.node_ptr == std::vector<std::size_t>({0, context.local.num_nodes}), "single graph must keep explicit node_ptr");
    require(context.local.edge_ptr == std::vector<std::size_t>({0, context.local.edges.size()}), "single graph must keep explicit edge_ptr");

    std::vector<bool> has_self(context.local.num_nodes, false);
    std::size_t previous_dst = 0;
    for (std::size_t edge_index = 0; edge_index < context.local.edges.size(); ++edge_index) {
        const auto& edge = context.local.edges[edge_index];
        if (edge_index != 0) {
            require(previous_dst <= edge.dst, "whole-graph edges must be stable-sorted by dst");
        }
        previous_dst = edge.dst;
        if (edge.src == edge.dst) {
            has_self[edge.dst] = true;
        }
    }
    for (std::size_t node = 0; node < has_self.size(); ++node) {
        require(has_self[node], "every node must receive an explicit self-loop");
    }
}

void test_multi_graph_batch_normalization_uses_real_ppi_data() {
    const auto config = ppi_batch_config();
    const auto dataset = gatzk::data::load_dataset(config);
    const auto local = gatzk::data::normalize_graph_input(dataset, config);
    require(local.graph_count == 2, "PPI batch config should normalize two graphs");
    require(local.public_input.G_batch == 2, "PPI batch must carry G_batch=2");
    require(local.node_ptr.size() == 3, "PPI batch node_ptr size mismatch");
    require(local.edge_ptr.size() == 3, "PPI batch edge_ptr size mismatch");
    for (std::size_t graph_id = 0; graph_id < local.graph_count; ++graph_id) {
        const auto node_begin = local.node_ptr[graph_id];
        const auto node_end = local.node_ptr[graph_id + 1];
        const auto edge_begin = local.edge_ptr[graph_id];
        const auto edge_end = local.edge_ptr[graph_id + 1];
        require(node_begin < node_end, "each PPI graph must contain nodes");
        require(edge_begin < edge_end, "each PPI graph must contain edges");
        std::size_t previous_dst = node_begin;
        for (std::size_t edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
            const auto& edge = local.edges[edge_index];
            require(edge.graph_id == graph_id, "edges must stay in contiguous graph blocks");
            require(edge.src >= node_begin && edge.src < node_end, "edge src must stay inside graph block");
            require(edge.dst >= node_begin && edge.dst < node_end, "edge dst must stay inside graph block");
            if (edge_index != edge_begin) {
                require(previous_dst <= edge.dst, "dst order must be non-decreasing inside each graph block");
            }
            previous_dst = edge.dst;
        }
    }
}

void test_edge_sort_rule_is_edge_gid_then_dst_stable() {
    gatzk::data::GraphDataset dataset;
    dataset.name = "toy_batch";
    dataset.num_nodes = 4;
    dataset.num_features = 1;
    dataset.num_classes = 2;
    dataset.graph_count = 2;
    dataset.node_ptr = {0, 2, 4};
    dataset.features_fp = {{1.0}, {2.0}, {3.0}, {4.0}};
    dataset.features = {
        {FieldElement(1)},
        {FieldElement(2)},
        {FieldElement(3)},
        {FieldElement(4)},
    };
    dataset.labels = {0, 0, 1, 1};
    dataset.edges = {
        {1, 0, 0, 0},
        {0, 1, 0, 1},
        {3, 2, 1, 2},
        {2, 3, 1, 3},
    };

    auto config = small_debug_config();
    config.dataset = "toy_batch";
    config.batch_graphs = 2;
    const auto local = gatzk::data::normalize_graph_input(dataset, config);
    require(local.edge_ptr == std::vector<std::size_t>({0, 4, 8}), "edge_ptr must reflect explicit per-graph edge blocks including self-loops");
    for (std::size_t graph_id = 0; graph_id < local.graph_count; ++graph_id) {
        std::size_t previous_dst = local.node_ptr[graph_id];
        for (std::size_t edge_index = local.edge_ptr[graph_id]; edge_index < local.edge_ptr[graph_id + 1]; ++edge_index) {
            require(local.edges[edge_index].graph_id == graph_id, "graph ids must stay grouped");
            if (edge_index != local.edge_ptr[graph_id]) {
                require(previous_dst <= local.edges[edge_index].dst, "dst sort rule violated");
            }
            previous_dst = local.edges[edge_index].dst;
        }
    }
}

void test_forward_multilayer_hidden_family_uses_concat_bridge() {
    const FloatMatrix features = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
    };
    const std::vector<gatzk::data::Edge> edges = {
        {0, 0, 0, 0},
        {1, 1, 0, 1},
        {2, 2, 0, 2},
    };
    const auto model = make_family_model(1);
    const auto trace = gatzk::model::forward_note_style(features, edges, model);
    require(trace.hidden_layer_traces.size() == 2, "expected two hidden layer traces");
    require(trace.hidden_layer_traces[0].concat.front().size() == 4, "first hidden concat width mismatch");
    require(trace.hidden_layer_traces[1].input.front().size() == 4, "second hidden layer input must equal previous concat width");
    require(trace.hidden_layer_traces[1].concat.front().size() == 3, "second hidden concat width mismatch");
}

void test_forward_k_out_average_matches_output_head_average() {
    const FloatMatrix features = {
        {1.0, 2.0},
        {3.0, 1.0},
    };
    const std::vector<gatzk::data::Edge> edges = {
        {0, 0, 0, 0},
        {1, 1, 0, 1},
    };
    const auto model = make_family_model(2);
    const auto trace = gatzk::model::forward_note_style(features, edges, model);
    require(trace.output_head_traces.size() == 2, "expected two output attention heads");
    for (std::size_t row = 0; row < trace.Y.size(); ++row) {
        for (std::size_t col = 0; col < trace.Y[row].size(); ++col) {
            const auto expected = 0.5 * (
                trace.output_head_traces[0].H_agg[row][col]
                + trace.output_head_traces[1].H_agg[row][col]);
            require(std::abs(trace.Y[row][col] - expected) < 1e-9, "final logits must average all output heads");
        }
    }
}

void test_metadata_contains_required_profile_fields() {
    const auto& context = full_graph_formal_context();
    const auto metadata = gatzk::protocol::canonical_public_metadata(context);
    require(metadata.dataset_name == "cora", "metadata dataset_name mismatch");
    require(metadata.task_type == "transductive_node_classification", "metadata task_type mismatch");
    require(metadata.report_unit == "node", "metadata report_unit mismatch");
    require(metadata.graph_count == "1", "metadata graph_count mismatch");
    require(metadata.L == "2", "metadata L mismatch");
    require(metadata.hidden_profile == "8x8", "metadata hidden_profile mismatch");
    require(metadata.d_in_profile == "1433", "metadata d_in_profile mismatch");
    require(metadata.K_out == "1", "metadata K_out mismatch");
    require(metadata.batching_rule == "whole_graph_single", "metadata batching_rule mismatch");
    require(metadata.subgraph_rule == "whole_graph", "metadata subgraph_rule mismatch");
    require(metadata.self_loop_rule == "per_node", "metadata self_loop_rule mismatch");
    require(metadata.edge_sort_rule == "edge_gid_then_dst_stable", "metadata edge_sort_rule mismatch");
    require(metadata.chunking_rule == "none", "metadata chunking_rule mismatch");
}

void test_formal_proof_round_trip_and_manifest_export() {
    const auto& fixture = full_cora_proof_fixture();
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "formal prove/verify round-trip must accept");

    gatzk::protocol::RunMetrics metrics;
    metrics.backend_name = "test";
    metrics.config = repo_path("configs/cora_full.cfg");
    metrics.dataset = fixture.context.dataset.name;
    metrics.node_count = fixture.context.local.num_nodes;
    metrics.edge_count = fixture.context.local.edges.size();
    metrics.proof_size_bytes = gatzk::protocol::proof_size_bytes(fixture.proof);

    auto export_context = fixture.context;
    export_context.config.export_dir = "runs/test_manifest_export";
    gatzk::protocol::export_run_artifacts(export_context, fixture.trace, fixture.proof, metrics, true);

    const auto manifest_path = repo_root() / "runs/test_manifest_export/run_manifest.json";
    require(std::filesystem::exists(manifest_path), "run_manifest.json must be exported");
    const auto manifest_text = [] (const std::filesystem::path& path) {
        std::ifstream input(path);
        std::ostringstream out;
        out << input.rdbuf();
        return out.str();
    }(manifest_path);
    for (const auto& key : {
             "dataset_name",
             "task_type",
             "report_unit",
             "graph_count",
             "L",
             "hidden_profile",
             "d_in_profile",
             "K_out",
             "batching_rule",
             "subgraph_rule",
             "self_loop_rule",
             "edge_sort_rule",
             "chunking_rule",
             "node_ptr",
             "edge_ptr",
         }) {
        require(manifest_text.find(key) != std::string::npos, std::string("run_manifest missing key: ") + key);
    }
}

void test_formal_output_bias_relation_is_enforced() {
    const auto& fixture = full_cora_proof_fixture();
    const auto challenges = fixture.proof.challenges;
    const auto y_out = challenges.at("y_out");

    FieldElement mu_out = FieldElement::zero();
    FieldElement mu_y_lin = FieldElement::zero();
    for (const auto& [name, value] : fixture.proof.external_evaluations) {
        if (name == "mu_out") {
            mu_out = value;
        } else if (name == "mu_Y_lin") {
            mu_y_lin = value;
        }
    }
    const auto expected_bias = bias_fold_from_output_head(
        fixture.context.model.output_layer.heads.front().output_bias_fp,
        fixture.context.local.num_nodes,
        y_out);
    require(mu_out == mu_y_lin + expected_bias, "final logits must equal pre-bias output plus output-head bias");
}

void test_config_conflict_fails_fast() {
    auto config = full_graph_formal_config();
    config.K_out = 2;
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_context(config);
    });
    require(error.find("K_out") != std::string::npos, "config/model K_out conflict must fail fast");
}

void test_unsupported_formal_family_shape_fails_fast() {
    auto context = full_graph_formal_context();
    context.model.K_out = 2;
    context.model.output_layer.head_count = 2;
    context.model.output_layer.heads.push_back(context.model.output_layer.heads.front());
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_trace(context);
    });
    require(error.find("K_out=1") != std::string::npos || error.find("supports only") != std::string::npos, "unsupported formal family must fail fast");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"full_graph_config_parse", test_full_graph_config_parse},
        {"whole_graph_normalization_has_explicit_ptrs_and_sort", test_whole_graph_normalization_has_explicit_ptrs_and_sort},
        {"multi_graph_batch_normalization_uses_real_ppi_data", test_multi_graph_batch_normalization_uses_real_ppi_data},
        {"edge_sort_rule_is_edge_gid_then_dst_stable", test_edge_sort_rule_is_edge_gid_then_dst_stable},
        {"forward_multilayer_hidden_family_uses_concat_bridge", test_forward_multilayer_hidden_family_uses_concat_bridge},
        {"forward_k_out_average_matches_output_head_average", test_forward_k_out_average_matches_output_head_average},
        {"metadata_contains_required_profile_fields", test_metadata_contains_required_profile_fields},
        {"formal_proof_round_trip_and_manifest_export", test_formal_proof_round_trip_and_manifest_export},
        {"formal_output_bias_relation_is_enforced", test_formal_output_bias_relation_is_enforced},
        {"config_conflict_fails_fast", test_config_conflict_fails_fast},
        {"unsupported_formal_family_shape_fails_fast", test_unsupported_formal_family_shape_fails_fast},
    };

    try {
        for (const auto& [name, test] : tests) {
            test();
            std::cout << "[PASS] " << name << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
    return 0;
}
