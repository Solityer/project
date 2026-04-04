#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
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

ProofFixture build_proof_fixture_with_stage(const gatzk::util::AppConfig& config) {
    try {
        auto context = gatzk::protocol::build_context(config);
        try {
            auto trace = gatzk::protocol::build_trace(context);
            try {
                auto proof = gatzk::protocol::prove(context, trace);
                return ProofFixture{std::move(context), std::move(trace), std::move(proof)};
            } catch (const std::exception& error) {
                throw std::runtime_error(std::string("prove: ") + error.what());
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("build_trace: ") + error.what());
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("build_context: ") + error.what());
    }
}

const ProofFixture& full_cora_proof_fixture() {
    static const auto fixture = []() {
        auto config = full_graph_formal_config();
        config.export_dir = "runs/test_full_cora";
        return build_proof_fixture_with_stage(config);
    }();
    return fixture;
}

gatzk::util::AppConfig synthetic_family_formal_config(std::size_t k_out) {
    auto config = small_debug_config();
    config.export_dir = "runs/test_synth_family_k" + std::to_string(k_out);
    config.layer_count = 3;
    config.hidden_profile = {{2, 1}, {1, 1}};
    config.d_in_profile = {1433, 2};
    config.K_out = k_out;
    config.hidden_dim = 1;
    config.num_classes = 7;
    config.local_nodes = 16;
    config.center_node = 0;
    config.allow_synthetic_model = true;
    config.prove_enabled = true;
    return config;
}

const ProofFixture& synthetic_multilayer_proof_fixture() {
    static const auto fixture = []() {
        const auto config = synthetic_family_formal_config(1);
        return build_proof_fixture_with_stage(config);
    }();
    return fixture;
}

const ProofFixture& synthetic_multioutput_proof_fixture() {
    static const auto fixture = []() {
        const auto config = synthetic_family_formal_config(2);
        return build_proof_fixture_with_stage(config);
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
    for (const auto& layer : model.hidden_layers) {
        model.hidden_heads.insert(model.hidden_heads.end(), layer.heads.begin(), layer.heads.end());
    }

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

FieldElement external_eval(const Proof& proof, const std::string& name) {
    for (const auto& [entry_name, value] : proof.external_evaluations) {
        if (entry_name == name) {
            return value;
        }
    }
    throw std::runtime_error("missing external evaluation: " + name);
}

void overwrite_external_eval(Proof* proof, const std::string& name, const FieldElement& value) {
    for (auto& [entry_name, entry_value] : proof->external_evaluations) {
        if (entry_name == name) {
            entry_value = value;
            return;
        }
    }
    throw std::runtime_error("missing external evaluation for overwrite: " + name);
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

void test_formal_multilayer_family_round_trip() {
    const auto& fixture = synthetic_multilayer_proof_fixture();
    require(fixture.context.model.hidden_layers.size() == 2, "expected two hidden layers in formal family fixture");
    require(fixture.context.model.L == 3, "expected L=3 in formal family fixture");
    if (!gatzk::protocol::verify(fixture.context, fixture.proof)) {
        const auto y_out = fixture.proof.challenges.at("y_out");
        const auto expected = external_eval(fixture.proof, "mu_Y_lin")
            + bias_fold_from_output_head(
                fixture.context.model.output_layer.heads.front().output_bias_fp,
                fixture.context.local.num_nodes,
                y_out);
        throw std::runtime_error(
            "multilayer family formal round-trip must verify; mu_out="
            + external_eval(fixture.proof, "mu_out").to_string()
            + " mu_Y_lin=" + external_eval(fixture.proof, "mu_Y_lin").to_string()
            + " expected=" + expected.to_string());
    }
}

void test_formal_k_out_average_is_verified() {
    const auto& fixture = synthetic_multioutput_proof_fixture();
    require(fixture.context.model.K_out == 2, "expected K_out=2 in formal average fixture");
    if (!gatzk::protocol::verify(fixture.context, fixture.proof)) {
        const auto y_out = fixture.proof.challenges.at("y_out");
        const auto expected0 = external_eval(fixture.proof, "mu_out0_y_lin")
            + bias_fold_from_output_head(
                fixture.context.model.output_layer.heads[0].output_bias_fp,
                fixture.context.local.num_nodes,
                y_out);
        const auto expected1 = external_eval(fixture.proof, "mu_out1_y_lin")
            + bias_fold_from_output_head(
                fixture.context.model.output_layer.heads[1].output_bias_fp,
                fixture.context.local.num_nodes,
                y_out);
        throw std::runtime_error(
            "multi-output formal round-trip must verify; mu_out0_y="
            + external_eval(fixture.proof, "mu_out0_y").to_string()
            + " expected0=" + expected0.to_string()
            + " mu_out1_y=" + external_eval(fixture.proof, "mu_out1_y").to_string()
            + " expected1=" + expected1.to_string());
    }

    const auto mu_y_lin_0 = external_eval(fixture.proof, "mu_out0_y_lin");
    const auto mu_y_lin_1 = external_eval(fixture.proof, "mu_out1_y_lin");
    const auto mu_y_0 = external_eval(fixture.proof, "mu_out0_y");
    const auto mu_y_1 = external_eval(fixture.proof, "mu_out1_y");
    require(FieldElement(2) * external_eval(fixture.proof, "mu_Y_lin") == mu_y_lin_0 + mu_y_lin_1, "formal proof must carry averaged pre-bias output");
    require(FieldElement(2) * external_eval(fixture.proof, "mu_out") == mu_y_0 + mu_y_1, "formal proof must carry averaged final output");
}

void test_formal_rejects_wrong_output_average() {
    const auto& fixture = synthetic_multioutput_proof_fixture();
    auto proof = fixture.proof;
    overwrite_external_eval(&proof, "mu_out", external_eval(proof, "mu_out") + FieldElement::one());
    require(!gatzk::protocol::verify(fixture.context, proof), "verifier must reject a witness with wrong final averaged logits");
}

void test_formal_hidden_family_dimension_chain_is_enforced() {
    auto context = synthetic_multilayer_proof_fixture().context;
    context.model.hidden_layers[1].input_dim += 1;
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_trace(context);
    });
    require(error.find("previous concat width") != std::string::npos, "dimension chain mismatch must fail fast at formal trace build");
}

void test_checkpoint_backed_formal_validation() {
    const auto& fixture = full_cora_proof_fixture();
    require(!fixture.context.config.checkpoint_bundle.empty(), "checkpoint-backed validation requires a real checkpoint bundle");
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "checkpoint-backed cora formal prove/verify must pass");
}

void test_config_conflict_fails_fast() {
    auto config = full_graph_formal_config();
    config.K_out = 2;
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_context(config);
    });
    require(error.find("K_out") != std::string::npos, "config/model K_out conflict must fail fast");
}

void test_fail_fast_boundary_matches_actual_supported_family() {
    const auto& supported_fixture = synthetic_multioutput_proof_fixture();
    require(gatzk::protocol::verify(supported_fixture.context, supported_fixture.proof), "supported family boundary should not be over-rejected");

    auto context = supported_fixture.context;
    context.model.output_layer.input_dim += 1;
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_trace(context);
    });
    require(error.find("output_layer input_dim") != std::string::npos, "fail-fast boundary must target the actual unsupported family inconsistency");
}

void test_no_regression_existing_paths() {
    const auto& full_fixture = full_cora_proof_fixture();
    require(gatzk::protocol::verify(full_fixture.context, full_fixture.proof), "existing cora checkpoint path must remain valid");

    const auto config = ppi_batch_config();
    const auto dataset = gatzk::data::load_dataset(config);
    const auto local = gatzk::data::normalize_graph_input(dataset, config);
    require(local.graph_count == 2, "PPI batch normalization path must remain intact");
    require(local.node_ptr.size() == 3, "PPI batch node_ptr regression");
    require(local.edge_ptr.size() == 3, "PPI batch edge_ptr regression");
}

}  // namespace

int main(int argc, char** argv) {
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
        {"formal_multilayer_family_round_trip", test_formal_multilayer_family_round_trip},
        {"formal_k_out_average_is_verified", test_formal_k_out_average_is_verified},
        {"formal_rejects_wrong_output_average", test_formal_rejects_wrong_output_average},
        {"formal_hidden_family_dimension_chain_is_enforced", test_formal_hidden_family_dimension_chain_is_enforced},
        {"checkpoint_backed_formal_validation", test_checkpoint_backed_formal_validation},
        {"config_conflict_fails_fast", test_config_conflict_fails_fast},
        {"fail_fast_boundary_matches_actual_supported_family", test_fail_fast_boundary_matches_actual_supported_family},
        {"no_regression_existing_paths", test_no_regression_existing_paths},
    };

    const auto run_all = [&]() {
        const std::string filter = argc > 1 ? argv[1] : "";
        for (const auto& [name, test] : tests) {
            if (!filter.empty() && name.find(filter) == std::string::npos) {
                continue;
            }
            test();
            std::cout << "[PASS] " << name << '\n';
        }
    };
    if (const char* no_catch = std::getenv("GATZK_TEST_NO_CATCH"); no_catch != nullptr && std::string(no_catch) == "1") {
        run_all();
        return 0;
    }
    try {
        run_all();
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
    return 0;
}
