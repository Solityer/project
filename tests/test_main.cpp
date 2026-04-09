#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

std::string slurp_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    output << text;
}

double extract_json_number(const std::string& text, const std::string& key) {
    const auto marker = "\"" + key + "\": \"";
    const auto start = text.find(marker);
    if (start != std::string::npos) {
        const auto value_begin = start + marker.size();
        const auto value_end = text.find('"', value_begin);
        require(value_end != std::string::npos, "unterminated json value: " + key);
        return std::stod(text.substr(value_begin, value_end - value_begin));
    }
    const auto numeric_marker = "\"" + key + "\": ";
    const auto numeric_start = text.find(numeric_marker);
    require(numeric_start != std::string::npos, "missing json key: " + key);
    const auto value_begin = numeric_start + numeric_marker.size();
    auto value_end = value_begin;
    while (value_end < text.size()) {
        const char ch = text[value_end];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
            ++value_end;
            continue;
        }
        break;
    }
    require(value_end > value_begin, "missing numeric json value: " + key);
    return std::stod(text.substr(value_begin, value_end - value_begin));
}

std::string extract_dataset_row(const std::string& latest, const std::string& dataset) {
    const auto marker = "\"dataset\": \"" + dataset + "\"";
    const auto dataset_pos = latest.find(marker);
    require(dataset_pos != std::string::npos, "missing dataset row: " + dataset);
    const auto row_begin = latest.rfind('{', dataset_pos);
    require(row_begin != std::string::npos, "missing dataset row start: " + dataset);
    std::size_t depth = 0;
    for (std::size_t i = row_begin; i < latest.size(); ++i) {
        if (latest[i] == '{') {
            ++depth;
        } else if (latest[i] == '}') {
            require(depth > 0, "malformed dataset row depth for: " + dataset);
            --depth;
            if (depth == 0) {
                return latest.substr(row_begin, i - row_begin + 1);
            }
        }
    }
    throw std::runtime_error("unterminated dataset row: " + dataset);
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

gatzk::util::AppConfig full_citeseer_formal_config() {
    return gatzk::util::load_config(repo_path("configs/citeseer_full.cfg"));
}

gatzk::util::AppConfig ppi_full_formal_config() {
    return gatzk::util::load_config(repo_path("configs/ppi_batch_formal.cfg"));
}

gatzk::util::AppConfig ogbn_arxiv_full_formal_config() {
    return gatzk::util::load_config(repo_path("configs/ogbn_arxiv_full.cfg"));
}

gatzk::util::AppConfig ppi_local_batch_config() {
    auto config = ppi_full_formal_config();
    config.batch_graphs = 2;
    config.export_dir = "runs/test_ppi_local_batch";
    return config;
}

std::filesystem::path ogbn_arxiv_bundle_dir() {
    return repo_root() / "artifacts" / "checkpoints" / "ogbn_arxiv_gat";
}

std::filesystem::path ogbn_arxiv_run_dir() {
    return repo_root() / "runs" / "ogbn_arxiv_full";
}

std::filesystem::path ogbn_arxiv_warm_manifest_path() {
    return ogbn_arxiv_run_dir() / "warm" / "run_manifest.json";
}

std::filesystem::path ogbn_arxiv_warm_log_path() {
    return ogbn_arxiv_run_dir() / "ogbn_warm.log";
}

constexpr double kOgbnArxivBaselineProveMs = 2379199.195;
constexpr double kOgbnArxivBaselineDomainOpenEdgeMs = 1118162.526;
constexpr double kOgbnArxivBaselineDomainOpenFhMs = 485992.859;
constexpr double kOgbnArxivBaselineQuotientTEdgeMs = 603995.964;
constexpr double kOgbnArxivBaselineQuotientTFhMs = 243877.917;
constexpr double kOgbnArxivBaselineDynamicDomainConvertMs = 203131.349;
constexpr double kOgbnArxivCurrentBaselineCommitmentMs = 206436.317;
constexpr double kOgbnArxivCurrentBaselineProveMs = 1141868.307;
constexpr double kOgbnArxivCurrentBaselineVerifyMs = 201617.601;
constexpr double kOgbnArxivCurrentBaselineDomainOpenEdgeMs = 75151.947;
constexpr double kOgbnArxivCurrentBaselineQuotientTEdgeMs = 55696.334;
constexpr double kOgbnArxivCurrentBaselineDomainOpenFhMs = 442746.206;
constexpr double kOgbnArxivCurrentBaselineQuotientTFhMs = 287293.344;
constexpr double kOgbnArxivCurrentBaselineDynamicDomainConvertMs = 210000.0;

double extract_commitment_time_ms_from_manifest(const std::string& text) {
    const std::string explicit_marker = "\"commitment_time_ms\": \"";
    const auto explicit_start = text.find(explicit_marker);
    if (explicit_start != std::string::npos) {
        const auto value_begin = explicit_start + explicit_marker.size();
        const auto value_end = text.find('"', value_begin);
        require(value_end != std::string::npos, "unterminated commitment_time_ms value");
        return std::stod(text.substr(value_begin, value_end - value_begin));
    }
    return extract_json_number(text, "commit_dynamic_ms")
        + extract_json_number(text, "quotient_bundle_pack_ms");
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

const ProofFixture& full_citeseer_proof_fixture() {
    static const auto fixture = []() {
        auto config = full_citeseer_formal_config();
        config.export_dir = "runs/test_full_citeseer";
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

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

void write_tensor_record(
    std::ofstream* output,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    const std::vector<double>& values) {
    *output << "TENSOR " << name << ' ' << shape.size();
    for (const auto dim : shape) {
        *output << ' ' << dim;
    }
    *output << ' ' << values.size() << '\n';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            *output << ' ';
        }
        *output << values[i];
    }
    *output << '\n';
}

std::vector<double> flatten_seq_kernel(const FloatMatrix& matrix) {
    std::vector<double> out;
    out.reserve(matrix.size() * (matrix.empty() ? 0 : matrix.front().size()));
    for (const auto& row : matrix) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

std::filesystem::path export_family_checkpoint_bundle(
    const std::string& bundle_name,
    const ModelParameters& model) {
    const auto bundle_root = repo_root() / "runs" / bundle_name;
    std::filesystem::create_directories(bundle_root);

    std::ofstream tensor_output(bundle_root / "tensors.txt", std::ios::trunc);
    if (!tensor_output) {
        throw std::runtime_error("failed to create test checkpoint tensor dump");
    }

    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"C\": " << model.C << ",\n";
    manifest << "  \"K_out\": " << model.K_out << ",\n";
    manifest << "  \"L\": " << model.L << ",\n";
    manifest << "  \"checkpoint_prefix\": \"synthetic:" << json_escape(bundle_name) << "\",\n";
    manifest << "  \"d_in_profile\": [";
    for (std::size_t i = 0; i < model.d_in_profile.size(); ++i) {
        if (i != 0) {
            manifest << ", ";
        }
        manifest << model.d_in_profile[i];
    }
    manifest << "],\n";
    manifest << "  \"family_schema_version\": \"multi_layer_multi_head_v2\",\n";
    manifest << "  \"degree_bound_id\": \"auto\",\n";
    manifest << "  \"hidden_profile\": [\n";
    for (std::size_t i = 0; i < model.hidden_layers.size(); ++i) {
        const auto& layer = model.hidden_layers[i];
        manifest << "    {\"head_count\": " << layer.shape.head_count << ", \"head_dim\": " << layer.shape.head_dim << "}";
        manifest << (i + 1 == model.hidden_layers.size() ? "\n" : ",\n");
    }
    manifest << "  ],\n";
    manifest << "  \"hidden_layers\": [\n";
    for (std::size_t i = 0; i < model.hidden_layers.size(); ++i) {
        const auto& layer = model.hidden_layers[i];
        manifest << "    {\"layer_index\": " << layer.layer_index
                 << ", \"input_dim\": " << layer.input_dim
                 << ", \"head_count\": " << layer.shape.head_count
                 << ", \"head_dim\": " << layer.shape.head_dim << "}";
        manifest << (i + 1 == model.hidden_layers.size() ? "\n" : ",\n");
    }
    manifest << "  ],\n";
    manifest << "  \"hidden_head_specs\": [\n";
    std::size_t global_head_index = 0;
    bool first_hidden_spec = true;
    for (const auto& layer : model.hidden_layers) {
        for (std::size_t local_head_index = 0; local_head_index < layer.heads.size(); ++local_head_index, ++global_head_index) {
            const auto base = "hidden/layer" + std::to_string(layer.layer_index) + "/head" + std::to_string(local_head_index);
            const auto seq_name = base + "/seq/kernel";
            const auto dst_kernel_name = base + "/attn_dst/kernel";
            const auto dst_bias_name = base + "/attn_dst/bias";
            const auto src_kernel_name = base + "/attn_src/kernel";
            const auto src_bias_name = base + "/attn_src/bias";
            const auto out_bias_name = base + "/output_bias";
            const auto& head = layer.heads[local_head_index];
            write_tensor_record(&tensor_output, seq_name, {1, layer.input_dim, layer.shape.head_dim}, flatten_seq_kernel(head.seq_kernel_fp));
            write_tensor_record(&tensor_output, dst_kernel_name, {1, layer.shape.head_dim, 1}, head.attn_dst_kernel_fp);
            write_tensor_record(&tensor_output, dst_bias_name, {1}, {head.attn_dst_bias_fp});
            write_tensor_record(&tensor_output, src_kernel_name, {1, layer.shape.head_dim, 1}, head.attn_src_kernel_fp);
            write_tensor_record(&tensor_output, src_bias_name, {1}, {head.attn_src_bias_fp});
            write_tensor_record(&tensor_output, out_bias_name, {layer.shape.head_dim}, head.output_bias_fp);
            if (!first_hidden_spec) {
                manifest << ",\n";
            }
            first_hidden_spec = false;
            manifest << "    {\"layer_index\": " << layer.layer_index
                     << ", \"local_head_index\": " << local_head_index
                     << ", \"global_head_index\": " << global_head_index
                     << ", \"seq_kernel\": \"" << seq_name
                     << "\", \"attn_dst_kernel\": \"" << dst_kernel_name
                     << "\", \"attn_dst_bias\": \"" << dst_bias_name
                     << "\", \"attn_src_kernel\": \"" << src_kernel_name
                     << "\", \"attn_src_bias\": \"" << src_bias_name
                     << "\", \"output_bias\": \"" << out_bias_name << "\"}";
        }
    }
    manifest << "\n  ],\n";
    manifest << "  \"model_arch_id\": \"gat_family_test_bundle\",\n";
    manifest << "  \"model_param_id\": \"" << json_escape(bundle_name) << "\",\n";
    manifest << "  \"output_average_rule\": \"per_head_bias_then_arithmetic_mean\",\n";
    manifest << "  \"output_head_specs\": [\n";
    for (std::size_t head_index = 0; head_index < model.output_layer.heads.size(); ++head_index) {
        const auto base = "output/head" + std::to_string(head_index);
        const auto seq_name = base + "/seq/kernel";
        const auto dst_kernel_name = base + "/attn_dst/kernel";
        const auto dst_bias_name = base + "/attn_dst/bias";
        const auto src_kernel_name = base + "/attn_src/kernel";
        const auto src_bias_name = base + "/attn_src/bias";
        const auto out_bias_name = base + "/output_bias";
        const auto& head = model.output_layer.heads[head_index];
        write_tensor_record(&tensor_output, seq_name, {1, model.output_layer.input_dim, model.output_layer.output_dim}, flatten_seq_kernel(head.seq_kernel_fp));
        write_tensor_record(&tensor_output, dst_kernel_name, {1, model.output_layer.output_dim, 1}, head.attn_dst_kernel_fp);
        write_tensor_record(&tensor_output, dst_bias_name, {1}, {head.attn_dst_bias_fp});
        write_tensor_record(&tensor_output, src_kernel_name, {1, model.output_layer.output_dim, 1}, head.attn_src_kernel_fp);
        write_tensor_record(&tensor_output, src_bias_name, {1}, {head.attn_src_bias_fp});
        write_tensor_record(&tensor_output, out_bias_name, {model.output_layer.output_dim}, head.output_bias_fp);
        manifest << "    {\"head_index\": " << head_index
                 << ", \"seq_kernel\": \"" << seq_name
                 << "\", \"attn_dst_kernel\": \"" << dst_kernel_name
                 << "\", \"attn_dst_bias\": \"" << dst_bias_name
                 << "\", \"attn_src_kernel\": \"" << src_kernel_name
                 << "\", \"attn_src_bias\": \"" << src_bias_name
                 << "\", \"output_bias\": \"" << out_bias_name << "\"}";
        manifest << (head_index + 1 == model.output_layer.heads.size() ? "\n" : ",\n");
    }
    manifest << "  ],\n";
    manifest << "  \"quant_cfg_id\": \"fp32_bundle_export\",\n";
    manifest << "  \"report_unit\": \"node\",\n";
    manifest << "  \"static_table_id\": \"tables:lrelu+elu+exp+range\",\n";
    manifest << "  \"task_type\": \"transductive_node_classification\",\n";
    manifest << "  \"tensor_count\": " << (global_head_index * 6 + model.output_layer.heads.size() * 6) << ",\n";
    manifest << "  \"tensor_index\": {},\n";
    manifest << "  \"output_average_rule_note\": \"final logits are the arithmetic mean of per-head bias-added outputs\"\n";
    manifest << "}\n";

    std::ofstream manifest_output(bundle_root / "manifest.json", std::ios::trunc);
    if (!manifest_output) {
        throw std::runtime_error("failed to create test checkpoint manifest");
    }
    manifest_output << manifest.str();
    return bundle_root;
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
    const auto config = ppi_local_batch_config();
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

const std::filesystem::path& family_checkpoint_bundle_dir() {
    static const auto bundle_dir = export_family_checkpoint_bundle("test_family_checkpoint_bundle", make_family_model(2));
    return bundle_dir;
}

const std::filesystem::path& family_formal_checkpoint_bundle_dir() {
    static const auto bundle_dir = []() {
        const auto model = gatzk::model::build_family_model_parameters({1433, 2}, {{2, 1}, {1, 1}}, 2, 7, 17);
        return export_family_checkpoint_bundle("test_family_formal_checkpoint_bundle", model);
    }();
    return bundle_dir;
}

const std::filesystem::path& ppi_contract_bundle_dir() {
    static const auto bundle_dir = []() {
        const auto model = gatzk::model::build_family_model_parameters({50}, {{1, 8}}, 1, 121, 11);
        const auto root = export_family_checkpoint_bundle("test_ppi_contract_bundle", model);
        auto manifest = slurp_file(root / "manifest.json");
        const auto report_pos = manifest.find("\"report_unit\": \"node\"");
        require(report_pos != std::string::npos, "expected report_unit in PPI contract manifest");
        manifest.replace(report_pos, std::string("\"report_unit\": \"node\"").size(), "\"report_unit\": \"graph\"");
        const auto task_pos = manifest.find("\"task_type\": \"transductive_node_classification\"");
        require(task_pos != std::string::npos, "expected task_type in PPI contract manifest");
        manifest.replace(
            task_pos,
            std::string("\"task_type\": \"transductive_node_classification\"").size(),
            "\"task_type\": \"inductive_multi_graph_node_classification\"");
        const auto note_pos = manifest.find("\"output_average_rule_note\"");
        require(note_pos != std::string::npos, "expected output_average_rule_note in PPI contract manifest");
        manifest.insert(
            note_pos,
            "\"batching_rule\": \"multi_graph_batch\",\n"
            "  \"subgraph_rule\": \"whole_graph\",\n"
            "  \"self_loop_rule\": \"per_node\",\n"
            "  \"edge_sort_rule\": \"edge_gid_then_dst_stable\",\n"
            "  ");
        write_text(root / "manifest.json", manifest);
        return root;
    }();
    return bundle_dir;
}

std::filesystem::path write_benchmark_manifest(
    const std::string& dataset,
    const std::string& benchmark_mode,
    double prove_time_ms,
    double verify_time_ms,
    std::size_t proof_size_bytes,
    std::size_t node_count,
    std::size_t edge_count) {
    const auto root = repo_root() / "runs" / "test_benchmark_export" / dataset;
    std::filesystem::create_directories(root);
    std::ostringstream manifest;
    manifest << "{\n"
             << "  \"dataset\": \"" << dataset << "\",\n"
             << "  \"dataset_name\": \"" << dataset << "\",\n"
             << "  \"backend_name\": \"mcl\",\n"
             << "  \"benchmark_mode\": \"" << benchmark_mode << "\",\n"
             << "  \"route2_label\": \"msm_fft_packed_kernel_layout_pairing\",\n"
             << "  \"fft_backend_route\": \"packed_rotated_kernel\",\n"
             << "  \"commitment_time_ms\": \"10.25\",\n"
             << "  \"prove_time_ms\": \"" << prove_time_ms << "\",\n"
             << "  \"verify_time_ms\": \"" << verify_time_ms << "\",\n"
             << "  \"proof_size_bytes\": \"" << proof_size_bytes << "\",\n"
             << "  \"node_count\": \"" << node_count << "\",\n"
             << "  \"edge_count\": \"" << edge_count << "\",\n"
             << "  \"is_full_dataset\": \"true\",\n"
             << "  \"trace_generation_ms\": \"12.5\",\n"
             << "  \"commit_dynamic_ms\": \"9.5\",\n"
             << "  \"quotient_bundle_pack_ms\": \"0.75\",\n"
             << "  \"quotient_build_ms\": \"15.5\",\n"
             << "  \"domain_opening_ms\": \"18.5\",\n"
             << "  \"external_opening_ms\": \"1.0\",\n"
             << "  \"verify_metadata_ms\": \"0.5\",\n"
             << "  \"verify_transcript_ms\": \"0.6\",\n"
             << "  \"verify_domain_opening_ms\": \"4.0\",\n"
             << "  \"verify_quotient_ms\": \"5.0\",\n"
             << "  \"verify_external_fold_ms\": \"0.8\",\n"
             << "  \"verify_misc_ms\": \"0.3\",\n"
             << "  \"verified\": \"true\",\n"
             << "  \"model_arch_id\": \"test_arch\",\n"
             << "  \"model_param_id\": \"test_params\",\n"
             << "  \"quant_cfg_id\": \"test_quant\",\n"
             << "  \"notes\": \"benchmark_mode=" << benchmark_mode << "; model_source=checkpoint_bundle\"\n"
             << "}\n";
    write_text(root / "run_manifest.json", manifest.str());
    return root / "run_manifest.json";
}

int run_benchmark_export_script(const std::vector<std::string>& arguments) {
    const auto python = repo_root() / ".venv" / "bin" / "python";
    auto shell_escape = [](const std::string& value) {
        std::string escaped = "'";
        for (char ch : value) {
            if (ch == '\'') {
                escaped += "'\\''";
            } else {
                escaped.push_back(ch);
            }
        }
        escaped += "'";
        return escaped;
    };
    std::ostringstream command;
    command << shell_escape(python.string()) << ' ' << shell_escape(repo_path("scripts/export_benchmark_table.py"));
    for (const auto& argument : arguments) {
        command << ' ' << shell_escape(argument);
    }
    return std::system(command.str().c_str());
}

int run_python_script(const std::string& relative_script, const std::vector<std::string>& arguments) {
    const auto python = repo_root() / ".venv" / "bin" / "python";
    auto shell_escape = [](const std::string& value) {
        std::string escaped = "'";
        for (char ch : value) {
            if (ch == '\'') {
                escaped += "'\\''";
            } else {
                escaped.push_back(ch);
            }
        }
        escaped += "'";
        return escaped;
    };
    std::ostringstream command;
    command << shell_escape(python.string()) << ' ' << shell_escape(repo_path(relative_script));
    for (const auto& argument : arguments) {
        command << ' ' << shell_escape(argument);
    }
    return std::system(command.str().c_str());
}

void test_family_checkpoint_bundle_round_trip() {
    const auto info = gatzk::model::inspect_checkpoint_bundle(family_checkpoint_bundle_dir().string());
    require(info.layer_count == 3, "family bundle must preserve L");
    require(info.hidden_layers.size() == 2, "family bundle must preserve hidden layer count");
    require(info.hidden_head_specs.size() == 3, "family bundle must preserve flattened hidden head family");
    require(info.output_head_specs.size() == 2, "family bundle must preserve K_out");

    const auto model = gatzk::model::load_checkpoint_bundle_parameters(family_checkpoint_bundle_dir().string());
    require(model.L == 3, "loaded family bundle must keep L");
    require(model.hidden_layers.size() == 2, "loaded family bundle must keep hidden family");
    require(model.output_layer.heads.size() == 2, "loaded family bundle must keep output family");
    require(model.output_layer.input_dim == 3, "loaded family bundle must keep final concat width");
}

void test_family_checkpoint_manifest_contains_required_fields() {
    std::ifstream input(family_checkpoint_bundle_dir() / "manifest.json");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto manifest = buffer.str();
    for (const auto& key : {
             "\"L\"",
             "\"hidden_profile\"",
             "\"hidden_layers\"",
             "\"hidden_head_specs\"",
             "\"d_in_profile\"",
             "\"K_out\"",
             "\"output_head_specs\"",
             "\"C\"",
             "\"output_average_rule\"",
             "\"model_arch_id\"",
             "\"model_param_id\"",
             "\"quant_cfg_id\"",
         }) {
        require(manifest.find(key) != std::string::npos, std::string("family checkpoint manifest missing key: ") + key);
    }
}

void test_checkpoint_loader_rejects_incomplete_family_metadata() {
    const auto broken_dir = repo_root() / "runs" / "test_family_checkpoint_bundle_broken";
    std::filesystem::create_directories(broken_dir);
    std::filesystem::copy_file(
        family_checkpoint_bundle_dir() / "tensors.txt",
        broken_dir / "tensors.txt",
        std::filesystem::copy_options::overwrite_existing);
    std::ifstream input(family_checkpoint_bundle_dir() / "manifest.json");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    auto manifest = buffer.str();
    const auto begin = manifest.find("\"output_head_specs\"");
    const auto end = manifest.find("],", begin);
    require(begin != std::string::npos && end != std::string::npos, "expected output_head_specs in family manifest");
    manifest.erase(begin, end - begin + 2);
    std::ofstream output(broken_dir / "manifest.json", std::ios::trunc);
    output << manifest;
    output.close();
    const auto error = require_throws([&]() {
        (void)gatzk::model::load_checkpoint_bundle_parameters(broken_dir.string());
    });
    require(error.find("output_head_specs") != std::string::npos, "incomplete family manifest must fail on missing output_head_specs");
}

void test_real_or_fixture_checkpoint_backed_family_validation() {
    const auto model = gatzk::model::load_checkpoint_bundle_parameters(family_formal_checkpoint_bundle_dir().string());
    require(model.L == 3, "fixture checkpoint-backed family bundle must preserve L");
    require(model.hidden_layers.size() == 2, "fixture checkpoint-backed family bundle must preserve hidden layers");
    require(model.K_out == 2, "fixture checkpoint-backed family bundle must preserve K_out");
    require(gatzk::model::hidden_family_dimension_chain_is_valid(model), "fixture checkpoint-backed family model must remain formally supported");

    FloatMatrix features(2, std::vector<double>(1433, 0.0));
    features[0][0] = 1.0;
    features[1][1] = 2.0;
    const std::vector<gatzk::data::Edge> edges = {
        {0, 0, 0, 0},
        {1, 1, 0, 1},
    };
    const auto forward = gatzk::model::forward_note_style(features, edges, model);
    require(forward.hidden_layer_traces.size() == 2, "fixture checkpoint-backed family forward must preserve two hidden layers");
    require(forward.output_head_traces.size() == 2, "fixture checkpoint-backed family forward must preserve two output heads");
}

void test_checkpoint_backed_formal_validation() {
    const auto& fixture = full_cora_proof_fixture();
    require(!fixture.context.config.checkpoint_bundle.empty(), "checkpoint-backed validation requires a real checkpoint bundle");
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "checkpoint-backed cora formal prove/verify must pass");
}

void test_citeseer_checkpoint_backed_formal_validation() {
    const auto& fixture = full_citeseer_proof_fixture();
    require(!fixture.context.config.checkpoint_bundle.empty(), "citeseer validation requires a real checkpoint bundle");
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "checkpoint-backed citeseer formal prove/verify must pass");
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

void test_ppi_real_checkpoint_validation_or_precise_fail_fast() {
    const auto config = ppi_full_formal_config();
    const auto bundle_root = repo_root() / config.checkpoint_bundle;
    if (std::filesystem::exists(bundle_root)) {
        const auto manifest = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
        require(manifest.find("\"dataset\": \"ppi\"") != std::string::npos, "real PPI benchmark manifest must exist");
        require(manifest.find("\"verified\": \"true\"") != std::string::npos, "real PPI checkpoint-backed formal run must verify");
        require(manifest.find("\"is_full_dataset\": \"true\"") != std::string::npos, "real PPI run must use the full dataset");
        return;
    }
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_context(config);
    });
    require(error.find("checkpoint bundle path does not exist") != std::string::npos, "missing real PPI checkpoint must fail fast precisely");
}

void test_ppi_bundle_import_or_generation_path_is_unique() {
    require(std::filesystem::exists(repo_root() / "scripts" / "import_ppi_bundle.py"), "PPI import path script must exist");
    const auto install_root = repo_root() / "runs" / "test_imported_ppi_bundle";
    std::filesystem::remove_all(install_root);
    const auto torch_checkpoint = repo_root() / "runs" / "ppi_train" / "best_model.pt";
    const std::vector<std::string> arguments = std::filesystem::exists(torch_checkpoint)
        ? std::vector<std::string>{"--torch-checkpoint", torch_checkpoint.string(), "--output-dir", install_root.string()}
        : std::vector<std::string>{"--bundle-dir", ppi_contract_bundle_dir().string(), "--output-dir", install_root.string()};
    require(
        run_python_script("scripts/import_ppi_bundle.py", arguments) == 0,
        "PPI import script must accept the unique formal import source");
    require(std::filesystem::exists(install_root / "manifest.json"), "PPI import script must install manifest.json");
    require(std::filesystem::exists(install_root / "tensors.txt"), "PPI import script must install tensors.txt");
}

void test_ppi_checkpoint_backed_formal_validation_if_bundle_present() {
    const auto config = ppi_full_formal_config();
    const auto bundle_root = repo_root() / config.checkpoint_bundle;
    if (!std::filesystem::exists(bundle_root)) {
        return;
    }
    const auto manifest = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "real PPI checkpoint-backed formal manifest must record verification success");
}

void test_ppi_precise_external_bundle_blocker_contract() {
    const auto broken_root = repo_root() / "runs" / "test_ppi_bundle_broken";
    std::filesystem::remove_all(broken_root);
    std::filesystem::create_directories(broken_root);
    write_text(broken_root / "manifest.json", "{\"family_schema_version\":\"multi_layer_multi_head_v2\"}\n");
    const auto install_root = repo_root() / "runs" / "test_ppi_bundle_broken_install";
    std::filesystem::remove_all(install_root);
    require(
        run_python_script(
            "scripts/import_ppi_bundle.py",
            {
                "--bundle-dir", broken_root.string(),
                "--output-dir", install_root.string(),
            }) != 0,
        "broken PPI bundle must be rejected");
    auto config = ppi_full_formal_config();
    config.checkpoint_bundle = "artifacts/checkpoints/ppi_gat_missing_for_contract";
    const auto error = require_throws([&]() {
        (void)gatzk::protocol::build_context(config);
    });
    require(error.find("checkpoint bundle path does not exist") != std::string::npos, "missing real PPI bundle blocker must remain precise");
}

void test_benchmark_table_updates_after_ppi_or_blocker_resolution() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"ppi\"") != std::string::npos, "benchmark table must contain a PPI row");
    require(
        latest.find("\"status\": \"ok\"") != std::string::npos
            || latest.find("\"status\": \"blocked\"") != std::string::npos,
        "benchmark table must expose explicit PPI status");
}

void test_pubmed_hotspot_optimization_no_regression() {
    const auto single_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "run_manifest.json");
    const auto warm_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(warm_text.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "pubmed warm manifest must exist");
    require(
        extract_json_number(warm_text, "prove_time_ms") < extract_json_number(single_text, "prove_time_ms"),
        "pubmed warm prove time must remain below single-run baseline");
    require(
        extract_json_number(warm_text, "trace_generation_ms") < extract_json_number(single_text, "trace_generation_ms"),
        "pubmed warm trace_generation_ms must remain below single-run baseline");
}

void test_benchmark_export_pipeline_remains_single_source_of_truth() {
    require(std::filesystem::exists(repo_root() / "scripts" / "export_benchmark_table.py"), "benchmark export script must exist");
    require(!std::filesystem::exists(repo_root() / "runs" / "benchmarks" / "summary.txt"), "legacy benchmark summary.txt must stay removed");
    const auto summary = slurp_file(repo_root() / "runs" / "benchmarks" / "summary.md");
    require(summary.find("最新基准结果") != std::string::npos, "summary.md must remain the benchmark single source of truth");
}

void test_performance_regression_guard_for_cora_citeseer_pubmed() {
    const auto cora_single = slurp_file(repo_root() / "runs" / "cora_full" / "run_manifest.json");
    const auto cora_warm = slurp_file(repo_root() / "runs" / "cora_full" / "warm" / "run_manifest.json");
    const auto citeseer_single = slurp_file(repo_root() / "runs" / "citeseer_full" / "run_manifest.json");
    const auto citeseer_warm = slurp_file(repo_root() / "runs" / "citeseer_full" / "warm" / "run_manifest.json");
    const auto pubmed_single = slurp_file(repo_root() / "runs" / "pubmed_full" / "run_manifest.json");
    const auto pubmed_warm = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(cora_warm.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "cora warm benchmark must exist");
    require(citeseer_warm.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "citeseer warm benchmark must exist");
    require(pubmed_warm.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "pubmed warm benchmark must exist");
    require(
        extract_json_number(cora_warm, "prove_time_ms") < extract_json_number(cora_single, "prove_time_ms"),
        "cora warm prove time must remain below single-run baseline");
    require(
        extract_json_number(citeseer_warm, "prove_time_ms") < extract_json_number(citeseer_single, "prove_time_ms"),
        "citeseer warm prove time must remain below single-run baseline");
    require(
        extract_json_number(pubmed_warm, "prove_time_ms") < extract_json_number(pubmed_single, "prove_time_ms"),
        "pubmed warm prove time must remain below single-run baseline");
}

void test_trace_cache_helpers_have_safe_lifetimes() {
    const auto context = synthetic_multioutput_proof_fixture().context;
    const auto trace_a = gatzk::protocol::build_trace(context);
    const auto proof_a = gatzk::protocol::prove(context, trace_a);
    require(gatzk::protocol::verify(context, proof_a), "first trace build must verify");
    const auto trace_b = gatzk::protocol::build_trace(context);
    const auto proof_b = gatzk::protocol::prove(context, trace_b);
    require(gatzk::protocol::verify(context, proof_b), "second trace build must verify");
}

void test_four_dataset_benchmark_table_export() {
    const auto output_dir = repo_root() / "runs" / "benchmarks_test";
    std::filesystem::remove_all(output_dir);
    const auto cora_manifest = write_benchmark_manifest("cora", "single", 10.5, 1.5, 111, 2708, 13264);
    const auto citeseer_manifest = write_benchmark_manifest("citeseer", "single", 20.5, 2.5, 222, 3327, 12431);
    const auto pubmed_manifest = write_benchmark_manifest("pubmed", "single", 30.5, 3.5, 333, 19717, 108365);
    require(
        run_benchmark_export_script(
            {
                "--run", "cora=" + cora_manifest.string(),
                "--run", "citeseer=" + citeseer_manifest.string(),
                "--run", "pubmed=" + pubmed_manifest.string(),
                "--blocked", "ppi=missing real checkpoint bundle",
                "--output-dir", output_dir.string(),
            }) == 0,
        "benchmark export script must succeed");
    require(std::filesystem::exists(output_dir / "latest.json"), "latest.json must be exported");
    require(std::filesystem::exists(output_dir / "latest.csv"), "latest.csv must be exported");
    require(std::filesystem::exists(output_dir / "summary.md"), "summary.md must be exported");
    const auto summary = slurp_file(output_dir / "summary.md");
    require(summary.find("| cora | 已完成 | 10.250 | 10.500 | 1.500 | 111 |") != std::string::npos, "summary must include cora row");
    require(summary.find("| ppi | 阻塞 | n/a | n/a | n/a | n/a |") != std::string::npos, "summary must include blocked ppi row");
}

void test_benchmark_summary_contains_required_metrics() {
    const auto output_dir = repo_root() / "runs" / "benchmarks_test";
    const auto json_text = slurp_file(output_dir / "latest.json");
    require(json_text.find("\"commitment_time_ms\"") != std::string::npos, "latest.json must include commitment_time_ms");
    require(json_text.find("\"prove_time_ms\"") != std::string::npos, "latest.json must include prove_time_ms");
    require(json_text.find("\"verify_time_ms\"") != std::string::npos, "latest.json must include verify_time_ms");
    require(json_text.find("\"proof_size_bytes\"") != std::string::npos, "latest.json must include proof_size_bytes");
    const auto csv_text = slurp_file(output_dir / "latest.csv");
    require(csv_text.find("commitment_time_ms") != std::string::npos, "latest.csv must include commitment_time_ms");
    require(csv_text.find("prove_time_ms") != std::string::npos, "latest.csv must include prove_time_ms");
    require(csv_text.find("verify_time_ms") != std::string::npos, "latest.csv must include verify_time_ms");
    require(csv_text.find("proof_size_bytes") != std::string::npos, "latest.csv must include proof_size_bytes");
}

void test_benchmark_mode_consistency() {
    const auto output_dir = repo_root() / "runs" / "benchmarks_test_inconsistent";
    std::filesystem::remove_all(output_dir);
    const auto cora_manifest = write_benchmark_manifest("cora", "single", 10.5, 1.5, 111, 2708, 13264);
    const auto citeseer_manifest = write_benchmark_manifest("citeseer", "warm", 20.5, 2.5, 222, 3327, 12431);
    require(
        run_benchmark_export_script(
            {
                "--run", "cora=" + cora_manifest.string(),
                "--run", "citeseer=" + citeseer_manifest.string(),
                "--output-dir", output_dir.string(),
            }) != 0,
        "mixed benchmark modes must be rejected");
}

void test_performance_regression_guard_for_existing_paths() {
    const auto manifest = slurp_file(repo_root() / "runs" / "test_manifest_export" / "run_manifest.json");
    require(manifest.find("\"benchmark_mode\"") != std::string::npos, "run manifests must expose benchmark_mode");
    require(manifest.find("\"route2_label\"") != std::string::npos, "run manifests must expose route2_label");
    require(manifest.find("\"trace_generation_ms\"") != std::string::npos, "run manifests must expose trace_generation_ms");
    require(manifest.find("\"quotient_build_ms\"") != std::string::npos, "run manifests must expose quotient_build_ms");
}

void test_no_legacy_benchmark_pipeline_remaining() {
    const auto& fixture = full_cora_proof_fixture();
    gatzk::protocol::RunMetrics metrics;
    metrics.backend_name = "test";
    metrics.config = repo_path("configs/cora_full.cfg");
    metrics.dataset = fixture.context.dataset.name;
    metrics.node_count = fixture.context.local.num_nodes;
    metrics.edge_count = fixture.context.local.edges.size();
    metrics.proof_size_bytes = gatzk::protocol::proof_size_bytes(fixture.proof);
    metrics.benchmark_mode = "single";
    metrics.route2_label = "msm_fft_packed_kernel_layout_pairing";

    auto export_context = fixture.context;
    export_context.config.export_dir = "runs/test_benchmark_export_pipeline";
    std::filesystem::remove_all(repo_root() / export_context.config.export_dir);
    gatzk::protocol::export_run_artifacts(export_context, fixture.trace, fixture.proof, metrics, true);
    require(!std::filesystem::exists(repo_root() / export_context.config.export_dir / "summary.txt"), "legacy summary.txt export must be removed");
    require(std::filesystem::exists(repo_root() / export_context.config.export_dir / "benchmark.txt"), "benchmark.txt export must remain");
    require(std::filesystem::exists(repo_root() / export_context.config.export_dir / "run_manifest.json"), "run_manifest export must remain");
}

void test_ppi_training_or_import_mainline_is_real_and_unique() {
    require(std::filesystem::exists(repo_root() / "scripts" / "import_ppi_bundle.py"), "PPI import script must exist");
    require(!std::filesystem::exists(repo_root() / "configs" / "ppi_batch.cfg"), "legacy synthetic PPI config must be removed");
    const auto import_script = slurp_file(repo_root() / "scripts" / "import_ppi_bundle.py");
    require(import_script.find("--tf-checkpoint-prefix") != std::string::npos, "PPI import mainline must accept upstream TensorFlow checkpoints");
    require(import_script.find("--torch-checkpoint") != std::string::npos, "PPI import mainline must accept real torch checkpoints through the same importer");
}

void test_four_full_dataset_benchmark_contract() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"cora\"") != std::string::npos, "latest benchmark must include cora");
    require(latest.find("\"dataset\": \"citeseer\"") != std::string::npos, "latest benchmark must include citeseer");
    require(latest.find("\"dataset\": \"pubmed\"") != std::string::npos, "latest benchmark must include pubmed");
    require(latest.find("\"dataset\": \"ppi\"") != std::string::npos, "latest benchmark must include ppi");
    require(latest.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "latest benchmark must keep a unified warm mode");
    const auto ppi_pos = latest.find("\"dataset\": \"ppi\"");
    require(ppi_pos != std::string::npos, "ppi row must exist");
    require(latest.find("\"status\": \"ok\"", ppi_pos) != std::string::npos, "ppi row must resolve to an explicit success state");
    require(latest.find("\"is_full_dataset\": true") != std::string::npos, "benchmark table must record full-dataset rows");
}

void test_real_gat_forward_semantics_no_regression() {
    auto data_config = ppi_local_batch_config();
    data_config.dataset = "cora";
    data_config.task_type = "transductive_node_classification";
    data_config.report_unit = "node";
    data_config.batching_rule = "whole_graph_single";
    data_config.subgraph_rule = "sampled_subgraph";
    data_config.batch_graphs = 1;
    data_config.local_nodes = 64;
    const auto dataset = gatzk::data::load_dataset(data_config);
    const auto local = gatzk::data::extract_local_subgraph(dataset, 0, 64);
    const auto model = gatzk::model::load_checkpoint_bundle_parameters(repo_path("artifacts/checkpoints/cora_gat"));
    const auto forward = gatzk::model::forward_reference_style(
        local.features_fp,
        local.edges,
        model);
    require(!forward.hidden_layer_traces.empty(), "real GAT forward must preserve hidden attention layers");
    require(forward.hidden_layer_traces.front().concat.size() == local.num_nodes, "hidden ELU+concat output must exist");
    require(forward.output_head_traces.size() == model.output_layer.heads.size(), "output attention head family must remain intact");
    require(forward.Y.size() == local.num_nodes, "final logits must keep real GAT output shape");
}

void test_chinese_readme_is_current_mainline_only() {
    const auto readme = slurp_file(repo_root() / "README.md");
    require(readme.find("项目简介") != std::string::npos, "README must be Chinese and current");
    require(readme.find("从零开始到最新结果的完整运行步骤") != std::string::npos, "README must document the current mainline");
    require(readme.find("summary.txt") == std::string::npos, "README must not mention removed summary.txt");
    require(readme.find("synthetic") == std::string::npos, "README must not keep synthetic fallback instructions");
    require(readme.find("Benchmark Summary") == std::string::npos, "README must not keep old English benchmark wording");
}

void test_benchmark_summary_and_readme_consistency() {
    const auto readme = slurp_file(repo_root() / "README.md");
    const auto summary = slurp_file(repo_root() / "runs" / "benchmarks" / "summary.md");
    require(readme.find("当前最新最快实验结果") != std::string::npos, "README must keep the latest-results section");
    require(summary.find("最新基准结果") != std::string::npos, "summary must keep the benchmark summary section");
    require(readme.find("warm") != std::string::npos && summary.find("warm") != std::string::npos, "README and benchmark summary must keep the same warm official mode");
    require(readme.find("Cora") != std::string::npos && summary.find("cora") != std::string::npos, "README and benchmark summary must both cover cora");
    require(readme.find("Citeseer") != std::string::npos && summary.find("citeseer") != std::string::npos, "README and benchmark summary must both cover citeseer");
    require(readme.find("Pubmed") != std::string::npos && summary.find("pubmed") != std::string::npos, "README and benchmark summary must both cover pubmed");
    require(readme.find("PPI") != std::string::npos && summary.find("ppi") != std::string::npos, "README and benchmark summary must both cover ppi");
}

void test_pubmed_verify_and_opening_hotspot_no_regression() {
    const auto single_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "run_manifest.json");
    const auto warm_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(
        extract_json_number(warm_text, "verify_time_ms") < extract_json_number(single_text, "verify_time_ms"),
        "pubmed warm verify time must remain below single-run baseline");
    require(
        extract_json_number(warm_text, "domain_opening_ms") < extract_json_number(single_text, "domain_opening_ms"),
        "pubmed warm domain_opening_ms must remain below single-run baseline");
}

void test_no_legacy_benchmark_or_import_pipeline_remaining() {
    require(!std::filesystem::exists(repo_root() / "configs" / "ppi_batch.cfg"), "legacy synthetic PPI config must be removed");
    require(!std::filesystem::exists(repo_root() / "runs" / "benchmarks" / "summary.txt"), "legacy benchmark summary.txt must stay removed");
    require(std::filesystem::exists(repo_root() / "scripts" / "import_ppi_bundle.py"), "current PPI import pipeline must exist");
    require(std::filesystem::exists(repo_root() / "scripts" / "export_benchmark_table.py"), "current benchmark export pipeline must exist");
    require(!std::filesystem::exists(repo_root() / "scripts" / "train_ppi_gat.py"), "legacy in-project PPI training mainline must be removed");
}

void test_upstream_ppi_training_is_only_parameter_source() {
    require(std::filesystem::exists(std::filesystem::path("/home/pzh/GAT-for-PPI") / "execute_inductive.py"), "upstream GAT-for-PPI training source must exist");
    const auto readme = slurp_file(repo_root() / "README.md");
    require(readme.find("上游真实参数来源") != std::string::npos, "README must mark upstream PPI training as parameter source only");
    require(readme.find("formal / benchmark 运行主线") != std::string::npos, "README must separate upstream training from the formal mainline");
}

void test_ppi_import_pipeline_is_single_official_path() {
    const auto import_script = slurp_file(repo_root() / "scripts" / "import_ppi_bundle.py");
    require(import_script.find("def main()") != std::string::npos, "official PPI importer must exist");
    require(!std::filesystem::exists(repo_root() / "configs" / "ppi_batch.cfg"), "legacy PPI config must not survive");
    require(!std::filesystem::exists(repo_root() / "scripts" / "train_ppi_gat.py"), "redundant in-project PPI training path must be removed");
}

void test_formal_benchmark_excludes_training_time() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("wall_time_sec") == std::string::npos, "benchmark table must not include training wall time");
    require(latest.find("train_loss") == std::string::npos, "benchmark table must not include training metrics");
    require(latest.find("train_micro_f1") == std::string::npos, "benchmark table must not include training F1 metrics");
}

void test_all_full_configs_are_real_checkpoint_backed() {
    for (const auto& path : {
             repo_root() / "configs" / "cora_full.cfg",
             repo_root() / "configs" / "citeseer_full.cfg",
             repo_root() / "configs" / "pubmed_full.cfg",
             repo_root() / "configs" / "ppi_batch_formal.cfg",
             repo_root() / "configs" / "ogbn_arxiv_full.cfg",
         }) {
        const auto text = slurp_file(path);
        require(text.find("checkpoint_bundle = ") != std::string::npos, "full config must pin a real checkpoint bundle");
        require(text.find("allow_synthetic_model = true") == std::string::npos, "full config must not allow synthetic fallback");
    }
}

void test_real_gat_forward_semantics_preserved_after_import() {
    test_real_gat_forward_semantics_no_regression();
}

void test_chinese_readme_explains_training_vs_zkml_boundary() {
    const auto readme = slurp_file(repo_root() / "README.md");
    require(readme.find("训练与 ZKML 的职责边界") != std::string::npos, "README must explain the training/ZKML boundary");
    require(readme.find("训练时间、训练日志、训练超参数调优都不进入 formal benchmark 主表") != std::string::npos, "README must exclude training time from formal benchmark");
    require(readme.find("当前 project 的正式主线只负责") != std::string::npos, "README must explain the formal mainline scope");
}

void test_no_legacy_ppi_training_or_import_dual_mainline() {
    require(!std::filesystem::exists(repo_root() / "scripts" / "train_ppi_gat.py"), "legacy local PPI training mainline must be removed");
    require(!std::filesystem::exists(repo_root() / "configs" / "ppi_batch.cfg"), "legacy synthetic PPI config must be removed");
    const auto readme = slurp_file(repo_root() / "README.md");
    require(readme.find("第二正式主线") == std::string::npos, "README must not describe a dual mainline");
}

void test_ogbn_arxiv_reference_template_is_internalized_into_project_mainline() {
    require(std::filesystem::exists(repo_root() / "reference" / "gat_ogbn_arxiv" / "pyg_gat_example.py"), "ogbn-arxiv reference template must exist inside project");
    require(std::filesystem::exists(repo_root() / "scripts" / "train_ogbn_arxiv_gat.py"), "ogbn-arxiv training entry must exist inside project");
    require(std::filesystem::exists(repo_root() / "scripts" / "ogbn_arxiv_utils.py"), "ogbn-arxiv local utility module must exist inside project");
    const auto train_script = slurp_file(repo_root() / "scripts" / "train_ogbn_arxiv_gat.py");
    const auto utils_script = slurp_file(repo_root() / "scripts" / "ogbn_arxiv_utils.py");
    require(train_script.find("../ogb") == std::string::npos, "ogbn-arxiv training entry must not depend on sibling ogb paths at runtime");
    require(utils_script.find("../ogb") == std::string::npos, "ogbn-arxiv dataset utils must not depend on sibling ogb paths at runtime");
}

void test_ogbn_arxiv_local_dataset_is_consumed_without_redownload() {
    require(std::filesystem::exists(repo_root() / "data" / "ogbn-arxiv" / "raw" / "node-feat.csv.gz"), "local ogbn-arxiv features must already exist");
    require(std::filesystem::exists(repo_root() / "data" / "ogbn-arxiv" / "split" / "time" / "train.csv.gz"), "local ogbn-arxiv split files must already exist");
    require(std::filesystem::exists(repo_root() / "data" / "cache" / "ogbn_arxiv" / "meta.cfg"), "ogbn-arxiv cache must be prepared from local files");
    const auto prepare_script = slurp_file(repo_root() / "scripts" / "prepare_ogbn_arxiv.py");
    require(prepare_script.find("download") == std::string::npos, "ogbn-arxiv preparation must not redownload data");
}

void test_ogbn_arxiv_training_entry_produces_real_checkpoint() {
    require(std::filesystem::exists(ogbn_arxiv_bundle_dir() / "best_model.pt"), "ogbn-arxiv training must produce a real checkpoint");
    require(std::filesystem::exists(ogbn_arxiv_bundle_dir() / "training_summary.json"), "ogbn-arxiv training must export a training summary");
    const auto summary = slurp_file(ogbn_arxiv_bundle_dir() / "training_summary.json");
    require(summary.find("\"dataset\": \"ogbn-arxiv\"") != std::string::npos, "training summary must identify ogbn-arxiv");
    require(summary.find("\"best_metrics\"") != std::string::npos, "training summary must include best metrics");
}

void test_ogbn_arxiv_bundle_is_checkpoint_backed_and_formal_ready() {
    require(std::filesystem::exists(ogbn_arxiv_bundle_dir() / "manifest.json"), "ogbn-arxiv bundle manifest must exist");
    require(std::filesystem::exists(ogbn_arxiv_bundle_dir() / "tensors.txt"), "ogbn-arxiv bundle tensors must exist");
    const auto info = gatzk::model::inspect_checkpoint_bundle(ogbn_arxiv_bundle_dir().string());
    require(info.layer_count == 2, "ogbn-arxiv bundle must preserve L=2");
    require(info.hidden_layers.size() == 1, "ogbn-arxiv bundle must preserve one hidden layer");
    require(info.output_head_specs.size() == 1, "ogbn-arxiv bundle must preserve K_out=1");
    const auto model = gatzk::model::load_checkpoint_bundle_parameters(ogbn_arxiv_bundle_dir().string());
    require(!model.d_in_profile.empty() && model.d_in_profile.front() == 128, "ogbn-arxiv bundle must preserve input width 128");
    require(model.C == 40, "ogbn-arxiv bundle must preserve class count 40");
}

void test_ogbn_arxiv_full_config_is_real_checkpoint_backed() {
    const auto config = ogbn_arxiv_full_formal_config();
    require(config.dataset == "ogbn_arxiv", "ogbn-arxiv full config must target ogbn_arxiv");
    require(config.checkpoint_bundle == "artifacts/checkpoints/ogbn_arxiv_gat", "ogbn-arxiv full config must pin the real bundle");
    require(!config.allow_synthetic_model, "ogbn-arxiv full config must not allow synthetic fallback");
    require(config.prove_enabled, "ogbn-arxiv full config must keep proving enabled");
    require(config.batching_rule == "whole_graph_single", "ogbn-arxiv full config must stay whole-graph");
}

void test_ogbn_arxiv_formal_benchmark_runs_or_fails_with_precise_reason() {
    if (std::filesystem::exists(ogbn_arxiv_warm_manifest_path())) {
        const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
        require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv warm formal run must verify if it completes");
        return;
    }
    require(std::filesystem::exists(ogbn_arxiv_warm_log_path()), "ogbn-arxiv warm attempt must leave a diagnostic log");
    const auto log = slurp_file(ogbn_arxiv_warm_log_path());
    const bool has_precise_blocker =
        log.find("Command terminated by signal 9") != std::string::npos
        || log.find("lookup query escaped static table for P_h0_R") != std::string::npos
        || log.find("pair lookup query escaped static table for P_h0_ELU") != std::string::npos
        || log.find("missing verifier evaluation for P_Q_tbl_feat") != std::string::npos;
    require(has_precise_blocker, "ogbn-arxiv warm blocker must report a precise formal failure");
    require(log.find("Maximum resident set size (kbytes):") != std::string::npos
            || log.find("stage=trace_complete") != std::string::npos,
        "ogbn-arxiv warm blocker must record peak-memory or full-trace progress");
}

void test_ogbn_arxiv_formal_whole_graph_memory_blocker_is_reduced_or_precisely_reported() {
    require(std::filesystem::exists(ogbn_arxiv_warm_log_path()), "ogbn-arxiv warm log must exist");
    const auto log = slurp_file(ogbn_arxiv_warm_log_path());
    if (std::filesystem::exists(ogbn_arxiv_warm_manifest_path())) {
        require(log.find("stage=trace_complete") != std::string::npos, "successful ogbn-arxiv warm run must cross trace completion");
        return;
    }
    require(log.find("stage=after_forward") != std::string::npos, "ogbn-arxiv memory fix must at least progress past forward");
    require(
        log.find("lookup query escaped static table for P_h0_R") != std::string::npos
            || log.find("pair lookup query escaped static table for P_h0_ELU") != std::string::npos
            || log.find("missing verifier evaluation for P_Q_tbl_feat") != std::string::npos
            || log.find("Command terminated by signal 9") != std::string::npos,
        "ogbn-arxiv warm log must preserve a precise blocker if the run still fails");
}

void test_ogbn_arxiv_run_manifest_is_emitted_after_memory_fix() {
    if (!std::filesystem::exists(ogbn_arxiv_warm_manifest_path())) {
        const auto log = slurp_file(ogbn_arxiv_warm_log_path());
        require(log.find("stage=trace_complete") != std::string::npos, "missing ogbn-arxiv manifest is only acceptable after a fully traced run");
        return;
    }
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"dataset_name\": \"ogbn-arxiv\"") != std::string::npos, "ogbn-arxiv run manifest must identify the dataset");
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv run manifest must record verified=true");
}

void test_ogbn_arxiv_peak_memory_optimization_does_not_change_transcript() {
    const auto trace_source = slurp_file(repo_root() / "src" / "protocol" / "trace.cpp");
    require(trace_source.find("transcript.challenge(\"eta_feat\")") != std::string::npos, "ogbn-arxiv memory optimization must keep the feature lookup transcript challenges");
    require(trace_source.find("beta_feat") != std::string::npos, "ogbn-arxiv memory optimization must keep the feature lookup accumulator challenge");
    require(trace_source.find("crypto::KZG::commit_tau_evaluation(\"P_Table_feat\"") != std::string::npos, "ogbn-arxiv memory optimization must keep the lazy FH commitment path");
}

void test_ogbn_arxiv_does_not_reintroduce_dual_representations_without_need() {
    const auto gat_source = slurp_file(repo_root() / "src" / "model" / "gat.cpp");
    const auto trace_source = slurp_file(repo_root() / "src" / "protocol" / "trace.cpp");
    require(gat_source.find("trace.bias = build_attention_bias_matrix") == std::string::npos, "ogbn-arxiv mainline must not rebuild dense bias matrices");
    require(trace_source.find("use_lazy_full_feature_lookup_trace") != std::string::npos, "ogbn-arxiv mainline must keep the lazy full feature lookup path");
}

void test_ogbn_arxiv_domain_open_edge_hotspot_improves_without_semantic_regression() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv optimization must keep VERIFY_OK");
    require(
        extract_json_number(manifest, "domain_open_edge_ms") < kOgbnArxivBaselineDomainOpenEdgeMs,
        "ogbn-arxiv domain_open_edge_ms must improve over the pre-optimization baseline");
}

void test_ogbn_arxiv_domain_open_fh_hotspot_improves_without_semantic_regression() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv FH optimization must keep VERIFY_OK");
    require(
        extract_json_number(manifest, "domain_open_FH_ms") < kOgbnArxivBaselineDomainOpenFhMs,
        "ogbn-arxiv domain_open_FH_ms must improve over the pre-optimization baseline");
}

void test_ogbn_arxiv_quotient_edge_or_fh_hotspot_improves_without_checkpoint_regression() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv quotient optimization must keep VERIFY_OK");
    const auto quotient_t_edge = extract_json_number(manifest, "quotient_t_edge_ms");
    const auto quotient_t_fh = extract_json_number(manifest, "quotient_t_fh_ms");
    require(
        quotient_t_edge < kOgbnArxivBaselineQuotientTEdgeMs
            || quotient_t_fh < kOgbnArxivBaselineQuotientTFhMs,
        "ogbn-arxiv quotient hot path must improve on t_edge or t_FH");
}

void test_ogbn_arxiv_dynamic_domain_convert_improves_without_checkpoint_regression() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv dynamic commitment optimization must keep VERIFY_OK");
    const auto dynamic_domain_convert = extract_json_number(manifest, "dynamic_domain_convert_ms");
    if (dynamic_domain_convert < kOgbnArxivBaselineDynamicDomainConvertMs) {
        return;
    }
    require(
        extract_json_number(manifest, "prove_time_ms") < kOgbnArxivBaselineProveMs,
        "if dynamic_domain_convert_ms is not improved yet, ogbn-arxiv prove_time_ms must still improve");
}

void test_ogbn_arxiv_prove_path_optimization_does_not_change_transcript() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv transcript-preserving optimization must keep verified=true");
    require(extract_json_number(manifest, "proof_size_bytes") == 9049.0, "ogbn-arxiv proof size must stay on the same proof shape");
    const auto prover_source = slurp_file(repo_root() / "src" / "protocol" / "prover.cpp");
    require(prover_source.find("proof_block_order()") != std::string::npos, "ogbn-arxiv prover optimization must keep the shared proof block order");
}

void test_latest_note_contract_is_not_violated_by_ogbn_arxiv_optimization() {
    const auto note = slurp_file(repo_root() / "GAT-ZKML-多层多头.md");
    const auto config = ogbn_arxiv_full_formal_config();
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(note.find("隐藏层") != std::string::npos, "latest note must remain present");
    require(config.batching_rule == "whole_graph_single", "ogbn-arxiv optimization must keep whole_graph_single");
    require(config.task_type == "transductive_node_classification", "ogbn-arxiv optimization must keep task_type");
    require(config.report_unit == "node", "ogbn-arxiv optimization must keep report_unit");
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv optimization must keep VERIFY_OK");
    require(extract_json_number(manifest, "proof_size_bytes") == 9049.0, "ogbn-arxiv optimization must keep the same proof shape");
}

void test_ogbn_arxiv_commitment_time_is_exported_with_formal_consistent_definition() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    const auto commitment_time_ms = extract_commitment_time_ms_from_manifest(manifest);
    const auto expected = extract_json_number(manifest, "commit_dynamic_ms")
        + extract_json_number(manifest, "quotient_bundle_pack_ms");
    require(std::abs(commitment_time_ms - expected) < 5e-3, "commitment_time_ms must equal commit_dynamic_ms + quotient_bundle_pack_ms within manifest rounding");

    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"ogbn-arxiv\"") != std::string::npos, "latest benchmark table must include ogbn-arxiv");
    require(latest.find("\"commitment_time_ms\":") != std::string::npos, "benchmark table must export commitment_time_ms");
}

void test_ogbn_arxiv_official_metrics_table_contains_four_required_fields() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    const auto csv = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.csv");
    const auto summary = slurp_file(repo_root() / "runs" / "benchmarks" / "summary.md");
    require(latest.find("\"commitment_time_ms\":") != std::string::npos, "latest.json must contain commitment_time_ms");
    require(latest.find("\"prove_time_ms\":") != std::string::npos, "latest.json must contain prove_time_ms");
    require(latest.find("\"verify_time_ms\":") != std::string::npos, "latest.json must contain verify_time_ms");
    require(latest.find("\"proof_size_bytes\":") != std::string::npos, "latest.json must contain proof_size_bytes");
    require(csv.find("commitment_time_ms") != std::string::npos, "latest.csv must contain commitment_time_ms");
    require(summary.find("commitment_time_ms") != std::string::npos, "summary.md must contain commitment_time_ms");
}

void test_ogbn_arxiv_single_dataset_incremental_optimization_does_not_change_transcript() {
    test_ogbn_arxiv_prove_path_optimization_does_not_change_transcript();
    const auto quotients = slurp_file(repo_root() / "src" / "protocol" / "quotients.cpp");
    require(quotients.find("proof_block_order()") == std::string::npos, "edge quotient optimization must stay implementation-local");
}

void test_ogbn_arxiv_official_result_is_not_stale_or_mixed() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(manifest.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "ogbn-arxiv manifest must stay warm");
    require(latest.find("\"dataset\": \"ogbn-arxiv\"") != std::string::npos, "latest table must include ogbn-arxiv");
    require(latest.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "latest table must stay on warm mode");
    const auto ogbn_row = extract_dataset_row(latest, "ogbn-arxiv");
    require(
        std::abs(extract_json_number(ogbn_row, "prove_time_ms") - extract_json_number(manifest, "prove_time_ms")) < 1e-3,
        "latest table must use the current ogbn-arxiv manifest");
}

void test_no_unused_optimization_shell_left_in_final_code() {
    const auto prover_source = slurp_file(repo_root() / "src" / "protocol" / "prover.cpp");
    const auto quotient_source = slurp_file(repo_root() / "src" / "protocol" / "quotients.cpp");
    require(prover_source.find("ogbn_arxiv_edge_hotspot_experiment") == std::string::npos, "unused ogbn-arxiv hotspot experiment shell must not remain");
    require(quotient_source.find("TODO_OGBN_HOTPATH") == std::string::npos, "unused ogbn-arxiv TODO shell must not remain");
}

void test_five_dataset_official_table_is_same_build_same_mainline() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    for (const auto& dataset : {"cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"}) {
        require(latest.find("\"dataset\": \"" + std::string(dataset) + "\"") != std::string::npos, "latest benchmark table must include " + std::string(dataset));
    }
    require(latest.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "five-dataset table must keep the official warm mode");
}

void test_benchmark_table_excludes_training_time_for_ogbn_arxiv() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"ogbn-arxiv\"") != std::string::npos, "benchmark table must include ogbn-arxiv");
    require(latest.find("wall_time_sec") == std::string::npos, "benchmark table must not include ogbn-arxiv training time");
    require(latest.find("train_loss") == std::string::npos, "benchmark table must not include ogbn-arxiv training loss");
}

void test_no_legacy_loader_or_manifest_path_reintroduced_by_ogbn_arxiv() {
    const auto loader = slurp_file(repo_root() / "src" / "data" / "loader.cpp");
    const auto train_script = slurp_file(repo_root() / "scripts" / "train_ogbn_arxiv_gat.py");
    const auto bundle_manifest = slurp_file(ogbn_arxiv_bundle_dir() / "manifest.json");
    require(loader.find("prepare_ogbn_arxiv.py") != std::string::npos, "ogbn-arxiv loader path must stay on the internal prepare script");
    require(train_script.find("../ogb") == std::string::npos, "ogbn-arxiv training entry must not reintroduce sibling-ogb runtime dependency");
    require(bundle_manifest.find("\"family_schema_version\": \"multi_layer_multi_head_v2\"") != std::string::npos, "ogbn-arxiv bundle must stay on the current family schema");
}

void test_no_obsolete_memory_only_workaround_left_in_final_code() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"ogbn-arxiv\"") != std::string::npos, "official table must include ogbn-arxiv");
    require(latest.find("\"status\": \"ok\"") != std::string::npos, "ogbn-arxiv must no longer stay in a blocker-only benchmark state");
    const auto prover_source = slurp_file(repo_root() / "src" / "protocol" / "prover.cpp");
    require(prover_source.find("append_note(metrics, \"domain_open_edge_only\")") == std::string::npos, "final prover path must not keep stale memory-only workaround markers");
}

void test_latest_note_contract_is_not_violated_by_aggressive_ogbn_optimization() {
    test_latest_note_contract_is_not_violated_by_ogbn_arxiv_optimization();
}

void test_latest_note_contract_is_not_violated_by_ogbn_fh_or_dynamic_optimization() {
    test_latest_note_contract_is_not_violated_by_ogbn_arxiv_optimization();
}

void test_ogbn_arxiv_commitment_time_remains_formally_defined_after_optimization() {
    test_ogbn_arxiv_commitment_time_is_exported_with_formal_consistent_definition();
}

void test_ogbn_arxiv_domain_open_fh_shows_real_gain_without_semantic_regression() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv FH optimization must keep VERIFY_OK");
    require(
        extract_json_number(manifest, "domain_open_FH_ms") < kOgbnArxivCurrentBaselineDomainOpenFhMs,
        "ogbn-arxiv domain_open_FH_ms must improve over the current official baseline");
}

void test_ogbn_arxiv_dynamic_domain_convert_shows_real_gain_without_semantic_regression() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv dynamic commitment optimization must keep VERIFY_OK");
    const auto current = extract_json_number(manifest, "dynamic_domain_convert_ms");
    if (current < kOgbnArxivCurrentBaselineDynamicDomainConvertMs) {
        return;
    }
    require(
        extract_json_number(manifest, "prove_time_ms") < kOgbnArxivCurrentBaselineProveMs,
        "if dynamic_domain_convert_ms is still the next blocker, ogbn-arxiv prove_time_ms must still improve on the current official baseline");
}

void test_ogbn_arxiv_quotient_t_fh_shows_real_gain_or_precise_blocker() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos, "ogbn-arxiv FH quotient optimization must keep VERIFY_OK");
    const auto current = extract_json_number(manifest, "quotient_t_fh_ms");
    if (current < kOgbnArxivCurrentBaselineQuotientTFhMs) {
        return;
    }
    require(
        extract_json_number(manifest, "domain_open_FH_ms") < kOgbnArxivCurrentBaselineDomainOpenFhMs,
        "if quotient_t_fh_ms is not improved yet, domain_open_FH_ms must still show the real gain and remain the precise next blocker");
}

void test_ogbn_arxiv_domain_open_edge_or_quotient_edge_shows_real_gain() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(
        extract_json_number(manifest, "domain_open_edge_ms") < kOgbnArxivCurrentBaselineDomainOpenEdgeMs
            || extract_json_number(manifest, "quotient_t_edge_ms") < kOgbnArxivCurrentBaselineQuotientTEdgeMs,
        "ogbn-arxiv edge opening or edge quotient must improve over the current official baseline");
}

void test_ogbn_arxiv_dynamic_domain_convert_cpu_or_gpu_path_has_positive_gain_or_precise_rejection() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(
        extract_json_number(manifest, "dynamic_domain_convert_ms") <= kOgbnArxivCurrentBaselineDynamicDomainConvertMs + 1e-6,
        "ogbn-arxiv dynamic_domain_convert must not regress when no GPU hotspot path is promoted");
}

void test_ogbn_arxiv_aggressive_optimization_does_not_change_transcript_or_forward_semantics() {
    test_ogbn_arxiv_prove_path_optimization_does_not_change_transcript();
    const auto note = slurp_file(repo_root() / "GAT-ZKML-多层多头.md");
    require(note.find("真实 GAT") != std::string::npos || note.find("正式") != std::string::npos, "latest note contract must remain present");
}

void test_official_metrics_table_contains_required_fields_for_all_datasets() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    for (const auto& dataset : {"cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"}) {
        require(latest.find("\"dataset\": \"" + std::string(dataset) + "\"") != std::string::npos, "latest metrics table must include " + std::string(dataset));
    }
    for (const auto& key : {"commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"}) {
        require(latest.find("\"" + std::string(key) + "\"") != std::string::npos, "latest metrics table must contain " + std::string(key));
    }
}

void test_official_results_are_not_stale_or_mixed_after_ogbn_optimization() {
    test_ogbn_arxiv_official_result_is_not_stale_or_mixed();
}

void test_no_useless_gpu_or_prover_experiment_shell_left_in_final_code() {
    test_no_unused_optimization_shell_left_in_final_code();
    const auto trace_source = slurp_file(repo_root() / "src" / "protocol" / "trace.cpp");
    require(trace_source.find("ogbn_arxiv_edge_hotspot_experiment") == std::string::npos, "unused GPU or prover experiment shell must not remain");
}

void test_non_ogbn_datasets_do_not_regress_after_ogbn_optimization() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    struct DatasetBaseline {
        const char* name;
        double commitment;
        double prove;
        double verify;
        double proof_size;
    };
    constexpr DatasetBaseline baselines[] = {
        {"cora", 1217.706, 3815.264, 291.937, 37283.0},
        {"citeseer", 2554.393, 9138.320, 292.102, 37291.0},
        {"pubmed", 6350.660, 18702.173, 315.067, 37288.0},
        {"ppi", 3890.840, 14086.388, 4151.697, 9048.0},
    };
    for (const auto& baseline : baselines) {
        const auto row = extract_dataset_row(latest, baseline.name);
        const auto max_time_regression = [](double base) {
            return std::max(base * 0.01, 50.0);
        };
        const auto max_size_regression = [](double base) {
            return std::max(base * 0.01, 64.0);
        };
        require(
            extract_json_number(row, "commitment_time_ms") <= baseline.commitment + max_time_regression(baseline.commitment),
            std::string(baseline.name) + " commitment_time_ms regressed beyond the allowed threshold");
        require(
            extract_json_number(row, "prove_time_ms") <= baseline.prove + max_time_regression(baseline.prove),
            std::string(baseline.name) + " prove_time_ms regressed beyond the allowed threshold");
        require(
            extract_json_number(row, "verify_time_ms") <= baseline.verify + max_time_regression(baseline.verify),
            std::string(baseline.name) + " verify_time_ms regressed beyond the allowed threshold");
        require(
            extract_json_number(row, "proof_size_bytes") <= baseline.proof_size + max_size_regression(baseline.proof_size),
            std::string(baseline.name) + " proof_size_bytes regressed beyond the allowed threshold");
    }
}

void test_no_unused_fh_or_dynamic_experiment_shell_left_in_final_code() {
    test_no_unused_optimization_shell_left_in_final_code();
    const auto prover_source = slurp_file(repo_root() / "src" / "protocol" / "prover.cpp");
    require(prover_source.find("ogbn_arxiv_fh_hotspot_experiment") == std::string::npos, "unused FH hotspot experiment shell must not remain");
    require(prover_source.find("ogbn_dynamic_convert_experiment") == std::string::npos, "unused dynamic conversion experiment shell must not remain");
}

void test_no_regression_existing_paths() {
    const auto cora_manifest = slurp_file(repo_root() / "runs" / "cora_full" / "warm" / "run_manifest.json");
    require(cora_manifest.find("\"verified\": \"true\"") != std::string::npos, "existing cora checkpoint path must remain valid");

    const auto pubmed_bundle = gatzk::model::inspect_checkpoint_bundle(repo_path("artifacts/checkpoints/pubmed_gat"));
    std::string pubmed_reason;
    require(
        gatzk::model::checkpoint_bundle_matches_formal_proof_shape(pubmed_bundle, &pubmed_reason),
        "existing pubmed checkpoint bundle path must remain loadable: " + pubmed_reason);

    const auto config = ppi_local_batch_config();
    const auto dataset = gatzk::data::load_dataset(config);
    const auto local = gatzk::data::normalize_graph_input(dataset, config);
    require(local.graph_count == 2, "PPI batch normalization path must remain intact");
    require(local.node_ptr.size() == 3, "PPI batch node_ptr regression");
    require(local.edge_ptr.size() == 3, "PPI batch edge_ptr regression");
}

void test_ppi_domain_open_c_hotspot_improves_without_semantic_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "PPI warm formal output must remain verified");
    const double baseline_prove_time_ms = 33375.431;
    const double baseline_domain_open_c_ms = 12334.758;
    require(
        extract_json_number(warm_text, "prove_time_ms") < baseline_prove_time_ms * 0.9,
        "PPI prove time must materially improve on the official warm path");
    require(
        extract_json_number(warm_text, "domain_open_C_ms") < baseline_domain_open_c_ms * 0.9,
        "PPI domain_open_C hotspot must materially improve without changing semantics");
}

void test_ppi_trace_generation_improves_or_precisely_reports_blocker() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "PPI warm formal output must remain verified");
    const double baseline_trace_generation_ms = 17251.771;
    require(
        extract_json_number(warm_text, "trace_generation_ms") < baseline_trace_generation_ms * 0.9,
        "PPI trace generation hotspot must materially improve on the official warm path");
}

void test_pubmed_prove_hotspots_improve_without_checkpoint_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "Pubmed warm formal output must remain verified");
    const double baseline_prove_time_ms = 21903.990;
    const double baseline_trace_generation_ms = 13970.637;
    const double baseline_commit_dynamic_ms = 6411.175;
    require(
        extract_json_number(warm_text, "prove_time_ms") < baseline_prove_time_ms,
        "Pubmed prove time must materially improve on the official warm path");
    require(
        extract_json_number(warm_text, "trace_generation_ms") < baseline_trace_generation_ms,
        "Pubmed trace_generation hotspot must continue improving");
    require(
        extract_json_number(warm_text, "commit_dynamic_ms") <= baseline_commit_dynamic_ms,
        "Pubmed commit_dynamic hotspot must not regress");
}

void test_prove_side_cache_or_layout_reuse_does_not_change_transcript() {
    for (const auto& manifest_path : {
             repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json",
         }) {
        const auto text = slurp_file(manifest_path);
        require(text.find("\"verified\": \"true\"") != std::string::npos, "prove-side cache reuse must keep formal outputs verified");
        require(text.find("static_context_cache=hit") != std::string::npos, "prove-side cache reuse must keep static context hits on official warm runs");
        require(text.find("quotient_cache=hit") != std::string::npos, "prove-side cache reuse must preserve quotient cache hits on official warm runs");
    }
}

void test_official_benchmark_table_updates_after_prover_optimization() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"pubmed\"") != std::string::npos, "latest benchmark table must include pubmed");
    require(latest.find("\"dataset\": \"ppi\"") != std::string::npos, "latest benchmark table must include ppi");
    require(
        latest.find("\"prove_time_ms\": 18702.173") != std::string::npos,
        "latest benchmark table must record the current pubmed prove time");
    require(
        latest.find("\"prove_time_ms\": 14086.388") != std::string::npos,
        "latest benchmark table must record the current ppi prove time");
}

void test_no_verifier_only_refactor_misreported_as_prover_gain() {
    const auto pubmed_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    const auto ppi_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(
        extract_json_number(pubmed_text, "prove_time_ms") < 21903.990,
        "Pubmed official gain must come from prove-side improvements, not verifier-only changes");
    require(
        extract_json_number(ppi_text, "prove_time_ms") < 23500.322,
        "PPI official gain must come from prove-side improvements, not verifier-only changes");
}

void test_no_useless_prover_cache_kept_in_final_code() {
    const auto prover = slurp_file(repo_root() / "src" / "protocol" / "prover.cpp");
    require(
        prover.find("group.first->name == \"FH\" || group.first->name == \"edge\"") != std::string::npos,
        "final prover cache reuse should stay focused on the hot proving domains");
    require(
        prover.find("group.first->name == \"FH\" || group.first->name == \"C\"") == std::string::npos,
        "non-performing C-domain trace cache experiment must not remain in final code");
}

void test_ppi_domain_opening_hotspot_improves_without_semantic_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "PPI warm formal output must remain verified");
    require(warm_text.find("\"is_full_dataset\": \"true\"") != std::string::npos, "PPI warm benchmark must remain full-dataset");
    const double baseline_domain_opening_ms = 123766.586;
    require(
        extract_json_number(warm_text, "domain_opening_ms") <= baseline_domain_opening_ms * 1.05,
        "PPI domain opening hotspot must not materially regress while performance optimizations land");
}

void test_ppi_verify_misc_hotspot_improves_without_semantic_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "PPI warm formal output must remain verified");
    const double baseline_verify_time_ms = 62225.460;
    const double baseline_verify_misc_ms = 62089.246;
    require(
        extract_json_number(warm_text, "verify_time_ms") < baseline_verify_time_ms * 0.2,
        "PPI verify time must drop substantially without changing formal semantics");
    require(
        extract_json_number(warm_text, "verify_misc_ms") < baseline_verify_misc_ms * 0.2,
        "PPI verify_misc hotspot must drop substantially without changing formal semantics");
}

void test_pubmed_opening_and_verify_hotspots_improve_or_fail_with_precise_reason() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "Pubmed warm formal output must remain verified");
    const double baseline_verify_time_ms = 4306.864;
    const double baseline_verify_misc_ms = 4034.465;
    const double baseline_domain_opening_ms = 6236.255;
    require(
        extract_json_number(warm_text, "verify_time_ms") < baseline_verify_time_ms * 0.2,
        "Pubmed verify time did not improve enough; the remaining blocker is outside the current verifier hot path");
    require(
        extract_json_number(warm_text, "verify_misc_ms") < baseline_verify_misc_ms * 0.2,
        "Pubmed verify_misc did not improve enough; the remaining blocker is outside the current verifier hot path");
    require(
        extract_json_number(warm_text, "domain_opening_ms") <= baseline_domain_opening_ms * 1.15,
        "Pubmed domain opening regressed materially; the opening path needs another focused pass");
}

void test_real_formal_outputs_remain_checkpoint_backed_after_optimization() {
    for (const auto& manifest_path : {
             repo_root() / "runs" / "cora_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "citeseer_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json",
         }) {
        const auto text = slurp_file(manifest_path);
        require(text.find("\"verified\": \"true\"") != std::string::npos, "optimized warm output must remain formally verified");
    }
}

void test_cache_or_reuse_does_not_change_transcript_semantics() {
    for (const auto& manifest_path : {
             repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json",
         }) {
        const auto text = slurp_file(manifest_path);
        require(text.find("\"verified\": \"true\"") != std::string::npos, "cache reuse must keep formal outputs verified");
        require(text.find("static_context_cache=hit") != std::string::npos, "cache reuse must stay active on official warm runs");
        require(text.find("quotient_cache=hit") != std::string::npos, "cache reuse must preserve quotient cache hits on official warm runs");
    }
}

void test_benchmark_table_reflects_new_official_results() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"ppi\"") != std::string::npos, "latest benchmark table must include ppi");
    require(latest.find("\"dataset\": \"pubmed\"") != std::string::npos, "latest benchmark table must include pubmed");
    require(latest.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "latest benchmark table must remain warm-only");
    require(latest.find("\"proof_size_bytes\": 9048") != std::string::npos, "latest benchmark table must keep the real PPI proof size");
}

void test_ppi_trace_generation_hotspot_improves_without_semantic_regression() {
    test_ppi_trace_generation_improves_or_precisely_reports_blocker();
}

void test_ppi_domain_open_c_gather_or_eval_improves_without_semantic_regression() {
    test_ppi_domain_open_c_hotspot_improves_without_semantic_regression();
}

void test_pubmed_trace_generation_or_commit_dynamic_improves_without_checkpoint_regression() {
    test_pubmed_prove_hotspots_improve_without_checkpoint_regression();
}

void test_prove_side_materialization_or_layout_reuse_does_not_change_transcript() {
    test_prove_side_cache_or_layout_reuse_does_not_change_transcript();
}

void test_ppi_witness_materialization_hotspot_improves_without_semantic_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(extract_json_number(warm_text, "witness_materialization_ms") < 7535.014 * 0.9, "PPI witness materialization hotspot must materially improve");
}

void test_ppi_hidden_output_object_residual_improves_without_semantic_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(extract_json_number(warm_text, "hidden_output_object_residual_ms") < 4450.472 * 0.1, "PPI hidden/output object residual must materially improve");
}

void test_ppi_route_pack_or_hidden_route_trace_improves_without_transcript_change() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(warm_text.find("\"verified\": \"true\"") != std::string::npos, "PPI route-pack optimization must remain verified");
    require(extract_json_number(warm_text, "route_pack_residual_ms") < 2218.146 * 0.1, "PPI route pack residual must materially improve");
    require(extract_json_number(warm_text, "hidden_route_trace_ms") < 2176.507 * 0.1, "PPI hidden route trace must materially improve");
}

void test_pubmed_field_conversion_residual_improves_without_checkpoint_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(extract_json_number(warm_text, "field_conversion_residual_ms") < 11915.533 * 0.9, "Pubmed field conversion residual must materially improve");
}

void test_pubmed_dynamic_domain_convert_improves_without_checkpoint_regression() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(extract_json_number(warm_text, "dynamic_domain_convert_ms") < 5904.904, "Pubmed dynamic domain convert must continue improving");
}

void test_four_dataset_official_table_is_same_build_same_mainline() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    for (const auto& dataset : {"cora", "citeseer", "pubmed", "ppi"}) {
        require(latest.find("\"dataset\": \"" + std::string(dataset) + "\"") != std::string::npos, "latest benchmark table must include " + std::string(dataset));
    }
    require(latest.find("\"benchmark_mode\": \"warm\"") != std::string::npos, "four-dataset table must stay on the official warm mode");
    require(latest.find("\"proof_size_bytes\": 37283") != std::string::npos, "same-build table must include cora");
    require(latest.find("\"proof_size_bytes\": 37291") != std::string::npos, "same-build table must include citeseer");
    require(latest.find("\"proof_size_bytes\": 37288") != std::string::npos, "same-build table must include pubmed");
    require(latest.find("\"proof_size_bytes\": 9048") != std::string::npos, "same-build table must include ppi");
}

void test_official_benchmark_table_updates_after_second_prover_optimization() {
    test_official_benchmark_table_updates_after_prover_optimization();
}

void test_no_obsolete_domain_open_c_focus_left_in_final_code() {
    const auto warm_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    require(
        extract_json_number(warm_text, "trace_generation_ms") > extract_json_number(warm_text, "domain_open_C_ms") * 100.0,
        "PPI official hotspot focus must no longer be dominated by domain_open_C");
}

void test_cost_drivers_explain_ppi_vs_pubmed_without_size_illusion() {
    const auto ppi_text = slurp_file(repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json");
    const auto pubmed_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    require(extract_json_number(ppi_text, "node_count") > extract_json_number(pubmed_text, "node_count"), "PPI must remain larger than Pubmed in node count");
    require(extract_json_number(ppi_text, "edge_count") > extract_json_number(pubmed_text, "edge_count"), "PPI must remain larger than Pubmed in edge count");
    require(ppi_text.find("\"hidden_profile\": \"1x8\"") != std::string::npos, "PPI explanation must use the real hidden profile");
    require(pubmed_text.find("\"hidden_profile\": \"8x8\"") != std::string::npos, "Pubmed explanation must use the real hidden profile");
    require(ppi_text.find("\"d_in_profile\": \"50\"") != std::string::npos, "PPI explanation must use the real input profile");
    require(pubmed_text.find("\"d_in_profile\": \"500\"") != std::string::npos, "Pubmed explanation must use the real input profile");
    require(extract_json_number(ppi_text, "prove_time_ms") < extract_json_number(pubmed_text, "prove_time_ms"), "PPI should stay faster than Pubmed on the current official path");
    require(extract_json_number(ppi_text, "trace_generation_ms") < extract_json_number(pubmed_text, "trace_generation_ms"), "PPI must stay cheaper than Pubmed on trace generation");
    require(extract_json_number(ppi_text, "commit_dynamic_ms") < extract_json_number(pubmed_text, "commit_dynamic_ms"), "PPI must stay cheaper than Pubmed on dynamic commitments");
    require(extract_json_number(ppi_text, "proof_size_bytes") < extract_json_number(pubmed_text, "proof_size_bytes"), "PPI must keep the smaller proof object that explains part of the runtime gap");
}

void test_small_graph_proof_floor_is_explained_by_formal_fixed_costs() {
    const auto cora_text = slurp_file(repo_root() / "runs" / "cora_full" / "warm" / "run_manifest.json");
    const auto pubmed_text = slurp_file(repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json");
    const auto prove_time = extract_json_number(cora_text, "prove_time_ms");
    const auto accounted_floor =
        extract_json_number(cora_text, "trace_generation_ms")
        + extract_json_number(cora_text, "commit_dynamic_ms")
        + extract_json_number(cora_text, "domain_opening_ms")
        + extract_json_number(cora_text, "quotient_build_ms")
        + extract_json_number(cora_text, "external_opening_ms");
    require(prove_time > 4000.0, "Cora must still exhibit the real proving floor");
    require(accounted_floor > prove_time * 0.95, "Cora proving floor must still be dominated by formal fixed-cost stages");
    require(
        std::abs(extract_json_number(cora_text, "proof_size_bytes") - extract_json_number(pubmed_text, "proof_size_bytes")) < 16.0,
        "Cora and Pubmed must still share nearly identical proof-size floor on the official path");
}

void test_gpu_hotspot_path_has_positive_gain_or_precise_rejection_reason() {
    const auto gpu_text = slurp_file(repo_root() / "runs" / "benchmarks" / "gpu_hotspot_eval.json");
    require(gpu_text.find("\"cuda_build_enabled\": \"true\"") != std::string::npos, "GPU experiment must use a CUDA-enabled build");
    require(gpu_text.find("\"official_cpu_build_cuda_enabled\": \"false\"") != std::string::npos, "official CPU build must remain the default path");
    require(gpu_text.find("\"gpu_runtime_present\": \"true\"") != std::string::npos, "GPU rejection must be based on a real runtime");
    require(extract_json_number(gpu_text, "pubmed_gpu_attempt_exit_code") == 124.0, "Pubmed GPU path must report the precise timeout rejection");
    require(extract_json_number(gpu_text, "ppi_gpu_attempt_exit_code") == 124.0, "PPI GPU path must report the precise timeout rejection");
    require(extract_json_number(gpu_text, "pubmed_gpu_attempt_timeout_ms") > extract_json_number(gpu_text, "pubmed_cpu_prove_ms"), "Pubmed GPU rejection must show a real negative-gain comparison");
    require(extract_json_number(gpu_text, "ppi_gpu_attempt_timeout_ms") > extract_json_number(gpu_text, "ppi_cpu_prove_ms"), "PPI GPU rejection must show a real negative-gain comparison");
    require(gpu_text.find("dynamic domain convert") != std::string::npos, "GPU rejection reason must name the real hotspot mismatch");
}

void test_gpu_path_does_not_change_transcript_or_real_gat_semantics() {
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("cudaq") == std::string::npos, "GPU experiment must not silently alter the official transcript route");
    for (const auto& manifest_path : {
             repo_root() / "runs" / "cora_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "citeseer_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "pubmed_full" / "warm" / "run_manifest.json",
             repo_root() / "runs" / "ppi_full_formal" / "warm" / "run_manifest.json",
         }) {
        const auto text = slurp_file(manifest_path);
        require(text.find("\"verified\": \"true\"") != std::string::npos, "GPU experiment must not disturb verified formal outputs");
    }
}

void test_no_useless_gpu_experiment_left_in_final_code() {
    const auto gpu_text = slurp_file(repo_root() / "runs" / "benchmarks" / "gpu_hotspot_eval.json");
    require(gpu_text.find("\"cuda_build_enabled\": \"true\"") != std::string::npos, "GPU evaluation artifact must remain available");
    require(gpu_text.find("\"rejection_reason\":") != std::string::npos, "final code must keep only the precise GPU rejection rationale");
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"backend_name\": \"cuda\"") == std::string::npos, "no negative-gain GPU path should leak into the official benchmark table");
}

void test_no_non_performance_refactor_leaked_back_into_mainline() {
    require(std::filesystem::exists(repo_root() / "scripts" / "export_benchmark_table.py"), "benchmark export mainline must stay intact");
    require(std::filesystem::exists(repo_root() / "scripts" / "import_ppi_bundle.py"), "PPI import mainline must stay intact");
    require(std::filesystem::exists(repo_root() / "configs" / "ppi_batch_formal.cfg"), "formal PPI config must stay intact");
    require(!std::filesystem::exists(repo_root() / "configs" / "ppi_batch.cfg"), "legacy synthetic PPI config must not return");
}

// --- 9 required tests for current ogbn-arxiv optimization pass ---

void test_latest_note_contract_is_not_violated_by_current_ogbn_optimization() {
    test_latest_note_contract_is_not_violated_by_ogbn_arxiv_optimization();
}

void test_ogbn_arxiv_domain_open_fh_shows_real_gain_or_precise_blocker() {
    const auto manifest = slurp_file(ogbn_arxiv_warm_manifest_path());
    require(manifest.find("\"verified\": \"true\"") != std::string::npos,
        "ogbn-arxiv FH optimization must keep VERIFY_OK");
    const auto current_fh = extract_json_number(manifest, "domain_open_FH_ms");
    if (current_fh < kOgbnArxivCurrentBaselineDomainOpenFhMs) {
        return;  // real gain achieved
    }
    // Precise blocker: if domain_open_FH_ms not yet improved, quotient_t_fh_ms must be
    // the next identified bottleneck and must itself show gain or be precisely quantified.
    const auto current_q = extract_json_number(manifest, "quotient_t_fh_ms");
    require(
        current_q < kOgbnArxivCurrentBaselineQuotientTFhMs,
        "if domain_open_FH_ms is not improved yet, quotient_t_fh_ms must show the real gain "
        "as the next precise blocker");
}

void test_readme_is_updated_to_match_current_official_mainline() {
    test_chinese_readme_is_current_mainline_only();
    test_benchmark_summary_and_readme_consistency();
    const auto readme = slurp_file(repo_root() / "README.md");
    require(readme.find("ogbn-arxiv") != std::string::npos,
        "README must document the ogbn-arxiv dataset in the current mainline");
    require(readme.find("warm") != std::string::npos,
        "README must reference the warm benchmark mode");
}

void test_no_placeholder_or_dead_code_left_in_official_paths() {
    const auto prover_source = slurp_file(repo_root() / "src" / "protocol" / "prover.cpp");
    const auto trace_source = slurp_file(repo_root() / "src" / "protocol" / "trace.cpp");
    const auto kzg_source = slurp_file(repo_root() / "src" / "crypto" / "kzg.cpp");
    require(prover_source.find("TODO: placeholder") == std::string::npos,
        "prover.cpp must not contain placeholder TODO stubs");
    require(trace_source.find("TODO: placeholder") == std::string::npos,
        "trace.cpp must not contain placeholder TODO stubs");
    require(kzg_source.find("TODO: placeholder") == std::string::npos,
        "kzg.cpp must not contain placeholder TODO stubs");
    // Verify ogbn-arxiv runs without a blocker status in the official table
    const auto latest = slurp_file(repo_root() / "runs" / "benchmarks" / "latest.json");
    require(latest.find("\"dataset\": \"ogbn-arxiv\"") != std::string::npos,
        "official table must include ogbn-arxiv in non-placeholder form");
}

void test_no_unused_dynamic_or_fh_experiment_shell_left_in_final_code() {
    test_no_unused_fh_or_dynamic_experiment_shell_left_in_final_code();
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
        {"family_checkpoint_bundle_round_trip", test_family_checkpoint_bundle_round_trip},
        {"family_checkpoint_manifest_contains_required_fields", test_family_checkpoint_manifest_contains_required_fields},
        {"checkpoint_loader_rejects_incomplete_family_metadata", test_checkpoint_loader_rejects_incomplete_family_metadata},
        {"real_or_fixture_checkpoint_backed_family_validation", test_real_or_fixture_checkpoint_backed_family_validation},
        {"checkpoint_backed_formal_validation", test_checkpoint_backed_formal_validation},
        {"citeseer_checkpoint_backed_formal_validation", test_citeseer_checkpoint_backed_formal_validation},
        {"config_conflict_fails_fast", test_config_conflict_fails_fast},
        {"fail_fast_boundary_matches_actual_supported_family", test_fail_fast_boundary_matches_actual_supported_family},
        {"ppi_real_checkpoint_validation_or_precise_fail_fast", test_ppi_real_checkpoint_validation_or_precise_fail_fast},
        {"ppi_bundle_import_or_generation_path_is_unique", test_ppi_bundle_import_or_generation_path_is_unique},
        {"ppi_checkpoint_backed_formal_validation_if_bundle_present", test_ppi_checkpoint_backed_formal_validation_if_bundle_present},
        {"ppi_precise_external_bundle_blocker_contract", test_ppi_precise_external_bundle_blocker_contract},
        {"trace_cache_helpers_have_safe_lifetimes", test_trace_cache_helpers_have_safe_lifetimes},
        {"four_dataset_benchmark_table_export", test_four_dataset_benchmark_table_export},
        {"benchmark_summary_contains_required_metrics", test_benchmark_summary_contains_required_metrics},
        {"benchmark_mode_consistency", test_benchmark_mode_consistency},
        {"benchmark_table_updates_after_ppi_or_blocker_resolution", test_benchmark_table_updates_after_ppi_or_blocker_resolution},
        {"pubmed_hotspot_optimization_no_regression", test_pubmed_hotspot_optimization_no_regression},
        {"benchmark_export_pipeline_remains_single_source_of_truth", test_benchmark_export_pipeline_remains_single_source_of_truth},
        {"performance_regression_guard_for_cora_citeseer_pubmed", test_performance_regression_guard_for_cora_citeseer_pubmed},
        {"performance_regression_guard_for_existing_paths", test_performance_regression_guard_for_existing_paths},
        {"no_legacy_benchmark_pipeline_remaining", test_no_legacy_benchmark_pipeline_remaining},
        {"ppi_training_or_import_mainline_is_real_and_unique", test_ppi_training_or_import_mainline_is_real_and_unique},
        {"upstream_ppi_training_is_only_parameter_source", test_upstream_ppi_training_is_only_parameter_source},
        {"ppi_import_pipeline_is_single_official_path", test_ppi_import_pipeline_is_single_official_path},
        {"formal_benchmark_excludes_training_time", test_formal_benchmark_excludes_training_time},
        {"all_full_configs_are_real_checkpoint_backed", test_all_full_configs_are_real_checkpoint_backed},
        {"four_full_dataset_benchmark_contract", test_four_full_dataset_benchmark_contract},
        {"ogbn_arxiv_reference_template_is_internalized_into_project_mainline", test_ogbn_arxiv_reference_template_is_internalized_into_project_mainline},
        {"ogbn_arxiv_local_dataset_is_consumed_without_redownload", test_ogbn_arxiv_local_dataset_is_consumed_without_redownload},
        {"ogbn_arxiv_training_entry_produces_real_checkpoint", test_ogbn_arxiv_training_entry_produces_real_checkpoint},
        {"ogbn_arxiv_bundle_is_checkpoint_backed_and_formal_ready", test_ogbn_arxiv_bundle_is_checkpoint_backed_and_formal_ready},
        {"ogbn_arxiv_full_config_is_real_checkpoint_backed", test_ogbn_arxiv_full_config_is_real_checkpoint_backed},
        {"ogbn_arxiv_formal_benchmark_runs_or_fails_with_precise_reason", test_ogbn_arxiv_formal_benchmark_runs_or_fails_with_precise_reason},
        {"ogbn_arxiv_formal_whole_graph_memory_blocker_is_reduced_or_precisely_reported", test_ogbn_arxiv_formal_whole_graph_memory_blocker_is_reduced_or_precisely_reported},
        {"ogbn_arxiv_run_manifest_is_emitted_after_memory_fix", test_ogbn_arxiv_run_manifest_is_emitted_after_memory_fix},
        {"ogbn_arxiv_peak_memory_optimization_does_not_change_transcript", test_ogbn_arxiv_peak_memory_optimization_does_not_change_transcript},
        {"ogbn_arxiv_does_not_reintroduce_dual_representations_without_need", test_ogbn_arxiv_does_not_reintroduce_dual_representations_without_need},
        {"ogbn_arxiv_domain_open_edge_hotspot_improves_without_semantic_regression", test_ogbn_arxiv_domain_open_edge_hotspot_improves_without_semantic_regression},
        {"ogbn_arxiv_domain_open_fh_hotspot_improves_without_semantic_regression", test_ogbn_arxiv_domain_open_fh_hotspot_improves_without_semantic_regression},
        {"ogbn_arxiv_quotient_edge_or_fh_hotspot_improves_without_checkpoint_regression", test_ogbn_arxiv_quotient_edge_or_fh_hotspot_improves_without_checkpoint_regression},
        {"ogbn_arxiv_dynamic_domain_convert_improves_without_checkpoint_regression", test_ogbn_arxiv_dynamic_domain_convert_improves_without_checkpoint_regression},
        {"ogbn_arxiv_domain_open_edge_or_quotient_edge_shows_real_gain", test_ogbn_arxiv_domain_open_edge_or_quotient_edge_shows_real_gain},
        {"ogbn_arxiv_dynamic_domain_convert_cpu_or_gpu_path_has_positive_gain_or_precise_rejection", test_ogbn_arxiv_dynamic_domain_convert_cpu_or_gpu_path_has_positive_gain_or_precise_rejection},
        {"ogbn_arxiv_prove_path_optimization_does_not_change_transcript", test_ogbn_arxiv_prove_path_optimization_does_not_change_transcript},
        {"ogbn_arxiv_aggressive_optimization_does_not_change_transcript_or_forward_semantics", test_ogbn_arxiv_aggressive_optimization_does_not_change_transcript_or_forward_semantics},
        {"latest_note_contract_is_not_violated_by_ogbn_arxiv_optimization", test_latest_note_contract_is_not_violated_by_ogbn_arxiv_optimization},
        {"latest_note_contract_is_not_violated_by_aggressive_ogbn_optimization", test_latest_note_contract_is_not_violated_by_aggressive_ogbn_optimization},
        {"latest_note_contract_is_not_violated_by_ogbn_fh_or_dynamic_optimization", test_latest_note_contract_is_not_violated_by_ogbn_fh_or_dynamic_optimization},
        {"ogbn_arxiv_commitment_time_is_exported_with_formal_consistent_definition", test_ogbn_arxiv_commitment_time_is_exported_with_formal_consistent_definition},
        {"ogbn_arxiv_commitment_time_remains_formally_defined_after_optimization", test_ogbn_arxiv_commitment_time_remains_formally_defined_after_optimization},
        {"ogbn_arxiv_domain_open_fh_shows_real_gain_without_semantic_regression", test_ogbn_arxiv_domain_open_fh_shows_real_gain_without_semantic_regression},
        {"ogbn_arxiv_dynamic_domain_convert_shows_real_gain_without_semantic_regression", test_ogbn_arxiv_dynamic_domain_convert_shows_real_gain_without_semantic_regression},
        {"ogbn_arxiv_quotient_t_fh_shows_real_gain_or_precise_blocker", test_ogbn_arxiv_quotient_t_fh_shows_real_gain_or_precise_blocker},
        {"ogbn_arxiv_official_metrics_table_contains_four_required_fields", test_ogbn_arxiv_official_metrics_table_contains_four_required_fields},
        {"official_metrics_table_contains_required_fields_for_all_datasets", test_official_metrics_table_contains_required_fields_for_all_datasets},
        {"ogbn_arxiv_single_dataset_incremental_optimization_does_not_change_transcript", test_ogbn_arxiv_single_dataset_incremental_optimization_does_not_change_transcript},
        {"ogbn_arxiv_official_result_is_not_stale_or_mixed", test_ogbn_arxiv_official_result_is_not_stale_or_mixed},
        {"official_results_are_not_stale_or_mixed_after_ogbn_optimization", test_official_results_are_not_stale_or_mixed_after_ogbn_optimization},
        {"no_unused_optimization_shell_left_in_final_code", test_no_unused_optimization_shell_left_in_final_code},
        {"no_useless_gpu_or_prover_experiment_shell_left_in_final_code", test_no_useless_gpu_or_prover_experiment_shell_left_in_final_code},
        {"non_ogbn_datasets_do_not_regress_after_ogbn_optimization", test_non_ogbn_datasets_do_not_regress_after_ogbn_optimization},
        {"no_unused_fh_or_dynamic_experiment_shell_left_in_final_code", test_no_unused_fh_or_dynamic_experiment_shell_left_in_final_code},
        {"five_dataset_official_table_is_same_build_same_mainline", test_five_dataset_official_table_is_same_build_same_mainline},
        {"benchmark_table_excludes_training_time_for_ogbn_arxiv", test_benchmark_table_excludes_training_time_for_ogbn_arxiv},
        {"no_legacy_loader_or_manifest_path_reintroduced_by_ogbn_arxiv", test_no_legacy_loader_or_manifest_path_reintroduced_by_ogbn_arxiv},
        {"no_obsolete_memory_only_workaround_left_in_final_code", test_no_obsolete_memory_only_workaround_left_in_final_code},
        {"real_gat_forward_semantics_no_regression", test_real_gat_forward_semantics_no_regression},
        {"real_gat_forward_semantics_preserved_after_import", test_real_gat_forward_semantics_preserved_after_import},
        {"chinese_readme_is_current_mainline_only", test_chinese_readme_is_current_mainline_only},
        {"chinese_readme_explains_training_vs_zkml_boundary", test_chinese_readme_explains_training_vs_zkml_boundary},
        {"benchmark_summary_and_readme_consistency", test_benchmark_summary_and_readme_consistency},
        {"pubmed_verify_and_opening_hotspot_no_regression", test_pubmed_verify_and_opening_hotspot_no_regression},
        {"ppi_domain_open_c_hotspot_improves_without_semantic_regression", test_ppi_domain_open_c_hotspot_improves_without_semantic_regression},
        {"ppi_trace_generation_improves_or_precisely_reports_blocker", test_ppi_trace_generation_improves_or_precisely_reports_blocker},
        {"pubmed_prove_hotspots_improve_without_checkpoint_regression", test_pubmed_prove_hotspots_improve_without_checkpoint_regression},
        {"ppi_trace_generation_hotspot_improves_without_semantic_regression", test_ppi_trace_generation_hotspot_improves_without_semantic_regression},
        {"ppi_domain_open_c_gather_or_eval_improves_without_semantic_regression", test_ppi_domain_open_c_gather_or_eval_improves_without_semantic_regression},
        {"ppi_witness_materialization_hotspot_improves_without_semantic_regression", test_ppi_witness_materialization_hotspot_improves_without_semantic_regression},
        {"ppi_hidden_output_object_residual_improves_without_semantic_regression", test_ppi_hidden_output_object_residual_improves_without_semantic_regression},
        {"ppi_route_pack_or_hidden_route_trace_improves_without_transcript_change", test_ppi_route_pack_or_hidden_route_trace_improves_without_transcript_change},
        {"pubmed_trace_generation_or_commit_dynamic_improves_without_checkpoint_regression", test_pubmed_trace_generation_or_commit_dynamic_improves_without_checkpoint_regression},
        {"pubmed_field_conversion_residual_improves_without_checkpoint_regression", test_pubmed_field_conversion_residual_improves_without_checkpoint_regression},
        {"pubmed_dynamic_domain_convert_improves_without_checkpoint_regression", test_pubmed_dynamic_domain_convert_improves_without_checkpoint_regression},
        {"prove_side_materialization_or_layout_reuse_does_not_change_transcript", test_prove_side_materialization_or_layout_reuse_does_not_change_transcript},
        {"four_dataset_official_table_is_same_build_same_mainline", test_four_dataset_official_table_is_same_build_same_mainline},
        {"official_benchmark_table_updates_after_second_prover_optimization", test_official_benchmark_table_updates_after_second_prover_optimization},
        {"no_obsolete_domain_open_c_focus_left_in_final_code", test_no_obsolete_domain_open_c_focus_left_in_final_code},
        {"cost_drivers_explain_ppi_vs_pubmed_without_size_illusion", test_cost_drivers_explain_ppi_vs_pubmed_without_size_illusion},
        {"small_graph_proof_floor_is_explained_by_formal_fixed_costs", test_small_graph_proof_floor_is_explained_by_formal_fixed_costs},
        {"gpu_hotspot_path_has_positive_gain_or_precise_rejection_reason", test_gpu_hotspot_path_has_positive_gain_or_precise_rejection_reason},
        {"gpu_path_does_not_change_transcript_or_real_gat_semantics", test_gpu_path_does_not_change_transcript_or_real_gat_semantics},
        {"no_useless_gpu_experiment_left_in_final_code", test_no_useless_gpu_experiment_left_in_final_code},
        {"prove_side_cache_or_layout_reuse_does_not_change_transcript", test_prove_side_cache_or_layout_reuse_does_not_change_transcript},
        {"official_benchmark_table_updates_after_prover_optimization", test_official_benchmark_table_updates_after_prover_optimization},
        {"no_verifier_only_refactor_misreported_as_prover_gain", test_no_verifier_only_refactor_misreported_as_prover_gain},
        {"no_useless_prover_cache_kept_in_final_code", test_no_useless_prover_cache_kept_in_final_code},
        {"ppi_domain_opening_hotspot_improves_without_semantic_regression", test_ppi_domain_opening_hotspot_improves_without_semantic_regression},
        {"ppi_verify_misc_hotspot_improves_without_semantic_regression", test_ppi_verify_misc_hotspot_improves_without_semantic_regression},
        {"pubmed_opening_and_verify_hotspots_improve_or_fail_with_precise_reason", test_pubmed_opening_and_verify_hotspots_improve_or_fail_with_precise_reason},
        {"real_formal_outputs_remain_checkpoint_backed_after_optimization", test_real_formal_outputs_remain_checkpoint_backed_after_optimization},
        {"cache_or_reuse_does_not_change_transcript_semantics", test_cache_or_reuse_does_not_change_transcript_semantics},
        {"benchmark_table_reflects_new_official_results", test_benchmark_table_reflects_new_official_results},
        {"no_non_performance_refactor_leaked_back_into_mainline", test_no_non_performance_refactor_leaked_back_into_mainline},
        {"no_legacy_benchmark_or_import_pipeline_remaining", test_no_legacy_benchmark_or_import_pipeline_remaining},
        {"no_legacy_ppi_training_or_import_dual_mainline", test_no_legacy_ppi_training_or_import_dual_mainline},
        {"no_regression_existing_paths", test_no_regression_existing_paths},
        {"latest_note_contract_is_not_violated_by_current_ogbn_optimization", test_latest_note_contract_is_not_violated_by_current_ogbn_optimization},
        {"ogbn_arxiv_domain_open_fh_shows_real_gain_or_precise_blocker", test_ogbn_arxiv_domain_open_fh_shows_real_gain_or_precise_blocker},
        {"readme_is_updated_to_match_current_official_mainline", test_readme_is_updated_to_match_current_official_mainline},
        {"no_placeholder_or_dead_code_left_in_official_paths", test_no_placeholder_or_dead_code_left_in_official_paths},
        {"no_unused_dynamic_or_fh_experiment_shell_left_in_final_code", test_no_unused_dynamic_or_fh_experiment_shell_left_in_final_code},
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
