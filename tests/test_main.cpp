#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <functional>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

using gatzk::algebra::FieldElement;
using gatzk::protocol::ProtocolContext;
using gatzk::protocol::Proof;
using gatzk::protocol::TraceArtifacts;

struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

struct ErrorStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
};

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::string repo_path(const std::string& relative) {
    return (repo_root() / relative).string();
}

std::filesystem::path reference_output_dir() {
    return repo_root() / "runs/cora_full/reference";
}

const std::filesystem::path& ensure_reference_outputs() {
    static const auto output_dir = []() {
        const auto out = reference_output_dir();
        if (std::filesystem::exists(out / "summary.json")) {
            return out;
        }

        std::filesystem::create_directories(out);
        const auto python = std::filesystem::exists(repo_root() / ".venv/bin/python")
            ? (repo_root() / ".venv/bin/python").string()
            : std::string("python3");
        const auto command = python
            + " " + repo_path("scripts/gat_reference.py")
            + " --checkpoint-dir " + repo_path("artifacts/checkpoints/cora_gat")
            + " --data-root " + repo_path("data")
            + " --dataset cora"
            + " --output-dir " + out.string();
        if (std::system(command.c_str()) != 0) {
            throw std::runtime_error("failed to regenerate full-cora reference artifacts");
        }
        return out;
    }();
    return output_dir;
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

std::string trim_copy(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), value.end());
    return value;
}

NpyArray load_npy_double(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "failed to open npy file: " + path.string());

    char magic[6] = {};
    input.read(magic, sizeof(magic));
    require(std::string(magic, sizeof(magic)) == "\x93NUMPY", "invalid npy magic: " + path.string());

    std::uint8_t major = 0;
    std::uint8_t minor = 0;
    input.read(reinterpret_cast<char*>(&major), 1);
    input.read(reinterpret_cast<char*>(&minor), 1);
    (void)minor;

    std::uint32_t header_len = 0;
    if (major == 1) {
        std::uint16_t short_len = 0;
        input.read(reinterpret_cast<char*>(&short_len), sizeof(short_len));
        header_len = short_len;
    } else if (major == 2 || major == 3) {
        input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        throw std::runtime_error("unsupported npy version in " + path.string());
    }

    std::string header(header_len, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));

    const auto descr_pos = header.find("'descr'");
    require(descr_pos != std::string::npos, "missing descr in npy header: " + path.string());
    const auto descr_quote = header.find('\'', header.find(':', descr_pos));
    const auto descr_end = header.find('\'', descr_quote + 1);
    const auto descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
    const bool is_float64 = descr == "<f8" || descr == "|f8";
    const bool is_float32 = descr == "<f4" || descr == "|f4";
    require(is_float64 || is_float32, "expected float32/float64 npy payload in " + path.string());

    const auto fortran_pos = header.find("fortran_order");
    require(fortran_pos != std::string::npos, "missing fortran_order in npy header: " + path.string());
    require(header.find("False", fortran_pos) != std::string::npos, "fortran-order arrays are not supported: " + path.string());

    const auto shape_pos = header.find('(', header.find("shape"));
    const auto shape_end = header.find(')', shape_pos);
    require(shape_pos != std::string::npos && shape_end != std::string::npos, "missing shape in npy header: " + path.string());
    const auto shape_text = header.substr(shape_pos + 1, shape_end - shape_pos - 1);

    NpyArray out;
    std::stringstream shape_stream(shape_text);
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        out.shape.push_back(static_cast<std::size_t>(std::stoull(token)));
    }
    require(!out.shape.empty(), "empty shape in npy header: " + path.string());

    std::size_t count = 1;
    for (const auto dim : out.shape) {
        count *= dim;
    }
    out.values.resize(count);
    if (is_float64) {
        input.read(reinterpret_cast<char*>(out.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
        require(
            input.gcount() == static_cast<std::streamsize>(count * sizeof(double)),
            "unexpected float64 payload size in npy file: " + path.string());
    } else {
        std::vector<float> float_values(count, 0.0f);
        input.read(reinterpret_cast<char*>(float_values.data()), static_cast<std::streamsize>(count * sizeof(float)));
        require(
            input.gcount() == static_cast<std::streamsize>(count * sizeof(float)),
            "unexpected float32 payload size in npy file: " + path.string());
        for (std::size_t i = 0; i < count; ++i) {
            out.values[i] = static_cast<double>(float_values[i]);
        }
    }
    return out;
}

std::vector<double> flatten_matrix(const gatzk::model::FloatMatrix& matrix) {
    std::vector<double> out;
    for (const auto& row : matrix) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

ErrorStats compare_flat_values(
    const std::vector<double>& actual,
    const NpyArray& expected,
    const std::vector<std::size_t>& expected_shape,
    const std::string& label,
    double max_threshold,
    double mean_threshold) {
    require(expected.shape == expected_shape, label + " shape mismatch");
    require(actual.size() == expected.values.size(), label + " flattened size mismatch");

    ErrorStats stats;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto actual_serialized = static_cast<double>(static_cast<float>(actual[i]));
        const auto abs_error = std::abs(actual_serialized - expected.values[i]);
        stats.max_abs = std::max(stats.max_abs, abs_error);
        stats.mean_abs += abs_error;
    }
    if (!actual.empty()) {
        stats.mean_abs /= static_cast<double>(actual.size());
    }

    std::ostringstream message;
    message << label << " parity mismatch: max_abs=" << stats.max_abs << " mean_abs=" << stats.mean_abs;
    require(stats.max_abs <= max_threshold, message.str());
    require(stats.mean_abs <= mean_threshold, message.str());
    return stats;
}

ErrorStats compare_matrix_npy(
    const gatzk::model::FloatMatrix& actual,
    const std::filesystem::path& path,
    const std::string& label,
    double max_threshold = 5e-5,
    double mean_threshold = 1e-6) {
    const auto expected = load_npy_double(path);
    require(!actual.empty(), label + " actual matrix is empty");
    return compare_flat_values(
        flatten_matrix(actual),
        expected,
        {actual.size(), actual.front().size()},
        label,
        max_threshold,
        mean_threshold);
}

ErrorStats compare_vector_npy(
    const std::vector<double>& actual,
    const std::filesystem::path& path,
    const std::string& label,
    double max_threshold = 5e-5,
    double mean_threshold = 1e-6) {
    const auto expected = load_npy_double(path);
    return compare_flat_values(actual, expected, {actual.size()}, label, max_threshold, mean_threshold);
}

gatzk::model::FloatMatrix dense_sum_scores(
    const std::vector<double>& e_src,
    const std::vector<double>& e_dst) {
    gatzk::model::FloatMatrix out(e_dst.size(), std::vector<double>(e_src.size(), 0.0));
    for (std::size_t dst = 0; dst < e_dst.size(); ++dst) {
        for (std::size_t src = 0; src < e_src.size(); ++src) {
            out[dst][src] = e_dst[dst] + e_src[src];
        }
    }
    return out;
}

gatzk::model::FloatMatrix dense_leaky_relu(const gatzk::model::FloatMatrix& scores) {
    auto out = scores;
    for (auto& row : out) {
        for (auto& value : row) {
            value = value >= 0.0 ? value : 0.2 * value;
        }
    }
    return out;
}

gatzk::model::FloatMatrix dense_masked_add(
    const gatzk::model::FloatMatrix& left,
    const gatzk::model::FloatMatrix& right) {
    require(left.size() == right.size(), "dense matrix row mismatch");
    require(left.empty() || left.front().size() == right.front().size(), "dense matrix col mismatch");
    auto out = left;
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t col = 0; col < left[row].size(); ++col) {
            out[row][col] = left[row][col] + right[row][col];
        }
    }
    return out;
}

gatzk::model::FloatMatrix dense_delta(
    const gatzk::model::FloatMatrix& masked_logits,
    const std::vector<double>& max_per_row) {
    auto out = masked_logits;
    for (std::size_t row = 0; row < masked_logits.size(); ++row) {
        for (std::size_t col = 0; col < masked_logits[row].size(); ++col) {
            out[row][col] = max_per_row[row] - masked_logits[row][col];
        }
    }
    return out;
}

gatzk::model::FloatMatrix dense_u(
    const gatzk::model::FloatMatrix& masked_logits,
    const std::vector<double>& max_per_row) {
    auto out = masked_logits;
    for (std::size_t row = 0; row < masked_logits.size(); ++row) {
        for (std::size_t col = 0; col < masked_logits[row].size(); ++col) {
            out[row][col] = std::exp(masked_logits[row][col] - max_per_row[row]);
        }
    }
    return out;
}

gatzk::model::FloatMatrix dense_alpha(
    const gatzk::model::FloatMatrix& u,
    const std::vector<double>& inv) {
    auto out = u;
    for (std::size_t row = 0; row < u.size(); ++row) {
        for (std::size_t col = 0; col < u[row].size(); ++col) {
            out[row][col] = u[row][col] * inv[row];
        }
    }
    return out;
}

gatzk::util::AppConfig small_debug_config() {
    gatzk::util::AppConfig config;
    config.project_root = repo_root().string();
    config.dataset = "cora";
    config.data_root = "data";
    config.cache_root = "data/cache";
    config.export_dir = "runs/test_small_real";
    config.hidden_dim = 4;
    config.num_classes = 7;
    config.range_bits = 12;
    config.seed = 11;
    config.local_nodes = 32;
    config.center_node = 0;
    config.allow_synthetic_model = true;
    config.dump_trace = false;
    config.auto_prepare_dataset = false;
    config.prove_enabled = true;
    return config;
}

gatzk::util::AppConfig full_graph_debug_config() {
    auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    config.allow_synthetic_model = true;
    return config;
}

const ProtocolContext& full_graph_debug_context() {
    static const auto context = []() {
        const auto config = full_graph_debug_config();
        return gatzk::protocol::build_context(config);
    }();
    return context;
}

struct ProofFixture {
    ProtocolContext context;
    TraceArtifacts trace;
    Proof proof;
};

const ProofFixture& small_proof_fixture() {
    static const auto fixture = []() {
        const auto context = gatzk::protocol::build_context(small_debug_config());
        const auto trace = gatzk::protocol::build_trace(context);
        const auto proof = gatzk::protocol::prove(context, trace);
        return ProofFixture{context, trace, proof};
    }();
    return fixture;
}

std::unordered_map<std::string, gatzk::crypto::Commitment> commitment_map(
    const std::vector<std::pair<std::string, gatzk::crypto::Commitment>>& entries) {
    std::unordered_map<std::string, gatzk::crypto::Commitment> out;
    for (const auto& [name, commitment] : entries) {
        out[name] = commitment;
    }
    return out;
}

const gatzk::protocol::DomainOpeningBundle& bundle_by_name(const Proof& proof, const std::string& name) {
    for (const auto& [bundle_name, bundle] : proof.domain_openings) {
        if (bundle_name == name) {
            return bundle;
        }
    }
    throw std::runtime_error("missing bundle " + name);
}

bool bundle_contains(const gatzk::protocol::DomainOpeningBundle& bundle, const std::string& label) {
    for (const auto& [name, _] : bundle.values) {
        if (name == label) {
            return true;
        }
    }
    return false;
}

bool external_eval_contains(const Proof& proof, const std::string& label) {
    for (const auto& [name, _] : proof.external_evaluations) {
        if (name == label) {
            return true;
        }
    }
    return false;
}

void test_full_graph_config_parse() {
    const auto cora = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    require(cora.dataset == "cora" && cora.local_nodes == 2708 && cora.prove_enabled, "cora_full.cfg mismatch");
    require(!cora.checkpoint_bundle.empty(), "cora_full.cfg must define checkpoint_bundle");
    require(!cora.allow_synthetic_model, "cora_full.cfg must not enable synthetic parameters");
}

void test_full_cora_edges_are_dst_sorted_and_self_looped() {
    const auto& context = full_graph_debug_context();
    require(!context.local.edges.empty(), "full cora edge list should not be empty");

    std::vector<bool> has_self(context.local.num_nodes, false);
    std::vector<std::size_t> indegree(context.local.num_nodes, 0);
    std::size_t previous_dst = 0;
    for (std::size_t edge_index = 0; edge_index < context.local.edges.size(); ++edge_index) {
        const auto& edge = context.local.edges[edge_index];
        if (edge_index > 0) {
            require(previous_dst <= edge.dst, "full cora edges must be stable-sorted by dst");
        }
        previous_dst = edge.dst;
        indegree[edge.dst] += 1;
        if (edge.src == edge.dst) {
            has_self[edge.dst] = true;
        }
    }
    for (std::size_t node = 0; node < context.local.num_nodes; ++node) {
        require(has_self[node], "every full cora node must have an explicit self-loop");
        require(indegree[node] >= 1, "every full cora node must have at least one incoming edge");
    }
}

void test_full_cora_bias_matches_reference() {
    const auto& context = full_graph_debug_context();
    const auto bias = gatzk::model::build_attention_bias_matrix(context.local.num_nodes, context.local.edges);
    compare_matrix_npy(bias, ensure_reference_outputs() / "inputs/bias.npy", "bias");
}

void test_checkpoint_bundle_manifest_detects_architecture_mismatch() {
    const auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    const auto bundle = gatzk::model::inspect_checkpoint_bundle(repo_path(config.checkpoint_bundle));
    require(bundle.hidden_head_count == 8, "expected original GAT bundle to report 8 hidden heads");
    require(bundle.has_output_attention_head, "expected original GAT bundle to include output attention head");

    std::string reason;
    require(
        !gatzk::model::checkpoint_bundle_matches_single_head_protocol(bundle, &reason),
        "original GAT bundle must not be accepted by the current single-head protocol model");
    require(reason.find("hidden_head_count=8") != std::string::npos, "mismatch reason must expose hidden head count");
}

void test_real_checkpoint_bundle_loads_multihead_parameters() {
    const auto parameters = gatzk::model::load_checkpoint_bundle_parameters(repo_path("artifacts/checkpoints/cora_gat"));
    require(parameters.has_real_multihead, "real checkpoint bundle must mark the model as multi-head");
    require(parameters.hidden_heads.size() == 8, "expected 8 hidden heads in real checkpoint bundle");
    require(parameters.hidden_heads.front().seq_kernel_fp.size() == 1433, "hidden head input dimension mismatch");
    require(parameters.hidden_heads.front().seq_kernel_fp.front().size() == 8, "hidden head width mismatch");
    require(parameters.output_head.seq_kernel_fp.size() == 64, "output head input width mismatch");
    require(parameters.output_head.seq_kernel_fp.front().size() == 7, "output head class width mismatch");
    require(parameters.output_head.output_bias_fp.size() == 7, "output head bias width mismatch");
}

void test_reference_style_multihead_forward_shapes() {
    const auto& context = full_graph_debug_context();
    const auto parameters = gatzk::model::load_checkpoint_bundle_parameters(repo_path("artifacts/checkpoints/cora_gat"));
    const auto trace = gatzk::model::forward_reference_style(
        context.local.features_fp,
        context.local.edges,
        parameters);

    require(trace.hidden_head_traces.size() == 8, "expected 8 hidden head traces");
    require(trace.H.size() == context.local.num_nodes, "input feature row count mismatch");
    require(trace.H.front().size() == context.local.num_features, "input feature width mismatch");
    require(trace.hidden_head_traces.front().H_prime.size() == context.local.num_nodes, "hidden H_prime row count mismatch");
    require(trace.hidden_head_traces.front().H_prime.front().size() == 8, "hidden H_prime width mismatch");
    require(trace.hidden_head_traces.front().S.size() == context.local.edges.size(), "hidden edge-domain S size mismatch");
    require(trace.hidden_head_traces.front().alpha.size() == context.local.edges.size(), "hidden edge-domain alpha size mismatch");
    require(trace.hidden_concat.size() == context.local.num_nodes, "hidden concat row count mismatch");
    require(trace.hidden_concat.front().size() == 64, "hidden concat width mismatch");
    require(trace.Y_lin.size() == context.local.num_nodes, "Y_lin row count mismatch");
    require(trace.Y_lin.front().size() == context.local.num_classes, "Y_lin class width mismatch");
    require(trace.Y.size() == context.local.num_nodes, "Y row count mismatch");
    require(trace.Y.front().size() == context.local.num_classes, "Y class width mismatch");
}

void test_reference_style_multihead_forward_matches_reference_artifacts() {
    const auto& context = full_graph_debug_context();
    const auto checkpoint_dir = repo_path("artifacts/checkpoints/cora_gat");
    const auto& reference_dir = ensure_reference_outputs();
    const auto parameters = gatzk::model::load_checkpoint_bundle_parameters(checkpoint_dir);
    const auto trace = gatzk::model::forward_reference_style(
        context.local.features_fp,
        context.local.edges,
        parameters);

    compare_matrix_npy(trace.H, reference_dir / "inputs/H.npy", "H");
    for (std::size_t head = 0; head < trace.hidden_head_traces.size(); ++head) {
        const auto head_dir = reference_dir / ("hidden_head_" + std::to_string(head));
        const auto head_prefix = "hidden_head_" + std::to_string(head) + "/";
        const auto s_dense = dense_sum_scores(trace.hidden_head_traces[head].E_src, trace.hidden_head_traces[head].E_dst);
        const auto z_dense = dense_leaky_relu(s_dense);
        const auto masked_logits = dense_masked_add(z_dense, trace.bias);
        const auto delta_dense = dense_delta(masked_logits, trace.hidden_head_traces[head].M);
        const auto u_dense = dense_u(masked_logits, trace.hidden_head_traces[head].M);
        const auto alpha_dense = dense_alpha(u_dense, trace.hidden_head_traces[head].inv);
        compare_matrix_npy(trace.hidden_head_traces[head].H_prime, head_dir / "H_prime.npy", head_prefix + "H_prime");
        compare_vector_npy(trace.hidden_head_traces[head].E_src, head_dir / "E_src.npy", head_prefix + "E_src");
        compare_vector_npy(trace.hidden_head_traces[head].E_dst, head_dir / "E_dst.npy", head_prefix + "E_dst");
        compare_matrix_npy(s_dense, head_dir / "S.npy", head_prefix + "S");
        compare_matrix_npy(z_dense, head_dir / "Z.npy", head_prefix + "Z");
        compare_vector_npy(trace.hidden_head_traces[head].M, head_dir / "M.npy", head_prefix + "M");
        compare_matrix_npy(delta_dense, head_dir / "Delta.npy", head_prefix + "Delta");
        compare_matrix_npy(u_dense, head_dir / "U.npy", head_prefix + "U");
        compare_vector_npy(trace.hidden_head_traces[head].Sum, head_dir / "Sum.npy", head_prefix + "Sum");
        compare_vector_npy(trace.hidden_head_traces[head].inv, head_dir / "inv.npy", head_prefix + "inv");
        compare_matrix_npy(alpha_dense, head_dir / "alpha.npy", head_prefix + "alpha");
        compare_matrix_npy(trace.hidden_head_traces[head].H_agg, head_dir / "H_agg.npy", head_prefix + "H_agg");
    }

    compare_matrix_npy(trace.hidden_concat, reference_dir / "hidden_concat.npy", "hidden_concat");
    const auto output_s_dense = dense_sum_scores(trace.output_head_trace.E_src, trace.output_head_trace.E_dst);
    const auto output_z_dense = dense_leaky_relu(output_s_dense);
    const auto output_masked_logits = dense_masked_add(output_z_dense, trace.bias);
    const auto output_delta_dense = dense_delta(output_masked_logits, trace.output_head_trace.M);
    const auto output_u_dense = dense_u(output_masked_logits, trace.output_head_trace.M);
    const auto output_alpha_dense = dense_alpha(output_u_dense, trace.output_head_trace.inv);
    compare_matrix_npy(trace.output_head_trace.H_prime, reference_dir / "output/H_prime.npy", "output/H_prime");
    compare_vector_npy(trace.output_head_trace.E_src, reference_dir / "output/E_src.npy", "output/E_src");
    compare_vector_npy(trace.output_head_trace.E_dst, reference_dir / "output/E_dst.npy", "output/E_dst");
    compare_matrix_npy(output_s_dense, reference_dir / "output/S.npy", "output/S");
    compare_matrix_npy(output_z_dense, reference_dir / "output/Z.npy", "output/Z");
    compare_vector_npy(trace.output_head_trace.M, reference_dir / "output/M.npy", "output/M");
    compare_matrix_npy(output_delta_dense, reference_dir / "output/Delta.npy", "output/Delta");
    compare_matrix_npy(output_u_dense, reference_dir / "output/U.npy", "output/U");
    compare_vector_npy(trace.output_head_trace.Sum, reference_dir / "output/Sum.npy", "output/Sum");
    compare_vector_npy(trace.output_head_trace.inv, reference_dir / "output/inv.npy", "output/inv");
    compare_matrix_npy(output_alpha_dense, reference_dir / "output/alpha.npy", "output/alpha");
    compare_matrix_npy(trace.output_head_trace.H_agg, reference_dir / "output/H_agg.npy", "output/H_agg");
    compare_matrix_npy(trace.Y_lin, reference_dir / "output/Y_lin.npy", "output/Y_lin");
    compare_matrix_npy(trace.Y, reference_dir / "output/Y.npy", "output/Y");
}

void test_extract_full_graph_fast_path() {
    gatzk::data::GraphDataset dataset;
    dataset.name = "synthetic";
    dataset.num_nodes = 3;
    dataset.num_features = 2;
    dataset.num_classes = 2;
    dataset.features_fp = {
        {1.0, 0.0},
        {0.0, 1.0},
        {0.5, 0.5},
    };
    dataset.features = {
        {FieldElement(1), FieldElement(0)},
        {FieldElement(0), FieldElement(1)},
        {FieldElement(1), FieldElement(1)},
    };
    dataset.labels = {0, 1, 0};
    dataset.edges = {{0, 1}, {1, 2}};

    const auto local = gatzk::data::extract_local_subgraph(dataset, 0, dataset.num_nodes);
    require(local.num_nodes == dataset.num_nodes, "full graph extraction node count mismatch");
    require(local.absolute_ids.size() == dataset.num_nodes, "full graph absolute id size mismatch");
    require(local.absolute_ids[0] == 0 && local.absolute_ids[1] == 1 && local.absolute_ids[2] == 2, "full graph extraction should preserve natural order");
    require(local.edges.size() == dataset.edges.size() + dataset.num_nodes, "full graph extraction should append self loops");
    require(local.features_fp == dataset.features_fp, "full graph extraction should preserve float features");
}

void test_is_full_dataset_context() {
    const auto& context = full_graph_debug_context();
    require(context.local.num_nodes == context.dataset.num_nodes, "full config should build full dataset context");
    require(context.local.absolute_ids.front() == 0, "full dataset order should start from node 0");
    require(context.local.absolute_ids.back() == context.dataset.num_nodes - 1, "full dataset order should preserve natural ids");
    require(!context.local.features_fp.empty(), "full dataset context should retain float features");
    require(context.local.features_fp.front().size() == context.local.num_features, "float feature width mismatch");
}

void test_no_absolute_path_dependency() {
    const auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    require(config.project_root == repo_root().string(), "project_root should derive from config location");
    require(config.export_dir.rfind("/home/", 0) != 0, "export_dir must not be an absolute home path");
    require(config.checkpoint_bundle.rfind("/home/", 0) != 0, "checkpoint_bundle must not be an absolute home path");
}

void test_selector_padding_consistency() {
    const auto& context = full_graph_debug_context();
    const auto& q_edge_valid = context.public_polynomials.at("P_Q_edge_valid").data;
    const auto& q_qry_src = context.public_polynomials.at("P_Q_qry_src").data;
    const auto& q_qry_dst = context.public_polynomials.at("P_Q_qry_dst").data;
    const auto& q_n = context.public_polynomials.at("P_Q_N").data;
    const auto& q_proj = context.public_polynomials.at("P_Q_proj_valid").data;
    const auto& q_d = context.public_polynomials.at("P_Q_d_valid").data;

    for (std::size_t i = 0; i < context.local.edges.size(); ++i) {
        require(q_edge_valid[i] == FieldElement::one(), "edge valid selector must be 1 on real edges");
        require(q_qry_src[i] == q_edge_valid[i] && q_qry_dst[i] == q_edge_valid[i], "route query selectors must match edge valid selector");
    }
    require(q_edge_valid[context.local.edges.size()] == FieldElement::zero(), "edge valid selector padding mismatch");

    for (std::size_t i = 0; i < context.local.num_nodes; ++i) {
        require(q_n[i] == FieldElement::one(), "node selector must be 1 on real nodes");
    }
    require(q_n[context.local.num_nodes] == FieldElement::zero(), "node selector padding mismatch");

    for (std::size_t i = 0; i < context.local.num_features; ++i) {
        require(q_proj[i] == FieldElement::one(), "projection selector must be 1 on real feature indices");
    }
    require(q_proj[context.local.num_features] == FieldElement::zero(), "projection selector padding mismatch");

    for (std::size_t i = 0; i < context.config.hidden_dim; ++i) {
        require(q_d[i] == FieldElement::one(), "d selector must be 1 on real hidden dimensions");
    }
    require(q_d[context.config.hidden_dim] == FieldElement::zero(), "d selector padding mismatch");
}

void test_kzg_batch_opening() {
    const auto key = gatzk::crypto::KZG::setup(7);
    const auto domain = gatzk::algebra::RootOfUnityDomain::create("tiny", 4);
    const auto poly_a = gatzk::algebra::Polynomial::from_coefficients(
        "A",
        {FieldElement(1), FieldElement(2), FieldElement(3)});
    const auto poly_b = gatzk::algebra::Polynomial::from_evaluations(
        "B",
        {FieldElement(4), FieldElement(5), FieldElement(6), FieldElement(7)},
        domain);
    const auto commitment_a = gatzk::crypto::KZG::commit("A", poly_a, key);
    const auto commitment_b = gatzk::crypto::KZG::commit("B", poly_b, key);
    const std::vector<FieldElement> points = {FieldElement(9), FieldElement(11)};
    const std::vector<std::vector<FieldElement>> values = {
        {poly_a.evaluate(points[0]), poly_a.evaluate(points[1])},
        {poly_b.evaluate(points[0]), poly_b.evaluate(points[1])},
    };
    const auto witness = gatzk::crypto::KZG::open_batch(
        {commitment_a, commitment_b},
        points,
        values,
        FieldElement(13),
        key);
    require(
        gatzk::crypto::KZG::verify_batch(
            {commitment_a, commitment_b},
            points,
            values,
            FieldElement(13),
            witness,
            key),
        "batch opening verification failed");
}

void test_transcript_order_consistency() {
    const auto& fixture = small_proof_fixture();
    const auto replayed = gatzk::protocol::replay_challenges(
        fixture.context,
        commitment_map(fixture.proof.dynamic_commitments),
        commitment_map(fixture.proof.quotient_commitments));
    require(replayed == fixture.proof.challenges, "replayed transcript must match proof challenges");
    require(fixture.proof.challenges.at("eta_dst") == fixture.trace.challenges.at("eta_dst"), "eta_dst mismatch");
    require(fixture.proof.challenges.at("lambda_psq") == fixture.trace.challenges.at("lambda_psq"), "lambda_psq mismatch");
}

void test_agg_witness_commitment_opening_consistency() {
    const auto& fixture = small_proof_fixture();
    const auto dynamic = commitment_map(fixture.proof.dynamic_commitments);
    require(dynamic.contains("P_H_agg"), "missing P_H_agg commitment");
    require(dynamic.contains("P_H_agg_star"), "missing P_H_agg_star commitment");
    require(dynamic.contains("P_a_agg"), "missing P_a_agg commitment");
    require(dynamic.contains("P_b_agg"), "missing P_b_agg commitment");
    require(dynamic.contains("P_Acc_agg"), "missing P_Acc_agg commitment");
    require(external_eval_contains(fixture.proof, "mu_agg"), "missing mu_agg external evaluation");

    const auto& bundle_d = bundle_by_name(fixture.proof, "d");
    const auto& bundle_n = bundle_by_name(fixture.proof, "N");
    require(bundle_contains(bundle_d, "P_a_agg"), "d bundle missing P_a_agg");
    require(bundle_contains(bundle_d, "P_b_agg"), "d bundle missing P_b_agg");
    require(bundle_contains(bundle_d, "P_Acc_agg"), "d bundle missing P_Acc_agg");
    require(bundle_contains(bundle_n, "P_H_agg_star"), "N bundle missing P_H_agg_star");
}

void test_prove_verify_round_trip() {
    const auto& fixture = small_proof_fixture();
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "round-trip prove/verify should accept");
}

void test_tampered_witness_fails() {
    const auto& fixture = small_proof_fixture();
    auto tampered = fixture.proof;
    require(!tampered.witness_scalars.empty(), "proof should contain route witness scalars");
    tampered.witness_scalars.front().second += FieldElement::one();
    require(!gatzk::protocol::verify(fixture.context, tampered), "tampered witness scalar must be rejected");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"full_graph_config_parse", test_full_graph_config_parse},
        {"full_cora_edges_are_dst_sorted_and_self_looped", test_full_cora_edges_are_dst_sorted_and_self_looped},
        {"full_cora_bias_matches_reference", test_full_cora_bias_matches_reference},
        {"checkpoint_bundle_manifest_detects_architecture_mismatch", test_checkpoint_bundle_manifest_detects_architecture_mismatch},
        {"real_checkpoint_bundle_loads_multihead_parameters", test_real_checkpoint_bundle_loads_multihead_parameters},
        {"reference_style_multihead_forward_shapes", test_reference_style_multihead_forward_shapes},
        {"reference_style_multihead_forward_matches_reference_artifacts", test_reference_style_multihead_forward_matches_reference_artifacts},
        {"extract_full_graph_fast_path", test_extract_full_graph_fast_path},
        {"is_full_dataset_context", test_is_full_dataset_context},
        {"no_absolute_path_dependency", test_no_absolute_path_dependency},
        {"selector_padding_consistency", test_selector_padding_consistency},
        {"kzg_batch_opening", test_kzg_batch_opening},
        {"transcript_order_consistency", test_transcript_order_consistency},
        {"agg_witness_commitment_opening_consistency", test_agg_witness_commitment_opening_consistency},
        {"prove_verify_round_trip", test_prove_verify_round_trip},
        {"tampered_witness_fails", test_tampered_witness_fails},
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
