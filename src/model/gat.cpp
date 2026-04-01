#include "gatzk/model/gat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace gatzk::model {
namespace {

std::string load_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint manifest: " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::size_t parse_required_size_t(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric key in checkpoint manifest: " + key);
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::size_t shape_product(const std::vector<std::size_t>& shape) {
    std::size_t out = 1;
    for (const auto dim : shape) {
        out *= dim;
    }
    return out;
}

std::unordered_map<std::string, DenseTensor> load_text_tensors(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint tensor dump: " + path.string());
    }

    std::unordered_map<std::string, DenseTensor> out;
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") {
            throw std::runtime_error("malformed tensor dump marker in " + path.string());
        }
        std::string name;
        std::size_t rank = 0;
        input >> name >> rank;
        DenseTensor tensor;
        tensor.shape.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            input >> tensor.shape[i];
        }
        std::size_t count = 0;
        input >> count;
        tensor.values.resize(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            input >> tensor.values[i];
        }
        if (count != shape_product(tensor.shape)) {
            throw std::runtime_error("tensor dump count mismatch for " + name);
        }
        out.emplace(name, std::move(tensor));
    }
    return out;
}

const DenseTensor& require_tensor(
    const std::unordered_map<std::string, DenseTensor>& tensors,
    const std::string& name) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor in checkpoint dump: " + name);
    }
    return it->second;
}

FloatMatrix reshape_seq_kernel(const DenseTensor& tensor) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1) {
        throw std::runtime_error("expected seq kernel tensor shape [1, in, out]");
    }
    const auto rows = tensor.shape[1];
    const auto cols = tensor.shape[2];
    FloatMatrix out(rows, std::vector<double>(cols, 0.0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[row][col] = tensor.values[row * cols + col];
        }
    }
    return out;
}

std::vector<double> reshape_attention_kernel(const DenseTensor& tensor) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1) {
        throw std::runtime_error("expected attention kernel tensor shape [1, width, 1]");
    }
    return tensor.values;
}

double reshape_scalar_bias(const DenseTensor& tensor) {
    if (tensor.shape.size() != 1 || tensor.shape[0] != 1 || tensor.values.size() != 1) {
        throw std::runtime_error("expected scalar bias tensor shape [1]");
    }
    return tensor.values.front();
}

std::vector<double> reshape_output_bias(const DenseTensor& tensor) {
    if (tensor.shape.size() != 1) {
        throw std::runtime_error("expected output bias tensor shape [width]");
    }
    return tensor.values;
}

AttentionHeadParameters load_head_parameters(
    const std::unordered_map<std::string, DenseTensor>& tensors,
    const std::string& seq_kernel_name,
    const std::string& attn_dst_kernel_name,
    const std::string& attn_dst_bias_name,
    const std::string& attn_src_kernel_name,
    const std::string& attn_src_bias_name,
    const std::string& output_bias_name) {
    AttentionHeadParameters out;
    out.seq_kernel_fp = reshape_seq_kernel(require_tensor(tensors, seq_kernel_name));
    out.attn_dst_kernel_fp = reshape_attention_kernel(require_tensor(tensors, attn_dst_kernel_name));
    out.attn_dst_bias_fp = reshape_scalar_bias(require_tensor(tensors, attn_dst_bias_name));
    out.attn_src_kernel_fp = reshape_attention_kernel(require_tensor(tensors, attn_src_kernel_name));
    out.attn_src_bias_fp = reshape_scalar_bias(require_tensor(tensors, attn_src_bias_name));
    out.output_bias_fp = reshape_output_bias(require_tensor(tensors, output_bias_name));
    return out;
}

std::string hidden_seq_kernel_name(std::size_t head_index) {
    if (head_index == 0) {
        return "conv1d/kernel";
    }
    return "conv1d_" + std::to_string(head_index * 3) + "/kernel";
}

std::string hidden_dst_kernel_name(std::size_t head_index) {
    return "conv1d_" + std::to_string(head_index == 0 ? 1 : head_index * 3 + 1) + "/kernel";
}

std::string hidden_dst_bias_name(std::size_t head_index) {
    return "conv1d_" + std::to_string(head_index == 0 ? 1 : head_index * 3 + 1) + "/bias";
}

std::string hidden_src_kernel_name(std::size_t head_index) {
    return "conv1d_" + std::to_string(head_index == 0 ? 2 : head_index * 3 + 2) + "/kernel";
}

std::string hidden_src_bias_name(std::size_t head_index) {
    return "conv1d_" + std::to_string(head_index == 0 ? 2 : head_index * 3 + 2) + "/bias";
}

std::string hidden_output_bias_name(std::size_t head_index) {
    if (head_index == 0) {
        return "BiasAdd/biases";
    }
    return "BiasAdd_" + std::to_string(head_index) + "/biases";
}

double leaky_relu(double value, double alpha = 0.2) {
    return value >= 0.0 ? value : alpha * value;
}

double elu(double value) {
    return value >= 0.0 ? value : std::expm1(value);
}

FloatMatrix matmul(const FloatMatrix& left, const FloatMatrix& right) {
    if (left.empty() || right.empty()) {
        return {};
    }
    const auto shared = left.front().size();
    if (right.size() != shared) {
        throw std::runtime_error("float matrix multiply dimension mismatch");
    }
    const auto rows = left.size();
    const auto cols = right.front().size();
    FloatMatrix out(rows, std::vector<double>(cols, 0.0));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < shared; ++k) {
            const auto lhs = left[i][k];
            for (std::size_t j = 0; j < cols; ++j) {
                out[i][j] += lhs * right[k][j];
            }
        }
    }
    return out;
}

std::vector<double> matvec(const FloatMatrix& matrix, const std::vector<double>& vector, double bias) {
    std::vector<double> out(matrix.size(), bias);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t col = 0; col < vector.size(); ++col) {
            out[row] += matrix[row][col] * vector[col];
        }
    }
    return out;
}

FloatMatrix concatenate_columns(const std::vector<FloatMatrix>& matrices) {
    if (matrices.empty()) {
        return {};
    }
    const auto rows = matrices.front().size();
    std::size_t cols = 0;
    for (const auto& matrix : matrices) {
        if (matrix.size() != rows) {
            throw std::runtime_error("cannot concatenate matrices with different row counts");
        }
        cols += matrix.empty() ? 0 : matrix.front().size();
    }

    FloatMatrix out(rows, std::vector<double>(cols, 0.0));
    std::size_t offset = 0;
    for (const auto& matrix : matrices) {
        const auto width = matrix.empty() ? 0 : matrix.front().size();
        for (std::size_t row = 0; row < rows; ++row) {
            std::copy(matrix[row].begin(), matrix[row].end(), out[row].begin() + static_cast<std::ptrdiff_t>(offset));
        }
        offset += width;
    }
    return out;
}

HeadForwardTrace attention_head_forward_impl(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const AttentionHeadParameters& parameters,
    bool apply_output_bias,
    bool apply_activation,
    HeadForwardProfile* profile) {
    using Clock = std::chrono::steady_clock;
    auto stage_start = Clock::now();
    if (features.empty()) {
        return {};
    }
    const auto node_count = features.size();
    const auto width = attention_head_output_width(parameters);
    for (const auto& edge : edges) {
        if (edge.src >= node_count || edge.dst >= node_count) {
            throw std::runtime_error("edge out of bounds while running attention head forward");
        }
    }

    HeadForwardTrace trace;
    trace.H_prime = matmul(features, parameters.seq_kernel_fp);
    if (profile != nullptr) {
        profile->projection_ms += std::chrono::duration<double, std::milli>(Clock::now() - stage_start).count();
    }

    stage_start = Clock::now();
    trace.E_dst = matvec(trace.H_prime, parameters.attn_dst_kernel_fp, parameters.attn_dst_bias_fp);
    trace.E_src = matvec(trace.H_prime, parameters.attn_src_kernel_fp, parameters.attn_src_bias_fp);
    trace.S.assign(edges.size(), 0.0);
    trace.Z.assign(edges.size(), 0.0);
    trace.M.assign(node_count, 0.0);
    trace.Delta.assign(edges.size(), 0.0);
    trace.U.assign(edges.size(), 0.0);
    trace.Sum.assign(node_count, 0.0);
    trace.inv.assign(node_count, 0.0);
    trace.alpha.assign(edges.size(), 0.0);
    trace.H_agg_pre_bias.assign(node_count, std::vector<double>(width, 0.0));
    trace.H_agg.assign(node_count, std::vector<double>(width, 0.0));

    std::vector<std::vector<std::size_t>> incoming_sources(node_count);
    for (const auto& edge : edges) {
        incoming_sources[edge.dst].push_back(edge.src);
    }
    for (std::size_t dst = 0; dst < node_count; ++dst) {
        const auto has_self = std::find(
                                  incoming_sources[dst].begin(),
                                  incoming_sources[dst].end(),
                                  dst)
            != incoming_sources[dst].end();
        if (!has_self) {
            incoming_sources[dst].push_back(dst);
        }
    }

    for (std::size_t dst = 0; dst < node_count; ++dst) {
        trace.M[dst] = -std::numeric_limits<double>::infinity();
    }
    for (std::size_t dst = 0; dst < node_count; ++dst) {
        for (const auto src : incoming_sources[dst]) {
            trace.M[dst] = std::max(trace.M[dst], leaky_relu(trace.E_dst[dst] + trace.E_src[src]));
        }
    }
    for (std::size_t dst = 0; dst < node_count; ++dst) {
        for (const auto src : incoming_sources[dst]) {
            trace.Sum[dst] += std::exp(leaky_relu(trace.E_dst[dst] + trace.E_src[src]) - trace.M[dst]);
        }
    }
    for (std::size_t dst = 0; dst < node_count; ++dst) {
        trace.inv[dst] = 1.0 / trace.Sum[dst];
    }
    for (std::size_t dst = 0; dst < node_count; ++dst) {
        for (const auto src : incoming_sources[dst]) {
            const auto alpha = std::exp(leaky_relu(trace.E_dst[dst] + trace.E_src[src]) - trace.M[dst]) * trace.inv[dst];
            for (std::size_t col = 0; col < width; ++col) {
                trace.H_agg_pre_bias[dst][col] += alpha * trace.H_prime[src][col];
            }
        }
    }
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        trace.S[edge_index] = trace.E_dst[edge.dst] + trace.E_src[edge.src];
        trace.Z[edge_index] = leaky_relu(trace.S[edge_index]);
        trace.Delta[edge_index] = trace.M[edge.dst] - trace.Z[edge_index];
        trace.U[edge_index] = std::exp(trace.Z[edge_index] - trace.M[edge.dst]);
        trace.alpha[edge_index] = trace.U[edge_index] * trace.inv[edge.dst];
    }
    if (profile != nullptr) {
        profile->attention_ms += std::chrono::duration<double, std::milli>(Clock::now() - stage_start).count();
    }

    stage_start = Clock::now();
    for (std::size_t dst = 0; dst < node_count; ++dst) {
        for (std::size_t col = 0; col < width; ++col) {
            auto value = trace.H_agg_pre_bias[dst][col];
            if (apply_output_bias && col < parameters.output_bias_fp.size()) {
                value += parameters.output_bias_fp[col];
            }
            trace.H_agg[dst][col] = apply_activation ? elu(value) : value;
        }
    }
    if (profile != nullptr) {
        profile->activation_ms += std::chrono::duration<double, std::milli>(Clock::now() - stage_start).count();
    }

    return trace;
}

}  // namespace

ModelParameters build_model_parameters(
    std::size_t input_dim,
    std::size_t hidden_dim,
    std::size_t num_classes,
    std::uint64_t seed) {
    ModelParameters params;
    params.W.assign(input_dim, std::vector<algebra::FieldElement>(hidden_dim, algebra::FieldElement::zero()));
    for (std::size_t row = 0; row < input_dim; ++row) {
        bool has_non_zero = false;
        for (std::size_t col = 0; col < hidden_dim; ++col) {
            const auto enabled = ((row * 17 + col * 31 + seed) % 7U) == 0U;
            params.W[row][col] = enabled ? algebra::FieldElement::one() : algebra::FieldElement::zero();
            has_non_zero = has_non_zero || enabled;
        }
        if (!has_non_zero) {
            params.W[row][row % hidden_dim] = algebra::FieldElement::one();
        }
    }

    params.a_src.resize(hidden_dim, algebra::FieldElement::one());
    params.a_dst.resize(hidden_dim, algebra::FieldElement::one());
    for (std::size_t i = 0; i < hidden_dim; ++i) {
        params.a_src[i] = algebra::FieldElement::from_signed(static_cast<std::int64_t>((i + seed) % 2U + 1U));
        params.a_dst[i] = algebra::FieldElement::from_signed(static_cast<std::int64_t>((i + seed + 1U) % 2U + 1U));
    }

    params.W_out.assign(hidden_dim, std::vector<algebra::FieldElement>(num_classes, algebra::FieldElement::zero()));
    for (std::size_t row = 0; row < hidden_dim; ++row) {
        bool has_non_zero = false;
        for (std::size_t col = 0; col < num_classes; ++col) {
            const auto enabled = ((row * 13 + col * 19 + seed) % 5U) == 0U;
            params.W_out[row][col] = enabled ? algebra::FieldElement::one() : algebra::FieldElement::zero();
            has_non_zero = has_non_zero || enabled;
        }
        if (!has_non_zero) {
            params.W_out[row][row % num_classes] = algebra::FieldElement::one();
        }
    }

    params.b.resize(num_classes, algebra::FieldElement::zero());
    for (std::size_t i = 0; i < num_classes; ++i) {
        params.b[i] = algebra::FieldElement::from_signed(static_cast<std::int64_t>(i % 3U));
    }
    return params;
}

CheckpointBundleInfo inspect_checkpoint_bundle(const std::string& bundle_root) {
    const auto manifest_path = std::filesystem::path(bundle_root) / "manifest.json";
    const auto manifest = load_text_file(manifest_path);

    CheckpointBundleInfo info;
    info.bundle_root = bundle_root;
    info.hidden_head_count = parse_required_size_t(manifest, "hidden_head_count");
    info.has_output_attention_head = manifest.find("\"output_head\"") != std::string::npos;
    return info;
}

bool checkpoint_bundle_matches_single_head_protocol(
    const CheckpointBundleInfo& info,
    std::string* reason) {
    std::vector<std::string> failures;
    if (info.hidden_head_count != 1) {
        failures.push_back(
            "hidden_head_count=" + std::to_string(info.hidden_head_count)
            + " but the current protocol model expects exactly 1 hidden attention head");
    }
    if (info.has_output_attention_head) {
        failures.push_back(
            "the exported bundle contains an output attention head, while the current protocol model expects affine output parameters W_out/b");
    }
    if (failures.empty()) {
        return true;
    }
    if (reason != nullptr) {
        std::ostringstream stream;
        for (std::size_t i = 0; i < failures.size(); ++i) {
            if (i != 0) {
                stream << "; ";
            }
            stream << failures[i];
        }
        *reason = stream.str();
    }
    return false;
}

ModelParameters load_checkpoint_bundle_parameters(const std::string& bundle_root) {
    const auto manifest_info = inspect_checkpoint_bundle(bundle_root);
    const auto tensors = load_text_tensors(std::filesystem::path(bundle_root) / "tensors.txt");

    ModelParameters out;
    out.has_real_multihead = true;
    out.hidden_heads.reserve(manifest_info.hidden_head_count);
    for (std::size_t head_index = 0; head_index < manifest_info.hidden_head_count; ++head_index) {
        out.hidden_heads.push_back(load_head_parameters(
            tensors,
            hidden_seq_kernel_name(head_index),
            hidden_dst_kernel_name(head_index),
            hidden_dst_bias_name(head_index),
            hidden_src_kernel_name(head_index),
            hidden_src_bias_name(head_index),
            hidden_output_bias_name(head_index)));
    }
    out.output_head = load_head_parameters(
        tensors,
        "conv1d_24/kernel",
        "conv1d_25/kernel",
        "conv1d_25/bias",
        "conv1d_26/kernel",
        "conv1d_26/bias",
        "BiasAdd_8/biases");
    return out;
}

std::size_t attention_head_output_width(const AttentionHeadParameters& parameters) {
    if (!parameters.seq_kernel_fp.empty()) {
        return parameters.seq_kernel_fp.front().size();
    }
    return parameters.output_bias_fp.size();
}

FloatMatrix build_attention_bias_matrix(std::size_t num_nodes, const std::vector<data::Edge>& edges) {
    FloatMatrix bias(num_nodes, std::vector<double>(num_nodes, -1e9));
    for (std::size_t node = 0; node < num_nodes; ++node) {
        bias[node][node] = 0.0;
    }
    for (const auto& edge : edges) {
        if (edge.src >= num_nodes || edge.dst >= num_nodes) {
            throw std::runtime_error("edge out of bounds while building attention bias");
        }
        bias[edge.dst][edge.src] = 0.0;
    }
    return bias;
}

HeadForwardTrace attention_head_forward(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const AttentionHeadParameters& parameters,
    HeadForwardProfile* profile) {
    return attention_head_forward_impl(features, edges, parameters, true, true, profile);
}

MultiHeadForwardTrace forward_reference_style(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const ModelParameters& parameters,
    ForwardProfile* profile) {
    if (!parameters.has_real_multihead) {
        throw std::runtime_error("forward_reference_style requires real multi-head checkpoint parameters");
    }

    MultiHeadForwardTrace trace;
    trace.H = features;
    trace.bias = build_attention_bias_matrix(features.size(), edges);

    std::vector<FloatMatrix> hidden_outputs;
    hidden_outputs.reserve(parameters.hidden_heads.size());
    trace.hidden_head_traces.reserve(parameters.hidden_heads.size());
    for (const auto& head : parameters.hidden_heads) {
        HeadForwardProfile head_profile;
        auto head_trace = attention_head_forward_impl(features, edges, head, true, true, &head_profile);
        if (profile != nullptr) {
            profile->hidden_projection_ms += head_profile.projection_ms;
            profile->hidden_attention_ms += head_profile.attention_ms;
            profile->hidden_activation_ms += head_profile.activation_ms;
        }
        hidden_outputs.push_back(head_trace.H_agg);
        trace.hidden_head_traces.push_back(std::move(head_trace));
    }
    if (profile != nullptr) {
        const auto concat_start = std::chrono::steady_clock::now();
        trace.hidden_concat = concatenate_columns(hidden_outputs);
        profile->hidden_concat_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - concat_start).count();
    } else {
        trace.hidden_concat = concatenate_columns(hidden_outputs);
    }
    HeadForwardProfile output_profile;
    trace.output_head_trace = attention_head_forward_impl(trace.hidden_concat, edges, parameters.output_head, true, true, &output_profile);
    if (profile != nullptr) {
        profile->output_projection_ms += output_profile.projection_ms;
        profile->output_attention_ms += output_profile.attention_ms;
        profile->output_activation_ms += output_profile.activation_ms;
    }
    trace.Y_lin = trace.output_head_trace.H_prime;
    trace.Y = trace.output_head_trace.H_agg;
    return trace;
}

MultiHeadForwardTrace forward_note_style(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const ModelParameters& parameters,
    ForwardProfile* profile) {
    if (!parameters.has_real_multihead) {
        throw std::runtime_error("forward_note_style requires real multi-head checkpoint parameters");
    }

    MultiHeadForwardTrace trace;
    trace.H = features;
    trace.bias = build_attention_bias_matrix(features.size(), edges);

    std::vector<FloatMatrix> hidden_outputs;
    hidden_outputs.reserve(parameters.hidden_heads.size());
    trace.hidden_head_traces.reserve(parameters.hidden_heads.size());
    for (const auto& head : parameters.hidden_heads) {
        HeadForwardProfile head_profile;
        auto head_trace = attention_head_forward_impl(features, edges, head, false, true, &head_profile);
        if (profile != nullptr) {
            profile->hidden_projection_ms += head_profile.projection_ms;
            profile->hidden_attention_ms += head_profile.attention_ms;
            profile->hidden_activation_ms += head_profile.activation_ms;
        }
        hidden_outputs.push_back(head_trace.H_agg);
        trace.hidden_head_traces.push_back(std::move(head_trace));
    }
    if (profile != nullptr) {
        const auto concat_start = std::chrono::steady_clock::now();
        trace.hidden_concat = concatenate_columns(hidden_outputs);
        profile->hidden_concat_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - concat_start).count();
    } else {
        trace.hidden_concat = concatenate_columns(hidden_outputs);
    }
    HeadForwardProfile output_profile;
    trace.output_head_trace = attention_head_forward_impl(trace.hidden_concat, edges, parameters.output_head, false, false, &output_profile);
    if (profile != nullptr) {
        profile->output_projection_ms += output_profile.projection_ms;
        profile->output_attention_ms += output_profile.attention_ms;
        profile->output_activation_ms += output_profile.activation_ms;
    }
    trace.Y_lin = trace.output_head_trace.H_prime;
    trace.Y = trace.output_head_trace.H_agg;
    return trace;
}

Matrix project_features(const Matrix& left, const Matrix& right) {
    if (left.empty() || right.empty()) {
        return {};
    }
    const std::size_t shared = left.front().size();
    if (right.size() != shared) {
        throw std::runtime_error("matrix multiply dimension mismatch");
    }
    const std::size_t rows = left.size();
    const std::size_t cols = right.front().size();
    Matrix out(rows, std::vector<algebra::FieldElement>(cols, algebra::FieldElement::zero()));
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const auto task_count = rows >= 64 && cpu_count > 1 ? std::min<std::size_t>(cpu_count, rows) : 1;
    const auto chunk_size = (rows + task_count - 1) / task_count;

    std::vector<std::future<void>> futures;
    futures.reserve(task_count);
    for (std::size_t task = 0; task < task_count; ++task) {
        const auto begin = task * chunk_size;
        const auto end = std::min(rows, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.push_back(std::async(std::launch::async, [&, begin, end]() {
            for (std::size_t i = begin; i < end; ++i) {
                std::vector<mcl::Fr> native_row(cols);
                for (auto& value : native_row) {
                    value.clear();
                }
                for (std::size_t k = 0; k < shared; ++k) {
                    const auto& lhs = left[i][k].native();
                    for (std::size_t j = 0; j < cols; ++j) {
                        mcl::Fr term;
                        mcl::Fr::mul(term, lhs, right[k][j].native());
                        mcl::Fr::add(native_row[j], native_row[j], term);
                    }
                }
                for (std::size_t j = 0; j < cols; ++j) {
                    out[i][j] = algebra::FieldElement::from_native(native_row[j]);
                }
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    return out;
}

std::vector<algebra::FieldElement> matvec_projection(const Matrix& matrix, const std::vector<algebra::FieldElement>& vector) {
    std::vector<algebra::FieldElement> out(matrix.size(), algebra::FieldElement::zero());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        mcl::Fr sum;
        sum.clear();
        for (std::size_t j = 0; j < vector.size(); ++j) {
            mcl::Fr term;
            mcl::Fr::mul(term, matrix[i][j].native(), vector[j].native());
            mcl::Fr::add(sum, sum, term);
        }
        out[i] = algebra::FieldElement::from_native(sum);
    }
    return out;
}

std::vector<algebra::FieldElement> compress_rows(const Matrix& matrix, const algebra::FieldElement& challenge) {
    std::vector<algebra::FieldElement> out(matrix.size(), algebra::FieldElement::zero());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        mcl::Fr power = 1;
        mcl::Fr sum;
        sum.clear();
        for (const auto& value : matrix[i]) {
            mcl::Fr term;
            mcl::Fr::mul(term, value.native(), power);
            mcl::Fr::add(sum, sum, term);
            mcl::Fr::mul(power, power, challenge.native());
        }
        out[i] = algebra::FieldElement::from_native(sum);
    }
    return out;
}

Matrix aggregate_by_edges(
    const Matrix& h_prime,
    const std::vector<algebra::FieldElement>& alpha,
    const std::vector<data::Edge>& edges,
    std::size_t num_nodes) {
    const std::size_t width = h_prime.front().size();
    Matrix out(num_nodes, std::vector<algebra::FieldElement>(width, algebra::FieldElement::zero()));
    std::vector<std::vector<mcl::Fr>> native_out(num_nodes, std::vector<mcl::Fr>(width));
    for (auto& row : native_out) {
        for (auto& value : row) {
            value.clear();
        }
    }
    for (std::size_t k = 0; k < edges.size(); ++k) {
        const auto& alpha_k = alpha[k].native();
        for (std::size_t j = 0; j < width; ++j) {
            mcl::Fr term;
            mcl::Fr::mul(term, alpha_k, h_prime[edges[k].src][j].native());
            mcl::Fr::add(native_out[edges[k].dst][j], native_out[edges[k].dst][j], term);
        }
    }
    for (std::size_t i = 0; i < num_nodes; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            out[i][j] = algebra::FieldElement::from_native(native_out[i][j]);
        }
    }
    return out;
}

Matrix output_projection(
    const Matrix& h_agg,
    const Matrix& w_out,
    const std::vector<algebra::FieldElement>& bias,
    Matrix* linear_part) {
    auto linear = project_features(h_agg, w_out);
    Matrix out = linear;
    for (auto& row : out) {
        for (std::size_t j = 0; j < bias.size(); ++j) {
            row[j] += bias[j];
        }
    }
    if (linear_part != nullptr) {
        *linear_part = linear;
    }
    return out;
}

}
