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
        throw std::runtime_error("failed to open file: " + path.string());
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

std::vector<std::size_t> parse_optional_size_array(
    const std::string& text,
    const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return {};
    }
    std::string body = match[1].str();
    std::replace(body.begin(), body.end(), ',', ' ');
    std::vector<std::size_t> out;
    std::istringstream input(body);
    std::size_t value = 0;
    while (input >> value) {
        out.push_back(value);
    }
    return out;
}

std::string parse_required_string(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing string key in checkpoint manifest: " + key);
    }
    return match[1].str();
}

std::string extract_array_body(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing array key in checkpoint manifest: " + key);
    }
    const auto array_begin = text.find('[', key_pos);
    if (array_begin == std::string::npos) {
        throw std::runtime_error("malformed array in checkpoint manifest: " + key);
    }
    std::size_t depth = 0;
    for (std::size_t index = array_begin; index < text.size(); ++index) {
        if (text[index] == '[') {
            ++depth;
        } else if (text[index] == ']') {
            --depth;
            if (depth == 0) {
                return text.substr(array_begin + 1, index - array_begin - 1);
            }
        }
    }
    throw std::runtime_error("unterminated array in checkpoint manifest: " + key);
}

std::vector<std::string> parse_object_array_entries(const std::string& text, const std::string& key) {
    const auto body = extract_array_body(text, key);
    std::vector<std::string> entries;
    std::size_t depth = 0;
    std::size_t object_begin = std::string::npos;
    for (std::size_t index = 0; index < body.size(); ++index) {
        if (body[index] == '{') {
            if (depth == 0) {
                object_begin = index;
            }
            ++depth;
        } else if (body[index] == '}') {
            if (depth == 0) {
                throw std::runtime_error("malformed object array in checkpoint manifest: " + key);
            }
            --depth;
            if (depth == 0 && object_begin != std::string::npos) {
                entries.push_back(body.substr(object_begin, index - object_begin + 1));
                object_begin = std::string::npos;
            }
        }
    }
    if (depth != 0) {
        throw std::runtime_error("unterminated object array in checkpoint manifest: " + key);
    }
    return entries;
}

CheckpointLayerInfo parse_checkpoint_layer_info(const std::string& entry) {
    CheckpointLayerInfo out;
    out.layer_index = parse_required_size_t(entry, "layer_index");
    out.input_dim = parse_required_size_t(entry, "input_dim");
    out.shape.head_count = parse_required_size_t(entry, "head_count");
    out.shape.head_dim = parse_required_size_t(entry, "head_dim");
    return out;
}

CheckpointHeadSpec parse_checkpoint_head_spec(const std::string& entry) {
    CheckpointHeadSpec out;
    out.layer_index = parse_required_size_t(entry, "layer_index");
    out.local_head_index = parse_required_size_t(entry, "local_head_index");
    out.global_head_index = parse_required_size_t(entry, "global_head_index");
    out.seq_kernel = parse_required_string(entry, "seq_kernel");
    out.attn_dst_kernel = parse_required_string(entry, "attn_dst_kernel");
    out.attn_dst_bias = parse_required_string(entry, "attn_dst_bias");
    out.attn_src_kernel = parse_required_string(entry, "attn_src_kernel");
    out.attn_src_bias = parse_required_string(entry, "attn_src_bias");
    out.output_bias = parse_required_string(entry, "output_bias");
    return out;
}

CheckpointOutputHeadSpec parse_checkpoint_output_head_spec(const std::string& entry) {
    CheckpointOutputHeadSpec out;
    out.head_index = parse_required_size_t(entry, "head_index");
    out.seq_kernel = parse_required_string(entry, "seq_kernel");
    out.attn_dst_kernel = parse_required_string(entry, "attn_dst_kernel");
    out.attn_dst_bias = parse_required_string(entry, "attn_dst_bias");
    out.attn_src_kernel = parse_required_string(entry, "attn_src_kernel");
    out.attn_src_bias = parse_required_string(entry, "attn_src_bias");
    out.output_bias = parse_required_string(entry, "output_bias");
    return out;
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

AttentionHeadParameters build_synthetic_head(
    std::size_t input_dim,
    std::size_t output_dim,
    std::uint64_t seed,
    std::size_t bias_seed) {
    AttentionHeadParameters head;
    head.seq_kernel_fp.assign(input_dim, std::vector<double>(output_dim, 0.0));
    for (std::size_t row = 0; row < input_dim; ++row) {
        bool has_non_zero = false;
        for (std::size_t col = 0; col < output_dim; ++col) {
            const auto enabled = ((row * 17 + col * 31 + seed + bias_seed) % 7U) == 0U;
            head.seq_kernel_fp[row][col] = enabled ? 1.0 : 0.0;
            has_non_zero = has_non_zero || enabled;
        }
        if (!has_non_zero && output_dim != 0) {
            head.seq_kernel_fp[row][row % output_dim] = 1.0;
        }
    }
    head.attn_src_kernel_fp.resize(output_dim, 0.0);
    head.attn_dst_kernel_fp.resize(output_dim, 0.0);
    for (std::size_t i = 0; i < output_dim; ++i) {
        head.attn_src_kernel_fp[i] = static_cast<double>(((i + seed + bias_seed) % 3U) + 1U) / 4.0;
        head.attn_dst_kernel_fp[i] = static_cast<double>(((i + seed + bias_seed + 1U) % 3U) + 1U) / 4.0;
    }
    head.attn_src_bias_fp = static_cast<double>(static_cast<std::int64_t>((seed + bias_seed) % 3U) - 1) / 8.0;
    head.attn_dst_bias_fp = static_cast<double>(static_cast<std::int64_t>((seed + bias_seed + 1U) % 3U) - 1) / 8.0;
    head.output_bias_fp.resize(output_dim, 0.0);
    for (std::size_t i = 0; i < output_dim; ++i) {
        head.output_bias_fp[i] = static_cast<double>(static_cast<std::int64_t>((i + bias_seed) % 5U) - 2) / 8.0;
    }
    return head;
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

FloatMatrix average_matrices(const std::vector<FloatMatrix>& matrices) {
    if (matrices.empty()) {
        return {};
    }
    auto out = matrices.front();
    for (std::size_t matrix_index = 1; matrix_index < matrices.size(); ++matrix_index) {
        if (matrices[matrix_index].size() != out.size()
            || (!out.empty() && matrices[matrix_index].front().size() != out.front().size())) {
            throw std::runtime_error("cannot average matrices with different shapes");
        }
        for (std::size_t row = 0; row < out.size(); ++row) {
            for (std::size_t col = 0; col < out[row].size(); ++col) {
                out[row][col] += matrices[matrix_index][row][col];
            }
        }
    }
    const auto scale = 1.0 / static_cast<double>(matrices.size());
    for (auto& row : out) {
        for (auto& value : row) {
            value *= scale;
        }
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
    params.L = 2;
    params.d_in_profile = {input_dim};
    params.hidden_profile = {{1, hidden_dim}};
    params.K_out = 1;
    params.C = num_classes;
    return params;
}

ModelParameters build_family_model_parameters(
    const std::vector<std::size_t>& d_in_profile,
    const std::vector<HiddenLayerShape>& hidden_profile,
    std::size_t k_out,
    std::size_t num_classes,
    std::uint64_t seed) {
    if (hidden_profile.empty()) {
        throw std::runtime_error("synthetic family model requires at least one hidden layer");
    }
    if (d_in_profile.size() != hidden_profile.size()) {
        throw std::runtime_error("synthetic family model requires d_in_profile size to match hidden_profile size");
    }
    if (k_out == 0) {
        throw std::runtime_error("synthetic family model requires K_out >= 1");
    }

    ModelParameters out;
    out.has_real_multihead = true;
    out.L = hidden_profile.size() + 1;
    out.d_in_profile = d_in_profile;
    out.hidden_profile = hidden_profile;
    out.K_out = k_out;
    out.C = num_classes;

    std::size_t global_hidden_head_index = 0;
    out.hidden_layers.reserve(hidden_profile.size());
    for (std::size_t layer_index = 0; layer_index < hidden_profile.size(); ++layer_index) {
        const auto& shape = hidden_profile[layer_index];
        HiddenLayerParameters layer;
        layer.layer_index = layer_index;
        layer.input_dim = d_in_profile[layer_index];
        layer.shape = shape;
        layer.heads.reserve(shape.head_count);
        for (std::size_t head_index = 0; head_index < shape.head_count; ++head_index) {
            layer.heads.push_back(build_synthetic_head(
                layer.input_dim,
                shape.head_dim,
                seed + layer_index,
                global_hidden_head_index));
            ++global_hidden_head_index;
        }
        out.hidden_layers.push_back(layer);
    }

    out.hidden_heads.reserve(global_hidden_head_index);
    for (const auto& layer : out.hidden_layers) {
        out.hidden_heads.insert(out.hidden_heads.end(), layer.heads.begin(), layer.heads.end());
    }

    out.output_layer.input_dim =
        hidden_profile.back().head_count * hidden_profile.back().head_dim;
    out.output_layer.output_dim = num_classes;
    out.output_layer.head_count = k_out;
    out.output_layer.heads.reserve(k_out);
    for (std::size_t head_index = 0; head_index < k_out; ++head_index) {
        out.output_layer.heads.push_back(build_synthetic_head(
            out.output_layer.input_dim,
            num_classes,
            seed + 101U,
            head_index));
    }
    out.output_head = out.output_layer.heads.front();
    return out;
}

CheckpointBundleInfo inspect_checkpoint_bundle(const std::string& bundle_root) {
    const auto bundle_path = std::filesystem::path(bundle_root);
    if (!std::filesystem::exists(bundle_path)) {
        throw std::runtime_error("checkpoint bundle path does not exist: " + bundle_path.string());
    }
    const auto manifest_path = bundle_path / "manifest.json";
    if (!std::filesystem::exists(manifest_path)) {
        throw std::runtime_error("checkpoint bundle manifest not found: " + manifest_path.string());
    }
    const auto manifest = load_text_file(manifest_path);

    CheckpointBundleInfo info;
    info.bundle_root = bundle_root;
    info.family_schema_version = parse_required_string(manifest, "family_schema_version");
    info.output_average_rule = parse_required_string(manifest, "output_average_rule");
    info.model_arch_id = parse_required_string(manifest, "model_arch_id");
    info.model_param_id = parse_required_string(manifest, "model_param_id");
    info.quant_cfg_id = parse_required_string(manifest, "quant_cfg_id");
    info.static_table_id = parse_required_string(manifest, "static_table_id");
    info.degree_bound_id = parse_required_string(manifest, "degree_bound_id");
    info.layer_count = parse_required_size_t(manifest, "L");
    info.output_head_count = parse_required_size_t(manifest, "K_out");
    info.class_count = parse_required_size_t(manifest, "C");
    info.d_in_profile = parse_optional_size_array(manifest, "d_in_profile");
    for (const auto& entry : parse_object_array_entries(manifest, "hidden_layers")) {
        info.hidden_layers.push_back(parse_checkpoint_layer_info(entry));
        info.hidden_profile.push_back(info.hidden_layers.back().shape);
    }
    for (const auto& entry : parse_object_array_entries(manifest, "hidden_head_specs")) {
        info.hidden_head_specs.push_back(parse_checkpoint_head_spec(entry));
    }
    for (const auto& entry : parse_object_array_entries(manifest, "output_head_specs")) {
        info.output_head_specs.push_back(parse_checkpoint_output_head_spec(entry));
    }
    info.has_output_attention_head = info.output_head_count > 0;
    return info;
}

bool checkpoint_bundle_matches_formal_proof_shape(
    const CheckpointBundleInfo& info,
    std::string* reason) {
    std::vector<std::string> failures;
    if (info.layer_count < 2) {
        failures.push_back("L=" + std::to_string(info.layer_count) + " but checkpoint manifest must expose L >= 2");
    }
    if (info.family_schema_version.empty()) {
        failures.push_back("checkpoint manifest must expose non-empty family_schema_version");
    }
    if (info.output_average_rule != "per_head_bias_then_arithmetic_mean") {
        failures.push_back(
            "checkpoint manifest output_average_rule="
            + info.output_average_rule
            + " but expected per_head_bias_then_arithmetic_mean");
    }
    if (info.hidden_layers.empty()) {
        failures.push_back("checkpoint manifest must expose non-empty hidden_layers");
    }
    if (info.hidden_profile.empty()) {
        failures.push_back(
            "hidden_profile size=" + std::to_string(info.hidden_profile.size())
            + " but the checkpoint manifest must expose a non-empty hidden_profile");
    }
    if (info.output_head_count == 0) {
        failures.push_back(
            "K_out=" + std::to_string(info.output_head_count)
            + " but checkpoint manifest must expose at least one output attention head");
    }
    if (info.output_head_specs.size() != info.output_head_count) {
        failures.push_back(
            "output_head_specs size=" + std::to_string(info.output_head_specs.size())
            + " conflicts with K_out=" + std::to_string(info.output_head_count));
    }
    if (!info.d_in_profile.empty() && info.d_in_profile.size() != info.hidden_profile.size()) {
        failures.push_back(
            "d_in_profile size=" + std::to_string(info.d_in_profile.size())
            + " conflicts with hidden_profile size=" + std::to_string(info.hidden_profile.size()));
    }
    if (info.model_arch_id.empty()) {
        failures.push_back("checkpoint manifest must expose non-empty model_arch_id");
    }
    if (info.model_param_id.empty()) {
        failures.push_back("checkpoint manifest must expose non-empty model_param_id");
    }
    if (info.quant_cfg_id.empty()) {
        failures.push_back("checkpoint manifest must expose non-empty quant_cfg_id");
    }
    std::size_t flattened_hidden_heads = 0;
    for (std::size_t layer_index = 0; layer_index < info.hidden_layers.size(); ++layer_index) {
        const auto& layer = info.hidden_layers[layer_index];
        if (layer.layer_index != layer_index) {
            failures.push_back("hidden_layers must use contiguous layer_index values");
            break;
        }
        if (!info.d_in_profile.empty() && info.d_in_profile[layer_index] != layer.input_dim) {
            failures.push_back(
                "hidden_layers[" + std::to_string(layer_index) + "].input_dim conflicts with d_in_profile");
        }
        if (info.hidden_profile[layer_index].head_count != layer.shape.head_count
            || info.hidden_profile[layer_index].head_dim != layer.shape.head_dim) {
            failures.push_back(
                "hidden_layers[" + std::to_string(layer_index) + "] conflicts with hidden_profile");
        }
        flattened_hidden_heads += layer.shape.head_count;
    }
    if (flattened_hidden_heads != info.hidden_head_specs.size()) {
        failures.push_back(
            "hidden_head_specs size=" + std::to_string(info.hidden_head_specs.size())
            + " conflicts with hidden_layers head counts=" + std::to_string(flattened_hidden_heads));
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
    std::string manifest_reason;
    if (!checkpoint_bundle_matches_formal_proof_shape(manifest_info, &manifest_reason)) {
        throw std::runtime_error("checkpoint bundle family metadata is invalid: " + manifest_reason);
    }
    const auto tensor_path = std::filesystem::path(bundle_root) / "tensors.txt";
    if (!std::filesystem::exists(tensor_path)) {
        throw std::runtime_error("checkpoint bundle tensor dump not found: " + tensor_path.string());
    }
    const auto tensors = load_text_tensors(tensor_path);

    ModelParameters out;
    out.has_real_multihead = true;
    out.L = manifest_info.layer_count;
    out.d_in_profile = manifest_info.d_in_profile;
    out.hidden_profile = manifest_info.hidden_profile;
    out.K_out = manifest_info.output_head_count;
    out.hidden_heads.reserve(manifest_info.hidden_head_specs.size());
    out.hidden_layers.reserve(manifest_info.hidden_layers.size());
    std::size_t expected_global_head_index = 0;
    for (const auto& layer_info : manifest_info.hidden_layers) {
        HiddenLayerParameters layer;
        layer.layer_index = layer_info.layer_index;
        layer.input_dim = layer_info.input_dim;
        layer.shape = layer_info.shape;
        layer.heads.reserve(layer.shape.head_count);
        for (const auto& head_spec : manifest_info.hidden_head_specs) {
            if (head_spec.layer_index != layer.layer_index) {
                continue;
            }
            if (head_spec.global_head_index != expected_global_head_index) {
                throw std::runtime_error(
                    "checkpoint manifest hidden_head_specs must use contiguous global_head_index values");
            }
            layer.heads.push_back(load_head_parameters(
                tensors,
                head_spec.seq_kernel,
                head_spec.attn_dst_kernel,
                head_spec.attn_dst_bias,
                head_spec.attn_src_kernel,
                head_spec.attn_src_bias,
                head_spec.output_bias));
            out.hidden_heads.push_back(layer.heads.back());
            ++expected_global_head_index;
        }
        if (layer.heads.size() != layer.shape.head_count) {
            throw std::runtime_error(
                "checkpoint manifest hidden_head_specs do not match hidden_layers head_count for layer "
                + std::to_string(layer.layer_index));
        }
        out.hidden_layers.push_back(std::move(layer));
    }
    out.output_layer.head_count = manifest_info.output_head_count;
    out.output_layer.heads.reserve(manifest_info.output_head_specs.size());
    for (const auto& head_spec : manifest_info.output_head_specs) {
        if (head_spec.head_index != out.output_layer.heads.size()) {
            throw std::runtime_error("checkpoint manifest output_head_specs must use contiguous head_index values");
        }
        out.output_layer.heads.push_back(load_head_parameters(
            tensors,
            head_spec.seq_kernel,
            head_spec.attn_dst_kernel,
            head_spec.attn_dst_bias,
            head_spec.attn_src_kernel,
            head_spec.attn_src_bias,
            head_spec.output_bias));
    }
    if (out.hidden_heads.empty()) {
        throw std::runtime_error("checkpoint bundle does not contain hidden attention heads");
    }
    if (out.output_layer.heads.empty()) {
        throw std::runtime_error("checkpoint bundle does not contain output attention heads");
    }
    if (out.d_in_profile.empty()) {
        for (const auto& layer : out.hidden_layers) {
            out.d_in_profile.push_back(layer.input_dim);
        }
    }
    if (out.hidden_profile.empty()) {
        for (const auto& layer : out.hidden_layers) {
            out.hidden_profile.push_back(layer.shape);
        }
    }
    out.C = manifest_info.class_count;
    out.output_layer.input_dim = out.hidden_layers.back().shape.head_count * out.hidden_layers.back().shape.head_dim;
    out.output_layer.output_dim = attention_head_output_width(out.output_layer.heads.front());
    if (out.C == 0) {
        out.C = out.output_layer.output_dim;
    }
    if (out.output_layer.output_dim != out.C) {
        throw std::runtime_error("checkpoint manifest C conflicts with output head output_dim");
    }
    out.output_head = out.output_layer.heads.front();
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
    auto layer_input = features;
    for (const auto& layer : parameters.hidden_layers) {
        HiddenLayerForwardTrace layer_trace;
        layer_trace.input = layer_input;
        std::vector<FloatMatrix> hidden_outputs;
        hidden_outputs.reserve(layer.heads.size());
        layer_trace.head_traces.reserve(layer.heads.size());
        for (const auto& head : layer.heads) {
            HeadForwardProfile head_profile;
            auto head_trace = attention_head_forward_impl(layer_input, edges, head, false, true, &head_profile);
            if (profile != nullptr) {
                profile->hidden_projection_ms += head_profile.projection_ms;
                profile->hidden_attention_ms += head_profile.attention_ms;
                profile->hidden_activation_ms += head_profile.activation_ms;
            }
            hidden_outputs.push_back(head_trace.H_agg);
            layer_trace.head_traces.push_back(head_trace);
            trace.hidden_head_traces.push_back(head_trace);
        }
        if (profile != nullptr) {
            const auto concat_start = std::chrono::steady_clock::now();
            layer_trace.concat = concatenate_columns(hidden_outputs);
            profile->hidden_concat_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - concat_start).count();
        } else {
            layer_trace.concat = concatenate_columns(hidden_outputs);
        }
        layer_input = layer_trace.concat;
        trace.hidden_concat = layer_trace.concat;
        trace.hidden_layer_traces.push_back(std::move(layer_trace));
    }

    std::vector<FloatMatrix> output_values;
    output_values.reserve(parameters.output_layer.heads.size());
    trace.output_head_traces.reserve(parameters.output_layer.heads.size());
    for (const auto& head : parameters.output_layer.heads) {
        HeadForwardProfile output_profile;
        auto head_trace = attention_head_forward_impl(layer_input, edges, head, true, false, &output_profile);
        if (profile != nullptr) {
            profile->output_projection_ms += output_profile.projection_ms;
            profile->output_attention_ms += output_profile.attention_ms;
            profile->output_activation_ms += output_profile.activation_ms;
        }
        output_values.push_back(head_trace.H_agg);
        trace.output_head_traces.push_back(head_trace);
    }
    trace.output_head_trace = trace.output_head_traces.front();
    trace.output_head_values = output_values;
    trace.Y_lin = average_matrices(output_values);
    trace.Y = trace.Y_lin;
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
    auto layer_input = features;
    for (const auto& layer : parameters.hidden_layers) {
        HiddenLayerForwardTrace layer_trace;
        layer_trace.input = layer_input;
        std::vector<FloatMatrix> hidden_outputs;
        hidden_outputs.reserve(layer.heads.size());
        layer_trace.head_traces.reserve(layer.heads.size());
        for (const auto& head : layer.heads) {
            HeadForwardProfile head_profile;
            auto head_trace = attention_head_forward_impl(layer_input, edges, head, false, true, &head_profile);
            if (profile != nullptr) {
                profile->hidden_projection_ms += head_profile.projection_ms;
                profile->hidden_attention_ms += head_profile.attention_ms;
                profile->hidden_activation_ms += head_profile.activation_ms;
            }
            hidden_outputs.push_back(head_trace.H_agg);
            layer_trace.head_traces.push_back(head_trace);
            if (layer.layer_index == 0) {
                trace.hidden_head_traces.push_back(std::move(head_trace));
            }
        }
        if (profile != nullptr) {
            const auto concat_start = std::chrono::steady_clock::now();
            layer_trace.concat = concatenate_columns(hidden_outputs);
            profile->hidden_concat_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - concat_start).count();
        } else {
            layer_trace.concat = concatenate_columns(hidden_outputs);
        }
        layer_input = layer_trace.concat;
        trace.hidden_concat = layer_trace.concat;
        trace.hidden_layer_traces.push_back(std::move(layer_trace));
    }

    std::vector<FloatMatrix> output_pre_bias;
    std::vector<FloatMatrix> output_values;
    output_pre_bias.reserve(parameters.output_layer.heads.size());
    output_values.reserve(parameters.output_layer.heads.size());
    trace.output_head_traces.reserve(parameters.output_layer.heads.size());
    for (const auto& head : parameters.output_layer.heads) {
        HeadForwardProfile output_profile;
        auto head_trace = attention_head_forward_impl(layer_input, edges, head, true, false, &output_profile);
        if (profile != nullptr) {
            profile->output_projection_ms += output_profile.projection_ms;
            profile->output_attention_ms += output_profile.attention_ms;
            profile->output_activation_ms += output_profile.activation_ms;
        }
            output_pre_bias.push_back(head_trace.H_agg_pre_bias);
            output_values.push_back(head_trace.H_agg);
            trace.output_head_traces.push_back(head_trace);
    }
    trace.output_head_trace = trace.output_head_traces.front();
    trace.output_head_values = output_values;
    trace.Y_lin = average_matrices(output_pre_bias);
    trace.Y = average_matrices(output_values);
    return trace;
}

bool supports_current_formal_proof_shape(const ModelParameters& parameters) {
    std::string reason;
    return parameters.has_real_multihead && hidden_family_dimension_chain_is_valid(parameters, &reason);
}

bool hidden_family_dimension_chain_is_valid(const ModelParameters& parameters, std::string* reason) {
    std::vector<std::string> failures;
    if (!parameters.has_real_multihead) {
        failures.push_back("formal family proof requires has_real_multihead=true");
    }
    if (parameters.L < 2) {
        failures.push_back("L must be >= 2");
    }
    if (parameters.hidden_layers.empty()) {
        failures.push_back("hidden_layers must be non-empty");
    }
    if (parameters.hidden_profile.size() != parameters.hidden_layers.size()) {
        failures.push_back("hidden_profile size conflicts with hidden_layers size");
    }
    if (!parameters.d_in_profile.empty() && parameters.d_in_profile.size() != parameters.hidden_layers.size()) {
        failures.push_back("d_in_profile size conflicts with hidden_layers size");
    }
    std::size_t expected_input_dim = 0;
    for (std::size_t layer_index = 0; layer_index < parameters.hidden_layers.size(); ++layer_index) {
        const auto& layer = parameters.hidden_layers[layer_index];
        if (layer.layer_index != layer_index) {
            failures.push_back("hidden_layers must use contiguous layer_index values");
            break;
        }
        if (layer.shape.head_count == 0 || layer.shape.head_dim == 0) {
            failures.push_back("hidden layer shape must have non-zero head_count and head_dim");
        }
        if (layer.heads.size() != layer.shape.head_count) {
            failures.push_back(
                "hidden layer " + std::to_string(layer_index)
                + " head_count conflicts with heads.size()");
        }
        const auto configured_input =
            !parameters.d_in_profile.empty() ? parameters.d_in_profile[layer_index] : layer.input_dim;
        if (layer.input_dim != configured_input) {
            failures.push_back(
                "hidden layer " + std::to_string(layer_index)
                + " input_dim conflicts with d_in_profile");
        }
        if (layer_index == 0) {
            expected_input_dim = configured_input;
        } else if (layer.input_dim != expected_input_dim) {
            failures.push_back(
                "hidden layer " + std::to_string(layer_index)
                + " input_dim=" + std::to_string(layer.input_dim)
                + " conflicts with previous concat width=" + std::to_string(expected_input_dim));
        }
        for (const auto& head : layer.heads) {
            if (head.seq_kernel_fp.size() != layer.input_dim) {
                failures.push_back(
                    "hidden layer " + std::to_string(layer_index)
                    + " seq kernel row count conflicts with layer input_dim");
                break;
            }
            if (attention_head_output_width(head) != layer.shape.head_dim) {
                failures.push_back(
                    "hidden layer " + std::to_string(layer_index)
                    + " head output width conflicts with head_dim");
                break;
            }
        }
        expected_input_dim = layer.shape.head_count * layer.shape.head_dim;
    }
    if (parameters.output_layer.head_count == 0 || parameters.output_layer.heads.empty()) {
        failures.push_back("output_layer must expose at least one output attention head");
    }
    if (parameters.output_layer.heads.size() != parameters.output_layer.head_count) {
        failures.push_back("output_layer head_count conflicts with heads.size()");
    }
    if (parameters.K_out != parameters.output_layer.head_count) {
        failures.push_back("K_out conflicts with output_layer head_count");
    }
    if (!parameters.hidden_layers.empty() && parameters.output_layer.input_dim != expected_input_dim) {
        failures.push_back(
            "output_layer input_dim=" + std::to_string(parameters.output_layer.input_dim)
            + " conflicts with final hidden concat width=" + std::to_string(expected_input_dim));
    }
    for (const auto& head : parameters.output_layer.heads) {
        if (head.seq_kernel_fp.size() != parameters.output_layer.input_dim) {
            failures.push_back("output head seq kernel row count conflicts with output_layer input_dim");
            break;
        }
        if (attention_head_output_width(head) != parameters.output_layer.output_dim) {
            failures.push_back("output head output width conflicts with output_layer output_dim");
            break;
        }
    }
    if (parameters.C != parameters.output_layer.output_dim) {
        failures.push_back("C conflicts with output_layer output_dim");
    }
    if (parameters.L != parameters.hidden_layers.size() + 1) {
        failures.push_back("L conflicts with hidden layer family size");
    }
    if (flattened_hidden_head_count(parameters) != parameters.hidden_heads.size()) {
        failures.push_back("hidden_heads compatibility view must flatten all hidden layers");
    }
    if (reason != nullptr && !failures.empty()) {
        std::ostringstream stream;
        for (std::size_t i = 0; i < failures.size(); ++i) {
            if (i != 0) {
                stream << "; ";
            }
            stream << failures[i];
        }
        *reason = stream.str();
    }
    return failures.empty();
}

std::size_t flattened_hidden_head_count(const ModelParameters& parameters) {
    std::size_t count = 0;
    for (const auto& layer : parameters.hidden_layers) {
        count += layer.heads.size();
    }
    return count;
}

std::size_t max_hidden_input_dim(const ModelParameters& parameters) {
    std::size_t value = 0;
    for (const auto& layer : parameters.hidden_layers) {
        value = std::max(value, layer.input_dim);
    }
    return value;
}

std::size_t max_hidden_head_dim(const ModelParameters& parameters) {
    std::size_t value = 0;
    for (const auto& layer : parameters.hidden_layers) {
        value = std::max(value, layer.shape.head_dim);
    }
    return value;
}

std::size_t max_hidden_concat_width(const ModelParameters& parameters) {
    std::size_t value = 0;
    for (const auto& layer : parameters.hidden_layers) {
        value = std::max(value, layer.shape.head_count * layer.shape.head_dim);
    }
    return value;
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
