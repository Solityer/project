#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/util/config.hpp"

namespace {

std::int64_t quantize_float_raw(double value) {
    return static_cast<std::int64_t>(value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5);
}

std::int64_t quantize_lrelu_raw(std::int64_t raw) {
    const auto input = static_cast<double>(raw) / 16.0;
    return quantize_float_raw(input >= 0.0 ? input : 0.2 * input);
}

using RawMatrix = std::vector<std::vector<std::int64_t>>;

std::int64_t checked_raw_value(__int128 value, std::string_view label) {
    if (value > static_cast<__int128>(std::numeric_limits<std::int64_t>::max())
        || value < static_cast<__int128>(std::numeric_limits<std::int64_t>::min())) {
        throw std::runtime_error("fixed-point raw value overflow while computing " + std::string(label));
    }
    return static_cast<std::int64_t>(value);
}

std::int64_t rescale_fixed_product_sum(__int128 value, std::string_view label) {
    constexpr __int128 kScale = 16;
    const auto rounded = value >= 0 ? value + kScale / 2 : value - kScale / 2;
    return checked_raw_value(rounded / kScale, label);
}

std::vector<std::int64_t> quantize_vector_raw(const std::vector<double>& values) {
    std::vector<std::int64_t> out(values.size(), 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = quantize_float_raw(values[i]);
    }
    return out;
}

RawMatrix quantize_matrix_raw(const gatzk::model::FloatMatrix& matrix) {
    RawMatrix out(matrix.size());
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        out[row] = quantize_vector_raw(matrix[row]);
    }
    return out;
}

RawMatrix project_raw_matrix(
    const RawMatrix& left,
    const RawMatrix& right,
    std::string_view label,
    bool rescale_products) {
    if (left.empty() || right.empty()) {
        return {};
    }
    const auto shared = left.front().size();
    if (right.size() != shared) {
        throw std::runtime_error("fixed-point matrix multiply dimension mismatch: " + std::string(label));
    }
    const auto rows = left.size();
    const auto cols = right.front().size();
    RawMatrix out(rows, std::vector<std::int64_t>(cols, 0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            __int128 sum = 0;
            for (std::size_t k = 0; k < shared; ++k) {
                sum += static_cast<__int128>(left[row][k]) * static_cast<__int128>(right[k][col]);
            }
            out[row][col] = rescale_products
                ? rescale_fixed_product_sum(sum, label)
                : checked_raw_value(sum, label);
        }
    }
    return out;
}

std::vector<std::int64_t> raw_matvec_no_bias(
    const RawMatrix& matrix,
    const std::vector<std::int64_t>& vector,
    std::string_view label,
    bool rescale_products) {
    std::vector<std::int64_t> out(matrix.size(), 0);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (matrix[row].size() != vector.size()) {
            throw std::runtime_error("fixed-point matvec dimension mismatch: " + std::string(label));
        }
        __int128 sum = 0;
        for (std::size_t col = 0; col < vector.size(); ++col) {
            sum += static_cast<__int128>(matrix[row][col]) * static_cast<__int128>(vector[col]);
        }
        out[row] = rescale_products
            ? rescale_fixed_product_sum(sum, label)
            : checked_raw_value(sum, label);
    }
    return out;
}

RawMatrix concat_raw_columns(const std::vector<RawMatrix>& matrices, std::string_view label) {
    if (matrices.empty()) {
        return {};
    }
    const auto rows = matrices.front().size();
    std::size_t cols = 0;
    for (const auto& matrix : matrices) {
        if (matrix.size() != rows) {
            throw std::runtime_error("fixed-point concat row mismatch: " + std::string(label));
        }
        cols += rows == 0 ? 0 : matrix.front().size();
    }
    RawMatrix out(rows, std::vector<std::int64_t>(cols, 0));
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t offset = 0;
        for (const auto& matrix : matrices) {
            std::copy(matrix[row].begin(), matrix[row].end(), out[row].begin() + static_cast<std::ptrdiff_t>(offset));
            offset += matrix[row].size();
        }
    }
    return out;
}

struct RangeStats {
    std::int64_t min_s = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_s = std::numeric_limits<std::int64_t>::min();
    std::int64_t min_z = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_z = std::numeric_limits<std::int64_t>::min();
    std::size_t edge_count = 0;
    std::size_t unique_s = 0;
};

RangeStats scan_raw_attention(
    const std::vector<std::int64_t>& e_src_raw,
    const std::vector<std::int64_t>& e_dst_raw,
    const std::vector<gatzk::data::Edge>& edges) {
    RangeStats stats;
    stats.edge_count = edges.size();
    std::unordered_set<std::int64_t> unique;
    unique.reserve(std::min<std::size_t>(edges.size(), 1ULL << 22));
    for (const auto& edge : edges) {
        if (edge.src >= e_src_raw.size() || edge.dst >= e_dst_raw.size()) {
            throw std::runtime_error("edge endpoint outside attention vector");
        }
        const auto s = e_src_raw[edge.src] + e_dst_raw[edge.dst];
        const auto z = quantize_lrelu_raw(s);
        stats.min_s = std::min(stats.min_s, s);
        stats.max_s = std::max(stats.max_s, s);
        stats.min_z = std::min(stats.min_z, z);
        stats.max_z = std::max(stats.max_z, z);
        unique.insert(s);
    }
    stats.unique_s = unique.size();
    return stats;
}

void print_stats(const std::string& label, std::size_t index, const RangeStats& stats) {
    const auto span = stats.max_s - stats.min_s + 1;
    std::cout
        << label << ','
        << index << ','
        << stats.edge_count << ','
        << stats.unique_s << ','
        << stats.min_s << ','
        << stats.max_s << ','
        << span << ','
        << stats.min_z << ','
        << stats.max_z << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) {
            std::cerr << "usage: gatzk_lrelu_range_diag <config> [--rescale-products]\n";
            return 2;
        }
        const bool rescale_products = argc == 3 && std::string(argv[2]) == "--rescale-products";
        if (argc == 3 && !rescale_products) {
            throw std::runtime_error("unsupported lrelu_range_diag flag: " + std::string(argv[2]));
        }
        const auto config = gatzk::util::load_config(argv[1]);
        const auto dataset = gatzk::data::load_dataset(config);
        const auto local = gatzk::data::normalize_graph_input(dataset, config);
        const auto checkpoint_root = std::filesystem::path(config.project_root) / config.checkpoint_bundle;
        const auto model = gatzk::model::load_checkpoint_bundle_parameters(checkpoint_root.string());
        gatzk::model::ForwardProfile profile;
        const auto forward = gatzk::model::forward_note_style(local.features_fp, local.edges, model, &profile);

        std::cout
            << "label,index,edge_count,unique_s,min_s_raw,max_s_raw,span_s_raw,min_z_raw,max_z_raw\n";
        std::size_t global_hidden_head_index = 0;
        std::vector<RawMatrix> hidden_layer_concat_raw;
        hidden_layer_concat_raw.reserve(forward.hidden_layer_traces.size());
        for (std::size_t layer_index = 0; layer_index < forward.hidden_layer_traces.size(); ++layer_index) {
            const auto& layer_trace = forward.hidden_layer_traces[layer_index];
            const RawMatrix input_raw =
                layer_index == 0 ? quantize_matrix_raw(layer_trace.input) : hidden_layer_concat_raw.back();
            std::vector<RawMatrix> layer_head_outputs_raw;
            layer_head_outputs_raw.reserve(layer_trace.head_traces.size());
            for (const auto& head_trace : layer_trace.head_traces) {
                const auto& head = model.hidden_heads.at(global_hidden_head_index);
                const auto h_prime_raw = project_raw_matrix(
                    input_raw,
                    quantize_matrix_raw(head.seq_kernel_fp),
                    "hidden head projection",
                    rescale_products);
                const auto e_src_raw = raw_matvec_no_bias(
                    h_prime_raw,
                    quantize_vector_raw(head.attn_src_kernel_fp),
                    "hidden source attention",
                    rescale_products);
                const auto e_dst_raw = raw_matvec_no_bias(
                    h_prime_raw,
                    quantize_vector_raw(head.attn_dst_kernel_fp),
                    "hidden destination attention",
                    rescale_products);
                print_stats("hidden", global_hidden_head_index, scan_raw_attention(e_src_raw, e_dst_raw, local.edges));
                layer_head_outputs_raw.push_back(quantize_matrix_raw(head_trace.H_agg));
                ++global_hidden_head_index;
            }
            hidden_layer_concat_raw.push_back(concat_raw_columns(layer_head_outputs_raw, "hidden layer concat"));
        }

        const RawMatrix output_input_raw =
            hidden_layer_concat_raw.empty() ? RawMatrix{} : hidden_layer_concat_raw.back();
        for (std::size_t i = 0; i < forward.output_head_traces.size(); ++i) {
            const auto& head = model.output_layer.heads.at(i);
            const auto y_prime_raw = project_raw_matrix(
                output_input_raw,
                quantize_matrix_raw(head.seq_kernel_fp),
                "output head projection",
                rescale_products);
            const auto e_src_raw = raw_matvec_no_bias(
                y_prime_raw,
                quantize_vector_raw(head.attn_src_kernel_fp),
                "output source attention",
                rescale_products);
            const auto e_dst_raw = raw_matvec_no_bias(
                y_prime_raw,
                quantize_vector_raw(head.attn_dst_kernel_fp),
                "output destination attention",
                rescale_products);
            print_stats("output", i, scan_raw_attention(e_src_raw, e_dst_raw, local.edges));
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "lrelu_range_diag: " << ex.what() << '\n';
        return 1;
    }
}
