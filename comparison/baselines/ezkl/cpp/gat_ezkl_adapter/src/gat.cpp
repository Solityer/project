#include "gat_ezkl/gat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gat_ezkl {
namespace {

FloatMatrix matmul(const FloatMatrix& left, const FloatMatrix& right) {
    if (left.empty() || right.empty()) {
        return {};
    }
    const auto shared = left.front().size();
    if (right.size() != shared) {
        throw std::runtime_error("GAT matmul dimension mismatch");
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

}  // namespace

GatSemanticSummary summarize_gat_semantics(
    const NormalizedGraph& graph,
    const CheckpointBundle& bundle) {
    GatSemanticSummary out;
    out.hidden_attention_head_count = bundle.hidden_heads.size();
    out.output_attention_head_count = bundle.output_heads.size();
    out.total_attention_head_count = out.hidden_attention_head_count + out.output_attention_head_count;
    out.dense_attention_elements_per_head = graph.node_count * graph.node_count;
    out.dense_attention_elements_all_heads = out.dense_attention_elements_per_head * out.total_attention_head_count;
    out.output_average_rule = bundle.output_average_rule;
    return out;
}

double leaky_relu(double value) {
    return value >= 0.0 ? value : 0.2 * value;
}

double elu(double value) {
    return value >= 0.0 ? value : std::expm1(value);
}

HeadForwardTrace attention_head_forward(
    const FloatMatrix& features,
    const std::vector<Edge>& edges,
    const HeadParameters& parameters,
    bool apply_output_bias,
    bool apply_hidden_elu) {
    if (features.empty()) {
        return {};
    }
    const auto node_count = features.size();
    const auto width = parameters.seq_kernel.empty() ? 0 : parameters.seq_kernel.front().size();

    HeadForwardTrace trace;
    trace.h_prime = matmul(features, parameters.seq_kernel);
    trace.e_src = matvec(trace.h_prime, parameters.attn_src_kernel, parameters.attn_src_bias);
    trace.e_dst = matvec(trace.h_prime, parameters.attn_dst_kernel, parameters.attn_dst_bias);
    trace.alpha.assign(edges.size(), 0.0);
    trace.h_agg.assign(node_count, std::vector<double>(width, 0.0));

    std::vector<std::vector<std::size_t>> incoming_edge_indices(node_count);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto& edge = edges[edge_index];
        if (edge.src >= node_count || edge.dst >= node_count) {
            throw std::runtime_error("edge endpoint out of bounds in GAT forward");
        }
        incoming_edge_indices[edge.dst].push_back(edge_index);
    }

    for (std::size_t dst = 0; dst < node_count; ++dst) {
        if (incoming_edge_indices[dst].empty()) {
            throw std::runtime_error("GAT grouped softmax requires at least one incoming edge per node");
        }
        double max_score = -std::numeric_limits<double>::infinity();
        for (const auto edge_index : incoming_edge_indices[dst]) {
            const auto& edge = edges[edge_index];
            max_score = std::max(max_score, leaky_relu(trace.e_src[edge.src] + trace.e_dst[edge.dst]));
        }
        double denom = 0.0;
        for (const auto edge_index : incoming_edge_indices[dst]) {
            const auto& edge = edges[edge_index];
            denom += std::exp(leaky_relu(trace.e_src[edge.src] + trace.e_dst[edge.dst]) - max_score);
        }
        for (const auto edge_index : incoming_edge_indices[dst]) {
            const auto& edge = edges[edge_index];
            const auto alpha = std::exp(leaky_relu(trace.e_src[edge.src] + trace.e_dst[edge.dst]) - max_score) / denom;
            trace.alpha[edge_index] = alpha;
            for (std::size_t col = 0; col < width; ++col) {
                trace.h_agg[dst][col] += alpha * trace.h_prime[edge.src][col];
            }
        }
    }

    for (auto& row : trace.h_agg) {
        for (std::size_t col = 0; col < row.size(); ++col) {
            if (apply_output_bias && col < parameters.output_bias.size()) {
                row[col] += parameters.output_bias[col];
            }
            if (apply_hidden_elu) {
                row[col] = elu(row[col]);
            }
        }
    }
    return trace;
}

}  // namespace gat_ezkl
