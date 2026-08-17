#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl 
{

    using FloatMatrix = std::vector<std::vector<double>>;

    struct GatSemanticSummary 
    {
        bool implemented_in_cpp = true;
        std::size_t hidden_attention_head_count = 0;
        std::size_t output_attention_head_count = 0;
        std::size_t total_attention_head_count = 0;
        std::size_t dense_attention_elements_per_head = 0;
        std::size_t dense_attention_elements_all_heads = 0;
        std::string hidden_activation = "elu";
        std::string attention_activation = "leaky_relu(alpha=0.2)";
        std::string output_activation = "identity_after_output_bias";
        std::string output_average_rule = "per_head_bias_then_arithmetic_mean";
    };

    struct HeadParameters {
        FloatMatrix seq_kernel;
        std::vector<double> attn_src_kernel;
        std::vector<double> attn_dst_kernel;
        double attn_src_bias = 0.0;
        double attn_dst_bias = 0.0;
        std::vector<double> output_bias;
    };

    // 注意力头前向传播的完整中间 trace
    struct HeadForwardTrace 
    {
        FloatMatrix h_prime;
        std::vector<double> e_src;
        std::vector<double> e_dst;
        std::vector<double> s;
        std::vector<double> z;
        std::vector<double> m;
        std::vector<double> delta;
        std::vector<double> u;
        std::vector<double> sum;
        std::vector<double> inv;
        std::vector<double> alpha;
        FloatMatrix h_agg;
    };

    GatSemanticSummary summarize_gat_semantics(
        const NormalizedGraph& graph,
        const CheckpointBundle& bundle);

    double leaky_relu(double value);
    double elu(double value);

    // 单个注意力头的前向传播，返回完整中间变量 trace
    HeadForwardTrace attention_head_forward(
        const FloatMatrix& features,
        const std::vector<Edge>& edges,
        const HeadParameters& parameters,
        bool apply_output_bias,
        bool apply_hidden_elu);
}