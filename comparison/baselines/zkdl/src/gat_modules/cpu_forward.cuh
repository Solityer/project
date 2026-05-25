#pragma once

// CPU-only GAT trace construction used by the CPU formal backend.

// CPU 路径运行单个 attention head；输出完整 witness trace，不写 official artifact。
HeadTraceHost run_attention_head_cpu(
    const std::vector<double>& features,
    std::size_t rows,
    std::size_t input_dim,
    const HeadParams& head,
    const LocalGraph& graph,
    bool apply_bias,
    bool apply_elu)
{
    if (features.size() != rows * input_dim) {
        throw std::runtime_error("CPU attention head input shape mismatch");
    }
    if (head.W.size() != input_dim * head.output_dim
        || head.attn_src.size() != head.output_dim
        || head.attn_dst.size() != head.output_dim) {
        throw std::runtime_error("CPU attention head parameter shape mismatch");
    }
    const std::size_t width = head.output_dim;
    HeadTraceHost trace;
    trace.rows = rows;
    trace.width = width;
    trace.h_prime.assign(rows * width, 0.0);
    trace.e_src.assign(rows, 0.0);
    trace.e_dst.assign(rows, 0.0);
    trace.z.assign(graph.edges.size(), 0.0);
    trace.m.assign(rows, -std::numeric_limits<double>::infinity());
    trace.u.assign(graph.edges.size(), 0.0);
    trace.sum.assign(rows, 0.0);
    trace.h_pre.assign(rows * width, 0.0);
    trace.h_out.assign(rows * width, 0.0);

    trace.h_prime = zkdl_fixed_point_projection_scale16(features, rows, input_dim, head.W, width);
    const auto fixed_attention = zkdl_fixed_point_attention_trace_scale16(
        trace.h_prime, rows, width, head, graph);
    trace.e_src = fixed_attention.e_src;
    trace.e_dst = fixed_attention.e_dst;
    trace.z = fixed_attention.z;
    trace.m = fixed_attention.m;
    trace.u = fixed_attention.u;
    trace.sum = fixed_attention.sum;
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto src = graph.edges[edge].src;
        const auto dst = graph.edges[edge].dst;
        if (trace.sum[dst] == 0.0) {
            throw std::runtime_error("CPU attention denominator is zero");
        }
        const double alpha = trace.u[edge] / trace.sum[dst];
        for (std::size_t col = 0; col < width; ++col) {
            trace.h_pre[dst * width + col] += alpha * trace.h_prime[src * width + col];
        }
    }
    trace.h_out = zkdl_fixed_point_activation_scale16(
        trace.h_pre, rows, width, head.output_bias, apply_bias, apply_elu);
    return trace;
}

// CPU official forward：逐层 hidden 拼接、逐输出头平均，输出 full-chain trace。
ForwardTraceHost run_gat_forward_cpu_trace(const LocalGraph& graph, const ModelParams& model)
{
    ForwardTraceHost trace;
    trace.hidden_layer_inputs.clear();
    trace.hidden_layer_input_dims.clear();
    trace.hidden_heads.clear();
    std::vector<double> current = graph.features;
    std::size_t current_dim = graph.num_features;
    const std::size_t rows = graph.num_nodes;

    for (std::size_t layer = 0; layer < model.hidden_profile.size(); ++layer) {
        trace.hidden_layer_inputs.push_back(current);
        trace.hidden_layer_input_dims.push_back(current_dim);
        trace.hidden_heads.emplace_back();
        std::vector<const HeadParams*> layer_heads;
        for (const auto& head : model.hidden_heads) {
            if (head.layer_index == layer) {
                layer_heads.push_back(&head);
            }
        }
        if (layer_heads.empty()) {
            throw std::runtime_error("CPU forward missing hidden heads for layer " + std::to_string(layer));
        }
        const std::size_t head_width = layer_heads.front()->output_dim;
        std::vector<double> concat(rows * layer_heads.size() * head_width, 0.0);
        for (std::size_t head_index = 0; head_index < layer_heads.size(); ++head_index) {
            auto head_trace = run_attention_head_cpu(
                current, rows, current_dim, *layer_heads[head_index], graph, false, true);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < head_width; ++col) {
                    concat[row * layer_heads.size() * head_width + head_index * head_width + col] =
                        head_trace.h_out[row * head_width + col];
                }
            }
            trace.hidden_heads.back().push_back(std::move(head_trace));
        }
        current = std::move(concat);
        current_dim = layer_heads.size() * head_width;
    }

    trace.output_layer_input = current;
    trace.output_layer_input_dim = current_dim;
    trace.output_heads.clear();
    trace.logits.assign(rows * model.C, 0.0);
    if (model.output_heads.empty()) {
        throw std::runtime_error("CPU forward missing output heads");
    }
    const double average_scale = 1.0 / static_cast<double>(model.output_heads.size());
    for (const auto& head : model.output_heads) {
        auto head_trace = run_attention_head_cpu(current, rows, current_dim, head, graph, true, false);
        if (head_trace.h_out.size() != rows * model.C) {
            throw std::runtime_error("CPU output head shape mismatch");
        }
        for (std::size_t i = 0; i < trace.logits.size(); ++i) {
            trace.logits[i] += head_trace.h_out[i] * average_scale;
        }
        trace.output_heads.push_back(std::move(head_trace));
    }
    return trace;
}
