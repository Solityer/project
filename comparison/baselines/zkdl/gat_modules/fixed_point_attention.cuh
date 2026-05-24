#pragma once

// GAT attention score 的固定点语义：H'、attention 向量与 bias 均量化到 scale=16。
// node dot 先做整数乘加，再按 half-away-from-zero 除以 scale；LeakyReLU 在 scale16 score 上执行。

struct ZkdlFixedPointAttentionTrace {
    std::vector<double> e_src;
    std::vector<double> e_dst;
    std::vector<double> z;
    std::vector<double> m;
    std::vector<double> u;
    std::vector<double> sum;
};

std::vector<std::int64_t> zkdl_quantize_scale16_vec_i64(const std::vector<double>& values)
{
    std::vector<std::int64_t> out(values.size(), 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = zkdl_quantize_scale16_i64(values[i]);
    }
    return out;
}

std::int64_t zkdl_fixed_attention_dot_raw(
    const std::vector<std::int64_t>& h_raw,
    const std::vector<std::int64_t>& attn_raw,
    std::size_t row,
    std::size_t width)
{
    __int128 acc = 0;
    for (std::size_t col = 0; col < width; ++col) {
        acc += static_cast<__int128>(h_raw[row * width + col])
            * static_cast<__int128>(attn_raw[col]);
    }
    const auto pre = zkdl_checked_i128_to_i64(acc, "fixed attention dot pre-rescale product");
    return zkdl_round_half_away_div_i64(pre, 16);
}

ZkdlFixedPointAttentionTrace zkdl_fixed_point_attention_trace_scale16(
    const std::vector<double>& h_prime,
    std::size_t rows,
    std::size_t width,
    const HeadParams& head,
    const LocalGraph& graph)
{
    if (h_prime.size() != rows * width
        || head.attn_src.size() != width
        || head.attn_dst.size() != width) {
        throw std::runtime_error("fixed attention trace 输入形状不一致");
    }
    ZkdlFixedPointAttentionTrace trace;
    trace.e_src.assign(rows, 0.0);
    trace.e_dst.assign(rows, 0.0);
    trace.z.assign(graph.edges.size(), 0.0);
    trace.m.assign(rows, -std::numeric_limits<double>::infinity());
    trace.u.assign(graph.edges.size(), 0.0);
    trace.sum.assign(rows, 0.0);

    const auto h_raw = zkdl_quantize_scale16_vec_i64(h_prime);
    const auto attn_src_raw = zkdl_quantize_scale16_vec_i64(head.attn_src);
    const auto attn_dst_raw = zkdl_quantize_scale16_vec_i64(head.attn_dst);
    const auto bias_src_raw = zkdl_quantize_scale16_i64(head.attn_src_bias);
    const auto bias_dst_raw = zkdl_quantize_scale16_i64(head.attn_dst_bias);
    std::vector<std::int64_t> e_src_raw(rows, 0);
    std::vector<std::int64_t> e_dst_raw(rows, 0);

    const std::size_t node_task_count = rows;
    const std::size_t cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(cpu_count, node_task_count == 0 ? 1 : node_task_count);
    auto node_worker = [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            const auto src_raw = zkdl_fixed_attention_dot_raw(h_raw, attn_src_raw, row, width) + bias_src_raw;
            const auto dst_raw = zkdl_fixed_attention_dot_raw(h_raw, attn_dst_raw, row, width) + bias_dst_raw;
            e_src_raw[row] = src_raw;
            e_dst_raw[row] = dst_raw;
            trace.e_src[row] = static_cast<double>(src_raw) / 16.0;
            trace.e_dst[row] = static_cast<double>(dst_raw) / 16.0;
        }
    };
    if (worker_count <= 1 || node_task_count < 4096) {
        node_worker(0, node_task_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        const std::size_t chunk = (node_task_count + worker_count - 1) / worker_count;
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            const std::size_t begin = worker_index * chunk;
            const std::size_t end = std::min(node_task_count, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back(node_worker, begin, end);
        }
        for (auto& thread : workers) thread.join();
    }

    std::vector<std::int64_t> z_raw(graph.edges.size(), 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto src = graph.edges[edge].src;
        const auto dst = graph.edges[edge].dst;
        if (src >= rows || dst >= rows) {
            throw std::runtime_error("fixed attention edge index out of range");
        }
        const auto score_raw = zkdl_checked_i128_to_i64(
            static_cast<__int128>(e_src_raw[src]) + static_cast<__int128>(e_dst_raw[dst]),
            "fixed attention score");
        const auto lrelu_raw = score_raw >= 0 ? score_raw : zkdl_round_half_away_div_i64(score_raw, 5);
        z_raw[edge] = lrelu_raw;
        trace.z[edge] = static_cast<double>(lrelu_raw) / 16.0;
        trace.m[dst] = std::max(trace.m[dst], trace.z[edge]);
    }
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto dst = graph.edges[edge].dst;
        const double u = std::exp(trace.z[edge] - trace.m[dst]);
        trace.u[edge] = u;
        trace.sum[dst] += u;
    }
    return trace;
}

std::vector<double> zkdl_fixed_point_activation_scale16(
    const std::vector<double>& pre,
    std::size_t rows,
    std::size_t width,
    const std::vector<double>& bias,
    bool apply_bias,
    bool apply_elu)
{
    if (pre.size() != rows * width) {
        throw std::runtime_error("fixed activation 输入形状不一致");
    }
    if (apply_bias && bias.size() != width) {
        throw std::runtime_error("fixed activation bias 形状不一致");
    }
    std::vector<std::int64_t> bias_raw(width, 0);
    if (apply_bias) {
        for (std::size_t col = 0; col < width; ++col) {
            bias_raw[col] = zkdl_quantize_scale16_i64(bias[col]);
        }
    }
    std::vector<double> out(pre.size(), 0.0);
    const std::size_t task_count = pre.size();
    const std::size_t cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(cpu_count, task_count == 0 ? 1 : task_count);
    auto worker = [&](std::size_t begin, std::size_t end) {
        for (std::size_t idx = begin; idx < end; ++idx) {
            const std::size_t col = idx % width;
            std::int64_t raw = zkdl_quantize_scale16_i64(pre[idx]);
            if (apply_bias) raw += bias_raw[col];
            if (apply_elu && raw < 0) {
                const double shifted = static_cast<double>(raw) / 16.0;
                raw = zkdl_quantize_scale16_i64(std::exp(shifted) - 1.0);
            }
            out[idx] = static_cast<double>(raw) / 16.0;
        }
    };
    if (worker_count <= 1 || task_count < 4096) {
        worker(0, task_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        const std::size_t chunk = (task_count + worker_count - 1) / worker_count;
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            const std::size_t begin = worker_index * chunk;
            const std::size_t end = std::min(task_count, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back(worker, begin, end);
        }
        for (auto& thread : workers) thread.join();
    }
    return out;
}
