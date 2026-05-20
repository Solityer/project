#pragma once

// CUDA forward trace 构造与 GPU kernel 执行路径。

__global__ void fill_double_kernel(double* out, double value, std::size_t n) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = value;
    }
}

__global__ void dense_matmul_kernel(const double* A, const double* B, double* C, int rows, int shared, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    double acc = 0.0;
    for (int k = 0; k < shared; ++k) {
        acc += A[static_cast<std::size_t>(row) * shared + k] * B[static_cast<std::size_t>(k) * cols + col];
    }
    C[static_cast<std::size_t>(row) * cols + col] = acc;
}

__global__ void node_attention_kernel(const double* H, const double* a_src, const double* a_dst,
                                      double b_src, double b_dst, double* E_src, double* E_dst,
                                      int rows, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    double src = b_src;
    double dst = b_dst;
    for (int j = 0; j < width; ++j) {
        const double value = H[static_cast<std::size_t>(row) * width + j];
        src += value * a_src[j];
        dst += value * a_dst[j];
    }
    E_src[row] = src;
    E_dst[row] = dst;
}

__device__ double atomic_max_double(double* address, double value) {
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    while (true) {
        assumed = old;
        if (__longlong_as_double(static_cast<long long>(assumed)) >= value) {
            break;
        }
        old = atomicCAS(address_as_ull, assumed, static_cast<unsigned long long int>(__double_as_longlong(value)));
        if (old == assumed) {
            break;
        }
    }
    return __longlong_as_double(static_cast<long long>(old));
}

__global__ void edge_score_kernel(const std::uint32_t* src, const std::uint32_t* dst,
                                  const double* E_src, const double* E_dst,
                                  double* Z, double* M, std::size_t edge_count) {
    const std::size_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) {
        return;
    }
    const double s = E_dst[dst[eid]] + E_src[src[eid]];
    const double z = s >= 0.0 ? s : 0.2 * s;
    Z[eid] = z;
    atomic_max_double(&M[dst[eid]], z);
}

__global__ void softmax_sum_kernel(const std::uint32_t* dst, const double* Z, const double* M,
                                   double* U, double* Sum, std::size_t edge_count) {
    const std::size_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) {
        return;
    }
    const double u = exp(Z[eid] - M[dst[eid]]);
    U[eid] = u;
    atomicAdd(&Sum[dst[eid]], u);
}

__global__ void aggregate_kernel(const std::uint32_t* src, const std::uint32_t* dst,
                                 const double* U, const double* Sum, const double* H_prime,
                                 double* H_agg_pre, int width, std::size_t edge_count) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = edge_count * static_cast<std::size_t>(width);
    if (gid >= total) {
        return;
    }
    const std::size_t eid = gid / width;
    const int col = static_cast<int>(gid % width);
    const double alpha = U[eid] / Sum[dst[eid]];
    const double contrib = alpha * H_prime[static_cast<std::size_t>(src[eid]) * width + col];
    atomicAdd(&H_agg_pre[static_cast<std::size_t>(dst[eid]) * width + col], contrib);
}

__global__ void activation_kernel(const double* pre, const double* bias, double* out,
                                  int rows, int width, bool apply_bias, bool apply_elu) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * width;
    if (gid >= total) {
        return;
    }
    const int col = static_cast<int>(gid % width);
    double value = pre[gid];
    if (apply_bias) {
        value += bias[col];
    }
    if (apply_elu && value < 0.0) {
        value = exp(value) - 1.0;
    }
    out[gid] = value;
}

__global__ void copy_head_to_concat_kernel(const double* head, double* concat,
                                           int rows, int head_width, int concat_width, int offset) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * head_width;
    if (gid >= total) {
        return;
    }
    const int row = static_cast<int>(gid / head_width);
    const int col = static_cast<int>(gid % head_width);
    concat[static_cast<std::size_t>(row) * concat_width + offset + col] = head[gid];
}

__global__ void add_average_kernel(const double* head, double* logits, double scale, std::size_t n) {
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        logits[gid] += head[gid] * scale;
    }
}

DeviceBuffer<double> run_attention_head(
    const DeviceBuffer<double>& features,
    int rows,
    int input_dim,
    const HeadParams& head,
    const DeviceBuffer<std::uint32_t>& src,
    const DeviceBuffer<std::uint32_t>& dst,
    std::size_t edge_count,
    bool apply_bias,
    bool apply_elu,
    double* projection_ms,
    double* attention_ms,
    double* activation_ms,
    HeadTraceHost* trace) {
    const int width = static_cast<int>(head.output_dim);
    auto d_W = to_device(head.W);
    auto d_a_src = to_device(head.attn_src);
    auto d_a_dst = to_device(head.attn_dst);
    DeviceBuffer<double> H_prime(static_cast<std::size_t>(rows) * width);
    {
        ScopedTimer timer;
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        dense_matmul_kernel<<<grid, block>>>(features.ptr, d_W.ptr, H_prime.ptr, rows, input_dim, width);
        cuda_sync("dense_matmul_kernel");
        *projection_ms += timer.elapsed_ms();
    }

    DeviceBuffer<double> E_src(rows);
    DeviceBuffer<double> E_dst(rows);
    DeviceBuffer<double> Z(edge_count);
    DeviceBuffer<double> M(rows);
    DeviceBuffer<double> U(edge_count);
    DeviceBuffer<double> Sum(rows);
    DeviceBuffer<double> H_pre(static_cast<std::size_t>(rows) * width);
    {
        ScopedTimer timer;
        node_attention_kernel<<<(rows + 255) / 256, 256>>>(
            H_prime.ptr, d_a_src.ptr, d_a_dst.ptr, head.attn_src_bias, head.attn_dst_bias,
            E_src.ptr, E_dst.ptr, rows, width);
        cuda_sync("node_attention_kernel");
        fill_double_kernel<<<(rows + 255) / 256, 256>>>(M.ptr, -std::numeric_limits<double>::infinity(), rows);
        fill_double_kernel<<<(rows + 255) / 256, 256>>>(Sum.ptr, 0.0, rows);
        fill_double_kernel<<<(static_cast<std::size_t>(rows) * width + 255) / 256, 256>>>(H_pre.ptr, 0.0, static_cast<std::size_t>(rows) * width);
        cuda_sync("attention_init");
        edge_score_kernel<<<(edge_count + 255) / 256, 256>>>(src.ptr, dst.ptr, E_src.ptr, E_dst.ptr, Z.ptr, M.ptr, edge_count);
        cuda_sync("edge_score_kernel");
        softmax_sum_kernel<<<(edge_count + 255) / 256, 256>>>(dst.ptr, Z.ptr, M.ptr, U.ptr, Sum.ptr, edge_count);
        cuda_sync("softmax_sum_kernel");
        aggregate_kernel<<<(edge_count * static_cast<std::size_t>(width) + 255) / 256, 256>>>(
            src.ptr, dst.ptr, U.ptr, Sum.ptr, H_prime.ptr, H_pre.ptr, width, edge_count);
        cuda_sync("aggregate_kernel");
        *attention_ms += timer.elapsed_ms();
    }

    auto d_bias = to_device(head.output_bias);
    DeviceBuffer<double> H_out(static_cast<std::size_t>(rows) * width);
    {
        ScopedTimer timer;
        activation_kernel<<<(static_cast<std::size_t>(rows) * width + 255) / 256, 256>>>(
            H_pre.ptr, d_bias.ptr, H_out.ptr, rows, width, apply_bias, apply_elu);
        cuda_sync("activation_kernel");
        *activation_ms += timer.elapsed_ms();
    }
    if (trace != nullptr) {
        trace->rows = static_cast<std::size_t>(rows);
        trace->width = static_cast<std::size_t>(width);
        trace->h_prime = to_host(H_prime);
        trace->e_src = to_host(E_src);
        trace->e_dst = to_host(E_dst);
        trace->z = to_host(Z);
        trace->m = to_host(M);
        trace->u = to_host(U);
        trace->sum = to_host(Sum);
        trace->h_pre = to_host(H_pre);
        trace->h_out = to_host(H_out);
    }
    return H_out;
}

struct ForwardMetrics {
    double hidden_projection_ms = 0.0;
    double hidden_attention_ms = 0.0;
    double hidden_activation_ms = 0.0;
    double hidden_concat_ms = 0.0;
    double output_projection_ms = 0.0;
    double output_attention_ms = 0.0;
    double output_activation_ms = 0.0;
    double total_ms = 0.0;
};

std::vector<double> run_gat_forward_gpu(const LocalGraph& graph, const ModelParams& model, ForwardMetrics* metrics, ForwardTraceHost* trace) {
    ScopedTimer total_timer;
    auto features = to_device(graph.features);
    int current_dim = static_cast<int>(graph.num_features);
    const int rows = static_cast<int>(graph.num_nodes);

    std::vector<std::uint32_t> src_host(graph.edges.size());
    std::vector<std::uint32_t> dst_host(graph.edges.size());
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        src_host[i] = static_cast<std::uint32_t>(graph.edges[i].src);
        dst_host[i] = static_cast<std::uint32_t>(graph.edges[i].dst);
    }
    const auto d_src = to_device(src_host);
    const auto d_dst = to_device(dst_host);

    DeviceBuffer<double> current = std::move(features);
    if (trace != nullptr) {
        trace->hidden_layer_inputs.clear();
        trace->hidden_layer_input_dims.clear();
        trace->hidden_heads.clear();
    }
    for (std::size_t layer = 0; layer < model.hidden_profile.size(); ++layer) {
        if (trace != nullptr) {
            trace->hidden_layer_inputs.push_back(to_host(current));
            trace->hidden_layer_input_dims.push_back(static_cast<std::size_t>(current_dim));
            trace->hidden_heads.emplace_back();
        }
        std::vector<const HeadParams*> layer_heads;
        for (const auto& head : model.hidden_heads) {
            if (head.layer_index == layer) {
                layer_heads.push_back(&head);
            }
        }
        const int head_width = static_cast<int>(model.hidden_profile[layer].second);
        const int concat_width = static_cast<int>(layer_heads.size()) * head_width;
        DeviceBuffer<double> concat(static_cast<std::size_t>(rows) * concat_width);
        int offset = 0;
        for (const auto* head : layer_heads) {
            HeadTraceHost hidden_head_trace;
            auto out = run_attention_head(
                current, rows, current_dim, *head, d_src, d_dst, graph.edges.size(),
                false, true,
                &metrics->hidden_projection_ms,
                &metrics->hidden_attention_ms,
                &metrics->hidden_activation_ms,
                trace == nullptr ? nullptr : &hidden_head_trace);
            if (trace != nullptr) {
                trace->hidden_heads.back().push_back(std::move(hidden_head_trace));
            }
            ScopedTimer concat_timer;
            copy_head_to_concat_kernel<<<(static_cast<std::size_t>(rows) * head_width + 255) / 256, 256>>>(
                out.ptr, concat.ptr, rows, head_width, concat_width, offset);
            cuda_sync("copy_head_to_concat_kernel");
            metrics->hidden_concat_ms += concat_timer.elapsed_ms();
            offset += head_width;
        }
        current = std::move(concat);
        current_dim = concat_width;
    }

    DeviceBuffer<double> logits(static_cast<std::size_t>(rows) * model.C);
    fill_double_kernel<<<(static_cast<std::size_t>(rows) * model.C + 255) / 256, 256>>>(logits.ptr, 0.0, static_cast<std::size_t>(rows) * model.C);
    cuda_sync("logits_init");
    const double average_scale = 1.0 / static_cast<double>(model.output_heads.size());
    if (trace != nullptr) {
        trace->output_heads.clear();
        trace->output_heads.reserve(model.output_heads.size());
        trace->output_layer_input = to_host(current);
        trace->output_layer_input_dim = static_cast<std::size_t>(current_dim);
    }
    for (const auto& head : model.output_heads) {
        HeadTraceHost head_trace;
        auto out = run_attention_head(
            current, rows, current_dim, head, d_src, d_dst, graph.edges.size(),
            true, false,
            &metrics->output_projection_ms,
            &metrics->output_attention_ms,
            &metrics->output_activation_ms,
            trace == nullptr ? nullptr : &head_trace);
        if (trace != nullptr) {
            trace->output_heads.push_back(std::move(head_trace));
        }
        add_average_kernel<<<(static_cast<std::size_t>(rows) * model.C + 255) / 256, 256>>>(
            out.ptr, logits.ptr, average_scale, static_cast<std::size_t>(rows) * model.C);
        cuda_sync("add_average_kernel");
    }

    metrics->total_ms = total_timer.elapsed_ms();
    auto host_logits = to_host(logits);
    if (trace != nullptr) {
        trace->logits = host_logits;
    }
    return host_logits;
}
