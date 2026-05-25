#pragma once

// 稀疏路由、Hadamard、多列 grouped sum 的 zkDL-native 工具。

std::vector<Fr_t> encode_padded_vector_scale16_raw(
    const std::vector<double>& values,
    std::size_t logical_count,
    std::size_t padded_count)
{
    if (values.size() != logical_count || logical_count > padded_count) {
        throw std::runtime_error("一维向量编码形状不一致");
    }
    std::vector<Fr_t> out(padded_count, fr_from_signed_host(0));
    for (std::size_t i = 0; i < logical_count; ++i) {
        out[i] = fr_from_signed_host(quantize_scale16_to_i64(values[i]));
    }
    return out;
}

std::vector<std::uint32_t> graph_src_u32(const LocalGraph& graph)
{
    std::vector<std::uint32_t> out(graph.edges.size(), 0);
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        if (graph.edges[i].src > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("src index 超出 uint32 范围");
        }
        out[i] = static_cast<std::uint32_t>(graph.edges[i].src);
    }
    return out;
}

std::vector<std::uint32_t> graph_dst_u32(const LocalGraph& graph)
{
    std::vector<std::uint32_t> out(graph.edges.size(), 0);
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        if (graph.edges[i].dst > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("dst index 超出 uint32 范围");
        }
        out[i] = static_cast<std::uint32_t>(graph.edges[i].dst);
    }
    return out;
}

std::vector<Fr_t> route_vector_by_topology_raw(
    const std::vector<Fr_t>& node_raw,
    const LocalGraph& graph,
    std::size_t padded_edge_count,
    bool use_source)
{
    std::vector<Fr_t> out(padded_edge_count, fr_from_signed_host(0));
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const std::size_t node = use_source ? graph.edges[edge].src : graph.edges[edge].dst;
        if (node >= node_raw.size()) throw std::runtime_error("sparse routing node index 超出 witness 范围");
        out[edge] = node_raw[node];
    }
    return out;
}

std::vector<Fr_t> grouped_sum_by_dst_raw(
    const std::vector<Fr_t>& edge_raw,
    const LocalGraph& graph,
    std::size_t padded_node_count)
{
    std::vector<Fr_t> out(padded_node_count, fr_from_signed_host(0));
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const std::size_t dst = graph.edges[edge].dst;
        if (dst >= padded_node_count || edge >= edge_raw.size()) {
            throw std::runtime_error("grouped sum index 超出 witness 范围");
        }
        out[dst] = Fr_host_add(out[dst], edge_raw[edge]);
    }
    return out;
}

std::vector<Fr_t> mont_vector_from_raw(const std::vector<Fr_t>& raw)
{
    std::vector<Fr_t> out(raw.size(), Fr_host_zero());
    for (std::size_t i = 0; i < raw.size(); ++i) out[i] = Fr_host_mont(raw[i]);
    return out;
}

std::vector<Fr_t> encode_nonnegative_scale16_raw(
    const std::vector<double>& values,
    std::size_t logical_count,
    std::size_t padded_count,
    std::vector<std::uint32_t>* quantized_out)
{
    if (values.size() != logical_count || logical_count > padded_count) {
        throw std::runtime_error("非负一维向量编码形状不一致");
    }
    std::vector<Fr_t> out(padded_count, fr_from_signed_host(0));
    if (quantized_out != nullptr) quantized_out->assign(padded_count, 0);
    for (std::size_t i = 0; i < logical_count; ++i) {
        const auto quantized = quantize_scale16_to_i64(values[i]);
        if (quantized < 0 || static_cast<unsigned long long>(quantized) > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("非负 scale16 编码超出 uint32 范围");
        }
        out[i] = fr_from_signed_host(quantized);
        if (quantized_out != nullptr) (*quantized_out)[i] = static_cast<std::uint32_t>(quantized);
    }
    return out;
}

std::vector<std::uint64_t> grouped_sum_by_dst_u64(
    const std::vector<std::uint32_t>& edge_values,
    const LocalGraph& graph,
    std::size_t padded_node_count)
{
    std::vector<std::uint64_t> out(padded_node_count, 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const std::size_t dst = graph.edges[edge].dst;
        if (dst >= padded_node_count || edge >= edge_values.size()) {
            throw std::runtime_error("uint grouped sum index 超出 witness 范围");
        }
        out[dst] += edge_values[edge];
        if (out[dst] > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("denominator 超出当前 small-inverse uint32 范围");
        }
    }
    return out;
}

std::vector<Fr_t> valid_one_mont_vector(std::size_t logical_count, std::size_t padded_count)
{
    if (logical_count > padded_count) throw std::runtime_error("valid one vector shape mismatch");
    std::vector<Fr_t> out(padded_count, Fr_host_zero());
    for (std::size_t i = 0; i < logical_count; ++i) out[i] = Fr_host_one_mont();
    return out;
}

std::vector<Fr_t> column_scale16_raw(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t col,
    std::size_t padded_rows)
{
    if (values.size() != rows * cols || col >= cols || rows > padded_rows) {
        throw std::runtime_error("列向量编码形状不一致");
    }
    std::vector<Fr_t> out(padded_rows, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        out[row] = fr_from_signed_host(quantize_scale16_to_i64(values[row * cols + col]));
    }
    return out;
}

std::vector<Fr_t> matrix_scale16_raw(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (values.size() != rows * cols || rows > padded_rows || cols > padded_cols) {
        throw std::runtime_error("矩阵编码形状不一致");
    }
    std::vector<Fr_t> out(padded_rows * padded_cols, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[row * padded_cols + col] = fr_from_signed_host(
                quantize_scale16_to_i64(values[row * cols + col]));
        }
    }
    return out;
}

std::vector<double> slice_matrix_columns(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t col_begin,
    std::size_t col_count)
{
    if (values.size() != rows * cols || col_begin > cols || col_count > cols - col_begin) {
        throw std::runtime_error("矩阵列切片形状不一致");
    }
    std::vector<double> out(rows * col_count, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t local_col = 0; local_col < col_count; ++local_col) {
            out[row * col_count + local_col] = values[row * cols + col_begin + local_col];
        }
    }
    return out;
}

std::vector<Fr_t> route_matrix_by_topology_raw(
    const std::vector<Fr_t>& node_matrix_raw,
    const LocalGraph& graph,
    std::size_t padded_node_count,
    std::size_t padded_edge_count,
    std::size_t padded_cols,
    bool use_source)
{
    if (node_matrix_raw.size() != padded_node_count * padded_cols) {
        throw std::runtime_error("矩阵 sparse routing 输入形状不一致");
    }
    std::vector<Fr_t> out(padded_edge_count * padded_cols, fr_from_signed_host(0));
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const std::size_t node = use_source ? graph.edges[edge].src : graph.edges[edge].dst;
        if (node >= padded_node_count) throw std::runtime_error("矩阵 sparse routing node index 超出范围");
        for (std::size_t col = 0; col < padded_cols; ++col) {
            out[edge * padded_cols + col] = node_matrix_raw[node * padded_cols + col];
        }
    }
    return out;
}

std::vector<Fr_t> grouped_sum_matrix_by_dst_mont(
    const std::vector<Fr_t>& edge_matrix_mont,
    const LocalGraph& graph,
    std::size_t padded_node_count,
    std::size_t padded_edge_count,
    std::size_t padded_cols)
{
    if (edge_matrix_mont.size() != padded_edge_count * padded_cols) {
        throw std::runtime_error("矩阵 grouped sum 输入形状不一致");
    }
    std::vector<Fr_t> out(padded_node_count * padded_cols, Fr_host_zero());
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const std::size_t dst = graph.edges[edge].dst;
        if (dst >= padded_node_count) throw std::runtime_error("矩阵 grouped sum dst index 超出范围");
        for (std::size_t col = 0; col < padded_cols; ++col) {
            out[dst * padded_cols + col] = Fr_host_add(
                out[dst * padded_cols + col],
                edge_matrix_mont[edge * padded_cols + col]);
        }
    }
    return out;
}

std::vector<Fr_t> weighted_matrix_by_alpha_mont(
    const std::vector<Fr_t>& alpha_mont,
    const std::vector<Fr_t>& message_edge_matrix_mont,
    std::size_t logical_edge_count,
    std::size_t padded_edge_count,
    std::size_t padded_cols)
{
    if (alpha_mont.size() != padded_edge_count
        || message_edge_matrix_mont.size() != padded_edge_count * padded_cols) {
        throw std::runtime_error("alpha 矩阵乘法输入形状不一致");
    }
    std::vector<Fr_t> out(padded_edge_count * padded_cols, Fr_host_zero());
    for (std::size_t edge = 0; edge < logical_edge_count; ++edge) {
        for (std::size_t col = 0; col < padded_cols; ++col) {
            out[edge * padded_cols + col] = Fr_host_mul(
                alpha_mont[edge],
                message_edge_matrix_mont[edge * padded_cols + col]);
        }
    }
    return out;
}

std::size_t floor_power_of_two_size(std::size_t value)
{
    if (value == 0) return 0;
    std::size_t out = 1;
    while (out <= value / 2) out <<= 1U;
    return out;
}

std::size_t formal_effective_matrix_commitment_block_size(
    std::size_t requested_row_block,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (padded_rows == 0 || padded_cols == 0) {
        throw std::runtime_error("matrix commitment block size needs non-empty tensors");
    }
    const std::size_t row_block = std::min(requested_row_block == 0 ? 64 : requested_row_block, padded_rows);
    std::size_t row_block_pow2 = std::max<std::size_t>(1, floor_power_of_two_size(row_block));
    // 矩阵 opening 的 inner proof 复杂度随 row_block * padded_cols 增长。
    // 这里限制总 inner block，避免 PPI/ogbn-arxiv 输出层宽列块把 opening 放大到不可完成。
    const std::size_t max_inner_block = 8192;
    if (row_block_pow2 > max_inner_block / padded_cols) {
        row_block_pow2 = std::max<std::size_t>(1, floor_power_of_two_size(max_inner_block / padded_cols));
    }
    return row_block_pow2 * padded_cols;
}

std::size_t formal_effective_commitment_block_size(
    std::size_t requested,
    std::size_t padded_node_count,
    std::size_t padded_edge_count)
{
    const std::size_t min_size = std::min(padded_node_count, padded_edge_count);
    if (min_size == 0) throw std::runtime_error("commitment block size needs non-empty tensors");
    if (requested == 0) requested = 64;
    const std::size_t capped = std::min(requested, min_size);
    const std::size_t block = floor_power_of_two_size(capped);
    return std::max<std::size_t>(1, block);
}

Commitment make_formal_commitment_generators(std::size_t block_size)
{
    if (block_size == 0 || block_size > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("formal commitment block size 超出 uint 范围");
    }
    if (block_size == 1) return Commitment(1, G1Jacobian_generator);
    auto scalars = deterministic_raw_fr_vec(
        block_size,
        "zkdl-gat-gpu-generators:block:" + std::to_string(block_size));
    std::vector<G1Jacobian_t> points(block_size, G1Jacobian_host_zero());
    for (std::size_t i = 0; i < block_size; ++i) {
        points[i] = G1Jacobian_host_mul(G1Jacobian_generator, scalars[i]);
    }
    return Commitment(static_cast<uint>(block_size), points.data());
}

struct MultiColumnSparseRoutingGadgetProof
{
    std::string label;
    std::string direction;
    std::uint64_t topology_hash = 0;
    uint node_count = 0;
    uint edge_count = 0;
    uint column_count = 0;
    uint padded_node_count = 0;
    uint padded_edge_count = 0;
    uint padded_column_count = 0;
    std::vector<Fr_t> column_point;
    std::vector<G1Jacobian_t> compressed_node_commitment_points;
    std::vector<G1Jacobian_t> compressed_edge_commitment_points;
    SparseRoutingRelationProof compressed_routing;
    Fr_t node_matrix_eval = Fr_host_zero();
    Fr_t edge_matrix_eval = Fr_host_zero();
    CommitmentOpeningProof opening_node_matrix;
    CommitmentOpeningProof opening_edge_matrix;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t) + label.size()
            + sizeof(uint32_t) + direction.size()
            + sizeof(std::uint64_t)
            + 6 * sizeof(uint32_t)
            + sizeof(uint32_t) + column_point.size() * sizeof(Fr_t)
            + sizeof(uint32_t) + compressed_node_commitment_points.size() * sizeof(G1Jacobian_t)
            + sizeof(uint32_t) + compressed_edge_commitment_points.size() * sizeof(G1Jacobian_t)
            + compressed_routing.serialized_size_bytes()
            + 2 * sizeof(Fr_t)
            + opening_node_matrix.serialized_size_bytes()
            + opening_edge_matrix.serialized_size_bytes()
            + sizeof(uint64_t);
    }
};

struct MultiColumnHadamardGadgetProof
{
    std::string label;
    uint edge_count = 0;
    uint column_count = 0;
    uint padded_edge_count = 0;
    uint padded_column_count = 0;
    std::vector<Fr_t> column_point;
    std::vector<G1Jacobian_t> compressed_message_commitment_points;
    std::vector<G1Jacobian_t> compressed_weighted_commitment_points;
    HadamardSumcheckProof compressed_hadamard;
    Fr_t message_matrix_eval = Fr_host_zero();
    Fr_t weighted_matrix_eval = Fr_host_zero();
    CommitmentOpeningProof opening_message_matrix;
    CommitmentOpeningProof opening_weighted_matrix;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t) + label.size()
            + 4 * sizeof(uint32_t)
            + sizeof(uint32_t) + column_point.size() * sizeof(Fr_t)
            + sizeof(uint32_t) + compressed_message_commitment_points.size() * sizeof(G1Jacobian_t)
            + sizeof(uint32_t) + compressed_weighted_commitment_points.size() * sizeof(G1Jacobian_t)
            + compressed_hadamard.serialized_size_bytes()
            + 2 * sizeof(Fr_t)
            + opening_message_matrix.serialized_size_bytes()
            + opening_weighted_matrix.serialized_size_bytes()
            + sizeof(uint64_t);
    }
};

void absorb_multi_column_sparse_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    const std::string& direction,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint padded_column_count,
    const G1TensorJacobian& com_node_matrix,
    const G1TensorJacobian& com_edge_matrix)
{
    transcript.absorb_label("zkdl_gat_multi_column_sparse_routing");
    transcript.absorb_label(label);
    transcript.absorb_label(direction);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(column_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(padded_column_count);
    formal_absorb_g1_tensor(transcript, com_node_matrix);
    formal_absorb_g1_tensor(transcript, com_edge_matrix);
}

void absorb_multi_column_hadamard_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    uint edge_count,
    uint column_count,
    uint padded_edge_count,
    uint padded_column_count,
    const G1TensorJacobian& com_alpha,
    const G1TensorJacobian& com_message_matrix,
    const G1TensorJacobian& com_weighted_matrix)
{
    transcript.absorb_label("zkdl_gat_multi_column_hadamard");
    transcript.absorb_label(label);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(column_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(padded_column_count);
    formal_absorb_g1_tensor(transcript, com_alpha);
    formal_absorb_g1_tensor(transcript, com_message_matrix);
    formal_absorb_g1_tensor(transcript, com_weighted_matrix);
}

MultiColumnSparseRoutingGadgetProof prove_multi_column_sparse_routing(
    const FrTensor& node_matrix,
    const FrTensor& edge_matrix,
    const Commitment& matrix_generators,
    const G1TensorJacobian& com_node_matrix,
    const G1TensorJacobian& com_edge_matrix,
    const Commitment& vector_generators,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_column_count,
    const std::string& direction,
    const std::string& relation_label)
{
    if (direction != "source" && direction != "destination") {
        throw std::runtime_error("multi-column sparse routing direction 必须是 source 或 destination");
    }
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (padded_column_count != formal_next_power_of_two_u32(column_count)) {
        throw std::runtime_error("multi-column sparse routing padded column mismatch");
    }
    if (node_matrix.size != padded_node_count * padded_column_count
        || edge_matrix.size != padded_edge_count * padded_column_count) {
        throw std::runtime_error("multi-column sparse routing matrix size mismatch");
    }

    MultiColumnSparseRoutingGadgetProof proof;
    proof.label = relation_label;
    proof.direction = direction;
    proof.topology_hash = formal_sparse_topology_hash(src, dst, node_count, edge_count);
    proof.node_count = node_count;
    proof.edge_count = edge_count;
    proof.column_count = column_count;
    proof.padded_node_count = padded_node_count;
    proof.padded_edge_count = padded_edge_count;
    proof.padded_column_count = padded_column_count;

    FormalTranscript transcript;
    absorb_multi_column_sparse_metadata(
        transcript, relation_label, direction, proof.topology_hash,
        node_count, edge_count, column_count, padded_node_count, padded_edge_count,
        padded_column_count, com_node_matrix, com_edge_matrix);
    proof.column_point = formal_make_challenge_point(
        transcript, ceilLog2(padded_column_count), "multi_column_sparse_column_point");

    const auto compressed_node_host = formal_matrix_column_compression_host(
        node_matrix, node_count, padded_node_count, column_count, padded_column_count, proof.column_point);
    const auto compressed_edge_host = formal_matrix_column_compression_host(
        edge_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    FrTensor compressed_node(static_cast<uint>(compressed_node_host.size()), compressed_node_host.data());
    FrTensor compressed_edge(static_cast<uint>(compressed_edge_host.size()), compressed_edge_host.data());
    const auto com_compressed_node = vector_generators.commit(compressed_node);
    const auto com_compressed_edge = vector_generators.commit(compressed_edge);
    proof.compressed_node_commitment_points = Commitment::host_points(com_compressed_node);
    proof.compressed_edge_commitment_points = Commitment::host_points(com_compressed_edge);
    proof.compressed_routing = formal_sparse_routing_relation_prove(
        compressed_node, compressed_edge,
        vector_generators, com_compressed_node,
        vector_generators, com_compressed_edge,
        src, dst, node_count, edge_count, direction,
        relation_label + ":compressed_sparse_routing");

    const auto node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_node.point});
    const auto edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_edge.point});
    proof.node_matrix_eval = matrix_generators.open_with_proof(
        node_matrix, com_node_matrix, node_matrix_point, proof.opening_node_matrix);
    proof.edge_matrix_eval = matrix_generators.open_with_proof(
        edge_matrix, com_edge_matrix, edge_matrix_point, proof.opening_edge_matrix);
    if (!Fr_host_eq(proof.node_matrix_eval, proof.compressed_routing.opening_node.value)
        || !Fr_host_eq(proof.edge_matrix_eval, proof.compressed_routing.opening_edge.value)) {
        throw std::runtime_error("multi-column sparse routing compression binding mismatch");
    }
    transcript.absorb_label(proof.compressed_routing.transcript_digest);
    transcript.absorb_fr(proof.node_matrix_eval);
    transcript.absorb_fr(proof.edge_matrix_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool verify_multi_column_sparse_routing(
    const MultiColumnSparseRoutingGadgetProof& proof,
    const Commitment& matrix_generators,
    const G1TensorJacobian& com_node_matrix,
    const G1TensorJacobian& com_edge_matrix,
    const Commitment& vector_generators,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    const std::string& direction,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label || proof.direction != direction) {
        return fail("multi-column sparse routing label/direction mismatch");
    }
    if (proof.padded_node_count != formal_next_power_of_two_u32(proof.node_count)
        || proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count)
        || proof.padded_column_count != formal_next_power_of_two_u32(proof.column_count)) {
        return fail("multi-column sparse routing padded shape mismatch");
    }
    const auto expected_hash = formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count);
    if (proof.topology_hash != expected_hash) return fail("multi-column sparse routing topology hash mismatch");

    FormalTranscript transcript;
    absorb_multi_column_sparse_metadata(
        transcript, relation_label, direction, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.column_count, proof.padded_node_count,
        proof.padded_edge_count, proof.padded_column_count, com_node_matrix, com_edge_matrix);
    const auto expected_column_point = formal_make_challenge_point(
        transcript, ceilLog2(proof.padded_column_count), "multi_column_sparse_column_point");
    if (!formal_points_equal(expected_column_point, proof.column_point)) {
        return fail("multi-column sparse routing column point mismatch");
    }
    G1TensorJacobian com_compressed_node(
        static_cast<uint>(proof.compressed_node_commitment_points.size()),
        proof.compressed_node_commitment_points.data());
    G1TensorJacobian com_compressed_edge(
        static_cast<uint>(proof.compressed_edge_commitment_points.size()),
        proof.compressed_edge_commitment_points.data());
    std::string inner_failure;
    if (!formal_sparse_routing_relation_verify(
            proof.compressed_routing,
            vector_generators, com_compressed_node,
            vector_generators, com_compressed_edge,
            src, dst, direction,
            relation_label + ":compressed_sparse_routing",
            &inner_failure)) {
        return fail("multi-column sparse routing compressed verifier failed: " + inner_failure);
    }
    const auto expected_node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_node.point});
    const auto expected_edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_edge.point});
    if (!formal_points_equal(proof.opening_node_matrix.point, expected_node_matrix_point)
        || !formal_points_equal(proof.opening_edge_matrix.point, expected_edge_matrix_point)) {
        return fail("multi-column sparse routing matrix opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_node_matrix.value, proof.node_matrix_eval)
        || !Fr_host_eq(proof.opening_edge_matrix.value, proof.edge_matrix_eval)) {
        return fail("multi-column sparse routing matrix opening value mismatch");
    }
    if (!Fr_host_eq(proof.node_matrix_eval, proof.compressed_routing.opening_node.value)
        || !Fr_host_eq(proof.edge_matrix_eval, proof.compressed_routing.opening_edge.value)) {
        return fail("multi-column sparse routing compression binding mismatch");
    }
    if (!Commitment::verify_opening(matrix_generators, com_node_matrix, proof.opening_node_matrix)
        || !Commitment::verify_opening(matrix_generators, com_edge_matrix, proof.opening_edge_matrix)) {
        return fail("multi-column sparse routing matrix opening rejected");
    }
    transcript.absorb_label(proof.compressed_routing.transcript_digest);
    transcript.absorb_fr(proof.node_matrix_eval);
    transcript.absorb_fr(proof.edge_matrix_eval);
    if (proof.transcript_digest != transcript.digest_hex()) {
        return fail("multi-column sparse routing transcript digest mismatch");
    }
    return true;
}

MultiColumnHadamardGadgetProof prove_multi_column_alpha_hadamard(
    const FrTensor& alpha,
    const FrTensor& message_matrix,
    const FrTensor& weighted_matrix,
    const Commitment& vector_generators,
    const G1TensorJacobian& com_alpha,
    const Commitment& matrix_generators,
    const G1TensorJacobian& com_message_matrix,
    const G1TensorJacobian& com_weighted_matrix,
    uint edge_count,
    uint column_count,
    uint padded_column_count,
    const std::string& relation_label)
{
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (padded_column_count != formal_next_power_of_two_u32(column_count)) {
        throw std::runtime_error("multi-column Hadamard padded column mismatch");
    }
    if (alpha.size != padded_edge_count
        || message_matrix.size != padded_edge_count * padded_column_count
        || weighted_matrix.size != padded_edge_count * padded_column_count) {
        throw std::runtime_error("multi-column Hadamard tensor size mismatch");
    }

    MultiColumnHadamardGadgetProof proof;
    proof.label = relation_label;
    proof.edge_count = edge_count;
    proof.column_count = column_count;
    proof.padded_edge_count = padded_edge_count;
    proof.padded_column_count = padded_column_count;

    FormalTranscript transcript;
    absorb_multi_column_hadamard_metadata(
        transcript, relation_label, edge_count, column_count, padded_edge_count,
        padded_column_count, com_alpha, com_message_matrix, com_weighted_matrix);
    proof.column_point = formal_make_challenge_point(
        transcript, ceilLog2(padded_column_count), "multi_column_hadamard_column_point");
    const auto compressed_message_host = formal_matrix_column_compression_host(
        message_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    const auto compressed_weighted_host = formal_matrix_column_compression_host(
        weighted_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    FrTensor compressed_message(static_cast<uint>(compressed_message_host.size()), compressed_message_host.data());
    FrTensor compressed_weighted(static_cast<uint>(compressed_weighted_host.size()), compressed_weighted_host.data());
    const auto com_compressed_message = vector_generators.commit(compressed_message);
    const auto com_compressed_weighted = vector_generators.commit(compressed_weighted);
    proof.compressed_message_commitment_points = Commitment::host_points(com_compressed_message);
    proof.compressed_weighted_commitment_points = Commitment::host_points(com_compressed_weighted);
    proof.compressed_hadamard = formal_hadamard_relation_prove(
        alpha, compressed_message, compressed_weighted,
        vector_generators, com_alpha, com_compressed_message, com_compressed_weighted,
        relation_label + ":compressed_alpha_times_message");
    const auto message_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_b.point});
    const auto weighted_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_c.point});
    proof.message_matrix_eval = matrix_generators.open_with_proof(
        message_matrix, com_message_matrix, message_matrix_point, proof.opening_message_matrix);
    proof.weighted_matrix_eval = matrix_generators.open_with_proof(
        weighted_matrix, com_weighted_matrix, weighted_matrix_point, proof.opening_weighted_matrix);
    if (!Fr_host_eq(proof.message_matrix_eval, proof.compressed_hadamard.opening_b.value)
        || !Fr_host_eq(proof.weighted_matrix_eval, proof.compressed_hadamard.opening_c.value)) {
        throw std::runtime_error("multi-column Hadamard compression binding mismatch");
    }
    transcript.absorb_label(proof.compressed_hadamard.transcript_digest);
    transcript.absorb_fr(proof.message_matrix_eval);
    transcript.absorb_fr(proof.weighted_matrix_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool verify_multi_column_alpha_hadamard(
    const MultiColumnHadamardGadgetProof& proof,
    const Commitment& vector_generators,
    const G1TensorJacobian& com_alpha,
    const Commitment& matrix_generators,
    const G1TensorJacobian& com_message_matrix,
    const G1TensorJacobian& com_weighted_matrix,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("multi-column Hadamard label mismatch");
    if (proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count)
        || proof.padded_column_count != formal_next_power_of_two_u32(proof.column_count)) {
        return fail("multi-column Hadamard padded shape mismatch");
    }
    FormalTranscript transcript;
    absorb_multi_column_hadamard_metadata(
        transcript, relation_label, proof.edge_count, proof.column_count,
        proof.padded_edge_count, proof.padded_column_count,
        com_alpha, com_message_matrix, com_weighted_matrix);
    const auto expected_column_point = formal_make_challenge_point(
        transcript, ceilLog2(proof.padded_column_count), "multi_column_hadamard_column_point");
    if (!formal_points_equal(expected_column_point, proof.column_point)) {
        return fail("multi-column Hadamard column point mismatch");
    }
    G1TensorJacobian com_compressed_message(
        static_cast<uint>(proof.compressed_message_commitment_points.size()),
        proof.compressed_message_commitment_points.data());
    G1TensorJacobian com_compressed_weighted(
        static_cast<uint>(proof.compressed_weighted_commitment_points.size()),
        proof.compressed_weighted_commitment_points.data());
    std::string inner_failure;
    if (!formal_hadamard_relation_verify(
            proof.compressed_hadamard,
            vector_generators, com_alpha, com_compressed_message, com_compressed_weighted,
            relation_label + ":compressed_alpha_times_message",
            &inner_failure)) {
        return fail("multi-column Hadamard compressed verifier failed: " + inner_failure);
    }
    const auto expected_message_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_b.point});
    const auto expected_weighted_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_c.point});
    if (!formal_points_equal(proof.opening_message_matrix.point, expected_message_matrix_point)
        || !formal_points_equal(proof.opening_weighted_matrix.point, expected_weighted_matrix_point)) {
        return fail("multi-column Hadamard matrix opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_message_matrix.value, proof.message_matrix_eval)
        || !Fr_host_eq(proof.opening_weighted_matrix.value, proof.weighted_matrix_eval)) {
        return fail("multi-column Hadamard matrix opening value mismatch");
    }
    if (!Fr_host_eq(proof.message_matrix_eval, proof.compressed_hadamard.opening_b.value)
        || !Fr_host_eq(proof.weighted_matrix_eval, proof.compressed_hadamard.opening_c.value)) {
        return fail("multi-column Hadamard compression binding mismatch");
    }
    if (!Commitment::verify_opening(matrix_generators, com_message_matrix, proof.opening_message_matrix)
        || !Commitment::verify_opening(matrix_generators, com_weighted_matrix, proof.opening_weighted_matrix)) {
        return fail("multi-column Hadamard matrix opening rejected");
    }
    transcript.absorb_label(proof.compressed_hadamard.transcript_digest);
    transcript.absorb_fr(proof.message_matrix_eval);
    transcript.absorb_fr(proof.weighted_matrix_eval);
    if (proof.transcript_digest != transcript.digest_hex()) {
        return fail("multi-column Hadamard transcript digest mismatch");
    }
    return true;
}
