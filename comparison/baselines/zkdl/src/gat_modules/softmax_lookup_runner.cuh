#pragma once

// 单个 attention head 的 zkDL-native softmax/weighted aggregation 子链。
// 当前实现证明 denominator、nonzero inverse、inverse routing、alpha consistency 和 weighted aggregation。

// 大维度 message 聚合按列分块，避免单个 Hyrax opening 过大。
// 根据列宽选择 softmax 聚合分块；避免超大矩阵 opening 影响可复现性。
std::size_t attention_softmax_column_chunk_limit(std::size_t width)
{
    if (width == 0) throw std::runtime_error("attention softmax column chunk width 为空");
    return std::min<std::size_t>(width, 64);
}

// GPU full-chain softmax/weighted aggregation relation。
// 证明 denominator、inverse、alpha、alpha*message 和按 dst 聚合；不单独提交虚拟 R_src 稠密矩阵。
// GPU 路径生成完整 softmax chain record；绑定 U/Sum/inv/alpha/聚合关系。
ZkdlFullRelationRecord prove_gpu_attention_softmax_chain(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& head,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    if (head.u.size() != graph.edges.size() || head.sum.size() != graph.num_nodes || head.h_prime.size() != graph.num_nodes * head.width) {
        throw std::runtime_error("attention head trace 与 graph 形状不一致");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_edge_count = next_power_of_two(std::max<std::size_t>(1, graph.edges.size()));
    const std::size_t padded_feature_columns = next_power_of_two(std::max<std::size_t>(1, head.width));
    if (padded_node_count > std::numeric_limits<uint>::max()
        || padded_edge_count > std::numeric_limits<uint>::max()
        || padded_feature_columns > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("attention softmax proof 形状超过 uint 上限");
    }

    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const uint node_count_u32 = static_cast<uint>(graph.num_nodes);
    const uint edge_count_u32 = static_cast<uint>(graph.edges.size());
    (void)padded_feature_columns;

    std::vector<std::uint32_t> u_quantized;
    auto u_raw = encode_nonnegative_scale16_raw(head.u, graph.edges.size(), padded_edge_count, &u_quantized);
    auto denom_raw = grouped_sum_by_dst_raw(u_raw, graph, padded_node_count);
    const auto denom_u64 = grouped_sum_by_dst_u64(u_quantized, graph, padded_node_count);
    auto u_mont = mont_vector_from_raw(u_raw);
    auto denom_mont = mont_vector_from_raw(denom_raw);
    std::vector<Fr_t> inv_node_mont(padded_node_count, Fr_host_zero());
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (denom_u64[node] == 0) throw std::runtime_error("attention denominator 为 0");
        inv_node_mont[node] = fr_inverse_uint_mont(static_cast<std::uint32_t>(denom_u64[node]));
    }
    const auto inv_edge_mont = route_vector_by_topology_raw(inv_node_mont, graph, padded_edge_count, false);
    const auto denom_edge_mont = route_vector_by_topology_raw(denom_mont, graph, padded_edge_count, false);
    auto valid_one = valid_one_mont_vector(graph.num_nodes, padded_node_count);
    std::vector<Fr_t> alpha_mont(padded_edge_count, Fr_host_zero());
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        alpha_mont[edge] = Fr_host_mul(u_mont[edge], inv_edge_mont[edge]);
    }

    ZkdlFullRelationRecord record;
    record.stage = "attention_softmax_weighted_aggregation";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + "_head" + std::to_string(head_index);
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    const std::size_t vector_block = formal_effective_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_edge_count);
    Commitment gen = make_formal_commitment_generators(vector_block);
    FrTensor u_tensor(static_cast<uint>(u_mont.size()), u_mont.data());
    FrTensor denom_tensor(static_cast<uint>(denom_mont.size()), denom_mont.data());
    FrTensor denom_edge_tensor(static_cast<uint>(denom_edge_mont.size()), denom_edge_mont.data());
    FrTensor inv_node_tensor(static_cast<uint>(inv_node_mont.size()), inv_node_mont.data());
    FrTensor inv_edge_tensor(static_cast<uint>(inv_edge_mont.size()), inv_edge_mont.data());
    FrTensor valid_one_tensor(static_cast<uint>(valid_one.size()), valid_one.data());
    FrTensor alpha_tensor(static_cast<uint>(alpha_mont.size()), alpha_mont.data());

    ScopedTimer commit_timer;
    const auto com_u = gen.commit(u_tensor);
    const auto com_denom = gen.commit(denom_tensor);
    const auto com_denom_edge = gen.commit(denom_edge_tensor);
    const auto com_inv_node = gen.commit(inv_node_tensor);
    const auto com_inv_edge = gen.commit(inv_edge_tensor);
    const auto com_valid_one = gen.commit(valid_one_tensor);
    const auto com_alpha = gen.commit(alpha_tensor);
    cuda_sync("full-chain attention base commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    const std::string denom_label = prefix + ":softmax_denominator_grouped_sum";
    const std::string inverse_label = prefix + ":softmax_inverse_nonzero";
    const std::string inv_route_label = prefix + ":softmax_inverse_destination_routing";
    const std::string alpha_from_inv_label = prefix + ":alpha_from_inverse";
    const std::string alpha_times_denom_label = prefix + ":alpha_times_denominator_equals_u";

    GroupedSumRelationProof denom_proof;
    HadamardSumcheckProof inverse_proof;
    SparseRoutingRelationProof inv_route_proof;
    HadamardSumcheckProof alpha_from_inv_proof;
    HadamardSumcheckProof alpha_times_denom_proof;
    {
        ScopedTimer timer;
        denom_proof = formal_grouped_sum_relation_prove(
            u_tensor, denom_tensor, gen, com_u, gen, com_denom,
            src, dst, node_count_u32, edge_count_u32, denom_label);
        inverse_proof = formal_hadamard_relation_prove(
            denom_tensor, inv_node_tensor, valid_one_tensor, gen, com_denom, com_inv_node, com_valid_one, inverse_label);
        inv_route_proof = formal_sparse_routing_relation_prove(
            inv_node_tensor, inv_edge_tensor, gen, com_inv_node, gen, com_inv_edge,
            src, dst, node_count_u32, edge_count_u32, "destination", inv_route_label);
        alpha_from_inv_proof = formal_hadamard_relation_prove(
            u_tensor, inv_edge_tensor, alpha_tensor, gen, com_u, com_inv_edge, com_alpha, alpha_from_inv_label);
        alpha_times_denom_proof = formal_hadamard_relation_prove(
            alpha_tensor, denom_edge_tensor, u_tensor, gen, com_alpha, com_denom_edge, com_u, alpha_times_denom_label);
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_grouped_sum_relation_verify(denom_proof, gen, com_u, gen, com_denom, src, dst, denom_label, &failure)) {
            throw std::runtime_error("denominator verifier rejected: " + failure);
        }
        if (!formal_hadamard_relation_verify(inverse_proof, gen, com_denom, com_inv_node, com_valid_one, inverse_label, &failure)) {
            throw std::runtime_error("inverse verifier rejected: " + failure);
        }
        if (!formal_sparse_routing_relation_verify(inv_route_proof, gen, com_inv_node, gen, com_inv_edge, src, dst, "destination", inv_route_label, &failure)) {
            throw std::runtime_error("inverse routing verifier rejected: " + failure);
        }
        if (!formal_hadamard_relation_verify(alpha_from_inv_proof, gen, com_u, com_inv_edge, com_alpha, alpha_from_inv_label, &failure)) {
            throw std::runtime_error("alpha-from-inverse verifier rejected: " + failure);
        }
        if (!formal_hadamard_relation_verify(alpha_times_denom_proof, gen, com_alpha, com_denom_edge, com_u, alpha_times_denom_label, &failure)) {
            throw std::runtime_error("alpha-times-denominator verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += denom_proof.serialized_size_bytes()
        + inverse_proof.serialized_size_bytes()
        + inv_route_proof.serialized_size_bytes()
        + alpha_from_inv_proof.serialized_size_bytes()
        + alpha_times_denom_proof.serialized_size_bytes();
    record.proof_digests.push_back("denominator=" + denom_proof.transcript_digest);
    record.proof_digests.push_back("inverse=" + inverse_proof.transcript_digest);
    record.proof_digests.push_back("inverse_route=" + inv_route_proof.transcript_digest);
    record.proof_digests.push_back("alpha_from_inverse=" + alpha_from_inv_proof.transcript_digest);
    record.proof_digests.push_back("alpha_times_denominator=" + alpha_times_denom_proof.transcript_digest);

    const std::size_t column_chunk_limit = attention_softmax_column_chunk_limit(head.width);
    for (std::size_t col_begin = 0; col_begin < head.width; col_begin += column_chunk_limit) {
        const std::size_t chunk_columns = std::min(column_chunk_limit, head.width - col_begin);
        const std::size_t padded_chunk_columns = next_power_of_two(std::max<std::size_t>(1, chunk_columns));
        const uint chunk_column_count_u32 = static_cast<uint>(chunk_columns);
        const uint padded_chunk_column_count_u32 = static_cast<uint>(padded_chunk_columns);
        const std::string chunk_suffix = ":cols" + std::to_string(col_begin)
            + "_" + std::to_string(col_begin + chunk_columns);
        std::cout << "full_chain_runner dataset=" << spec.dataset
                  << " device=gpu stage=prove_" << layer_kind
                  << "_attention_chunk layer=" << layer_index
                  << " head=" << head_index
                  << " cols=" << col_begin << "_" << (col_begin + chunk_columns)
                  << "\n";
        const std::size_t matrix_block = formal_effective_matrix_commitment_block_size(
            vector_block,
            std::min(padded_node_count, padded_edge_count),
            padded_chunk_columns);
        Commitment matrix_gen = make_formal_commitment_generators(matrix_block);
        const auto h_prime_chunk = slice_matrix_columns(
            head.h_prime, graph.num_nodes, head.width, col_begin, chunk_columns);
        auto message_node_matrix_raw = matrix_scale16_raw(
            h_prime_chunk, graph.num_nodes, chunk_columns, padded_node_count, padded_chunk_columns);
        auto message_edge_matrix_raw = route_matrix_by_topology_raw(
            message_node_matrix_raw, graph, padded_node_count, padded_edge_count, padded_chunk_columns, true);
        auto message_node_matrix_mont = mont_vector_from_raw(message_node_matrix_raw);
        auto message_edge_matrix_mont = mont_vector_from_raw(message_edge_matrix_raw);
        auto weighted_edge_matrix_mont = weighted_matrix_by_alpha_mont(
            alpha_mont, message_edge_matrix_mont, graph.edges.size(), padded_edge_count, padded_chunk_columns);
        auto aggregated_matrix_mont = grouped_sum_matrix_by_dst_mont(
            weighted_edge_matrix_mont, graph, padded_node_count, padded_edge_count, padded_chunk_columns);
        FrTensor message_node_matrix_tensor(static_cast<uint>(message_node_matrix_mont.size()), message_node_matrix_mont.data());
        FrTensor message_edge_matrix_tensor(static_cast<uint>(message_edge_matrix_mont.size()), message_edge_matrix_mont.data());
        FrTensor weighted_edge_matrix_tensor(static_cast<uint>(weighted_edge_matrix_mont.size()), weighted_edge_matrix_mont.data());
        FrTensor aggregated_matrix_tensor(static_cast<uint>(aggregated_matrix_mont.size()), aggregated_matrix_mont.data());

        ScopedTimer matrix_commit_timer;
        const auto com_message_node_matrix = matrix_gen.commit(message_node_matrix_tensor);
        const auto com_message_edge_matrix = matrix_gen.commit(message_edge_matrix_tensor);
        const auto com_weighted_edge_matrix = matrix_gen.commit(weighted_edge_matrix_tensor);
        const auto com_aggregated_matrix = matrix_gen.commit(aggregated_matrix_tensor);
        cuda_sync("full-chain attention weighted aggregation commit " + record.relation_scope + chunk_suffix);
        record.commitment_ms += matrix_commit_timer.elapsed_ms();

        const std::string message_route_label = prefix + ":weighted_message_source_routing" + chunk_suffix;
        const std::string alpha_message_label = prefix + ":alpha_times_message" + chunk_suffix;
        const std::string aggregation_label = prefix + ":weighted_aggregation_by_dst" + chunk_suffix;
        MultiColumnSparseRoutingGadgetProof message_route_proof;
        MultiColumnHadamardGadgetProof alpha_message_proof;
        MultiColumnGroupedSumRelationProof aggregation_proof;
        {
            ScopedTimer timer;
            message_route_proof = prove_multi_column_sparse_routing(
                message_node_matrix_tensor, message_edge_matrix_tensor,
                matrix_gen, com_message_node_matrix, com_message_edge_matrix,
                gen, src, dst, node_count_u32, edge_count_u32,
                chunk_column_count_u32, padded_chunk_column_count_u32,
                "source", message_route_label);
            alpha_message_proof = prove_multi_column_alpha_hadamard(
                alpha_tensor, message_edge_matrix_tensor, weighted_edge_matrix_tensor,
                gen, com_alpha, matrix_gen, com_message_edge_matrix, com_weighted_edge_matrix,
                edge_count_u32, chunk_column_count_u32, padded_chunk_column_count_u32, alpha_message_label);
            aggregation_proof = formal_multi_column_grouped_sum_relation_prove(
                weighted_edge_matrix_tensor, aggregated_matrix_tensor,
                matrix_gen, com_weighted_edge_matrix,
                matrix_gen, com_aggregated_matrix,
                gen, src, dst, node_count_u32, edge_count_u32,
                chunk_column_count_u32, padded_chunk_column_count_u32, aggregation_label);
            record.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            if (!verify_multi_column_sparse_routing(
                    message_route_proof, matrix_gen, com_message_node_matrix, com_message_edge_matrix,
                    gen, src, dst, "source", message_route_label, &failure)) {
                throw std::runtime_error("weighted message routing verifier rejected: " + failure);
            }
            if (!verify_multi_column_alpha_hadamard(
                    alpha_message_proof, gen, com_alpha, matrix_gen,
                    com_message_edge_matrix, com_weighted_edge_matrix,
                    alpha_message_label, &failure)) {
                throw std::runtime_error("alpha-message verifier rejected: " + failure);
            }
            if (!formal_multi_column_grouped_sum_relation_verify(
                    aggregation_proof, matrix_gen, com_weighted_edge_matrix,
                    matrix_gen, com_aggregated_matrix, gen, src, dst, aggregation_label, &failure)) {
                throw std::runtime_error("weighted aggregation verifier rejected: " + failure);
            }
            record.verify_ms += timer.elapsed_ms();
        }
        record.proof_size_bytes += message_route_proof.serialized_size_bytes()
            + alpha_message_proof.serialized_size_bytes()
            + aggregation_proof.serialized_size_bytes();
        record.proof_digests.push_back("message_route" + chunk_suffix + "=" + message_route_proof.transcript_digest);
        record.proof_digests.push_back("alpha_message" + chunk_suffix + "=" + alpha_message_proof.transcript_digest);
        record.proof_digests.push_back("weighted_aggregation" + chunk_suffix + "=" + aggregation_proof.transcript_digest);
    }
    record.proof_generated = true;
    record.verified = true;
    return record;
}
