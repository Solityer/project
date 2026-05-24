#pragma once

// attention dot 与 LeakyReLU lookup/range 子链。
// 证明内容包括 source/destination attention dot 的 zkDL matmul、dot rescale 后与 node attention 的 bias 绑定、
// node->edge routing、edge score 绑定，以及 LeakyReLU 的 selector/branch algebra。

struct AttentionDotWitness {
    uint node_count = 0;
    uint edge_count = 0;
    uint width = 0;
    uint padded_nodes = 0;
    uint padded_edges = 0;
    uint padded_width = 0;
    std::vector<Fr_t> h_prime_mont;
    std::vector<Fr_t> attn_src_mont;
    std::vector<Fr_t> attn_dst_mont;
    std::vector<Fr_t> pre_src_mont;
    std::vector<Fr_t> pre_dst_mont;
    std::vector<Fr_t> dot_src_mont;
    std::vector<Fr_t> dot_dst_mont;
    std::vector<Fr_t> residual_src_mont;
    std::vector<Fr_t> residual_dst_mont;
    std::vector<Fr_t> e_src_mont;
    std::vector<Fr_t> e_dst_mont;
    std::vector<Fr_t> valid_node_mont;
    std::vector<Fr_t> edge_src_mont;
    std::vector<Fr_t> edge_dst_mont;
    std::vector<Fr_t> score_mont;
    std::vector<Fr_t> z_mont;
    std::vector<Fr_t> valid_edge_mont;
    std::vector<Fr_t> pos_selector_mont;
    std::vector<Fr_t> neg_selector_mont;
    std::vector<Fr_t> pos_score_mont;
    std::vector<Fr_t> pos_z_mont;
    std::vector<Fr_t> neg_score_mont;
    std::vector<Fr_t> neg_z_mont;
    std::vector<Fr_t> neg_residual_mont;
    std::size_t attention_dot_mismatch_count = 0;
    std::size_t lrelu_mismatch_count = 0;
    std::int64_t max_abs_dot_residual = 0;
    std::int64_t max_abs_lrelu_residual = 0;
};

std::vector<Fr_t> attention_i64_to_mont(const std::vector<std::int64_t>& values)
{
    return signed_i64_to_mont_vector(values);
}

// 构造 source/destination attention dot、bias、score 与 LeakyReLU 分支见证。
// 该见证只参与 relation proof，official proof object 只保留摘要与 mismatch counter。
AttentionDotWitness build_attention_dot_witness(
    const LocalGraph& graph,
    const HeadParams& head,
    const HeadTraceHost& trace)
{
    if (trace.h_prime.size() != graph.num_nodes * head.output_dim
        || trace.e_src.size() != graph.num_nodes
        || trace.e_dst.size() != graph.num_nodes
        || trace.z.size() != graph.edges.size()
        || head.attn_src.size() != head.output_dim
        || head.attn_dst.size() != head.output_dim) {
        throw std::runtime_error("attention dot witness 输入形状不一致");
    }
    AttentionDotWitness witness;
    witness.node_count = static_cast<uint>(graph.num_nodes);
    witness.edge_count = static_cast<uint>(graph.edges.size());
    witness.width = static_cast<uint>(head.output_dim);
    witness.padded_nodes = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.num_nodes)));
    witness.padded_edges = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.edges.size())));
    witness.padded_width = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head.output_dim)));

    const auto h_i64 = matrix_scale16_i64(
        trace.h_prime, graph.num_nodes, head.output_dim, witness.padded_nodes, witness.padded_width);
    const auto attn_src_i64 = matrix_scale16_i64(
        head.attn_src, head.output_dim, 1, witness.padded_width, 1);
    const auto attn_dst_i64 = matrix_scale16_i64(
        head.attn_dst, head.output_dim, 1, witness.padded_width, 1);
    const auto pre_src_i64 = projection_product_i64(
        h_i64, attn_src_i64, witness.padded_nodes, witness.padded_width, 1);
    const auto pre_dst_i64 = projection_product_i64(
        h_i64, attn_dst_i64, witness.padded_nodes, witness.padded_width, 1);
    const auto bias_src_i64 = quantize_scale16_to_i64(head.attn_src_bias);
    const auto bias_dst_i64 = quantize_scale16_to_i64(head.attn_dst_bias);

    std::vector<std::int64_t> dot_src_i64(witness.padded_nodes, 0);
    std::vector<std::int64_t> dot_dst_i64(witness.padded_nodes, 0);
    std::vector<std::int64_t> residual_src_i64(witness.padded_nodes, 0);
    std::vector<std::int64_t> residual_dst_i64(witness.padded_nodes, 0);
    std::vector<std::int64_t> e_src_i64(witness.padded_nodes, 0);
    std::vector<std::int64_t> e_dst_i64(witness.padded_nodes, 0);
    for (std::size_t node = 0; node < witness.padded_nodes; ++node) {
        if (node >= graph.num_nodes) continue;
        dot_src_i64[node] = round_half_away_div_i64(pre_src_i64[node], 16);
        dot_dst_i64[node] = round_half_away_div_i64(pre_dst_i64[node], 16);
        residual_src_i64[node] = checked_i128_to_i64(
            static_cast<__int128>(pre_src_i64[node]) - static_cast<__int128>(16) * dot_src_i64[node],
            "attention source dot residual");
        residual_dst_i64[node] = checked_i128_to_i64(
            static_cast<__int128>(pre_dst_i64[node]) - static_cast<__int128>(16) * dot_dst_i64[node],
            "attention destination dot residual");
        witness.max_abs_dot_residual = std::max(
            witness.max_abs_dot_residual,
            std::max<std::int64_t>(std::llabs(residual_src_i64[node]), std::llabs(residual_dst_i64[node])));
        e_src_i64[node] = dot_src_i64[node] + bias_src_i64;
        e_dst_i64[node] = dot_dst_i64[node] + bias_dst_i64;
        if (e_src_i64[node] != quantize_scale16_to_i64(trace.e_src[node])
            || e_dst_i64[node] != quantize_scale16_to_i64(trace.e_dst[node])) {
            ++witness.attention_dot_mismatch_count;
        }
    }

    std::vector<std::int64_t> edge_src_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> edge_dst_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> score_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> z_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> valid_edge_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> pos_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> neg_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> pos_score_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> pos_z_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> neg_score_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> neg_z_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> neg_residual_i64(witness.padded_edges, 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto src = graph.edges[edge].src;
        const auto dst = graph.edges[edge].dst;
        edge_src_i64[edge] = e_src_i64[src];
        edge_dst_i64[edge] = e_dst_i64[dst];
        score_i64[edge] = checked_i128_to_i64(
            static_cast<__int128>(edge_src_i64[edge]) + static_cast<__int128>(edge_dst_i64[edge]),
            "attention edge score");
        z_i64[edge] = score_i64[edge] >= 0 ? score_i64[edge] : round_half_away_div_i64(score_i64[edge], 5);
        valid_edge_i64[edge] = 1;
        pos_i64[edge] = score_i64[edge] >= 0 ? 1 : 0;
        neg_i64[edge] = score_i64[edge] < 0 ? 1 : 0;
        pos_score_i64[edge] = pos_i64[edge] * score_i64[edge];
        pos_z_i64[edge] = pos_i64[edge] * z_i64[edge];
        neg_score_i64[edge] = neg_i64[edge] * score_i64[edge];
        neg_z_i64[edge] = neg_i64[edge] * z_i64[edge];
        neg_residual_i64[edge] = checked_i128_to_i64(
            static_cast<__int128>(neg_score_i64[edge]) - static_cast<__int128>(5) * neg_z_i64[edge],
            "attention LeakyReLU negative residual");
        witness.max_abs_lrelu_residual = std::max(
            witness.max_abs_lrelu_residual,
            static_cast<std::int64_t>(std::llabs(neg_residual_i64[edge])));
        if (z_i64[edge] != quantize_scale16_to_i64(trace.z[edge])) ++witness.lrelu_mismatch_count;
    }

    witness.h_prime_mont = attention_i64_to_mont(h_i64);
    witness.attn_src_mont = attention_i64_to_mont(attn_src_i64);
    witness.attn_dst_mont = attention_i64_to_mont(attn_dst_i64);
    witness.pre_src_mont = attention_i64_to_mont(pre_src_i64);
    witness.pre_dst_mont = attention_i64_to_mont(pre_dst_i64);
    witness.dot_src_mont = attention_i64_to_mont(dot_src_i64);
    witness.dot_dst_mont = attention_i64_to_mont(dot_dst_i64);
    witness.residual_src_mont = attention_i64_to_mont(residual_src_i64);
    witness.residual_dst_mont = attention_i64_to_mont(residual_dst_i64);
    witness.e_src_mont = attention_i64_to_mont(e_src_i64);
    witness.e_dst_mont = attention_i64_to_mont(e_dst_i64);
    witness.valid_node_mont = valid_one_mont_vector(graph.num_nodes, witness.padded_nodes);
    witness.edge_src_mont = attention_i64_to_mont(edge_src_i64);
    witness.edge_dst_mont = attention_i64_to_mont(edge_dst_i64);
    witness.score_mont = attention_i64_to_mont(score_i64);
    witness.z_mont = attention_i64_to_mont(z_i64);
    witness.valid_edge_mont = attention_i64_to_mont(valid_edge_i64);
    witness.pos_selector_mont = attention_i64_to_mont(pos_i64);
    witness.neg_selector_mont = attention_i64_to_mont(neg_i64);
    witness.pos_score_mont = attention_i64_to_mont(pos_score_i64);
    witness.pos_z_mont = attention_i64_to_mont(pos_z_i64);
    witness.neg_score_mont = attention_i64_to_mont(neg_score_i64);
    witness.neg_z_mont = attention_i64_to_mont(neg_z_i64);
    witness.neg_residual_mont = attention_i64_to_mont(neg_residual_i64);
    return witness;
}

// GPU full-chain attention-dot/LReLU relation：绑定 dot product、bias、routing 与 LReLU lookup/range。
ZkdlFullRelationRecord prove_gpu_attention_dot_lrelu_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadParams& head,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_attention_dot_witness(graph, head, trace);
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const std::size_t matmul_block = projection_common_commitment_block_size(
        requested_commitment_block_size, witness.h_prime_mont.size(), witness.attn_src_mont.size(), witness.pre_src_mont.size());
    const std::size_t edge_block = formal_effective_commitment_block_size(
        requested_commitment_block_size, witness.padded_nodes, witness.padded_edges);

    ZkdlFullRelationRecord record;
    record.stage = "attention_dot_lrelu_lookup_range";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":attention_dot_lrelu";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    Commitment matmul_gen = make_formal_commitment_generators(matmul_block);
    Commitment edge_gen = make_formal_commitment_generators(edge_block);
    FrTensor h_tensor(static_cast<uint>(witness.h_prime_mont.size()), witness.h_prime_mont.data());
    FrTensor attn_src_tensor(static_cast<uint>(witness.attn_src_mont.size()), witness.attn_src_mont.data());
    FrTensor attn_dst_tensor(static_cast<uint>(witness.attn_dst_mont.size()), witness.attn_dst_mont.data());
    FrTensor pre_src_tensor(static_cast<uint>(witness.pre_src_mont.size()), witness.pre_src_mont.data());
    FrTensor pre_dst_tensor(static_cast<uint>(witness.pre_dst_mont.size()), witness.pre_dst_mont.data());
    FrTensor dot_src_tensor(static_cast<uint>(witness.dot_src_mont.size()), witness.dot_src_mont.data());
    FrTensor dot_dst_tensor(static_cast<uint>(witness.dot_dst_mont.size()), witness.dot_dst_mont.data());
    FrTensor residual_src_tensor(static_cast<uint>(witness.residual_src_mont.size()), witness.residual_src_mont.data());
    FrTensor residual_dst_tensor(static_cast<uint>(witness.residual_dst_mont.size()), witness.residual_dst_mont.data());
    FrTensor e_src_tensor(static_cast<uint>(witness.e_src_mont.size()), witness.e_src_mont.data());
    FrTensor e_dst_tensor(static_cast<uint>(witness.e_dst_mont.size()), witness.e_dst_mont.data());
    FrTensor valid_node_tensor(static_cast<uint>(witness.valid_node_mont.size()), witness.valid_node_mont.data());
    FrTensor edge_src_tensor(static_cast<uint>(witness.edge_src_mont.size()), witness.edge_src_mont.data());
    FrTensor edge_dst_tensor(static_cast<uint>(witness.edge_dst_mont.size()), witness.edge_dst_mont.data());
    FrTensor score_tensor(static_cast<uint>(witness.score_mont.size()), witness.score_mont.data());
    FrTensor z_tensor(static_cast<uint>(witness.z_mont.size()), witness.z_mont.data());
    FrTensor valid_edge_tensor(static_cast<uint>(witness.valid_edge_mont.size()), witness.valid_edge_mont.data());
    FrTensor pos_tensor(static_cast<uint>(witness.pos_selector_mont.size()), witness.pos_selector_mont.data());
    FrTensor neg_tensor(static_cast<uint>(witness.neg_selector_mont.size()), witness.neg_selector_mont.data());
    FrTensor pos_score_tensor(static_cast<uint>(witness.pos_score_mont.size()), witness.pos_score_mont.data());
    FrTensor pos_z_tensor(static_cast<uint>(witness.pos_z_mont.size()), witness.pos_z_mont.data());
    FrTensor neg_score_tensor(static_cast<uint>(witness.neg_score_mont.size()), witness.neg_score_mont.data());
    FrTensor neg_z_tensor(static_cast<uint>(witness.neg_z_mont.size()), witness.neg_z_mont.data());
    FrTensor neg_residual_tensor(static_cast<uint>(witness.neg_residual_mont.size()), witness.neg_residual_mont.data());

    ScopedTimer commit_timer;
    const auto com_h = matmul_gen.commit(h_tensor);
    const auto com_attn_src = matmul_gen.commit(attn_src_tensor);
    const auto com_attn_dst = matmul_gen.commit(attn_dst_tensor);
    const auto com_pre_src = matmul_gen.commit(pre_src_tensor);
    const auto com_pre_dst = matmul_gen.commit(pre_dst_tensor);
    const auto com_dot_src = matmul_gen.commit(dot_src_tensor);
    const auto com_dot_dst = matmul_gen.commit(dot_dst_tensor);
    const auto com_residual_src = matmul_gen.commit(residual_src_tensor);
    const auto com_residual_dst = matmul_gen.commit(residual_dst_tensor);
    const auto com_e_src = matmul_gen.commit(e_src_tensor);
    const auto com_e_dst = matmul_gen.commit(e_dst_tensor);
    const auto com_valid_node = matmul_gen.commit(valid_node_tensor);
    const auto com_edge_src = edge_gen.commit(edge_src_tensor);
    const auto com_edge_dst = edge_gen.commit(edge_dst_tensor);
    const auto com_score = edge_gen.commit(score_tensor);
    const auto com_z = edge_gen.commit(z_tensor);
    const auto com_valid_edge = edge_gen.commit(valid_edge_tensor);
    const auto com_pos = edge_gen.commit(pos_tensor);
    const auto com_neg = edge_gen.commit(neg_tensor);
    const auto com_pos_score = edge_gen.commit(pos_score_tensor);
    const auto com_pos_z = edge_gen.commit(pos_z_tensor);
    const auto com_neg_score = edge_gen.commit(neg_score_tensor);
    const auto com_neg_z = edge_gen.commit(neg_z_tensor);
    const auto com_neg_residual = edge_gen.commit(neg_residual_tensor);
    cuda_sync("attention dot/lrelu commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const uint num = 1;
    const uint k = 1;
    const std::string prefix = record.relation_scope;
    MatmulRelationProof src_dot_proof;
    MatmulRelationProof dst_dot_proof;
    {
        ScopedTimer timer;
        src_dot_proof = formal_matmul_relation_prove(
            h_tensor, attn_src_tensor, pre_src_tensor, matmul_gen, com_h, com_attn_src, com_pre_src,
            num, witness.node_count, witness.width, k, num, witness.padded_nodes, witness.padded_width, k,
            prefix + ":source_attention_dot_matmul");
        dst_dot_proof = formal_matmul_relation_prove(
            h_tensor, attn_dst_tensor, pre_dst_tensor, matmul_gen, com_h, com_attn_dst, com_pre_dst,
            num, witness.node_count, witness.width, k, num, witness.padded_nodes, witness.padded_width, k,
            prefix + ":destination_attention_dot_matmul");
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_matmul_relation_verify(src_dot_proof, matmul_gen, com_h, com_attn_src, com_pre_src, prefix + ":source_attention_dot_matmul", &failure)) {
            throw std::runtime_error("GPU source attention dot verifier rejected: " + failure);
        }
        if (!formal_matmul_relation_verify(dst_dot_proof, matmul_gen, com_h, com_attn_dst, com_pre_dst, prefix + ":destination_attention_dot_matmul", &failure)) {
            throw std::runtime_error("GPU destination attention dot verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += src_dot_proof.serialized_size_bytes() + dst_dot_proof.serialized_size_bytes();
    record.proof_digests.push_back("source_dot=" + src_dot_proof.transcript_digest);
    record.proof_digests.push_back("destination_dot=" + dst_dot_proof.transcript_digest);

    const auto node_point = deterministic_mont_fr_vec(ceilLog2(witness.padded_nodes), prefix + ":node_point");
    const auto edge_point = deterministic_mont_fr_vec(ceilLog2(witness.padded_edges), prefix + ":edge_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    const auto neg_scale = fr_from_signed_mont_host(-16);
    const std::vector<uint> node_shape = {witness.padded_nodes};
    const std::vector<uint> edge_shape = {witness.padded_edges};
    const auto node_valid_eval = evaluate_valid_selector_mle(graph.num_nodes, node_point);

    std::vector<LinearRelationProof> linear_proofs;
    auto prove_linear = [&](const std::string& label, const std::vector<FormalLinearTerm>& terms,
                            const Fr_t& public_coeff, const Fr_t& public_eval,
                            const Fr_t& constant, const std::vector<const G1TensorJacobian*>& commitments) {
        ScopedTimer timer;
        auto proof = formal_linear_relation_prove(label, terms.front().point.size() == node_point.size() ? node_shape : edge_shape,
            terms, terms.front().point.size() == node_point.size() ? matmul_gen : edge_gen, public_coeff, public_eval, constant);
        record.prove_ms += timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const auto& gen = terms.front().point.size() == node_point.size() ? matmul_gen : edge_gen;
        if (!formal_linear_relation_verify(proof, gen, commitments, public_eval, &failure)) {
            throw std::runtime_error("GPU attention linear verifier rejected: " + label + " " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
        record.proof_size_bytes += proof.serialized_size_bytes();
        record.proof_digests.push_back(label.substr(prefix.size() + 1) + "=" + proof.transcript_digest);
    };
    prove_linear(prefix + ":source_dot_rescale_identity",
        {{"pre_src", &pre_src_tensor, &com_pre_src, one, node_point},
         {"dot_src", &dot_src_tensor, &com_dot_src, neg_scale, node_point},
         {"residual_src", &residual_src_tensor, &com_residual_src, neg_one, node_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pre_src, &com_dot_src, &com_residual_src});
    prove_linear(prefix + ":destination_dot_rescale_identity",
        {{"pre_dst", &pre_dst_tensor, &com_pre_dst, one, node_point},
         {"dot_dst", &dot_dst_tensor, &com_dot_dst, neg_scale, node_point},
         {"residual_dst", &residual_dst_tensor, &com_residual_dst, neg_one, node_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pre_dst, &com_dot_dst, &com_residual_dst});
    prove_linear(prefix + ":source_attention_bias_binding",
        {{"e_src", &e_src_tensor, &com_e_src, one, node_point},
         {"dot_src", &dot_src_tensor, &com_dot_src, neg_one, node_point}},
        fr_from_signed_mont_host(-quantize_scale16_to_i64(head.attn_src_bias)), node_valid_eval, Fr_host_zero(), {&com_e_src, &com_dot_src});
    prove_linear(prefix + ":destination_attention_bias_binding",
        {{"e_dst", &e_dst_tensor, &com_e_dst, one, node_point},
         {"dot_dst", &dot_dst_tensor, &com_dot_dst, neg_one, node_point}},
        fr_from_signed_mont_host(-quantize_scale16_to_i64(head.attn_dst_bias)), node_valid_eval, Fr_host_zero(), {&com_e_dst, &com_dot_dst});
    prove_linear(prefix + ":attention_score_binding",
        {{"edge_src", &edge_src_tensor, &com_edge_src, one, edge_point},
         {"edge_dst", &edge_dst_tensor, &com_edge_dst, one, edge_point},
         {"score", &score_tensor, &com_score, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_edge_src, &com_edge_dst, &com_score});
    prove_linear(prefix + ":lrelu_selector_partition",
        {{"pos", &pos_tensor, &com_pos, one, edge_point},
         {"neg", &neg_tensor, &com_neg, one, edge_point},
         {"valid_edge", &valid_edge_tensor, &com_valid_edge, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pos, &com_neg, &com_valid_edge});
    prove_linear(prefix + ":lrelu_positive_branch",
        {{"pos_score", &pos_score_tensor, &com_pos_score, one, edge_point},
         {"pos_z", &pos_z_tensor, &com_pos_z, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pos_score, &com_pos_z});
    prove_linear(prefix + ":lrelu_negative_branch_rescale",
        {{"neg_score", &neg_score_tensor, &com_neg_score, one, edge_point},
         {"neg_z", &neg_z_tensor, &com_neg_z, fr_from_signed_mont_host(-5), edge_point},
         {"neg_residual", &neg_residual_tensor, &com_neg_residual, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_neg_score, &com_neg_z, &com_neg_residual});

    SparseRoutingRelationProof source_route;
    SparseRoutingRelationProof destination_route;
    {
        ScopedTimer timer;
        source_route = formal_sparse_routing_relation_prove(
            e_src_tensor, edge_src_tensor, matmul_gen, com_e_src, edge_gen, com_edge_src,
            src, dst, witness.node_count, witness.edge_count, "source", prefix + ":source_attention_edge_routing");
        destination_route = formal_sparse_routing_relation_prove(
            e_dst_tensor, edge_dst_tensor, matmul_gen, com_e_dst, edge_gen, com_edge_dst,
            src, dst, witness.node_count, witness.edge_count, "destination", prefix + ":destination_attention_edge_routing");
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_sparse_routing_relation_verify(source_route, matmul_gen, com_e_src, edge_gen, com_edge_src, src, dst, "source", prefix + ":source_attention_edge_routing", &failure)) {
            throw std::runtime_error("GPU source attention routing verifier rejected: " + failure);
        }
        if (!formal_sparse_routing_relation_verify(destination_route, matmul_gen, com_e_dst, edge_gen, com_edge_dst, src, dst, "destination", prefix + ":destination_attention_edge_routing", &failure)) {
            throw std::runtime_error("GPU destination attention routing verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += source_route.serialized_size_bytes() + destination_route.serialized_size_bytes();
    record.proof_digests.push_back("source_route=" + source_route.transcript_digest);
    record.proof_digests.push_back("destination_route=" + destination_route.transcript_digest);

    auto prove_hadamard = [&](const std::string& label, const FrTensor& a, const FrTensor& b, const FrTensor& c,
                              const G1TensorJacobian& com_a, const G1TensorJacobian& com_b, const G1TensorJacobian& com_c) {
        ScopedTimer timer;
        auto proof = formal_hadamard_relation_prove(a, b, c, edge_gen, com_a, com_b, com_c, label);
        record.prove_ms += timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        if (!formal_hadamard_relation_verify(proof, edge_gen, com_a, com_b, com_c, label, &failure)) {
            throw std::runtime_error("GPU attention hadamard verifier rejected: " + label + " " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
        record.proof_size_bytes += proof.serialized_size_bytes();
        record.proof_digests.push_back(label.substr(prefix.size() + 1) + "=" + proof.transcript_digest);
    };
    prove_hadamard(prefix + ":pos_selector_boolean", pos_tensor, pos_tensor, pos_tensor, com_pos, com_pos, com_pos);
    prove_hadamard(prefix + ":neg_selector_boolean", neg_tensor, neg_tensor, neg_tensor, com_neg, com_neg, com_neg);
    prove_hadamard(prefix + ":positive_score_mask", pos_tensor, score_tensor, pos_score_tensor, com_pos, com_score, com_pos_score);
    prove_hadamard(prefix + ":positive_z_mask", pos_tensor, z_tensor, pos_z_tensor, com_pos, com_z, com_pos_z);
    prove_hadamard(prefix + ":negative_score_mask", neg_tensor, score_tensor, neg_score_tensor, com_neg, com_score, com_neg_score);
    prove_hadamard(prefix + ":negative_z_mask", neg_tensor, z_tensor, neg_z_tensor, com_neg, com_z, com_neg_z);

    record.proof_digests.push_back("quant_contract_id=" + quant.contract_id);
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("lrelu_table_digest=" + table.lrelu_digest);
    record.proof_digests.push_back("attention_dot_mismatch_count=" + std::to_string(witness.attention_dot_mismatch_count));
    record.proof_digests.push_back("lrelu_mismatch_count=" + std::to_string(witness.lrelu_mismatch_count));
    record.proof_digests.push_back("max_abs_dot_residual=" + std::to_string(witness.max_abs_dot_residual));
    record.proof_digests.push_back("max_abs_lrelu_residual=" + std::to_string(witness.max_abs_lrelu_residual));
    record.proof_generated = true;
    record.verified = true;
    return record;
}

// CPU full-chain attention-dot/LReLU relation：与 GPU 共享 public table/quant digest gate。
ZkdlFullRelationRecord prove_cpu_attention_dot_lrelu_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadParams& head,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_attention_dot_witness(graph, head, trace);
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const std::size_t matmul_block = projection_common_commitment_block_size(
        requested_commitment_block_size, witness.h_prime_mont.size(), witness.attn_src_mont.size(), witness.pre_src_mont.size());
    const std::size_t edge_block = formal_effective_commitment_block_size(
        requested_commitment_block_size, witness.padded_nodes, witness.padded_edges);

    ZkdlFullRelationRecord record;
    record.stage = "cpu_attention_dot_lrelu_lookup_range";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":attention_dot_lrelu";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    CpuCommitment matmul_gen = CpuCommitment::deterministic(
        static_cast<uint>(matmul_block), spec.dataset + ":cpu_attention_matmul_block" + std::to_string(matmul_block));
    CpuCommitment edge_gen = CpuCommitment::deterministic(
        static_cast<uint>(edge_block), spec.dataset + ":cpu_attention_edge_block" + std::to_string(edge_block));
    CpuFrTensor h_tensor(witness.h_prime_mont);
    CpuFrTensor attn_src_tensor(witness.attn_src_mont);
    CpuFrTensor attn_dst_tensor(witness.attn_dst_mont);
    CpuFrTensor pre_src_tensor(witness.pre_src_mont);
    CpuFrTensor pre_dst_tensor(witness.pre_dst_mont);
    CpuFrTensor dot_src_tensor(witness.dot_src_mont);
    CpuFrTensor dot_dst_tensor(witness.dot_dst_mont);
    CpuFrTensor residual_src_tensor(witness.residual_src_mont);
    CpuFrTensor residual_dst_tensor(witness.residual_dst_mont);
    CpuFrTensor e_src_tensor(witness.e_src_mont);
    CpuFrTensor e_dst_tensor(witness.e_dst_mont);
    CpuFrTensor valid_node_tensor(witness.valid_node_mont);
    CpuFrTensor edge_src_tensor(witness.edge_src_mont);
    CpuFrTensor edge_dst_tensor(witness.edge_dst_mont);
    CpuFrTensor score_tensor(witness.score_mont);
    CpuFrTensor z_tensor(witness.z_mont);
    CpuFrTensor valid_edge_tensor(witness.valid_edge_mont);
    CpuFrTensor pos_tensor(witness.pos_selector_mont);
    CpuFrTensor neg_tensor(witness.neg_selector_mont);
    CpuFrTensor pos_score_tensor(witness.pos_score_mont);
    CpuFrTensor pos_z_tensor(witness.pos_z_mont);
    CpuFrTensor neg_score_tensor(witness.neg_score_mont);
    CpuFrTensor neg_z_tensor(witness.neg_z_mont);
    CpuFrTensor neg_residual_tensor(witness.neg_residual_mont);

    ScopedTimer commit_timer;
    const auto com_h = matmul_gen.commit(h_tensor);
    const auto com_attn_src = matmul_gen.commit(attn_src_tensor);
    const auto com_attn_dst = matmul_gen.commit(attn_dst_tensor);
    const auto com_pre_src = matmul_gen.commit(pre_src_tensor);
    const auto com_pre_dst = matmul_gen.commit(pre_dst_tensor);
    const auto com_dot_src = matmul_gen.commit(dot_src_tensor);
    const auto com_dot_dst = matmul_gen.commit(dot_dst_tensor);
    const auto com_residual_src = matmul_gen.commit(residual_src_tensor);
    const auto com_residual_dst = matmul_gen.commit(residual_dst_tensor);
    const auto com_e_src = matmul_gen.commit(e_src_tensor);
    const auto com_e_dst = matmul_gen.commit(e_dst_tensor);
    const auto com_valid_node = matmul_gen.commit(valid_node_tensor);
    const auto com_edge_src = edge_gen.commit(edge_src_tensor);
    const auto com_edge_dst = edge_gen.commit(edge_dst_tensor);
    const auto com_score = edge_gen.commit(score_tensor);
    const auto com_z = edge_gen.commit(z_tensor);
    const auto com_valid_edge = edge_gen.commit(valid_edge_tensor);
    const auto com_pos = edge_gen.commit(pos_tensor);
    const auto com_neg = edge_gen.commit(neg_tensor);
    const auto com_pos_score = edge_gen.commit(pos_score_tensor);
    const auto com_pos_z = edge_gen.commit(pos_z_tensor);
    const auto com_neg_score = edge_gen.commit(neg_score_tensor);
    const auto com_neg_z = edge_gen.commit(neg_z_tensor);
    const auto com_neg_residual = edge_gen.commit(neg_residual_tensor);
    record.commitment_ms += commit_timer.elapsed_ms();

    const uint num = 1;
    const uint k = 1;
    const std::string prefix = record.relation_scope;
    MatmulRelationProof src_dot_proof;
    MatmulRelationProof dst_dot_proof;
    {
        ScopedTimer timer;
        src_dot_proof = cpu_matmul_relation_prove(
            h_tensor, attn_src_tensor, pre_src_tensor, matmul_gen, com_h, com_attn_src, com_pre_src,
            num, witness.node_count, witness.width, k, num, witness.padded_nodes, witness.padded_width, k,
            prefix + ":source_attention_dot_matmul");
        dst_dot_proof = cpu_matmul_relation_prove(
            h_tensor, attn_dst_tensor, pre_dst_tensor, matmul_gen, com_h, com_attn_dst, com_pre_dst,
            num, witness.node_count, witness.width, k, num, witness.padded_nodes, witness.padded_width, k,
            prefix + ":destination_attention_dot_matmul");
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_matmul_relation_verify(src_dot_proof, matmul_gen, com_h, com_attn_src, com_pre_src, prefix + ":source_attention_dot_matmul", &failure)) {
            throw std::runtime_error("CPU source attention dot verifier rejected: " + failure);
        }
        if (!cpu_matmul_relation_verify(dst_dot_proof, matmul_gen, com_h, com_attn_dst, com_pre_dst, prefix + ":destination_attention_dot_matmul", &failure)) {
            throw std::runtime_error("CPU destination attention dot verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += src_dot_proof.serialized_size_bytes() + dst_dot_proof.serialized_size_bytes();
    record.proof_digests.push_back("source_dot=" + src_dot_proof.transcript_digest);
    record.proof_digests.push_back("destination_dot=" + dst_dot_proof.transcript_digest);

    const auto node_point = cpu_deterministic_mont_fr_vec(ceilLog2(witness.padded_nodes), prefix + ":node_point");
    const auto edge_point = cpu_deterministic_mont_fr_vec(ceilLog2(witness.padded_edges), prefix + ":edge_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    const auto neg_scale = fr_from_signed_mont_host(-16);
    const std::vector<uint> node_shape = {witness.padded_nodes};
    const std::vector<uint> edge_shape = {witness.padded_edges};
    const auto node_valid_eval = evaluate_valid_selector_mle(graph.num_nodes, node_point);

    auto prove_cpu_linear = [&](const std::string& label, const std::vector<CpuLinearTerm>& terms,
                                const Fr_t& public_coeff, const Fr_t& public_eval,
                                const Fr_t& constant, const std::vector<const CpuG1Tensor*>& commitments) {
        const bool node_relation = terms.front().point.size() == node_point.size();
        const auto& gen = node_relation ? matmul_gen : edge_gen;
        ScopedTimer timer;
        auto proof = cpu_linear_relation_prove(
            label, node_relation ? node_shape : edge_shape, terms, gen, public_coeff, public_eval, constant);
        record.prove_ms += timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        if (!cpu_linear_relation_verify(proof, gen, commitments, public_eval, &failure)) {
            throw std::runtime_error("CPU attention linear verifier rejected: " + label + " " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
        record.proof_size_bytes += proof.serialized_size_bytes();
        record.proof_digests.push_back(label.substr(prefix.size() + 1) + "=" + proof.transcript_digest);
    };
    prove_cpu_linear(prefix + ":source_dot_rescale_identity",
        {{"pre_src", &pre_src_tensor, &com_pre_src, one, node_point},
         {"dot_src", &dot_src_tensor, &com_dot_src, neg_scale, node_point},
         {"residual_src", &residual_src_tensor, &com_residual_src, neg_one, node_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pre_src, &com_dot_src, &com_residual_src});
    prove_cpu_linear(prefix + ":destination_dot_rescale_identity",
        {{"pre_dst", &pre_dst_tensor, &com_pre_dst, one, node_point},
         {"dot_dst", &dot_dst_tensor, &com_dot_dst, neg_scale, node_point},
         {"residual_dst", &residual_dst_tensor, &com_residual_dst, neg_one, node_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pre_dst, &com_dot_dst, &com_residual_dst});
    prove_cpu_linear(prefix + ":source_attention_bias_binding",
        {{"e_src", &e_src_tensor, &com_e_src, one, node_point},
         {"dot_src", &dot_src_tensor, &com_dot_src, neg_one, node_point}},
        fr_from_signed_mont_host(-quantize_scale16_to_i64(head.attn_src_bias)), node_valid_eval, Fr_host_zero(), {&com_e_src, &com_dot_src});
    prove_cpu_linear(prefix + ":destination_attention_bias_binding",
        {{"e_dst", &e_dst_tensor, &com_e_dst, one, node_point},
         {"dot_dst", &dot_dst_tensor, &com_dot_dst, neg_one, node_point}},
        fr_from_signed_mont_host(-quantize_scale16_to_i64(head.attn_dst_bias)), node_valid_eval, Fr_host_zero(), {&com_e_dst, &com_dot_dst});
    prove_cpu_linear(prefix + ":attention_score_binding",
        {{"edge_src", &edge_src_tensor, &com_edge_src, one, edge_point},
         {"edge_dst", &edge_dst_tensor, &com_edge_dst, one, edge_point},
         {"score", &score_tensor, &com_score, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_edge_src, &com_edge_dst, &com_score});
    prove_cpu_linear(prefix + ":lrelu_selector_partition",
        {{"pos", &pos_tensor, &com_pos, one, edge_point},
         {"neg", &neg_tensor, &com_neg, one, edge_point},
         {"valid_edge", &valid_edge_tensor, &com_valid_edge, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pos, &com_neg, &com_valid_edge});
    prove_cpu_linear(prefix + ":lrelu_positive_branch",
        {{"pos_score", &pos_score_tensor, &com_pos_score, one, edge_point},
         {"pos_z", &pos_z_tensor, &com_pos_z, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_pos_score, &com_pos_z});
    prove_cpu_linear(prefix + ":lrelu_negative_branch_rescale",
        {{"neg_score", &neg_score_tensor, &com_neg_score, one, edge_point},
         {"neg_z", &neg_z_tensor, &com_neg_z, fr_from_signed_mont_host(-5), edge_point},
         {"neg_residual", &neg_residual_tensor, &com_neg_residual, neg_one, edge_point}},
        Fr_host_zero(), Fr_host_zero(), Fr_host_zero(), {&com_neg_score, &com_neg_z, &com_neg_residual});

    SparseRoutingRelationProof source_route;
    SparseRoutingRelationProof destination_route;
    {
        ScopedTimer timer;
        source_route = cpu_sparse_routing_relation_prove(
            e_src_tensor, edge_src_tensor, matmul_gen, com_e_src, edge_gen, com_edge_src,
            src, dst, witness.node_count, witness.edge_count, "source", prefix + ":source_attention_edge_routing");
        destination_route = cpu_sparse_routing_relation_prove(
            e_dst_tensor, edge_dst_tensor, matmul_gen, com_e_dst, edge_gen, com_edge_dst,
            src, dst, witness.node_count, witness.edge_count, "destination", prefix + ":destination_attention_edge_routing");
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_sparse_routing_relation_verify(source_route, matmul_gen, com_e_src, edge_gen, com_edge_src, src, dst, "source", prefix + ":source_attention_edge_routing", &failure)) {
            throw std::runtime_error("CPU source attention routing verifier rejected: " + failure);
        }
        if (!cpu_sparse_routing_relation_verify(destination_route, matmul_gen, com_e_dst, edge_gen, com_edge_dst, src, dst, "destination", prefix + ":destination_attention_edge_routing", &failure)) {
            throw std::runtime_error("CPU destination attention routing verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += source_route.serialized_size_bytes() + destination_route.serialized_size_bytes();
    record.proof_digests.push_back("source_route=" + source_route.transcript_digest);
    record.proof_digests.push_back("destination_route=" + destination_route.transcript_digest);

    auto prove_cpu_hadamard = [&](const std::string& label, const CpuFrTensor& a, const CpuFrTensor& b, const CpuFrTensor& c,
                                  const CpuG1Tensor& com_a, const CpuG1Tensor& com_b, const CpuG1Tensor& com_c) {
        ScopedTimer timer;
        auto proof = cpu_hadamard_relation_prove(a, b, c, edge_gen, com_a, com_b, com_c, label);
        record.prove_ms += timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        if (!cpu_hadamard_relation_verify(proof, edge_gen, com_a, com_b, com_c, label, &failure)) {
            throw std::runtime_error("CPU attention hadamard verifier rejected: " + label + " " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
        record.proof_size_bytes += proof.serialized_size_bytes();
        record.proof_digests.push_back(label.substr(prefix.size() + 1) + "=" + proof.transcript_digest);
    };
    prove_cpu_hadamard(prefix + ":pos_selector_boolean", pos_tensor, pos_tensor, pos_tensor, com_pos, com_pos, com_pos);
    prove_cpu_hadamard(prefix + ":neg_selector_boolean", neg_tensor, neg_tensor, neg_tensor, com_neg, com_neg, com_neg);
    prove_cpu_hadamard(prefix + ":positive_score_mask", pos_tensor, score_tensor, pos_score_tensor, com_pos, com_score, com_pos_score);
    prove_cpu_hadamard(prefix + ":positive_z_mask", pos_tensor, z_tensor, pos_z_tensor, com_pos, com_z, com_pos_z);
    prove_cpu_hadamard(prefix + ":negative_score_mask", neg_tensor, score_tensor, neg_score_tensor, com_neg, com_score, com_neg_score);
    prove_cpu_hadamard(prefix + ":negative_z_mask", neg_tensor, z_tensor, neg_z_tensor, com_neg, com_z, com_neg_z);

    record.proof_digests.push_back("quant_contract_id=" + quant.contract_id);
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("lrelu_table_digest=" + table.lrelu_digest);
    record.proof_digests.push_back("attention_dot_mismatch_count=" + std::to_string(witness.attention_dot_mismatch_count));
    record.proof_digests.push_back("lrelu_mismatch_count=" + std::to_string(witness.lrelu_mismatch_count));
    record.proof_digests.push_back("max_abs_dot_residual=" + std::to_string(witness.max_abs_dot_residual));
    record.proof_digests.push_back("max_abs_lrelu_residual=" + std::to_string(witness.max_abs_lrelu_residual));
    record.proof_generated = true;
    record.verified = true;
    return record;
}
