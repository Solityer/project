#pragma once

// sparse/grouped 与 softmax aggregation partial runner。

struct GatSoftmaxAggregationGadgetRun
{
    bool denominator_ok = false;
    bool inverse_ok = false;
    bool inverse_routing_ok = false;
    bool alpha_from_inverse_ok = false;
    bool alpha_times_denominator_ok = false;
    bool weighted_aggregation_ok = false;
    std::size_t available_feature_columns = 0;
    std::size_t feature_columns = 0;
    std::size_t verified_feature_columns = 0;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::uint64_t topology_hash = 0;
    std::size_t commitment_block_size = 1;
    std::vector<std::string> relation_details;
    fs::path proof_path;
    fs::path evidence_path;
};

GatSoftmaxAggregationGadgetRun run_softmax_aggregation_relations_with_trace(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const ForwardTraceHost& trace,
    const fs::path& output_root,
    std::size_t max_feature_columns,
    std::size_t requested_commitment_block_size)
{
    if (trace.hidden_heads.empty() || trace.hidden_heads.front().empty()) {
        throw std::runtime_error("缺少 hidden layer0/head0 trace，无法生成 softmax/aggregation relation evidence");
    }
    const auto& head = trace.hidden_heads.front().front();
    if (head.u.size() != graph.edges.size() || head.sum.size() != graph.num_nodes || head.h_prime.size() != graph.num_nodes * head.width) {
        throw std::runtime_error("hidden layer0/head0 trace 与 graph 形状不一致");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_edge_count = next_power_of_two(std::max<std::size_t>(1, graph.edges.size()));
    if (padded_node_count > std::numeric_limits<uint>::max() || padded_edge_count > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("softmax/aggregation relation 当前实现超过 uint 形状上限");
    }
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const uint node_count_u32 = static_cast<uint>(graph.num_nodes);
    const uint edge_count_u32 = static_cast<uint>(graph.edges.size());

    std::vector<std::uint32_t> u_quantized;
    auto u_raw = encode_nonnegative_scale16_raw(head.u, graph.edges.size(), padded_edge_count, &u_quantized);
    auto denom_raw = grouped_sum_by_dst_raw(u_raw, graph, padded_node_count);
    const auto denom_u64 = grouped_sum_by_dst_u64(u_quantized, graph, padded_node_count);
    auto u_mont = mont_vector_from_raw(u_raw);
    auto denom_mont = mont_vector_from_raw(denom_raw);
    std::vector<Fr_t> inv_node_mont(padded_node_count, Fr_host_zero());
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (denom_u64[node] == 0) {
            throw std::runtime_error("denominator 为 0，不能生成 inverse relation");
        }
        inv_node_mont[node] = fr_inverse_uint_mont(static_cast<std::uint32_t>(denom_u64[node]));
    }
    const auto inv_edge_mont = route_vector_by_topology_raw(inv_node_mont, graph, padded_edge_count, false);
    const auto denom_edge_mont = route_vector_by_topology_raw(denom_mont, graph, padded_edge_count, false);
    auto valid_one = valid_one_mont_vector(graph.num_nodes, padded_node_count);
    std::vector<Fr_t> alpha_mont(padded_edge_count, Fr_host_zero());
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        alpha_mont[edge] = Fr_host_mul(u_mont[edge], inv_edge_mont[edge]);
    }

    GatSoftmaxAggregationGadgetRun result;
    result.available_feature_columns = head.width;
    result.feature_columns = max_feature_columns == 0 ? head.width : std::min(head.width, max_feature_columns);
    result.topology_hash = formal_sparse_topology_hash(src, dst, node_count_u32, edge_count_u32);
    result.commitment_block_size = formal_effective_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_edge_count);
    Commitment gen = make_formal_commitment_generators(result.commitment_block_size);

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
    cuda_sync("formal softmax/aggregation base commit " + spec.dataset);
    result.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = spec.dataset + ":hidden_layer0_head0";
    const std::string denom_label = prefix + ":softmax_denominator_grouped_sum:U_by_dst";
    const std::string inverse_label = prefix + ":softmax_inverse:denominator_times_inverse";
    const std::string inv_route_label = prefix + ":softmax_inverse:inverse_destination_routing";
    const std::string alpha_from_inv_label = prefix + ":alpha_from_inverse:U_times_inv_edge";
    const std::string alpha_times_denom_label = prefix + ":alpha_times_denom_equals_U";

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
        result.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        result.denominator_ok = formal_grouped_sum_relation_verify(
            denom_proof, gen, com_u, gen, com_denom, src, dst, denom_label, &failure);
        if (!result.denominator_ok) throw std::runtime_error("denominator grouped sum verifier rejected: " + failure);
        result.inverse_ok = formal_hadamard_relation_verify(
            inverse_proof, gen, com_denom, com_inv_node, com_valid_one, inverse_label, &failure);
        if (!result.inverse_ok) throw std::runtime_error("inverse Hadamard verifier rejected: " + failure);
        result.inverse_routing_ok = formal_sparse_routing_relation_verify(
            inv_route_proof, gen, com_inv_node, gen, com_inv_edge, src, dst, "destination", inv_route_label, &failure);
        if (!result.inverse_routing_ok) throw std::runtime_error("inverse destination routing verifier rejected: " + failure);
        result.alpha_from_inverse_ok = formal_hadamard_relation_verify(
            alpha_from_inv_proof, gen, com_u, com_inv_edge, com_alpha, alpha_from_inv_label, &failure);
        if (!result.alpha_from_inverse_ok) throw std::runtime_error("alpha-from-inverse verifier rejected: " + failure);
        result.alpha_times_denominator_ok = formal_hadamard_relation_verify(
            alpha_times_denom_proof, gen, com_alpha, com_denom_edge, com_u, alpha_times_denom_label, &failure);
        if (!result.alpha_times_denominator_ok) throw std::runtime_error("alpha-times-denominator verifier rejected: " + failure);
        result.verify_ms += timer.elapsed_ms();
    }
    result.proof_size_bytes += denom_proof.serialized_size_bytes()
        + inverse_proof.serialized_size_bytes()
        + inv_route_proof.serialized_size_bytes()
        + alpha_from_inv_proof.serialized_size_bytes()
        + alpha_times_denom_proof.serialized_size_bytes();
    result.relation_details.push_back("denominator_grouped_sum status=PASS transcript_digest=" + denom_proof.transcript_digest);
    result.relation_details.push_back("inverse_denominator_times_inverse status=PASS transcript_digest=" + inverse_proof.transcript_digest);
    result.relation_details.push_back("inverse_destination_routing status=PASS transcript_digest=" + inv_route_proof.transcript_digest);
    result.relation_details.push_back("alpha_from_inverse status=PASS transcript_digest=" + alpha_from_inv_proof.transcript_digest);
    result.relation_details.push_back("alpha_times_denominator status=PASS transcript_digest=" + alpha_times_denom_proof.transcript_digest);

    bool weighted_all_ok = true;
    if (result.feature_columns == result.available_feature_columns && result.feature_columns > 0) {
        const std::size_t padded_feature_columns = next_power_of_two(result.available_feature_columns);
        if (padded_feature_columns > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("multi-column feature columns 超出 uint 范围");
        }
        const std::size_t matrix_block_size = formal_effective_matrix_commitment_block_size(
            result.commitment_block_size,
            std::min(padded_node_count, padded_edge_count),
            padded_feature_columns);
        Commitment matrix_gen = make_formal_commitment_generators(matrix_block_size);
        auto message_node_matrix_raw = matrix_scale16_raw(
            head.h_prime, graph.num_nodes, head.width, padded_node_count, padded_feature_columns);
        auto message_edge_matrix_raw = route_matrix_by_topology_raw(
            message_node_matrix_raw, graph, padded_node_count, padded_edge_count,
            padded_feature_columns, true);
        auto message_node_matrix_mont = mont_vector_from_raw(message_node_matrix_raw);
        auto message_edge_matrix_mont = mont_vector_from_raw(message_edge_matrix_raw);
        auto weighted_edge_matrix_mont = weighted_matrix_by_alpha_mont(
            alpha_mont, message_edge_matrix_mont, graph.edges.size(),
            padded_edge_count, padded_feature_columns);
        auto aggregated_matrix_mont = grouped_sum_matrix_by_dst_mont(
            weighted_edge_matrix_mont, graph, padded_node_count,
            padded_edge_count, padded_feature_columns);

        FrTensor message_node_matrix_tensor(
            static_cast<uint>(message_node_matrix_mont.size()), message_node_matrix_mont.data());
        FrTensor message_edge_matrix_tensor(
            static_cast<uint>(message_edge_matrix_mont.size()), message_edge_matrix_mont.data());
        FrTensor weighted_edge_matrix_tensor(
            static_cast<uint>(weighted_edge_matrix_mont.size()), weighted_edge_matrix_mont.data());
        FrTensor aggregated_matrix_tensor(
            static_cast<uint>(aggregated_matrix_mont.size()), aggregated_matrix_mont.data());

        ScopedTimer feature_commit_timer;
        const auto com_message_node_matrix = matrix_gen.commit(message_node_matrix_tensor);
        const auto com_message_edge_matrix = matrix_gen.commit(message_edge_matrix_tensor);
        const auto com_weighted_edge_matrix = matrix_gen.commit(weighted_edge_matrix_tensor);
        const auto com_aggregated_matrix = matrix_gen.commit(aggregated_matrix_tensor);
        cuda_sync("formal multi-column weighted aggregation commit " + spec.dataset);
        result.commitment_ms += feature_commit_timer.elapsed_ms();

        const std::string message_route_label = prefix + ":weighted_aggregation:multi_column_message_source_routing";
        const std::string alpha_message_label = prefix + ":alpha_multiplication:multi_column_alpha_times_message";
        const std::string aggregation_label = prefix + ":weighted_aggregation:multi_column_weighted_by_dst";
        MultiColumnSparseRoutingGadgetProof message_route_proof;
        MultiColumnHadamardGadgetProof alpha_message_proof;
        MultiColumnGroupedSumRelationProof aggregation_proof;
        {
            ScopedTimer timer;
            message_route_proof = prove_multi_column_sparse_routing(
                message_node_matrix_tensor, message_edge_matrix_tensor,
                matrix_gen, com_message_node_matrix, com_message_edge_matrix,
                gen, src, dst, node_count_u32, edge_count_u32,
                static_cast<uint>(result.available_feature_columns),
                static_cast<uint>(padded_feature_columns),
                "source", message_route_label);
            alpha_message_proof = prove_multi_column_alpha_hadamard(
                alpha_tensor, message_edge_matrix_tensor, weighted_edge_matrix_tensor,
                gen, com_alpha, matrix_gen, com_message_edge_matrix, com_weighted_edge_matrix,
                edge_count_u32,
                static_cast<uint>(result.available_feature_columns),
                static_cast<uint>(padded_feature_columns),
                alpha_message_label);
            aggregation_proof = formal_multi_column_grouped_sum_relation_prove(
                weighted_edge_matrix_tensor, aggregated_matrix_tensor,
                matrix_gen, com_weighted_edge_matrix,
                matrix_gen, com_aggregated_matrix,
                gen, src, dst, node_count_u32, edge_count_u32,
                static_cast<uint>(result.available_feature_columns),
                static_cast<uint>(padded_feature_columns),
                aggregation_label);
            result.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            const bool message_ok = verify_multi_column_sparse_routing(
                message_route_proof, matrix_gen, com_message_node_matrix, com_message_edge_matrix,
                gen, src, dst, "source", message_route_label, &failure);
            if (!message_ok) {
                weighted_all_ok = false;
                result.relation_details.push_back("multi_column_weighted_aggregation message_source_routing status=REJECT failure=\"" + failure + "\"");
            }
            const bool alpha_message_ok = message_ok && verify_multi_column_alpha_hadamard(
                alpha_message_proof, gen, com_alpha, matrix_gen,
                com_message_edge_matrix, com_weighted_edge_matrix,
                alpha_message_label, &failure);
            if (message_ok && !alpha_message_ok) {
                weighted_all_ok = false;
                result.relation_details.push_back("multi_column_weighted_aggregation alpha_times_message status=REJECT failure=\"" + failure + "\"");
            }
            const bool aggregation_ok = alpha_message_ok && formal_multi_column_grouped_sum_relation_verify(
                aggregation_proof,
                matrix_gen, com_weighted_edge_matrix,
                matrix_gen, com_aggregated_matrix,
                gen, src, dst, aggregation_label, &failure);
            if (alpha_message_ok && !aggregation_ok) {
                weighted_all_ok = false;
                result.relation_details.push_back("multi_column_weighted_aggregation weighted_grouped_sum status=REJECT failure=\"" + failure + "\"");
            }
            if (message_ok && alpha_message_ok && aggregation_ok) {
                result.verified_feature_columns = result.available_feature_columns;
                result.proof_size_bytes += message_route_proof.serialized_size_bytes()
                    + alpha_message_proof.serialized_size_bytes()
                    + aggregation_proof.serialized_size_bytes();
                result.relation_details.push_back("multi_column_weighted_aggregation status=PASS feature_columns="
                    + std::to_string(result.verified_feature_columns)
                    + " matrix_commitment_block_size=" + std::to_string(matrix_block_size)
                    + " message_route_digest=" + message_route_proof.transcript_digest
                    + " alpha_message_digest=" + alpha_message_proof.transcript_digest
                    + " aggregation_digest=" + aggregation_proof.transcript_digest);
            }
            result.verify_ms += timer.elapsed_ms();
        }
    } else {
    for (std::size_t col = 0; col < result.feature_columns; ++col) {
        auto message_node_raw = column_scale16_raw(head.h_prime, graph.num_nodes, head.width, col, padded_node_count);
        auto message_edge_raw = route_vector_by_topology_raw(message_node_raw, graph, padded_edge_count, true);
        auto message_node_mont = mont_vector_from_raw(message_node_raw);
        auto message_edge_mont = mont_vector_from_raw(message_edge_raw);
        std::vector<Fr_t> weighted_edge_mont(padded_edge_count, Fr_host_zero());
        for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
            weighted_edge_mont[edge] = Fr_host_mul(alpha_mont[edge], message_edge_mont[edge]);
        }
        auto aggregated_mont = grouped_sum_by_dst_raw(weighted_edge_mont, graph, padded_node_count);

        FrTensor message_node_tensor(static_cast<uint>(message_node_mont.size()), message_node_mont.data());
        FrTensor message_edge_tensor(static_cast<uint>(message_edge_mont.size()), message_edge_mont.data());
        FrTensor weighted_tensor(static_cast<uint>(weighted_edge_mont.size()), weighted_edge_mont.data());
        FrTensor aggregated_tensor(static_cast<uint>(aggregated_mont.size()), aggregated_mont.data());
        ScopedTimer feature_commit_timer;
        const auto com_message_node = gen.commit(message_node_tensor);
        const auto com_message_edge = gen.commit(message_edge_tensor);
        const auto com_weighted = gen.commit(weighted_tensor);
        const auto com_aggregated = gen.commit(aggregated_tensor);
        cuda_sync("formal weighted aggregation commit " + spec.dataset + " col " + std::to_string(col));
        result.commitment_ms += feature_commit_timer.elapsed_ms();

        const std::string col_suffix = ":feature_col" + std::to_string(col);
        const std::string message_route_label = prefix + ":weighted_aggregation:message_source_routing" + col_suffix;
        const std::string alpha_message_label = prefix + ":alpha_multiplication:alpha_times_message" + col_suffix;
        const std::string aggregation_label = prefix + ":weighted_aggregation:weighted_by_dst" + col_suffix;
        SparseRoutingRelationProof message_route_proof;
        HadamardSumcheckProof alpha_message_proof;
        GroupedSumRelationProof aggregation_proof;
        {
            ScopedTimer timer;
            message_route_proof = formal_sparse_routing_relation_prove(
                message_node_tensor, message_edge_tensor, gen, com_message_node, gen, com_message_edge,
                src, dst, node_count_u32, edge_count_u32, "source", message_route_label);
            alpha_message_proof = formal_hadamard_relation_prove(
                alpha_tensor, message_edge_tensor, weighted_tensor, gen, com_alpha, com_message_edge, com_weighted, alpha_message_label);
            aggregation_proof = formal_grouped_sum_relation_prove(
                weighted_tensor, aggregated_tensor, gen, com_weighted, gen, com_aggregated,
                src, dst, node_count_u32, edge_count_u32, aggregation_label);
            result.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            const bool message_ok = formal_sparse_routing_relation_verify(
                message_route_proof, gen, com_message_node, gen, com_message_edge,
                src, dst, "source", message_route_label, &failure);
            if (!message_ok) {
                weighted_all_ok = false;
                result.relation_details.push_back("weighted_feature_col=" + std::to_string(col)
                    + " message_source_routing status=REJECT failure=\"" + failure + "\"");
            }
            const bool alpha_message_ok = message_ok && formal_hadamard_relation_verify(
                alpha_message_proof, gen, com_alpha, com_message_edge, com_weighted, alpha_message_label, &failure);
            if (message_ok && !alpha_message_ok) {
                weighted_all_ok = false;
                result.relation_details.push_back("weighted_feature_col=" + std::to_string(col)
                    + " alpha_times_message status=REJECT failure=\"" + failure + "\"");
            }
            const bool aggregation_ok = alpha_message_ok && formal_grouped_sum_relation_verify(
                aggregation_proof, gen, com_weighted, gen, com_aggregated, src, dst, aggregation_label, &failure);
            if (alpha_message_ok && !aggregation_ok) {
                weighted_all_ok = false;
                result.relation_details.push_back("weighted_feature_col=" + std::to_string(col)
                    + " weighted_grouped_sum status=REJECT failure=\"" + failure + "\"");
            }
            if (message_ok && alpha_message_ok && aggregation_ok) {
                ++result.verified_feature_columns;
                result.proof_size_bytes += message_route_proof.serialized_size_bytes()
                    + alpha_message_proof.serialized_size_bytes()
                    + aggregation_proof.serialized_size_bytes();
                result.relation_details.push_back("weighted_feature_col=" + std::to_string(col)
                    + " status=PASS message_route_digest=" + message_route_proof.transcript_digest
                    + " alpha_message_digest=" + alpha_message_proof.transcript_digest
                    + " aggregation_digest=" + aggregation_proof.transcript_digest);
            }
            result.verify_ms += timer.elapsed_ms();
        }
    }
    }
    result.weighted_aggregation_ok = weighted_all_ok
        && result.verified_feature_columns == result.feature_columns
        && result.feature_columns == result.available_feature_columns;
    const bool weighted_executed_ok = weighted_all_ok && result.verified_feature_columns == result.feature_columns;

    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    result.proof_path = run_dir / "formal_artifacts" / "zkdl_softmax_aggregation_relations.proof.txt";
    {
        std::ofstream out(result.proof_path);
        out << "ZKDLGATA\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "relation_scope=hidden_layer0_head0_softmax_denominator_inverse_alpha_weighted_aggregation\n";
        out << "node_count=" << graph.num_nodes << "\n";
        out << "edge_count=" << graph.edges.size() << "\n";
        out << "padded_node_count=" << padded_node_count << "\n";
        out << "padded_edge_count=" << padded_edge_count << "\n";
        out << "feature_columns=" << result.feature_columns << "\n";
        out << "available_feature_columns=" << result.available_feature_columns << "\n";
        out << "verified_feature_columns=" << result.verified_feature_columns << "\n";
        out << "commitment_block_size=" << result.commitment_block_size << "\n";
        out << "topology_hash=" << result.topology_hash << "\n";
        out << "denominator_grouped_sum_status=" << (result.denominator_ok ? "PASS" : "FAIL") << "\n";
        out << "inverse_status=" << (result.inverse_ok ? "PASS" : "FAIL") << "\n";
        out << "inverse_destination_routing_status=" << (result.inverse_routing_ok ? "PASS" : "FAIL") << "\n";
        out << "alpha_from_inverse_status=" << (result.alpha_from_inverse_ok ? "PASS" : "FAIL") << "\n";
        out << "alpha_times_denominator_status=" << (result.alpha_times_denominator_ok ? "PASS" : "FAIL") << "\n";
        out << "weighted_aggregation_status="
            << (result.weighted_aggregation_ok ? "PASS" : (weighted_executed_ok ? "PARTIAL_PASS" : "FAIL")) << "\n";
        out << "proof_size_bytes=" << result.proof_size_bytes << "\n";
        out << "relation_details=" << result.relation_details.size() << "\n";
        for (const auto& detail : result.relation_details) out << detail << "\n";
        out << "说明=本 proof artifact 只包含 relation metadata、证明大小与 transcript 摘要，不包含完整 witness、完整 tensor 或 debug dump。\n";
    }
    result.evidence_path = output_root / "evidence" / ("softmax_aggregation_relation_" + spec.dataset + ".json");
    {
        std::ofstream out(result.evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"relation_scope\": \"hidden_layer0_head0_softmax_denominator_inverse_alpha_weighted_aggregation\",\n";
        out << "  \"node_count\": \"" << graph.num_nodes << "\",\n";
        out << "  \"edge_count\": \"" << graph.edges.size() << "\",\n";
        out << "  \"feature_columns\": \"" << result.feature_columns << "\",\n";
        out << "  \"available_feature_columns\": \"" << result.available_feature_columns << "\",\n";
        out << "  \"verified_feature_columns\": \"" << result.verified_feature_columns << "\",\n";
        out << "  \"commitment_block_size\": \"" << result.commitment_block_size << "\",\n";
        out << "  \"topology_hash\": \"" << result.topology_hash << "\",\n";
        out << "  \"denominator_grouped_sum\": \"" << (result.denominator_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"inverse\": \"" << (result.inverse_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"inverse_destination_routing\": \"" << (result.inverse_routing_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"alpha_from_inverse\": \"" << (result.alpha_from_inverse_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"alpha_times_denominator\": \"" << (result.alpha_times_denominator_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"weighted_aggregation\": \""
            << (result.weighted_aggregation_ok ? "PASS" : (weighted_executed_ok ? "PARTIAL_PASS" : "FAIL")) << "\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(result.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(result.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(result.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << result.proof_size_bytes << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(result.proof_path.string()) << "\",\n";
        out << "  \"note\": \"该 evidence 证明 hidden layer0/head0 的 U 分母、inverse、alpha 与 weighted aggregation 的 zkDL relation-level 链条；ExpMap、max/range、ELU、output aggregation、rescale 与 rounding 仍未闭合，因此不写入 official latest。\"\n";
        out << "}\n";
    }
    {
        std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
        debug << "dataset=" << spec.dataset << " stage=formal_softmax_aggregation status="
              << (result.weighted_aggregation_ok ? "PASS" : "PARTIAL") << "\n";
        debug << "nodes=" << graph.num_nodes << " edges=" << graph.edges.size()
              << " topology_hash=" << result.topology_hash
              << " commitment_block_size=" << result.commitment_block_size
              << " feature_columns=" << result.feature_columns
              << " available_feature_columns=" << result.available_feature_columns
              << " verified_feature_columns=" << result.verified_feature_columns << "\n";
        debug << "commitment_ms=" << format_ms(result.commitment_ms)
              << " prove_ms=" << format_ms(result.prove_ms)
              << " verify_ms=" << format_ms(result.verify_ms)
              << " proof_size_bytes=" << result.proof_size_bytes << "\n";
    }
    {
        std::ofstream resume(run_dir / "RESUME_POINT.md", std::ios::app);
        resume << "\n## softmax/aggregation relation 继续点\n\n";
        resume << "- 本轮 relation: `denominator grouped sum`, `inverse Hadamard`, `alpha multiplication`, `weighted aggregation`\n";
        resume << "- 证明范围: hidden layer0/head0 full graph / full edge set，feature columns `"
               << result.verified_feature_columns << "/" << result.available_feature_columns
               << "`；本次实际运行 columns `" << result.feature_columns << "`\n";
        resume << "- commitment block size: `" << result.commitment_block_size << "`\n";
        resume << "- 结果: `" << (result.weighted_aggregation_ok ? "PASS" : "PARTIAL") << "`\n";
        resume << "- proof artifact: `" << result.proof_path.string() << "`\n";
        resume << "- evidence: `" << result.evidence_path.string() << "`\n";
        resume << "- Level A 状态: blocked；softmax max/ExpMap、LeakyReLU/ELU lookup-range、output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍未闭合。\n";
    }
    return result;
}

struct SparseGroupedGadgetRun
{
    bool source_ok = false;
    bool destination_ok = false;
    bool grouped_sum_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::uint64_t topology_hash = 0;
    std::size_t commitment_block_size = 1;
    std::string source_digest;
    std::string destination_digest;
    std::string grouped_sum_digest;
    fs::path proof_path;
    fs::path evidence_path;
};

SparseGroupedGadgetRun run_sparse_grouped_relations_with_trace(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const ForwardTraceHost& trace,
    const fs::path& output_root,
    std::size_t requested_commitment_block_size)
{
    if (trace.hidden_heads.empty() || trace.hidden_heads.front().empty()) {
        throw std::runtime_error("缺少 hidden layer0/head0 trace，无法生成 sparse routing evidence");
    }
    const auto& head = trace.hidden_heads.front().front();
    if (head.e_src.size() != graph.num_nodes || head.e_dst.size() != graph.num_nodes) {
        throw std::runtime_error("attention node trace 与 graph node_count 不一致");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_edge_count = next_power_of_two(std::max<std::size_t>(1, graph.edges.size()));
    if (padded_node_count > std::numeric_limits<uint>::max() || padded_edge_count > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("sparse routing 当前实现超过 uint 形状上限");
    }
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const uint node_count_u32 = static_cast<uint>(graph.num_nodes);
    const uint edge_count_u32 = static_cast<uint>(graph.edges.size());

    auto node_src_raw = encode_padded_vector_scale16_raw(head.e_src, graph.num_nodes, padded_node_count);
    auto node_dst_raw = encode_padded_vector_scale16_raw(head.e_dst, graph.num_nodes, padded_node_count);
    auto edge_src_raw = route_vector_by_topology_raw(node_src_raw, graph, padded_edge_count, true);
    auto edge_dst_raw = route_vector_by_topology_raw(node_dst_raw, graph, padded_edge_count, false);
    auto grouped_sum_raw = grouped_sum_by_dst_raw(edge_src_raw, graph, padded_node_count);

    FrTensor node_src_tensor(static_cast<uint>(node_src_raw.size()), node_src_raw.data());
    FrTensor node_dst_tensor(static_cast<uint>(node_dst_raw.size()), node_dst_raw.data());
    FrTensor edge_src_tensor(static_cast<uint>(edge_src_raw.size()), edge_src_raw.data());
    FrTensor edge_dst_tensor(static_cast<uint>(edge_dst_raw.size()), edge_dst_raw.data());
    FrTensor grouped_input_tensor(static_cast<uint>(edge_src_raw.size()), edge_src_raw.data());
    FrTensor grouped_output_tensor(static_cast<uint>(grouped_sum_raw.size()), grouped_sum_raw.data());
    node_src_tensor.mont();
    node_dst_tensor.mont();
    edge_src_tensor.mont();
    edge_dst_tensor.mont();
    grouped_input_tensor.mont();
    grouped_output_tensor.mont();

    SparseGroupedGadgetRun result;
    result.topology_hash = formal_sparse_topology_hash(src, dst, node_count_u32, edge_count_u32);
    result.commitment_block_size = formal_effective_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_edge_count);
    Commitment gen = make_formal_commitment_generators(result.commitment_block_size);
    ScopedTimer commit_timer;
    const auto com_node_src = gen.commit(node_src_tensor);
    const auto com_node_dst = gen.commit(node_dst_tensor);
    const auto com_edge_src = gen.commit(edge_src_tensor);
    const auto com_edge_dst = gen.commit(edge_dst_tensor);
    const auto com_grouped_input = gen.commit(grouped_input_tensor);
    const auto com_grouped_output = gen.commit(grouped_output_tensor);
    cuda_sync("formal sparse/grouped commit " + spec.dataset);
    result.commitment_ms = commit_timer.elapsed_ms();

    const std::string source_label = spec.dataset + ":hidden_layer0_head0:source_sparse_routing:full_edges";
    const std::string destination_label = spec.dataset + ":hidden_layer0_head0:destination_sparse_routing:full_edges";
    const std::string grouped_label = spec.dataset + ":hidden_layer0_head0:grouped_sum_by_dst:source_attention_edge_values";
    SparseRoutingRelationProof source_proof;
    SparseRoutingRelationProof destination_proof;
    GroupedSumRelationProof grouped_proof;
    {
        ScopedTimer timer;
        source_proof = formal_sparse_routing_relation_prove(
            node_src_tensor, edge_src_tensor, gen, com_node_src, gen, com_edge_src,
            src, dst, node_count_u32, edge_count_u32, "source", source_label);
        destination_proof = formal_sparse_routing_relation_prove(
            node_dst_tensor, edge_dst_tensor, gen, com_node_dst, gen, com_edge_dst,
            src, dst, node_count_u32, edge_count_u32, "destination", destination_label);
        grouped_proof = formal_grouped_sum_relation_prove(
            grouped_input_tensor, grouped_output_tensor, gen, com_grouped_input, gen, com_grouped_output,
            src, dst, node_count_u32, edge_count_u32, grouped_label);
        result.prove_ms = timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        result.source_ok = formal_sparse_routing_relation_verify(
            source_proof, gen, com_node_src, gen, com_edge_src, src, dst, "source", source_label, &failure);
        if (!result.source_ok) throw std::runtime_error("source sparse routing verifier rejected: " + failure);
        result.destination_ok = formal_sparse_routing_relation_verify(
            destination_proof, gen, com_node_dst, gen, com_edge_dst, src, dst, "destination", destination_label, &failure);
        if (!result.destination_ok) throw std::runtime_error("destination sparse routing verifier rejected: " + failure);
        result.grouped_sum_ok = formal_grouped_sum_relation_verify(
            grouped_proof, gen, com_grouped_input, gen, com_grouped_output, src, dst, grouped_label, &failure);
        if (!result.grouped_sum_ok) throw std::runtime_error("grouped sum verifier rejected: " + failure);
        result.verify_ms = timer.elapsed_ms();
    }

    result.proof_size_bytes = source_proof.serialized_size_bytes()
        + destination_proof.serialized_size_bytes()
        + grouped_proof.serialized_size_bytes();
    result.source_digest = source_proof.transcript_digest;
    result.destination_digest = destination_proof.transcript_digest;
    result.grouped_sum_digest = grouped_proof.transcript_digest;
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    result.proof_path = run_dir / "formal_artifacts" / "zkdl_sparse_grouped_relations.proof.txt";
    {
        std::ofstream out(result.proof_path);
        out << "ZKDLGATS\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "relation_scope=hidden_layer0_head0_sparse_routing_and_one_dim_grouped_sum\n";
        out << "node_count=" << graph.num_nodes << "\n";
        out << "edge_count=" << graph.edges.size() << "\n";
        out << "padded_node_count=" << padded_node_count << "\n";
        out << "padded_edge_count=" << padded_edge_count << "\n";
        out << "edge_range=0.." << graph.edges.size() << "\n";
        out << "commitment_block_size=" << result.commitment_block_size << "\n";
        out << "topology_hash=" << result.topology_hash << "\n";
        out << "source_sparse_routing_status=PASS\n";
        out << "destination_sparse_routing_status=PASS\n";
        out << "grouped_sum_status=PASS\n";
        out << "source_transcript_digest=" << result.source_digest << "\n";
        out << "destination_transcript_digest=" << result.destination_digest << "\n";
        out << "grouped_sum_transcript_digest=" << result.grouped_sum_digest << "\n";
        out << "proof_size_bytes=" << result.proof_size_bytes << "\n";
        out << "说明=本 proof artifact 只包含 relation metadata、承诺/opening/sumcheck 证明大小与 transcript 摘要，不包含完整 witness、完整 tensor 或 debug dump。\n";
    }
    result.evidence_path = output_root / "evidence" / ("sparse_grouped_relation_" + spec.dataset + ".json");
    {
        std::ofstream out(result.evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"relation_scope\": \"hidden_layer0_head0_sparse_routing_and_one_dim_grouped_sum\",\n";
        out << "  \"node_count\": \"" << graph.num_nodes << "\",\n";
        out << "  \"edge_count\": \"" << graph.edges.size() << "\",\n";
        out << "  \"edge_range\": \"0.." << graph.edges.size() << "\",\n";
        out << "  \"commitment_block_size\": \"" << result.commitment_block_size << "\",\n";
        out << "  \"topology_hash\": \"" << result.topology_hash << "\",\n";
        out << "  \"source_sparse_routing\": \"PASS\",\n";
        out << "  \"destination_sparse_routing\": \"PASS\",\n";
        out << "  \"grouped_sum_by_dst\": \"PASS\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(result.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(result.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(result.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << result.proof_size_bytes << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(result.proof_path.string()) << "\",\n";
        out << "  \"note\": \"一维 grouped sum 使用 hidden layer0 head0 的 source attention edge 值作为输入；它证明 grouped sum gadget 本身，不等同于 softmax denominator 或 weighted aggregation。\"\n";
        out << "}\n";
    }
    {
        std::ofstream debug(run_dir / "proof_debug.log");
        debug << "dataset=" << spec.dataset << " stage=formal_sparse_grouped status=PASS\n";
        debug << "nodes=" << graph.num_nodes << " edges=" << graph.edges.size()
              << " topology_hash=" << result.topology_hash
              << " commitment_block_size=" << result.commitment_block_size << "\n";
        debug << "source_sparse_routing_digest=" << result.source_digest << "\n";
        debug << "destination_sparse_routing_digest=" << result.destination_digest << "\n";
        debug << "grouped_sum_digest=" << result.grouped_sum_digest << "\n";
        debug << "commitment_ms=" << format_ms(result.commitment_ms)
              << " prove_ms=" << format_ms(result.prove_ms)
              << " verify_ms=" << format_ms(result.verify_ms)
              << " proof_size_bytes=" << result.proof_size_bytes << "\n";
    }
    {
        std::ofstream resume(run_dir / "RESUME_POINT.md");
        resume << "# zkDL GAT 数据集继续点\n\n";
        resume << "- 数据集: `" << spec.dataset << "`\n";
        resume << "- 本轮 relation: `SparseSourceRoutingRelationProof`, `SparseDestinationRoutingRelationProof`, `GroupedSumRelationProof`\n";
        resume << "- 证明范围: full graph / full edge set，edge range `0.." << graph.edges.size() << "`\n";
        resume << "- commitment block size: `" << result.commitment_block_size << "`\n";
        resume << "- 结果: `PASS`\n";
        resume << "- proof artifact: `" << result.proof_path.string() << "`\n";
        resume << "- evidence: `" << result.evidence_path.string() << "`\n";
        resume << "- Level A 状态: blocked；softmax ExpMap/inverse、LeakyReLU/ELU lookup-range、weighted/output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍未闭合。\n";
    }
    return result;
}
