#pragma once

// CPU full-chain runner 使用 CPU commitment/opening 后端复现与 GPU 相同的 official proof scope。
// 它覆盖 hidden/output 全部 head、lookup/range、rescale/rounding、softmax、concat 与 public logits binding。

// CPU attention softmax/weighted aggregation 子链。
// 输入为单个 head 的 forward trace；输出 relation record 会进入 full proof manifest，
// 其中 commitment/prove/verify 时间和 proof_size 都计入 official 四项。
ZkdlFullRelationRecord prove_cpu_attention_softmax_chain(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& head,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    if (head.u.size() != graph.edges.size() || head.sum.size() != graph.num_nodes) {
        throw std::runtime_error("CPU attention head trace 与 graph 形状不一致");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_edge_count = next_power_of_two(std::max<std::size_t>(1, graph.edges.size()));
    if (padded_node_count > std::numeric_limits<uint>::max() || padded_edge_count > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("CPU attention softmax proof 形状超过 uint 上限");
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
        if (denom_u64[node] == 0) throw std::runtime_error("CPU attention denominator 为 0");
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
    record.stage = "cpu_attention_softmax_weighted_aggregation";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + "_head" + std::to_string(head_index);
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    const std::size_t vector_block = formal_effective_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_edge_count);
    CpuCommitment gen = CpuCommitment::deterministic(static_cast<uint>(vector_block), record.relation_scope + ":cpu_vector_gen");

    CpuFrTensor u_tensor(mont_vector_from_raw(u_raw));
    CpuFrTensor denom_tensor(mont_vector_from_raw(denom_raw));
    CpuFrTensor denom_edge_tensor(denom_edge_mont);
    CpuFrTensor inv_node_tensor(inv_node_mont);
    CpuFrTensor inv_edge_tensor(inv_edge_mont);
    CpuFrTensor valid_one_tensor(valid_one);
    CpuFrTensor alpha_tensor(alpha_mont);

    ScopedTimer commit_timer;
    const auto com_u = gen.commit(u_tensor);
    const auto com_denom = gen.commit(denom_tensor);
    const auto com_denom_edge = gen.commit(denom_edge_tensor);
    const auto com_inv_node = gen.commit(inv_node_tensor);
    const auto com_inv_edge = gen.commit(inv_edge_tensor);
    const auto com_valid_one = gen.commit(valid_one_tensor);
    const auto com_alpha = gen.commit(alpha_tensor);
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
        denom_proof = cpu_grouped_sum_relation_prove(
            u_tensor, denom_tensor, gen, com_u, gen, com_denom,
            src, dst, node_count_u32, edge_count_u32, denom_label);
        inverse_proof = cpu_hadamard_relation_prove(
            denom_tensor, inv_node_tensor, valid_one_tensor, gen, com_denom, com_inv_node, com_valid_one, inverse_label);
        inv_route_proof = cpu_sparse_routing_relation_prove(
            inv_node_tensor, inv_edge_tensor, gen, com_inv_node, gen, com_inv_edge,
            src, dst, node_count_u32, edge_count_u32, "destination", inv_route_label);
        alpha_from_inv_proof = cpu_hadamard_relation_prove(
            u_tensor, inv_edge_tensor, alpha_tensor, gen, com_u, com_inv_edge, com_alpha, alpha_from_inv_label);
        alpha_times_denom_proof = cpu_hadamard_relation_prove(
            alpha_tensor, denom_edge_tensor, u_tensor, gen, com_alpha, com_denom_edge, com_u, alpha_times_denom_label);
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_grouped_sum_relation_verify(denom_proof, gen, com_u, gen, com_denom, src, dst, denom_label, &failure)) {
            throw std::runtime_error("CPU denominator verifier rejected: " + failure);
        }
        if (!cpu_hadamard_relation_verify(inverse_proof, gen, com_denom, com_inv_node, com_valid_one, inverse_label, &failure)) {
            throw std::runtime_error("CPU inverse verifier rejected: " + failure);
        }
        if (!cpu_sparse_routing_relation_verify(inv_route_proof, gen, com_inv_node, gen, com_inv_edge, src, dst, "destination", inv_route_label, &failure)) {
            throw std::runtime_error("CPU inverse routing verifier rejected: " + failure);
        }
        if (!cpu_hadamard_relation_verify(alpha_from_inv_proof, gen, com_u, com_inv_edge, com_alpha, alpha_from_inv_label, &failure)) {
            throw std::runtime_error("CPU alpha verifier rejected: " + failure);
        }
        if (!cpu_hadamard_relation_verify(alpha_times_denom_proof, gen, com_alpha, com_denom_edge, com_u, alpha_times_denom_label, &failure)) {
            throw std::runtime_error("CPU alpha denominator verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = denom_proof.serialized_size_bytes()
        + inverse_proof.serialized_size_bytes()
        + inv_route_proof.serialized_size_bytes()
        + alpha_from_inv_proof.serialized_size_bytes()
        + alpha_times_denom_proof.serialized_size_bytes();
    record.proof_digests.push_back("denominator=" + denom_proof.transcript_digest);
    record.proof_digests.push_back("inverse=" + inverse_proof.transcript_digest);
    record.proof_digests.push_back("inverse_route=" + inv_route_proof.transcript_digest);
    record.proof_digests.push_back("alpha_from_inverse=" + alpha_from_inv_proof.transcript_digest);
    record.proof_digests.push_back("alpha_times_denominator=" + alpha_times_denom_proof.transcript_digest);

    if (head.h_prime.size() != graph.num_nodes * head.width) {
        throw std::runtime_error("CPU attention weighted aggregation trace 与 graph 形状不一致");
    }
    const std::size_t padded_feature_columns = next_power_of_two(std::max<std::size_t>(1, head.width));
    if (padded_feature_columns > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("CPU attention feature column count 超过 uint 上限");
    }
    const std::size_t column_chunk_limit = attention_softmax_column_chunk_limit(head.width);
    for (std::size_t col_begin = 0; col_begin < head.width; col_begin += column_chunk_limit) {
        const std::size_t chunk_columns = std::min(column_chunk_limit, head.width - col_begin);
        const std::size_t padded_chunk_columns = next_power_of_two(std::max<std::size_t>(1, chunk_columns));
        const uint chunk_column_count_u32 = static_cast<uint>(chunk_columns);
        const uint padded_chunk_column_count_u32 = static_cast<uint>(padded_chunk_columns);
        const std::string chunk_suffix = ":cols" + std::to_string(col_begin)
            + "_" + std::to_string(col_begin + chunk_columns);
        std::cout << "full_chain_runner dataset=" << spec.dataset
                  << " device=cpu stage=prove_" << layer_kind
                  << "_attention_chunk layer=" << layer_index
                  << " head=" << head_index
                  << " cols=" << col_begin << "_" << (col_begin + chunk_columns)
                  << "\n";
        const std::size_t matrix_block = formal_effective_matrix_commitment_block_size(
            vector_block,
            std::min(padded_node_count, padded_edge_count),
            padded_chunk_columns);
        CpuCommitment matrix_gen = CpuCommitment::deterministic(
            static_cast<uint>(matrix_block),
            record.relation_scope + ":cpu_matrix_gen" + chunk_suffix);
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
        CpuFrTensor message_node_matrix_tensor(message_node_matrix_mont);
        CpuFrTensor message_edge_matrix_tensor(message_edge_matrix_mont);
        CpuFrTensor weighted_edge_matrix_tensor(weighted_edge_matrix_mont);
        CpuFrTensor aggregated_matrix_tensor(aggregated_matrix_mont);

        ScopedTimer matrix_commit_timer;
        const auto com_message_node_matrix = matrix_gen.commit(message_node_matrix_tensor);
        const auto com_message_edge_matrix = matrix_gen.commit(message_edge_matrix_tensor);
        const auto com_weighted_edge_matrix = matrix_gen.commit(weighted_edge_matrix_tensor);
        const auto com_aggregated_matrix = matrix_gen.commit(aggregated_matrix_tensor);
        record.commitment_ms += matrix_commit_timer.elapsed_ms();

        const std::string message_route_label = prefix + ":weighted_message_source_routing" + chunk_suffix;
        const std::string alpha_message_label = prefix + ":alpha_times_message" + chunk_suffix;
        const std::string aggregation_label = prefix + ":weighted_aggregation_by_dst" + chunk_suffix;
        MultiColumnSparseRoutingGadgetProof message_route_proof;
        MultiColumnHadamardGadgetProof alpha_message_proof;
        MultiColumnGroupedSumRelationProof aggregation_proof;
        {
            ScopedTimer timer;
            message_route_proof = cpu_prove_multi_column_sparse_routing(
                message_node_matrix_tensor, message_edge_matrix_tensor,
                matrix_gen, com_message_node_matrix, com_message_edge_matrix,
                gen, src, dst, node_count_u32, edge_count_u32,
                chunk_column_count_u32, padded_chunk_column_count_u32,
                "source", message_route_label);
            alpha_message_proof = cpu_prove_multi_column_alpha_hadamard(
                alpha_tensor, message_edge_matrix_tensor, weighted_edge_matrix_tensor,
                gen, com_alpha, matrix_gen, com_message_edge_matrix, com_weighted_edge_matrix,
                edge_count_u32, chunk_column_count_u32, padded_chunk_column_count_u32, alpha_message_label);
            aggregation_proof = cpu_multi_column_grouped_sum_relation_prove(
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
            if (!cpu_verify_multi_column_sparse_routing(
                    message_route_proof, matrix_gen, com_message_node_matrix, com_message_edge_matrix,
                    gen, src, dst, "source", message_route_label, &failure)) {
                throw std::runtime_error("CPU weighted message routing verifier rejected: " + failure);
            }
            if (!cpu_verify_multi_column_alpha_hadamard(
                    alpha_message_proof, gen, com_alpha, matrix_gen,
                    com_message_edge_matrix, com_weighted_edge_matrix,
                    alpha_message_label, &failure)) {
                throw std::runtime_error("CPU alpha-message verifier rejected: " + failure);
            }
            if (!cpu_multi_column_grouped_sum_relation_verify(
                    aggregation_proof, matrix_gen, com_weighted_edge_matrix,
                    matrix_gen, com_aggregated_matrix, gen, src, dst, aggregation_label, &failure)) {
                throw std::runtime_error("CPU weighted aggregation verifier rejected: " + failure);
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

// CPU official 路径：读取 full dataset/cache/checkpoint，生成 CPU forward trace，
// 遍历所有 hidden/output relation，并写 proof object、full proof manifest 和 verifier manifest。
ZkdlFullProofObject run_cpu_full_chain_proof(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    std::size_t requested_commitment_block_size)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset, "cpu");
    ensure_dir(run_dir / "formal_artifacts");
    auto log_line = [&](const std::string& line) {
        std::cout << line << "\n";
    };

    log_line("full_chain_runner dataset=" + spec.dataset + " device=cpu stage=load_dataset");
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    const auto graph = normalize_graph(dataset, spec);
    if (graph.num_nodes != dataset.num_nodes) {
        throw std::runtime_error("CPU full-chain runner 只允许 full dataset/full graph");
    }
    log_line("full_chain_runner dataset=" + spec.dataset + " device=cpu stage=load_model");
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);

    log_line("full_chain_runner dataset=" + spec.dataset + " device=cpu stage=cpu_forward_trace");
    ScopedTimer forward_timer;
    const auto trace = run_gat_forward_cpu_trace(graph, model);
    log_line("full_chain_runner dataset=" + spec.dataset
        + " device=cpu forward_ms=" + format_ms(forward_timer.elapsed_ms())
        + " hidden_layers=" + std::to_string(trace.hidden_heads.size())
        + " output_heads=" + std::to_string(trace.output_heads.size()));

    zkdl_require_trace_range_bits_within_spec(graph, trace, spec);
    const auto metadata = build_zkdl_gat_public_metadata(spec, dataset, graph, model);
    const auto quant = build_zkdl_gat_quant_contract(spec, model);
    const auto table = build_zkdl_gat_table_context(spec, model);
    write_zkdl_gat_protocol_context_json(run_dir / "formal_artifacts" / "zkdl_full_chain_public_metadata.json", metadata, quant, table);

    ZkdlFullProofObject proof;
    proof.dataset = spec.dataset;
    proof.device = "cpu";
    proof.metadata_id = metadata.metadata_id;
    proof.vk_static_id = table.context_id;
    proof.vk_model_id = model.model_param_id;
    proof.quant_contract_id = quant.contract_id;
    proof.is_full_dataset = true;
    proof.proof_object_path = run_dir / "formal_artifacts" / "zkdl_full_proof_object.txt";
    proof.proof_manifest_path = run_dir / "full_proof_manifest.json";
    proof.verifier_manifest_path = run_dir / "full_verifier_manifest.json";

    for (std::size_t layer = 0; layer < trace.hidden_heads.size(); ++layer) {
        for (std::size_t head = 0; head < trace.hidden_heads[layer].size(); ++head) {
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_projection layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_projection_matmul_full_record(
                    spec,
                    graph,
                    trace.hidden_layer_inputs.at(layer),
                    trace.hidden_layer_input_dims.at(layer),
                    require_hidden_head_params(model, layer, head),
                    "hidden",
                    layer,
                    head,
                    requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_rescale_rounding layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_rescale_rounding_full_record(
                    spec,
                    graph,
                    trace.hidden_layer_inputs.at(layer),
                    trace.hidden_layer_input_dims.at(layer),
                    require_hidden_head_params(model, layer, head),
                    trace.hidden_heads[layer][head],
                    quant,
                    "hidden",
                    layer,
                    head,
                    requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_attention_dot_lrelu layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_attention_dot_lrelu_full_record(
                    spec,
                    graph,
                    require_hidden_head_params(model, layer, head),
                    trace.hidden_heads[layer][head],
                    quant,
                    table,
                    "hidden",
                    layer,
                    head,
                    requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_target_max_shift layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_target_max_shift_full_record(
                    spec,
                    graph,
                    trace.hidden_heads[layer][head],
                    table,
                    "hidden",
                    layer,
                    head,
                    requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_expmap_lookup layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_expmap_lookup_full_record(
                    spec,
                    graph,
                    trace.hidden_heads[layer][head],
                    quant,
                    table,
                    "hidden",
                    layer,
                    head,
                    requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_attention layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_attention_softmax_chain(
                    spec, graph, trace.hidden_heads[layer][head], "hidden", layer, head, requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=cpu stage=prove_hidden_elu_lookup layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_cpu_hidden_elu_lookup_full_record(
                    spec,
                    trace.hidden_heads[layer][head],
                    quant,
                    table,
                    layer,
                    head,
                    requested_commitment_block_size));
        }
        const bool final_hidden_layer = layer + 1 == trace.hidden_heads.size();
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_hidden_concat layer=" + std::to_string(layer));
        append_full_relation_record(
            proof,
            prove_cpu_hidden_concat_binding_full_record(
                spec,
                graph,
                trace.hidden_heads[layer],
                final_hidden_layer ? trace.output_layer_input : trace.hidden_layer_inputs.at(layer + 1),
                final_hidden_layer ? trace.output_layer_input_dim : trace.hidden_layer_input_dims.at(layer + 1),
                layer,
                requested_commitment_block_size));
    }
    for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_projection head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_projection_matmul_full_record(
                spec,
                graph,
                trace.output_layer_input,
                trace.output_layer_input_dim,
                model.output_heads.at(head),
                "output",
                0,
                head,
                requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_rescale_rounding head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_rescale_rounding_full_record(
                spec,
                graph,
                trace.output_layer_input,
                trace.output_layer_input_dim,
                model.output_heads.at(head),
                trace.output_heads[head],
                quant,
                "output",
                0,
                head,
                requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_attention_dot_lrelu head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_attention_dot_lrelu_full_record(
                spec,
                graph,
                model.output_heads.at(head),
                trace.output_heads[head],
                quant,
                table,
                "output",
                0,
                head,
                requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_target_max_shift head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_target_max_shift_full_record(
                spec,
                graph,
                trace.output_heads[head],
                table,
                "output",
                0,
                head,
                requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_expmap_lookup head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_expmap_lookup_full_record(
                spec,
                graph,
                trace.output_heads[head],
                quant,
                table,
                "output",
                0,
                head,
                requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_attention head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_attention_softmax_chain(
                spec, graph, trace.output_heads[head], "output", 0, head, requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=cpu stage=prove_output_attention_bias head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_cpu_output_attention_bias_full_record(
                spec,
                graph,
                model.output_heads.at(head),
                trace.output_heads[head],
                head,
                requested_commitment_block_size));
    }
    log_line("full_chain_runner dataset=" + spec.dataset
        + " device=cpu stage=prove_output_head_average_public_logits");
    append_full_relation_record(
        proof,
        prove_cpu_output_logits_binding_full_record(
            spec, graph, model, trace, requested_commitment_block_size));

    {
        ScopedTimer verifier_timer;
        proof.missing_relations = zkdl_verify_full_proof_manifest_clean(
            proof, spec, model, trace, metadata, quant, table, "cpu");
        proof.verify_ms += verifier_timer.elapsed_ms();
    }
    finalize_full_proof_status(proof);
    write_zkdl_full_proof_object(proof);
    log_line("full_chain_runner dataset=" + spec.dataset
        + " device=cpu proof_generated=" + std::string(proof.proof_generated ? "true" : "false")
        + " verified=" + std::string(proof.verified ? "true" : "false")
        + " relation_records=" + std::to_string(proof.records.size())
        + " missing_relations=" + std::to_string(proof.missing_relations.size())
        + " proof_size_bytes=" + std::to_string(proof.proof_size_bytes));
    return proof;
}
