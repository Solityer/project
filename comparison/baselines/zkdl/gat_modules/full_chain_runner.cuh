#pragma once

// full-chain runner 负责遍历 hidden/output 所有 attention head，并写 full proof / verifier manifest。
// 该 runner 不负责 official 导出；只有 proof.generated && verifier.verified && missing_relations 为空时 exporter 才能接收。

ZkdlFullProofObject run_gpu_full_chain_proof(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    std::size_t requested_commitment_block_size)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset, "gpu");
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    std::ofstream log(run_dir / "full_chain_runner.log");
    auto log_line = [&](const std::string& line) {
        log << line << "\n";
        log.flush();
        std::cout << line << "\n";
    };

    log_line("full_chain_runner dataset=" + spec.dataset + " device=gpu stage=load_dataset");
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    const auto graph = normalize_graph(dataset, spec);
    if (graph.num_nodes != dataset.num_nodes) {
        throw std::runtime_error("full-chain runner 只允许 full dataset/full graph");
    }
    log_line("full_chain_runner dataset=" + spec.dataset + " device=gpu stage=load_model");
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);

    log_line("full_chain_runner dataset=" + spec.dataset + " device=gpu stage=gpu_forward_trace");
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    log_line("full_chain_runner dataset=" + spec.dataset
        + " device=gpu forward_ms=" + format_ms(forward.total_ms)
        + " hidden_layers=" + std::to_string(trace.hidden_heads.size())
        + " output_heads=" + std::to_string(trace.output_heads.size()));

    const auto effective_range_bits = zkdl_required_trace_range_bits(graph, trace, spec.range_bits);
    const auto effective_spec = zkdl_spec_with_range_bits(spec, effective_range_bits);
    if (effective_range_bits != spec.range_bits) {
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=gpu stage=table_context_range_bits value=" + std::to_string(effective_range_bits));
    }
    const auto metadata = build_zkdl_gat_public_metadata(effective_spec, dataset, graph, model);
    const auto quant = build_zkdl_gat_quant_contract(effective_spec, model);
    const auto table = build_zkdl_gat_table_context(effective_spec, model);
    write_zkdl_gat_protocol_context_json(run_dir / "formal_artifacts" / "zkdl_full_chain_public_metadata.json", metadata, quant, table);

    ZkdlFullProofObject proof;
    proof.dataset = spec.dataset;
    proof.device = "gpu";
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
                + " device=gpu stage=prove_hidden_projection layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_projection_matmul_full_record(
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
                + " device=gpu stage=prove_hidden_rescale_rounding layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_rescale_rounding_full_record(
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
                + " device=gpu stage=prove_hidden_attention_dot_lrelu layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_attention_dot_lrelu_full_record(
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
                + " device=gpu stage=prove_hidden_target_max_shift layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_target_max_shift_full_record(
                    spec,
                    graph,
                    trace.hidden_heads[layer][head],
                    table,
                    "hidden",
                    layer,
                    head,
                    requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=gpu stage=prove_hidden_expmap_lookup layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_expmap_lookup_full_record(
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
                + " device=gpu stage=prove_hidden_attention layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_attention_softmax_chain(
                    spec, graph, trace.hidden_heads[layer][head], "hidden", layer, head, requested_commitment_block_size));
            log_line("full_chain_runner dataset=" + spec.dataset
                + " device=gpu stage=prove_hidden_elu_lookup layer=" + std::to_string(layer)
                + " head=" + std::to_string(head));
            append_full_relation_record(
                proof,
                prove_gpu_hidden_elu_lookup_full_record(
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
            + " device=gpu stage=prove_hidden_concat layer=" + std::to_string(layer));
        append_full_relation_record(
            proof,
            prove_gpu_hidden_concat_binding_full_record(
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
            + " device=gpu stage=prove_output_projection head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_projection_matmul_full_record(
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
            + " device=gpu stage=prove_output_rescale_rounding head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_rescale_rounding_full_record(
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
            + " device=gpu stage=prove_output_attention_dot_lrelu head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_attention_dot_lrelu_full_record(
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
            + " device=gpu stage=prove_output_target_max_shift head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_target_max_shift_full_record(
                spec,
                graph,
                trace.output_heads[head],
                table,
                "output",
                0,
                head,
                requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=gpu stage=prove_output_expmap_lookup head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_expmap_lookup_full_record(
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
            + " device=gpu stage=prove_output_attention head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_attention_softmax_chain(
                spec, graph, trace.output_heads[head], "output", 0, head, requested_commitment_block_size));
        log_line("full_chain_runner dataset=" + spec.dataset
            + " device=gpu stage=prove_output_attention_bias head=" + std::to_string(head));
        append_full_relation_record(
            proof,
            prove_gpu_output_attention_bias_full_record(
                spec,
                graph,
                model.output_heads.at(head),
                trace.output_heads[head],
                head,
                requested_commitment_block_size));
    }
    log_line("full_chain_runner dataset=" + spec.dataset
        + " device=gpu stage=prove_output_head_average_public_logits");
    append_full_relation_record(
        proof,
        prove_gpu_output_logits_binding_full_record(
            spec, graph, model, trace, requested_commitment_block_size));

    {
        ScopedTimer verifier_timer;
        proof.missing_relations = zkdl_verify_full_proof_manifest_clean(
            proof, spec, model, trace, metadata, quant, table, "gpu");
        proof.verify_ms += verifier_timer.elapsed_ms();
    }
    finalize_full_proof_status(proof);
    write_zkdl_full_proof_object(proof);
    {
        std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
        debug << "dataset=" << spec.dataset
              << " device=gpu stage=full_chain_runner proof_generated=" << (proof.proof_generated ? "true" : "false")
              << " verified=" << (proof.verified ? "true" : "false")
              << " relation_records=" << proof.records.size()
              << " missing_relations=" << proof.missing_relations.size()
              << " proof_size_bytes=" << proof.proof_size_bytes
              << " full_proof_manifest=" << proof.proof_manifest_path.string()
              << " full_verifier_manifest=" << proof.verifier_manifest_path.string()
              << "\n";
    }
    log_line("full_chain_runner dataset=" + spec.dataset
        + " device=gpu proof_generated=" + std::string(proof.proof_generated ? "true" : "false")
        + " verified=" + std::string(proof.verified ? "true" : "false")
        + " relation_records=" + std::to_string(proof.records.size())
        + " missing_relations=" + std::to_string(proof.missing_relations.size())
        + " proof_size_bytes=" + std::to_string(proof.proof_size_bytes));
    return proof;
}
