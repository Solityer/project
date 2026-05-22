#pragma once

// 输出绑定 runner：在真实 GAT full graph trace 上证明 output heads 的平均值
// 与 public logits 的公开 MLE 取值一致。该 relation 仍是 partial_formal，
// 不覆盖 output attention 前序链、bias 前序链、rescale/rounding 或完整 verifier。

struct GpuOutputBindingRun
{
    bool output_average_ok = false;
    bool public_logits_binding_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::size_t output_head_count = 0;
    std::size_t class_count = 0;
    std::size_t commitment_block_size = 1;
    std::string transcript_digest;
    fs::path proof_path;
    fs::path evidence_path;
};

std::vector<Fr_t> encode_output_matrix_mont(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    return mont_vector_from_raw(matrix_scale16_raw(values, rows, cols, padded_rows, padded_cols));
}

GpuOutputBindingRun run_output_binding_relation_with_trace(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    const fs::path& output_root,
    std::size_t requested_commitment_block_size)
{
    if (trace.output_heads.empty()) {
        throw std::runtime_error("缺少 output head trace，无法生成 output binding relation");
    }
    if (trace.logits.size() != graph.num_nodes * model.C) {
        throw std::runtime_error("public logits trace 与 graph/model 形状不一致");
    }
    for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
        if (trace.output_heads[head].h_out.size() != graph.num_nodes * model.C) {
            throw std::runtime_error("output head h_out 形状不一致");
        }
    }

    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_class_count = next_power_of_two(std::max<std::size_t>(1, model.C));
    if (padded_node_count > std::numeric_limits<uint>::max()
        || padded_class_count > std::numeric_limits<uint>::max()
        || padded_node_count > std::numeric_limits<std::size_t>::max() / padded_class_count) {
        throw std::runtime_error("output binding tensor shape 超出当前 zkDL uint/size_t 范围");
    }
    const std::size_t total_size = padded_node_count * padded_class_count;
    if (total_size > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("output binding tensor 总长度超出当前 FrTensor uint 范围");
    }

    GpuOutputBindingRun result;
    result.output_head_count = trace.output_heads.size();
    result.class_count = model.C;
    result.commitment_block_size = formal_effective_matrix_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_class_count);
    Commitment gen = make_formal_commitment_generators(result.commitment_block_size);

    std::vector<std::unique_ptr<FrTensor>> head_tensors;
    std::vector<std::unique_ptr<G1TensorJacobian>> head_commitments;
    head_tensors.reserve(trace.output_heads.size());
    head_commitments.reserve(trace.output_heads.size());
    std::vector<FormalLinearTerm> terms;
    std::vector<const G1TensorJacobian*> verify_commitments;
    const auto point = deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(total_size)),
        spec.dataset + ":output_binding:public_logits_point");
    const Fr_t head_average_coeff = fr_inverse_uint_mont(static_cast<std::uint32_t>(trace.output_heads.size()));

    for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
        auto encoded = encode_output_matrix_mont(
            trace.output_heads[head].h_out,
            graph.num_nodes,
            model.C,
            padded_node_count,
            padded_class_count);
        head_tensors.emplace_back(std::make_unique<FrTensor>(static_cast<uint>(encoded.size()), encoded.data()));
        {
            ScopedTimer timer;
            head_commitments.emplace_back(std::make_unique<G1TensorJacobian>(gen.commit(*head_tensors.back())));
            cuda_sync("formal output binding commit " + spec.dataset + " head " + std::to_string(head));
            result.commitment_ms += timer.elapsed_ms();
        }
        terms.push_back(FormalLinearTerm{
            "output_head" + std::to_string(head),
            head_tensors.back().get(),
            head_commitments.back().get(),
            head_average_coeff,
            point,
        });
        verify_commitments.push_back(head_commitments.back().get());
    }

    auto logits_encoded = encode_output_matrix_mont(
        trace.logits, graph.num_nodes, model.C, padded_node_count, padded_class_count);
    FrTensor logits_tensor(static_cast<uint>(logits_encoded.size()), logits_encoded.data());
    const Fr_t public_logits_eval = logits_tensor(point);

    LinearRelationProof proof;
    const std::string label = spec.dataset + ":output_binding:head_average_equals_public_logits";
    {
        ScopedTimer timer;
        proof = formal_linear_relation_prove(
            label,
            {static_cast<uint>(padded_node_count), static_cast<uint>(padded_class_count)},
            terms,
            gen,
            fr_neg_one_mont(),
            public_logits_eval,
            Fr_host_zero());
        result.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        const bool ok = formal_linear_relation_verify(
            proof, gen, verify_commitments, public_logits_eval, &failure);
        if (!ok) {
            throw std::runtime_error("output binding linear verifier rejected: " + failure);
        }
        result.verify_ms += timer.elapsed_ms();
    }

    result.output_average_ok = true;
    result.public_logits_binding_ok = true;
    result.proof_size_bytes = proof.serialized_size_bytes();
    result.transcript_digest = proof.transcript_digest;

    GpuOutputBindingArtifactPayload payload;
    payload.node_count = graph.num_nodes;
    payload.class_count = model.C;
    payload.padded_node_count = padded_node_count;
    payload.padded_class_count = padded_class_count;
    payload.output_head_count = result.output_head_count;
    payload.commitment_block_size = result.commitment_block_size;
    payload.output_average_ok = result.output_average_ok;
    payload.public_logits_binding_ok = result.public_logits_binding_ok;
    payload.commitment_ms = result.commitment_ms;
    payload.prove_ms = result.prove_ms;
    payload.verify_ms = result.verify_ms;
    payload.proof_size_bytes = result.proof_size_bytes;
    payload.transcript_digest = result.transcript_digest;
    const auto paths = write_gpu_output_binding_artifact(spec, output_root, payload);
    result.proof_path = paths.first;
    result.evidence_path = paths.second;
    write_gpu_output_binding_debug_resume(spec, output_root, payload, result.proof_path, result.evidence_path);
    return result;
}

GpuOutputBindingRun run_formal_output_binding_gadget(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    std::size_t requested_commitment_block_size)
{
    std::cout << "formal_output_binding dataset=" << spec.dataset << " stage=load_dataset\n";
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    std::cout << "formal_output_binding dataset=" << spec.dataset << " stage=normalize_graph\n";
    const auto graph = normalize_graph(dataset, spec);
    if (graph.num_nodes != dataset.num_nodes) {
        throw std::runtime_error("formal output binding 只允许 full dataset/full graph");
    }
    std::cout << "formal_output_binding dataset=" << spec.dataset << " stage=load_model\n";
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    std::cout << "formal_output_binding dataset=" << spec.dataset << " stage=gpu_forward_trace\n";
    ForwardMetrics metrics;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &metrics, &trace);
    std::cout << "formal_output_binding dataset=" << spec.dataset
              << " forward_ms=" << format_ms(metrics.total_ms)
              << " output_heads=" << trace.output_heads.size()
              << " classes=" << model.C << "\n";
    auto result = run_output_binding_relation_with_trace(
        spec, graph, model, trace, output_root, requested_commitment_block_size);
    std::cout << "formal_output_binding dataset=" << spec.dataset
              << " status=PASS"
              << " commitment_ms=" << format_ms(result.commitment_ms)
              << " prove_ms=" << format_ms(result.prove_ms)
              << " verify_ms=" << format_ms(result.verify_ms)
              << " proof_size_bytes=" << result.proof_size_bytes
              << " proof=" << result.proof_path.string() << "\n";
    return result;
}
