#pragma once

// full proof object 的 output-head average / public logits binding 子关系。
// 该文件只生成 official full-chain proof record：验证者从 public logits 本地计算 MLE，
// 并检查所有 output head 在共同尺度上的算术平均，不读取 witness。

// 按协议的 N x C 行优先展平规则编码 output head 或 public logits。
// 输入是 CPU/GPU forward trace 中的 double 值；输出是 scale16 后的 Fr montgomery 向量。
std::vector<Fr_t> encode_output_matrix_mont(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    return mont_vector_from_raw(matrix_scale16_raw(values, rows, cols, padded_rows, padded_cols));
}

// GPU full-chain 的 output-head average/public logits 绑定。
// 该函数承诺每个 output head 的 h_out，不把 public logits 作为 witness；
// proof_size 与三段 timing 会进入 full_proof_manifest 的 official 统计。
ZkdlFullRelationRecord prove_gpu_output_logits_binding_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    std::size_t requested_commitment_block_size)
{
    if (trace.output_heads.empty()) {
        throw std::runtime_error("GPU full output binding 缺少 output head trace");
    }
    if (trace.logits.size() != graph.num_nodes * model.C) {
        throw std::runtime_error("GPU full output binding public logits 形状不一致");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_class_count = next_power_of_two(std::max<std::size_t>(1, model.C));
    const std::size_t total_size = padded_node_count * padded_class_count;
    if (total_size > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("GPU full output binding tensor 超出 uint 范围");
    }

    ZkdlFullRelationRecord record;
    record.stage = "output_head_average_public_logits_binding";
    record.relation_scope = spec.dataset + ":output_layer:all_heads:public_logits";
    record.layer_kind = "output";

    const std::size_t block_size = formal_effective_matrix_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_class_count);
    Commitment gen = make_formal_commitment_generators(block_size);
    std::vector<std::unique_ptr<FrTensor>> head_tensors;
    std::vector<std::unique_ptr<G1TensorJacobian>> head_commitments;
    std::vector<FormalLinearTerm> terms;
    std::vector<const G1TensorJacobian*> verify_commitments;
    head_tensors.reserve(trace.output_heads.size());
    head_commitments.reserve(trace.output_heads.size());

    const auto point = deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(total_size)),
        spec.dataset + ":full_output_binding:public_logits_point");
    const Fr_t head_average_coeff = fr_inverse_uint_mont(static_cast<std::uint32_t>(trace.output_heads.size()));

    for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
        if (trace.output_heads[head].h_out.size() != graph.num_nodes * model.C) {
            throw std::runtime_error("GPU full output binding output head 形状不一致");
        }
        auto encoded = encode_output_matrix_mont(
            trace.output_heads[head].h_out,
            graph.num_nodes,
            model.C,
            padded_node_count,
            padded_class_count);
        head_tensors.emplace_back(std::make_unique<FrTensor>(static_cast<uint>(encoded.size()), encoded.data()));
        ScopedTimer commit_timer;
        head_commitments.emplace_back(std::make_unique<G1TensorJacobian>(gen.commit(*head_tensors.back())));
        cuda_sync("full output binding commit " + spec.dataset + " head " + std::to_string(head));
        record.commitment_ms += commit_timer.elapsed_ms();
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

    const std::string label = spec.dataset + ":full_output_binding:head_average_equals_public_logits";
    LinearRelationProof proof;
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
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(proof, gen, verify_commitments, public_logits_eval, &failure)) {
            throw std::runtime_error("GPU full output binding verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = proof.serialized_size_bytes();
    record.proof_digests.push_back("head_average_public_logits=" + proof.transcript_digest);
    record.proof_generated = true;
    record.verified = true;
    return record;
}

// CPU full-chain 的 output-head average/public logits 绑定。
// 语义与 GPU 版本一致，只是使用 CPU commitment/opening 后端，保证 CPU/GPU 分表互不借用。
ZkdlFullRelationRecord prove_cpu_output_logits_binding_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    std::size_t requested_commitment_block_size)
{
    if (trace.output_heads.empty()) {
        throw std::runtime_error("CPU full output binding 缺少 output head trace");
    }
    if (trace.logits.size() != graph.num_nodes * model.C) {
        throw std::runtime_error("CPU full output binding public logits 形状不一致");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_class_count = next_power_of_two(std::max<std::size_t>(1, model.C));
    const std::size_t total_size = padded_node_count * padded_class_count;
    if (total_size > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("CPU full output binding tensor 超出 uint 范围");
    }

    ZkdlFullRelationRecord record;
    record.stage = "cpu_output_head_average_public_logits_binding";
    record.relation_scope = spec.dataset + ":output_layer:all_heads:public_logits";
    record.layer_kind = "output";

    const std::size_t block_size = formal_effective_matrix_commitment_block_size(
        requested_commitment_block_size, padded_node_count, padded_class_count);
    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block_size),
        spec.dataset + ":cpu_full_output_binding:matrix_gen");
    std::vector<std::unique_ptr<CpuFrTensor>> head_tensors;
    std::vector<std::unique_ptr<CpuG1Tensor>> head_commitments;
    std::vector<CpuLinearTerm> terms;
    std::vector<const CpuG1Tensor*> verify_commitments;
    head_tensors.reserve(trace.output_heads.size());
    head_commitments.reserve(trace.output_heads.size());

    const auto point = cpu_deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(total_size)),
        spec.dataset + ":cpu_full_output_binding:public_logits_point");
    const Fr_t head_average_coeff = fr_inverse_uint_mont(static_cast<std::uint32_t>(trace.output_heads.size()));

    for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
        if (trace.output_heads[head].h_out.size() != graph.num_nodes * model.C) {
            throw std::runtime_error("CPU full output binding output head 形状不一致");
        }
        auto encoded = mont_vector_from_raw(matrix_scale16_raw(
            trace.output_heads[head].h_out,
            graph.num_nodes,
            model.C,
            padded_node_count,
            padded_class_count));
        head_tensors.emplace_back(std::make_unique<CpuFrTensor>(std::move(encoded)));
        ScopedTimer commit_timer;
        head_commitments.emplace_back(std::make_unique<CpuG1Tensor>(gen.commit(*head_tensors.back())));
        record.commitment_ms += commit_timer.elapsed_ms();
        terms.push_back(CpuLinearTerm{
            "output_head" + std::to_string(head),
            head_tensors.back().get(),
            head_commitments.back().get(),
            head_average_coeff,
            point,
        });
        verify_commitments.push_back(head_commitments.back().get());
    }

    auto logits_encoded = mont_vector_from_raw(matrix_scale16_raw(
        trace.logits, graph.num_nodes, model.C, padded_node_count, padded_class_count));
    CpuFrTensor logits_tensor(std::move(logits_encoded));
    const Fr_t public_logits_eval = formal_host_mle_vector(logits_tensor.values, point.begin(), point.end());

    const std::string label = spec.dataset + ":cpu_full_output_binding:head_average_equals_public_logits";
    LinearRelationProof proof;
    {
        ScopedTimer timer;
        proof = cpu_linear_relation_prove(
            label,
            {static_cast<uint>(padded_node_count), static_cast<uint>(padded_class_count)},
            terms,
            gen,
            fr_neg_one_mont(),
            public_logits_eval,
            Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(proof, gen, verify_commitments, public_logits_eval, &failure)) {
            throw std::runtime_error("CPU full output binding verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = proof.serialized_size_bytes();
    record.proof_digests.push_back("head_average_public_logits=" + proof.transcript_digest);
    record.proof_generated = true;
    record.verified = true;
    return record;
}
