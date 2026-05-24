#pragma once

// output attention 的 bias/尺度对齐子链：证明 output aggregation h_pre 加 output bias 后得到 h_out。

struct OutputBiasWitness {
    uint rows = 0;
    uint cols = 0;
    uint padded_rows = 0;
    uint padded_cols = 0;
    std::vector<Fr_t> pre_mont;
    std::vector<Fr_t> bias_mont;
    std::vector<Fr_t> out_mont;
    std::size_t bias_mismatch_count = 0;
};

// 构造输出头偏置见证，证明协议定义的 h_out = h_pre + b_out。
// 输出层没有 ELU；该函数只处理 bias/尺度已对齐后的线性绑定。
OutputBiasWitness build_output_bias_witness(
    const LocalGraph& graph,
    const HeadParams& head,
    const HeadTraceHost& trace)
{
    if (trace.h_pre.size() != graph.num_nodes * head.output_dim
        || trace.h_out.size() != graph.num_nodes * head.output_dim
        || head.output_bias.size() != head.output_dim) {
        throw std::runtime_error("output bias witness 形状不一致");
    }
    OutputBiasWitness witness;
    witness.rows = static_cast<uint>(graph.num_nodes);
    witness.cols = static_cast<uint>(head.output_dim);
    witness.padded_rows = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.num_nodes)));
    witness.padded_cols = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head.output_dim)));
    std::vector<std::int64_t> pre_i64(witness.padded_rows * witness.padded_cols, 0);
    std::vector<std::int64_t> bias_i64(witness.padded_rows * witness.padded_cols, 0);
    std::vector<std::int64_t> out_i64(witness.padded_rows * witness.padded_cols, 0);
    for (std::size_t row = 0; row < graph.num_nodes; ++row) {
        for (std::size_t col = 0; col < head.output_dim; ++col) {
            const auto idx = row * witness.padded_cols + col;
            const auto dense_idx = row * head.output_dim + col;
            pre_i64[idx] = quantize_scale16_to_i64(trace.h_pre[dense_idx]);
            bias_i64[idx] = quantize_scale16_to_i64(head.output_bias[col]);
            out_i64[idx] = quantize_scale16_to_i64(trace.h_out[dense_idx]);
            const auto expected = pre_i64[idx] + bias_i64[idx];
            if (expected != out_i64[idx]) ++witness.bias_mismatch_count;
        }
    }
    witness.pre_mont = attention_i64_to_mont(pre_i64);
    witness.bias_mont = attention_i64_to_mont(bias_i64);
    witness.out_mont = attention_i64_to_mont(out_i64);
    return witness;
}

// GPU full-chain output bias relation：计入 output attention head 的 official proof_size/timing。
ZkdlFullRelationRecord prove_gpu_output_attention_bias_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadParams& head,
    const HeadTraceHost& trace,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_output_bias_witness(graph, head, trace);
    const auto size = witness.pre_mont.size();
    const auto block = rescale_common_commitment_block_size(requested_commitment_block_size, size);
    ZkdlFullRelationRecord record;
    record.stage = "output_attention_bias_scale_binding";
    record.relation_scope = spec.dataset + ":output_layer0:head" + std::to_string(head_index) + ":attention_bias_binding";
    record.layer_kind = "output";
    record.layer_index = 0;
    record.head_index = head_index;

    Commitment gen = make_formal_commitment_generators(block);
    FrTensor pre_tensor(static_cast<uint>(witness.pre_mont.size()), witness.pre_mont.data());
    FrTensor bias_tensor(static_cast<uint>(witness.bias_mont.size()), witness.bias_mont.data());
    FrTensor out_tensor(static_cast<uint>(witness.out_mont.size()), witness.out_mont.data());
    ScopedTimer commit_timer;
    const auto com_pre = gen.commit(pre_tensor);
    const auto com_bias = gen.commit(bias_tensor);
    const auto com_out = gen.commit(out_tensor);
    cuda_sync("output attention bias commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const auto point = deterministic_mont_fr_vec(ceilLog2(static_cast<uint>(size)), record.relation_scope + ":point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof proof;
    {
        std::vector<FormalLinearTerm> terms = {
            {"h_out", &out_tensor, &com_out, one, point},
            {"h_pre", &pre_tensor, &com_pre, neg_one, point},
            {"bias", &bias_tensor, &com_bias, neg_one, point},
        };
        ScopedTimer timer;
        proof = formal_linear_relation_prove(
            record.relation_scope + ":h_pre_plus_bias_equals_h_out",
            {witness.padded_rows, witness.padded_cols}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(proof, gen, {&com_out, &com_pre, &com_bias}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU output attention bias verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = proof.serialized_size_bytes();
    record.proof_digests.push_back("bias_binding=" + proof.transcript_digest);
    record.proof_digests.push_back("bias_mismatch_count=" + std::to_string(witness.bias_mismatch_count));
    record.proof_generated = true;
    record.verified = true;
    return record;
}

// CPU full-chain output bias relation：与 GPU 使用相同 public logits 前序语义。
ZkdlFullRelationRecord prove_cpu_output_attention_bias_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadParams& head,
    const HeadTraceHost& trace,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_output_bias_witness(graph, head, trace);
    const auto size = witness.pre_mont.size();
    const auto block = rescale_common_commitment_block_size(requested_commitment_block_size, size);
    ZkdlFullRelationRecord record;
    record.stage = "cpu_output_attention_bias_scale_binding";
    record.relation_scope = spec.dataset + ":output_layer0:head" + std::to_string(head_index) + ":attention_bias_binding";
    record.layer_kind = "output";
    record.layer_index = 0;
    record.head_index = head_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block), spec.dataset + ":cpu_output_bias_block" + std::to_string(block));
    CpuFrTensor pre_tensor(witness.pre_mont);
    CpuFrTensor bias_tensor(witness.bias_mont);
    CpuFrTensor out_tensor(witness.out_mont);
    ScopedTimer commit_timer;
    const auto com_pre = gen.commit(pre_tensor);
    const auto com_bias = gen.commit(bias_tensor);
    const auto com_out = gen.commit(out_tensor);
    record.commitment_ms += commit_timer.elapsed_ms();

    const auto point = cpu_deterministic_mont_fr_vec(ceilLog2(static_cast<uint>(size)), record.relation_scope + ":point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof proof;
    {
        std::vector<CpuLinearTerm> terms = {
            {"h_out", &out_tensor, &com_out, one, point},
            {"h_pre", &pre_tensor, &com_pre, neg_one, point},
            {"bias", &bias_tensor, &com_bias, neg_one, point},
        };
        ScopedTimer timer;
        proof = cpu_linear_relation_prove(
            record.relation_scope + ":h_pre_plus_bias_equals_h_out",
            {witness.padded_rows, witness.padded_cols}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(proof, gen, {&com_out, &com_pre, &com_bias}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU output attention bias verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = proof.serialized_size_bytes();
    record.proof_digests.push_back("bias_binding=" + proof.transcript_digest);
    record.proof_digests.push_back("bias_mismatch_count=" + std::to_string(witness.bias_mismatch_count));
    record.proof_generated = true;
    record.verified = true;
    return record;
}
