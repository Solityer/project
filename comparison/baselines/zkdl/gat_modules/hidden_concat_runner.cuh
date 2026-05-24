#pragma once

std::vector<Fr_t> boolean_offset_point_bits(std::size_t offset, std::size_t bit_count)
{
    std::vector<Fr_t> out(bit_count, Fr_host_zero());
    for (std::size_t bit = 0; bit < bit_count; ++bit) {
        out[bit] = ((offset >> bit) & 1ULL) != 0ULL ? Fr_host_one_mont() : Fr_host_zero();
    }
    return out;
}

std::vector<Fr_t> concat_points_for_hidden_head(
    const std::vector<Fr_t>& row_point,
    const std::vector<Fr_t>& local_col_point,
    std::size_t head_index,
    std::size_t padded_head_width,
    std::size_t padded_concat_width)
{
    const std::size_t local_bits = local_col_point.size();
    const std::size_t concat_bits = ceilLog2(static_cast<uint>(padded_concat_width));
    if ((std::size_t{1} << local_bits) != padded_head_width || (std::size_t{1} << concat_bits) != padded_concat_width) {
        throw std::runtime_error("hidden concat point padding 不是 power-of-two");
    }
    const std::size_t offset = head_index * padded_head_width;
    if (offset + padded_head_width > padded_concat_width) {
        throw std::runtime_error("hidden concat head offset 超出 concat 宽度");
    }
    auto fixed = boolean_offset_point_bits(offset, concat_bits);
    for (std::size_t bit = 0; bit < local_bits; ++bit) {
        fixed[bit] = local_col_point[bit];
    }
    std::vector<Fr_t> out;
    out.reserve(concat_bits + row_point.size());
    out.insert(out.end(), fixed.begin(), fixed.end());
    out.insert(out.end(), row_point.begin(), row_point.end());
    return out;
}

std::vector<Fr_t> hidden_head_point(
    const std::vector<Fr_t>& row_point,
    const std::vector<Fr_t>& local_col_point)
{
    std::vector<Fr_t> out;
    out.reserve(local_col_point.size() + row_point.size());
    out.insert(out.end(), local_col_point.begin(), local_col_point.end());
    out.insert(out.end(), row_point.begin(), row_point.end());
    return out;
}

ZkdlFullRelationRecord prove_gpu_hidden_concat_binding_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const std::vector<HeadTraceHost>& heads,
    const std::vector<double>& concat_matrix,
    std::size_t concat_dim,
    std::size_t layer_index,
    std::size_t requested_commitment_block_size)
{
    if (heads.empty()) throw std::runtime_error("GPU hidden concat binding 缺少 hidden heads");
    const std::size_t rows = graph.num_nodes;
    const std::size_t head_width = heads.front().width;
    for (const auto& head : heads) {
        if (head.rows != rows || head.width != head_width || head.h_out.size() != rows * head_width) {
            throw std::runtime_error("GPU hidden concat binding head 形状不一致");
        }
    }
    if (concat_dim != heads.size() * head_width || concat_matrix.size() != rows * concat_dim) {
        throw std::runtime_error("GPU hidden concat binding concat 形状不一致");
    }
    const std::size_t padded_rows = next_power_of_two(std::max<std::size_t>(1, rows));
    const std::size_t padded_head_width = next_power_of_two(std::max<std::size_t>(1, head_width));
    const std::size_t padded_concat_width = next_power_of_two(std::max<std::size_t>(1, concat_dim));
    const std::size_t block_size = formal_effective_matrix_commitment_block_size(
        requested_commitment_block_size, padded_rows, std::max(padded_head_width, padded_concat_width));

    ZkdlFullRelationRecord record;
    record.stage = "hidden_concat_binding";
    record.relation_scope = spec.dataset + ":hidden_layer" + std::to_string(layer_index) + ":concat";
    record.layer_kind = "hidden";
    record.layer_index = layer_index;

    Commitment gen = make_formal_commitment_generators(block_size);
    auto concat_encoded = mont_vector_from_raw(matrix_scale16_raw(
        concat_matrix, rows, concat_dim, padded_rows, padded_concat_width));
    FrTensor concat_tensor(static_cast<uint>(concat_encoded.size()), concat_encoded.data());
    ScopedTimer concat_commit_timer;
    const auto concat_commitment = gen.commit(concat_tensor);
    cuda_sync("hidden concat binding commit concat " + record.relation_scope);
    record.commitment_ms += concat_commit_timer.elapsed_ms();

    const auto row_point = deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(padded_rows)),
        record.relation_scope + ":row_point");
    const auto local_col_point = deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(padded_head_width)),
        record.relation_scope + ":local_col_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();

    for (std::size_t head_index = 0; head_index < heads.size(); ++head_index) {
        auto head_encoded = mont_vector_from_raw(matrix_scale16_raw(
            heads[head_index].h_out, rows, head_width, padded_rows, padded_head_width));
        FrTensor head_tensor(static_cast<uint>(head_encoded.size()), head_encoded.data());
        ScopedTimer commit_timer;
        const auto head_commitment = gen.commit(head_tensor);
        cuda_sync("hidden concat binding commit head " + std::to_string(head_index));
        record.commitment_ms += commit_timer.elapsed_ms();

        const auto head_eval_point = hidden_head_point(row_point, local_col_point);
        const auto concat_eval_point = concat_points_for_hidden_head(
            row_point, local_col_point, head_index, padded_head_width, padded_concat_width);
        const std::string label = record.relation_scope + ":head" + std::to_string(head_index);
        std::vector<FormalLinearTerm> terms = {
            FormalLinearTerm{"head_h_out", &head_tensor, &head_commitment, one, head_eval_point},
            FormalLinearTerm{"concat_slice", &concat_tensor, &concat_commitment, neg_one, concat_eval_point},
        };
        LinearRelationProof proof;
        {
            ScopedTimer timer;
            proof = formal_linear_relation_prove(
                label,
                {static_cast<uint>(padded_rows), static_cast<uint>(padded_concat_width)},
                terms,
                gen,
                Fr_host_zero(),
                Fr_host_zero(),
                Fr_host_zero());
            record.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            std::vector<const G1TensorJacobian*> commitments = {&head_commitment, &concat_commitment};
            if (!formal_linear_relation_verify(proof, gen, commitments, Fr_host_zero(), &failure)) {
                throw std::runtime_error("GPU hidden concat binding verifier rejected: " + failure);
            }
            record.verify_ms += timer.elapsed_ms();
        }
        record.proof_size_bytes += proof.serialized_size_bytes();
        record.proof_digests.push_back("head" + std::to_string(head_index) + "=" + proof.transcript_digest);
    }
    record.proof_generated = true;
    record.verified = true;
    return record;
}

ZkdlFullRelationRecord prove_cpu_hidden_concat_binding_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const std::vector<HeadTraceHost>& heads,
    const std::vector<double>& concat_matrix,
    std::size_t concat_dim,
    std::size_t layer_index,
    std::size_t requested_commitment_block_size)
{
    if (heads.empty()) throw std::runtime_error("CPU hidden concat binding 缺少 hidden heads");
    const std::size_t rows = graph.num_nodes;
    const std::size_t head_width = heads.front().width;
    for (const auto& head : heads) {
        if (head.rows != rows || head.width != head_width || head.h_out.size() != rows * head_width) {
            throw std::runtime_error("CPU hidden concat binding head 形状不一致");
        }
    }
    if (concat_dim != heads.size() * head_width || concat_matrix.size() != rows * concat_dim) {
        throw std::runtime_error("CPU hidden concat binding concat 形状不一致");
    }
    const std::size_t padded_rows = next_power_of_two(std::max<std::size_t>(1, rows));
    const std::size_t padded_head_width = next_power_of_two(std::max<std::size_t>(1, head_width));
    const std::size_t padded_concat_width = next_power_of_two(std::max<std::size_t>(1, concat_dim));
    const std::size_t block_size = formal_effective_matrix_commitment_block_size(
        requested_commitment_block_size, padded_rows, std::max(padded_head_width, padded_concat_width));

    ZkdlFullRelationRecord record;
    record.stage = "cpu_hidden_concat_binding";
    record.relation_scope = spec.dataset + ":hidden_layer" + std::to_string(layer_index) + ":concat";
    record.layer_kind = "hidden";
    record.layer_index = layer_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block_size),
        spec.dataset + ":cpu_hidden_concat_block" + std::to_string(block_size));
    auto concat_encoded = mont_vector_from_raw(matrix_scale16_raw(
        concat_matrix, rows, concat_dim, padded_rows, padded_concat_width));
    CpuFrTensor concat_tensor(std::move(concat_encoded));
    ScopedTimer concat_commit_timer;
    const auto concat_commitment = gen.commit(concat_tensor);
    record.commitment_ms += concat_commit_timer.elapsed_ms();

    const auto row_point = cpu_deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(padded_rows)),
        record.relation_scope + ":cpu_row_point");
    const auto local_col_point = cpu_deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(padded_head_width)),
        record.relation_scope + ":cpu_local_col_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();

    for (std::size_t head_index = 0; head_index < heads.size(); ++head_index) {
        auto head_encoded = mont_vector_from_raw(matrix_scale16_raw(
            heads[head_index].h_out, rows, head_width, padded_rows, padded_head_width));
        CpuFrTensor head_tensor(std::move(head_encoded));
        ScopedTimer commit_timer;
        const auto head_commitment = gen.commit(head_tensor);
        record.commitment_ms += commit_timer.elapsed_ms();

        const auto head_eval_point = hidden_head_point(row_point, local_col_point);
        const auto concat_eval_point = concat_points_for_hidden_head(
            row_point, local_col_point, head_index, padded_head_width, padded_concat_width);
        const std::string label = record.relation_scope + ":head" + std::to_string(head_index);
        std::vector<CpuLinearTerm> terms = {
            CpuLinearTerm{"head_h_out", &head_tensor, &head_commitment, one, head_eval_point},
            CpuLinearTerm{"concat_slice", &concat_tensor, &concat_commitment, neg_one, concat_eval_point},
        };
        LinearRelationProof proof;
        {
            ScopedTimer timer;
            proof = cpu_linear_relation_prove(
                label,
                {static_cast<uint>(padded_rows), static_cast<uint>(padded_concat_width)},
                terms,
                gen,
                Fr_host_zero(),
                Fr_host_zero(),
                Fr_host_zero());
            record.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            std::vector<const CpuG1Tensor*> commitments = {&head_commitment, &concat_commitment};
            if (!cpu_linear_relation_verify(proof, gen, commitments, Fr_host_zero(), &failure)) {
                throw std::runtime_error("CPU hidden concat binding verifier rejected: " + failure);
            }
            record.verify_ms += timer.elapsed_ms();
        }
        record.proof_size_bytes += proof.serialized_size_bytes();
        record.proof_digests.push_back("head" + std::to_string(head_index) + "=" + proof.transcript_digest);
    }
    record.proof_generated = true;
    record.verified = true;
    return record;
}
