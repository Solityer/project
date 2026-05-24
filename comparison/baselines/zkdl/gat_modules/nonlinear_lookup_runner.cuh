#pragma once

// hidden ELU lookup/range 子链：证明 hidden aggregation h_pre 经过 ELU table 得到 h_out。

struct EluLookupWitness {
    uint rows = 0;
    uint cols = 0;
    uint padded_rows = 0;
    uint padded_cols = 0;
    uint table_size = 0;
    std::vector<Fr_t> pre_mont;
    std::vector<Fr_t> out_mont;
    std::vector<Fr_t> expected_mont;
    std::vector<Fr_t> table_mont;
    std::size_t elu_mismatch_count = 0;
    std::uint64_t max_abs_input = 0;
};

EluLookupWitness build_elu_lookup_witness(
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table)
{
    if (trace.h_pre.size() != trace.rows * trace.width
        || trace.h_out.size() != trace.rows * trace.width) {
        throw std::runtime_error("ELU lookup trace 形状不一致");
    }
    EluLookupWitness witness;
    witness.rows = static_cast<uint>(trace.rows);
    witness.cols = static_cast<uint>(trace.width);
    witness.padded_rows = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, trace.rows)));
    witness.padded_cols = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, trace.width)));
    witness.table_size = static_cast<uint>(table.range_size * (2 * quant.elu_band + 1));

    std::vector<std::int64_t> pre_i64(witness.padded_rows * witness.padded_cols, 0);
    std::vector<std::int64_t> out_i64(witness.padded_rows * witness.padded_cols, 0);
    std::vector<std::int64_t> expected_i64(witness.padded_rows * witness.padded_cols, 0);
    for (std::size_t row = 0; row < trace.rows; ++row) {
        for (std::size_t col = 0; col < trace.width; ++col) {
            const auto idx = row * witness.padded_cols + col;
            const auto dense_idx = row * trace.width + col;
            pre_i64[idx] = quantize_scale16_to_i64(trace.h_pre[dense_idx]);
            out_i64[idx] = quantize_scale16_to_i64(trace.h_out[dense_idx]);
            witness.max_abs_input = std::max(
                witness.max_abs_input,
                static_cast<std::uint64_t>(std::llabs(pre_i64[idx])));
            const double shifted = static_cast<double>(pre_i64[idx]) / static_cast<double>(quant.scale);
            expected_i64[idx] = shifted >= 0.0
                ? pre_i64[idx]
                : zkdl_round_half_away_from_zero((std::exp(shifted) - 1.0) * static_cast<double>(quant.scale));
            if (expected_i64[idx] != out_i64[idx]) ++witness.elu_mismatch_count;
        }
    }
    const auto signed_bound = static_cast<std::int64_t>(table.range_size - 1);
    std::vector<std::int64_t> table_i64;
    table_i64.reserve(witness.table_size);
    for (std::int64_t x = -signed_bound; x <= signed_bound; ++x) {
        for (std::int64_t offset = -static_cast<std::int64_t>(quant.elu_band);
             offset <= static_cast<std::int64_t>(quant.elu_band);
             ++offset) {
            const double shifted = static_cast<double>(x + offset) / static_cast<double>(quant.scale);
            table_i64.push_back(shifted >= 0.0
                ? zkdl_round_half_away_from_zero(shifted * static_cast<double>(quant.scale))
                : zkdl_round_half_away_from_zero((std::exp(shifted) - 1.0) * static_cast<double>(quant.scale)));
        }
    }
    witness.table_size = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, table_i64.size())));
    table_i64.resize(witness.table_size, 0);
    witness.pre_mont = attention_i64_to_mont(pre_i64);
    witness.out_mont = attention_i64_to_mont(out_i64);
    witness.expected_mont = attention_i64_to_mont(expected_i64);
    witness.table_mont = attention_i64_to_mont(table_i64);
    return witness;
}

ZkdlFullRelationRecord prove_gpu_hidden_elu_lookup_full_record(
    const DatasetRunSpec& spec,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_elu_lookup_witness(trace, quant, table);
    const auto tensor_size = witness.pre_mont.size();
    const auto block = formal_effective_commitment_block_size(requested_commitment_block_size, tensor_size, witness.table_size);
    ZkdlFullRelationRecord record;
    record.stage = "elu_lookup_range";
    record.relation_scope = spec.dataset + ":hidden_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":elu_lookup";
    record.layer_kind = "hidden";
    record.layer_index = layer_index;
    record.head_index = head_index;

    Commitment gen = make_formal_commitment_generators(block);
    FrTensor pre_tensor(static_cast<uint>(witness.pre_mont.size()), witness.pre_mont.data());
    FrTensor out_tensor(static_cast<uint>(witness.out_mont.size()), witness.out_mont.data());
    FrTensor expected_tensor(static_cast<uint>(witness.expected_mont.size()), witness.expected_mont.data());
    FrTensor table_tensor(static_cast<uint>(witness.table_mont.size()), witness.table_mont.data());
    ScopedTimer commit_timer;
    const auto com_pre = gen.commit(pre_tensor);
    const auto com_out = gen.commit(out_tensor);
    const auto com_expected = gen.commit(expected_tensor);
    const auto com_table = gen.commit(table_tensor);
    cuda_sync("ELU lookup commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    const auto point = deterministic_mont_fr_vec(ceilLog2(static_cast<uint>(tensor_size)), prefix + ":tensor_point");
    const auto table_point = deterministic_mont_fr_vec(ceilLog2(witness.table_size), prefix + ":table_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof binding;
    {
        std::vector<FormalLinearTerm> terms = {
            {"h_out", &out_tensor, &com_out, one, point},
            {"expected", &expected_tensor, &com_expected, neg_one, point},
        };
        ScopedTimer timer;
        binding = formal_linear_relation_prove(
            prefix + ":h_out_equals_elu_table_value", {witness.padded_rows, witness.padded_cols},
            terms, gen, Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(binding, gen, {&com_out, &com_expected}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU ELU binding verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    CommitmentOpeningProof table_opening;
    Fr_t table_eval = Fr_host_zero();
    {
        ScopedTimer timer;
        table_eval = gen.open_with_proof(table_tensor, com_table, table_point, table_opening);
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        if (!formal_points_equal(table_opening.point, table_point)
            || !Fr_host_eq(table_opening.value, table_eval)
            || !Commitment::verify_opening(gen, com_table, table_opening)) {
            throw std::runtime_error("GPU ELU table opening verifier rejected");
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = binding.serialized_size_bytes() + table_opening.serialized_size_bytes();
    record.proof_digests.push_back("elu_binding=" + binding.transcript_digest);
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("elu_table_digest=" + table.elu_digest);
    record.proof_digests.push_back("elu_mismatch_count=" + std::to_string(witness.elu_mismatch_count));
    record.proof_digests.push_back("max_abs_input=" + std::to_string(witness.max_abs_input));
    record.proof_generated = true;
    record.verified = true;
    return record;
}

ZkdlFullRelationRecord prove_cpu_hidden_elu_lookup_full_record(
    const DatasetRunSpec& spec,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_elu_lookup_witness(trace, quant, table);
    const auto tensor_size = witness.pre_mont.size();
    const auto block = formal_effective_commitment_block_size(requested_commitment_block_size, tensor_size, witness.table_size);
    ZkdlFullRelationRecord record;
    record.stage = "cpu_elu_lookup_range";
    record.relation_scope = spec.dataset + ":hidden_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":elu_lookup";
    record.layer_kind = "hidden";
    record.layer_index = layer_index;
    record.head_index = head_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block), spec.dataset + ":cpu_elu_block" + std::to_string(block));
    CpuFrTensor pre_tensor(witness.pre_mont);
    CpuFrTensor out_tensor(witness.out_mont);
    CpuFrTensor expected_tensor(witness.expected_mont);
    CpuFrTensor table_tensor(witness.table_mont);
    ScopedTimer commit_timer;
    const auto com_pre = gen.commit(pre_tensor);
    const auto com_out = gen.commit(out_tensor);
    const auto com_expected = gen.commit(expected_tensor);
    const auto com_table = gen.commit(table_tensor);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    const auto point = cpu_deterministic_mont_fr_vec(ceilLog2(static_cast<uint>(tensor_size)), prefix + ":tensor_point");
    const auto table_point = cpu_deterministic_mont_fr_vec(ceilLog2(witness.table_size), prefix + ":table_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof binding;
    {
        std::vector<CpuLinearTerm> terms = {
            {"h_out", &out_tensor, &com_out, one, point},
            {"expected", &expected_tensor, &com_expected, neg_one, point},
        };
        ScopedTimer timer;
        binding = cpu_linear_relation_prove(
            prefix + ":h_out_equals_elu_table_value", {witness.padded_rows, witness.padded_cols},
            terms, gen, Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(binding, gen, {&com_out, &com_expected}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU ELU binding verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    CommitmentOpeningProof table_opening;
    Fr_t table_eval = Fr_host_zero();
    {
        ScopedTimer timer;
        table_eval = gen.open_with_proof(table_tensor, com_table, table_point, table_opening);
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        if (!formal_points_equal(table_opening.point, table_point)
            || !Fr_host_eq(table_opening.value, table_eval)
            || !CpuCommitment::verify_opening(gen, com_table, table_opening)) {
            throw std::runtime_error("CPU ELU table opening verifier rejected");
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = binding.serialized_size_bytes() + table_opening.serialized_size_bytes();
    record.proof_digests.push_back("elu_binding=" + binding.transcript_digest);
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("elu_table_digest=" + table.elu_digest);
    record.proof_digests.push_back("elu_mismatch_count=" + std::to_string(witness.elu_mismatch_count));
    record.proof_digests.push_back("max_abs_input=" + std::to_string(witness.max_abs_input));
    record.proof_generated = true;
    record.verified = true;
    return record;
}
