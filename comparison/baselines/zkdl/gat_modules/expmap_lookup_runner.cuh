#pragma once

// ExpMap lookup/range 子链：将 target max/shift 产生的 delta=m-z 绑定到 u=exp(-delta/scale) 的 GAT table 语义。
// proof object 记录 table commitment/opening、u 与 table-evaluated expected tensor 的绑定，以及 range/table context 摘要。

struct ExpMapLookupWitness {
    uint edge_count = 0;
    uint padded_edges = 0;
    uint table_size = 0;
    std::vector<Fr_t> delta_mont;
    std::vector<Fr_t> u_mont;
    std::vector<Fr_t> expected_mont;
    std::vector<Fr_t> table_mont;
    std::size_t exp_mismatch_count = 0;
    std::uint64_t max_delta = 0;
    std::uint64_t max_u = 0;
};

ExpMapLookupWitness build_expmap_lookup_witness(
    const LocalGraph& graph,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table)
{
    if (trace.z.size() != graph.edges.size()
        || trace.m.size() != graph.num_nodes
        || trace.u.size() != graph.edges.size()) {
        throw std::runtime_error("ExpMap lookup trace 形状不一致");
    }
    if (table.range_size == 0 || table.range_size > static_cast<std::size_t>(std::numeric_limits<uint>::max())) {
        throw std::runtime_error("ExpMap table range_size 非法");
    }
    ExpMapLookupWitness witness;
    witness.edge_count = static_cast<uint>(graph.edges.size());
    witness.padded_edges = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.edges.size())));
    witness.table_size = static_cast<uint>(table.range_size);

    std::vector<std::int64_t> z_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> node_max_i64(graph.num_nodes, std::numeric_limits<std::int64_t>::min());
    std::vector<std::uint32_t> incoming_count(graph.num_nodes, 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto dst = graph.edges[edge].dst;
        z_i64[edge] = quantize_scale16_to_i64(trace.z[edge]);
        node_max_i64[dst] = std::max(node_max_i64[dst], z_i64[edge]);
        ++incoming_count[dst];
    }
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (incoming_count[node] == 0) throw std::runtime_error("ExpMap lookup 发现无入边节点");
    }

    std::vector<std::int64_t> delta_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> u_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> expected_i64(witness.padded_edges, 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto dst = graph.edges[edge].dst;
        const auto delta = checked_i128_to_i64(
            static_cast<__int128>(node_max_i64[dst]) - static_cast<__int128>(z_i64[edge]),
            "ExpMap delta");
        if (delta < 0) throw std::runtime_error("ExpMap delta 为负");
        if (static_cast<std::uint64_t>(delta) >= table.range_size) {
            throw std::runtime_error("ExpMap delta 超出公开 table range");
        }
        delta_i64[edge] = delta;
        witness.max_delta = std::max(witness.max_delta, static_cast<std::uint64_t>(delta));
        const double shifted = static_cast<double>(delta) / static_cast<double>(quant.scale);
        expected_i64[edge] = zkdl_round_half_away_from_zero(std::exp(-shifted) * static_cast<double>(quant.scale));
        u_i64[edge] = quantize_scale16_to_i64(trace.u[edge]);
        witness.max_u = std::max(witness.max_u, static_cast<std::uint64_t>(std::max<std::int64_t>(0, u_i64[edge])));
        if (expected_i64[edge] != u_i64[edge]) ++witness.exp_mismatch_count;
    }

    std::vector<std::int64_t> table_i64(table.range_size, 0);
    for (std::size_t delta = 0; delta < table.range_size; ++delta) {
        const double shifted = static_cast<double>(delta) / static_cast<double>(quant.scale);
        table_i64[delta] = zkdl_round_half_away_from_zero(std::exp(-shifted) * static_cast<double>(quant.scale));
    }
    witness.delta_mont = attention_i64_to_mont(delta_i64);
    witness.u_mont = attention_i64_to_mont(u_i64);
    witness.expected_mont = attention_i64_to_mont(expected_i64);
    witness.table_mont = attention_i64_to_mont(table_i64);
    return witness;
}

ZkdlFullRelationRecord prove_gpu_expmap_lookup_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_expmap_lookup_witness(graph, trace, quant, table);
    const auto block = formal_effective_commitment_block_size(
        requested_commitment_block_size, witness.padded_edges, witness.table_size);

    ZkdlFullRelationRecord record;
    record.stage = "expmap_lookup_range";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":expmap_lookup";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    Commitment gen = make_formal_commitment_generators(block);
    FrTensor delta_tensor(static_cast<uint>(witness.delta_mont.size()), witness.delta_mont.data());
    FrTensor u_tensor(static_cast<uint>(witness.u_mont.size()), witness.u_mont.data());
    FrTensor expected_tensor(static_cast<uint>(witness.expected_mont.size()), witness.expected_mont.data());
    FrTensor table_tensor(static_cast<uint>(witness.table_mont.size()), witness.table_mont.data());
    ScopedTimer commit_timer;
    const auto com_delta = gen.commit(delta_tensor);
    const auto com_u = gen.commit(u_tensor);
    const auto com_expected = gen.commit(expected_tensor);
    const auto com_table = gen.commit(table_tensor);
    cuda_sync("ExpMap lookup commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    const auto edge_point = deterministic_mont_fr_vec(ceilLog2(witness.padded_edges), prefix + ":edge_point");
    const auto table_point = deterministic_mont_fr_vec(ceilLog2(witness.table_size), prefix + ":table_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof u_binding;
    {
        std::vector<FormalLinearTerm> terms = {
            {"u", &u_tensor, &com_u, one, edge_point},
            {"expected", &expected_tensor, &com_expected, neg_one, edge_point},
        };
        ScopedTimer timer;
        u_binding = formal_linear_relation_prove(
            prefix + ":u_equals_expmap_table_value", {witness.padded_edges}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(u_binding, gen, {&com_u, &com_expected}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU ExpMap u binding verifier rejected: " + failure);
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
            throw std::runtime_error("GPU ExpMap table opening verifier rejected");
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = u_binding.serialized_size_bytes() + table_opening.serialized_size_bytes();
    record.proof_digests.push_back("u_binding=" + u_binding.transcript_digest);
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("exp_table_digest=" + table.exp_digest);
    record.proof_digests.push_back("table_opening_value_recorded=true");
    record.proof_digests.push_back("exp_mismatch_count=" + std::to_string(witness.exp_mismatch_count));
    record.proof_digests.push_back("max_delta=" + std::to_string(witness.max_delta));
    record.proof_digests.push_back("max_u=" + std::to_string(witness.max_u));
    record.proof_generated = true;
    record.verified = true;
    return record;
}

ZkdlFullRelationRecord prove_cpu_expmap_lookup_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& trace,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_expmap_lookup_witness(graph, trace, quant, table);
    const auto block = formal_effective_commitment_block_size(
        requested_commitment_block_size, witness.padded_edges, witness.table_size);

    ZkdlFullRelationRecord record;
    record.stage = "cpu_expmap_lookup_range";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":expmap_lookup";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block), spec.dataset + ":cpu_expmap_block" + std::to_string(block));
    CpuFrTensor delta_tensor(witness.delta_mont);
    CpuFrTensor u_tensor(witness.u_mont);
    CpuFrTensor expected_tensor(witness.expected_mont);
    CpuFrTensor table_tensor(witness.table_mont);
    ScopedTimer commit_timer;
    const auto com_delta = gen.commit(delta_tensor);
    const auto com_u = gen.commit(u_tensor);
    const auto com_expected = gen.commit(expected_tensor);
    const auto com_table = gen.commit(table_tensor);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    const auto edge_point = cpu_deterministic_mont_fr_vec(ceilLog2(witness.padded_edges), prefix + ":edge_point");
    const auto table_point = cpu_deterministic_mont_fr_vec(ceilLog2(witness.table_size), prefix + ":table_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof u_binding;
    {
        std::vector<CpuLinearTerm> terms = {
            {"u", &u_tensor, &com_u, one, edge_point},
            {"expected", &expected_tensor, &com_expected, neg_one, edge_point},
        };
        ScopedTimer timer;
        u_binding = cpu_linear_relation_prove(
            prefix + ":u_equals_expmap_table_value", {witness.padded_edges}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(u_binding, gen, {&com_u, &com_expected}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU ExpMap u binding verifier rejected: " + failure);
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
            throw std::runtime_error("CPU ExpMap table opening verifier rejected");
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = u_binding.serialized_size_bytes() + table_opening.serialized_size_bytes();
    record.proof_digests.push_back("u_binding=" + u_binding.transcript_digest);
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("exp_table_digest=" + table.exp_digest);
    record.proof_digests.push_back("table_opening_value_recorded=true");
    record.proof_digests.push_back("exp_mismatch_count=" + std::to_string(witness.exp_mismatch_count));
    record.proof_digests.push_back("max_delta=" + std::to_string(witness.max_delta));
    record.proof_digests.push_back("max_u=" + std::to_string(witness.max_u));
    record.proof_generated = true;
    record.verified = true;
    return record;
}
