#pragma once

// target-node max/shift comparison 子链。
// 证明每条边的 shifted score delta = m_dst - z_edge 非负，并证明每个 target node 的 max 被某条入边达到。

struct TargetMaxShiftWitness {
    uint node_count = 0;
    uint edge_count = 0;
    uint padded_nodes = 0;
    uint padded_edges = 0;
    uint bit_width = 0;
    std::vector<Fr_t> node_max_mont;
    std::vector<Fr_t> edge_max_mont;
    std::vector<Fr_t> z_mont;
    std::vector<Fr_t> delta_mont;
    std::vector<std::vector<Fr_t>> delta_bits_mont;
    std::vector<Fr_t> representative_mont;
    std::vector<Fr_t> zero_edge_mont;
    std::vector<Fr_t> valid_node_mont;
    std::size_t max_mismatch_count = 0;
    std::uint64_t max_delta = 0;
};

uint unsigned_bit_width_u64(std::uint64_t value)
{
    uint bits = 1;
    std::uint64_t limit = 1;
    while (limit <= value && bits < 63U) {
        limit <<= 1U;
        ++bits;
    }
    return std::max<uint>(1U, bits);
}

uint zkdl_table_range_bits(const ZkdlGatTableContext& table)
{
    if (table.range_size == 0 || (table.range_size & (table.range_size - 1U)) != 0) {
        throw std::runtime_error("table context range_size 必须为非零二次幂");
    }
    uint bits = 0;
    std::size_t value = table.range_size;
    while (value > 1U) {
        value >>= 1U;
        ++bits;
    }
    if (bits == 0 || bits >= 63U) {
        throw std::runtime_error("table context range_bits 超出支持范围");
    }
    return bits;
}

// 构造每个 dst 组的 M、M_edge-Z_edge、代表最大边与位分解见证。
// 该子关系保证 Softmax 稳定化使用的是公开目标组内最大值，而不是调试比较。
TargetMaxShiftWitness build_target_max_shift_witness(
    const LocalGraph& graph,
    const HeadTraceHost& trace,
    uint public_bit_width)
{
    if (trace.z.size() != graph.edges.size() || trace.m.size() != graph.num_nodes) {
        throw std::runtime_error("target max/shift trace 形状不一致");
    }
    TargetMaxShiftWitness witness;
    witness.node_count = static_cast<uint>(graph.num_nodes);
    witness.edge_count = static_cast<uint>(graph.edges.size());
    witness.padded_nodes = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.num_nodes)));
    witness.padded_edges = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.edges.size())));

    std::vector<std::int64_t> z_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> node_max_i64(witness.padded_nodes, std::numeric_limits<std::int64_t>::min());
    std::vector<std::uint32_t> incoming_count(graph.num_nodes, 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto dst = graph.edges[edge].dst;
        z_i64[edge] = quantize_scale16_to_i64(trace.z[edge]);
        node_max_i64[dst] = std::max(node_max_i64[dst], z_i64[edge]);
        ++incoming_count[dst];
    }
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (incoming_count[node] == 0) {
            throw std::runtime_error("target max/shift 发现无入边节点，self-loop 规则未满足");
        }
        if (node_max_i64[node] != quantize_scale16_to_i64(trace.m[node])) {
            ++witness.max_mismatch_count;
        }
    }
    for (std::size_t node = graph.num_nodes; node < witness.padded_nodes; ++node) {
        node_max_i64[node] = 0;
    }

    std::vector<std::int64_t> edge_max_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> delta_i64(witness.padded_edges, 0);
    std::vector<std::int64_t> representative_i64(witness.padded_edges, 0);
    std::vector<char> representative_seen(graph.num_nodes, 0);
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto dst = graph.edges[edge].dst;
        edge_max_i64[edge] = node_max_i64[dst];
        const auto delta = checked_i128_to_i64(
            static_cast<__int128>(edge_max_i64[edge]) - static_cast<__int128>(z_i64[edge]),
            "target max delta");
        if (delta < 0) throw std::runtime_error("target max/shift delta 为负");
        delta_i64[edge] = delta;
        witness.max_delta = std::max(witness.max_delta, static_cast<std::uint64_t>(delta));
        if (delta == 0 && representative_seen[dst] == 0) {
            representative_i64[edge] = 1;
            representative_seen[dst] = 1;
        }
    }
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (representative_seen[node] == 0) {
            throw std::runtime_error("target max/shift 缺少 max representative edge");
        }
    }

    if (public_bit_width == 0 || public_bit_width >= 63U) {
        throw std::runtime_error("target max/shift public bit width 非法");
    }
    if (witness.max_delta >= (std::uint64_t{1} << public_bit_width)) {
        throw std::runtime_error("target max/shift delta 超出 public range_bits");
    }
    witness.bit_width = public_bit_width;
    witness.delta_bits_mont.assign(witness.bit_width, std::vector<Fr_t>(witness.padded_edges, Fr_host_zero()));
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const auto value = static_cast<std::uint64_t>(delta_i64[edge]);
        for (uint bit = 0; bit < witness.bit_width; ++bit) {
            witness.delta_bits_mont[bit][edge] = ((value >> bit) & 1ULL) ? Fr_host_one_mont() : Fr_host_zero();
        }
    }
    witness.node_max_mont = attention_i64_to_mont(node_max_i64);
    witness.edge_max_mont = attention_i64_to_mont(edge_max_i64);
    witness.z_mont = attention_i64_to_mont(z_i64);
    witness.delta_mont = attention_i64_to_mont(delta_i64);
    witness.representative_mont = attention_i64_to_mont(representative_i64);
    witness.zero_edge_mont.assign(witness.padded_edges, Fr_host_zero());
    witness.valid_node_mont = valid_one_mont_vector(graph.num_nodes, witness.padded_nodes);
    return witness;
}

// GPU full-chain target max/shift relation：证明差分非负、代表边达到最大值且 mismatch 为 0。
ZkdlFullRelationRecord prove_gpu_target_max_shift_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& trace,
    const ZkdlGatTableContext& table,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_target_max_shift_witness(graph, trace, zkdl_table_range_bits(table));
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const auto block = formal_effective_commitment_block_size(
        requested_commitment_block_size, witness.padded_nodes, witness.padded_edges);

    ZkdlFullRelationRecord record;
    record.stage = "target_max_shift_comparison";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":target_max_shift";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    Commitment gen = make_formal_commitment_generators(block);
    FrTensor node_max_tensor(static_cast<uint>(witness.node_max_mont.size()), witness.node_max_mont.data());
    FrTensor edge_max_tensor(static_cast<uint>(witness.edge_max_mont.size()), witness.edge_max_mont.data());
    FrTensor z_tensor(static_cast<uint>(witness.z_mont.size()), witness.z_mont.data());
    FrTensor delta_tensor(static_cast<uint>(witness.delta_mont.size()), witness.delta_mont.data());
    FrTensor representative_tensor(static_cast<uint>(witness.representative_mont.size()), witness.representative_mont.data());
    FrTensor zero_edge_tensor(static_cast<uint>(witness.zero_edge_mont.size()), witness.zero_edge_mont.data());
    FrTensor valid_node_tensor(static_cast<uint>(witness.valid_node_mont.size()), witness.valid_node_mont.data());
    std::vector<std::unique_ptr<FrTensor>> bit_tensors;
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_tensors.emplace_back(std::make_unique<FrTensor>(static_cast<uint>(witness.delta_bits_mont[bit].size()), witness.delta_bits_mont[bit].data()));
    }

    ScopedTimer commit_timer;
    const auto com_node_max = gen.commit(node_max_tensor);
    const auto com_edge_max = gen.commit(edge_max_tensor);
    const auto com_z = gen.commit(z_tensor);
    const auto com_delta = gen.commit(delta_tensor);
    const auto com_representative = gen.commit(representative_tensor);
    const auto com_zero_edge = gen.commit(zero_edge_tensor);
    const auto com_valid_node = gen.commit(valid_node_tensor);
    std::vector<std::unique_ptr<G1TensorJacobian>> bit_commitments;
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_commitments.emplace_back(std::make_unique<G1TensorJacobian>(gen.commit(*bit_tensors[bit])));
    }
    cuda_sync("target max/shift commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    SparseRoutingRelationProof max_route;
    GroupedSumRelationProof representative_grouped;
    HadamardSumcheckProof representative_boolean;
    HadamardSumcheckProof representative_delta_zero;
    {
        ScopedTimer timer;
        max_route = formal_sparse_routing_relation_prove(
            node_max_tensor, edge_max_tensor, gen, com_node_max, gen, com_edge_max,
            src, dst, witness.node_count, witness.edge_count, "destination", prefix + ":max_destination_routing");
        representative_grouped = formal_grouped_sum_relation_prove(
            representative_tensor, valid_node_tensor, gen, com_representative, gen, com_valid_node,
            src, dst, witness.node_count, witness.edge_count, prefix + ":representative_one_per_target");
        representative_boolean = formal_hadamard_relation_prove(
            representative_tensor, representative_tensor, representative_tensor,
            gen, com_representative, com_representative, com_representative, prefix + ":representative_boolean");
        representative_delta_zero = formal_hadamard_relation_prove(
            representative_tensor, delta_tensor, zero_edge_tensor,
            gen, com_representative, com_delta, com_zero_edge, prefix + ":representative_delta_zero");
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_sparse_routing_relation_verify(max_route, gen, com_node_max, gen, com_edge_max, src, dst, "destination", prefix + ":max_destination_routing", &failure)) {
            throw std::runtime_error("GPU target max routing verifier rejected: " + failure);
        }
        if (!formal_grouped_sum_relation_verify(representative_grouped, gen, com_representative, gen, com_valid_node, src, dst, prefix + ":representative_one_per_target", &failure)) {
            throw std::runtime_error("GPU max representative grouped verifier rejected: " + failure);
        }
        if (!formal_hadamard_relation_verify(representative_boolean, gen, com_representative, com_representative, com_representative, prefix + ":representative_boolean", &failure)) {
            throw std::runtime_error("GPU max representative boolean verifier rejected: " + failure);
        }
        if (!formal_hadamard_relation_verify(representative_delta_zero, gen, com_representative, com_delta, com_zero_edge, prefix + ":representative_delta_zero", &failure)) {
            throw std::runtime_error("GPU max representative delta verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += max_route.serialized_size_bytes()
        + representative_grouped.serialized_size_bytes()
        + representative_boolean.serialized_size_bytes()
        + representative_delta_zero.serialized_size_bytes();
    record.proof_digests.push_back("max_route=" + max_route.transcript_digest);
    record.proof_digests.push_back("representative_grouped=" + representative_grouped.transcript_digest);
    record.proof_digests.push_back("representative_boolean=" + representative_boolean.transcript_digest);
    record.proof_digests.push_back("representative_delta_zero=" + representative_delta_zero.transcript_digest);

    const auto edge_point = deterministic_mont_fr_vec(ceilLog2(witness.padded_edges), prefix + ":edge_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof delta_identity;
    {
        std::vector<FormalLinearTerm> terms = {
            {"edge_max", &edge_max_tensor, &com_edge_max, one, edge_point},
            {"z", &z_tensor, &com_z, neg_one, edge_point},
            {"delta", &delta_tensor, &com_delta, neg_one, edge_point},
        };
        ScopedTimer timer;
        delta_identity = formal_linear_relation_prove(
            prefix + ":max_minus_z_equals_shift_delta", {witness.padded_edges}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(delta_identity, gen, {&com_edge_max, &com_z, &com_delta}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU max delta identity verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += delta_identity.serialized_size_bytes();
    record.proof_digests.push_back("delta_identity=" + delta_identity.transcript_digest);

    std::vector<FormalLinearTerm> decomposition_terms = {{"delta", &delta_tensor, &com_delta, one, edge_point}};
    std::vector<const G1TensorJacobian*> decomposition_commitments = {&com_delta};
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        decomposition_terms.push_back({
            "delta_bit" + std::to_string(bit),
            bit_tensors[bit].get(),
            bit_commitments[bit].get(),
            fr_from_signed_mont_host(-static_cast<std::int64_t>(1ULL << bit)),
            edge_point,
        });
        decomposition_commitments.push_back(bit_commitments[bit].get());
    }
    LinearRelationProof delta_decomposition;
    {
        ScopedTimer timer;
        delta_decomposition = formal_linear_relation_prove(
            prefix + ":delta_bit_decomposition", {witness.padded_edges}, decomposition_terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(delta_decomposition, gen, decomposition_commitments, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU max delta decomposition verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += delta_decomposition.serialized_size_bytes();
    record.proof_digests.push_back("delta_decomposition=" + delta_decomposition.transcript_digest);
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        const auto label = prefix + ":delta_bit" + std::to_string(bit) + ":boolean";
        ScopedTimer timer;
        auto bitness = formal_hadamard_relation_prove(
            *bit_tensors[bit], *bit_tensors[bit], *bit_tensors[bit],
            gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label);
        record.prove_ms += timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        if (!formal_hadamard_relation_verify(bitness, gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label, &failure)) {
            throw std::runtime_error("GPU max delta bitness verifier rejected: " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
        record.proof_size_bytes += bitness.serialized_size_bytes();
        record.proof_digests.push_back("delta_bit" + std::to_string(bit) + "=" + bitness.transcript_digest);
    }
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("max_mismatch_count=" + std::to_string(witness.max_mismatch_count));
    record.proof_digests.push_back("max_delta=" + std::to_string(witness.max_delta));
    record.proof_digests.push_back("delta_bit_width=" + std::to_string(witness.bit_width));
    record.proof_generated = true;
    record.verified = true;
    return record;
}

// CPU full-chain target max/shift relation：使用相同 table_context_id 绑定 range 位宽。
ZkdlFullRelationRecord prove_cpu_target_max_shift_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& trace,
    const ZkdlGatTableContext& table,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_target_max_shift_witness(graph, trace, zkdl_table_range_bits(table));
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const auto block = formal_effective_commitment_block_size(
        requested_commitment_block_size, witness.padded_nodes, witness.padded_edges);

    ZkdlFullRelationRecord record;
    record.stage = "cpu_target_max_shift_comparison";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":target_max_shift";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block), spec.dataset + ":cpu_target_max_block" + std::to_string(block));
    CpuFrTensor node_max_tensor(witness.node_max_mont);
    CpuFrTensor edge_max_tensor(witness.edge_max_mont);
    CpuFrTensor z_tensor(witness.z_mont);
    CpuFrTensor delta_tensor(witness.delta_mont);
    CpuFrTensor representative_tensor(witness.representative_mont);
    CpuFrTensor zero_edge_tensor(witness.zero_edge_mont);
    CpuFrTensor valid_node_tensor(witness.valid_node_mont);
    std::vector<std::unique_ptr<CpuFrTensor>> bit_tensors;
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_tensors.emplace_back(std::make_unique<CpuFrTensor>(witness.delta_bits_mont[bit]));
    }

    ScopedTimer commit_timer;
    const auto com_node_max = gen.commit(node_max_tensor);
    const auto com_edge_max = gen.commit(edge_max_tensor);
    const auto com_z = gen.commit(z_tensor);
    const auto com_delta = gen.commit(delta_tensor);
    const auto com_representative = gen.commit(representative_tensor);
    const auto com_zero_edge = gen.commit(zero_edge_tensor);
    const auto com_valid_node = gen.commit(valid_node_tensor);
    std::vector<std::unique_ptr<CpuG1Tensor>> bit_commitments;
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_commitments.emplace_back(std::make_unique<CpuG1Tensor>(gen.commit(*bit_tensors[bit])));
    }
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string prefix = record.relation_scope;
    SparseRoutingRelationProof max_route;
    GroupedSumRelationProof representative_grouped;
    HadamardSumcheckProof representative_boolean;
    HadamardSumcheckProof representative_delta_zero;
    {
        ScopedTimer timer;
        max_route = cpu_sparse_routing_relation_prove(
            node_max_tensor, edge_max_tensor, gen, com_node_max, gen, com_edge_max,
            src, dst, witness.node_count, witness.edge_count, "destination", prefix + ":max_destination_routing");
        representative_grouped = cpu_grouped_sum_relation_prove(
            representative_tensor, valid_node_tensor, gen, com_representative, gen, com_valid_node,
            src, dst, witness.node_count, witness.edge_count, prefix + ":representative_one_per_target");
        representative_boolean = cpu_hadamard_relation_prove(
            representative_tensor, representative_tensor, representative_tensor,
            gen, com_representative, com_representative, com_representative, prefix + ":representative_boolean");
        representative_delta_zero = cpu_hadamard_relation_prove(
            representative_tensor, delta_tensor, zero_edge_tensor,
            gen, com_representative, com_delta, com_zero_edge, prefix + ":representative_delta_zero");
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_sparse_routing_relation_verify(max_route, gen, com_node_max, gen, com_edge_max, src, dst, "destination", prefix + ":max_destination_routing", &failure)) {
            throw std::runtime_error("CPU target max routing verifier rejected: " + failure);
        }
        if (!cpu_grouped_sum_relation_verify(representative_grouped, gen, com_representative, gen, com_valid_node, src, dst, prefix + ":representative_one_per_target", &failure)) {
            throw std::runtime_error("CPU max representative grouped verifier rejected: " + failure);
        }
        if (!cpu_hadamard_relation_verify(representative_boolean, gen, com_representative, com_representative, com_representative, prefix + ":representative_boolean", &failure)) {
            throw std::runtime_error("CPU max representative boolean verifier rejected: " + failure);
        }
        if (!cpu_hadamard_relation_verify(representative_delta_zero, gen, com_representative, com_delta, com_zero_edge, prefix + ":representative_delta_zero", &failure)) {
            throw std::runtime_error("CPU max representative delta verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += max_route.serialized_size_bytes()
        + representative_grouped.serialized_size_bytes()
        + representative_boolean.serialized_size_bytes()
        + representative_delta_zero.serialized_size_bytes();
    record.proof_digests.push_back("max_route=" + max_route.transcript_digest);
    record.proof_digests.push_back("representative_grouped=" + representative_grouped.transcript_digest);
    record.proof_digests.push_back("representative_boolean=" + representative_boolean.transcript_digest);
    record.proof_digests.push_back("representative_delta_zero=" + representative_delta_zero.transcript_digest);

    const auto edge_point = cpu_deterministic_mont_fr_vec(ceilLog2(witness.padded_edges), prefix + ":edge_point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    LinearRelationProof delta_identity;
    {
        std::vector<CpuLinearTerm> terms = {
            {"edge_max", &edge_max_tensor, &com_edge_max, one, edge_point},
            {"z", &z_tensor, &com_z, neg_one, edge_point},
            {"delta", &delta_tensor, &com_delta, neg_one, edge_point},
        };
        ScopedTimer timer;
        delta_identity = cpu_linear_relation_prove(
            prefix + ":max_minus_z_equals_shift_delta", {witness.padded_edges}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(delta_identity, gen, {&com_edge_max, &com_z, &com_delta}, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU max delta identity verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += delta_identity.serialized_size_bytes();
    record.proof_digests.push_back("delta_identity=" + delta_identity.transcript_digest);

    std::vector<CpuLinearTerm> decomposition_terms = {{"delta", &delta_tensor, &com_delta, one, edge_point}};
    std::vector<const CpuG1Tensor*> decomposition_commitments = {&com_delta};
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        decomposition_terms.push_back({
            "delta_bit" + std::to_string(bit),
            bit_tensors[bit].get(),
            bit_commitments[bit].get(),
            fr_from_signed_mont_host(-static_cast<std::int64_t>(1ULL << bit)),
            edge_point,
        });
        decomposition_commitments.push_back(bit_commitments[bit].get());
    }
    LinearRelationProof delta_decomposition;
    {
        ScopedTimer timer;
        delta_decomposition = cpu_linear_relation_prove(
            prefix + ":delta_bit_decomposition", {witness.padded_edges}, decomposition_terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(delta_decomposition, gen, decomposition_commitments, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU max delta decomposition verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes += delta_decomposition.serialized_size_bytes();
    record.proof_digests.push_back("delta_decomposition=" + delta_decomposition.transcript_digest);
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        const auto label = prefix + ":delta_bit" + std::to_string(bit) + ":boolean";
        ScopedTimer timer;
        auto bitness = cpu_hadamard_relation_prove(
            *bit_tensors[bit], *bit_tensors[bit], *bit_tensors[bit],
            gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label);
        record.prove_ms += timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        if (!cpu_hadamard_relation_verify(bitness, gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label, &failure)) {
            throw std::runtime_error("CPU max delta bitness verifier rejected: " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
        record.proof_size_bytes += bitness.serialized_size_bytes();
        record.proof_digests.push_back("delta_bit" + std::to_string(bit) + "=" + bitness.transcript_digest);
    }
    record.proof_digests.push_back("table_context_id=" + table.context_id);
    record.proof_digests.push_back("max_mismatch_count=" + std::to_string(witness.max_mismatch_count));
    record.proof_digests.push_back("max_delta=" + std::to_string(witness.max_delta));
    record.proof_digests.push_back("delta_bit_width=" + std::to_string(witness.bit_width));
    record.proof_generated = true;
    record.verified = true;
    return record;
}
