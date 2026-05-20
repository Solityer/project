#pragma once

// CPU-only zkDL-native relation backend. This mirrors the local Hyrax-style
// tensor commitment/opening semantics without using FrTensor/G1Tensor CUDA
// storage, so CPU results remain independent from GPU partial artifacts.

struct CpuFrTensor {
    std::vector<Fr_t> values;

    CpuFrTensor() = default;
    explicit CpuFrTensor(std::vector<Fr_t> v) : values(std::move(v)) {}

    uint size() const
    {
        if (values.size() > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("CPU tensor exceeds uint size");
        }
        return static_cast<uint>(values.size());
    }
};

struct CpuG1Tensor {
    std::vector<G1Jacobian_t> points;

    CpuG1Tensor() = default;
    explicit CpuG1Tensor(std::vector<G1Jacobian_t> p) : points(std::move(p)) {}

    uint size() const
    {
        if (points.size() > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("CPU G1 tensor exceeds uint size");
        }
        return static_cast<uint>(points.size());
    }
};

bool cpu_raw_fr_to_u64(const Fr_t& value, std::uint64_t* out)
{
    for (uint i = 2; i < 8; ++i) {
        if (value.val[i] != 0U) return false;
    }
    *out = (static_cast<std::uint64_t>(value.val[1]) << 32U) | value.val[0];
    return true;
}

G1Jacobian_t cpu_g1_mul_u64(G1Jacobian_t base, std::uint64_t scalar)
{
    G1Jacobian_t out = G1Jacobian_host_zero();
    while (scalar != 0U) {
        if ((scalar & 1ULL) != 0ULL) {
            out = G1Jacobian_host_add(out, base);
        }
        scalar >>= 1U;
        if (scalar != 0U) {
            base = G1Jacobian_host_double(base);
        }
    }
    return out;
}

G1Jacobian_t cpu_g1_mul_fr_raw(G1Jacobian_t base, const Fr_t& raw_scalar)
{
    std::uint64_t positive = 0;
    if (cpu_raw_fr_to_u64(raw_scalar, &positive)) {
        return cpu_g1_mul_u64(base, positive);
    }
    const Fr_t neg_raw = Fr_host_sub(Fr_host_modulus(), raw_scalar);
    std::uint64_t negative = 0;
    if (cpu_raw_fr_to_u64(neg_raw, &negative)) {
        return G1Jacobian_host_minus(cpu_g1_mul_u64(base, negative));
    }
    return G1Jacobian_host_mul(base, raw_scalar);
}

void cpu_absorb_g1_tensor(FormalTranscript& transcript, const CpuG1Tensor& commitment)
{
    transcript.absorb_u64(commitment.points.size());
    for (const auto& point : commitment.points) transcript.absorb_g1(point);
}

std::vector<Fr_t> cpu_partial_me_checked(
    std::vector<Fr_t> current,
    std::vector<Fr_t>::const_iterator begin,
    std::vector<Fr_t>::const_iterator end,
    uint window_size)
{
    if (window_size == 0) throw std::runtime_error("CPU partial MLE invalid window size");
    for (auto it = begin; it != end; ++it) {
        const uint in_size = static_cast<uint>(current.size());
        const uint out_size = Fr_partial_me_out_size(in_size, window_size);
        std::vector<Fr_t> next(out_size, Fr_host_zero());
        for (uint gid = 0; gid < out_size; ++gid) {
            const uint window_id = gid / window_size;
            const uint idx_in_window = gid % window_size;
            const uint gid0 = 2U * window_id * window_size + idx_in_window;
            const uint gid1 = (2U * window_id + 1U) * window_size + idx_in_window;
            const Fr_t low = gid0 < in_size ? current[gid0] : Fr_host_zero();
            const Fr_t high = gid1 < in_size ? current[gid1] : Fr_host_zero();
            next[gid] = Fr_host_mle_step(low, high, *it);
        }
        current.swap(next);
    }
    return current;
}

struct CpuCommitment {
    std::vector<G1Jacobian_t> generators;

    CpuCommitment() = default;
    explicit CpuCommitment(std::vector<G1Jacobian_t> g) : generators(std::move(g)) {}

    uint size() const
    {
        if (generators.size() > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("CPU generator tensor exceeds uint size");
        }
        return static_cast<uint>(generators.size());
    }

    static CpuCommitment deterministic(uint size, const std::string& label)
    {
        if (size == 0) throw std::runtime_error("CPU commitment needs non-empty generators");
        auto scalars = deterministic_raw_fr_vec(size, "zkdl-gat-cpu-generators:" + label + ":" + std::to_string(size));
        std::vector<G1Jacobian_t> points(size, G1Jacobian_host_zero());
        for (uint i = 0; i < size; ++i) {
            points[i] = cpu_g1_mul_fr_raw(G1Jacobian_generator, scalars[i]);
        }
        return CpuCommitment(std::move(points));
    }

    CpuG1Tensor commit(const CpuFrTensor& tensor) const
    {
        if (generators.empty()) throw std::runtime_error("CPU commitment has no generators");
        if ((tensor.values.size() % generators.size()) != 0) {
            throw std::runtime_error("CPU commitment incompatible dimensions");
        }
        const std::size_t rows = tensor.values.size() / generators.size();
        std::vector<G1Jacobian_t> out(rows, G1Jacobian_host_zero());
        for (std::size_t row = 0; row < rows; ++row) {
            G1Jacobian_t acc = G1Jacobian_host_zero();
            for (std::size_t col = 0; col < generators.size(); ++col) {
                const Fr_t raw = Fr_host_unmont(tensor.values[row * generators.size() + col]);
                acc = G1Jacobian_host_add(acc, cpu_g1_mul_fr_raw(generators[col], raw));
            }
            out[row] = acc;
        }
        return CpuG1Tensor(std::move(out));
    }

    Fr_t open_with_proof(
        const CpuFrTensor& tensor,
        const CpuG1Tensor& commitment,
        const std::vector<Fr_t>& point,
        CommitmentOpeningProof& proof) const;

    static bool verify_opening(
        const CpuCommitment& generators,
        const CpuG1Tensor& commitment,
        const CommitmentOpeningProof& proof);
};

Fr_t cpu_me_open_fixed_recursive(
    const std::vector<Fr_t>& scalars,
    const CpuCommitment& generators,
    std::vector<Fr_t>::const_iterator begin,
    std::vector<Fr_t>::const_iterator end,
    std::vector<G1Jacobian_t>& proof)
{
    if (scalars.size() != generators.generators.size()) {
        throw std::runtime_error("CPU opening incompatible dimensions");
    }
    if (begin >= end) {
        proof.push_back(generators.generators.front());
        return scalars.empty() ? Fr_host_zero() : scalars.front();
    }
    if ((scalars.size() % 2U) != 0U) {
        throw std::runtime_error("CPU opening requires power-of-two generator size");
    }
    const uint new_size = static_cast<uint>(scalars.size() / 2U);
    std::vector<Fr_t> left_scalars(new_size, Fr_host_zero());
    std::vector<Fr_t> right_scalars(new_size, Fr_host_zero());
    std::vector<Fr_t> folded_scalars(new_size, Fr_host_zero());
    std::vector<G1Jacobian_t> left_points(new_size, G1Jacobian_host_zero());
    std::vector<G1Jacobian_t> right_points(new_size, G1Jacobian_host_zero());
    for (uint i = 0; i < new_size; ++i) {
        left_scalars[i] = scalars[2U * i];
        right_scalars[i] = scalars[2U * i + 1U];
        folded_scalars[i] = Fr_host_mle_step(left_scalars[i], right_scalars[i], *begin);
        left_points[i] = generators.generators[2U * i];
        right_points[i] = generators.generators[2U * i + 1U];
    }

    CpuCommitment left_generators(left_points);
    CpuCommitment right_generators(right_points);
    proof.push_back(generators.commit(CpuFrTensor(scalars)).points.front());
    proof.push_back(right_generators.commit(CpuFrTensor(left_scalars)).points.front());
    proof.push_back(left_generators.commit(CpuFrTensor(right_scalars)).points.front());

    const Fr_t u_raw = Fr_host_unmont(*begin);
    const Fr_t one_minus_u_raw = Fr_host_unmont(Fr_host_sub(Fr_host_one_mont(), *begin));
    std::vector<G1Jacobian_t> folded_points(new_size, G1Jacobian_host_zero());
    for (uint i = 0; i < new_size; ++i) {
        folded_points[i] = G1Jacobian_host_add(
            G1Jacobian_host_mul(left_points[i], u_raw),
            G1Jacobian_host_mul(right_points[i], one_minus_u_raw));
    }
    return cpu_me_open_fixed_recursive(folded_scalars, CpuCommitment(std::move(folded_points)), begin + 1, end, proof);
}

Fr_t CpuCommitment::open_with_proof(
    const CpuFrTensor& tensor,
    const CpuG1Tensor& commitment,
    const std::vector<Fr_t>& point,
    CommitmentOpeningProof& proof) const
{
    const uint outer_log = ceilLog2(commitment.size());
    if (point.size() < outer_log) throw std::runtime_error("CPU opening point too short");
    const uint inner_log = static_cast<uint>(point.size()) - outer_log;
    if (size() != (1U << inner_log)) throw std::runtime_error("CPU opening generator dimension mismatch");
    proof.point = point;
    proof.transcript.clear();
    const std::vector<Fr_t> u_out(point.end() - outer_log, point.end());
    const std::vector<Fr_t> u_in(point.begin(), point.end() - outer_log);
    auto reduced = cpu_partial_me_checked(tensor.values, u_out.begin(), u_out.end(), size());
    proof.value = cpu_me_open_fixed_recursive(reduced, *this, u_in.begin(), u_in.end(), proof.transcript);
    return proof.value;
}

bool CpuCommitment::verify_opening(
    const CpuCommitment& generators,
    const CpuG1Tensor& commitment,
    const CommitmentOpeningProof& proof)
{
    const uint outer_log = ceilLog2(commitment.size());
    if (proof.point.size() < outer_log) return false;
    const uint inner_log = static_cast<uint>(proof.point.size()) - outer_log;
    if (generators.size() != (1U << inner_log)) return false;
    if (proof.transcript.size() != static_cast<std::size_t>(3 * inner_log + 1)) return false;
    std::vector<Fr_t> u_out(proof.point.end() - outer_log, proof.point.end());
    const G1Jacobian_t public_commitment = outer_log == 0
        ? commitment.points.front()
        : G1Jacobian_host_mle(commitment.points, u_out.begin(), u_out.end());
    if (inner_log == 0) {
        if (!g1_equal_host(generators.generators.front(), proof.transcript.back())) return false;
        return g1_equal_host(public_commitment, g1_final_commitment_host(proof.transcript.back(), proof.value));
    }
    if (!g1_equal_host(public_commitment, proof.transcript.front())) return false;
    for (uint round = 0; round < inner_log; ++round) {
        const auto current = proof.transcript.at(3 * round);
        const auto cross0 = proof.transcript.at(3 * round + 1);
        const auto cross1 = proof.transcript.at(3 * round + 2);
        const auto expected_next = g1_open_expected_next_host(current, cross0, cross1, proof.point.at(round));
        const auto actual_next = (round + 1 < inner_log)
            ? proof.transcript.at(3 * (round + 1))
            : g1_final_commitment_host(proof.transcript.back(), proof.value);
        if (!g1_equal_host(expected_next, actual_next)) return false;
    }
    return true;
}

InnerProductSumcheckProof cpu_inner_product_sumcheck_prove(
    const std::vector<Fr_t>& a,
    const std::vector<Fr_t>& b,
    const std::string& relation_label)
{
    if (a.size() != b.size()) throw std::runtime_error("CPU inner product size mismatch");
    InnerProductSumcheckProof proof;
    proof.size = static_cast<uint>(a.size());
    std::vector<Fr_t> cur_a = a;
    std::vector<Fr_t> cur_b = b;
    FormalTranscript transcript;
    transcript.absorb_label("inner_product_sumcheck");
    transcript.absorb_label(relation_label);
    transcript.absorb_u64(proof.size);
    proof.claimed_sum = formal_inner_product_claim_host(cur_a, cur_b);
    transcript.absorb_fr(proof.claimed_sum);
    Fr_t running_claim = proof.claimed_sum;
    const uint rounds = ceilLog2(proof.size);
    for (uint round_idx = 0; round_idx < rounds; ++round_idx) {
        const uint out_size = static_cast<uint>((cur_a.size() + 1U) / 2U);
        SumcheckQuadraticRound round;
        std::vector<Fr_t> next_a(out_size, Fr_host_zero());
        std::vector<Fr_t> next_b(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i) {
            const Fr_t a0 = (2U * i < cur_a.size()) ? cur_a[2U * i] : Fr_host_zero();
            const Fr_t b0 = (2U * i < cur_b.size()) ? cur_b[2U * i] : Fr_host_zero();
            const Fr_t a1 = (2U * i + 1U < cur_a.size()) ? cur_a[2U * i + 1U] : Fr_host_zero();
            const Fr_t b1 = (2U * i + 1U < cur_b.size()) ? cur_b[2U * i + 1U] : Fr_host_zero();
            const Fr_t da = Fr_host_sub(a1, a0);
            const Fr_t db = Fr_host_sub(b1, b0);
            round.c0 = Fr_host_add(round.c0, Fr_host_mul(a0, b0));
            round.c1 = Fr_host_add(round.c1, Fr_host_add(Fr_host_mul(a0, db), Fr_host_mul(b0, da)));
            round.c2 = Fr_host_add(round.c2, Fr_host_mul(da, db));
        }
        if (!Fr_host_eq(running_claim, Fr_host_sum_quadratic_endpoints(round))) {
            throw std::runtime_error("CPU inner product inconsistent round");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("sumcheck_round");
        for (uint i = 0; i < out_size; ++i) {
            const Fr_t a0 = (2U * i < cur_a.size()) ? cur_a[2U * i] : Fr_host_zero();
            const Fr_t b0 = (2U * i < cur_b.size()) ? cur_b[2U * i] : Fr_host_zero();
            const Fr_t a1 = (2U * i + 1U < cur_a.size()) ? cur_a[2U * i + 1U] : Fr_host_zero();
            const Fr_t b1 = (2U * i + 1U < cur_b.size()) ? cur_b[2U * i + 1U] : Fr_host_zero();
            next_a[i] = Fr_host_mle_step(a0, a1, challenge);
            next_b[i] = Fr_host_mle_step(b0, b1, challenge);
        }
        running_claim = Fr_host_quadratic_eval(round, challenge);
        proof.rounds.push_back(round);
        proof.challenges.push_back(challenge);
        cur_a.swap(next_a);
        cur_b.swap(next_b);
    }
    proof.final_a = cur_a.empty() ? Fr_host_zero() : cur_a.front();
    proof.final_b = cur_b.empty() ? Fr_host_zero() : cur_b.front();
    transcript.absorb_fr(proof.final_a);
    transcript.absorb_fr(proof.final_b);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

void cpu_absorb_sparse_routing_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    const std::string& direction,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint edge_begin,
    uint edge_end,
    const CpuG1Tensor& com_node,
    const CpuG1Tensor& com_edge)
{
    transcript.absorb_label("sparse_routing_relation_cpu");
    transcript.absorb_label(label);
    transcript.absorb_label(direction);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(edge_begin);
    transcript.absorb_u64(edge_end);
    cpu_absorb_g1_tensor(transcript, com_node);
    cpu_absorb_g1_tensor(transcript, com_edge);
}

void cpu_absorb_grouped_sum_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint edge_begin,
    uint edge_end,
    const CpuG1Tensor& com_edge_input,
    const CpuG1Tensor& com_node_output)
{
    transcript.absorb_label("grouped_sum_relation_cpu");
    transcript.absorb_label(label);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(edge_begin);
    transcript.absorb_u64(edge_end);
    cpu_absorb_g1_tensor(transcript, com_edge_input);
    cpu_absorb_g1_tensor(transcript, com_node_output);
}

void cpu_absorb_hadamard_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    uint size,
    const CpuG1Tensor& com_a,
    const CpuG1Tensor& com_b,
    const CpuG1Tensor& com_c)
{
    transcript.absorb_label("hadamard_relation_cpu");
    transcript.absorb_label(label);
    transcript.absorb_u64(size);
    cpu_absorb_g1_tensor(transcript, com_a);
    cpu_absorb_g1_tensor(transcript, com_b);
    cpu_absorb_g1_tensor(transcript, com_c);
}

SparseRoutingRelationProof cpu_sparse_routing_relation_prove(
    const CpuFrTensor& node_values,
    const CpuFrTensor& edge_values,
    const CpuCommitment& node_generators,
    const CpuG1Tensor& com_node,
    const CpuCommitment& edge_generators,
    const CpuG1Tensor& com_edge,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    const std::string& direction,
    const std::string& relation_label)
{
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (node_values.size() != padded_node_count || edge_values.size() != padded_edge_count) {
        throw std::runtime_error("CPU sparse routing tensor size mismatch");
    }
    SparseRoutingRelationProof proof;
    proof.label = relation_label;
    proof.direction = direction;
    proof.topology_hash = formal_sparse_topology_hash(src, dst, node_count, edge_count);
    proof.node_count = node_count;
    proof.edge_count = edge_count;
    proof.padded_node_count = padded_node_count;
    proof.padded_edge_count = padded_edge_count;
    proof.edge_begin = 0;
    proof.edge_end = edge_count;
    proof.commitment_node_label = relation_label + ":node";
    proof.commitment_edge_label = relation_label + ":edge";

    FormalTranscript transcript;
    cpu_absorb_sparse_routing_metadata(transcript, relation_label, direction, proof.topology_hash,
        node_count, edge_count, padded_node_count, padded_edge_count, 0, edge_count, com_node, com_edge);
    proof.edge_point = formal_make_challenge_point(transcript, ceilLog2(padded_edge_count), "sparse_route_edge_point");
    proof.edge_eval = edge_generators.open_with_proof(edge_values, com_edge, proof.edge_point, proof.opening_edge);
    transcript.absorb_fr(proof.edge_eval);

    const auto coeff = formal_sparse_route_public_coefficients(
        src, dst, node_count, edge_count, padded_node_count, padded_edge_count, proof.edge_point, direction);
    proof.inner_product = cpu_inner_product_sumcheck_prove(coeff, node_values.values, relation_label + ":inner");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.edge_eval)) {
        throw std::runtime_error("CPU sparse routing claim mismatch");
    }
    proof.public_weight_eval = formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end());
    proof.inner_product.final_a = proof.public_weight_eval;
    proof.inner_product.final_b = node_generators.open_with_proof(
        node_values, com_node, proof.inner_product.challenges, proof.opening_node);
    proof.inner_product.opening_b = proof.opening_node;
    transcript.absorb_label(proof.inner_product.transcript_digest);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_sparse_routing_relation_verify(
    const SparseRoutingRelationProof& proof,
    const CpuCommitment& node_generators,
    const CpuG1Tensor& com_node,
    const CpuCommitment& edge_generators,
    const CpuG1Tensor& com_edge,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    const std::string& direction,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label || proof.direction != direction) return fail("CPU sparse label/direction mismatch");
    if (proof.topology_hash != formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count)) {
        return fail("CPU sparse topology hash mismatch");
    }
    FormalTranscript transcript;
    cpu_absorb_sparse_routing_metadata(transcript, relation_label, direction, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.padded_node_count, proof.padded_edge_count,
        proof.edge_begin, proof.edge_end, com_node, com_edge);
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_edge_count), "sparse_route_edge_point"), proof.edge_point)) {
        return fail("CPU sparse edge point replay mismatch");
    }
    if (!CpuCommitment::verify_opening(edge_generators, com_edge, proof.opening_edge)) return fail("CPU sparse edge opening rejected");
    transcript.absorb_fr(proof.edge_eval);
    const auto coeff = formal_sparse_route_public_coefficients(
        src, dst, proof.node_count, proof.edge_count, proof.padded_node_count, proof.padded_edge_count, proof.edge_point, direction);
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.edge_eval)) return fail("CPU sparse claim mismatch");
    std::string inner_failure;
    if (!formal_inner_product_sumcheck_verify(proof.inner_product, relation_label + ":inner", &inner_failure)) {
        return fail("CPU sparse inner sumcheck failed: " + inner_failure);
    }
    if (!Fr_host_eq(proof.public_weight_eval,
            formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end()))) {
        return fail("CPU sparse public coefficient eval mismatch");
    }
    if (!Fr_host_eq(proof.inner_product.final_a, proof.public_weight_eval)) return fail("CPU sparse final public eval mismatch");
    if (!CpuCommitment::verify_opening(node_generators, com_node, proof.opening_node)) return fail("CPU sparse node opening rejected");
    if (!Fr_host_eq(proof.opening_node.value, proof.inner_product.final_b)) return fail("CPU sparse node opening value mismatch");
    transcript.absorb_label(proof.inner_product.transcript_digest);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("CPU sparse transcript digest mismatch");
    return true;
}

GroupedSumRelationProof cpu_grouped_sum_relation_prove(
    const CpuFrTensor& edge_input,
    const CpuFrTensor& node_output,
    const CpuCommitment& edge_generators,
    const CpuG1Tensor& com_edge_input,
    const CpuCommitment& node_generators,
    const CpuG1Tensor& com_node_output,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    const std::string& relation_label)
{
    (void)src;
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (edge_input.size() != padded_edge_count || node_output.size() != padded_node_count) {
        throw std::runtime_error("CPU grouped sum tensor size mismatch");
    }
    GroupedSumRelationProof proof;
    proof.label = relation_label;
    proof.topology_hash = formal_sparse_topology_hash(src, dst, node_count, edge_count);
    proof.node_count = node_count;
    proof.edge_count = edge_count;
    proof.padded_node_count = padded_node_count;
    proof.padded_edge_count = padded_edge_count;
    proof.edge_begin = 0;
    proof.edge_end = edge_count;
    proof.commitment_edge_input_label = relation_label + ":edge_input";
    proof.commitment_node_output_label = relation_label + ":node_output";

    FormalTranscript transcript;
    cpu_absorb_grouped_sum_metadata(transcript, relation_label, proof.topology_hash,
        node_count, edge_count, padded_node_count, padded_edge_count, 0, edge_count, com_edge_input, com_node_output);
    proof.node_point = formal_make_challenge_point(transcript, ceilLog2(padded_node_count), "grouped_sum_node_point");
    proof.output_eval = node_generators.open_with_proof(node_output, com_node_output, proof.node_point, proof.opening_node_output);
    transcript.absorb_fr(proof.output_eval);
    const auto coeff = formal_grouped_sum_public_coefficients(dst, node_count, edge_count, padded_edge_count, proof.node_point);
    proof.inner_product = cpu_inner_product_sumcheck_prove(coeff, edge_input.values, relation_label + ":inner");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.output_eval)) {
        throw std::runtime_error("CPU grouped sum claim mismatch");
    }
    proof.public_weight_eval = formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end());
    proof.inner_product.final_a = proof.public_weight_eval;
    proof.inner_product.final_b = edge_generators.open_with_proof(
        edge_input, com_edge_input, proof.inner_product.challenges, proof.opening_edge_input);
    proof.inner_product.opening_b = proof.opening_edge_input;
    transcript.absorb_label(proof.inner_product.transcript_digest);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_grouped_sum_relation_verify(
    const GroupedSumRelationProof& proof,
    const CpuCommitment& edge_generators,
    const CpuG1Tensor& com_edge_input,
    const CpuCommitment& node_generators,
    const CpuG1Tensor& com_node_output,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("CPU grouped label mismatch");
    if (proof.topology_hash != formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count)) {
        return fail("CPU grouped topology hash mismatch");
    }
    FormalTranscript transcript;
    cpu_absorb_grouped_sum_metadata(transcript, relation_label, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.padded_node_count, proof.padded_edge_count,
        proof.edge_begin, proof.edge_end, com_edge_input, com_node_output);
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_node_count), "grouped_sum_node_point"), proof.node_point)) {
        return fail("CPU grouped node point replay mismatch");
    }
    if (!CpuCommitment::verify_opening(node_generators, com_node_output, proof.opening_node_output)) return fail("CPU grouped output opening rejected");
    if (!Fr_host_eq(proof.opening_node_output.value, proof.output_eval)) return fail("CPU grouped output opening value mismatch");
    transcript.absorb_fr(proof.output_eval);
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.output_eval)) return fail("CPU grouped claim mismatch");
    const auto coeff = formal_grouped_sum_public_coefficients(
        dst, proof.node_count, proof.edge_count, proof.padded_edge_count, proof.node_point);
    std::string inner_failure;
    if (!formal_inner_product_sumcheck_verify(proof.inner_product, relation_label + ":inner", &inner_failure)) {
        return fail("CPU grouped inner sumcheck failed: " + inner_failure);
    }
    if (!Fr_host_eq(proof.public_weight_eval,
            formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end()))) {
        return fail("CPU grouped public coefficient eval mismatch");
    }
    if (!CpuCommitment::verify_opening(edge_generators, com_edge_input, proof.opening_edge_input)) return fail("CPU grouped edge opening rejected");
    if (!Fr_host_eq(proof.opening_edge_input.value, proof.inner_product.final_b)) return fail("CPU grouped edge opening value mismatch");
    transcript.absorb_label(proof.inner_product.transcript_digest);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("CPU grouped transcript digest mismatch");
    return true;
}

HadamardSumcheckProof cpu_hadamard_relation_prove(
    const CpuFrTensor& a,
    const CpuFrTensor& b,
    const CpuFrTensor& c,
    const CpuCommitment& generators,
    const CpuG1Tensor& com_a,
    const CpuG1Tensor& com_b,
    const CpuG1Tensor& com_c,
    const std::string& relation_label)
{
    if (a.values.size() != b.values.size() || a.values.size() != c.values.size()) {
        throw std::runtime_error("CPU hadamard tensor size mismatch");
    }
    HadamardSumcheckProof proof;
    proof.label = relation_label;
    proof.size = a.size();
    proof.shape = {proof.size};
    proof.commitment_a_label = relation_label + ":A";
    proof.commitment_b_label = relation_label + ":B";
    proof.commitment_c_label = relation_label + ":C";
    FormalTranscript transcript;
    cpu_absorb_hadamard_metadata(transcript, relation_label, proof.size, com_a, com_b, com_c);
    const uint rounds = ceilLog2(proof.size);
    proof.point = formal_make_challenge_point(transcript, rounds, "hadamard_output_point");
    proof.eval_c = generators.open_with_proof(c, com_c, proof.point, proof.opening_c);
    proof.claimed_eval_c = proof.eval_c;
    transcript.absorb_fr(proof.eval_c);

    std::vector<Fr_t> cur_a = a.values;
    std::vector<Fr_t> cur_b = b.values;
    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < rounds; ++round_idx) {
        const uint out_size = static_cast<uint>((cur_a.size() + 1U) / 2U);
        std::vector<Fr_t> coeff0(out_size, Fr_host_zero());
        std::vector<Fr_t> coeff1(out_size, Fr_host_zero());
        std::vector<Fr_t> coeff2(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i) {
            const Fr_t a0 = (2U * i < cur_a.size()) ? cur_a[2U * i] : Fr_host_zero();
            const Fr_t b0 = (2U * i < cur_b.size()) ? cur_b[2U * i] : Fr_host_zero();
            const Fr_t a1 = (2U * i + 1U < cur_a.size()) ? cur_a[2U * i + 1U] : Fr_host_zero();
            const Fr_t b1 = (2U * i + 1U < cur_b.size()) ? cur_b[2U * i + 1U] : Fr_host_zero();
            const Fr_t da = Fr_host_sub(a1, a0);
            const Fr_t db = Fr_host_sub(b1, b0);
            coeff0[i] = Fr_host_mul(a0, b0);
            coeff1[i] = Fr_host_add(Fr_host_mul(a0, db), Fr_host_mul(b0, da));
            coeff2[i] = Fr_host_mul(da, db);
        }
        SumcheckQuadraticRound round;
        round.c0 = formal_host_mle_vector(coeff0, proof.point.begin() + round_idx + 1, proof.point.end());
        round.c1 = formal_host_mle_vector(coeff1, proof.point.begin() + round_idx + 1, proof.point.end());
        round.c2 = formal_host_mle_vector(coeff2, proof.point.begin() + round_idx + 1, proof.point.end());
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point[round_idx]))) {
            throw std::runtime_error("CPU hadamard inconsistent round");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("hadamard_fold");
        std::vector<Fr_t> next_a(out_size, Fr_host_zero());
        std::vector<Fr_t> next_b(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i) {
            const Fr_t a0 = (2U * i < cur_a.size()) ? cur_a[2U * i] : Fr_host_zero();
            const Fr_t b0 = (2U * i < cur_b.size()) ? cur_b[2U * i] : Fr_host_zero();
            const Fr_t a1 = (2U * i + 1U < cur_a.size()) ? cur_a[2U * i + 1U] : Fr_host_zero();
            const Fr_t b1 = (2U * i + 1U < cur_b.size()) ? cur_b[2U * i + 1U] : Fr_host_zero();
            next_a[i] = Fr_host_mle_step(a0, a1, challenge);
            next_b[i] = Fr_host_mle_step(b0, b1, challenge);
        }
        running_claim = Fr_host_quadratic_eval(round, challenge);
        proof.rounds.push_back(round);
        proof.challenges.push_back(challenge);
        cur_a.swap(next_a);
        cur_b.swap(next_b);
    }
    proof.final_a = cur_a.empty() ? Fr_host_zero() : cur_a.front();
    proof.final_b = cur_b.empty() ? Fr_host_zero() : cur_b.front();
    proof.eval_a = generators.open_with_proof(a, com_a, proof.challenges, proof.opening_a);
    proof.eval_b = generators.open_with_proof(b, com_b, proof.challenges, proof.opening_b);
    transcript.absorb_fr(proof.final_a);
    transcript.absorb_fr(proof.final_b);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_hadamard_relation_verify(
    const HadamardSumcheckProof& proof,
    const CpuCommitment& generators,
    const CpuG1Tensor& com_a,
    const CpuG1Tensor& com_b,
    const CpuG1Tensor& com_c,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("CPU hadamard label mismatch");
    FormalTranscript transcript;
    cpu_absorb_hadamard_metadata(transcript, relation_label, proof.size, com_a, com_b, com_c);
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.size), "hadamard_output_point"), proof.point)) {
        return fail("CPU hadamard output point replay mismatch");
    }
    if (!CpuCommitment::verify_opening(generators, com_c, proof.opening_c)) return fail("CPU hadamard C opening rejected");
    transcript.absorb_fr(proof.eval_c);
    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < proof.rounds.size(); ++round_idx) {
        const auto& round = proof.rounds[round_idx];
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point[round_idx]))) {
            return fail("CPU hadamard weighted endpoint mismatch");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("hadamard_fold");
        if (!Fr_host_eq(challenge, proof.challenges[round_idx])) return fail("CPU hadamard challenge mismatch");
        running_claim = Fr_host_quadratic_eval(round, challenge);
    }
    if (!Fr_host_eq(running_claim, Fr_host_mul(proof.final_a, proof.final_b))) return fail("CPU hadamard terminal product mismatch");
    if (!CpuCommitment::verify_opening(generators, com_a, proof.opening_a)) return fail("CPU hadamard A opening rejected");
    if (!CpuCommitment::verify_opening(generators, com_b, proof.opening_b)) return fail("CPU hadamard B opening rejected");
    if (!Fr_host_eq(proof.opening_a.value, proof.final_a) || !Fr_host_eq(proof.opening_b.value, proof.final_b)) {
        return fail("CPU hadamard terminal opening value mismatch");
    }
    transcript.absorb_fr(proof.final_a);
    transcript.absorb_fr(proof.final_b);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("CPU hadamard transcript digest mismatch");
    return true;
}

struct CpuFormalRelationRun {
    bool source_ok = false;
    bool destination_ok = false;
    bool grouped_sum_ok = false;
    bool denominator_ok = false;
    bool inverse_ok = false;
    bool inverse_routing_ok = false;
    bool alpha_from_inverse_ok = false;
    bool alpha_times_denominator_ok = false;
    bool weighted_aggregation_ok = false;
    std::size_t available_feature_columns = 0;
    std::size_t verified_feature_columns = 0;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::uint64_t topology_hash = 0;
    std::size_t commitment_block_size = 1;
    fs::path proof_path;
    fs::path evidence_path;
    std::vector<std::string> relation_details;
};

CpuFormalRelationRun run_cpu_formal_relations_with_trace(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const HeadTraceHost& head,
    const fs::path& output_root)
{
    if (head.e_src.size() != graph.num_nodes || head.e_dst.size() != graph.num_nodes
        || head.u.size() != graph.edges.size() || head.sum.size() != graph.num_nodes) {
        throw std::runtime_error("CPU formal trace shape mismatch");
    }
    const std::size_t padded_node_count = next_power_of_two(std::max<std::size_t>(1, graph.num_nodes));
    const std::size_t padded_edge_count = next_power_of_two(std::max<std::size_t>(1, graph.edges.size()));
    const auto src = graph_src_u32(graph);
    const auto dst = graph_dst_u32(graph);
    const uint node_count_u32 = static_cast<uint>(graph.num_nodes);
    const uint edge_count_u32 = static_cast<uint>(graph.edges.size());

    auto node_src_raw = encode_padded_vector_scale16_raw(head.e_src, graph.num_nodes, padded_node_count);
    auto node_dst_raw = encode_padded_vector_scale16_raw(head.e_dst, graph.num_nodes, padded_node_count);
    auto edge_src_raw = route_vector_by_topology_raw(node_src_raw, graph, padded_edge_count, true);
    auto edge_dst_raw = route_vector_by_topology_raw(node_dst_raw, graph, padded_edge_count, false);
    auto grouped_sum_raw = grouped_sum_by_dst_raw(edge_src_raw, graph, padded_node_count);
    std::vector<std::uint32_t> u_quantized;
    auto u_raw = encode_nonnegative_scale16_raw(head.u, graph.edges.size(), padded_edge_count, &u_quantized);
    auto denominator_raw = grouped_sum_by_dst_raw(u_raw, graph, padded_node_count);
    const auto denominator_u64 = grouped_sum_by_dst_u64(u_quantized, graph, padded_node_count);
    auto denominator_mont = mont_vector_from_raw(denominator_raw);
    std::vector<Fr_t> inverse_node_mont(padded_node_count, Fr_host_zero());
    for (std::size_t node = 0; node < graph.num_nodes; ++node) {
        if (denominator_u64[node] == 0) {
            throw std::runtime_error("CPU denominator 为 0，不能生成 inverse relation");
        }
        inverse_node_mont[node] = fr_inverse_uint_mont(static_cast<std::uint32_t>(denominator_u64[node]));
    }
    auto u_mont = mont_vector_from_raw(u_raw);
    auto inverse_edge_mont = route_vector_by_topology_raw(inverse_node_mont, graph, padded_edge_count, false);
    auto denominator_edge_mont = route_vector_by_topology_raw(denominator_mont, graph, padded_edge_count, false);
    auto valid_one_mont = valid_one_mont_vector(graph.num_nodes, padded_node_count);
    std::vector<Fr_t> alpha_mont(padded_edge_count, Fr_host_zero());
    for (std::size_t edge = 0; edge < graph.edges.size(); ++edge) {
        alpha_mont[edge] = Fr_host_mul(u_mont[edge], inverse_edge_mont[edge]);
    }

    CpuFormalRelationRun result;
    result.topology_hash = formal_sparse_topology_hash(src, dst, node_count_u32, edge_count_u32);
    result.commitment_block_size = 1;
    const auto gen = CpuCommitment::deterministic(static_cast<uint>(result.commitment_block_size),
        spec.dataset + ":cpu_formal_block1");

    CpuFrTensor node_src(mont_vector_from_raw(node_src_raw));
    CpuFrTensor node_dst(mont_vector_from_raw(node_dst_raw));
    CpuFrTensor edge_src(mont_vector_from_raw(edge_src_raw));
    CpuFrTensor edge_dst(mont_vector_from_raw(edge_dst_raw));
    CpuFrTensor grouped_input(mont_vector_from_raw(edge_src_raw));
    CpuFrTensor grouped_output(mont_vector_from_raw(grouped_sum_raw));
    CpuFrTensor u_tensor(u_mont);
    CpuFrTensor denominator_tensor(denominator_mont);
    CpuFrTensor denominator_edge_tensor(denominator_edge_mont);
    CpuFrTensor inverse_node_tensor(inverse_node_mont);
    CpuFrTensor inverse_edge_tensor(inverse_edge_mont);
    CpuFrTensor valid_one_tensor(valid_one_mont);
    CpuFrTensor alpha_tensor(alpha_mont);
    ScopedTimer commit_timer;
    const auto com_node_src = gen.commit(node_src);
    const auto com_node_dst = gen.commit(node_dst);
    const auto com_edge_src = gen.commit(edge_src);
    const auto com_edge_dst = gen.commit(edge_dst);
    const auto com_grouped_input = gen.commit(grouped_input);
    const auto com_grouped_output = gen.commit(grouped_output);
    const auto com_u = gen.commit(u_tensor);
    const auto com_denominator = gen.commit(denominator_tensor);
    const auto com_denominator_edge = gen.commit(denominator_edge_tensor);
    const auto com_inverse_node = gen.commit(inverse_node_tensor);
    const auto com_inverse_edge = gen.commit(inverse_edge_tensor);
    const auto com_valid_one = gen.commit(valid_one_tensor);
    const auto com_alpha = gen.commit(alpha_tensor);
    result.commitment_ms = commit_timer.elapsed_ms();

    const std::string prefix = spec.dataset + ":cpu:hidden_layer0_head0";
    const std::string source_label = prefix + ":source_sparse_routing:full_edges";
    const std::string destination_label = prefix + ":destination_sparse_routing:full_edges";
    const std::string grouped_label = prefix + ":grouped_sum_by_dst:source_attention_edge_values";
    const std::string denom_label = prefix + ":softmax_denominator_grouped_sum:U_by_dst";
    const std::string inverse_label = prefix + ":softmax_inverse:denominator_times_inverse";
    const std::string inv_route_label = prefix + ":softmax_inverse:inverse_destination_routing";
    const std::string alpha_from_inv_label = prefix + ":alpha_from_inverse:U_times_inv_edge";
    const std::string alpha_times_denom_label = prefix + ":alpha_times_denom_equals_U";

    SparseRoutingRelationProof source_proof;
    SparseRoutingRelationProof destination_proof;
    GroupedSumRelationProof grouped_proof;
    GroupedSumRelationProof denominator_proof;
    HadamardSumcheckProof inverse_proof;
    SparseRoutingRelationProof inverse_route_proof;
    HadamardSumcheckProof alpha_from_inverse_proof;
    HadamardSumcheckProof alpha_times_denominator_proof;
    {
        ScopedTimer timer;
        source_proof = cpu_sparse_routing_relation_prove(
            node_src, edge_src, gen, com_node_src, gen, com_edge_src, src, dst,
            node_count_u32, edge_count_u32, "source", source_label);
        destination_proof = cpu_sparse_routing_relation_prove(
            node_dst, edge_dst, gen, com_node_dst, gen, com_edge_dst, src, dst,
            node_count_u32, edge_count_u32, "destination", destination_label);
        grouped_proof = cpu_grouped_sum_relation_prove(
            grouped_input, grouped_output, gen, com_grouped_input, gen, com_grouped_output,
            src, dst, node_count_u32, edge_count_u32, grouped_label);
        denominator_proof = cpu_grouped_sum_relation_prove(
            u_tensor, denominator_tensor, gen, com_u, gen, com_denominator,
            src, dst, node_count_u32, edge_count_u32, denom_label);
        inverse_proof = cpu_hadamard_relation_prove(
            denominator_tensor, inverse_node_tensor, valid_one_tensor,
            gen, com_denominator, com_inverse_node, com_valid_one, inverse_label);
        inverse_route_proof = cpu_sparse_routing_relation_prove(
            inverse_node_tensor, inverse_edge_tensor, gen, com_inverse_node, gen, com_inverse_edge,
            src, dst, node_count_u32, edge_count_u32, "destination", inv_route_label);
        alpha_from_inverse_proof = cpu_hadamard_relation_prove(
            u_tensor, inverse_edge_tensor, alpha_tensor,
            gen, com_u, com_inverse_edge, com_alpha, alpha_from_inv_label);
        alpha_times_denominator_proof = cpu_hadamard_relation_prove(
            alpha_tensor, denominator_edge_tensor, u_tensor,
            gen, com_alpha, com_denominator_edge, com_u, alpha_times_denom_label);
        result.prove_ms = timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        result.source_ok = cpu_sparse_routing_relation_verify(
            source_proof, gen, com_node_src, gen, com_edge_src, src, dst, "source", source_label, &failure);
        if (!result.source_ok) throw std::runtime_error("CPU source sparse verifier rejected: " + failure);
        result.destination_ok = cpu_sparse_routing_relation_verify(
            destination_proof, gen, com_node_dst, gen, com_edge_dst, src, dst, "destination", destination_label, &failure);
        if (!result.destination_ok) throw std::runtime_error("CPU destination sparse verifier rejected: " + failure);
        result.grouped_sum_ok = cpu_grouped_sum_relation_verify(
            grouped_proof, gen, com_grouped_input, gen, com_grouped_output, src, dst, grouped_label, &failure);
        if (!result.grouped_sum_ok) throw std::runtime_error("CPU grouped verifier rejected: " + failure);
        result.denominator_ok = cpu_grouped_sum_relation_verify(
            denominator_proof, gen, com_u, gen, com_denominator, src, dst, denom_label, &failure);
        if (!result.denominator_ok) throw std::runtime_error("CPU denominator verifier rejected: " + failure);
        result.inverse_ok = cpu_hadamard_relation_verify(
            inverse_proof, gen, com_denominator, com_inverse_node, com_valid_one, inverse_label, &failure);
        if (!result.inverse_ok) throw std::runtime_error("CPU inverse verifier rejected: " + failure);
        result.inverse_routing_ok = cpu_sparse_routing_relation_verify(
            inverse_route_proof, gen, com_inverse_node, gen, com_inverse_edge,
            src, dst, "destination", inv_route_label, &failure);
        if (!result.inverse_routing_ok) throw std::runtime_error("CPU inverse routing verifier rejected: " + failure);
        result.alpha_from_inverse_ok = cpu_hadamard_relation_verify(
            alpha_from_inverse_proof, gen, com_u, com_inverse_edge, com_alpha, alpha_from_inv_label, &failure);
        if (!result.alpha_from_inverse_ok) throw std::runtime_error("CPU alpha-from-inverse verifier rejected: " + failure);
        result.alpha_times_denominator_ok = cpu_hadamard_relation_verify(
            alpha_times_denominator_proof, gen, com_alpha, com_denominator_edge, com_u, alpha_times_denom_label, &failure);
        if (!result.alpha_times_denominator_ok) throw std::runtime_error("CPU alpha-times-denominator verifier rejected: " + failure);
        result.verify_ms = timer.elapsed_ms();
    }

    result.proof_size_bytes =
        source_proof.serialized_size_bytes()
        + destination_proof.serialized_size_bytes()
        + grouped_proof.serialized_size_bytes()
        + denominator_proof.serialized_size_bytes()
        + inverse_proof.serialized_size_bytes()
        + inverse_route_proof.serialized_size_bytes()
        + alpha_from_inverse_proof.serialized_size_bytes()
        + alpha_times_denominator_proof.serialized_size_bytes();
    result.available_feature_columns = head.width;
    result.relation_details = {
        "source_sparse_routing status=PASS digest=" + source_proof.transcript_digest,
        "destination_sparse_routing status=PASS digest=" + destination_proof.transcript_digest,
        "grouped_sum_by_dst status=PASS digest=" + grouped_proof.transcript_digest,
        "denominator_grouped_sum status=PASS digest=" + denominator_proof.transcript_digest,
        "inverse_denominator_times_inverse status=PASS digest=" + inverse_proof.transcript_digest,
        "inverse_destination_routing status=PASS digest=" + inverse_route_proof.transcript_digest,
        "alpha_from_inverse status=PASS digest=" + alpha_from_inverse_proof.transcript_digest,
        "alpha_times_denominator status=PASS digest=" + alpha_times_denominator_proof.transcript_digest,
    };

    const auto run_dir = device_run_dir(output_root, spec.dataset, "cpu");
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    result.proof_path = run_dir / "formal_artifacts" / "zkdl_cpu_relation_partial.proof.txt";
    {
        std::ofstream out(result.proof_path);
        out << "ZKDLGATCPUFORMAL\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "backend=cpu_only_host_fr_g1_commitment\n";
        out << "relation_scope=hidden_layer0_head0_sparse_softmax_partial\n";
        out << "node_count=" << graph.num_nodes << "\n";
        out << "edge_count=" << graph.edges.size() << "\n";
        out << "padded_node_count=" << padded_node_count << "\n";
        out << "padded_edge_count=" << padded_edge_count << "\n";
        out << "commitment_block_size=" << result.commitment_block_size << "\n";
        out << "topology_hash=" << result.topology_hash << "\n";
        out << "source_sparse_routing_status=PASS\n";
        out << "destination_sparse_routing_status=PASS\n";
        out << "grouped_sum_status=PASS\n";
        out << "denominator_grouped_sum_status=PASS\n";
        out << "inverse_status=PASS\n";
        out << "inverse_destination_routing_status=PASS\n";
        out << "alpha_from_inverse_status=PASS\n";
        out << "alpha_times_denominator_status=PASS\n";
        out << "weighted_aggregation_status=NOT_RUN_CPU_EXTENSION\n";
        out << "available_feature_columns=" << result.available_feature_columns << "\n";
        out << "verified_feature_columns=" << result.verified_feature_columns << "\n";
        out << "proof_size_bytes=" << result.proof_size_bytes << "\n";
        out << "relation_details=" << result.relation_details.size() << "\n";
        for (const auto& detail : result.relation_details) out << detail << "\n";
        out << "说明=CPU proof artifact 只包含 relation metadata、证明大小和 transcript 摘要；不包含完整 witness、完整 tensor 或 debug dump。\n";
    }
    result.evidence_path = output_root / "evidence" / ("cpu_relation_partial_" + spec.dataset + ".json");
    {
        std::ofstream out(result.evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"device\": \"cpu\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"backend\": \"cpu_only_host_fr_g1_commitment\",\n";
        out << "  \"relation_scope\": \"hidden_layer0_head0_sparse_softmax_partial\",\n";
        out << "  \"node_count\": \"" << graph.num_nodes << "\",\n";
        out << "  \"edge_count\": \"" << graph.edges.size() << "\",\n";
        out << "  \"commitment_block_size\": \"" << result.commitment_block_size << "\",\n";
        out << "  \"topology_hash\": \"" << result.topology_hash << "\",\n";
        out << "  \"source_sparse_routing\": \"PASS\",\n";
        out << "  \"destination_sparse_routing\": \"PASS\",\n";
        out << "  \"grouped_sum_by_dst\": \"PASS\",\n";
        out << "  \"denominator_grouped_sum\": \"PASS\",\n";
        out << "  \"inverse\": \"PASS\",\n";
        out << "  \"inverse_destination_routing\": \"PASS\",\n";
        out << "  \"alpha_from_inverse\": \"PASS\",\n";
        out << "  \"alpha_times_denominator\": \"PASS\",\n";
        out << "  \"weighted_aggregation\": \"NOT_RUN_CPU_EXTENSION\",\n";
        out << "  \"available_feature_columns\": \"" << result.available_feature_columns << "\",\n";
        out << "  \"verified_feature_columns\": \"" << result.verified_feature_columns << "\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(result.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(result.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(result.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << result.proof_size_bytes << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(result.proof_path.string()) << "\",\n";
        out << "  \"note\": \"CPU-only hidden layer0/head0 sparse + denominator/inverse/alpha relation-level formal result; weighted aggregation、max/ExpMap/lookup/range/ELU/output aggregation/rescale remain open.\"\n";
        out << "}\n";
    }
    return result;
}
