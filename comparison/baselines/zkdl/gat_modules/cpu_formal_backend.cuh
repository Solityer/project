#pragma once

// CPU-only zkDL-native relation backend：复用本地 Hyrax-style tensor
// commitment/opening 语义，但不依赖 FrTensor/G1Tensor CUDA storage，
// 因此 CPU 结果与 GPU partial artifact 保持独立。

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

bool cpu_raw_fr_is_zero(const Fr_t& value)
{
    for (uint i = 0; i < 8; ++i) {
        if (value.val[i] != 0U) return false;
    }
    return true;
}

bool cpu_raw_fr_is_one(const Fr_t& value)
{
    if (value.val[0] != 1U) return false;
    for (uint i = 1; i < 8; ++i) {
        if (value.val[i] != 0U) return false;
    }
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
        const std::size_t cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t worker_count = std::min<std::size_t>(rows == 0 ? 1 : rows, cpu_count);
        auto commit_rows = [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            G1Jacobian_t acc = G1Jacobian_host_zero();
            for (std::size_t col = 0; col < generators.size(); ++col) {
                const Fr_t raw = Fr_host_unmont(tensor.values[row * generators.size() + col]);
                if (cpu_raw_fr_is_zero(raw)) {
                    continue;
                }
                const G1Jacobian_t term = cpu_raw_fr_is_one(raw)
                    ? generators[col]
                    : cpu_g1_mul_fr_raw(generators[col], raw);
                acc = G1Jacobian_host_add(acc, term);
            }
            out[row] = acc;
        }
        };
        if (worker_count <= 1 || rows < 1024) {
            commit_rows(0, rows);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            const std::size_t chunk = (rows + worker_count - 1) / worker_count;
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                const std::size_t begin = worker * chunk;
                const std::size_t end = std::min(rows, begin + chunk);
                if (begin >= end) break;
                workers.emplace_back(commit_rows, begin, end);
            }
            for (auto& worker : workers) worker.join();
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

