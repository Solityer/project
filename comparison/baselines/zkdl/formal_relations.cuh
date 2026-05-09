#ifndef FORMAL_RELATIONS_CUH
#define FORMAL_RELATIONS_CUH

#include "commitment.cuh"

#include <algorithm>
#include <limits>
#include <memory>

inline uint formal_next_power_of_two_u32(uint value)
{
    if (value <= 1U) return 1U;
    uint out = 1U;
    while (out < value)
    {
        if (out > (std::numeric_limits<uint>::max() >> 1U))
        {
            throw std::runtime_error("power-of-two padding overflows uint");
        }
        out <<= 1U;
    }
    return out;
}

inline void formal_absorb_g1_tensor(FormalTranscript& transcript, const G1TensorJacobian& commitment)
{
    transcript.absorb_u64(commitment.size);
    const auto points = Commitment::host_points(commitment);
    for (const auto& point : points) transcript.absorb_g1(point);
}

inline void formal_absorb_string_vector(FormalTranscript& transcript, const vector<std::string>& values)
{
    transcript.absorb_u64(values.size());
    for (const auto& value : values) transcript.absorb_label(value);
}

inline vector<Fr_t> formal_make_challenge_point(FormalTranscript& transcript, uint dim, const std::string& label)
{
    vector<Fr_t> out;
    out.reserve(dim);
    for (uint i = 0; i < dim; ++i)
    {
        transcript.absorb_u64(i);
        out.push_back(transcript.challenge(label));
    }
    return out;
}

inline vector<Fr_t> formal_concat_points(const vector<vector<Fr_t>>& points)
{
    vector<Fr_t> out;
    for (const auto& point : points) out.insert(out.end(), point.begin(), point.end());
    return out;
}

inline bool formal_points_equal(const vector<Fr_t>& lhs, const vector<Fr_t>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        if (!Fr_host_eq(lhs[i], rhs[i])) return false;
    }
    return true;
}

inline Fr_t formal_host_mle_vector(const vector<Fr_t>& values, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end)
{
    vector<Fr_t> current = values;
    for (auto it = begin; it != end; ++it)
    {
        const uint out_size = static_cast<uint>((current.size() + 1U) / 2U);
        vector<Fr_t> next(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i)
        {
            const uint low_idx = 2U * i;
            const uint high_idx = low_idx + 1U;
            const Fr_t low = low_idx < current.size() ? current[low_idx] : Fr_host_zero();
            const Fr_t high = high_idx < current.size() ? current[high_idx] : Fr_host_zero();
            next[i] = Fr_host_mle_step(low, high, *it);
        }
        current.swap(next);
    }
    return current.empty() ? Fr_host_zero() : current.front();
}

inline vector<Fr_t> formal_partial_mle_by_window_host(
    const vector<Fr_t>& values,
    vector<Fr_t>::const_iterator begin,
    vector<Fr_t>::const_iterator end,
    uint window_size)
{
    if (window_size == 0) throw std::runtime_error("invalid zero window size");
    vector<Fr_t> current = values;
    for (auto it = begin; it != end; ++it)
    {
        const uint in_size = static_cast<uint>(current.size());
        const uint out_size = Fr_partial_me_out_size(in_size, window_size);
        vector<Fr_t> next(out_size, Fr_host_zero());
        for (uint gid = 0; gid < out_size; ++gid)
        {
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

inline Fr_t formal_host_sum(const vector<Fr_t>& values)
{
    Fr_t out = Fr_host_zero();
    for (const auto& value : values) out = Fr_host_add(out, value);
    return out;
}

inline void formal_absorb_hadamard_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    uint size,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c)
{
    transcript.absorb_label("hadamard_relation");
    transcript.absorb_label(label);
    transcript.absorb_u64(size);
    formal_absorb_g1_tensor(transcript, com_a);
    formal_absorb_g1_tensor(transcript, com_b);
    formal_absorb_g1_tensor(transcript, com_c);
}

inline HadamardSumcheckProof formal_hadamard_relation_prove(
    const FrTensor& a,
    const FrTensor& b,
    const FrTensor& c,
    const Commitment& generators,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c,
    const std::string& relation_label)
{
    if (a.size != b.size || a.size != c.size) throw std::runtime_error("hadamard relation size mismatch");
    HadamardSumcheckProof proof;
    proof.label = relation_label;
    proof.size = a.size;
    proof.shape = {a.size};
    proof.commitment_a_label = relation_label + ":A";
    proof.commitment_b_label = relation_label + ":B";
    proof.commitment_c_label = relation_label + ":C";

    FormalTranscript transcript;
    formal_absorb_hadamard_metadata(transcript, relation_label, proof.size, com_a, com_b, com_c);
    const uint rounds = ceilLog2(proof.size);
    proof.point = formal_make_challenge_point(transcript, rounds, "hadamard_output_point");
    proof.eval_c = generators.open_with_proof(c, com_c, proof.point, proof.opening_c);
    proof.claimed_eval_c = proof.eval_c;
    transcript.absorb_fr(proof.eval_c);

    vector<Fr_t> cur_a = a.to_host();
    vector<Fr_t> cur_b = b.to_host();
    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < rounds; ++round_idx)
    {
        const uint out_size = static_cast<uint>((cur_a.size() + 1U) / 2U);
        vector<Fr_t> coeff0(out_size, Fr_host_zero());
        vector<Fr_t> coeff1(out_size, Fr_host_zero());
        vector<Fr_t> coeff2(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i)
        {
            const uint low_idx = 2U * i;
            const uint high_idx = low_idx + 1U;
            const Fr_t a0 = low_idx < cur_a.size() ? cur_a[low_idx] : Fr_host_zero();
            const Fr_t b0 = low_idx < cur_b.size() ? cur_b[low_idx] : Fr_host_zero();
            const Fr_t a1 = high_idx < cur_a.size() ? cur_a[high_idx] : Fr_host_zero();
            const Fr_t b1 = high_idx < cur_b.size() ? cur_b[high_idx] : Fr_host_zero();
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
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point[round_idx])))
        {
            throw std::runtime_error("hadamard prover generated inconsistent weighted endpoint");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("hadamard_fold");

        vector<Fr_t> next_a(out_size, Fr_host_zero());
        vector<Fr_t> next_b(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i)
        {
            const uint low_idx = 2U * i;
            const uint high_idx = low_idx + 1U;
            const Fr_t a0 = low_idx < cur_a.size() ? cur_a[low_idx] : Fr_host_zero();
            const Fr_t b0 = low_idx < cur_b.size() ? cur_b[low_idx] : Fr_host_zero();
            const Fr_t a1 = high_idx < cur_a.size() ? cur_a[high_idx] : Fr_host_zero();
            const Fr_t b1 = high_idx < cur_b.size() ? cur_b[high_idx] : Fr_host_zero();
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

inline bool formal_hadamard_relation_verify(
    const HadamardSumcheckProof& proof,
    const Commitment& generators,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.size == 0) return fail("empty hadamard relation");
    if (proof.label != relation_label) return fail("relation label mismatch");
    const uint rounds = ceilLog2(proof.size);
    if (proof.point.size() != rounds || proof.rounds.size() != rounds || proof.challenges.size() != rounds)
    {
        return fail("hadamard round/point count mismatch");
    }

    FormalTranscript transcript;
    formal_absorb_hadamard_metadata(transcript, relation_label, proof.size, com_a, com_b, com_c);
    const auto expected_point = formal_make_challenge_point(transcript, rounds, "hadamard_output_point");
    if (!formal_points_equal(expected_point, proof.point)) return fail("hadamard output point replay mismatch");
    if (!formal_points_equal(proof.opening_c.point, proof.point)) return fail("hadamard C opening point mismatch");
    if (!Fr_host_eq(proof.opening_c.value, proof.eval_c)) return fail("hadamard C opening value mismatch");
    if (!Fr_host_eq(proof.claimed_eval_c, proof.eval_c)) return fail("hadamard claimed C evaluation mismatch");
    if (!Commitment::verify_opening(generators, com_c, proof.opening_c)) return fail("hadamard C opening rejected");
    transcript.absorb_fr(proof.eval_c);

    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < rounds; ++round_idx)
    {
        const auto& round = proof.rounds[round_idx];
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point[round_idx])))
        {
            return fail("hadamard weighted endpoint mismatch");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("hadamard_fold");
        if (!Fr_host_eq(challenge, proof.challenges[round_idx])) return fail("hadamard challenge replay mismatch");
        running_claim = Fr_host_quadratic_eval(round, challenge);
    }
    if (!Fr_host_eq(running_claim, Fr_host_mul(proof.final_a, proof.final_b)))
    {
        return fail("hadamard terminal product mismatch");
    }
    if (!formal_points_equal(proof.opening_a.point, proof.challenges)
        || !formal_points_equal(proof.opening_b.point, proof.challenges))
    {
        return fail("hadamard terminal opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_a.value, proof.final_a)
        || !Fr_host_eq(proof.opening_b.value, proof.final_b))
    {
        return fail("hadamard terminal opening value mismatch");
    }
    if (!Commitment::verify_opening(generators, com_a, proof.opening_a)) return fail("hadamard A opening rejected");
    if (!Commitment::verify_opening(generators, com_b, proof.opening_b)) return fail("hadamard B opening rejected");

    transcript.absorb_fr(proof.final_a);
    transcript.absorb_fr(proof.final_b);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("hadamard transcript digest mismatch");
    return true;
}

inline vector<Fr_t> formal_matmul_reduce_a_over_m(
    const vector<Fr_t>& a,
    uint padded_num,
    uint padded_m,
    uint padded_n,
    const vector<Fr_t>& point_m)
{
    if (a.size() != static_cast<std::size_t>(padded_num) * padded_m * padded_n)
    {
        throw std::runtime_error("matmul A padded shape mismatch");
    }
    vector<Fr_t> out(static_cast<std::size_t>(padded_num) * padded_n, Fr_host_zero());
    vector<Fr_t> column(padded_m, Fr_host_zero());
    for (uint batch = 0; batch < padded_num; ++batch)
    {
        for (uint t = 0; t < padded_n; ++t)
        {
            for (uint row = 0; row < padded_m; ++row)
            {
                column[row] = a[(static_cast<std::size_t>(batch) * padded_m + row) * padded_n + t];
            }
            out[static_cast<std::size_t>(batch) * padded_n + t] =
                formal_host_mle_vector(column, point_m.begin(), point_m.end());
        }
    }
    return out;
}

inline vector<Fr_t> formal_matmul_reduce_b_over_k(
    const vector<Fr_t>& b,
    uint padded_num,
    uint padded_n,
    uint padded_k,
    const vector<Fr_t>& point_k)
{
    if (b.size() != static_cast<std::size_t>(padded_num) * padded_n * padded_k)
    {
        throw std::runtime_error("matmul B padded shape mismatch");
    }
    vector<Fr_t> out(static_cast<std::size_t>(padded_num) * padded_n, Fr_host_zero());
    vector<Fr_t> row_values(padded_k, Fr_host_zero());
    for (uint batch = 0; batch < padded_num; ++batch)
    {
        for (uint t = 0; t < padded_n; ++t)
        {
            for (uint col = 0; col < padded_k; ++col)
            {
                row_values[col] = b[(static_cast<std::size_t>(batch) * padded_n + t) * padded_k + col];
            }
            out[static_cast<std::size_t>(batch) * padded_n + t] =
                formal_host_mle_vector(row_values, point_k.begin(), point_k.end());
        }
    }
    return out;
}

inline void formal_absorb_matmul_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    uint num,
    uint m,
    uint n,
    uint k,
    uint padded_num,
    uint padded_m,
    uint padded_n,
    uint padded_k,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c)
{
    transcript.absorb_label("matmul_relation");
    transcript.absorb_label(label);
    transcript.absorb_u64(num);
    transcript.absorb_u64(m);
    transcript.absorb_u64(n);
    transcript.absorb_u64(k);
    transcript.absorb_u64(padded_num);
    transcript.absorb_u64(padded_m);
    transcript.absorb_u64(padded_n);
    transcript.absorb_u64(padded_k);
    formal_absorb_g1_tensor(transcript, com_a);
    formal_absorb_g1_tensor(transcript, com_b);
    formal_absorb_g1_tensor(transcript, com_c);
}

inline MatmulRelationProof formal_matmul_relation_prove(
    const FrTensor& a,
    const FrTensor& b,
    const FrTensor& c,
    const Commitment& generators,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c,
    uint num,
    uint m,
    uint n,
    uint k,
    uint padded_num,
    uint padded_m,
    uint padded_n,
    uint padded_k,
    const std::string& relation_label)
{
    if (a.size != padded_num * padded_m * padded_n
        || b.size != padded_num * padded_n * padded_k
        || c.size != padded_num * padded_m * padded_k)
    {
        throw std::runtime_error("matmul relation padded tensor size mismatch");
    }
    MatmulRelationProof proof;
    proof.label = relation_label;
    proof.num = num;
    proof.m = m;
    proof.n = n;
    proof.k = k;
    proof.padded_num = padded_num;
    proof.padded_m = padded_m;
    proof.padded_n = padded_n;
    proof.padded_k = padded_k;
    proof.commitment_a_label = relation_label + ":A";
    proof.commitment_b_label = relation_label + ":B";
    proof.commitment_c_label = relation_label + ":C";

    FormalTranscript transcript;
    formal_absorb_matmul_metadata(transcript, relation_label, num, m, n, k,
        padded_num, padded_m, padded_n, padded_k, com_a, com_b, com_c);
    proof.point_num = formal_make_challenge_point(transcript, ceilLog2(padded_num), "matmul_num_point");
    proof.point_m = formal_make_challenge_point(transcript, ceilLog2(padded_m), "matmul_m_point");
    proof.point_k = formal_make_challenge_point(transcript, ceilLog2(padded_k), "matmul_k_point");
    proof.point_c = formal_concat_points({proof.point_k, proof.point_m, proof.point_num});
    proof.eval_c = generators.open_with_proof(c, com_c, proof.point_c, proof.opening_c);
    transcript.absorb_fr(proof.eval_c);

    vector<Fr_t> reduced_a = formal_matmul_reduce_a_over_m(a.to_host(), padded_num, padded_m, padded_n, proof.point_m);
    vector<Fr_t> reduced_b = formal_matmul_reduce_b_over_k(b.to_host(), padded_num, padded_n, padded_k, proof.point_k);
    uint cur_num = padded_num;
    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < proof.point_num.size(); ++round_idx)
    {
        const uint out_num = (cur_num + 1U) / 2U;
        vector<Fr_t> coeff0(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        vector<Fr_t> coeff1(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        vector<Fr_t> coeff2(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        for (uint batch = 0; batch < out_num; ++batch)
        {
            for (uint t = 0; t < padded_n; ++t)
            {
                const uint low_batch = 2U * batch;
                const uint high_batch = low_batch + 1U;
                const Fr_t a0 = low_batch < cur_num ? reduced_a[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b0 = low_batch < cur_num ? reduced_b[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t a1 = high_batch < cur_num ? reduced_a[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b1 = high_batch < cur_num ? reduced_b[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t da = Fr_host_sub(a1, a0);
                const Fr_t db = Fr_host_sub(b1, b0);
                const std::size_t idx = static_cast<std::size_t>(batch) * padded_n + t;
                coeff0[idx] = Fr_host_mul(a0, b0);
                coeff1[idx] = Fr_host_add(Fr_host_mul(a0, db), Fr_host_mul(b0, da));
                coeff2[idx] = Fr_host_mul(da, db);
            }
        }
        const auto rest0 = formal_partial_mle_by_window_host(
            coeff0, proof.point_num.begin() + round_idx + 1, proof.point_num.end(), padded_n);
        const auto rest1 = formal_partial_mle_by_window_host(
            coeff1, proof.point_num.begin() + round_idx + 1, proof.point_num.end(), padded_n);
        const auto rest2 = formal_partial_mle_by_window_host(
            coeff2, proof.point_num.begin() + round_idx + 1, proof.point_num.end(), padded_n);
        SumcheckQuadraticRound round;
        round.c0 = formal_host_sum(rest0);
        round.c1 = formal_host_sum(rest1);
        round.c2 = formal_host_sum(rest2);
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point_num[round_idx])))
        {
            throw std::runtime_error("matmul prover generated inconsistent batch round");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("matmul_batch_fold");

        vector<Fr_t> next_a(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        vector<Fr_t> next_b(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        for (uint batch = 0; batch < out_num; ++batch)
        {
            for (uint t = 0; t < padded_n; ++t)
            {
                const uint low_batch = 2U * batch;
                const uint high_batch = low_batch + 1U;
                const Fr_t a0 = low_batch < cur_num ? reduced_a[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b0 = low_batch < cur_num ? reduced_b[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t a1 = high_batch < cur_num ? reduced_a[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b1 = high_batch < cur_num ? reduced_b[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const std::size_t idx = static_cast<std::size_t>(batch) * padded_n + t;
                next_a[idx] = Fr_host_mle_step(a0, a1, challenge);
                next_b[idx] = Fr_host_mle_step(b0, b1, challenge);
            }
        }
        running_claim = Fr_host_quadratic_eval(round, challenge);
        proof.batch_rounds.push_back(round);
        proof.batch_challenges.push_back(challenge);
        reduced_a.swap(next_a);
        reduced_b.swap(next_b);
        cur_num = out_num;
    }
    if (reduced_a.size() != padded_n || reduced_b.size() != padded_n)
    {
        throw std::runtime_error("matmul batch reduction did not end at inner dimension");
    }

    FrTensor reduced_a_tensor(padded_n, reduced_a.data());
    FrTensor reduced_b_tensor(padded_n, reduced_b.data());
    proof.inner_product = formal_inner_product_sumcheck_prove(reduced_a_tensor, reduced_b_tensor, relation_label + ":inner");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, running_claim))
    {
        throw std::runtime_error("matmul inner-product claim mismatch");
    }
    proof.reduced_inner_product_claim = running_claim;
    proof.inner_product.opening_a = CommitmentOpeningProof{};
    proof.inner_product.opening_b = CommitmentOpeningProof{};
    const auto point_a = formal_concat_points({proof.inner_product.challenges, proof.point_m, proof.batch_challenges});
    const auto point_b = formal_concat_points({proof.point_k, proof.inner_product.challenges, proof.batch_challenges});
    proof.inner_product.final_a = generators.open_with_proof(a, com_a, point_a, proof.opening_a);
    proof.inner_product.final_b = generators.open_with_proof(b, com_b, point_b, proof.opening_b);
    proof.inner_product.opening_a = proof.opening_a;
    proof.inner_product.opening_b = proof.opening_b;
    proof.openings = {proof.opening_a, proof.opening_b, proof.opening_c};
    transcript.absorb_label(proof.inner_product.transcript_digest);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

inline bool formal_matmul_relation_verify(
    const MatmulRelationProof& proof,
    const Commitment& generators,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("matmul relation label mismatch");
    if (proof.padded_num == 0 || proof.padded_m == 0 || proof.padded_n == 0 || proof.padded_k == 0)
    {
        return fail("matmul invalid padded shape");
    }
    if (proof.point_num.size() != ceilLog2(proof.padded_num)
        || proof.point_m.size() != ceilLog2(proof.padded_m)
        || proof.point_k.size() != ceilLog2(proof.padded_k)
        || proof.batch_rounds.size() != proof.point_num.size()
        || proof.batch_challenges.size() != proof.point_num.size())
    {
        return fail("matmul point/round count mismatch");
    }

    FormalTranscript transcript;
    formal_absorb_matmul_metadata(transcript, relation_label, proof.num, proof.m, proof.n, proof.k,
        proof.padded_num, proof.padded_m, proof.padded_n, proof.padded_k, com_a, com_b, com_c);
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_num), "matmul_num_point"), proof.point_num))
    {
        return fail("matmul num point replay mismatch");
    }
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_m), "matmul_m_point"), proof.point_m))
    {
        return fail("matmul m point replay mismatch");
    }
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_k), "matmul_k_point"), proof.point_k))
    {
        return fail("matmul k point replay mismatch");
    }
    const auto expected_c_point = formal_concat_points({proof.point_k, proof.point_m, proof.point_num});
    if (!formal_points_equal(expected_c_point, proof.point_c)) return fail("matmul C point mismatch");
    if (!formal_points_equal(proof.opening_c.point, proof.point_c)) return fail("matmul C opening point mismatch");
    if (!Fr_host_eq(proof.opening_c.value, proof.eval_c)) return fail("matmul C opening value mismatch");
    if (!Commitment::verify_opening(generators, com_c, proof.opening_c)) return fail("matmul C opening rejected");
    transcript.absorb_fr(proof.eval_c);

    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < proof.batch_rounds.size(); ++round_idx)
    {
        const auto& round = proof.batch_rounds[round_idx];
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point_num[round_idx])))
        {
            return fail("matmul batch weighted endpoint mismatch");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("matmul_batch_fold");
        if (!Fr_host_eq(challenge, proof.batch_challenges[round_idx])) return fail("matmul batch challenge mismatch");
        running_claim = Fr_host_quadratic_eval(round, challenge);
    }
    if (!Fr_host_eq(proof.reduced_inner_product_claim, running_claim)) return fail("matmul reduced claim mismatch");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, running_claim)) return fail("matmul inner claimed sum mismatch");
    std::string inner_failure;
    if (!formal_inner_product_sumcheck_verify(proof.inner_product, relation_label + ":inner", &inner_failure))
    {
        return fail("matmul inner sumcheck failed: " + inner_failure);
    }
    const auto expected_a_point = formal_concat_points({proof.inner_product.challenges, proof.point_m, proof.batch_challenges});
    const auto expected_b_point = formal_concat_points({proof.point_k, proof.inner_product.challenges, proof.batch_challenges});
    if (!formal_points_equal(proof.opening_a.point, expected_a_point)) return fail("matmul A opening point mismatch");
    if (!formal_points_equal(proof.opening_b.point, expected_b_point)) return fail("matmul B opening point mismatch");
    if (!Fr_host_eq(proof.opening_a.value, proof.inner_product.final_a)
        || !Fr_host_eq(proof.opening_b.value, proof.inner_product.final_b))
    {
        return fail("matmul terminal opening value mismatch");
    }
    if (!Commitment::verify_opening(generators, com_a, proof.opening_a)) return fail("matmul A opening rejected");
    if (!Commitment::verify_opening(generators, com_b, proof.opening_b)) return fail("matmul B opening rejected");
    transcript.absorb_label(proof.inner_product.transcript_digest);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("matmul transcript digest mismatch");
    return true;
}

struct FormalLinearTerm
{
    std::string label;
    const FrTensor* tensor = nullptr;
    const G1TensorJacobian* commitment = nullptr;
    Fr_t coefficient = Fr_host_zero();
    vector<Fr_t> point;
};

inline void formal_absorb_linear_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    const vector<uint>& shape,
    const vector<std::string>& tensor_labels,
    const vector<Fr_t>& coefficients,
    const vector<vector<Fr_t>>& points,
    const Fr_t& public_coefficient,
    const Fr_t& public_evaluation,
    const Fr_t& constant,
    const vector<const G1TensorJacobian*>& commitments)
{
    transcript.absorb_label("linear_relation");
    transcript.absorb_label(label);
    transcript.absorb_u64(shape.size());
    for (const auto dim : shape) transcript.absorb_u64(dim);
    formal_absorb_string_vector(transcript, tensor_labels);
    transcript.absorb_u64(coefficients.size());
    for (const auto& coeff : coefficients) transcript.absorb_fr(coeff);
    transcript.absorb_u64(points.size());
    for (const auto& point : points)
    {
        transcript.absorb_u64(point.size());
        for (const auto& coord : point) transcript.absorb_fr(coord);
    }
    transcript.absorb_fr(public_coefficient);
    transcript.absorb_fr(public_evaluation);
    transcript.absorb_fr(constant);
    transcript.absorb_u64(commitments.size());
    for (const auto* commitment : commitments)
    {
        if (commitment == nullptr) throw std::runtime_error("null linear commitment");
        formal_absorb_g1_tensor(transcript, *commitment);
    }
}

inline LinearRelationProof formal_linear_relation_prove(
    const std::string& relation_label,
    const vector<uint>& shape,
    const vector<FormalLinearTerm>& terms,
    const Commitment& generators,
    const Fr_t& public_coefficient,
    const Fr_t& public_evaluation,
    const Fr_t& constant)
{
    if (terms.empty()) throw std::runtime_error("linear relation needs at least one committed term");
    LinearRelationProof proof;
    proof.label = relation_label;
    proof.shape = shape;
    proof.public_coefficient = public_coefficient;
    proof.public_evaluation = public_evaluation;
    proof.constant = constant;
    vector<const G1TensorJacobian*> commitments;
    for (const auto& term : terms)
    {
        if (term.tensor == nullptr || term.commitment == nullptr) throw std::runtime_error("linear relation null term");
        proof.tensor_labels.push_back(term.label);
        proof.coefficients.push_back(term.coefficient);
        proof.points.push_back(term.point);
        commitments.push_back(term.commitment);
    }
    FormalTranscript transcript;
    formal_absorb_linear_metadata(transcript, relation_label, shape, proof.tensor_labels, proof.coefficients,
        proof.points, public_coefficient, public_evaluation, constant, commitments);
    proof.openings.resize(terms.size());
    proof.evaluations.resize(terms.size(), Fr_host_zero());
    for (std::size_t i = 0; i < terms.size(); ++i)
    {
        proof.evaluations[i] = generators.open_with_proof(*terms[i].tensor, *terms[i].commitment, terms[i].point, proof.openings[i]);
        transcript.absorb_fr(proof.evaluations[i]);
    }
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

inline bool formal_linear_relation_verify(
    const LinearRelationProof& proof,
    const Commitment& generators,
    const vector<const G1TensorJacobian*>& commitments,
    const Fr_t& verifier_public_evaluation,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (commitments.size() != proof.coefficients.size()
        || proof.openings.size() != proof.coefficients.size()
        || proof.evaluations.size() != proof.coefficients.size()
        || proof.points.size() != proof.coefficients.size()
        || proof.tensor_labels.size() != proof.coefficients.size())
    {
        return fail("linear relation vector size mismatch");
    }
    if (!Fr_host_eq(proof.public_evaluation, verifier_public_evaluation))
    {
        return fail("linear public evaluation mismatch");
    }
    FormalTranscript transcript;
    formal_absorb_linear_metadata(transcript, proof.label, proof.shape, proof.tensor_labels, proof.coefficients,
        proof.points, proof.public_coefficient, proof.public_evaluation, proof.constant, commitments);
    Fr_t acc = proof.constant;
    acc = Fr_host_add(acc, Fr_host_mul(proof.public_coefficient, verifier_public_evaluation));
    for (std::size_t i = 0; i < proof.coefficients.size(); ++i)
    {
        if (commitments[i] == nullptr) return fail("linear null verifier commitment");
        if (!formal_points_equal(proof.openings[i].point, proof.points[i])) return fail("linear opening point mismatch");
        if (!Fr_host_eq(proof.openings[i].value, proof.evaluations[i])) return fail("linear opening value mismatch");
        if (!Commitment::verify_opening(generators, *commitments[i], proof.openings[i])) return fail("linear opening rejected");
        acc = Fr_host_add(acc, Fr_host_mul(proof.coefficients[i], proof.evaluations[i]));
        transcript.absorb_fr(proof.evaluations[i]);
    }
    if (!Fr_host_eq(acc, Fr_host_zero())) return fail("linear relation equation mismatch");
    if (proof.transcript_digest != transcript.digest_hex()) return fail("linear transcript digest mismatch");
    return true;
}

#endif
