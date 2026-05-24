#ifndef FORMAL_RELATIONS_CUH
#define FORMAL_RELATIONS_CUH

#include "commitment.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

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

inline vector<Fr_t> formal_mle_basis_weights_host(const vector<Fr_t>& point)
{
    vector<Fr_t> weights(std::size_t{1} << point.size(), Fr_host_zero());
    weights[0] = Fr_host_one_mont();
    std::size_t active = 1;
    for (const auto& u : point)
    {
        const Fr_t one_minus_u = Fr_host_sub(Fr_host_one_mont(), u);
        for (std::size_t i = 0; i < active; ++i)
        {
            const Fr_t base = weights[i];
            weights[i] = Fr_host_mul(base, one_minus_u);
            weights[i + active] = Fr_host_mul(base, u);
        }
        active <<= 1U;
    }
    return weights;
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
    const auto row_weights = formal_mle_basis_weights_host(point_m);
    if (row_weights.size() != padded_m) throw std::runtime_error("matmul A row basis size mismatch");
    for (uint batch = 0; batch < padded_num; ++batch)
    {
        for (uint t = 0; t < padded_n; ++t)
        {
            Fr_t acc = Fr_host_zero();
            for (uint row = 0; row < padded_m; ++row)
            {
                const auto value = a[(static_cast<std::size_t>(batch) * padded_m + row) * padded_n + t];
                acc = Fr_host_add(acc, Fr_host_mul(value, row_weights[row]));
            }
            out[static_cast<std::size_t>(batch) * padded_n + t] = acc;
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
    const auto column_weights = formal_mle_basis_weights_host(point_k);
    if (column_weights.size() != padded_k) throw std::runtime_error("matmul B column basis size mismatch");
    for (uint batch = 0; batch < padded_num; ++batch)
    {
        for (uint t = 0; t < padded_n; ++t)
        {
            Fr_t acc = Fr_host_zero();
            for (uint col = 0; col < padded_k; ++col)
            {
                const auto value = b[(static_cast<std::size_t>(batch) * padded_n + t) * padded_k + col];
                acc = Fr_host_add(acc, Fr_host_mul(value, column_weights[col]));
            }
            out[static_cast<std::size_t>(batch) * padded_n + t] = acc;
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

inline void formal_hash_absorb_u64(std::uint64_t* state, std::uint64_t value)
{
    for (uint i = 0; i < 8U; ++i)
    {
        const unsigned char byte = static_cast<unsigned char>((value >> (8U * i)) & 0xffU);
        *state ^= byte;
        *state *= 1099511628211ULL;
    }
}

inline std::uint64_t formal_sparse_topology_hash(
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count)
{
    if (src.size() < edge_count || dst.size() < edge_count)
    {
        throw std::runtime_error("sparse topology vector is shorter than edge_count");
    }
    std::uint64_t state = 1469598103934665603ULL;
    formal_hash_absorb_u64(&state, node_count);
    formal_hash_absorb_u64(&state, edge_count);
    for (uint i = 0; i < edge_count; ++i)
    {
        formal_hash_absorb_u64(&state, src[i]);
        formal_hash_absorb_u64(&state, dst[i]);
    }
    return state;
}

inline Fr_t formal_eq_basis_eval(const vector<Fr_t>& point, uint index)
{
    Fr_t acc = Fr_host_one_mont();
    for (uint bit = 0; bit < point.size(); ++bit)
    {
        const bool one = ((index >> bit) & 1U) != 0U;
        const Fr_t factor = one ? point[bit] : Fr_host_sub(Fr_host_one_mont(), point[bit]);
        acc = Fr_host_mul(acc, factor);
    }
    return acc;
}

inline vector<Fr_t> formal_sparse_route_public_coefficients(
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    uint padded_node_count,
    uint padded_edge_count,
    const vector<Fr_t>& edge_point,
    const std::string& direction)
{
    if (src.size() < edge_count || dst.size() < edge_count) throw std::runtime_error("route topology vector too short");
    if (edge_point.size() != ceilLog2(padded_edge_count)) throw std::runtime_error("route edge point dimension mismatch");
    vector<Fr_t> coeff(padded_node_count, Fr_host_zero());
    for (uint edge = 0; edge < edge_count; ++edge)
    {
        const std::uint32_t node = direction == "source" ? src[edge] : dst[edge];
        if (node >= node_count) throw std::runtime_error("route topology node index out of range");
        coeff[node] = Fr_host_add(coeff[node], formal_eq_basis_eval(edge_point, edge));
    }
    return coeff;
}

inline vector<Fr_t> formal_grouped_sum_public_coefficients(
    const vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    uint padded_edge_count,
    const vector<Fr_t>& node_point)
{
    if (dst.size() < edge_count) throw std::runtime_error("grouped-sum topology vector too short");
    if (node_point.size() != ceilLog2(formal_next_power_of_two_u32(node_count)))
    {
        throw std::runtime_error("grouped-sum node point dimension mismatch");
    }
    vector<Fr_t> coeff(padded_edge_count, Fr_host_zero());
    for (uint edge = 0; edge < edge_count; ++edge)
    {
        if (dst[edge] >= node_count) throw std::runtime_error("grouped-sum dst index out of range");
        coeff[edge] = formal_eq_basis_eval(node_point, dst[edge]);
    }
    return coeff;
}

inline void formal_absorb_sparse_routing_metadata(
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
    const G1TensorJacobian& com_node,
    const G1TensorJacobian& com_edge)
{
    transcript.absorb_label("sparse_routing_relation");
    transcript.absorb_label(label);
    transcript.absorb_label(direction);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(edge_begin);
    transcript.absorb_u64(edge_end);
    formal_absorb_g1_tensor(transcript, com_node);
    formal_absorb_g1_tensor(transcript, com_edge);
}

inline void formal_absorb_grouped_sum_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint edge_begin,
    uint edge_end,
    const G1TensorJacobian& com_edge_input,
    const G1TensorJacobian& com_node_output)
{
    transcript.absorb_label("grouped_sum_relation");
    transcript.absorb_label(label);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(edge_begin);
    transcript.absorb_u64(edge_end);
    formal_absorb_g1_tensor(transcript, com_edge_input);
    formal_absorb_g1_tensor(transcript, com_node_output);
}

inline SparseRoutingRelationProof formal_sparse_routing_relation_prove(
    const FrTensor& node_values,
    const FrTensor& edge_values,
    const Commitment& node_generators,
    const G1TensorJacobian& com_node,
    const Commitment& edge_generators,
    const G1TensorJacobian& com_edge,
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    const std::string& direction,
    const std::string& relation_label)
{
    if (direction != "source" && direction != "destination") throw std::runtime_error("sparse routing direction must be source or destination");
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (node_values.size != padded_node_count || edge_values.size != padded_edge_count)
    {
        throw std::runtime_error("sparse routing tensor size mismatch");
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
    formal_absorb_sparse_routing_metadata(transcript, relation_label, direction, proof.topology_hash,
        node_count, edge_count, padded_node_count, padded_edge_count, proof.edge_begin, proof.edge_end, com_node, com_edge);
    proof.edge_point = formal_make_challenge_point(transcript, ceilLog2(padded_edge_count), "sparse_route_edge_point");
    proof.edge_eval = edge_generators.open_with_proof(edge_values, com_edge, proof.edge_point, proof.opening_edge);
    transcript.absorb_fr(proof.edge_eval);

    const auto coeff = formal_sparse_route_public_coefficients(
        src, dst, node_count, edge_count, padded_node_count, padded_edge_count, proof.edge_point, direction);
    FrTensor public_coeff(static_cast<uint>(coeff.size()), coeff.data());
    proof.inner_product = formal_inner_product_sumcheck_prove(public_coeff, node_values, relation_label + ":inner");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.edge_eval))
    {
        throw std::runtime_error("sparse routing witness does not satisfy public topology relation");
    }
    proof.public_weight_eval = formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end());
    if (!Fr_host_eq(proof.public_weight_eval, proof.inner_product.final_a))
    {
        throw std::runtime_error("sparse routing public coefficient terminal mismatch");
    }
    const Fr_t node_eval = node_generators.open_with_proof(
        node_values, com_node, proof.inner_product.challenges, proof.opening_node);
    if (!Fr_host_eq(node_eval, proof.inner_product.final_b))
    {
        throw std::runtime_error("sparse routing node opening mismatch");
    }
    transcript.absorb_label(proof.inner_product.transcript_digest);
    transcript.absorb_fr(proof.public_weight_eval);
    transcript.absorb_fr(node_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

inline bool formal_sparse_routing_relation_verify(
    const SparseRoutingRelationProof& proof,
    const Commitment& node_generators,
    const G1TensorJacobian& com_node,
    const Commitment& edge_generators,
    const G1TensorJacobian& com_edge,
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    const std::string& direction,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("sparse routing label mismatch");
    if (proof.direction != direction) return fail("sparse routing direction mismatch");
    if (direction != "source" && direction != "destination") return fail("invalid sparse routing direction");
    if (proof.node_count == 0 || proof.edge_count == 0) return fail("empty sparse routing relation");
    if (proof.padded_node_count != formal_next_power_of_two_u32(proof.node_count)
        || proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count))
    {
        return fail("sparse routing padded shape mismatch");
    }
    if (proof.edge_begin != 0 || proof.edge_end != proof.edge_count) return fail("sparse routing verifier currently expects full edge range");
    const auto expected_hash = formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count);
    if (proof.topology_hash != expected_hash) return fail("sparse routing topology hash mismatch");

    FormalTranscript transcript;
    formal_absorb_sparse_routing_metadata(transcript, relation_label, direction, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.padded_node_count, proof.padded_edge_count,
        proof.edge_begin, proof.edge_end, com_node, com_edge);
    const auto expected_edge_point = formal_make_challenge_point(transcript, ceilLog2(proof.padded_edge_count), "sparse_route_edge_point");
    if (!formal_points_equal(expected_edge_point, proof.edge_point)) return fail("sparse routing edge point replay mismatch");
    if (!formal_points_equal(proof.opening_edge.point, proof.edge_point)) return fail("sparse routing edge opening point mismatch");
    if (!Fr_host_eq(proof.opening_edge.value, proof.edge_eval)) return fail("sparse routing edge opening value mismatch");
    if (!Commitment::verify_opening(edge_generators, com_edge, proof.opening_edge)) return fail("sparse routing edge opening rejected");
    transcript.absorb_fr(proof.edge_eval);

    const auto coeff = formal_sparse_route_public_coefficients(
        src, dst, proof.node_count, proof.edge_count, proof.padded_node_count, proof.padded_edge_count, proof.edge_point, direction);
    if (proof.inner_product.size != proof.padded_node_count) return fail("sparse routing inner-product size mismatch");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.edge_eval)) return fail("sparse routing claim does not match edge evaluation");
    std::string inner_failure;
    if (!formal_inner_product_sumcheck_verify(proof.inner_product, relation_label + ":inner", &inner_failure))
    {
        return fail("sparse routing inner sumcheck failed: " + inner_failure);
    }
    const Fr_t public_eval = formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end());
    if (!Fr_host_eq(public_eval, proof.public_weight_eval)) return fail("sparse routing stored public coefficient evaluation mismatch");
    if (!Fr_host_eq(public_eval, proof.inner_product.final_a)) return fail("sparse routing terminal public coefficient mismatch");
    if (!formal_points_equal(proof.opening_node.point, proof.inner_product.challenges)) return fail("sparse routing node opening point mismatch");
    if (!Fr_host_eq(proof.opening_node.value, proof.inner_product.final_b)) return fail("sparse routing node opening value mismatch");
    if (!Commitment::verify_opening(node_generators, com_node, proof.opening_node)) return fail("sparse routing node opening rejected");

    transcript.absorb_label(proof.inner_product.transcript_digest);
    transcript.absorb_fr(proof.public_weight_eval);
    transcript.absorb_fr(proof.opening_node.value);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("sparse routing transcript digest mismatch");
    return true;
}

inline GroupedSumRelationProof formal_grouped_sum_relation_prove(
    const FrTensor& edge_input,
    const FrTensor& node_output,
    const Commitment& edge_generators,
    const G1TensorJacobian& com_edge_input,
    const Commitment& node_generators,
    const G1TensorJacobian& com_node_output,
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    const std::string& relation_label)
{
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (edge_input.size != padded_edge_count || node_output.size != padded_node_count)
    {
        throw std::runtime_error("grouped-sum tensor size mismatch");
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
    formal_absorb_grouped_sum_metadata(transcript, relation_label, proof.topology_hash,
        node_count, edge_count, padded_node_count, padded_edge_count, proof.edge_begin, proof.edge_end,
        com_edge_input, com_node_output);
    proof.node_point = formal_make_challenge_point(transcript, ceilLog2(padded_node_count), "grouped_sum_node_point");
    proof.output_eval = node_generators.open_with_proof(node_output, com_node_output, proof.node_point, proof.opening_node_output);
    transcript.absorb_fr(proof.output_eval);

    const auto coeff = formal_grouped_sum_public_coefficients(dst, node_count, edge_count, padded_edge_count, proof.node_point);
    FrTensor public_coeff(static_cast<uint>(coeff.size()), coeff.data());
    proof.inner_product = formal_inner_product_sumcheck_prove(public_coeff, edge_input, relation_label + ":inner");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.output_eval))
    {
        throw std::runtime_error("grouped-sum witness does not satisfy public dst grouping relation");
    }
    proof.public_weight_eval = formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end());
    if (!Fr_host_eq(proof.public_weight_eval, proof.inner_product.final_a))
    {
        throw std::runtime_error("grouped-sum public coefficient terminal mismatch");
    }
    const Fr_t edge_eval = edge_generators.open_with_proof(
        edge_input, com_edge_input, proof.inner_product.challenges, proof.opening_edge_input);
    if (!Fr_host_eq(edge_eval, proof.inner_product.final_b))
    {
        throw std::runtime_error("grouped-sum edge opening mismatch");
    }
    transcript.absorb_label(proof.inner_product.transcript_digest);
    transcript.absorb_fr(proof.public_weight_eval);
    transcript.absorb_fr(edge_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

inline bool formal_grouped_sum_relation_verify(
    const GroupedSumRelationProof& proof,
    const Commitment& edge_generators,
    const G1TensorJacobian& com_edge_input,
    const Commitment& node_generators,
    const G1TensorJacobian& com_node_output,
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("grouped-sum label mismatch");
    if (proof.node_count == 0 || proof.edge_count == 0) return fail("empty grouped-sum relation");
    if (proof.padded_node_count != formal_next_power_of_two_u32(proof.node_count)
        || proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count))
    {
        return fail("grouped-sum padded shape mismatch");
    }
    if (proof.edge_begin != 0 || proof.edge_end != proof.edge_count) return fail("grouped-sum verifier currently expects full edge range");
    const auto expected_hash = formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count);
    if (proof.topology_hash != expected_hash) return fail("grouped-sum topology hash mismatch");

    FormalTranscript transcript;
    formal_absorb_grouped_sum_metadata(transcript, relation_label, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.padded_node_count, proof.padded_edge_count,
        proof.edge_begin, proof.edge_end, com_edge_input, com_node_output);
    const auto expected_node_point = formal_make_challenge_point(transcript, ceilLog2(proof.padded_node_count), "grouped_sum_node_point");
    if (!formal_points_equal(expected_node_point, proof.node_point)) return fail("grouped-sum node point replay mismatch");
    if (!formal_points_equal(proof.opening_node_output.point, proof.node_point)) return fail("grouped-sum output opening point mismatch");
    if (!Fr_host_eq(proof.opening_node_output.value, proof.output_eval)) return fail("grouped-sum output opening value mismatch");
    if (!Commitment::verify_opening(node_generators, com_node_output, proof.opening_node_output)) return fail("grouped-sum output opening rejected");
    transcript.absorb_fr(proof.output_eval);

    const auto coeff = formal_grouped_sum_public_coefficients(dst, proof.node_count, proof.edge_count, proof.padded_edge_count, proof.node_point);
    if (proof.inner_product.size != proof.padded_edge_count) return fail("grouped-sum inner-product size mismatch");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, proof.output_eval)) return fail("grouped-sum claim does not match output evaluation");
    std::string inner_failure;
    if (!formal_inner_product_sumcheck_verify(proof.inner_product, relation_label + ":inner", &inner_failure))
    {
        return fail("grouped-sum inner sumcheck failed: " + inner_failure);
    }
    const Fr_t public_eval = formal_host_mle_vector(coeff, proof.inner_product.challenges.begin(), proof.inner_product.challenges.end());
    if (!Fr_host_eq(public_eval, proof.public_weight_eval)) return fail("grouped-sum stored public coefficient evaluation mismatch");
    if (!Fr_host_eq(public_eval, proof.inner_product.final_a)) return fail("grouped-sum terminal public coefficient mismatch");
    if (!formal_points_equal(proof.opening_edge_input.point, proof.inner_product.challenges)) return fail("grouped-sum edge opening point mismatch");
    if (!Fr_host_eq(proof.opening_edge_input.value, proof.inner_product.final_b)) return fail("grouped-sum edge opening value mismatch");
    if (!Commitment::verify_opening(edge_generators, com_edge_input, proof.opening_edge_input)) return fail("grouped-sum edge opening rejected");

    transcript.absorb_label(proof.inner_product.transcript_digest);
    transcript.absorb_fr(proof.public_weight_eval);
    transcript.absorb_fr(proof.opening_edge_input.value);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("grouped-sum transcript digest mismatch");
    return true;
}

inline vector<Fr_t> formal_matrix_column_compression_host(
    const FrTensor& matrix,
    uint logical_rows,
    uint padded_rows,
    uint logical_columns,
    uint padded_columns,
    const vector<Fr_t>& column_point)
{
    if (logical_rows > padded_rows || logical_columns > padded_columns)
    {
        throw std::runtime_error("multi-column compression logical shape exceeds padded shape");
    }
    if (matrix.size != padded_rows * padded_columns)
    {
        throw std::runtime_error("multi-column compression matrix size mismatch");
    }
    if (column_point.size() != ceilLog2(padded_columns))
    {
        throw std::runtime_error("multi-column compression column point dimension mismatch");
    }
    const auto host = matrix.to_host();
    vector<Fr_t> out(padded_rows, Fr_host_zero());
    vector<Fr_t> row_values(padded_columns, Fr_host_zero());
    for (uint row = 0; row < padded_rows; ++row)
    {
        for (uint col = 0; col < padded_columns; ++col)
        {
            row_values[col] = host[static_cast<std::size_t>(row) * padded_columns + col];
        }
        out[row] = formal_host_mle_vector(row_values, column_point.begin(), column_point.end());
    }
    return out;
}

inline void formal_absorb_multi_column_grouped_sum_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint padded_column_count,
    const G1TensorJacobian& com_edge_matrix,
    const G1TensorJacobian& com_node_matrix)
{
    transcript.absorb_label("multi_column_grouped_sum_relation");
    transcript.absorb_label(label);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(column_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(padded_column_count);
    formal_absorb_g1_tensor(transcript, com_edge_matrix);
    formal_absorb_g1_tensor(transcript, com_node_matrix);
}

inline MultiColumnGroupedSumRelationProof formal_multi_column_grouped_sum_relation_prove(
    const FrTensor& edge_matrix,
    const FrTensor& node_matrix,
    const Commitment& edge_matrix_generators,
    const G1TensorJacobian& com_edge_matrix,
    const Commitment& node_matrix_generators,
    const G1TensorJacobian& com_node_matrix,
    const Commitment& vector_generators,
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_column_count,
    const std::string& relation_label)
{
    (void)src;
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (padded_column_count != formal_next_power_of_two_u32(column_count))
    {
        throw std::runtime_error("multi-column grouped-sum padded column mismatch");
    }
    if (edge_matrix.size != padded_edge_count * padded_column_count
        || node_matrix.size != padded_node_count * padded_column_count)
    {
        throw std::runtime_error("multi-column grouped-sum matrix size mismatch");
    }
    MultiColumnGroupedSumRelationProof proof;
    proof.label = relation_label;
    proof.topology_hash = formal_sparse_topology_hash(src, dst, node_count, edge_count);
    proof.node_count = node_count;
    proof.edge_count = edge_count;
    proof.column_count = column_count;
    proof.padded_node_count = padded_node_count;
    proof.padded_edge_count = padded_edge_count;
    proof.padded_column_count = padded_column_count;
    proof.commitment_edge_matrix_label = relation_label + ":edge_matrix";
    proof.commitment_node_matrix_label = relation_label + ":node_matrix";
    proof.commitment_compressed_edge_label = relation_label + ":compressed_edge";
    proof.commitment_compressed_node_label = relation_label + ":compressed_node";

    FormalTranscript transcript;
    formal_absorb_multi_column_grouped_sum_metadata(
        transcript,
        relation_label,
        proof.topology_hash,
        node_count,
        edge_count,
        column_count,
        padded_node_count,
        padded_edge_count,
        padded_column_count,
        com_edge_matrix,
        com_node_matrix);
    proof.column_point = formal_make_challenge_point(
        transcript,
        ceilLog2(padded_column_count),
        "multi_column_grouped_sum_column_point");

    const auto compressed_edge_host = formal_matrix_column_compression_host(
        edge_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    const auto compressed_node_host = formal_matrix_column_compression_host(
        node_matrix, node_count, padded_node_count, column_count, padded_column_count, proof.column_point);
    FrTensor compressed_edge(static_cast<uint>(compressed_edge_host.size()), compressed_edge_host.data());
    FrTensor compressed_node(static_cast<uint>(compressed_node_host.size()), compressed_node_host.data());
    const auto com_compressed_edge = vector_generators.commit(compressed_edge);
    const auto com_compressed_node = vector_generators.commit(compressed_node);
    proof.compressed_edge_commitment_points = Commitment::host_points(com_compressed_edge);
    proof.compressed_node_commitment_points = Commitment::host_points(com_compressed_node);

    proof.compressed_grouped_sum = formal_grouped_sum_relation_prove(
        compressed_edge,
        compressed_node,
        vector_generators,
        com_compressed_edge,
        vector_generators,
        com_compressed_node,
        src,
        dst,
        node_count,
        edge_count,
        relation_label + ":compressed_grouped_sum");

    const auto edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_grouped_sum.opening_edge_input.point});
    const auto node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_grouped_sum.opening_node_output.point});
    proof.edge_matrix_eval = edge_matrix_generators.open_with_proof(
        edge_matrix, com_edge_matrix, edge_matrix_point, proof.opening_edge_matrix);
    proof.node_matrix_eval = node_matrix_generators.open_with_proof(
        node_matrix, com_node_matrix, node_matrix_point, proof.opening_node_matrix);
    if (!Fr_host_eq(proof.edge_matrix_eval, proof.compressed_grouped_sum.opening_edge_input.value)
        || !Fr_host_eq(proof.node_matrix_eval, proof.compressed_grouped_sum.opening_node_output.value))
    {
        throw std::runtime_error("multi-column grouped-sum compression binding mismatch");
    }
    transcript.absorb_label(proof.compressed_grouped_sum.transcript_digest);
    transcript.absorb_fr(proof.edge_matrix_eval);
    transcript.absorb_fr(proof.node_matrix_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

inline bool formal_multi_column_grouped_sum_relation_verify(
    const MultiColumnGroupedSumRelationProof& proof,
    const Commitment& edge_matrix_generators,
    const G1TensorJacobian& com_edge_matrix,
    const Commitment& node_matrix_generators,
    const G1TensorJacobian& com_node_matrix,
    const Commitment& vector_generators,
    const vector<std::uint32_t>& src,
    const vector<std::uint32_t>& dst,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("multi-column grouped-sum label mismatch");
    if (proof.node_count == 0 || proof.edge_count == 0 || proof.column_count == 0)
    {
        return fail("empty multi-column grouped-sum relation");
    }
    if (proof.padded_node_count != formal_next_power_of_two_u32(proof.node_count)
        || proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count)
        || proof.padded_column_count != formal_next_power_of_two_u32(proof.column_count))
    {
        return fail("multi-column grouped-sum padded shape mismatch");
    }
    const auto expected_hash = formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count);
    if (proof.topology_hash != expected_hash) return fail("multi-column grouped-sum topology hash mismatch");

    FormalTranscript transcript;
    formal_absorb_multi_column_grouped_sum_metadata(
        transcript,
        relation_label,
        proof.topology_hash,
        proof.node_count,
        proof.edge_count,
        proof.column_count,
        proof.padded_node_count,
        proof.padded_edge_count,
        proof.padded_column_count,
        com_edge_matrix,
        com_node_matrix);
    const auto expected_column_point = formal_make_challenge_point(
        transcript,
        ceilLog2(proof.padded_column_count),
        "multi_column_grouped_sum_column_point");
    if (!formal_points_equal(expected_column_point, proof.column_point))
    {
        return fail("multi-column grouped-sum column point replay mismatch");
    }

    G1TensorJacobian com_compressed_edge(
        static_cast<uint>(proof.compressed_edge_commitment_points.size()),
        proof.compressed_edge_commitment_points.data());
    G1TensorJacobian com_compressed_node(
        static_cast<uint>(proof.compressed_node_commitment_points.size()),
        proof.compressed_node_commitment_points.data());
    std::string grouped_failure;
    if (!formal_grouped_sum_relation_verify(
            proof.compressed_grouped_sum,
            vector_generators,
            com_compressed_edge,
            vector_generators,
            com_compressed_node,
            src,
            dst,
            relation_label + ":compressed_grouped_sum",
            &grouped_failure))
    {
        return fail("multi-column grouped-sum compressed verifier failed: " + grouped_failure);
    }

    const auto expected_edge_matrix_point = formal_concat_points({
        proof.column_point,
        proof.compressed_grouped_sum.opening_edge_input.point});
    const auto expected_node_matrix_point = formal_concat_points({
        proof.column_point,
        proof.compressed_grouped_sum.opening_node_output.point});
    if (!formal_points_equal(proof.opening_edge_matrix.point, expected_edge_matrix_point))
    {
        return fail("multi-column grouped-sum edge matrix opening point mismatch");
    }
    if (!formal_points_equal(proof.opening_node_matrix.point, expected_node_matrix_point))
    {
        return fail("multi-column grouped-sum node matrix opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_edge_matrix.value, proof.edge_matrix_eval)
        || !Fr_host_eq(proof.opening_node_matrix.value, proof.node_matrix_eval))
    {
        return fail("multi-column grouped-sum matrix opening value mismatch");
    }
    if (!Fr_host_eq(proof.edge_matrix_eval, proof.compressed_grouped_sum.opening_edge_input.value)
        || !Fr_host_eq(proof.node_matrix_eval, proof.compressed_grouped_sum.opening_node_output.value))
    {
        return fail("multi-column grouped-sum compression binding mismatch");
    }
    if (!Commitment::verify_opening(edge_matrix_generators, com_edge_matrix, proof.opening_edge_matrix))
    {
        return fail("multi-column grouped-sum edge matrix opening rejected");
    }
    if (!Commitment::verify_opening(node_matrix_generators, com_node_matrix, proof.opening_node_matrix))
    {
        return fail("multi-column grouped-sum node matrix opening rejected");
    }

    transcript.absorb_label(proof.compressed_grouped_sum.transcript_digest);
    transcript.absorb_fr(proof.edge_matrix_eval);
    transcript.absorb_fr(proof.node_matrix_eval);
    if (proof.transcript_digest != transcript.digest_hex())
    {
        return fail("multi-column grouped-sum transcript digest mismatch");
    }
    return true;
}

#endif
