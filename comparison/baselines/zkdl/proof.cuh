#ifndef PROOF_CUH
#define PROOF_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

#ifndef COMMITMENT_OPENING_PROOF_DEFINED
#define COMMITMENT_OPENING_PROOF_DEFINED
struct CommitmentOpeningProof
{
    vector<Fr_t> point;
    Fr_t value;
    vector<G1Jacobian_t> transcript;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t)
            + point.size() * sizeof(Fr_t)
            + sizeof(Fr_t)
            + sizeof(uint32_t)
            + transcript.size() * sizeof(G1Jacobian_t);
    }
};
#endif

vector<Fr_t> random_vec(uint len)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    vector<Fr_t> out(len);
    for (uint i = 0; i < len; ++ i) out[i] = {dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt) % 1944954707};
    return out;
}

uint ceilLog2(uint num) {
    if (num == 0) return 0;
    
    // Decrease num to handle the case where num is already a power of 2
    num--;

    uint result = 0;
    
    // Keep shifting the number to the right until it becomes zero. 
    // Each shift means the number is halved, which corresponds to 
    // a division by 2 in logarithmic terms.
    while (num > 0) {
        num >>= 1;
        result++;
    }

    return result;
}

template<typename T>
std::vector<T> concatenate(const std::vector<std::vector<T>>& vecs) {
    // First, compute the total size for the result vector.
    size_t totalSize = 0;
    for (const auto& v : vecs) {
        totalSize += v.size();
    }

    // Allocate space for the result vector.
    std::vector<T> result;
    result.reserve(totalSize);

    // Append each vector's contents to the result vector.
    for (const auto& v : vecs) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}


KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    proof.push_back(out0.sum());
    proof.push_back(out1.sum());
    proof.push_back(out2.sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_ip_sc(a_new, b_new, begin + 1, end, proof);
}

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u)
{
    vector<Fr_t> proof;
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_ip_sc(a, b, u.begin(), u.end(), proof);
    return proof;
}

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 5");
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_hp_sc(a_new, b_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions 1");
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 2");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions 3");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions 4");

    Fr_hp_sc(a, b, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    Fr_t a0 = (2 * gid < in_size) ? a[2 * gid] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (2 * gid + 1 < in_size) ? a[2 * gid + 1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(a0, a0), a0);
    Fr_t diff = blstrs__scalar__Scalar_sub(a1, a0);
    out1[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_double(a0), diff), diff);
    out2[gid] = blstrs__scalar__Scalar_sqr(diff);
}

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_bin_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_bin_sc(a_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions");
    uint log_size = u.size();
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_bin_sc(a, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

struct FormalTranscript
{
    std::uint64_t state = 1469598103934665603ULL;

    void absorb_bytes(const void* data, std::size_t n)
    {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            state ^= bytes[i];
            state *= 1099511628211ULL;
        }
    }

    void absorb_label(const std::string& label)
    {
        absorb_bytes(label.data(), label.size());
        const char sep = '\0';
        absorb_bytes(&sep, 1);
    }

    void absorb_u64(std::uint64_t value)
    {
        absorb_bytes(&value, sizeof(value));
    }

    void absorb_fr(const Fr_t& value)
    {
        absorb_bytes(&value, sizeof(value));
    }

    void absorb_g1(const G1Jacobian_t& value)
    {
        absorb_bytes(&value, sizeof(value));
    }

    Fr_t challenge(const std::string& label)
    {
        absorb_label(label);
        std::uint64_t x = state;
        x ^= x >> 33U;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33U;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33U;
        Fr_t raw = Fr_host_zero();
        raw.val[0] = static_cast<uint>(x & 0xffffffffULL);
        raw.val[1] = static_cast<uint>((x >> 32U) & 0xffffffffULL);
        raw.val[2] = static_cast<uint>((state ^ x) & 0xffffffffULL);
        const Fr_t out = Fr_host_mont(raw);
        absorb_fr(out);
        return out;
    }

    std::string digest_hex() const
    {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << state;
        return out.str();
    }
};

struct SumcheckQuadraticRound
{
    Fr_t c0 = Fr_host_zero();
    Fr_t c1 = Fr_host_zero();
    Fr_t c2 = Fr_host_zero();

    std::size_t serialized_size_bytes() const
    {
        return 3 * sizeof(Fr_t);
    }
};

inline Fr_t Fr_host_double_value(const Fr_t& x)
{
    return Fr_host_add(x, x);
}

inline Fr_t Fr_host_quadratic_eval(const SumcheckQuadraticRound& round, const Fr_t& x)
{
    return Fr_host_add(round.c0, Fr_host_add(Fr_host_mul(round.c1, x), Fr_host_mul(round.c2, Fr_host_mul(x, x))));
}

inline Fr_t Fr_host_sum_quadratic_endpoints(const SumcheckQuadraticRound& round)
{
    return Fr_host_add(Fr_host_double_value(round.c0), Fr_host_add(round.c1, round.c2));
}

struct InnerProductSumcheckProof
{
    uint size = 0;
    Fr_t claimed_sum = Fr_host_zero();
    vector<SumcheckQuadraticRound> rounds;
    vector<Fr_t> challenges;
    Fr_t final_a = Fr_host_zero();
    Fr_t final_b = Fr_host_zero();
    CommitmentOpeningProof opening_a;
    CommitmentOpeningProof opening_b;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + sizeof(Fr_t) + sizeof(uint32_t);
        for (const auto& round : rounds) out += round.serialized_size_bytes();
        out += sizeof(uint32_t) + challenges.size() * sizeof(Fr_t);
        out += 2 * sizeof(Fr_t);
        out += opening_a.serialized_size_bytes() + opening_b.serialized_size_bytes();
        out += sizeof(uint64_t);
        return out;
    }
};

struct HadamardSumcheckProof
{
    uint size = 0;
    vector<Fr_t> point;
    Fr_t eval_a = Fr_host_zero();
    Fr_t eval_b = Fr_host_zero();
    Fr_t eval_c = Fr_host_zero();
    CommitmentOpeningProof opening_a;
    CommitmentOpeningProof opening_b;
    CommitmentOpeningProof opening_c;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t)
            + sizeof(uint32_t) + point.size() * sizeof(Fr_t)
            + 3 * sizeof(Fr_t)
            + opening_a.serialized_size_bytes()
            + opening_b.serialized_size_bytes()
            + opening_c.serialized_size_bytes()
            + sizeof(uint64_t);
    }
};

using BinarySumcheckProof = HadamardSumcheckProof;

struct LinearRelationProof
{
    std::string label;
    vector<Fr_t> coefficients;
    vector<CommitmentOpeningProof> openings;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + label.size() + sizeof(uint32_t) + coefficients.size() * sizeof(Fr_t);
        out += sizeof(uint32_t);
        for (const auto& opening : openings) out += opening.serialized_size_bytes();
        out += sizeof(uint64_t);
        return out;
    }
};

struct MatmulRelationProof
{
    std::string label;
    uint num = 0;
    uint m = 0;
    uint n = 0;
    uint k = 0;
    InnerProductSumcheckProof inner_product;
    vector<CommitmentOpeningProof> openings;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + label.size() + 4 * sizeof(uint32_t);
        out += inner_product.serialized_size_bytes();
        out += sizeof(uint32_t);
        for (const auto& opening : openings) out += opening.serialized_size_bytes();
        out += sizeof(uint64_t);
        return out;
    }
};

struct FormalProofObject
{
    std::string protocol = "zkdl-gat-formal-partial";
    std::string metadata;
    vector<G1Jacobian_t> commitments;
    vector<InnerProductSumcheckProof> inner_products;
    vector<HadamardSumcheckProof> hadamards;
    vector<LinearRelationProof> linear_relations;
    vector<MatmulRelationProof> matmuls;
    vector<CommitmentOpeningProof> openings;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + protocol.size() + sizeof(uint32_t) + metadata.size();
        out += sizeof(uint32_t) + commitments.size() * sizeof(G1Jacobian_t);
        out += sizeof(uint32_t);
        for (const auto& proof : inner_products) out += proof.serialized_size_bytes();
        out += sizeof(uint32_t);
        for (const auto& proof : hadamards) out += proof.serialized_size_bytes();
        out += sizeof(uint32_t);
        for (const auto& proof : linear_relations) out += proof.serialized_size_bytes();
        out += sizeof(uint32_t);
        for (const auto& proof : matmuls) out += proof.serialized_size_bytes();
        out += sizeof(uint32_t);
        for (const auto& opening : openings) out += opening.serialized_size_bytes();
        out += sizeof(uint64_t);
        return out;
    }
};

inline Fr_t formal_inner_product_claim_host(const vector<Fr_t>& a, const vector<Fr_t>& b)
{
    if (a.size() != b.size()) throw std::runtime_error("Incompatible dimensions");
    Fr_t out = Fr_host_zero();
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        out = Fr_host_add(out, Fr_host_mul(a[i], b[i]));
    }
    return out;
}

inline InnerProductSumcheckProof formal_inner_product_sumcheck_prove(
    const FrTensor& a,
    const FrTensor& b,
    const std::string& relation_label)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    InnerProductSumcheckProof proof;
    proof.size = a.size;
    vector<Fr_t> cur_a = a.to_host();
    vector<Fr_t> cur_b = b.to_host();

    FormalTranscript transcript;
    transcript.absorb_label("inner_product_sumcheck");
    transcript.absorb_label(relation_label);
    transcript.absorb_u64(proof.size);
    proof.claimed_sum = formal_inner_product_claim_host(cur_a, cur_b);
    transcript.absorb_fr(proof.claimed_sum);

    const uint rounds = ceilLog2(a.size);
    Fr_t running_claim = proof.claimed_sum;
    for (uint round_idx = 0; round_idx < rounds; ++round_idx)
    {
        const uint out_size = static_cast<uint>((cur_a.size() + 1U) / 2U);
        SumcheckQuadraticRound round;
        vector<Fr_t> next_a(out_size, Fr_host_zero());
        vector<Fr_t> next_b(out_size, Fr_host_zero());
        for (uint i = 0; i < out_size; ++i)
        {
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
        if (!Fr_host_eq(running_claim, Fr_host_sum_quadratic_endpoints(round)))
        {
            throw std::runtime_error("inner product prover generated inconsistent round");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("sumcheck_round");
        for (uint i = 0; i < out_size; ++i)
        {
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

inline bool formal_inner_product_sumcheck_verify(
    const InnerProductSumcheckProof& proof,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    FormalTranscript transcript;
    transcript.absorb_label("inner_product_sumcheck");
    transcript.absorb_label(relation_label);
    transcript.absorb_u64(proof.size);
    transcript.absorb_fr(proof.claimed_sum);
    Fr_t running_claim = proof.claimed_sum;
    if (proof.rounds.size() != ceilLog2(proof.size)) return fail("round count mismatch");
    for (uint round_idx = 0; round_idx < proof.rounds.size(); ++round_idx)
    {
        const auto& round = proof.rounds[round_idx];
        if (!Fr_host_eq(running_claim, Fr_host_sum_quadratic_endpoints(round)))
        {
            return fail("sumcheck endpoint mismatch");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("sumcheck_round");
        if (round_idx >= proof.challenges.size() || !Fr_host_eq(challenge, proof.challenges[round_idx]))
        {
            return fail("challenge replay mismatch");
        }
        running_claim = Fr_host_quadratic_eval(round, challenge);
    }
    if (!Fr_host_eq(running_claim, Fr_host_mul(proof.final_a, proof.final_b)))
    {
        return fail("final product mismatch");
    }
    transcript.absorb_fr(proof.final_a);
    transcript.absorb_fr(proof.final_b);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("transcript digest mismatch");
    return true;
}

#endif
