#ifndef PROOF_CUH
#define PROOF_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

#include <vector>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

#ifndef COMMITMENT_OPENING_PROOF_DEFINED
#define COMMITMENT_OPENING_PROOF_DEFINED
// 承诺打开证明记录评值点、声明值和 G1 transcript
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

// 计算 MLE/sumcheck 所需的向上取整 log2
uint ceilLog2(uint num) {
    if (num == 0) return 0;

    // 先减一，使 2^k 的输入仍返回 k。
    num--;

    uint result = 0;

    while (num > 0) {
        num >>= 1;
        result++;
    }

    return result;
}

// Fiat-Shamir transcript吸入标签、承诺和声明后派生 relation 内部挑战值域
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

// 单轮二次多项式声明
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

// 域元素乘以2
inline Fr_t Fr_host_double_value(const Fr_t& x)
{
    return Fr_host_add(x, x);
}

// 在给定挑战上计算一轮二次多项式；verifier 用它重放 sumcheck
inline Fr_t Fr_host_quadratic_eval(const SumcheckQuadraticRound& round, const Fr_t& x)
{
    return Fr_host_add(round.c0, Fr_host_add(Fr_host_mul(round.c1, x), Fr_host_mul(round.c2, Fr_host_mul(x, x))));
}

// 计算 g(0)+g(1)
inline Fr_t Fr_host_sum_quadratic_endpoints(const SumcheckQuadraticRound& round)
{
    return Fr_host_add(Fr_host_double_value(round.c0), Fr_host_add(round.c1, round.c2));
}

// 计算 (1-u)g(0)+u g(1)
inline Fr_t Fr_host_weighted_quadratic_endpoints(const SumcheckQuadraticRound& round, const Fr_t& u)
{
    const Fr_t one_minus_u = Fr_host_sub(Fr_host_one_mont(), u);
    const Fr_t at_zero = round.c0;
    const Fr_t at_one = Fr_host_add(round.c0, Fr_host_add(round.c1, round.c2));
    return Fr_host_add(Fr_host_mul(one_minus_u, at_zero), Fr_host_mul(u, at_one));
}

// 内积
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
    std::string label;
    uint size = 0;
    vector<uint> shape;
    std::string commitment_a_label;
    std::string commitment_b_label;
    std::string commitment_c_label;
    vector<Fr_t> point;
    vector<SumcheckQuadraticRound> rounds;
    vector<Fr_t> challenges;
    Fr_t claimed_eval_c = Fr_host_zero();
    Fr_t eval_a = Fr_host_zero();
    Fr_t eval_b = Fr_host_zero();
    Fr_t eval_c = Fr_host_zero();
    Fr_t final_a = Fr_host_zero();
    Fr_t final_b = Fr_host_zero();
    CommitmentOpeningProof opening_a;
    CommitmentOpeningProof opening_b;
    CommitmentOpeningProof opening_c;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + label.size()
            + sizeof(uint32_t)
            + sizeof(uint32_t) + shape.size() * sizeof(uint32_t)
            + sizeof(uint32_t) + commitment_a_label.size()
            + sizeof(uint32_t) + commitment_b_label.size()
            + sizeof(uint32_t) + commitment_c_label.size()
            + sizeof(uint32_t) + point.size() * sizeof(Fr_t)
            + sizeof(uint32_t);
        for (const auto& round : rounds) out += round.serialized_size_bytes();
        out += sizeof(uint32_t) + challenges.size() * sizeof(Fr_t)
            + 6 * sizeof(Fr_t)
            + opening_a.serialized_size_bytes()
            + opening_b.serialized_size_bytes()
            + opening_c.serialized_size_bytes()
            + sizeof(uint64_t);
        return out;
    }
};

using BinarySumcheckProof = HadamardSumcheckProof;

// 记录多张量线性组合的 opening
struct LinearRelationProof
{
    std::string label;
    vector<uint> shape;
    vector<std::string> tensor_labels;
    vector<Fr_t> coefficients;
    vector<vector<Fr_t>> points;
    vector<Fr_t> evaluations;
    Fr_t public_evaluation = Fr_host_zero();
    Fr_t public_coefficient = Fr_host_zero();
    Fr_t constant = Fr_host_zero();
    vector<CommitmentOpeningProof> openings;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + label.size()
            + sizeof(uint32_t) + shape.size() * sizeof(uint32_t)
            + sizeof(uint32_t);
        for (const auto& tensor_label : tensor_labels) out += sizeof(uint32_t) + tensor_label.size();
        out += sizeof(uint32_t) + coefficients.size() * sizeof(Fr_t);
        out += sizeof(uint32_t);
        for (const auto& point : points) out += sizeof(uint32_t) + point.size() * sizeof(Fr_t);
        out += sizeof(uint32_t) + evaluations.size() * sizeof(Fr_t);
        out += 3 * sizeof(Fr_t);
        out += sizeof(uint32_t);
        for (const auto& opening : openings) out += opening.serialized_size_bytes();
        out += sizeof(uint64_t);
        return out;
    }
};

// 矩阵乘法
struct MatmulRelationProof
{
    std::string label;
    uint num = 0;
    uint m = 0;
    uint n = 0;
    uint k = 0;
    uint padded_num = 0;
    uint padded_m = 0;
    uint padded_n = 0;
    uint padded_k = 0;
    std::string commitment_a_label;
    std::string commitment_b_label;
    std::string commitment_c_label;
    vector<Fr_t> point_num;
    vector<Fr_t> point_m;
    vector<Fr_t> point_k;
    vector<Fr_t> point_c;
    Fr_t eval_c = Fr_host_zero();
    vector<SumcheckQuadraticRound> batch_rounds;
    vector<Fr_t> batch_challenges;
    Fr_t reduced_inner_product_claim = Fr_host_zero();
    InnerProductSumcheckProof inner_product;
    CommitmentOpeningProof opening_a;
    CommitmentOpeningProof opening_b;
    CommitmentOpeningProof opening_c;
    vector<CommitmentOpeningProof> openings;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        std::size_t out = sizeof(uint32_t) + label.size() + 8 * sizeof(uint32_t)
            + sizeof(uint32_t) + commitment_a_label.size()
            + sizeof(uint32_t) + commitment_b_label.size()
            + sizeof(uint32_t) + commitment_c_label.size()
            + sizeof(uint32_t) + point_num.size() * sizeof(Fr_t)
            + sizeof(uint32_t) + point_m.size() * sizeof(Fr_t)
            + sizeof(uint32_t) + point_k.size() * sizeof(Fr_t)
            + sizeof(uint32_t) + point_c.size() * sizeof(Fr_t)
            + sizeof(Fr_t)
            + sizeof(uint32_t);
        for (const auto& round : batch_rounds) out += round.serialized_size_bytes();
        out += sizeof(uint32_t) + batch_challenges.size() * sizeof(Fr_t)
            + sizeof(Fr_t);
        out += inner_product.serialized_size_bytes();
        out += opening_a.serialized_size_bytes()
            + opening_b.serialized_size_bytes()
            + opening_c.serialized_size_bytes();
        out += sizeof(uint32_t);
        for (const auto& opening : openings) out += opening.serialized_size_bytes();
        out += sizeof(uint64_t);
        return out;
    }
};

// 稀疏路由，绑定公开 src/dst 拓扑
struct SparseRoutingRelationProof
{
    std::string label;
    std::string direction;
    std::uint64_t topology_hash = 0;
    uint node_count = 0;
    uint edge_count = 0;
    uint padded_node_count = 0;
    uint padded_edge_count = 0;
    uint edge_begin = 0;
    uint edge_end = 0;
    std::string commitment_node_label;
    std::string commitment_edge_label;
    vector<Fr_t> edge_point;
    Fr_t edge_eval = Fr_host_zero();
    Fr_t public_weight_eval = Fr_host_zero();
    InnerProductSumcheckProof inner_product;
    CommitmentOpeningProof opening_node;
    CommitmentOpeningProof opening_edge;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t) + label.size()
            + sizeof(uint32_t) + direction.size()
            + sizeof(std::uint64_t)
            + 6 * sizeof(uint32_t)
            + sizeof(uint32_t) + commitment_node_label.size()
            + sizeof(uint32_t) + commitment_edge_label.size()
            + sizeof(uint32_t) + edge_point.size() * sizeof(Fr_t)
            + 2 * sizeof(Fr_t)
            + inner_product.serialized_size_bytes()
            + opening_node.serialized_size_bytes()
            + opening_edge.serialized_size_bytes()
            + sizeof(uint64_t);
    }
};

// 按目标节点聚合，用于softmax分母和聚合输出绑定
struct GroupedSumRelationProof
{
    std::string label;
    std::uint64_t topology_hash = 0;
    uint node_count = 0;
    uint edge_count = 0;
    uint padded_node_count = 0;
    uint padded_edge_count = 0;
    uint edge_begin = 0;
    uint edge_end = 0;
    std::string commitment_edge_input_label;
    std::string commitment_node_output_label;
    vector<Fr_t> node_point;
    Fr_t output_eval = Fr_host_zero();
    Fr_t public_weight_eval = Fr_host_zero();
    InnerProductSumcheckProof inner_product;
    CommitmentOpeningProof opening_edge_input;
    CommitmentOpeningProof opening_node_output;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t) + label.size()
            + sizeof(std::uint64_t)
            + 6 * sizeof(uint32_t)
            + sizeof(uint32_t) + commitment_edge_input_label.size()
            + sizeof(uint32_t) + commitment_node_output_label.size()
            + sizeof(uint32_t) + node_point.size() * sizeof(Fr_t)
            + 2 * sizeof(Fr_t)
            + inner_product.serialized_size_bytes()
            + opening_edge_input.serialized_size_bytes()
            + opening_node_output.serialized_size_bytes()
            + sizeof(uint64_t);
    }
};

struct MultiColumnGroupedSumRelationProof
{
    std::string label;
    std::uint64_t topology_hash = 0;
    uint node_count = 0;
    uint edge_count = 0;
    uint column_count = 0;
    uint padded_node_count = 0;
    uint padded_edge_count = 0;
    uint padded_column_count = 0;
    std::string commitment_edge_matrix_label;
    std::string commitment_node_matrix_label;
    std::string commitment_compressed_edge_label;
    std::string commitment_compressed_node_label;
    vector<Fr_t> column_point;
    vector<G1Jacobian_t> compressed_edge_commitment_points;
    vector<G1Jacobian_t> compressed_node_commitment_points;
    GroupedSumRelationProof compressed_grouped_sum;
    Fr_t edge_matrix_eval = Fr_host_zero();
    Fr_t node_matrix_eval = Fr_host_zero();
    CommitmentOpeningProof opening_edge_matrix;
    CommitmentOpeningProof opening_node_matrix;
    std::string transcript_digest;

    std::size_t serialized_size_bytes() const
    {
        return sizeof(uint32_t) + label.size()
            + sizeof(std::uint64_t)
            + 6 * sizeof(uint32_t)
            + sizeof(uint32_t) + commitment_edge_matrix_label.size()
            + sizeof(uint32_t) + commitment_node_matrix_label.size()
            + sizeof(uint32_t) + commitment_compressed_edge_label.size()
            + sizeof(uint32_t) + commitment_compressed_node_label.size()
            + sizeof(uint32_t) + column_point.size() * sizeof(Fr_t)
            + sizeof(uint32_t) + compressed_edge_commitment_points.size() * sizeof(G1Jacobian_t)
            + sizeof(uint32_t) + compressed_node_commitment_points.size() * sizeof(G1Jacobian_t)
            + compressed_grouped_sum.serialized_size_bytes()
            + 2 * sizeof(Fr_t)
            + opening_edge_matrix.serialized_size_bytes()
            + opening_node_matrix.serialized_size_bytes()
            + sizeof(uint64_t);
    }
};

// 在 CPU 上计算内积
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

// 生成 formal inner-product sumcheck proof；写入 rounds/challenges，不写 official manifest。
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

// 验证 inner-product sumcheck transcript；只重放承诺评值声明，不读取 witness。
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
