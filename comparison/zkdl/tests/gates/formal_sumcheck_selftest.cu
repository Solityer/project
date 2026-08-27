#include "support/commitment.cuh"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

// 比较 Fr 原始编码；Gate 2 用它确认 opening 末端值匹配。
bool fr_equal_host(const Fr_t& lhs, const Fr_t& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(Fr_t)) == 0;
}

// FNV-1a 哈希；输入标签，输出确定性测试种子。
std::uint64_t fnv1a64(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// 将小整数编码到 raw Fr；用于构造可复现的 sumcheck witness。
Fr_t fr_from_signed_host(std::int64_t value)
{
    Fr_t out = Fr_host_zero();
    const Fr_t p = Fr_host_modulus();
    if (value >= 0) {
        std::uint64_t v = static_cast<std::uint64_t>(value);
        out.val[0] = static_cast<uint>(v & 0xffffffffULL);
        out.val[1] = static_cast<uint>((v >> 32U) & 0xffffffffULL);
        return out;
    }
    Fr_t abs_value = Fr_host_zero();
    const std::uint64_t v = static_cast<std::uint64_t>(-value);
    abs_value.val[0] = static_cast<uint>(v & 0xffffffffULL);
    abs_value.val[1] = static_cast<uint>((v >> 32U) & 0xffffffffULL);
    return Fr_host_sub(p, abs_value);
}

// 生成 Montgomery Fr 向量；只服务 Gate 2，不进入 official manifest。
std::vector<Fr_t> deterministic_mont_vec(std::size_t n, const std::string& label)
{
    std::mt19937_64 rng(fnv1a64(label));
    std::vector<Fr_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto value = static_cast<std::int64_t>((rng() % 31ULL) - 15ULL);
        out[i] = Fr_host_mont(fr_from_signed_host(value));
    }
    return out;
}

// 运行 inner-product sumcheck 自测；验证 transcript 与 opening proof 均可通过。
bool run_inner_product_case(const std::string& label, uint size, std::ostream& log)
{
    auto a_host = deterministic_mont_vec(size, label + ":a");
    auto b_host = deterministic_mont_vec(size, label + ":b");
    FrTensor a(size, a_host.data());
    FrTensor b(size, b_host.data());
    Commitment gen(1, G1Jacobian_generator);
    auto com_a = gen.commit(a);
    auto com_b = gen.commit(b);

    bool passed = true;
    std::string failure;
    try {
        auto proof = formal_inner_product_sumcheck_prove(a, b, label);
        proof.opening_a = CommitmentOpeningProof{};
        proof.opening_b = CommitmentOpeningProof{};
        gen.open_with_proof(a, com_a, proof.challenges, proof.opening_a);
        gen.open_with_proof(b, com_b, proof.challenges, proof.opening_b);
        std::string verify_failure;
        passed = formal_inner_product_sumcheck_verify(proof, label, &verify_failure);
        if (!passed) failure = verify_failure;
        if (passed && !Commitment::verify_opening(gen, com_a, proof.opening_a)) {
            passed = false;
            failure = "opening_a failed";
        }
        if (passed && !Commitment::verify_opening(gen, com_b, proof.opening_b)) {
            passed = false;
            failure = "opening_b failed";
        }
        if (passed && (!fr_equal_host(proof.opening_a.value, proof.final_a) || !fr_equal_host(proof.opening_b.value, proof.final_b))) {
            passed = false;
            failure = "final opening value mismatch";
        }
        log << "INNER_PRODUCT label=" << label
            << " size=" << size
            << " padded_size=" << (std::size_t{1} << ceilLog2(size))
            << " rounds=" << proof.rounds.size()
            << " transcript_digest=" << proof.transcript_digest
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " backend=cpu_sumcheck_hybrid_opening"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "INNER_PRODUCT label=" << label
            << " size=" << size
            << " backend=cpu_sumcheck_hybrid_opening status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

}  // namespace

// Gate 2 入口；全部 sumcheck 子项通过时返回 0。
int main()
{
    bool ok = true;
    ok = run_inner_product_case("size4_power2", 4, std::cout) && ok;
    ok = run_inner_product_case("size7_non_power2", 7, std::cout) && ok;
    ok = run_inner_product_case("size15_non_power2", 15, std::cout) && ok;
    std::cout << "SUMMARY status=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
