#include "commitment.cuh"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool fr_equal_host(const Fr_t& lhs, const Fr_t& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(Fr_t)) == 0;
}

std::uint64_t fnv1a64(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

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

Commitment deterministic_generators(uint size, const std::string& label)
{
    auto scalars = deterministic_mont_vec(size, "formal-sumcheck-gens:" + label);
    std::vector<Fr_t> raw(scalars.size());
    for (std::size_t i = 0; i < scalars.size(); ++i) {
        raw[i] = Fr_host_unmont(scalars[i]);
    }
    FrTensor scalar_tensor(size, raw.data());
    Commitment out(size, G1Jacobian_generator);
    out *= scalar_tensor;
    return out;
}

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

bool run_hadamard_point_case(const std::string& label, uint size, std::ostream& log)
{
    auto a_host = deterministic_mont_vec(size, label + ":a");
    auto b_host = deterministic_mont_vec(size, label + ":b");
    std::vector<Fr_t> c_host(size, Fr_host_zero());
    for (uint i = 0; i < size; ++i) c_host[i] = Fr_host_mul(a_host[i], b_host[i]);
    FrTensor a(size, a_host.data());
    FrTensor b(size, b_host.data());
    FrTensor c(size, c_host.data());
    Commitment gen(1, G1Jacobian_generator);
    auto com_a = gen.commit(a);
    auto com_b = gen.commit(b);
    auto com_c = gen.commit(c);

    bool passed = true;
    std::string failure;
    try {
        HadamardSumcheckProof proof;
        proof.size = size;
        FormalTranscript transcript;
        transcript.absorb_label("hadamard_point_relation");
        transcript.absorb_label(label);
        transcript.absorb_u64(size);
        const uint rounds = ceilLog2(size);
        for (uint i = 0; i < rounds; ++i) {
            proof.point.push_back(transcript.challenge("hadamard_point"));
        }
        proof.eval_a = gen.open_with_proof(a, com_a, proof.point, proof.opening_a);
        proof.eval_b = gen.open_with_proof(b, com_b, proof.point, proof.opening_b);
        proof.eval_c = gen.open_with_proof(c, com_c, proof.point, proof.opening_c);
        transcript.absorb_fr(proof.eval_a);
        transcript.absorb_fr(proof.eval_b);
        transcript.absorb_fr(proof.eval_c);
        proof.transcript_digest = transcript.digest_hex();
        if (!Commitment::verify_opening(gen, com_a, proof.opening_a)) {
            passed = false;
            failure = "opening_a failed";
        }
        if (passed && !Commitment::verify_opening(gen, com_b, proof.opening_b)) {
            passed = false;
            failure = "opening_b failed";
        }
        if (passed && !Commitment::verify_opening(gen, com_c, proof.opening_c)) {
            passed = false;
            failure = "opening_c failed";
        }
        if (passed && !fr_equal_host(Fr_host_mul(proof.eval_a, proof.eval_b), proof.eval_c)) {
            passed = false;
            failure = "hadamard point relation mismatch";
        }
        log << "HADAMARD_POINT label=" << label
            << " size=" << size
            << " padded_size=" << (std::size_t{1} << ceilLog2(size))
            << " point_dim=" << proof.point.size()
            << " transcript_digest=" << proof.transcript_digest
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " backend=point_relation_with_openings"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "HADAMARD_POINT label=" << label
            << " size=" << size
            << " backend=point_relation_with_openings status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

}  // namespace

int main()
{
    bool ok = true;
    ok = run_inner_product_case("size4_power2", 4, std::cout) && ok;
    ok = run_inner_product_case("size7_non_power2", 7, std::cout) && ok;
    ok = run_inner_product_case("size15_non_power2", 15, std::cout) && ok;
    std::cout << "SUMMARY status=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
