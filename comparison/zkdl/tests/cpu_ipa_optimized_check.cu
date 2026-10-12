#include "../cpu-backend.cuh"

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace zkdl;

bool require(bool condition, const char* label) {
    if (condition) return true;
    std::cerr << "cpu_ipa_optimized_check status=FAIL case=" << label << '\n';
    return false;
}

CpuG1Point scalar_msm_oracle(const CpuG1Point* points, const FrWire* scalars,
                             std::size_t count) {
    CpuG1Point result = CpuG1Point::zero();
    for (std::size_t index = 0U; index < count; ++index)
        result = cpu_g1_add(result, cpu_g1_mul(points[index], scalars[index]));
    return result;
}

CpuG1Point old_l_fold_oracle(const CpuG1Point* generators, const FrWire* a,
                             const FrWire* b, std::size_t half,
                             const CpuG1Point& u) {
    CpuG1Point result = CpuG1Point::zero();
    for (std::size_t index = 0U; index < half; ++index) {
        result = cpu_g1_add(
            result, cpu_g1_mul(generators[half + index], a[index]));
        result = cpu_g1_add(
            result, cpu_g1_mul(
                u, fr_mul(a[index], b[half + index])));
    }
    return result;
}

CpuG1Point old_r_fold_oracle(const CpuG1Point* generators, const FrWire* a,
                             const FrWire* b, std::size_t half,
                             const CpuG1Point& u) {
    CpuG1Point result = CpuG1Point::zero();
    for (std::size_t index = 0U; index < half; ++index) {
        result = cpu_g1_add(
            result, cpu_g1_mul(generators[index], a[half + index]));
        result = cpu_g1_add(
            result, cpu_g1_mul(
                u, fr_mul(a[half + index], b[index])));
    }
    return result;
}

bool run_check() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    std::cerr << "cpu_ipa_optimized_check status=FAIL reason=BLST backend is required\n";
    return false;
#else
    static_assert(!std::is_assignable<CpuIpaBasisContext&,
                                      std::vector<CpuG1Point>>::value,
                  "an authenticated basis must not accept a raw vector assignment");

    constexpr std::uint32_t kGeneratorCount = 8192U;
    const std::uint32_t padded_size = cpu_ipa_padded_size(kGeneratorCount);
    if (!require(padded_size == 16384U && padded_size > kGeneratorCount + 1U,
                 "hiding-and-zero-padding-shape")) return false;

    const CpuG1Point seed = cpu_g1_generator();
    CpuG1Point h = cpu_g1_mul(seed, fr_from_u32(kGeneratorCount + 17U));
    CpuG1Point u = cpu_g1_mul(seed, fr_from_u32(kGeneratorCount + 19U));
    std::vector<CpuG1Point> decoded_generators;
    decoded_generators.reserve(kGeneratorCount);
    for (std::uint32_t index = 0U; index < kGeneratorCount; ++index)
        decoded_generators.push_back(
            cpu_g1_mul(seed, fr_from_u32(index + 1U)));

    CpuIpaBasisContext prover_basis;
    std::string error;
    if (!require(CpuIpaBasisContext::authenticate(
                     std::move(decoded_generators), h, u, prover_basis, &error),
                 "strict-prover-basis-factory")) {
        std::cerr << " reason=" << error << '\n';
        return false;
    }
    const auto& generators = prover_basis.generators();
    if (!require(prover_basis.owns_exact_generators(generators, h, u) &&
                     cpu_pedersen_basis_valid(prover_basis, h, u,
                                              kGeneratorCount, &error),
                 "immutable-owned-basis")) return false;

    std::vector<CpuG1Point> foreign_generators = generators;
    foreign_generators[0U] = cpu_g1_add(foreign_generators[0U], h);
    if (!require(!prover_basis.owns_exact_generators(foreign_generators, h, u) &&
                     !cpu_pedersen_basis_valid(
                         prover_basis, cpu_g1_add(h, seed), u, kGeneratorCount,
                         &error),
                 "foreign-or-mutated-basis-rejected")) return false;
    std::vector<CpuG1Point> invalid_generators = generators;
    invalid_generators[0U] = h;
    CpuIpaBasisContext invalid_basis;
    if (!require(!CpuIpaBasisContext::authenticate(
                     std::move(invalid_generators), h, u, invalid_basis, &error) &&
                     !invalid_basis.valid(),
                 "strict-factory-rejects-H-reuse")) return false;

    std::vector<FrWire> values(kGeneratorCount);
    for (std::uint32_t index = 0U; index < kGeneratorCount; ++index)
        values[index] = (index % 29U == 0U)
            ? FrWire{} : fr_from_u32((index % 101U) + 1U);
    std::vector<FrWire> point(13U);
    for (std::size_t index = 0U; index < point.size(); ++index)
        point[index] = fr_from_u32(static_cast<std::uint32_t>(index + 3U));
    const std::vector<FrWire> weights = cpu_ipa_mle_weights(point);
    std::vector<FrWire> padded_values(padded_size);
    std::vector<FrWire> padded_weights(padded_size);
    std::copy(values.begin(), values.end(), padded_values.begin());
    std::copy(weights.begin(), weights.end(), padded_weights.begin());

    const FrWire randomness = fr_from_u32(37U);
    const FrWire tau_blinder = fr_from_u32(41U);
    const CpuG1Point expected_xi = cpu_g1_add(
        cpu_g1_sum(generators, values), cpu_g1_mul(h, randomness));
    const CpuG1Point xi_oracle = cpu_g1_add(
        scalar_msm_oracle(generators.data(), values.data(), values.size()),
        cpu_g1_mul(h, randomness));
    const FrWire evaluation = cpu_ipa_inner_product(values, weights);
    const CpuG1Point expected_tau = cpu_g1_add(
        cpu_g1_mul(u, evaluation), cpu_g1_mul(h, tau_blinder));
    if (!require(cpu_g1_equal(expected_xi, xi_oracle) &&
                     !cpu_g1_equal(expected_xi, cpu_g1_sum(generators, values)) &&
                     !cpu_g1_equal(expected_tau, cpu_g1_mul(u, evaluation)),
                 "fixed-hiding-blinders-match-scalar-oracle")) return false;

    // The IPA reserves the next power-of-two domain for generator padding;
    // its hiding scalar stays in the H term of xi/tau rather than becoming a
    // mutable generator-vector alias.
    std::vector<CpuG1Point> folded_generators(padded_size, CpuG1Point::zero());
    std::copy(generators.begin(), generators.end(), folded_generators.begin());
    std::vector<FrWire> challenges;
    std::vector<FrWire> inverses;
    for (std::size_t remaining = padded_size; remaining > 1U; remaining >>= 1U) {
        const FrWire challenge = fr_from_u32(
            static_cast<std::uint32_t>(17U + challenges.size()));
        const FrWire inverse = cpu_fr_inv(challenge);
        challenges.push_back(challenge);
        inverses.push_back(inverse);
        const std::size_t half = folded_generators.size() / 2U;
        std::vector<CpuG1Point> next(half);
        for (std::size_t index = 0U; index < half; ++index)
            next[index] = cpu_g1_add(
                cpu_g1_mul(folded_generators[index], inverse),
                cpu_g1_mul(folded_generators[half + index], challenge));
        folded_generators.swap(next);
    }
    std::vector<FrWire> reverse_weights(1U, fr_from_u32(1U));
    for (std::size_t round = challenges.size(); round-- > 0U;) {
        const std::size_t half = reverse_weights.size();
        std::vector<FrWire> expanded(2U * half);
        for (std::size_t index = 0U; index < half; ++index) {
            expanded[index] = fr_mul(inverses[round], reverse_weights[index]);
            expanded[half + index] = fr_mul(challenges[round], reverse_weights[index]);
        }
        reverse_weights.swap(expanded);
    }
    const CpuG1Point weighted_g_hat = cpu_g1_sum(
        generators.data(), reverse_weights.data(), kGeneratorCount);
    const CpuG1Point scalar_g_hat = scalar_msm_oracle(
        generators.data(), reverse_weights.data(), kGeneratorCount);
    if (!require(reverse_weights.size() == padded_size &&
                     cpu_g1_equal(folded_generators.front(), weighted_g_hat) &&
                     cpu_g1_equal(weighted_g_hat, scalar_g_hat),
                 "reverse-challenge-Gfold-matches-original-fold")) return false;

    // Compare the production L/R MSM + U dot-product equations with the old
    // per-entry group-operation oracle on a reachable, nonzero folded round.
    std::vector<CpuG1Point> round_generators(padded_size, CpuG1Point::zero());
    std::copy(generators.begin(), generators.end(), round_generators.begin());
    std::vector<FrWire> round_a = padded_values;
    std::vector<FrWire> round_b = padded_weights;
    const std::size_t first_half = padded_size / 2U;
    std::vector<CpuG1Point> first_fold_generators(first_half);
    std::vector<FrWire> first_fold_a(first_half), first_fold_b(first_half);
    for (std::size_t index = 0U; index < first_half; ++index) {
        first_fold_generators[index] = cpu_g1_add(
            cpu_g1_mul(round_generators[index], inverses[0U]),
            cpu_g1_mul(round_generators[first_half + index], challenges[0U]));
        first_fold_a[index] = fr_add(
            fr_mul(round_a[index], challenges[0U]),
            fr_mul(round_a[first_half + index], inverses[0U]));
        first_fold_b[index] = fr_add(
            fr_mul(round_b[index], inverses[0U]),
            fr_mul(round_b[first_half + index], challenges[0U]));
    }
    const std::size_t lr_half = first_half / 2U;
    const CpuG1Point l_msm = cpu_g1_add(
        cpu_g1_sum(first_fold_generators.data() + lr_half,
                   first_fold_a.data(), lr_half),
        cpu_g1_mul(u, cpu_ipa_inner_product(
                          first_fold_a.data(), first_fold_b.data() + lr_half,
                          lr_half)));
    const CpuG1Point r_msm = cpu_g1_add(
        cpu_g1_sum(first_fold_generators.data(),
                   first_fold_a.data() + lr_half, lr_half),
        cpu_g1_mul(u, cpu_ipa_inner_product(
                          first_fold_a.data() + lr_half, first_fold_b.data(),
                          lr_half)));
    const CpuG1Point l_old = old_l_fold_oracle(
        first_fold_generators.data(), first_fold_a.data(), first_fold_b.data(),
        lr_half, u);
    const CpuG1Point r_old = old_r_fold_oracle(
        first_fold_generators.data(), first_fold_a.data(), first_fold_b.data(),
        lr_half, u);
    if (!require(lr_half == 4096U && cpu_g1_equal(l_msm, l_old) &&
                     cpu_g1_equal(r_msm, r_old),
                 "large-LR-MSM-matches-per-entry-oracle")) return false;

    CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = 2U;
    metadata.evaluation_commitment_index = 7U;
    metadata.outer_point = {fr_from_u32(23U)};
    metadata.descriptor_digest = sha256(std::vector<zkdl::byte>{'d'});
    metadata.registry_digest = sha256(std::vector<zkdl::byte>{'r'});
    metadata.srs_digest = sha256(std::vector<zkdl::byte>{'s'});
    HyraxOpeningProverResult opened = cpu_open_hyrax_v3_authenticated(
        prover_basis, values, point, randomness, tau_blinder, expected_tau,
        metadata, expected_xi);
    if (!require(cpu_g1_equal(cpu_g1_from_native(opened.proof.xi), expected_xi) &&
                     cpu_g1_equal(cpu_g1_from_native(opened.proof.tau), expected_tau) &&
                     opened.proof.generator_size == kGeneratorCount &&
                     opened.proof.padded_generator_size == padded_size,
                 "production-prover-fixed-blinder-and-padding")) return false;

    std::vector<CpuG1Point> verifier_generators = generators;
    CpuIpaBasisContext verifier_basis;
    if (!require(CpuIpaBasisContext::authenticate(
                     std::move(verifier_generators), h, u, verifier_basis, &error),
                 "independent-verifier-basis-factory")) return false;
    if (!require(cpu_ipa_basis_digest(prover_basis, kGeneratorCount, padded_size) ==
                     cpu_ipa_basis_digest(generators, h, u, padded_size) &&
                     cpu_verify_hyrax_v3_authenticated(
                         opened.proof, expected_xi, expected_tau, verifier_basis,
                         point, metadata, &error),
                 "independent-authenticated-verify")) {
        std::cerr << " reason=" << error << '\n';
        return false;
    }
    if (!require(!cpu_verify_hyrax_v3_authenticated(
                     opened.proof, cpu_g1_add(expected_xi, h), expected_tau,
                     verifier_basis, point, metadata, &error),
                 "independent-verify-rejects-wrong-checked-xi")) return false;
    if (!require(cpu_verify_hyrax_v3(
                     opened.proof, expected_xi, expected_tau, generators, point,
                     h, u, metadata, &error),
                 "raw-full-basis-legacy-verify")) return false;
    std::vector<CpuG1Point> raw_prefix = generators;
    raw_prefix.pop_back();
    if (!require(!cpu_verify_hyrax_v3(
                     opened.proof, expected_xi, expected_tau, raw_prefix, point,
                     h, u, metadata, &error),
                 "raw-prefix-rejected")) return false;
    if (!require(!cpu_verify_hyrax_v3_impl(
                     opened.proof, expected_xi, expected_tau, generators, point,
                     h, u, metadata, nullptr, &error, &verifier_basis,
                     kGeneratorCount),
                 "foreign-context-cannot-authenticate-prover-vector")) return false;

    std::vector<zkdl::byte> encoded;
    serialize_hyrax_opening_v3(opened.proof, encoded);
    HyraxOpeningProofV3 decoded;
    std::size_t offset = 0U;
    if (!require(deserialize_hyrax_opening_v3(encoded, offset, decoded) &&
                     offset == encoded.size(),
                 "opening-deserialize")) return false;
    std::vector<zkdl::byte> reencoded;
    serialize_hyrax_opening_v3(decoded, reencoded);
    if (!require(reencoded == encoded &&
                     cpu_g1_equal(cpu_g1_from_native(decoded.xi), expected_xi) &&
                     cpu_g1_equal(cpu_g1_from_native(decoded.tau), expected_tau) &&
                     cpu_verify_hyrax_v3_authenticated(
                         decoded, expected_xi, expected_tau, verifier_basis,
                         point, metadata, &error),
                 "opening-wire-roundtrip-and-verify")) return false;

    std::vector<zkdl::byte> tampered_wire = encoded;
    const std::size_t z1_offset = tampered_wire.size() - 2U * sizeof(FrWire);
    tampered_wire[z1_offset] ^= 1U;
    HyraxOpeningProofV3 tampered;
    offset = 0U;
    const bool tampered_decoded =
        deserialize_hyrax_opening_v3(tampered_wire, offset, tampered) &&
        offset == tampered_wire.size();
    if (!require(!tampered_decoded ||
                     !cpu_verify_hyrax_v3_authenticated(
                         tampered, expected_xi, expected_tau, verifier_basis,
                         point, metadata, &error),
                 "serialized-response-tamper-rejected")) return false;

    std::cout << "cpu_ipa_optimized_check status=PASS generators="
              << kGeneratorCount << " padded=" << padded_size
              << " rounds=" << opened.proof.rounds.size() / 2U
              << " lr_oracle_terms=" << 2U * lr_half << '\n';
    return true;
#endif
}

}  // namespace

int main() {
    try {
        return run_check() ? 0 : 1;
    } catch (const std::exception& exception) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL exception="
                  << exception.what() << '\n';
        return 1;
    }
}
