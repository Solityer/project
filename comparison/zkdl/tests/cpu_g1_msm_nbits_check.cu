#include "../cpu-backend.cuh"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace zkdl;

bool require(bool condition, const char* name) {
    if (condition) return true;
    std::cerr << "cpu_g1_msm_nbits_check status=FAIL case=" << name << '\n';
    return false;
}

FrWire scalar_u64(std::uint64_t value) {
    FrWire scalar{};
    scalar[0] = static_cast<std::uint32_t>(value);
    scalar[1] = static_cast<std::uint32_t>(value >> 32U);
    return scalar;
}

FrWire scalar_with_nbits(std::size_t nbits) {
    FrWire scalar{};
    if (nbits != 0U) scalar[(nbits - 1U) / 32U] =
        static_cast<std::uint32_t>(1U << ((nbits - 1U) % 32U));
    return scalar;
}

FrWire scalar_or(const std::vector<FrWire>& scalars) {
    FrWire aggregate{};
    for (const FrWire& scalar : scalars)
        for (std::size_t limb = 0U; limb < aggregate.size(); ++limb)
            aggregate[limb] |= scalar[limb];
    return aggregate;
}

CpuG1Point fixed_256_oracle(const std::vector<CpuG1Point>& points,
                            const std::vector<FrWire>& scalars) {
    CpuG1Point result = CpuG1Point::zero();
    for (std::size_t index = 0U; index < scalars.size(); ++index) {
        if (fr_compare(scalars[index], FrWire{}) == 0) continue;
        result = cpu_g1_add(result,
                            cpu_g1_mul_integer_bits(points[index], scalars[index]));
    }
    return result;
}

bool check_msm(const char* name, const std::vector<CpuG1Point>& points,
               const std::vector<FrWire>& scalars, std::size_t expected_nbits) {
    if (!require(points.size() == scalars.size(), "test_shape")) return false;
    if (!require(cpu_g1_msm_or_nbits(scalar_or(scalars)) == expected_nbits,
                 "exact_chunk_bit_length")) return false;
    const CpuG1Point actual = cpu_g1_sum(points, scalars);
    const CpuG1Point expected = fixed_256_oracle(points, scalars);
    if (!require(cpu_g1_equal_portable(actual, expected), name)) return false;
    return true;
}

bool tiny_independent_opening() {
    const CpuG1Point seed = cpu_g1_generator();
    const CpuG1Point h = cpu_g1_mul(seed, scalar_u64(710003U));
    const CpuG1Point u = cpu_g1_mul(seed, scalar_u64(710009U));
    std::vector<CpuG1Point> generators;
    for (std::uint32_t index = 0U; index < 4U; ++index)
        generators.push_back(cpu_g1_mul(seed, scalar_u64(index + 11U)));
    CpuIpaBasisContext prover_basis;
    std::string error;
    if (!require(CpuIpaBasisContext::authenticate(
                     std::move(generators), h, u, prover_basis, &error),
                 "prover_basis_authenticate")) return false;

    const std::vector<FrWire> values{
        scalar_u64(3U), scalar_u64(5U), scalar_u64(7U), scalar_u64(11U)};
    const std::vector<FrWire> point{scalar_u64(13U), scalar_u64(17U)};
    const FrWire randomness = scalar_u64(19U);
    const FrWire tau_blinder = scalar_u64(23U);
    const std::vector<FrWire> mle = cpu_ipa_mle_weights(point);
    const CpuG1Point xi = cpu_g1_add(
        cpu_g1_sum(prover_basis.generators(), values), cpu_g1_mul(h, randomness));
    const CpuG1Point tau = cpu_g1_add(
        cpu_g1_mul(u, cpu_ipa_inner_product(values, mle)),
        cpu_g1_mul(h, tau_blinder));
    CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = 1U;
    metadata.evaluation_commitment_index = 2U;
    metadata.outer_point = point;
    metadata.logical_domain = 4U;
    metadata.actual_length = 4U;
    metadata.padded_length = 4U;
    metadata.descriptor_digest = sha256(std::vector<zkdl::byte>{'d'});
    metadata.registry_digest = sha256(std::vector<zkdl::byte>{'r'});
    metadata.srs_digest = sha256(std::vector<zkdl::byte>{'s'});
    const HyraxOpeningProverResult opened = cpu_open_hyrax_v3_authenticated(
        prover_basis, values, point, randomness, tau_blinder, tau, metadata, xi);

    CpuIpaBasisContext verifier_basis;
    if (!require(CpuIpaBasisContext::authenticate(
                     std::vector<CpuG1Point>(prover_basis.generators()), h, u,
                     verifier_basis, &error),
                 "independent_verifier_basis")) return false;
    if (!require(cpu_verify_hyrax_v3_authenticated(
                     opened.proof, xi, tau, verifier_basis, point, metadata,
                     &error),
                 "independent_opening_verify")) return false;

    std::vector<zkdl::byte> encoded;
    serialize_hyrax_opening_v3(opened.proof, encoded);
    if (!require(encoded.size() >= 2U * sizeof(FrWire), "opening_serialized"))
        return false;
    encoded[encoded.size() - 2U * sizeof(FrWire)] ^= 1U;
    HyraxOpeningProofV3 tampered;
    std::size_t offset = 0U;
    if (!require(deserialize_hyrax_opening_v3(encoded, offset, tampered) &&
                     offset == encoded.size(),
                 "tampered_opening_decode")) return false;
    return require(!cpu_verify_hyrax_v3_authenticated(
                       tampered, xi, tau, verifier_basis, point, metadata,
                       &error),
                   "independent_verify_rejects_tamper");
}

}  // namespace

int main() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    std::cerr << "cpu_g1_msm_nbits_check status=FAIL reason=BLST backend required\n";
    return 2;
#else
    const CpuG1Point generator = cpu_g1_generator();
    const std::vector<CpuG1Point> bit_points{
        generator, cpu_g1_double(generator),
        cpu_g1_mul(generator, scalar_u64(3U)), CpuG1Point::zero()};
    const std::vector<std::size_t> widths{1U, 31U, 32U, 33U, 63U,
                                          64U, 127U, 128U, 254U};
    for (std::size_t width : widths) {
        const std::vector<FrWire> values{
            scalar_with_nbits(width), scalar_u64(1U), FrWire{}, FrWire{}};
        if (!check_msm("bit_boundary_matches_fixed_256_oracle", bit_points,
                       values, width)) return 1;
    }

    const FrWire p_minus_one = fr_sub_raw(kScalarModulus, scalar_u64(1U));
    if (!check_msm("p_minus_one_matches_fixed_256_oracle", bit_points,
                   {p_minus_one, scalar_u64(1U), FrWire{}, FrWire{}}, 255U))
        return 1;
    if (!check_msm("all_zero_matches_fixed_256_oracle", bit_points,
                   {FrWire{}, FrWire{}, FrWire{}, FrWire{}}, 0U)) return 1;

    std::vector<CpuG1Point> small_points{generator};
    if (!check_msm("count_one", small_points, {scalar_u64(1U)}, 1U)) return 1;
    for (std::size_t count : {5U, 128U}) {
        std::vector<CpuG1Point> points;
        std::vector<FrWire> scalars;
        points.reserve(count);
        scalars.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
            points.push_back(cpu_g1_mul(
                generator, scalar_u64(static_cast<std::uint64_t>(index + 2U))));
            scalars.push_back(scalar_u64(
                static_cast<std::uint64_t>((index * 17U + 3U) % 97U)));
        }
        if (!check_msm(count == 5U ? "count_five" : "count_128",
                       points, scalars, 7U)) return 1;
    }
    if (!require(cpu_g1_msm_workspace_bound_bytes(1U) <
                     cpu_g1_msm_workspace_bound_bytes(5U) &&
                     cpu_g1_msm_workspace_bound_bytes(5U) <
                     cpu_g1_msm_workspace_bound_bytes(128U) &&
                     cpu_g1_msm_workspace_bound_bytes(128U) <
                     cpu_g1_msm_workspace_bound_bytes(kCpuG1MsmChunkPoints),
                 "generic_workspace_scales_with_count")) return 1;

    std::vector<CpuG1Point> binary_points(128U, generator);
    std::vector<FrWire> dense_binary(128U);
    for (std::size_t index = 0U; index < dense_binary.size(); index += 2U)
        dense_binary[index] = scalar_u64(1U);
    if (!check_msm("dense_zero_one_scalars", binary_points, dense_binary, 1U))
        return 1;
    std::vector<FrWire> sparse_binary(128U);
    sparse_binary[73U] = scalar_u64(1U);
    if (!check_msm("sparse_zero_one_scalars", binary_points, sparse_binary, 1U))
        return 1;

    std::vector<CpuG1Point> chunk_points(kCpuG1MsmChunkPoints + 1U,
                                         CpuG1Point::zero());
    std::vector<FrWire> chunk_scalars(kCpuG1MsmChunkPoints + 1U);
    chunk_points[0U] = generator;
    chunk_scalars[0U] = scalar_u64(1U);
    chunk_points.back() = generator;
    chunk_scalars.back() = p_minus_one;
    if (!require(cpu_g1_msm_workspace_bound_bytes(kCpuG1MsmChunkPoints) >
                     cpu_g1_msm_workspace_bound_bytes(128U),
                 "full_chunk_workspace_bound")) return 1;
    if (!require(cpu_g1_sum(chunk_points, chunk_scalars).is_zero(),
                 "16384_chunk_and_tail_have_independent_widths")) return 1;

    std::vector<CpuG1Point> basis_points;
    for (std::uint32_t index = 0U; index < 8U; ++index)
        basis_points.push_back(cpu_g1_mul(generator, scalar_u64(index + 2U)));
    const CpuG1Point h = cpu_g1_mul(generator, scalar_u64(101U));
    const CpuG1Point u = cpu_g1_mul(generator, scalar_u64(103U));
    CpuIpaBasisContext basis;
    std::string error;
    if (!require(CpuIpaBasisContext::authenticate(
                     std::move(basis_points), h, u, basis, &error),
                 "session_basis_authenticate")) return 1;
    CpuG1MsmSession session(basis);
    std::vector<FrWire> zero_scalars(basis.size());
    if (!require(session.sum(basis, basis.generators(), h, u,
                             zero_scalars.data(), zero_scalars.size()).is_zero() &&
                     session.workspace_count() == 0U,
                 "session_zero_scalars_validate_then_skip")) return 1;
    std::vector<FrWire> session_scalars{
        scalar_u64(1U), scalar_with_nbits(31U), scalar_with_nbits(64U),
        scalar_with_nbits(127U), scalar_with_nbits(254U), scalar_u64(0U),
        scalar_u64(17U), p_minus_one};
    if (!require(cpu_g1_equal_portable(
                     session.sum(basis, basis.generators(), h, u,
                                 session_scalars.data(), session_scalars.size()),
                     fixed_256_oracle(basis.generators(), session_scalars)),
                 "session_sum_matches_fixed_256_oracle")) return 1;

    const std::vector<std::uint32_t> indices{7U, 1U, 3U, 1U, 0U, 6U, 2U, 5U};
    const std::vector<FrWire> indexed_scalars{
        scalar_with_nbits(1U), scalar_with_nbits(33U), scalar_with_nbits(63U),
        scalar_u64(0U), scalar_with_nbits(128U), scalar_u64(1U),
        scalar_with_nbits(254U), scalar_u64(0U)};
    std::vector<CpuG1Point> indexed_points;
    for (std::uint32_t index : indices)
        indexed_points.push_back(basis.generators()[index]);
    if (!require(cpu_g1_equal_portable(
                     session.sum_indexed(basis, basis.generators(), h, u,
                                         static_cast<std::uint32_t>(basis.size()),
                                         indices.data(), indexed_scalars.data(),
                                         indices.size()),
                     fixed_256_oracle(indexed_points, indexed_scalars)),
                 "indexed_reordering_matches_fixed_256_oracle")) return 1;

    CpuG1MsmSession validation_session(basis);
    const std::uint32_t bad_index = static_cast<std::uint32_t>(basis.size());
    const FrWire zero{};
    bool bad_index_rejected = false;
    try {
        (void)validation_session.sum_indexed(
            basis, basis.generators(), h, u,
            static_cast<std::uint32_t>(basis.size()), &bad_index, &zero, 1U);
    } catch (const std::invalid_argument&) {
        bad_index_rejected = true;
    }
    if (!require(bad_index_rejected && validation_session.workspace_count() == 0U,
                 "zero_scalar_does_not_skip_index_validation")) return 1;

    CpuG1Point invalid_point;
    invalid_point.x = cpu_fp_one();
    invalid_point.y = cpu_fp_one();
    invalid_point.z = cpu_fp_one();
    bool bad_zero_point_rejected = false;
    try {
        (void)cpu_g1_sum(std::vector<CpuG1Point>{invalid_point},
                         std::vector<FrWire>{FrWire{}});
    } catch (const std::invalid_argument&) {
        bad_zero_point_rejected = true;
    }
    if (!require(bad_zero_point_rejected,
                 "zero_scalar_does_not_skip_generic_point_validation")) return 1;

    if (!tiny_independent_opening()) return 1;
    std::cout << "cpu_g1_msm_nbits_check status=PASS widths=9 chunk_points="
              << kCpuG1MsmChunkPoints << " independent_verify=PASS tamper=REJECTED\n";
    return 0;
#endif
}
