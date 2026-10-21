#include "../cpu-backend.cuh"
#include "../gat_adapter/hyrax_gpu_fold.cu"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace zkdl;

bool fixture_fail(const char* case_name, const std::string& detail) {
    std::cerr << "hyrax_gpu_msm_fixture status=FAIL case=" << case_name
              << " detail=" << detail << '\n';
    return false;
}

bool fixture_require(bool condition, const char* case_name,
                     const std::string& detail) {
    return condition || fixture_fail(case_name, detail);
}

FrWire scalar_pattern(std::size_t index, std::size_t variant) {
    const std::size_t selector = (index + 3U * variant) % 17U;
    if (selector == 0U) return FrWire{};
    if (selector == 1U) return fr_from_u32(1U);
    if (selector == 2U)
        return fr_sub_raw(kScalarModulus, fr_from_u32(1U));
    if (selector >= 3U && selector <= 14U) {
        constexpr unsigned int kBoundaryBits[] = {
            3U, 4U, 31U, 32U, 63U, 64U,
            95U, 96U, 127U, 128U, 191U, 254U};
        const unsigned int bit = kBoundaryBits[selector - 3U];
        FrWire value{};
        value[bit / 32U] = 1U << (bit % 32U);
        return value;
    }
    return fr_from_u32(static_cast<std::uint32_t>(index * 29U + variant * 7U + 5U));
}

std::vector<CpuG1Point> make_generators(std::size_t count) {
    std::vector<CpuG1Point> generators;
    generators.reserve(count);
    const CpuG1Point base = cpu_g1_generator();
    for (std::size_t index = 0U; index < count; ++index)
        generators.push_back(cpu_g1_mul(
            base, fr_from_u32(static_cast<std::uint32_t>(index + 2U))));
    if (count > 257U) {
        generators[256U] = cpu_g1_neg(generators[0U]);
        generators[257U] = generators[0U];
    }
    return generators;
}

bool same_canonical_g1(const CpuG1Point& actual, const CpuG1Point& expected) {
    if (!cpu_g1_equal(actual, expected)) return false;
    std::array<zkdl::byte, 48U> actual_bytes{};
    std::array<zkdl::byte, 48U> expected_bytes{};
    return cpu_pedersen_strict_g1(actual, nullptr, &actual_bytes) &&
           cpu_pedersen_strict_g1(expected, nullptr, &expected_bytes) &&
           actual_bytes == expected_bytes;
}

struct PointOperationCounts {
    std::uint64_t add_calls = 0U;
    std::uint64_t double_calls = 0U;
};

void add_counts(PointOperationCounts& target,
                const PointOperationCounts& source) {
    target.add_calls += source.add_calls;
    target.double_calls += source.double_calls;
}

int highest_set_bit(const FrWire& scalar) {
    for (int limb = static_cast<int>(scalar.size()) - 1; limb >= 0; --limb) {
        if (scalar[static_cast<std::size_t>(limb)] != 0U) {
            const uint32_t word = scalar[static_cast<std::size_t>(limb)];
            return limb * 32 + 31 - __builtin_clz(word);
        }
    }
    return -1;
}

std::uint64_t scalar_popcount(const FrWire& scalar) {
    std::uint64_t count = 0U;
    for (const std::uint32_t limb : scalar) count += __builtin_popcount(limb);
    return count;
}

FrWire nibble_at(const FrWire& scalar, std::uint32_t window) {
    return fr_from_u32((scalar[window >> 3U] >> ((window & 7U) * 4U)) & 0x0fU);
}

void count_scalar_mul(PointOperationCounts& count, const FrWire& scalar) {
    const int highest = highest_set_bit(scalar);
    if (highest < 0) return;
    count.add_calls += scalar_popcount(scalar);
    count.double_calls += static_cast<std::uint64_t>(highest + 1);
}

void count_lr_algorithms(const std::vector<FrWire>& scalars, std::size_t half,
                         PointOperationCounts& legacy,
                         PointOperationCounts& optimized) {
    const std::uint64_t legacy_adds_before = legacy.add_calls;
    const std::uint64_t legacy_doubles_before = legacy.double_calls;
    for (std::size_t index = 0U; index < half; ++index) {
        count_scalar_mul(legacy, scalars[index]);
        count_scalar_mul(legacy, scalars[half + index]);
    }
    legacy.add_calls += 2U * (half - 1U);
    if (half < 128U) {
        optimized.add_calls += legacy.add_calls - legacy_adds_before;
        optimized.double_calls += legacy.double_calls - legacy_doubles_before;
        return;
    }

    const std::size_t cta_count = (half + 127U) / 128U;
    const std::size_t block_reduction_adds = cta_count - 1U;
    for (std::size_t side = 0U; side < 2U; ++side) {
        for (std::uint32_t window = 0U; window < 64U; ++window) {
            for (std::size_t block = 0U; block < cta_count; ++block) {
                for (std::size_t warp = 0U; warp < 4U; ++warp) {
                    std::uint32_t bucket_counts[16]{};
                    for (std::size_t lane = 0U; lane < 32U; ++lane) {
                        const std::size_t index = block * 128U + warp * 32U + lane;
                        const std::uint32_t digit = index < half
                            ? nibble_at(scalars[side * half + index], window)[0U]
                            : 0U;
                        ++bucket_counts[digit];
                    }
                    for (std::uint32_t bucket = 0U; bucket < 16U; ++bucket)
                        if (bucket_counts[bucket] > 1U)
                            optimized.add_calls += bucket_counts[bucket] - 1U;
                }
                /* 45 safe adds merge four warp buckets; the new 15-bucket
                 * suffix scan/reduction uses 45 + 15 safe adds. */
                optimized.add_calls += 105U;
            }
        }
        optimized.add_calls += 64U * block_reduction_adds + 64U;
        optimized.double_calls += 4U * 64U;
    }
}

void count_gfold_algorithms(PointOperationCounts& legacy,
                            PointOperationCounts& optimized,
                            const FrWire& challenge, const FrWire& inverse,
                            std::size_t pair_count) {
    const int challenge_high = highest_set_bit(challenge);
    const int inverse_high = highest_set_bit(inverse);
    const std::uint64_t old_doubles =
        static_cast<std::uint64_t>(challenge_high + 1) +
        static_cast<std::uint64_t>(inverse_high + 1);
    const std::uint64_t old_adds =
        scalar_popcount(challenge) + scalar_popcount(inverse) + 1U;
    FrWire bit_union{};
    for (std::size_t limb = 0U; limb < bit_union.size(); ++limb)
        bit_union[limb] = challenge[limb] | inverse[limb];
    const int union_high = highest_set_bit(bit_union);
    const std::uint64_t new_doubles =
        union_high < 0 ? 0U : static_cast<std::uint64_t>(union_high + 1);
    const std::uint64_t new_adds = scalar_popcount(bit_union) + 1U;
    legacy.add_calls += old_adds * pair_count;
    legacy.double_calls += old_doubles * pair_count;
    optimized.add_calls += new_adds * pair_count;
    optimized.double_calls += new_doubles * pair_count;
}

FrWire nonzero_csprng_scalar() {
    FrWire value{};
    do {
        value = cpu_fresh_csprng_scalar();
    } while (fr_compare(value, FrWire{}) == 0);
    return value;
}

FrWire cross_sum(const std::vector<FrWire>& a,
                 const std::vector<FrWire>& b,
                 std::size_t half, bool left) {
    FrWire result{};
    for (std::size_t index = 0U; index < half; ++index) {
        const FrWire term = left
            ? fr_mul(a[index], b[half + index])
            : fr_mul(a[half + index], b[index]);
        result = fr_add(result, term);
    }
    return result;
}

bool check_direct_gpu_session(const CpuIpaBasisContext& basis,
                              CpuHyraxGpuFoldSession& gpu_fold,
                              PointOperationCounts& legacy_ops,
                              PointOperationCounts& optimized_ops) {
    constexpr std::size_t kGeneratorCount = 258U;
    gpu_fold.reset(kGeneratorCount);
    const std::size_t current_size = cpu_ipa_padded_size(
        static_cast<std::uint32_t>(kGeneratorCount));
    std::vector<CpuG1Point> current_g(current_size, CpuG1Point::zero());
    std::copy(basis.generators().begin(),
              basis.generators().begin() + kGeneratorCount, current_g.begin());
    std::vector<FrWire> current_a(current_size), current_b(current_size);
    for (std::size_t index = 0U; index < current_size; ++index) {
        current_a[index] = scalar_pattern(index, 0U);
        current_b[index] = scalar_pattern(index, 1U);
    }
    /* Force an empty four-bit window while retaining canonical field scalars. */
    constexpr std::uint32_t kEmptyWindow = 8U;
    constexpr std::uint32_t kEmptyLimb = kEmptyWindow >> 3U;
    constexpr std::uint32_t kEmptyShift = (kEmptyWindow & 7U) * 4U;
    for (FrWire& scalar : current_a)
        scalar[kEmptyLimb] &= ~(0x0fU << kEmptyShift);

    /* Bucket 1 has an opposite-point cancellation in its first warp. */
    current_a[0U] = fr_from_u32(1U);
    current_a[1U] = fr_from_u32(1U);
    current_a[2U] = fr_from_u32(15U);
    current_a[current_size / 2U + 1U] = fr_from_u32(1U);
    current_a[current_size / 2U + 2U] = fr_from_u32(15U);
    for (const FrWire& scalar : current_a) {
        if (nibble_at(scalar, kEmptyWindow)[0U] != 0U)
            return fixture_fail("empty-digit-window",
                                "test scalar unexpectedly repopulates the cleared window");
    }
    if (!cpu_g1_equal(cpu_g1_add(current_g[current_size / 2U],
                                 current_g[current_size / 2U + 1U]),
                     CpuG1Point::zero()))
        return fixture_fail("opposite-bucket-points",
                            "bucket-1 fixture points do not cancel");
    gpu_fold.initialize_scalars(current_a, current_b);

    std::size_t active = current_size;
    std::size_t rounds = 0U;
    const FrWire challenges[] = {
        fr_from_u32(1U), fr_sub_raw(kScalarModulus, fr_from_u32(1U)),
        fr_from_u32(17U), fr_from_u32(0x10001U),
        fr_from_u32(0x80000001U), fr_from_u32(29U),
        fr_from_u32(43U), fr_from_u32(71U), fr_from_u32(101U)};
    while (active > 1U) {
        const std::size_t half = active / 2U;
        count_lr_algorithms(current_a, half, legacy_ops, optimized_ops);
        const CpuHyraxGpuFoldRound actual = gpu_fold.compute_lr(active);
        const CpuG1Point expected_left = cpu_g1_sum(
            current_g.data() + half, current_a.data(), half);
        const CpuG1Point expected_right = cpu_g1_sum(
            current_g.data(), current_a.data() + half, half);
        if (!same_canonical_g1(actual.left, expected_left) ||
            !same_canonical_g1(actual.right, expected_right))
            return fixture_fail("direct-lr-oracle",
                                "GPU L/R MSM differs from BLST CPU Pippenger");
        if (fr_compare(actual.cross_left,
                       cross_sum(current_a, current_b, half, true)) != 0 ||
            fr_compare(actual.cross_right,
                       cross_sum(current_a, current_b, half, false)) != 0)
            return fixture_fail("direct-cross-oracle",
                                "GPU cross scalar differs from CPU field arithmetic");

        const FrWire challenge = challenges[rounds % 9U];
        const FrWire inverse = cpu_fr_inv(challenge);
        count_gfold_algorithms(legacy_ops, optimized_ops, challenge, inverse, half);
        gpu_fold.fold(challenge, inverse, half);
        std::vector<CpuG1Point> next_g(half);
        std::vector<FrWire> next_a(half), next_b(half);
        for (std::size_t index = 0U; index < half; ++index) {
            next_g[index] = cpu_g1_add(
                cpu_g1_mul(current_g[index], inverse),
                cpu_g1_mul(current_g[half + index], challenge));
            next_a[index] = fr_add(
                fr_mul(current_a[index], challenge),
                fr_mul(current_a[half + index], inverse));
            next_b[index] = fr_add(
                fr_mul(current_b[index], inverse),
                fr_mul(current_b[half + index], challenge));
        }
        current_g.swap(next_g);
        current_a.swap(next_a);
        current_b.swap(next_b);
        active = half;
        ++rounds;
    }

    const CpuHyraxGpuFoldFinalState actual = gpu_fold.final_state();
    if (!same_canonical_g1(actual.generator, current_g[0U]) ||
        fr_compare(actual.a, current_a[0U]) != 0 ||
        fr_compare(actual.b, current_b[0U]) != 0)
        return fixture_fail("direct-final-state",
                            "GPU multiround fold differs from CPU oracle");
    return true;
}

bool check_full_gpu_opening(const CpuIpaBasisContext& prover_basis,
                            const CpuG1Point& h, const CpuG1Point& u,
                            CpuHyraxGpuFoldSession& gpu_fold,
                            std::uint64_t prior_kernel_count) {
    constexpr std::uint32_t kGeneratorCount = 4096U;
    const auto& generators = prover_basis.generators();
    std::vector<FrWire> values(kGeneratorCount);
    for (std::uint32_t index = 0U; index < kGeneratorCount; ++index)
        values[index] = fr_from_u32(index * 13U + 7U);
    std::vector<FrWire> point(12U);
    for (std::uint32_t index = 0U; index < point.size(); ++index)
        point[index] = fr_from_u32(index * 19U + 3U);
    const FrWire randomness = nonzero_csprng_scalar();
    const FrWire tau_blinder = nonzero_csprng_scalar();
    const CpuG1Point xi = cpu_g1_add(
        cpu_g1_sum(generators.data(), values.data(), values.size()),
        cpu_g1_mul(h, randomness));
    const std::vector<FrWire> weights = cpu_ipa_mle_weights(point);
    const FrWire evaluation = cpu_ipa_inner_product(
        values, 0U, weights, 0U, values.size());
    const CpuG1Point tau = cpu_g1_add(
        cpu_g1_mul(u, evaluation), cpu_g1_mul(h, tau_blinder));

    CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = 11U;
    metadata.evaluation_commitment_index = 12U;
    metadata.logical_domain = kGeneratorCount;
    metadata.actual_length = kGeneratorCount;
    metadata.padded_length = kGeneratorCount;
    metadata.descriptor_digest = sha256(
        std::vector<zkdl::byte>{'h', 'y', 'r', 'a', 'x'});
    metadata.registry_digest = sha256(
        std::vector<zkdl::byte>{'g', 'p', 'u', 'm', 's', 'm'});
    metadata.srs_digest = sha256(
        std::vector<zkdl::byte>{'b', 'l', 's', '1', '2', '-', '3', '8', '1'});

    CpuG1MsmSession prover_session(prover_basis);
    prover_session.set_hyrax_gpu_fold_session(&gpu_fold);
    const HyraxOpeningProverResult opened = cpu_open_hyrax_v3_authenticated(
        prover_basis, values, point, randomness, tau_blinder, tau, metadata,
        xi, &prover_session);
    if (!fixture_require(opened.proof.generator_size == kGeneratorCount &&
                             opened.proof.padded_generator_size ==
                                 cpu_ipa_padded_size(kGeneratorCount),
                         "opening-shape", "opening did not use the 4096-point domain"))
        return false;

    const CpuHyraxGpuFoldStats after_open = gpu_fold.stats();
    if (!fixture_require(after_open.kernel_count > prior_kernel_count,
                         "opening-gpu-dispatch",
                         "4096-point opening did not add GPU session kernel launches"))
        return false;

    std::vector<zkdl::byte> encoded;
    serialize_hyrax_opening_v3(opened.proof, encoded);
    HyraxOpeningProofV3 decoded;
    std::size_t offset = 0U;
    if (!fixture_require(deserialize_hyrax_opening_v3(encoded, offset, decoded) &&
                             offset == encoded.size(),
                         "opening-serialize-load", "serialized opening did not load"))
        return false;
    std::vector<zkdl::byte> reencoded;
    serialize_hyrax_opening_v3(decoded, reencoded);
    if (!fixture_require(reencoded == encoded,
                         "opening-canonical-wire",
                         "loaded opening did not retain canonical wire bytes"))
        return false;

    CpuIpaBasisContext verifier_basis;
    std::vector<CpuG1Point> verifier_generators(generators);
    std::string error;
    if (!fixture_require(CpuIpaBasisContext::authenticate(
                             std::move(verifier_generators), h, u,
                             verifier_basis, &error),
                         "independent-basis", error))
        return false;
    CpuG1MsmSession verifier_session(verifier_basis);
    if (!fixture_require(cpu_verify_hyrax_v3_authenticated(
                             decoded, xi, tau, verifier_basis, point, metadata,
                             &error, &verifier_session),
                         "independent-cpu-verify", error))
        return false;

    std::vector<zkdl::byte> tampered_wire = encoded;
    const std::size_t z1_offset = tampered_wire.size() - 2U * sizeof(FrWire);
    tampered_wire[z1_offset] ^= 1U;
    HyraxOpeningProofV3 tampered;
    offset = 0U;
    const bool tampered_loaded =
        deserialize_hyrax_opening_v3(tampered_wire, offset, tampered) &&
        offset == tampered_wire.size();
    if (!fixture_require(!tampered_loaded ||
                             !cpu_verify_hyrax_v3_authenticated(
                                 tampered, xi, tau, verifier_basis, point,
                                 metadata, &error, &verifier_session),
                         "tamper-rejection",
                         "independent CPU verifier accepted changed opening bytes"))
        return false;
    return true;
}

}  // namespace

int main() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    fixture_fail("configuration", "BLST CPU oracle is required");
    return 1;
#else
    try {
        constexpr std::size_t kBasisSize = 4096U;
        std::vector<CpuG1Point> generators = make_generators(kBasisSize);
        const CpuG1Point base = cpu_g1_generator();
        const CpuG1Point h = cpu_g1_mul(base, fr_from_u32(9001U));
        const CpuG1Point u = cpu_g1_mul(base, fr_from_u32(9003U));
        CpuIpaBasisContext prover_basis;
        std::string error;
        if (!fixture_require(CpuIpaBasisContext::authenticate(
                                 std::move(generators), h, u, prover_basis, &error),
                             "basis-authentication", error))
            return 1;
        std::unique_ptr<CpuHyraxGpuFoldSession> gpu_fold =
            make_cpu_hyrax_gpu_fold_session(prover_basis, h, u);
        PointOperationCounts legacy_ops, optimized_ops;
        if (!check_direct_gpu_session(prover_basis, *gpu_fold,
                                      legacy_ops, optimized_ops)) return 1;
        const CpuHyraxGpuFoldStats before_open = gpu_fold->stats();
        if (!check_full_gpu_opening(prover_basis, h, u, *gpu_fold,
                                    before_open.kernel_count)) return 1;
        const CpuHyraxGpuFoldStats stats = gpu_fold->stats();
        if (!fixture_require(stats.kernel_count != 0U && stats.peak_allocated_bytes != 0U,
                             "gpu-session-stats",
                             "GPU session did not report its execution and workspace"))
            return 1;
        const std::uint64_t legacy_total =
            legacy_ops.add_calls + legacy_ops.double_calls;
        const std::uint64_t optimized_total =
            optimized_ops.add_calls + optimized_ops.double_calls;
        std::cout << "hyrax_gpu_msm_fixture status=PASS direct_multiround=PASS"
                  << " opening_generators=" << kBasisSize
                  << " serialize_load=PASS independent_cpu_verify=PASS tamper=REJECTED"
                  << " modeled_point_ops_legacy(add,double,total)="
                  << legacy_ops.add_calls << ',' << legacy_ops.double_calls << ','
                  << legacy_total
                  << " pippenger(add,double,total)=" << optimized_ops.add_calls
                  << ',' << optimized_ops.double_calls << ',' << optimized_total
                  << " weighted_bucket_safeadds_lr_per_cta_window(legacy,new)=60,120"
                  << " weighted_bucket_serial_add_depth_per_side(legacy,new)=30,8"
                  << " edge_cases=bucket1,bucket15,empty_window,opposite_cancel"
                  << " gpu_kernel_count=" << stats.kernel_count
                  << " peak_allocated_bytes=" << stats.peak_allocated_bytes
                  << " device=" << stats.device_name << '\n';
        return 0;
    } catch (const std::exception& exception) {
        fixture_fail("exception", exception.what());
        return 1;
    }
#endif
}
