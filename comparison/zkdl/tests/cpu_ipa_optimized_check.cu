#include "../cpu-backend.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
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

struct GfoldTrace {
    std::vector<std::pair<CpuG1Point, CpuG1Point>> rounds;
    CpuG1Point final_generator = CpuG1Point::zero();
};

struct FullProjectiveFoldTrace {
    std::vector<std::pair<CpuG1Point, CpuG1Point>> rounds;
    std::vector<std::vector<CpuG1Point>> generator_states;
    std::vector<std::vector<FrWire>> a_states;
    std::vector<std::vector<FrWire>> b_states;
    CpuG1Point final_generator = CpuG1Point::zero();
    double lrs_seconds = 0.0;
    double fold_seconds = 0.0;
};

bool cpu_g1_projective_tuple_equal(const CpuG1Point& left,
                                  const CpuG1Point& right) {
    return cpu_fp_compare(left.x, right.x) == 0 &&
           cpu_fp_compare(left.y, right.y) == 0 &&
           cpu_fp_compare(left.z, right.z) == 0;
}

std::vector<FrWire> fixed_fold_challenges(std::uint32_t padded_size) {
    std::vector<FrWire> challenges;
    for (std::uint32_t remaining = padded_size; remaining > 1U; remaining >>= 1U)
        challenges.push_back(fr_from_u32(
            17U + static_cast<std::uint32_t>(challenges.size() * 12U + 1U)));
    return challenges;
}

GfoldTrace legacy_projective_gfold(
    const std::vector<CpuG1Point>& generators, std::uint32_t generator_count,
    const std::vector<FrWire>& values, const std::vector<FrWire>& challenges) {
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_count);
    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    std::copy(generators.begin(), generators.begin() + generator_count,
              current_generators.begin());
    std::vector<FrWire> current_a(padded_size);
    std::copy(values.begin(), values.end(), current_a.begin());
    GfoldTrace trace;
    trace.rounds.reserve(challenges.size());
    std::uint32_t current_size = padded_size;
    for (const FrWire& challenge : challenges) {
        const FrWire inverse = cpu_fr_inv(challenge);
        const std::uint32_t half = current_size / 2U;
        const CpuG1Point left = cpu_g1_sum(
            current_generators.data() + half, current_a.data(), half);
        const CpuG1Point right = cpu_g1_sum(
            current_generators.data(), current_a.data() + half, half);
        trace.rounds.emplace_back(left, right);
        std::vector<CpuG1Point> next_generators(half);
        std::vector<FrWire> next_a(half);
        cpu_parallel_for(half, 128U, [&](std::size_t index) {
            next_generators[index] = cpu_g1_add(
                cpu_g1_mul(current_generators[index], inverse),
                cpu_g1_mul(current_generators[half + index], challenge));
            next_a[index] = fr_add(fr_mul(current_a[index], challenge),
                                   fr_mul(current_a[half + index], inverse));
        });
        current_generators.swap(next_generators);
        current_a.swap(next_a);
        current_size = half;
    }
    trace.final_generator = current_generators.front();
    return trace;
}

FrWire deterministic_full_field_scalar(const std::string& label,
                                       std::uint32_t index) {
    std::vector<zkdl::byte> input{
        'z', 'k', 'd', 'l', '-', 'h', 'y', 'r', 'a', 'x', '-',
        'f', 'o', 'l', 'd', '-', 'f', 'i', 'x', 't', 'u', 'r', 'e', '-', 'v', '1'};
    input.push_back(0U);
    input.insert(input.end(), label.begin(), label.end());
    input.push_back(0U);
    append_u32(input, index);
    const std::array<zkdl::byte, 32U> digest = sha256(input);
    FrWire value{};
    for (std::size_t limb = 0U; limb < value.size(); ++limb)
        value[limb] = static_cast<std::uint32_t>(digest[4U * limb]) |
            (static_cast<std::uint32_t>(digest[4U * limb + 1U]) << 8U) |
            (static_cast<std::uint32_t>(digest[4U * limb + 2U]) << 16U) |
            (static_cast<std::uint32_t>(digest[4U * limb + 3U]) << 24U);
    while (!fr_canonical(value)) value = fr_sub_raw(value, kScalarModulus);
    return value;
}

FullProjectiveFoldTrace legacy_full_projective_fold(
    const std::vector<CpuG1Point>& generators, std::uint32_t generator_count,
    const std::vector<FrWire>& values_a, const std::vector<FrWire>& values_b,
    const std::vector<FrWire>& challenges) {
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_count);
    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    std::copy(generators.begin(), generators.begin() + generator_count,
              current_generators.begin());
    std::vector<FrWire> current_a(padded_size), current_b(padded_size);
    std::copy(values_a.begin(), values_a.end(), current_a.begin());
    std::copy(values_b.begin(), values_b.end(), current_b.begin());
    FullProjectiveFoldTrace trace;
    trace.rounds.reserve(challenges.size());
    trace.generator_states.reserve(challenges.size());
    trace.a_states.reserve(challenges.size());
    trace.b_states.reserve(challenges.size());
    std::uint32_t current_size = padded_size;
    for (const FrWire& challenge : challenges) {
        const FrWire inverse = cpu_fr_inv(challenge);
        const std::uint32_t half = current_size / 2U;
        const auto lrs_start = std::chrono::steady_clock::now();
        const CpuG1Point left = cpu_g1_sum(
            current_generators.data() + half, current_a.data(), half);
        const CpuG1Point right = cpu_g1_sum(
            current_generators.data(), current_a.data() + half, half);
        trace.rounds.emplace_back(left, right);
        trace.lrs_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - lrs_start).count();

        const auto fold_start = std::chrono::steady_clock::now();
        std::vector<CpuG1Point> next_generators(half);
        std::vector<FrWire> next_a(half), next_b(half);
        cpu_parallel_for(half, 128U, [&](std::size_t index) {
            next_generators[index] = cpu_g1_add(
                cpu_g1_mul(current_generators[index], inverse),
                cpu_g1_mul(current_generators[half + index], challenge));
            next_a[index] = fr_add(fr_mul(current_a[index], challenge),
                                   fr_mul(current_a[half + index], inverse));
            next_b[index] = fr_add(fr_mul(current_b[index], inverse),
                                   fr_mul(current_b[half + index], challenge));
        });
        trace.fold_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - fold_start).count();
        trace.generator_states.push_back(next_generators);
        trace.a_states.push_back(next_a);
        trace.b_states.push_back(next_b);
        current_generators.swap(next_generators);
        current_a.swap(next_a);
        current_b.swap(next_b);
        current_size = half;
    }
    trace.final_generator = current_generators.front();
    return trace;
}

FullProjectiveFoldTrace parallel_full_projective_fold(
    const std::vector<CpuG1Point>& generators, std::uint32_t generator_count,
    const std::vector<FrWire>& values_a, const std::vector<FrWire>& values_b,
    const std::vector<FrWire>& challenges) {
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_count);
    std::vector<CpuG1Point> current_generators(padded_size, CpuG1Point::zero());
    std::copy(generators.begin(), generators.begin() + generator_count,
              current_generators.begin());
    std::vector<FrWire> current_a(padded_size), current_b(padded_size);
    std::copy(values_a.begin(), values_a.end(), current_a.begin());
    std::copy(values_b.begin(), values_b.end(), current_b.begin());
    FullProjectiveFoldTrace trace;
    trace.rounds.reserve(challenges.size());
    trace.generator_states.reserve(challenges.size());
    trace.a_states.reserve(challenges.size());
    trace.b_states.reserve(challenges.size());
    std::uint32_t current_size = padded_size;
    for (const FrWire& challenge : challenges) {
        const FrWire inverse = cpu_fr_inv(challenge);
        const std::uint32_t half = current_size / 2U;
        const auto lrs_start = std::chrono::steady_clock::now();
        const CpuG1Point left = cpu_g1_sum(
            current_generators.data() + half, current_a.data(), half);
        const CpuG1Point right = cpu_g1_sum(
            current_generators.data(), current_a.data() + half, half);
        trace.rounds.emplace_back(left, right);
        trace.lrs_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - lrs_start).count();

        const auto fold_start = std::chrono::steady_clock::now();
        std::vector<CpuG1Point> next_generators;
        std::vector<FrWire> next_a, next_b;
        cpu_ipa_fold_projective_vectors_parallel(
            current_generators, current_a, current_b, challenge, inverse,
            next_generators, next_a, next_b);
        trace.fold_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - fold_start).count();
        trace.generator_states.push_back(next_generators);
        trace.a_states.push_back(next_a);
        trace.b_states.push_back(next_b);
        current_generators.swap(next_generators);
        current_a.swap(next_a);
        current_b.swap(next_b);
        current_size = half;
    }
    trace.final_generator = current_generators.front();
    return trace;
}

bool projective_fold_rejects_noncanonical_input() {
    const std::vector<CpuG1Point> generators{
        cpu_g1_generator(), cpu_g1_generator()};
    std::vector<FrWire> a(2U, fr_from_u32(3U));
    std::vector<FrWire> b(2U, fr_from_u32(5U));
    std::vector<CpuG1Point> next_generators;
    std::vector<FrWire> next_a, next_b;
    const FrWire challenge = fr_from_u32(7U);
    const FrWire inverse = cpu_fr_inv(challenge);
    auto rejects = [&](std::vector<FrWire>& target) {
        const FrWire saved = target[1U];
        target[1U] = kScalarModulus;
        bool rejected = false;
        try {
            cpu_ipa_fold_projective_vectors_parallel(
                generators, a, b, challenge, inverse,
                next_generators, next_a, next_b);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        target[1U] = saved;
        return rejected;
    };
    const bool rejects_a = rejects(a);
    const bool rejects_b = rejects(b);
    FrWire invalid_challenge = kScalarModulus;
    bool rejects_challenge = false;
    try {
        cpu_ipa_fold_projective_vectors_parallel(
            generators, a, b, invalid_challenge, inverse,
            next_generators, next_a, next_b);
    } catch (const std::invalid_argument&) {
        rejects_challenge = true;
    }
    return rejects_a && rejects_b && rejects_challenge;
}

FrWire reference_inner_product(const FrWire* left, const FrWire* right,
                               std::size_t count) {
    FrWire result{};
    for (std::size_t index = 0U; index < count; ++index)
        result = fr_add(result, fr_mul(left[index], right[index]));
    return result;
}

bool dot_product_validation_boundaries() {
    const std::vector<FrWire> left{
        fr_from_u32(2U), fr_from_u32(3U), fr_from_u32(4U)};
    const std::vector<FrWire> right{
        fr_from_u32(5U), fr_from_u32(7U), fr_from_u32(11U)};
    const FrWire expected = fr_add(fr_mul(left[1U], right[0U]),
                                   fr_mul(left[2U], right[1U]));
    if (cpu_ipa_inner_product(left, 1U, right, 0U, 2U) != expected ||
        cpu_ipa_inner_product(nullptr, nullptr, 0U) != FrWire{}) return false;
    const std::vector<FrWire> zero_values(5U, FrWire{});
    const std::vector<FrWire> nonzero_values(5U, fr_from_u32(9U));
    if (cpu_ipa_inner_product(zero_values, nonzero_values) != FrWire{}) return false;
    bool range_rejected = false;
    try {
        (void)cpu_ipa_inner_product(left, 2U, right, 0U, 2U);
    } catch (const std::invalid_argument&) {
        range_rejected = true;
    }
    bool pointer_range_rejected = false;
    const std::uintptr_t near_address_end =
        std::numeric_limits<std::uintptr_t>::max() - sizeof(FrWire) / 2U;
    const FrWire* near_address_end_pointer =
        reinterpret_cast<const FrWire*>(near_address_end);
    try {
        (void)cpu_ipa_inner_product(near_address_end_pointer,
                                    left.data(), 1U);
    } catch (const std::length_error&) {
        pointer_range_rejected = true;
    }
    const std::vector<FrWire> noncanonical{kScalarModulus};
    const std::vector<FrWire> zero{FrWire{}};
    bool noncanonical_rejected = false;
    try {
        (void)cpu_ipa_inner_product(noncanonical, zero);
    } catch (const std::invalid_argument&) {
        noncanonical_rejected = true;
    }
    bool shape_rejected = false;
    try {
        (void)cpu_ipa_inner_product(left, std::vector<FrWire>(2U));
    } catch (const std::invalid_argument&) {
        shape_rejected = true;
    }
    return range_rejected && pointer_range_rejected &&
           noncanonical_rejected && shape_rejected;
}

GfoldTrace indexed_gfold(
    const CpuIpaBasisContext& basis, CpuG1MsmSession& session,
    const CpuG1Point& h, const CpuG1Point& u,
    std::uint32_t generator_count, const std::vector<FrWire>& values,
    const std::vector<FrWire>& challenges) {
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_count);
    std::vector<FrWire> current_a(padded_size);
    std::copy(values.begin(), values.end(), current_a.begin());
    std::vector<FrWire> current_weights(1U, fr_from_u32(1U));
    std::vector<std::uint32_t> indices;
    std::vector<FrWire> scalars;
    indices.reserve(generator_count);
    scalars.reserve(generator_count);
    GfoldTrace trace;
    trace.rounds.reserve(challenges.size());
    std::uint32_t current_size = padded_size;
    for (const FrWire& challenge : challenges) {
        const FrWire inverse = cpu_fr_inv(challenge);
        const std::uint32_t half = current_size / 2U;
        CpuG1Point left = CpuG1Point::zero();
        CpuG1Point right = CpuG1Point::zero();
        cpu_ipa_fold_generator_lrs_indexed(
            session, basis, basis.generators(), generator_count, h, u,
            current_weights, current_a, indices, scalars, left, right);
        trace.rounds.emplace_back(left, right);
        std::vector<FrWire> next_a(half);
        for (std::size_t index = 0U; index < half; ++index)
            next_a[index] = fr_add(fr_mul(current_a[index], challenge),
                                   fr_mul(current_a[half + index], inverse));
        current_a.swap(next_a);
        current_weights = cpu_ipa_next_generator_weights_fast(
            current_weights, challenge, inverse);
        current_size = half;
    }
    trace.final_generator = session.sum(
        basis, basis.generators(), h, u, current_weights.data(),
        generator_count);
    return trace;
}

bool prefix_digest_matches_legacy(const CpuIpaBasisContext& basis,
                                  std::uint32_t generator_count,
                                  const CpuG1Point& h, const CpuG1Point& u) {
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_count);
    std::vector<CpuG1Point> prefix(
        basis.generators().begin(),
        basis.generators().begin() + generator_count);
    return cpu_ipa_basis_digest(basis, generator_count, padded_size) ==
           cpu_ipa_basis_digest(prefix, h, u, padded_size);
}

bool small_fold_case(std::uint32_t generator_count) {
    const CpuG1Point seed = cpu_g1_generator();
    const CpuG1Point h = cpu_g1_mul(seed, fr_from_u32(700001U));
    const CpuG1Point u = cpu_g1_mul(seed, fr_from_u32(700003U));
    std::vector<CpuG1Point> points;
    points.reserve(generator_count);
    for (std::uint32_t index = 0U; index < generator_count; ++index)
        points.push_back(cpu_g1_mul(seed, fr_from_u32(index + 5U)));
    CpuIpaBasisContext basis;
    std::string error;
    if (!CpuIpaBasisContext::authenticate(
            std::move(points), h, u, basis, &error)) {
        std::cerr << " reason=" << error << '\n';
        return false;
    }
    if (!prefix_digest_matches_legacy(basis, generator_count, h, u)) return false;
    CpuG1MsmSession session(basis);
    std::vector<FrWire> values(generator_count);
    for (std::uint32_t index = 0U; index < generator_count; ++index)
        values[index] = index % 3U == 1U
            ? FrWire{} : fr_from_u32(index + 2U);
    const auto challenges = fixed_fold_challenges(
        cpu_ipa_padded_size(generator_count));
    const GfoldTrace legacy = legacy_projective_gfold(
        basis.generators(), generator_count, values, challenges);
    const GfoldTrace indexed = indexed_gfold(
        basis, session, h, u, generator_count, values, challenges);
    if (legacy.rounds.size() != indexed.rounds.size()) return false;
    for (std::size_t round = 0U; round < legacy.rounds.size(); ++round)
        if (!cpu_g1_equal(legacy.rounds[round].first,
                          indexed.rounds[round].first) ||
            !cpu_g1_equal(legacy.rounds[round].second,
                          indexed.rounds[round].second))
            return false;
    return cpu_g1_equal(legacy.final_generator, indexed.final_generator);
}

bool repeated_basis_large_fold_case(
    const CpuIpaBasisContext& basis, CpuG1MsmSession& session,
    const CpuG1Point& h, const CpuG1Point& u,
    std::uint32_t generator_count) {
    const std::uint32_t padded_size = cpu_ipa_padded_size(generator_count);
    const CpuG1Point repeated_point = basis.generators().front();
    if (!prefix_digest_matches_legacy(basis, generator_count, h, u)) return false;
    std::vector<FrWire> current_a(padded_size);
    for (std::uint32_t index = 0U; index < generator_count; ++index)
        current_a[index] = index % 23U == 0U
            ? FrWire{} : fr_from_u32(index % 127U + 1U);
    std::vector<FrWire> legacy_coefficients(padded_size);
    std::fill(legacy_coefficients.begin(),
              legacy_coefficients.begin() + generator_count,
              fr_from_u32(1U));
    std::vector<FrWire> current_weights(1U, fr_from_u32(1U));
    std::vector<std::uint32_t> indices;
    std::vector<FrWire> scalars;
    indices.reserve(generator_count);
    scalars.reserve(generator_count);
    const auto challenges = fixed_fold_challenges(padded_size);
    std::uint32_t current_size = padded_size;
    for (const FrWire& challenge : challenges) {
        const FrWire inverse = cpu_fr_inv(challenge);
        const std::uint32_t half = current_size / 2U;
        CpuG1Point indexed_left = CpuG1Point::zero();
        CpuG1Point indexed_right = CpuG1Point::zero();
        cpu_ipa_fold_generator_lrs_indexed(
            session, basis, basis.generators(), generator_count, h, u,
            current_weights, current_a, indices, scalars,
            indexed_left, indexed_right);
        const FrWire old_left_scalar = cpu_ipa_inner_product(
            legacy_coefficients.data() + half, current_a.data(), half);
        const FrWire old_right_scalar = cpu_ipa_inner_product(
            legacy_coefficients.data(), current_a.data() + half, half);
        const CpuG1Point old_left = cpu_g1_mul(repeated_point, old_left_scalar);
        const CpuG1Point old_right = cpu_g1_mul(repeated_point, old_right_scalar);
        if (!cpu_g1_equal(old_left, indexed_left) ||
            !cpu_g1_equal(old_right, indexed_right)) return false;

        std::vector<FrWire> next_legacy_coefficients(half);
        std::vector<FrWire> next_a(half);
        for (std::size_t index = 0U; index < half; ++index) {
            next_legacy_coefficients[index] = fr_add(
                fr_mul(legacy_coefficients[index], inverse),
                fr_mul(legacy_coefficients[half + index], challenge));
            next_a[index] = fr_add(
                fr_mul(current_a[index], challenge),
                fr_mul(current_a[half + index], inverse));
        }
        legacy_coefficients.swap(next_legacy_coefficients);
        current_a.swap(next_a);
        current_weights = cpu_ipa_next_generator_weights_fast(
            current_weights, challenge, inverse);
        current_size = half;

        // Every original G point is the same valid point in this large
        // bounded fixture.  Equality of these scalar coefficients therefore
        // establishes every old/new folded point, including zero-padded tails.
        const std::size_t blocks = current_weights.size();
        for (std::size_t index = 0U; index < current_size; ++index) {
            FrWire expanded_coefficient{};
            for (std::size_t block = 0U; block < blocks; ++block) {
                const std::size_t original_index = block * current_size + index;
                if (original_index < generator_count)
                    expanded_coefficient = fr_add(
                        expanded_coefficient, current_weights[block]);
            }
            if (fr_compare(expanded_coefficient,
                           legacy_coefficients[index]) != 0) return false;
        }
        const std::array<std::size_t, 3U> samples = {
            0U, current_size / 2U, current_size - 1U};
        for (const std::size_t sample : samples) {
            FrWire expanded_coefficient{};
            for (std::size_t block = 0U; block < blocks; ++block) {
                const std::size_t original_index = block * current_size + sample;
                if (original_index < generator_count)
                    expanded_coefficient = fr_add(
                        expanded_coefficient, current_weights[block]);
            }
            if (!cpu_g1_equal(
                    cpu_g1_mul(repeated_point, legacy_coefficients[sample]),
                    cpu_g1_mul(repeated_point, expanded_coefficient)))
                return false;
        }
    }
    const CpuG1Point legacy_final =
        cpu_g1_mul(repeated_point, legacy_coefficients.front());
    const CpuG1Point indexed_final = session.sum(
        basis, basis.generators(), h, u, current_weights.data(),
        generator_count);
    return cpu_g1_equal(legacy_final, indexed_final);
}

bool tiny_independent_opening_case() {
    const CpuG1Point seed = cpu_g1_generator();
    const CpuG1Point h = cpu_g1_mul(seed, fr_from_u32(710003U));
    const CpuG1Point u = cpu_g1_mul(seed, fr_from_u32(710009U));
    std::vector<CpuG1Point> prover_points;
    prover_points.reserve(8U);
    for (std::uint32_t index = 0U; index < 8U; ++index)
        prover_points.push_back(cpu_g1_mul(seed, fr_from_u32(index + 11U)));
    CpuIpaBasisContext prover_basis;
    std::string error;
    if (!CpuIpaBasisContext::authenticate(
            std::move(prover_points), h, u, prover_basis, &error)) return false;
    std::vector<FrWire> values(8U);
    for (std::uint32_t index = 0U; index < 8U; ++index)
        values[index] = fr_from_u32(index * 3U + 1U);
    const std::vector<FrWire> point{
        fr_from_u32(2U), fr_from_u32(5U), fr_from_u32(7U)};
    const FrWire randomness = fr_from_u32(29U);
    const FrWire tau_blinder = fr_from_u32(31U);
    const auto mle = cpu_ipa_mle_weights(point);
    const CpuG1Point xi = cpu_g1_add(
        cpu_g1_sum(prover_basis.generators(), values),
        cpu_g1_mul(h, randomness));
    const CpuG1Point tau = cpu_g1_add(
        cpu_g1_mul(u, cpu_ipa_inner_product(values, mle)),
        cpu_g1_mul(h, tau_blinder));
    CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = 1U;
    metadata.evaluation_commitment_index = 2U;
    metadata.logical_domain = 8U;
    metadata.actual_length = 8U;
    metadata.padded_length = 8U;
    metadata.descriptor_digest = sha256(std::vector<zkdl::byte>{'d'});
    metadata.registry_digest = sha256(std::vector<zkdl::byte>{'r'});
    metadata.srs_digest = sha256(std::vector<zkdl::byte>{'s'});
    CpuG1MsmSession prover_session(prover_basis);
    const HyraxOpeningProverResult opened = cpu_open_hyrax_v3_authenticated(
        prover_basis, values, point, randomness, tau_blinder, tau,
        metadata, xi, &prover_session);
    if (opened.proof.rounds.capacity() != opened.proof.rounds.size() ||
        opened.proof.inner_point.capacity() != opened.proof.inner_point.size() ||
        opened.proof.outer_point.capacity() != opened.proof.outer_point.size())
        return false;
    bool wrong_tau_rejected = false;
    try {
        const CpuG1Point wrong_tau = cpu_g1_add(tau, seed);
        (void)cpu_open_hyrax_v3_authenticated(
            prover_basis, values, point, randomness, tau_blinder, wrong_tau,
            metadata, xi, &prover_session);
    } catch (const std::invalid_argument&) {
        wrong_tau_rejected = true;
    }
    if (!wrong_tau_rejected) return false;
    bool value_shape_rejected = false;
    try {
        std::vector<FrWire> short_values(values.begin(), values.end() - 1);
        (void)cpu_open_hyrax_v3_authenticated(
            prover_basis, short_values, point, randomness, tau_blinder, tau,
            metadata, xi, &prover_session);
    } catch (const std::invalid_argument&) {
        value_shape_rejected = true;
    }
    if (!value_shape_rejected) return false;
    bool point_shape_rejected = false;
    try {
        const std::vector<FrWire> short_point{fr_from_u32(2U), fr_from_u32(5U)};
        (void)cpu_open_hyrax_v3_authenticated(
            prover_basis, values, short_point, randomness, tau_blinder, tau,
            metadata, xi, &prover_session);
    } catch (const std::invalid_argument&) {
        point_shape_rejected = true;
    }
    if (!point_shape_rejected) return false;
    CpuIpaBasisContext verifier_basis;
    if (!CpuIpaBasisContext::authenticate(
            std::vector<CpuG1Point>(prover_basis.generators()), h, u,
            verifier_basis, &error)) return false;
    CpuG1MsmSession verifier_session(verifier_basis);
    if (!cpu_verify_hyrax_v3_authenticated(
            opened.proof, xi, tau, verifier_basis, point, metadata, &error,
            &verifier_session)) return false;
    std::vector<zkdl::byte> encoded;
    serialize_hyrax_opening_v3(opened.proof, encoded);
    const std::size_t z1_offset = encoded.size() - 2U * sizeof(FrWire);
    encoded[z1_offset] ^= 1U;
    HyraxOpeningProofV3 tampered;
    std::size_t offset = 0U;
    if (!deserialize_hyrax_opening_v3(encoded, offset, tampered) ||
        offset != encoded.size() ||
        cpu_verify_hyrax_v3_authenticated(
            tampered, xi, tau, verifier_basis, point, metadata, &error,
            &verifier_session)) return false;
    std::cout << "tiny_opening=PASS independent_verify=PASS tamper=REJECTED"
              << " wrong_tau=REJECTED value_shape=REJECTED point_shape=REJECTED"
              << " rounds_size=" << opened.proof.rounds.size()
              << " rounds_capacity=" << opened.proof.rounds.capacity()
              << " inner_size=" << opened.proof.inner_point.size()
              << " inner_capacity=" << opened.proof.inner_point.capacity()
              << " outer_size=" << opened.proof.outer_point.size()
              << " outer_capacity=" << opened.proof.outer_point.capacity()
              << " opening_sizeof=" << sizeof(HyraxOpeningProofV3)
              << " point_sizeof=" << sizeof(G1Jacobian_t) << '\n';
    return true;
}

bool run_fold_only_check() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    std::cerr << "cpu_ipa_optimized_check status=FAIL reason=BLST backend is required\n";
    return false;
#else
    for (const std::uint32_t count : {1U, 3U, 6U}) {
        if (!small_fold_case(count)) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL fold_count="
                      << count << '\n';
            return false;
        }
        std::cout << "gfold_small=PASS generators=" << count
                  << " padded=" << cpu_ipa_padded_size(count)
                  << " prefix_digest=LEGACY_EQUAL rounds="
                  << fixed_fold_challenges(cpu_ipa_padded_size(count)).size()
                  << '\n';
    }

    constexpr std::uint32_t kLargeCount = 65553U;
    const CpuG1Point seed = cpu_g1_generator();
    const CpuG1Point h = cpu_g1_mul(seed, fr_from_u32(720007U));
    const CpuG1Point u = cpu_g1_mul(seed, fr_from_u32(720011U));
    std::vector<CpuG1Point> large_points(kLargeCount, cpu_g1_mul(
        seed, fr_from_u32(720017U)));
    CpuIpaBasisContext large_basis;
    std::string error;
    const auto auth_start = std::chrono::steady_clock::now();
    if (!CpuIpaBasisContext::authenticate(
            std::move(large_points), h, u, large_basis, &error)) {
        std::cerr << " reason=" << error << '\n';
        return false;
    }
    const double auth_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - auth_start).count();
    const auto session_start = std::chrono::steady_clock::now();
    CpuG1MsmSession large_session(large_basis);
    const double preparation_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - session_start).count();

    std::vector<FrWire> large_scalars(kLargeCount);
    std::vector<std::uint32_t> large_indices(kLargeCount);
    for (std::uint32_t index = 0U; index < kLargeCount; ++index) {
        large_indices[index] = index;
        large_scalars[index] = index % 31U == 0U
            ? FrWire{} : fr_from_u32(index % 113U + 1U);
    }
    const auto raw_start = std::chrono::steady_clock::now();
    const CpuG1Point raw_msm = cpu_g1_sum(
        large_basis.generators().data(), large_scalars.data(), kLargeCount);
    const double raw_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - raw_start).count();
    const auto indexed_start = std::chrono::steady_clock::now();
    const CpuG1Point indexed_msm = large_session.sum_indexed(
        large_basis, large_basis.generators(), h, u, kLargeCount,
        large_indices.data(), large_scalars.data(), kLargeCount);
    const double indexed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - indexed_start).count();
    if (!cpu_g1_equal(raw_msm, indexed_msm) ||
        large_session.workspace_count() < 5U ||
        !prefix_digest_matches_legacy(large_basis, kLargeCount, h, u))
        return false;

    bool bad_index_rejected = false;
    const std::uint32_t bad_index = kLargeCount;
    const FrWire valid_scalar = fr_from_u32(1U);
    try {
        (void)large_session.sum_indexed(
            large_basis, large_basis.generators(), h, u, kLargeCount,
            &bad_index, &valid_scalar, 1U);
    } catch (const std::invalid_argument&) {
        bad_index_rejected = true;
    }
    FrWire noncanonical{};
    std::fill(reinterpret_cast<unsigned char*>(&noncanonical),
              reinterpret_cast<unsigned char*>(&noncanonical) + sizeof(noncanonical),
              0xffU);
    const std::uint32_t valid_index = 0U;
    bool bad_scalar_rejected = false;
    try {
        (void)large_session.sum_indexed(
            large_basis, large_basis.generators(), h, u, kLargeCount,
            &valid_index, &noncanonical, 1U);
    } catch (const std::invalid_argument&) {
        bad_scalar_rejected = true;
    }
    if (!bad_index_rejected || !bad_scalar_rejected) return false;

    const auto fold_start = std::chrono::steady_clock::now();
    const bool large_fold_ok = repeated_basis_large_fold_case(
        large_basis, large_session, h, u, kLargeCount);
    const double large_fold_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fold_start).count();
    if (!large_fold_ok) return false;
    std::cout << "gfold_large=PASS generators=" << kLargeCount
              << " padded=" << cpu_ipa_padded_size(kLargeCount)
              << " prefix_digest=LEGACY_EQUAL rounds="
              << fixed_fold_challenges(cpu_ipa_padded_size(kLargeCount)).size()
              << " all_folded_coefficients=EQUAL sampled_points=EQUAL"
              << " indexed_workspace_lanes=" << large_session.workspace_count()
              << " affine_basis_bytes=" << large_session.prepared_basis_bytes()
              << " workspace_bytes=" << large_session.workspace_bytes()
              << " authenticate_seconds=" << auth_seconds
              << " prepared_basis_seconds=" << preparation_seconds
              << " raw_msm_seconds=" << raw_seconds
              << " indexed_msm_seconds=" << indexed_seconds
              << " large_fold_seconds=" << large_fold_seconds
              << " invalid_index=REJECTED noncanonical_scalar=REJECTED\n";

    constexpr std::uint32_t kTimingCount = 8192U;
    std::vector<CpuG1Point> timing_points(kTimingCount, seed);
    CpuIpaBasisContext timing_basis;
    if (!CpuIpaBasisContext::authenticate(
            std::move(timing_points), h, u, timing_basis, &error)) return false;
    CpuG1MsmSession timing_session(timing_basis);
    std::vector<FrWire> timing_values(kTimingCount);
    for (std::uint32_t index = 0U; index < kTimingCount; ++index)
        timing_values[index] = fr_from_u32(index % 97U + 1U);
    const auto timing_challenges = fixed_fold_challenges(
        cpu_ipa_padded_size(kTimingCount));
    const auto legacy_start = std::chrono::steady_clock::now();
    const GfoldTrace timing_legacy = legacy_projective_gfold(
        timing_basis.generators(), kTimingCount, timing_values,
        timing_challenges);
    const double legacy_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - legacy_start).count();
    const auto optimized_start = std::chrono::steady_clock::now();
    const GfoldTrace timing_indexed = indexed_gfold(
        timing_basis, timing_session, h, u, kTimingCount, timing_values,
        timing_challenges);
    const double optimized_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - optimized_start).count();
    if (timing_legacy.rounds.size() != timing_indexed.rounds.size()) return false;
    for (std::size_t round = 0U; round < timing_legacy.rounds.size(); ++round)
        if (!cpu_g1_equal(timing_legacy.rounds[round].first,
                          timing_indexed.rounds[round].first) ||
            !cpu_g1_equal(timing_legacy.rounds[round].second,
                          timing_indexed.rounds[round].second)) return false;
    if (!cpu_g1_equal(timing_legacy.final_generator,
                      timing_indexed.final_generator)) return false;
    std::cout << "gfold_timing=PASS generators=" << kTimingCount
              << " padded=" << cpu_ipa_padded_size(kTimingCount)
              << " old_projective_seconds=" << legacy_seconds
              << " indexed_prepared_seconds=" << optimized_seconds
              << " speedup=" << legacy_seconds / optimized_seconds
              << " semantics=G_PARTS_ONLY\n";

    if (!tiny_independent_opening_case()) return false;
    std::cout << "cpu_ipa_optimized_check status=PASS mode=fold-only"
                 " full_datasets=0\n";
    return true;
#endif
}

bool run_fast_fold_only_check() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    std::cerr << "cpu_ipa_optimized_check status=FAIL reason=BLST backend is required\n";
    return false;
#else
    for (const std::uint32_t count : {1U, 3U, 6U}) {
        if (!small_fold_case(count)) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL fold_count="
                      << count << '\n';
            return false;
        }
    }

    constexpr std::uint32_t kLargeCount = 65553U;
    const std::uint32_t padded_size = cpu_ipa_padded_size(kLargeCount);
    const CpuG1Point seed = cpu_g1_generator();
    const CpuG1Point h = cpu_g1_mul(seed, fr_from_u32(720007U));
    const CpuG1Point u = cpu_g1_mul(seed, fr_from_u32(720011U));
    const CpuG1Point repeated_point = cpu_g1_mul(seed, fr_from_u32(720017U));
    std::vector<CpuG1Point> points(kLargeCount, repeated_point);
    CpuIpaBasisContext basis;
    std::string error;
    if (!CpuIpaBasisContext::authenticate(
            std::move(points), h, u, basis, &error)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL basis=" << error << '\n';
        return false;
    }
    if (!prefix_digest_matches_legacy(basis, kLargeCount, h, u)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL prefix-digest\n";
        return false;
    }
    CpuG1MsmSession session(basis);
    std::vector<FrWire> values(kLargeCount);
    for (std::uint32_t index = 0U; index < kLargeCount; ++index)
        values[index] = index % 23U == 0U
            ? FrWire{} : fr_from_u32(index % 127U + 1U);
    values.back() = fr_sub(kScalarModulus, fr_from_u32(1U));
    const auto challenges = fixed_fold_challenges(padded_size);

    const auto old_start = std::chrono::steady_clock::now();
    const GfoldTrace legacy = legacy_projective_gfold(
        basis.generators(), kLargeCount, values, challenges);
    const double legacy_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - old_start).count();
    const auto optimized_start = std::chrono::steady_clock::now();
    const GfoldTrace optimized = indexed_gfold(
        basis, session, h, u, kLargeCount, values, challenges);
    const double optimized_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - optimized_start).count();
    if (legacy.rounds.size() != challenges.size() ||
        optimized.rounds.size() != legacy.rounds.size()) return false;
    for (std::size_t round = 0U; round < legacy.rounds.size(); ++round)
        if (!cpu_g1_equal(legacy.rounds[round].first,
                          optimized.rounds[round].first) ||
            !cpu_g1_equal(legacy.rounds[round].second,
                          optimized.rounds[round].second)) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL large-LR round="
                      << round << '\n';
            return false;
        }
    if (!cpu_g1_equal(legacy.final_generator, optimized.final_generator)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL large-final-ghat\n";
        return false;
    }

    // Independently compare every expanded original-basis coefficient with
    // the scalar recurrence, including the non-power-of-two padded tail.
    std::vector<FrWire> legacy_coefficients(padded_size);
    std::fill(legacy_coefficients.begin(),
              legacy_coefficients.begin() + kLargeCount,
              fr_from_u32(1U));
    std::vector<FrWire> weights(1U, fr_from_u32(1U));
    std::uint32_t current_size = padded_size;
    for (const FrWire& challenge : challenges) {
        const FrWire inverse = cpu_fr_inv(challenge);
        const std::uint32_t half = current_size / 2U;
        std::vector<FrWire> next_legacy_coefficients(half);
        for (std::size_t index = 0U; index < half; ++index)
            next_legacy_coefficients[index] = fr_add(
                fr_mul(legacy_coefficients[index], inverse),
                fr_mul(legacy_coefficients[half + index], challenge));
        const std::vector<FrWire> generic_next =
            cpu_ipa_next_generator_weights(weights, challenge, inverse);
        const std::vector<FrWire> fast_next =
            cpu_ipa_next_generator_weights_fast(weights, challenge, inverse);
        if (generic_next != fast_next) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL fast-next-W\n";
            return false;
        }
        for (std::size_t index = 0U; index < half; ++index) {
            FrWire expanded{};
            for (std::size_t block = 0U; block < fast_next.size(); ++block) {
                const std::size_t original_index = block * half + index;
                if (original_index < kLargeCount)
                    expanded = fr_add(expanded, fast_next[block]);
            }
            if (fr_compare(expanded, next_legacy_coefficients[index]) != 0) {
                std::cerr << "cpu_ipa_optimized_check status=FAIL large-coefficient index="
                          << index << '\n';
                return false;
            }
        }
        legacy_coefficients.swap(next_legacy_coefficients);
        weights = fast_next;
        current_size = half;
    }

    FrWire noncanonical{};
    std::fill(reinterpret_cast<unsigned char*>(&noncanonical),
              reinterpret_cast<unsigned char*>(&noncanonical) + sizeof(noncanonical),
              0xffU);
    std::vector<FrWire> padded_scalars(padded_size);
    padded_scalars[kLargeCount] = noncanonical;
    std::vector<FrWire> initial_weight(1U, fr_from_u32(1U));
    std::vector<std::uint32_t> indices;
    std::vector<FrWire> scalars;
    CpuG1Point left, right;
    bool padded_scalar_rejected = false;
    try {
        cpu_ipa_fold_generator_lrs_indexed(
            session, basis, basis.generators(), kLargeCount, h, u,
            initial_weight, padded_scalars, indices, scalars, left, right);
    } catch (const std::invalid_argument&) {
        padded_scalar_rejected = true;
    }
    std::vector<FrWire> small_a(2U);
    std::vector<FrWire> weights_with_padded_block(4U, fr_from_u32(1U));
    weights_with_padded_block.back() = noncanonical;
    bool padded_weight_rejected = false;
    try {
        cpu_ipa_fold_generator_lrs_indexed(
            session, basis, basis.generators(), 5U, h, u,
            weights_with_padded_block, small_a, indices, scalars,
            left, right);
    } catch (const std::invalid_argument&) {
        padded_weight_rejected = true;
    }
    if (!padded_scalar_rejected || !padded_weight_rejected) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL padded-canonical-validation\n";
        return false;
    }

    std::cout << "gfold_fast_large=PASS generators=" << kLargeCount
              << " padded=" << padded_size << " rounds=" << challenges.size()
              << " prefix_digest=LEGACY_EQUAL every_LR=EQUAL final_ghat=EQUAL"
              << " every_folded_coefficient=EQUAL next_W=GENERIC_EQUAL"
              << " padded_scalar=REJECTED padded_weight=REJECTED"
              << " old_projective_seconds=" << legacy_seconds
              << " fast_indexed_seconds=" << optimized_seconds
              << " speedup=" << legacy_seconds / optimized_seconds
              << " fold_montgomery_scratch_bytes="
              << sizeof(zkdl_blst::blst_fr) * (padded_size + 1U)
              << " production_max_scratch_bytes="
              << sizeof(zkdl_blst::blst_fr) * ((1U << 19U) + 1U)
              << '\n';

    if (!tiny_independent_opening_case()) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL tiny-opening\n";
        return false;
    }
    std::cout << "cpu_ipa_optimized_check status=PASS mode=fast-fold-only"
                 " full_datasets=0\n";
    return true;
#endif
}

bool run_representative_fold_only_check() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    std::cerr << "cpu_ipa_optimized_check status=FAIL reason=BLST backend is required\n";
    return false;
#else
    constexpr std::uint32_t kGeneratorCount = 65553U;
    const std::uint32_t padded_size = cpu_ipa_padded_size(kGeneratorCount);
    if (padded_size != (1U << 17U)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative padded-size\n";
        return false;
    }

    std::cerr << "gfold_representative_progress stage=transcript-material\n" << std::flush;
    const auto material_start = std::chrono::steady_clock::now();
    Transcript source("cpu-ipa-representative-fold-v1");
    source.absorb("fixture.fixed-public-seed", std::vector<zkdl::byte>{
        'z', 'k', 'd', 'l', '-', 'f', 'o', 'l', 'd', '-', 'r', 'e', 'p', '-', '1'});
    std::vector<FrWire> values_a(kGeneratorCount), values_b(kGeneratorCount);
    std::size_t full_field_a = 0U;
    std::size_t full_field_b = 0U;
    for (std::uint32_t index = 0U; index < kGeneratorCount; ++index) {
        values_a[index] = deterministic_full_field_scalar("fold.a", index);
        values_b[index] = deterministic_full_field_scalar("fold.b", index);
        if (!fr_canonical(values_a[index]) || !fr_canonical(values_b[index])) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL representative noncanonical-input\n";
            return false;
        }
        const std::size_t high_limb = values_a[index].size() / 2U;
        bool high_a_nonzero = false, high_b_nonzero = false;
        for (std::size_t limb = high_limb; limb < values_a[index].size(); ++limb) {
            high_a_nonzero = high_a_nonzero || values_a[index][limb] != 0U;
            high_b_nonzero = high_b_nonzero || values_b[index][limb] != 0U;
        }
        full_field_a += high_a_nonzero ? 1U : 0U;
        full_field_b += high_b_nonzero ? 1U : 0U;
    }
    std::vector<FrWire> challenges;
    for (std::uint32_t remaining = padded_size; remaining > 1U; remaining >>= 1U) {
        const std::size_t round = challenges.size();
        challenges.push_back(cpu_ipa_nonzero_challenge(
            source, "fixture.fold.challenge." + std::to_string(round)));
    }
    std::size_t full_field_challenges = 0U;
    for (const FrWire& challenge : challenges) {
        bool high_limbs_nonzero = false;
        for (std::size_t limb = challenge.size() / 2U;
             limb < challenge.size(); ++limb)
            high_limbs_nonzero = high_limbs_nonzero || challenge[limb] != 0U;
        full_field_challenges += high_limbs_nonzero ? 1U : 0U;
    }
    const double transcript_material_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - material_start).count();
    if (full_field_a < (static_cast<std::size_t>(kGeneratorCount) * 99U) / 100U ||
        full_field_b < (static_cast<std::size_t>(kGeneratorCount) * 99U) / 100U ||
        full_field_challenges != challenges.size()) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative full-field-material\n";
        return false;
    }
    std::cerr << "gfold_representative_progress stage=pool-warmup\n" << std::flush;

    // Warm the existing fixed-size worker pool before either timed path so
    // the comparison excludes one-time pool construction for both variants.
    std::vector<std::uint8_t> pool_probe(1U << 15U);
    cpu_parallel_for(pool_probe.size(), 1024U, [&](std::size_t index) {
        pool_probe[index] = static_cast<std::uint8_t>(index);
    });
    const std::size_t pool_workers = cpu_bounded_thread_pool().size();

    const CpuG1Point seed = cpu_g1_generator();
    const CpuG1Point h = cpu_g1_mul(seed, fr_from_u32(731007U));
    const CpuG1Point u = cpu_g1_mul(seed, fr_from_u32(731011U));
    const CpuG1Point repeated_point = cpu_g1_mul(seed, fr_from_u32(731017U));
    std::vector<CpuG1Point> points(kGeneratorCount, repeated_point);
    CpuIpaBasisContext basis;
    std::string error;
    std::cerr << "gfold_representative_progress stage=basis-authentication\n" << std::flush;
    const auto basis_start = std::chrono::steady_clock::now();
    if (!CpuIpaBasisContext::authenticate(
            std::move(points), h, u, basis, &error)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative basis="
                  << error << '\n';
        return false;
    }
    const double basis_authentication_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - basis_start).count();
    if (!prefix_digest_matches_legacy(basis, kGeneratorCount, h, u)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative prefix-digest\n";
        return false;
    }

    if (!projective_fold_rejects_noncanonical_input()) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative canonical-rejection\n";
        return false;
    }
    if (!dot_product_validation_boundaries()) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative dot-boundaries\n";
        return false;
    }

    std::cerr << "gfold_representative_progress stage=legacy-generic-pool-fold\n" << std::flush;
    const auto legacy_start = std::chrono::steady_clock::now();
    const FullProjectiveFoldTrace legacy = legacy_full_projective_fold(
        basis.generators(), kGeneratorCount, values_a, values_b, challenges);
    const double legacy_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - legacy_start).count();

    std::cerr << "gfold_representative_progress stage=parallel-prepared-blst-fold\n" << std::flush;
    const auto parallel_start = std::chrono::steady_clock::now();
    const FullProjectiveFoldTrace parallel = parallel_full_projective_fold(
        basis.generators(), kGeneratorCount, values_a, values_b, challenges);
    const double parallel_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - parallel_start).count();
    if (legacy.rounds.size() != challenges.size() ||
        parallel.rounds.size() != legacy.rounds.size() ||
        legacy.generator_states.size() != parallel.generator_states.size() ||
        legacy.a_states.size() != parallel.a_states.size() ||
        legacy.b_states.size() != parallel.b_states.size()) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative rounds\n";
        return false;
    }
    std::size_t compared_generator_points = 0U;
    std::size_t compared_a_values = 0U;
    std::size_t compared_b_values = 0U;
    for (std::size_t round = 0U; round < legacy.rounds.size(); ++round) {
        if (!cpu_g1_equal(legacy.rounds[round].first,
                          parallel.rounds[round].first) ||
            !cpu_g1_equal(legacy.rounds[round].second,
                          parallel.rounds[round].second)) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL representative LR round="
                      << round << '\n';
            return false;
        }
        if (legacy.generator_states[round].size() !=
                parallel.generator_states[round].size() ||
            legacy.a_states[round] != parallel.a_states[round] ||
            legacy.b_states[round] != parallel.b_states[round]) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL representative folded-scalars round="
                      << round << '\n';
            return false;
        }
        for (std::size_t index = 0U;
             index < legacy.generator_states[round].size(); ++index) {
            if (!cpu_g1_projective_tuple_equal(
                    legacy.generator_states[round][index],
                    parallel.generator_states[round][index])) {
                std::cerr << "cpu_ipa_optimized_check status=FAIL representative folded-generator round="
                          << round << " index=" << index << '\n';
                return false;
            }
            ++compared_generator_points;
        }
        compared_a_values += legacy.a_states[round].size();
        compared_b_values += legacy.b_states[round].size();
    }
    if (!cpu_g1_projective_tuple_equal(legacy.final_generator,
                                       parallel.final_generator) ||
        !cpu_g1_equal(legacy.final_generator, parallel.final_generator)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative final-ghat\n";
        return false;
    }

    auto dot_reference_start = std::chrono::steady_clock::now();
    const FrWire initial_dot_reference = reference_inner_product(
        values_a.data(), values_b.data(), kGeneratorCount);
    const double initial_dot_reference_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - dot_reference_start).count();
    auto dot_parallel_start = std::chrono::steady_clock::now();
    const FrWire initial_dot_parallel = cpu_ipa_inner_product(values_a, values_b);
    double dot_parallel_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - dot_parallel_start).count();
    if (initial_dot_reference != initial_dot_parallel ||
        !fr_canonical(initial_dot_parallel)) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative initial-dot\n";
        return false;
    }
    double round_dot_reference_seconds = 0.0;
    double round_dot_parallel_seconds = 0.0;
    std::uint64_t dot_product_slots = kGeneratorCount;
    std::vector<FrWire> dot_current_a(padded_size), dot_current_b(padded_size);
    std::copy(values_a.begin(), values_a.end(), dot_current_a.begin());
    std::copy(values_b.begin(), values_b.end(), dot_current_b.begin());
    std::size_t dot_current_size = padded_size;
    for (std::size_t round = 0U; round < challenges.size(); ++round) {
        const std::size_t half = dot_current_size / 2U;
        dot_reference_start = std::chrono::steady_clock::now();
        const FrWire left_reference = reference_inner_product(
            dot_current_a.data(), dot_current_b.data() + half, half);
        const FrWire right_reference = reference_inner_product(
            dot_current_a.data() + half, dot_current_b.data(), half);
        round_dot_reference_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - dot_reference_start).count();
        dot_parallel_start = std::chrono::steady_clock::now();
        const FrWire left_parallel = cpu_ipa_inner_product(
            dot_current_a, 0U, dot_current_b, half, half);
        const FrWire right_parallel = cpu_ipa_inner_product(
            dot_current_a, half, dot_current_b, 0U, half);
        round_dot_parallel_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - dot_parallel_start).count();
        if (left_reference != left_parallel || right_reference != right_parallel) {
            std::cerr << "cpu_ipa_optimized_check status=FAIL representative U-dot round="
                      << round << '\n';
            return false;
        }
        dot_product_slots += 2U * half;
        dot_current_a = parallel.a_states[round];
        dot_current_b = parallel.b_states[round];
        dot_current_size = half;
    }

    std::cout << "gfold_representative=PASS generator_count=" << kGeneratorCount
              << " padded=" << padded_size << " rounds=" << challenges.size()
              << " full_field_a=" << full_field_a << '/' << kGeneratorCount
              << " full_field_b=" << full_field_b << '/' << kGeneratorCount
              << " full_field_challenges=" << full_field_challenges << '/'
              << challenges.size() << " fixed_domain_separated_seed=YES"
              << " raw_prefix_digest=PREPARED_EQUAL every_LR=EQUAL"
              << " every_folded_G=EQUAL every_a_b=EQUAL final_ghat=EQUAL"
              << " compared_G_points=" << compared_generator_points
              << " compared_a=" << compared_a_values
              << " compared_b=" << compared_b_values
              << " noncanonical_a_b_challenge=REJECTED"
              << " dot_offset_shape_zero_noncanonical=CHECKED"
              << " initial_dot=EQUAL every_U_dot=EQUAL"
              << " dot_slots=" << dot_product_slots << '\n'
              << "gfold_representative_timing_seconds transcript_material="
              << transcript_material_seconds
              << " basis_authentication=" << basis_authentication_seconds
              << " legacy_generic_pool_total=" << legacy_seconds
              << " legacy_LR=" << legacy.lrs_seconds
              << " legacy_fold=" << legacy.fold_seconds
              << " parallel_prepared_total=" << parallel_seconds
              << " parallel_LR=" << parallel.lrs_seconds
              << " parallel_fold=" << parallel.fold_seconds
              << " fold_speedup=" << legacy.fold_seconds / parallel.fold_seconds
              << " dot_reference_initial=" << initial_dot_reference_seconds
              << " dot_parallel_initial=" << dot_parallel_seconds
              << " dot_reference_U=" << round_dot_reference_seconds
              << " dot_parallel_U=" << round_dot_parallel_seconds
              << " pool_workers=" << pool_workers
              << " first_round_fold_outputs=" << (padded_size / 2U)
              << " formal_segment_S=" << (1U << 18U)
              << " formal_segment_P=" << (1U << 19U)
              << " formal_dot_slots_per_opening="
              << ((1U << 18U) + 2ULL * ((1U << 19U) - 1U))
              << " formal_proof=0 full_datasets=0\n";
    if (!tiny_independent_opening_case()) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL representative tiny-opening\n";
        return false;
    }
    std::cout << "cpu_ipa_optimized_check status=PASS mode=representative-fold-only"
                 " full_datasets=0\n";
    return true;
#endif
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

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--fold-only")
            return run_fold_only_check() ? 0 : 1;
        if (argc == 2 && std::string(argv[1]) == "--fast-fold-only")
            return run_fast_fold_only_check() ? 0 : 1;
        if (argc == 2 && std::string(argv[1]) == "--representative-fold-only")
            return run_representative_fold_only_check() ? 0 : 1;
        if (argc != 1) {
            std::cerr << "usage: cpu_ipa_optimized_check [--fold-only|--fast-fold-only|--representative-fold-only]\n";
            return 2;
        }
        return run_check() ? 0 : 1;
    } catch (const std::exception& exception) {
        std::cerr << "cpu_ipa_optimized_check status=FAIL exception="
                  << exception.what() << '\n';
        return 1;
    }
}
