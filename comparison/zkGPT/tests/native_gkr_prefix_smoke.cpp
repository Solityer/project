#include "../src/native_gkr_sumcheck.hpp"

#include <mcl/bn256.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace mcl::bn;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(const quadratic_poly& left, const quadratic_poly& right,
                   const char* message) {
    require(left.a == right.a && left.b == right.b && left.c == right.c,
            message);
}

void challenged_compaction_case(std::uint64_t lane, std::uint64_t extent, unsigned rounds) {
    std::vector<F> source(static_cast<std::size_t>(extent));
    std::vector<F> multiplier(static_cast<std::size_t>(extent));
    zkgpt::NativeVector<linear_poly> old_mult(static_cast<std::size_t>(extent)), old_values(static_cast<std::size_t>(extent));
    zkgpt::NativeVector<linear_poly> old_fold_mult, old_fold_values;
    zkgpt::native_gkr::PackedAffinePrefixArena packed_mult, packed_values;
    packed_mult.initialize_scalars(extent);
    packed_values.initialize_source_view(extent, zkgpt::NativeFieldView::from_vector(source));
    require(packed_values.capacity() == 0U, "round-zero source view allocated a full value prefix");
    F claim = F_ZERO;
    for (std::uint64_t index = 0U; index < extent; ++index) {
        source[static_cast<std::size_t>(index)] = F(static_cast<int>(index * 11U + lane + 3U));
        multiplier[static_cast<std::size_t>(index)] = F(static_cast<int>(index * 7U + lane + 5U));
        old_values[static_cast<std::size_t>(index)] = linear_poly(source[static_cast<std::size_t>(index)]);
        old_mult[static_cast<std::size_t>(index)] = linear_poly(multiplier[static_cast<std::size_t>(index)]);
        packed_mult.set_scalar(index, multiplier[static_cast<std::size_t>(index)]);
        claim += source[static_cast<std::size_t>(index)] * multiplier[static_cast<std::size_t>(index)];
    }
    std::uint64_t active = extent;
    F previous_random = F_ZERO;
    for (unsigned round = 0U; round < rounds; ++round) {
        const quadratic_poly old_poly = zkgpt::native_gkr::fold_prefix_product_round(old_mult, old_values, old_fold_mult, old_fold_values, active, previous_random);
        const quadratic_poly packed_poly = zkgpt::native_gkr::fold_prefix_product_round_packed(packed_mult, packed_values, active, previous_random);
        require_equal(packed_poly, old_poly, "challenge-compacted packed round differs from the original dense affine round");
        require(old_poly.eval(F_ZERO) + old_poly.eval(F_ONE) == claim, "challenge-compacted round changed its prior claim");
        const F challenge(static_cast<int>(round * 13U + lane + 17U));
        claim = old_poly.eval(challenge);
        packed_mult.apply_challenge(challenge);
        packed_values.apply_challenge(challenge);
        old_mult.swap(old_fold_mult);
        old_values.swap(old_fold_values);
        active = active / 2U + active % 2U;
        previous_random = challenge;
        for (std::uint64_t index = 0U; index < active; ++index) {
            require(packed_mult.coordinate_value(index, previous_random) == old_mult[static_cast<std::size_t>(index)].eval(previous_random), "compacted multiplier differs after native challenge");
            require(packed_values.coordinate_value(index, previous_random) == old_values[static_cast<std::size_t>(index)].eval(previous_random), "compacted value differs after native challenge");
        }
        require(!packed_mult.affine_mode() && !packed_values.affine_mode(), "challenge transition retained affine mode");
        if (active != 0U) require(!packed_values.source_view_mode(), "challenge transition retained a source view");
    }
    const F old_terminal = active == 0U ? F_ZERO : old_mult[0].eval(previous_random) * old_values[0].eval(previous_random);
    const F packed_terminal = packed_mult.first_value(previous_random) * packed_values.first_value(previous_random);
    require(packed_terminal == old_terminal && packed_terminal == claim, "challenge-compacted terminal differs from the original proof claim");
}

void challenged_compaction_cases() {
    challenged_compaction_case(0U, 0U, 6U);
    challenged_compaction_case(1U, 1U, 6U);
    challenged_compaction_case(2U, 3U, 6U);
    challenged_compaction_case(3U, 9U, 6U);
    challenged_compaction_case(4U, 32U, 6U);
    std::cout << "NATIVE_GKR_CHALLENGE_COMPACTION_OK cases=5 rounds=6\n";
}

void parallel_challenge_apply_equivalence() {
    constexpr std::uint64_t source_extent = (std::uint64_t(1) << 16U) + 1U;
    constexpr std::uint64_t folded_extent = source_extent / 2U + source_extent % 2U;
    require(zkgpt::native_gkr::challenge_apply_worker_count(folded_extent) > 1U,
            "parallel challenge fixture did not select multiple workers");
    std::vector<F> source(static_cast<std::size_t>(source_extent));
    for (std::uint64_t index = 0U; index < source_extent; ++index)
        source[static_cast<std::size_t>(index)] = F(static_cast<int>(index % 101U + 3U));
    for (const F challenge : {F_ZERO, F_ONE}) {
        zkgpt::native_gkr::PackedAffinePrefixArena source_pending;
        source_pending.initialize_source_view(source_extent, zkgpt::NativeFieldView::from_vector(source));
        source_pending.finish_fold(folded_extent);
        source_pending.apply_challenge(challenge);
        for (std::uint64_t index = 0U; index < folded_extent; ++index) {
            const std::uint64_t even = index * 2U;
            const std::uint64_t odd = even + 1U;
            const F zero_value = source[static_cast<std::size_t>(even)];
            const F one_value = odd < source_extent ? source[static_cast<std::size_t>(odd)] : F_ZERO;
            const F expected = (one_value - zero_value).isZero() ? zero_value : (one_value - zero_value) * challenge + zero_value;
            require(source_pending.coordinate_value(index, challenge) == expected,
                    "parallel pending-source challenge differs from the serial affine formula");
        }
        zkgpt::native_gkr::PackedAffinePrefixArena affine;
        affine.initialize_scalars(source_extent);
        for (std::uint64_t index = 0U; index < folded_extent; ++index) {
            const F slope = index % 7U == 0U ? F_ZERO : F(static_cast<int>(index % 23U + 5U));
            const F intercept(static_cast<int>(index % 29U + 7U));
            affine.store_folded(index, linear_poly(slope, intercept));
        }
        affine.finish_fold(folded_extent);
        affine.apply_challenge(challenge);
        for (std::uint64_t index = 0U; index < folded_extent; ++index) {
            const F slope = index % 7U == 0U ? F_ZERO : F(static_cast<int>(index % 23U + 5U));
            const F intercept(static_cast<int>(index % 29U + 7U));
            const F expected = slope.isZero() ? intercept : slope * challenge + intercept;
            require(affine.coordinate_value(index, challenge) == expected,
                    "parallel affine challenge differs from the serial affine formula");
        }
    }
    std::cout << "NATIVE_GKR_PARALLEL_CHALLENGE_OK extent=" << folded_extent << " challenges=zero,one\n";
}

void source_materialization_microbenchmark() {
    constexpr std::uint64_t extent = 300001U;
    const F challenge(29);
    std::vector<F> source(static_cast<std::size_t>(extent));
    std::vector<F> multiplier(static_cast<std::size_t>(extent));
    for (std::uint64_t index = 0U; index < extent; ++index) {
        source[static_cast<std::size_t>(index)] = F(static_cast<int>(index % 101U + 3U));
        multiplier[static_cast<std::size_t>(index)] = F(static_cast<int>(index % 97U + 5U));
    }
    quadratic_poly old_poly;
    F old_terminal = F_ZERO;
    const auto old_begin = std::chrono::steady_clock::now();
    {
        zkgpt::NativeVector<linear_poly> old_mult(static_cast<std::size_t>(extent)), old_values(static_cast<std::size_t>(extent));
        zkgpt::NativeVector<linear_poly> folded_mult, folded_values;
        for (std::uint64_t index = 0U; index < extent; ++index) {
            old_mult[static_cast<std::size_t>(index)] = linear_poly(multiplier[static_cast<std::size_t>(index)]);
            old_values[static_cast<std::size_t>(index)] = linear_poly(source[static_cast<std::size_t>(index)]);
        }
        old_poly = zkgpt::native_gkr::fold_prefix_product_round(old_mult, old_values, folded_mult, folded_values, extent, F_ZERO);
        old_terminal = folded_mult[0].eval(challenge) * folded_values[0].eval(challenge);
    }
    const auto old_end = std::chrono::steady_clock::now();
    std::atomic<std::uint64_t> getter_reads{0U};
    quadratic_poly packed_poly;
    F packed_terminal = F_ZERO;
    const auto packed_begin = std::chrono::steady_clock::now();
    {
        zkgpt::native_gkr::PackedAffinePrefixArena packed_mult, packed_values;
        packed_mult.initialize_scalars(extent);
        for (std::uint64_t index = 0U; index < extent; ++index) packed_mult.set_scalar(index, multiplier[static_cast<std::size_t>(index)]);
        packed_values.initialize_source_view(extent, zkgpt::NativeFieldView::borrowed(static_cast<std::size_t>(extent), &source, [&](std::size_t index) {
            getter_reads.fetch_add(1U, std::memory_order_relaxed);
            return source.at(index);
        }));
        packed_poly = zkgpt::native_gkr::fold_prefix_product_round_packed(packed_mult, packed_values, extent, F_ZERO);
        packed_mult.apply_challenge(challenge);
        packed_values.apply_challenge(challenge);
        packed_terminal = packed_mult.first_value(challenge) * packed_values.first_value(challenge);
    }
    const auto packed_end = std::chrono::steady_clock::now();
    require_equal(packed_poly, old_poly, "microbenchmark packed polynomial differs from its old-path control");
    require(packed_terminal == old_terminal, "microbenchmark compacted terminal differs from its old-path control");
    const auto old_us = std::chrono::duration_cast<std::chrono::microseconds>(old_end - old_begin).count();
    const auto packed_us = std::chrono::duration_cast<std::chrono::microseconds>(packed_end - packed_begin).count();
    std::cout << "NATIVE_GKR_PREFIX_MICROBENCH extent=" << extent << " source_reads_new=" << getter_reads.load() << " source_reads_old=" << extent << " old_us=" << old_us << " compacted_us=" << packed_us << "\n";
}

void dense_round_equivalence() {
    constexpr std::uint64_t domain = 64;
    constexpr std::uint64_t logical = 9;
    zkgpt::NativeVector<linear_poly> prefix_mult(logical), prefix_values(logical);
    zkgpt::NativeVector<linear_poly> prefix_fold_mult, prefix_fold_values;
    zkgpt::native_gkr::PackedAffinePrefixArena packed_mult, packed_values;
    packed_mult.initialize_scalars(logical);
    packed_values.initialize_scalars(logical);
    zkgpt::NativeVector<linear_poly> dense_mult(domain), dense_values(domain);
    zkgpt::NativeVector<linear_poly> dense_fold_mult, dense_fold_values;
    F claim = F_ZERO;
    for (std::uint64_t index = 0; index < logical; ++index) {
        const F m(static_cast<int>(index * 3 + 1));
        const F v(static_cast<int>(index * 5 + 2));
        prefix_mult[index] = linear_poly(m);
        prefix_values[index] = linear_poly(v);
        packed_mult.set_scalar(index, m);
        packed_values.set_scalar(index, v);
        dense_mult[index] = linear_poly(m);
        dense_values[index] = linear_poly(v);
        claim += m * v;
    }

    std::uint64_t prefix_extent = logical;
    std::uint64_t dense_extent = domain;
    F previous_random = F_ZERO;
    for (unsigned round = 0; round < 6; ++round) {
        const quadratic_poly prefix_poly = zkgpt::native_gkr::fold_prefix_product_round(
            prefix_mult, prefix_values, prefix_fold_mult, prefix_fold_values,
            prefix_extent, previous_random);
        const quadratic_poly dense_poly = zkgpt::native_gkr::fold_prefix_product_round(
            dense_mult, dense_values, dense_fold_mult, dense_fold_values,
            dense_extent, previous_random);
        const quadratic_poly packed_poly = zkgpt::native_gkr::fold_prefix_product_round_packed(
            packed_mult, packed_values, prefix_extent, previous_random);
        require_equal(prefix_poly, dense_poly,
                      "prefix fold polynomial differs from dense zero-padded reference");
        require_equal(packed_poly, dense_poly,
                      "packed affine fold polynomial differs from dense zero-padded reference");
        require(prefix_poly.eval(F_ZERO) + prefix_poly.eval(F_ONE) == claim,
                "sumcheck round endpoints do not sum to the prior claim");
        if (round == 0) {
            require(packed_mult.first_slope() != F_ZERO && packed_values.first_slope() != F_ZERO,
                    "packed affine fold failed to preserve nonzero slopes");
        }

        const F challenge(static_cast<int>(round + 2));
        claim = prefix_poly.eval(challenge);
        previous_random = challenge;
        prefix_mult.swap(prefix_fold_mult);
        prefix_values.swap(prefix_fold_values);
        dense_mult.swap(dense_fold_mult);
        dense_values.swap(dense_fold_values);
        prefix_extent = prefix_extent / 2 + prefix_extent % 2;
        dense_extent >>= 1;
    }
    require(prefix_extent == 1 && dense_extent == 1,
            "dense-reference test did not retain the original round count");
    const F terminal = prefix_mult[0].eval(previous_random) *
                       prefix_values[0].eval(previous_random);
    require(terminal == claim,
            "prefix terminal does not equal the final challenged claim");
    require(packed_mult.first_value(previous_random) * packed_values.first_value(previous_random) == claim,
            "packed prefix terminal does not equal the final challenged claim");
    require(packed_mult.affine_mode() && packed_values.affine_mode(),
            "packed prefix did not enter affine mode after folding");
}

void asymmetric_lanes_and_chunk_boundary() {
    constexpr std::uint64_t domain = 128;
    constexpr std::uint64_t lane_extents[] = {0U, 1U, 3U, 9U};
    constexpr unsigned rounds = 7U;
    F previous_random = F_ZERO;
    std::vector<zkgpt::native_gkr::PackedAffinePrefixArena> packed_mult(4U), packed_values(4U);
    std::vector<zkgpt::NativeVector<linear_poly>> dense_mult(4U), dense_values(4U),
        dense_fold_mult(4U), dense_fold_values(4U);
    std::uint64_t active[4] = {0U, 1U, 3U, 9U};
    F claims[4] = {F_ZERO, F_ZERO, F_ZERO, F_ZERO};
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        packed_mult[lane].initialize_scalars(active[lane]);
        packed_values[lane].initialize_scalars(active[lane]);
        dense_mult[lane].resize(static_cast<std::size_t>(active[lane]));
        dense_values[lane].resize(static_cast<std::size_t>(active[lane]));
        for (std::uint64_t index = 0U; index < active[lane]; ++index) {
            const F mult(static_cast<int>(2U + lane * 7U + index * 5U));
            const F value(static_cast<int>(3U + lane * 11U + index * 9U));
            packed_mult[lane].set_scalar(index, mult);
            packed_values[lane].set_scalar(index, value);
            dense_mult[lane][index] = linear_poly(mult);
            dense_values[lane][index] = linear_poly(value);
            claims[lane] += mult * value;
        }
    }

    for (unsigned round = 0U; round < rounds; ++round) {
        for (std::size_t lane = 0U; lane < 4U; ++lane) {
            const quadratic_poly packed = zkgpt::native_gkr::fold_prefix_product_round_packed(
                packed_mult[lane], packed_values[lane], active[lane], previous_random);
            const quadratic_poly dense = zkgpt::native_gkr::fold_prefix_product_round(
                dense_mult[lane], dense_values[lane], dense_fold_mult[lane],
                dense_fold_values[lane], active[lane], previous_random);
            require_equal(packed, dense, "asymmetric packed lane differs from dense reference");
            require(packed.eval(F_ZERO) + packed.eval(F_ONE) == claims[lane],
                    "asymmetric packed lane changed its sumcheck claim");
            claims[lane] = packed.eval(F(static_cast<int>(round + 13U)));
            dense_mult[lane].swap(dense_fold_mult[lane]);
            dense_values[lane].swap(dense_fold_values[lane]);
            active[lane] = active[lane] / 2U + active[lane] % 2U;
        }
        previous_random = F(static_cast<int>(round + 13U));
    }
    require(domain == (std::uint64_t(1) << rounds), "asymmetric test common domain changed");
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        const F packed_terminal = packed_mult[lane].first_value(previous_random) *
                                  packed_values[lane].first_value(previous_random);
        const F dense_terminal = dense_mult[lane].empty() ? F_ZERO :
            dense_mult[lane][0].eval(previous_random) * dense_values[lane][0].eval(previous_random);
        require(packed_terminal == claims[lane] && packed_terminal == dense_terminal,
                "asymmetric packed lane terminal differs from dense zero-padding");
    }

    constexpr std::uint64_t logical = (std::uint64_t(1) << 17U) + 1U;
    constexpr std::uint64_t fold_extent = logical / 2U + logical % 2U;
    zkgpt::native_gkr::PackedAffinePrefixArena packed_mult_boundary, packed_values_boundary;
    packed_mult_boundary.initialize_scalars(logical);
    packed_values_boundary.initialize_scalars(logical);
    zkgpt::NativeVector<linear_poly> dense_mult_boundary(static_cast<std::size_t>(logical));
    zkgpt::NativeVector<linear_poly> dense_values_boundary(static_cast<std::size_t>(logical));
    zkgpt::NativeVector<linear_poly> dense_fold_mult_boundary, dense_fold_values_boundary;
    F boundary_claim = F_ZERO;
    for (std::uint64_t index = 0U; index < logical; ++index) {
        const F mult(static_cast<int>((index % 97U) + 1U));
        const F value(static_cast<int>((index % 89U) + 3U));
        packed_mult_boundary.set_scalar(index, mult);
        packed_values_boundary.set_scalar(index, value);
        dense_mult_boundary[index] = linear_poly(mult);
        dense_values_boundary[index] = linear_poly(value);
        boundary_claim += mult * value;
    }
    const F boundary_challenge(19);
    const quadratic_poly packed_boundary = zkgpt::native_gkr::fold_prefix_product_round_packed(
        packed_mult_boundary, packed_values_boundary, logical, boundary_challenge);
    const quadratic_poly dense_boundary = zkgpt::native_gkr::fold_prefix_product_round(
        dense_mult_boundary, dense_values_boundary, dense_fold_mult_boundary,
        dense_fold_values_boundary, logical, boundary_challenge);
    require_equal(packed_boundary, dense_boundary,
                  "packed fold crossing its bounded scratch chunk differs from dense reference");
    require(packed_boundary.eval(F_ZERO) + packed_boundary.eval(F_ONE) == boundary_claim,
            "chunk-boundary packed fold changed its sumcheck claim");
    require(fold_extent == packed_values_boundary.logical_extent() &&
                packed_values_boundary.capacity() == logical + 1U,
            "packed chunk-boundary fold lost its odd-padded in-place arena extent");
}

void virtual_37_bit_domain() {
    constexpr int bits = 37;
    const std::uint64_t original_domain =
        zkgpt::native_gkr::checked_domain_size(bits, "37-bit test domain");
    require(original_domain == (std::uint64_t(1) << 37),
            "37-bit full domain was narrowed");

    std::vector<F> point(bits);
    for (int bit = 0; bit < bits; ++bit) point[bit] = F(bit + 2);
    const std::uint64_t absolute_coordinate = (std::uint64_t(1) << 35) + 5;
    zkgpt::native_gkr::EqualityWeightProvider equality(point, bits);
    F direct = F_ONE;
    for (int bit = 0; bit < bits; ++bit) {
        direct *= ((absolute_coordinate >> bit) & 1)
            ? point[bit] : F_ONE - point[bit];
    }
    require(equality.at(absolute_coordinate) == direct,
            "on-demand equality weight differs from the dense LSB-first formula");

    constexpr std::uint64_t logical = 3;
    zkgpt::NativeVector<linear_poly> mult(logical), values(logical), folded_mult,
        folded_values;
    zkgpt::native_gkr::PackedAffinePrefixArena packed_mult, packed_values;
    packed_mult.initialize_scalars(logical);
    packed_values.initialize_scalars(logical);
    for (std::uint64_t index = 0; index < logical; ++index) {
        const F mult_value(static_cast<int>(index + 4));
        const F value(static_cast<int>(index * 2 + 1));
        mult[index] = linear_poly(mult_value);
        values[index] = linear_poly(value);
        packed_mult.set_scalar(index, mult_value);
        packed_values.set_scalar(index, value);
    }
    std::vector<linear_poly> reference_mult(logical), reference_values(logical);
    for (std::uint64_t index = 0; index < logical; ++index) {
        reference_mult[index] = linear_poly(F(static_cast<int>(index + 4)));
        reference_values[index] = linear_poly(F(static_cast<int>(index * 2 + 1)));
    }
    F claim = F_ZERO;
    for (std::uint64_t index = 0; index < logical; ++index)
        claim += mult[index].b * values[index].b;

    std::uint64_t active_extent = logical;
    std::uint64_t remaining_domain = original_domain;
    F previous_random = F_ZERO;
    for (int round = 0; round < bits; ++round) {
        const quadratic_poly polynomial =
            zkgpt::native_gkr::fold_prefix_product_round(
                mult, values, folded_mult, folded_values, active_extent,
                previous_random);
        const quadratic_poly packed_polynomial =
            zkgpt::native_gkr::fold_prefix_product_round_packed(
                packed_mult, packed_values, active_extent, previous_random);
        quadratic_poly dense_reference;
        dense_reference.clear();
        const std::uint64_t reference_fold_extent = reference_mult.size() / 2 + reference_mult.size() % 2;
        for (std::uint64_t pair = 0; pair < reference_fold_extent; ++pair) {
            const std::uint64_t even = pair * 2, odd = even + 1;
            const F mult_zero = reference_mult[even].eval(previous_random);
            const F mult_one = odd < reference_mult.size() ? reference_mult[odd].eval(previous_random) : F_ZERO;
            const F value_zero = reference_values[even].eval(previous_random);
            const F value_one = odd < reference_values.size() ? reference_values[odd].eval(previous_random) : F_ZERO;
            dense_reference = dense_reference + linear_poly(mult_one - mult_zero, mult_zero) * linear_poly(value_one - value_zero, value_zero);
        }
        require_equal(polynomial, dense_reference, "37-bit virtual fold differs from independent dense-support reference");
        require_equal(packed_polynomial, dense_reference,
                      "37-bit packed fold differs from independent dense-support reference");
        require(polynomial.eval(F_ZERO) + polynomial.eval(F_ONE) == claim,
                "37-round prefix fold changed the sumcheck claim");
        const F challenge(round + 3);
        claim = polynomial.eval(challenge);
        require(packed_polynomial.eval(challenge) == claim,
                "37-bit packed fold changed the challenged claim");
        for (std::uint64_t pair = 0; pair < reference_fold_extent; ++pair) {
            const std::uint64_t even = pair * 2, odd = even + 1;
            const F mult_zero = reference_mult[even].eval(previous_random);
            const F mult_one = odd < reference_mult.size() ? reference_mult[odd].eval(previous_random) : F_ZERO;
            const F value_zero = reference_values[even].eval(previous_random);
            const F value_one = odd < reference_values.size() ? reference_values[odd].eval(previous_random) : F_ZERO;
            reference_mult[pair] = linear_poly(mult_one - mult_zero, mult_zero);
            reference_values[pair] = linear_poly(value_one - value_zero, value_zero);
        }
        reference_mult.resize(static_cast<std::size_t>(reference_fold_extent));
        reference_values.resize(static_cast<std::size_t>(reference_fold_extent));
        previous_random = challenge;
        mult.swap(folded_mult);
        values.swap(folded_values);
        active_extent = active_extent / 2 + active_extent % 2;
        remaining_domain >>= 1;
    }
    require(remaining_domain == 1 && active_extent == 1,
            "37-bit prefix fold stopped before the original terminal round");
    require(mult[0].eval(previous_random) * values[0].eval(previous_random) == claim,
            "37-bit prefix fold terminal differs from the challenged claim");
    require(reference_mult[0].eval(previous_random) * reference_values[0].eval(previous_random) == claim,
            "37-bit virtual terminal differs from independent dense-support reference");
    require(packed_mult.first_value(previous_random) * packed_values.first_value(previous_random) == claim,
            "37-bit packed terminal differs from the challenged claim");
}

void empty_lane_zero_terminal() {
    zkgpt::NativeVector<linear_poly> mult, values, folded_mult, folded_values;
    quadratic_poly polynomial = zkgpt::native_gkr::fold_prefix_product_round(mult, values, folded_mult, folded_values, 0, F(7));
    require(polynomial.eval(F_ZERO).isZero() && polynomial.eval(F_ONE).isZero(), "empty lane contributed to a sumcheck round");
    require(folded_mult.size() == 1 && folded_values.size() == 1, "empty lane lacks its zero terminal slot");
    require(folded_mult[0].eval(F(11)).isZero() && folded_values[0].eval(F(13)).isZero(), "empty lane terminal slot is nonzero");
    zkgpt::native_gkr::PackedAffinePrefixArena packed_mult, packed_values;
    packed_mult.initialize_scalars(0U);
    packed_values.initialize_scalars(0U);
    const quadratic_poly packed = zkgpt::native_gkr::fold_prefix_product_round_packed(
        packed_mult, packed_values, 0U, F(7));
    require(packed.eval(F_ZERO).isZero() && packed.eval(F_ONE).isZero(),
            "packed empty lane contributed to a sumcheck round");
    require(packed_mult.capacity() == 2U && packed_values.capacity() == 2U &&
                packed_mult.first_value(F(11)).isZero() && packed_values.first_value(F(13)).isZero(),
            "packed empty lane does not preserve its zero sentinel");
}

} // namespace

int main() {
    initPairing(mcl::BN_SNARK1);
    challenged_compaction_cases();
    parallel_challenge_apply_equivalence();
    source_materialization_microbenchmark();
    dense_round_equivalence();
    asymmetric_lanes_and_chunk_boundary();
    virtual_37_bit_domain();
    empty_lane_zero_terminal();
    std::cout << "NATIVE_GKR_PREFIX_OK rounds=37 logical=3 full_domain=2^37 Fr_bytes=" << sizeof(F) << " linear_poly_bytes=" << sizeof(linear_poly) << "\n";
}
