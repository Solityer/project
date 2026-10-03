#include "../src/native_gkr_sumcheck.hpp"

#include <mcl/bn256.hpp>
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

void dense_round_equivalence() {
    constexpr std::uint64_t domain = 64;
    constexpr std::uint64_t logical = 9;
    zkgpt::NativeVector<linear_poly> prefix_mult(logical), prefix_values(logical);
    zkgpt::NativeVector<linear_poly> prefix_fold_mult, prefix_fold_values;
    zkgpt::NativeVector<linear_poly> dense_mult(domain), dense_values(domain);
    zkgpt::NativeVector<linear_poly> dense_fold_mult, dense_fold_values;
    F claim = F_ZERO;
    for (std::uint64_t index = 0; index < logical; ++index) {
        const F m(static_cast<int>(index * 3 + 1));
        const F v(static_cast<int>(index * 5 + 2));
        prefix_mult[index] = linear_poly(m);
        prefix_values[index] = linear_poly(v);
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
        require_equal(prefix_poly, dense_poly,
                      "prefix fold polynomial differs from dense zero-padded reference");
        require(prefix_poly.eval(F_ZERO) + prefix_poly.eval(F_ONE) == claim,
                "sumcheck round endpoints do not sum to the prior claim");

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
    for (std::uint64_t index = 0; index < logical; ++index) {
        mult[index] = linear_poly(F(static_cast<int>(index + 4)));
        values[index] = linear_poly(F(static_cast<int>(index * 2 + 1)));
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
        require(polynomial.eval(F_ZERO) + polynomial.eval(F_ONE) == claim,
                "37-round prefix fold changed the sumcheck claim");
        const F challenge(round + 3);
        claim = polynomial.eval(challenge);
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
}

void empty_lane_zero_terminal() {
    zkgpt::NativeVector<linear_poly> mult, values, folded_mult, folded_values;
    quadratic_poly polynomial = zkgpt::native_gkr::fold_prefix_product_round(mult, values, folded_mult, folded_values, 0, F(7));
    require(polynomial.eval(F_ZERO).isZero() && polynomial.eval(F_ONE).isZero(), "empty lane contributed to a sumcheck round");
    require(folded_mult.size() == 1 && folded_values.size() == 1, "empty lane lacks its zero terminal slot");
    require(folded_mult[0].eval(F(11)).isZero() && folded_values[0].eval(F(13)).isZero(), "empty lane terminal slot is nonzero");
}

} // namespace

int main() {
    initPairing(mcl::BN_SNARK1);
    dense_round_equivalence();
    virtual_37_bit_domain();
    empty_lane_zero_terminal();
    std::cout << "NATIVE_GKR_PREFIX_OK rounds=37 logical=3 full_domain=2^37 Fr_bytes=" << sizeof(F) << " linear_poly_bytes=" << sizeof(linear_poly) << "\n";
}
