#pragma once

#include "polynomial.h"
#include "../gat_adapter/native_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace zkgpt::native_gkr {

inline std::uint64_t checked_domain_size(int bits, const char* name) {
    if (bits < 0) return 0;
    if (bits >= std::numeric_limits<std::uint64_t>::digits ||
        bits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error(std::string(name) + " bit length exceeds native index width");
    }
    return std::uint64_t(1) << bits;
}

inline std::size_t checked_extent(std::uint64_t extent, const char* name) {
    if (extent > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error(std::string(name) + " extent exceeds native storage index width");
    }
    return static_cast<std::size_t>(extent);
}

class EqualityWeightProvider {
    std::vector<F> point_;
    std::vector<F> low_weights_;
    unsigned bits_ = 0;
    unsigned low_bits_ = 0;
    std::uint64_t low_mask_ = 0;
    F scale_ = F_ONE;

    void initialize_low_weights() {
        low_bits_ = std::min<unsigned>(12, bits_);
        const std::size_t count = std::size_t(1) << low_bits_;
        low_weights_.assign(count, F_ZERO);
        low_weights_[0] = scale_;
        for (unsigned bit = 0; bit < low_bits_; ++bit) {
            const std::size_t old_count = std::size_t(1) << bit;
            for (std::size_t index = 0; index < old_count; ++index) {
                const F weight = low_weights_[index];
                low_weights_[index] = weight * (F_ONE - point_[bit]);
                low_weights_[index + old_count] = weight * point_[bit];
            }
        }
        low_mask_ = count - 1;
    }

public:
    EqualityWeightProvider() = default;

    EqualityWeightProvider(const std::vector<F>& point, int bits, const F& scale = F_ONE)
        : point_(point), scale_(scale) {
        if (bits < 0 || bits >= std::numeric_limits<std::uint64_t>::digits ||
            bits >= std::numeric_limits<std::size_t>::digits) {
            throw std::length_error("native equality point bit length is invalid");
        }
        bits_ = static_cast<unsigned>(bits);
        if (point_.size() < bits_ && !scale_.isZero()) throw std::invalid_argument("native equality point is shorter than its domain");
        if (point_.size() < bits_) point_.resize(bits_, F_ZERO);
        initialize_low_weights();
    }

    std::uint64_t domain_size() const {
        return std::uint64_t(1) << bits_;
    }

    F at(std::uint64_t coordinate) const {
        if (coordinate >= domain_size()) throw std::out_of_range("native equality coordinate exceeds its domain");
        F result = low_weights_[static_cast<std::size_t>(coordinate & low_mask_)];
        for (unsigned bit = low_bits_; bit < bits_; ++bit) {
            result *= ((coordinate >> bit) & 1) ? point_[bit] : F_ONE - point_[bit];
        }
        return result;
    }
};

class CombinedEqualityWeightProvider {
    EqualityWeightProvider first_;
    EqualityWeightProvider second_;

public:
    CombinedEqualityWeightProvider(const std::vector<F>& first_point,
                                   const std::vector<F>& second_point,
                                   int bits, const F& first_scale,
                                   const F& second_scale)
        : first_(first_point, bits, first_scale),
          second_(second_point, bits, second_scale) {}

    F at(std::uint64_t coordinate) const {
        return first_.at(coordinate) + second_.at(coordinate);
    }
};

inline linear_poly fold_prefix_linear(const zkgpt::NativeVector<linear_poly>& values,
                                      std::uint64_t logical_extent,
                                      std::uint64_t pair_index,
                                      const F& previous_random) {
    if (logical_extent == 0 || pair_index >= logical_extent / 2 + logical_extent % 2) {
        throw std::out_of_range("native sumcheck prefix fold pair is outside the active extent");
    }
    const std::uint64_t even = pair_index * 2;
    const std::uint64_t odd = even + 1;
    if (even >= values.size()) throw std::out_of_range("native sumcheck prefix value is missing");
    const F zero_value = values[static_cast<std::size_t>(even)].eval(previous_random);
    const F one_value = odd < logical_extent
        ? values.at(static_cast<std::size_t>(odd)).eval(previous_random)
        : F_ZERO;
    return linear_poly(one_value - zero_value, zero_value);
}

inline quadratic_poly fold_prefix_product_round(
    const zkgpt::NativeVector<linear_poly>& mult,
    const zkgpt::NativeVector<linear_poly>& values,
    zkgpt::NativeVector<linear_poly>& folded_mult,
    zkgpt::NativeVector<linear_poly>& folded_values,
    std::uint64_t logical_extent,
    const F& previous_random) {
    if (logical_extent > mult.size() || logical_extent > values.size()) {
        throw std::out_of_range("native sumcheck active prefix exceeds its source vectors");
    }
    const std::uint64_t folded_extent = logical_extent / 2 + logical_extent % 2;
    const std::size_t storage_extent = checked_extent(std::max<std::uint64_t>(1, folded_extent), "native sumcheck fold");
    folded_mult.resize(storage_extent);
    folded_values.resize(storage_extent);
    quadratic_poly result;
    result.clear();
    for (std::uint64_t pair = 0; pair < folded_extent; ++pair) {
        folded_mult[static_cast<std::size_t>(pair)] = fold_prefix_linear(mult, logical_extent, pair, previous_random);
        folded_values[static_cast<std::size_t>(pair)] = fold_prefix_linear(values, logical_extent, pair, previous_random);
        result = result + folded_mult[static_cast<std::size_t>(pair)] * folded_values[static_cast<std::size_t>(pair)];
    }
    return result;
}

} // namespace zkgpt::native_gkr
