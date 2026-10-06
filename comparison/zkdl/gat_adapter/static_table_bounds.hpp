#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace zkdl_gat {
namespace static_table_bounds {

// Compute the exact output interval of a symmetric input interval in an
// authenticated monotone lookup table. The caller validates monotonicity and
// table identity before using this bound.
inline bool symmetric_output_interval(const std::vector<std::int64_t> &values, std::int64_t table_min,
                                     std::int64_t table_max, std::int64_t input_abs,
                                     std::int64_t &output_min, std::int64_t &output_max) noexcept {
    if (values.empty() || input_abs < 0 || table_max < table_min) return false;
    const __int128 expected_rows = static_cast<__int128>(table_max) - table_min + 1;
    const __int128 low_input = -static_cast<__int128>(input_abs);
    const __int128 high_input = static_cast<__int128>(input_abs);
    if (expected_rows <= 0 || expected_rows != static_cast<__int128>(values.size()) ||
        low_input < table_min || high_input > table_max) return false;

    const std::size_t first = static_cast<std::size_t>(low_input - table_min);
    const std::size_t last = static_cast<std::size_t>(high_input - table_min);
    if (first > last || last >= values.size()) return false;
    for (std::size_t index = first; index < last; ++index)
        if (values[index] > values[index + 1U]) return false;
    output_min = values[first];
    output_max = values[last];
    return output_min <= output_max;
}

inline bool exp_domain_covers_delta(std::int64_t delta_bound, std::uint64_t exp_max) noexcept {
    return delta_bound >= 0 && static_cast<std::uint64_t>(delta_bound) <= exp_max;
}

// A positive-weighted average cannot leave the range of its inputs.  The
// returned bound includes the canonical nearest-ties-away integer rounding
// used by aggregate_head: floor((2*s_num*input_abs+s_den)/(2*s_den)).
// The native caller separately authenticates the table and graph boundaries;
// this helper repeats the positivity/nonempty checks at the bound boundary.
inline bool positive_weighted_round_abs_bound(const std::vector<std::int64_t> &weights,
                                             std::uint64_t minimum_group_degree,
                                             std::int64_t input_abs,
                                             std::uint64_t numerator_scale,
                                             std::uint64_t denominator_scale,
                                             std::int64_t &output_abs) noexcept {
    if (weights.empty() || minimum_group_degree == 0U || input_abs < 0 || numerator_scale == 0U || denominator_scale == 0U)
        return false;
    for (const auto weight : weights)
        if (weight <= 0) return false;

    const __int128 product = static_cast<__int128>(numerator_scale) * static_cast<__int128>(input_abs);
    const __int128 max_i128 = ((static_cast<__int128>(1) << 126U) - 1) * 2 + 1;
    const __int128 denominator = static_cast<__int128>(denominator_scale);
    if (product > (max_i128 - denominator) / 2) return false;
    const __int128 numerator = 2 * product + denominator;
    const __int128 rounded_bound = numerator / (2 * denominator);
    if (rounded_bound < 0 || rounded_bound > static_cast<__int128>(std::numeric_limits<std::int64_t>::max())) return false;
    output_abs = static_cast<std::int64_t>(rounded_bound);
    return true;
}

}  // namespace static_table_bounds
}  // namespace zkdl_gat
