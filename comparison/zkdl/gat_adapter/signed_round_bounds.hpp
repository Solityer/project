#pragma once

#include "gat_adapter.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace zkdl_gat::signed_round_bounds {

struct Widths {
    std::int64_t denominator_scale = 0;
    std::int64_t a_max = 0;
    std::int64_t q_max = 0;
    std::int64_t r_max = 0;
    std::int64_t delta_max = 0;
    std::int64_t divisor_max = 0;
    std::uint32_t a_bits = 0;
    std::uint32_t q_bits = 0;
    std::uint32_t r_bits = 0;
    std::uint32_t delta_bits = 0;
    std::uint32_t divisor_bits = 0;
    bool binds_raw_denominator = false;
};

inline std::int64_t checked_bound(__int128 value, const std::string &label) {
    if (value < 0 || value > std::numeric_limits<std::int64_t>::max())
        throw std::runtime_error("signed-round bound exceeds nonnegative int64: " + label);
    return static_cast<std::int64_t>(value);
}

inline std::uint32_t unsigned_width(std::int64_t inclusive_max, const std::string &label) {
    if (inclusive_max < 0) throw std::runtime_error("negative unsigned signed-round bound: " + label);
    std::uint64_t value = static_cast<std::uint64_t>(inclusive_max);
    std::uint32_t bits = 0U;
    do {
        ++bits;
        value >>= 1U;
    } while (value != 0U);
    if (bits > 63U) throw std::runtime_error("signed-round bit width exceeds signed wire capacity: " + label);
    return bits;
}

inline std::uint64_t decimal_component(const std::string &text, std::size_t begin, std::size_t end,
                                       const std::string &label) {
    if (begin >= end) throw std::runtime_error("missing signed-round operation coordinate: " + label);
    std::uint64_t value = 0U;
    for (std::size_t index = begin; index < end; ++index) {
        const char digit = text[index];
        if (digit < '0' || digit > '9') throw std::runtime_error("invalid signed-round operation coordinate: " + label);
        const std::uint64_t numeric = static_cast<std::uint64_t>(digit - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - numeric) / 10U)
            throw std::runtime_error("signed-round operation coordinate overflows: " + label);
        value = value * 10U + numeric;
    }
    return value;
}

inline const IntegerBoundFamily &family_for(const IntegerBoundLedger &ledger, std::uint64_t layer,
                                             std::uint64_t head, bool output,
                                             const std::string &operation_id) {
    const IntegerBoundFamily *found = nullptr;
    for (const auto &family : ledger.families) {
        if (family.layer == layer && family.head == head && family.output == output) {
            if (found != nullptr) throw std::runtime_error("duplicate authenticated signed-round ledger family: " + operation_id);
            found = &family;
        }
    }
    if (found == nullptr) throw std::runtime_error("missing authenticated signed-round ledger family: " + operation_id);
    const std::string expected = output ? "output/head" + std::to_string(head)
                                        : "hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head);
    if (found->family_id != expected || found->max_dividend_abs < 0 || found->max_aux_abs < 0 || found->max_divisor <= 0)
        throw std::runtime_error("malformed authenticated signed-round ledger family: " + operation_id);
    return *found;
}

inline std::int64_t ceil_positive_div(std::int64_t numerator, std::int64_t denominator,
                                      const std::string &label) {
    if (numerator < 0 || denominator <= 0) throw std::runtime_error("invalid signed-round division bound: " + label);
    return checked_bound((static_cast<__int128>(numerator) + denominator - 1) / denominator, label);
}

inline std::int64_t positive_config_value(std::uint64_t value, const std::string &label) {
    if (value == 0U || value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::runtime_error("signed-round config value is outside positive int64: " + label);
    return static_cast<std::int64_t>(value);
}

inline void finish_widths(Widths &widths, const std::string &operation_id) {
    widths.a_bits = unsigned_width(widths.a_max, operation_id + "/a");
    widths.q_bits = unsigned_width(widths.q_max, operation_id + "/q");
    widths.r_bits = unsigned_width(widths.r_max, operation_id + "/r");
    widths.delta_bits = unsigned_width(widths.delta_max, operation_id + "/delta");
    widths.divisor_bits = unsigned_width(widths.divisor_max, operation_id + "/divisor");
}

inline Widths derive(const Config &config, const IntegerBoundLedger &ledger,
                     const std::string &operation_id) {
    if (!ledger.authenticated || ledger.schema != "zkDL_gat_integer_bound_ledger_v1")
        throw std::runtime_error("signed-round widths require the authenticated integer-bound ledger");

    Widths widths;
    if (operation_id == "final_average") {
        const std::int64_t head_count = positive_config_value(config.architecture.output_heads, operation_id + "/heads");
        widths.denominator_scale = 1;
        std::int64_t final_sum_bound = 0;
        std::int64_t final_aux_bound = 0;
        for (std::uint64_t head = 0U; head < config.architecture.output_heads; ++head) {
            const auto &family = family_for(ledger, config.architecture.hidden_dims.size(), head, true, operation_id);
            final_sum_bound = std::max(final_sum_bound, family.max_dividend_abs);
            final_aux_bound = std::max(final_aux_bound, family.max_aux_abs);
        }
        widths.a_max = final_sum_bound;
        widths.divisor_max = head_count;
        widths.q_max = ceil_positive_div(widths.a_max, widths.divisor_max, operation_id + "/q");
        widths.r_max = checked_bound(static_cast<__int128>(2) * widths.divisor_max - 1, operation_id + "/r");
        widths.delta_max = widths.r_max;
        if (std::max({widths.a_max, widths.q_max, widths.r_max, widths.divisor_max}) > final_aux_bound)
            throw std::runtime_error("final-average signed-round range exceeds its authenticated family ledger");
        finish_widths(widths, operation_id);
        return widths;
    }

    const std::string hidden_prefix = "hidden/layer";
    const std::string output_prefix = "output/head";
    if (operation_id.compare(0U, hidden_prefix.size(), hidden_prefix) == 0) {
        const std::size_t layer_begin = hidden_prefix.size();
        const std::size_t layer_end = operation_id.find("/head", layer_begin);
        const std::size_t head_begin = layer_end == std::string::npos ? layer_end : layer_end + 5U;
        const std::size_t suffix = operation_id.find('/', head_begin);
        if (layer_end == std::string::npos || suffix == std::string::npos || operation_id.substr(suffix) != "/signed_round_div")
            throw std::runtime_error("noncanonical hidden signed-round operation id: " + operation_id);
        const std::uint64_t layer = decimal_component(operation_id, layer_begin, layer_end, operation_id + "/layer");
        const std::uint64_t head = decimal_component(operation_id, head_begin, suffix, operation_id + "/head");
        if (layer >= config.quantization.hidden_h_prime.size() || head >= config.quantization.hidden_h_prime.at(layer).size())
            throw std::runtime_error("hidden signed-round scale is outside the quantization chain: " + operation_id);
        const std::uint64_t scale = config.quantization.hidden_h_prime.at(layer).at(head);
        if (scale == 0U || scale > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::runtime_error("hidden signed-round denominator scale is invalid: " + operation_id);
        const auto &family = family_for(ledger, layer, head, false, operation_id);
        widths.binds_raw_denominator = true;
        widths.denominator_scale = static_cast<std::int64_t>(scale);
        widths.a_max = family.max_dividend_abs;
        widths.divisor_max = family.max_divisor;
        widths.q_max = ceil_positive_div(widths.a_max, widths.denominator_scale, operation_id + "/q");
        widths.r_max = checked_bound(static_cast<__int128>(2) * widths.divisor_max - 1, operation_id + "/r");
        widths.delta_max = widths.r_max;
        if (std::max({widths.a_max, widths.q_max, widths.r_max, widths.divisor_max}) > family.max_aux_abs)
            throw std::runtime_error("hidden signed-round range exceeds its authenticated family ledger: " + operation_id);
        finish_widths(widths, operation_id);
        return widths;
    }

    if (operation_id.compare(0U, output_prefix.size(), output_prefix) == 0) {
        const std::size_t head_begin = output_prefix.size();
        const std::size_t suffix = operation_id.find('/', head_begin);
        if (suffix == std::string::npos) throw std::runtime_error("noncanonical output signed-round operation id: " + operation_id);
        const std::uint64_t head = decimal_component(operation_id, head_begin, suffix, operation_id + "/head");
        const auto &family = family_for(ledger, config.architecture.hidden_dims.size(), head, true, operation_id);
        if (operation_id.substr(suffix) == "/signed_round_div") {
            if (head >= config.quantization.output_y_prime.size() || config.quantization.output_y_prime.at(head) == 0U ||
                config.quantization.output_y_prime.at(head) > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::runtime_error("output signed-round denominator scale is invalid: " + operation_id);
            widths.binds_raw_denominator = true;
            widths.denominator_scale = static_cast<std::int64_t>(config.quantization.output_y_prime.at(head));
            widths.a_max = family.max_dividend_abs;
            widths.divisor_max = family.max_divisor;
            widths.q_max = ceil_positive_div(widths.a_max, widths.denominator_scale, operation_id + "/q");
        } else if (operation_id.substr(suffix) == "/output_scale_alignment") {
            widths.denominator_scale = 1;
            // The authenticated output-family dividend bound includes both the
            // prior rounded quotient and its output_head scaling used here.
            widths.a_max = family.max_dividend_abs;
            widths.divisor_max = positive_config_value(config.quantization.output_pre, operation_id + "/output_pre");
            widths.q_max = ceil_positive_div(widths.a_max, widths.divisor_max, operation_id + "/q");
        } else {
            throw std::runtime_error("unsupported output signed-round operation: " + operation_id);
        }
        widths.r_max = checked_bound(static_cast<__int128>(2) * widths.divisor_max - 1, operation_id + "/r");
        widths.delta_max = widths.r_max;
        if (std::max({widths.a_max, widths.q_max, widths.r_max, widths.divisor_max}) > family.max_aux_abs)
            throw std::runtime_error("output signed-round range exceeds its authenticated family ledger: " + operation_id);
        finish_widths(widths, operation_id);
        return widths;
    }

    throw std::runtime_error("unsupported signed-round operation id: " + operation_id);
}

} // namespace zkdl_gat::signed_round_bounds
