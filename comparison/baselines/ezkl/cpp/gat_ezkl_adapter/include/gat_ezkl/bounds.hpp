#pragma once

#include <cstdint>
#include <string>

#include "gat_ezkl/quant.hpp"

namespace gat_ezkl {

struct ScoreBounds {
    bool inferred = false;
    std::size_t range_bits = 0;
    std::int64_t score_lower_bound = 0;
    std::int64_t score_upper_bound = 0;
    std::int64_t invalid_score_sentinel = 0;
    bool sentinel_bound_verified = false;
    std::string blocker;
    std::string explanation;
};

ScoreBounds infer_score_bounds_from_quant_cfg(
    const QuantConfig& quant,
    std::size_t range_bits);
ScoreBounds choose_invalid_score_sentinel(const ScoreBounds& bounds);
bool verify_sentinel_bound(const ScoreBounds& bounds, std::string* reason);

}  // namespace gat_ezkl
