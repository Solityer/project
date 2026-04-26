#include "gat_ezkl/bounds.hpp"

#include <limits>
#include <sstream>

namespace gat_ezkl {

ScoreBounds infer_score_bounds_from_quant_cfg(
    const QuantConfig& quant,
    std::size_t range_bits) {
    ScoreBounds out;
    out.range_bits = range_bits;
    if (range_bits == 0 || range_bits >= 62) {
        out.blocker = "ezkl_segment_softmax_sentinel_bound_unproven";
        out.explanation = "range_bits is missing or too large for a signed int64 bound proof";
        return out;
    }

    const auto magnitude = std::int64_t{1} << (range_bits - 1);
    out.inferred = true;
    out.score_lower_bound = -magnitude;
    out.score_upper_bound = magnitude - 1;
    std::ostringstream explanation;
    explanation
        << "inferred signed fixed-point score range from quant_cfg_id="
        << quant.quant_cfg_id
        << " and range_bits=" << range_bits
        << " as [" << out.score_lower_bound
        << "," << out.score_upper_bound << "]";
    out.explanation = explanation.str();
    return out;
}

ScoreBounds choose_invalid_score_sentinel(const ScoreBounds& bounds) {
    ScoreBounds out = bounds;
    if (!out.inferred) {
        if (out.blocker.empty()) {
            out.blocker = "ezkl_segment_softmax_sentinel_bound_unproven";
        }
        return out;
    }
    if (out.score_lower_bound == std::numeric_limits<std::int64_t>::min()) {
        out.blocker = "ezkl_segment_softmax_sentinel_bound_unproven";
        out.explanation = "score lower bound cannot be decremented without int64 overflow";
        return out;
    }
    out.invalid_score_sentinel = out.score_lower_bound - 1;
    out.sentinel_bound_verified = out.invalid_score_sentinel < out.score_lower_bound;
    if (!out.sentinel_bound_verified) {
        out.blocker = "ezkl_segment_softmax_sentinel_bound_unproven";
    }
    return out;
}

bool verify_sentinel_bound(const ScoreBounds& bounds, std::string* reason) {
    if (!bounds.inferred) {
        if (reason) *reason = bounds.explanation;
        return false;
    }
    if (!bounds.sentinel_bound_verified
        || !(bounds.invalid_score_sentinel < bounds.score_lower_bound)) {
        if (reason) *reason = "invalid score sentinel is not strictly below the legal score lower bound";
        return false;
    }
    if (reason) {
        std::ostringstream out;
        out << bounds.explanation
            << "; sentinel=" << bounds.invalid_score_sentinel
            << " is strictly below lower_bound=" << bounds.score_lower_bound;
        *reason = out.str();
    }
    return true;
}

}  // namespace gat_ezkl
