#include "gat_proof.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace zkgpt {
namespace gat {
namespace {

std::size_t oracle_next_power_of_two(std::size_t value) {
    std::size_t result = 1;
    while (result < std::max<std::size_t>(1, value)) {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("segmented input oracle domain overflow");
        result <<= 1;
    }
    return result;
}

std::size_t oracle_log2(std::size_t value) {
    if (value == 0 || (value & (value - 1)) != 0)
        throw std::invalid_argument("segmented input oracle requires a power-of-two domain");
    std::size_t bits = 0;
    while (value > 1) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

bool valid_layout(const InputLayout &layout, std::string *reason) {
    const auto fail = [reason](const char *message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (layout.n_in == 0 || (layout.n_in & (layout.n_in - 1)) != 0)
        return fail("segmented input oracle layout domain is not a power of two");
    if (layout.selector != "00:X_pub,01:W,10:Aux,11:L_qry" ||
        layout.segments.size() != 4)
        return fail("segmented input oracle selector/layout is not canonical");
    static const char *names[] = {"X_pub", "W", "Aux", "L_qry"};
    std::size_t maximum = 1;
    for (std::size_t index = 0; index < layout.segments.size(); ++index) {
        const InputSegment &segment = layout.segments[index];
        if (segment.name != names[index] ||
            segment.padded_length != layout.n_in ||
            segment.logical_length > layout.n_in)
            return fail("segmented input oracle segment shape is not canonical");
        if (index > (std::numeric_limits<std::size_t>::max() / layout.n_in) ||
            segment.offset != index * layout.n_in)
            return fail("segmented input oracle segment offset is not canonical");
        maximum = std::max(maximum, segment.logical_length);
    }
    if (layout.n_raw != maximum || oracle_next_power_of_two(layout.n_raw) != layout.n_in)
        return fail("segmented input oracle raw/padded sizes are inconsistent");
    return true;
}

} // namespace

SegmentedInputOracle::SegmentedInputOracle(
    const InputLayout &layout, std::array<std::vector<Fr>, 4> segments)
    : layout_(layout), segments_(std::move(segments)) {
    std::string reason;
    if (!valid(&reason))
        throw std::invalid_argument("invalid segmented input oracle: " + reason);
}

SegmentedInputOracle SegmentedInputOracle::from_int_segments(
    const InputLayout &layout,
    const std::array<std::vector<Int>, 4> &segments) {
    std::array<std::vector<Fr>, 4> converted;
    for (std::size_t segment = 0; segment < converted.size(); ++segment) {
        converted[segment].reserve(segments[segment].size());
        for (Int value : segments[segment]) converted[segment].emplace_back(value);
    }
    return SegmentedInputOracle(layout, std::move(converted));
}

bool SegmentedInputOracle::valid(std::string *reason) const {
    if (!valid_layout(layout_, reason)) return false;
    for (std::size_t index = 0; index < segments_.size(); ++index) {
        if (segments_[index].size() != layout_.segments[index].logical_length) {
            if (reason != nullptr) *reason = "segmented input oracle logical length mismatch";
            return false;
        }
    }
    return true;
}

Fr SegmentedInputOracle::value_at(std::size_t global_coordinate) const {
    if (!valid_layout(layout_, nullptr))
        throw std::invalid_argument("segmented input oracle has an invalid layout");
    const std::size_t total = layout_.n_in * layout_.segments.size();
    if (global_coordinate >= total)
        throw std::out_of_range("segmented input oracle global coordinate is out of range");
    const std::size_t segment = global_coordinate / layout_.n_in;
    return segment_value(segment, global_coordinate % layout_.n_in);
}

Fr SegmentedInputOracle::segment_value(std::size_t segment,
                                       std::size_t local_coordinate) const {
    if (segment >= segments_.size())
        throw std::out_of_range("segmented input oracle segment is out of range");
    if (local_coordinate >= layout_.n_in)
        throw std::out_of_range("segmented input oracle local coordinate is out of range");
    if (local_coordinate >= segments_[segment].size()) return Fr(0);
    return segments_[segment][local_coordinate];
}

void SegmentedInputOracle::stream(std::size_t segment, std::size_t begin,
                                  std::size_t count,
                                  const ChunkCallback &callback) const {
    if (!callback) throw std::invalid_argument("segmented input oracle callback is empty");
    if (segment >= segments_.size() || begin > layout_.n_in ||
        count > layout_.n_in - begin)
        throw std::out_of_range("segmented input oracle stream range is invalid");
    for (std::size_t index = 0; index < count; ++index)
        callback(begin + index, segment_value(segment, begin + index));
}

std::vector<Fr> SegmentedInputOracle::chunk(std::size_t segment,
                                            std::size_t begin,
                                            std::size_t count) const {
    std::vector<Fr> result;
    result.reserve(count);
    stream(segment, begin, count,
           [&result](std::size_t, const Fr &value) { result.push_back(value); });
    return result;
}

Fr SegmentedInputOracle::evaluate_mle(std::size_t segment,
                                      const std::vector<Fr> &point) const {
    if (segment >= segments_.size())
        throw std::out_of_range("segmented input oracle MLE segment is invalid");
    if (point.size() != oracle_log2(layout_.n_in))
        throw std::invalid_argument("segmented input oracle MLE point has the wrong dimension");
    Fr result = Fr(0);
    // Evaluate directly from logical values.  Coordinates beyond the logical
    // prefix are canonical zero padding and therefore need no iteration.
    for (std::size_t index = 0; index < segments_[segment].size(); ++index) {
        Fr weight = Fr(1);
        std::size_t coordinate = index;
        for (const Fr &challenge : point) {
            weight *= (coordinate & 1U) != 0 ? challenge : (Fr(1) - challenge);
            coordinate >>= 1;
        }
        result += segments_[segment][index] * weight;
    }
    return result;
}

Fr SegmentedInputOracle::evaluate_full_mle(const std::vector<Fr> &point) const {
    const std::size_t local_bits = oracle_log2(layout_.n_in);
    if (point.size() != local_bits + 2)
        throw std::invalid_argument("segmented input oracle full MLE point has the wrong dimension");
    const std::vector<Fr> local_point(point.begin(), point.begin() + local_bits);
    const Fr s2 = point[local_bits];
    const Fr s1 = point[local_bits + 1];
    const Fr c_x = (Fr(1) - s1) * (Fr(1) - s2);
    const Fr c_w = (Fr(1) - s1) * s2;
    const Fr c_aux = s1 * (Fr(1) - s2);
    const Fr c_l = s1 * s2;
    return evaluate_mle(0, local_point) * c_x +
           evaluate_mle(1, local_point) * c_w +
           evaluate_mle(2, local_point) * c_aux +
           evaluate_mle(3, local_point) * c_l;
}

std::vector<Fr> SegmentedInputOracle::fold(
    std::size_t segment, const std::vector<Fr> &prefix) const {
    if (segment >= segments_.size())
        throw std::out_of_range("segmented input oracle fold segment is invalid");
    const std::size_t bits = oracle_log2(layout_.n_in);
    if (prefix.size() > bits)
        throw std::invalid_argument("segmented input oracle fold prefix is too long");
    const std::size_t remaining_bits = bits - prefix.size();
    const std::size_t remaining = std::size_t(1) << remaining_bits;
    // Folding a complete large domain would itself defeat the oracle.  The
    // native caller must request bounded prefixes/chunks for such domains;
    // tiny synthetic relations remain fully supported.
    constexpr std::size_t kMaxFoldEntries = std::size_t(1) << 24;
    if (remaining > kMaxFoldEntries)
        throw std::length_error("segmented input oracle fold exceeds bounded chunk domain");
    std::vector<Fr> result(remaining, Fr(0));
    for (std::size_t index = 0; index < segments_[segment].size(); ++index) {
        Fr weight = Fr(1);
        std::size_t coordinate = index;
        for (const Fr &challenge : prefix) {
            weight *= (coordinate & 1U) != 0 ? challenge : (Fr(1) - challenge);
            coordinate >>= 1;
        }
        result[coordinate] += segments_[segment][index] * weight;
    }
    return result;
}

Fr SegmentedInputOracle::weighted_sum(
    std::size_t segment,
    const std::function<Fr(std::size_t)> &weight) const {
    if (segment >= segments_.size())
        throw std::out_of_range("segmented input oracle weighted-sum segment is invalid");
    if (!weight) throw std::invalid_argument("segmented input oracle weight callback is empty");
    Fr result = Fr(0);
    for (std::size_t index = 0; index < segments_[segment].size(); ++index)
        result += segments_[segment][index] * weight(index);
    return result;
}

} // namespace gat
} // namespace zkgpt
