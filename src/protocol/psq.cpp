#include "gatzk/protocol/psq.hpp"

#include <stdexcept>

namespace gatzk::protocol {

std::vector<algebra::FieldElement> build_group_prefix_state(
    const std::vector<algebra::FieldElement>& values,
    const std::vector<algebra::FieldElement>& q_new) {
    if (values.size() != q_new.size()) {
        throw std::runtime_error("prefix state input size mismatch");
    }
    if (values.empty()) {
        return {};
    }
    std::vector<algebra::FieldElement> out(values.size(), algebra::FieldElement::zero());
    out[0] = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        out[i] = q_new[i] * values[i] + (algebra::FieldElement::one() - q_new[i]) * (out[i - 1] + values[i]);
    }
    return out;
}

std::vector<algebra::FieldElement> build_max_counter_state(
    const std::vector<algebra::FieldElement>& s_max,
    const std::vector<algebra::FieldElement>& q_new) {
    if (s_max.size() != q_new.size()) {
        throw std::runtime_error("max counter input size mismatch");
    }
    if (s_max.empty()) {
        return {};
    }
    std::vector<algebra::FieldElement> out(s_max.size(), algebra::FieldElement::zero());
    out[0] = s_max[0];
    for (std::size_t i = 1; i < s_max.size(); ++i) {
        out[i] = q_new[i] * s_max[i] + (algebra::FieldElement::one() - q_new[i]) * (out[i - 1] + s_max[i]);
    }
    return out;
}
}