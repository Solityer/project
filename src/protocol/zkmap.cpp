#include "gatzk/protocol/zkmap.hpp"

#include <stdexcept>

namespace gatzk::protocol {

ZkMapAccumulatorTrace build_zkmap_trace(
    const std::vector<algebra::FieldElement>& a_values,
    const std::vector<algebra::FieldElement>& b_values,
    std::size_t domain_size) {
    if (a_values.size() != b_values.size()) {
        throw std::runtime_error("zkMaP input size mismatch");
    }

    ZkMapAccumulatorTrace trace;
    trace.a_values.assign(domain_size, algebra::FieldElement::zero());
    trace.b_values.assign(domain_size, algebra::FieldElement::zero());
    trace.accumulator.assign(domain_size, algebra::FieldElement::zero());
    for (std::size_t i = 0; i < a_values.size(); ++i) {
        trace.a_values[i] = a_values[i];
        trace.b_values[i] = b_values[i];
        trace.mu += a_values[i] * b_values[i];
    }
    for (std::size_t i = 0; i < a_values.size(); ++i) {
        trace.accumulator[i + 1] = trace.accumulator[i] + a_values[i] * b_values[i];
    }
    for (std::size_t i = a_values.size() + 1; i < domain_size; ++i) {
        trace.accumulator[i] = trace.accumulator[a_values.size()];
    }
    return trace;
}

}  // namespace gatzk::protocol
