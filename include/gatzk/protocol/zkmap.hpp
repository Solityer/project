#pragma once

#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

struct ZkMapAccumulatorTrace {
    std::vector<algebra::FieldElement> a_values;
    std::vector<algebra::FieldElement> b_values;
    std::vector<algebra::FieldElement> accumulator;
    algebra::FieldElement mu = algebra::FieldElement::zero();
};

ZkMapAccumulatorTrace build_zkmap_trace(
    const std::vector<algebra::FieldElement>& a_values,
    const std::vector<algebra::FieldElement>& b_values,
    std::size_t domain_size);

}  // namespace gatzk::protocol
