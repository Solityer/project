#pragma once

#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

std::vector<algebra::FieldElement> build_group_prefix_state(
    const std::vector<algebra::FieldElement>& values,
    const std::vector<algebra::FieldElement>& q_new);

std::vector<algebra::FieldElement> build_max_counter_state(
    const std::vector<algebra::FieldElement>& s_max,
    const std::vector<algebra::FieldElement>& q_new);

}  // namespace gatzk::protocol
