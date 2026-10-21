#pragma once

#include "gat_types.hpp"

#include <array>
#include <string>
#include <vector>

namespace zkgpt::gat::detail {

ForwardResult compute_formal_forward_unvalidated(const Graph& graph,
                                                 const Tensor2D& features,
                                                 const Model& model,
                                                 const QuantConfig& quant);
bool validate_formal_forward_value_admission(const Graph& graph, const Model& model, const QuantConfig& quant,
                                             const ForwardResult& forward,
                                             const std::vector<std::array<unsigned, 5>>& division_profiles,
                                             std::string* reason);

}  // namespace zkgpt::gat::detail
