#pragma once

#include "gat_types.hpp"

namespace zkgpt::gat::detail {

ForwardResult compute_formal_forward_unvalidated(const Graph& graph,
                                                 const Tensor2D& features,
                                                 const Model& model,
                                                 const QuantConfig& quant);

}  // namespace zkgpt::gat::detail
