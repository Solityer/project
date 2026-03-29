#pragma once

#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/packed_field.hpp"

namespace gatzk::algebra {

enum class AlgebraBackend {
    Cpu,
    Cuda,
};

AlgebraBackend configured_algebra_backend();
std::string configured_algebra_backend_name();
bool cuda_backend_available();

FieldElement dot_product(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs);
FieldElement dot_product_native_weights(
    const std::vector<FieldElement>& lhs,
    const std::vector<mcl::Fr>& rhs);
FieldElement dot_product_packed_native_weights(
    const std::vector<FieldElement>& lhs,
    const std::vector<mcl::Fr>& rhs,
    const PackedFieldBuffer& packed_rhs);

}  // namespace gatzk::algebra
