#pragma once

#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/packed_field.hpp"

namespace gatzk::algebra
{
    // 代数后端类型：CPU计算或GPU计算
    enum class AlgebraBackend
    {
        Cpu,
        Cuda,
    };

     // 返回当前配置的代数后端
    AlgebraBackend configured_algebra_backend();
    std::string configured_algebra_backend_name();
    bool cuda_backend_build_enabled();
    bool cuda_backend_available();

    // 计算两个域元素向量的点积：sum(lhs[i] * rhs[i])
    FieldElement dot_product(
        const std::vector<FieldElement>& lhs,
        const std::vector<FieldElement>& rhs);

    // 计算左向量（域元素）与右向量（原生 mcl::Fr 权重）的点积
    FieldElement dot_product_native_weights(
        const std::vector<FieldElement>& lhs,
        const std::vector<mcl::Fr>& rhs);
    
    // 计算左向量与右向量的点积，其中右向量已打包为设备端就绪格式（用于 GPU 加速）
    FieldElement dot_product_packed_native_weights(
        const std::vector<FieldElement>& lhs,
        const std::vector<mcl::Fr>& rhs,
        const PackedFieldBuffer& packed_rhs);
}