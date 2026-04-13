#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

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
    bool cuda_backend_unified_addressing_available();

    // GPU field hotspot acceleration (BLS12-381 Fr arithmetic on GPU)
    bool cuda_field_hotspots_build_enabled();
    bool cuda_field_hotspots_available();

    // Centralized, size-aware routing policy for full_gpu. These helpers only
    // choose an implementation route; they do not change proof semantics.
    bool should_use_cuda_barycentric_weights(
        std::size_t domain_size,
        bool points_precomputed);
    bool should_use_cuda_weighted_evaluation(
        std::size_t domain_size,
        std::size_t vector_count,
        std::size_t point_count = 1);
    bool should_use_cuda_lookup_histogram(
        std::size_t index_count,
        std::size_t bucket_count);

    // GPU barycentric weight computation
    // omega/x/zero_eval/inv_n: 4 × uint64 limbs in Montgomery form (matching mcl::Fr internal)
    // output_weight_limbs: domain_size × 4 uint64 limbs
    bool barycentric_weights_gpu(
        const uint64_t omega_limbs[4],
        const uint64_t x_limbs[4],
        const uint64_t zero_eval_limbs[4],
        const uint64_t inv_n_limbs[4],
        uint64_t* output_weight_limbs,
        std::size_t domain_size);

    // GPU weighted polynomial evaluation (interleaved layout)
    // weights: domain_size × 4 uint64 limbs  
    // interleaved_values: domain_size × row_count × 4 uint64 limbs
    // output: row_count × 4 uint64 limbs
    bool weighted_poly_eval_gpu(
        const uint64_t* weight_limbs,
        const uint64_t* interleaved_value_limbs,
        uint64_t* output_limbs,
        std::size_t domain_size,
        std::size_t row_count);

    // GPU weighted polynomial evaluation (columnar / zero-copy for unified memory)
    // weights: host ptr, domain_size × 4 uint64 limbs
    // column_ptrs: array of row_count host pointers, each -> domain_size × 4 uint64 limbs
    // output: host ptr, row_count × 4 uint64 limbs
    bool weighted_poly_eval_gpu_columnar(
        const uint64_t* weight_limbs,
        const uint64_t* const* column_ptrs,
        uint64_t* output_limbs,
        std::size_t domain_size,
        std::size_t row_count);

    // GPU multi-polynomial evaluation with shared weights (weight upload once)
    // column_value_ptrs: array of poly_count host pointers, each -> domain_size × 4 uint64 limbs
    // output: poly_count × 4 uint64 limbs
    bool weighted_poly_eval_gpu_multi(
        const uint64_t* weight_limbs,
        const uint64_t* const* column_value_ptrs,
        uint64_t* output_limbs,
        std::size_t domain_size,
        std::size_t poly_count);

    // 计算两个域元素向量的点积：sum(lhs[i] * rhs[i])
    FieldElement dot_product(
        const std::vector<FieldElement>& lhs,
        const std::vector<FieldElement>& rhs);

    // 计算左向量（域元素）与右向量（原生 mcl::Fr 权重）的点积
    FieldElement dot_product_native_weights(
        const std::vector<FieldElement>& lhs,
        const std::vector<mcl::Fr>& rhs);
}
