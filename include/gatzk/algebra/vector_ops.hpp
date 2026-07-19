#pragma once
// 本文件声明 algebra 层的统一执行入口，包括 CPU / CUDA 路由、
// 热点判定和域元素向量运算，不改变 proof schema 与 transcript。

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include <mcl/ntt.hpp>

namespace gatzk::algebra
{
    // 代数实现后端。它只影响执行放置，不引入第二套协议语义。
    enum class AlgebraBackend
    {
        Cpu,
        Cuda,
    };

    // 返回当前构建与运行环境解析出的代数后端信息。
    AlgebraBackend configured_algebra_backend();
    std::string configured_algebra_backend_name();
    bool cuda_backend_build_enabled();
    bool cuda_backend_available();
    bool cuda_backend_unified_addressing_available();
    std::string cuda_device_summary();
    std::size_t cuda_kernel_invocation_count();
    void record_cuda_kernel_invocations(std::size_t count);

    // partial CUDA Fr 热点能力探针；KZG、pairing 与 quotient 仍走 CPU。
    bool cuda_field_hotspots_build_enabled();
    bool cuda_field_hotspots_available();

    // partial CUDA 的统一 size-aware 路由策略，只决定实现路线，不改变 formal 结果。
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

    // 在 GPU 上构造重心权重，供 domain opening 与大列求值热点复用。
    bool barycentric_weights_gpu(
        const uint64_t omega_limbs[4],
        const uint64_t x_limbs[4],
        const uint64_t zero_eval_limbs[4],
        const uint64_t inv_n_limbs[4],
        uint64_t* output_weight_limbs,
        std::size_t domain_size);

    // 使用共享权重对打包列做 GPU 加权求值，是 partial CUDA PackedEvaluationBackend 的加速入口。
    bool weighted_poly_eval_gpu(
        const uint64_t* weight_limbs,
        const uint64_t* interleaved_value_limbs,
        uint64_t* output_limbs,
        std::size_t domain_size,
        std::size_t row_count);

    // 列式布局版本，适合统一内存或零拷贝场景下的大列评估。
    bool weighted_poly_eval_gpu_columnar(
        const uint64_t* weight_limbs,
        const uint64_t* const* column_ptrs,
        uint64_t* output_limbs,
        std::size_t domain_size,
        std::size_t row_count);

    // 多列共享一次权重上传，减少 prove / verify 热路径中的重复设备传输。
    bool weighted_poly_eval_gpu_multi(
        const uint64_t* weight_limbs,
        const uint64_t* const* column_value_ptrs,
        uint64_t* output_limbs,
        std::size_t domain_size,
        std::size_t poly_count);

    // CPU 侧域元素点积，是小规模 fallback 和校验路径的通用原语。
    FieldElement dot_product(
        const std::vector<FieldElement>& lhs,
        const std::vector<FieldElement>& rhs);

    // 共享 NTT plan 缓存，供 quotient materialization 与可复用 prover-side material 转换使用。
    std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan(std::size_t size);

    // 大型逆变换按偶/奇频点拆成两个独立半长 INTT；结果与标准
    // `ntt_plan(size)->intt` 逐元素一致，小规模保持标准实现。
    void intt_in_place(mcl::Fr* values, std::size_t size);

    // Execute two independent same-size forward NTTs concurrently at formal
    // transform sizes.  Small inputs retain the canonical serial plan calls.
    void ntt_pair_in_place(mcl::Fr* first, mcl::Fr* second, std::size_t size);

    // 直接消费原生权重，避免 barycentric 热路径中反复包装 `FieldElement`。
    FieldElement dot_product_native_weights(
        const std::vector<FieldElement>& lhs,
        const std::vector<mcl::Fr>& rhs);
}
