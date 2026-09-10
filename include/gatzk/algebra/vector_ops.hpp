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

    void release_cuda_ntt_workspace();
    void release_cuda_scaled_accumulate_workspace();

    class CudaScaledAccumulateSession
    {
    public:
        virtual ~CudaScaledAccumulateSession() = default;

        virtual void accumulate_range(
            const mcl::Fr *source,
            const mcl::Fr &scale,
            std::size_t target_offset,
            std::size_t element_count) = 0;

        virtual void finish() = 0;
    };

    std::unique_ptr<CudaScaledAccumulateSession>
    create_cuda_scaled_accumulate_session(
        mcl::Fr *target,
        std::size_t element_count);

    // 返回当前构建与运行环境解析出的代数后端信息。
    AlgebraBackend configured_algebra_backend();
    std::string configured_algebra_backend_name();
    bool cuda_backend_build_enabled();
    bool cuda_backend_available();
    // True only when the active CUDA device has UVA and may coherently read
    // pageable host buffers without explicit registration.
    bool cuda_backend_unified_addressing_available();
    std::string cuda_device_summary();
    std::size_t cuda_kernel_invocation_count();
    void record_cuda_kernel_invocations(std::size_t count);

    struct CudaExecutionStats
    {
        std::uint64_t h2d_bytes = 0;
        std::uint64_t d2h_bytes = 0;
        std::uint64_t h2d_operations = 0;
        std::uint64_t d2h_operations = 0;
        std::uint64_t stream_synchronizations = 0;
        std::uint64_t ntt_calls = 0;
        std::uint64_t intt_calls = 0;
        std::uint64_t quotient_calls = 0;
        std::uint64_t t_edge_calls = 0;
        std::uint64_t opening_fold_calls = 0;
        std::uint64_t msm_single_calls = 0;
        std::uint64_t msm_batch_calls = 0;
        std::uint64_t resident_hits = 0;
        std::uint64_t resident_misses = 0;
    };

    CudaExecutionStats cuda_execution_stats_snapshot();

    void record_cuda_h2d(std::size_t bytes);
    void record_cuda_d2h(std::size_t bytes);
    void record_cuda_stream_sync();
    void record_cuda_ntt_call();
    void record_cuda_intt_call();
    void record_cuda_quotient_call();
    void record_cuda_t_edge_call();
    void record_cuda_opening_fold_call();
    void record_cuda_msm_single_call();
    void record_cuda_msm_batch_call();
    void record_cuda_resident_hit();
    void record_cuda_resident_miss();

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

    // Production paired-NTT routing.  CPUOnly or a false predicate means the
    // caller owns the canonical CPU path.  After a large CUDA route has been
    // selected by policy and capability, an exact CUDA execution failure is
    // fail-closed; callers must not silently retry on CPU.
    bool should_use_cuda_intt(
        std::size_t size);

    bool should_use_cuda_ntt(
        std::size_t size);

    bool should_use_cuda_ntt_pair(
        std::size_t size);

    // Bounded multi-stage quotient-coset transform route.  This is distinct
    // from the generic paired-NTT route: it decomposes a target transform into
    // ICICLE-supported 2^24 subtransforms plus exact radix-2 combine kernels.
    // The predicate is intentionally size-aware so callers retain the CPU
    // path only when the CUDA route was not selected.  Once selected, an
    // admission or execution failure must be surfaced by the caller.
    bool should_use_cuda_coset_transform(
        std::size_t target_size);

    // Exact Montgomery-limb update `target[i] += source[i] * scale`.  Returns
    // true only when the active route executed the update on CUDA; false means
    // that the caller must retain its canonical CPU path.  Once CUDA is
    // selected, launch/synchronization errors are reported fail-closed.
    bool try_cuda_scaled_accumulate_native(
        mcl::Fr *target,
        const mcl::Fr *source,
        const mcl::Fr &scale,
        std::size_t element_count);

    // One-pass two-target variant used when two independent opening point
    // groups fold the same immutable semantic source.  The two scales and
    // targets remain independent; only the source traversal is shared.
    bool try_cuda_scaled_accumulate_pair_native(
        mcl::Fr *first_target,
        mcl::Fr *second_target,
        const mcl::Fr *source,
        const mcl::Fr &first_scale,
        const mcl::Fr &second_scale,
        std::size_t element_count);

    // 在 GPU 上构造重心权重，供 domain opening 与大列求值热点复用。
    bool barycentric_weights_gpu(
        const uint64_t omega_limbs[4],
        const uint64_t x_limbs[4],
        const uint64_t zero_eval_limbs[4],
        const uint64_t inv_n_limbs[4],
        uint64_t *output_weight_limbs,
        std::size_t domain_size);

    // 使用共享权重对打包列做 GPU 加权求值，是 partial CUDA PackedEvaluationBackend 的加速入口。
    bool weighted_poly_eval_gpu(
        const uint64_t *weight_limbs,
        const uint64_t *interleaved_value_limbs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t row_count);

    // 列式布局版本，适合统一内存或零拷贝场景下的大列评估。
    bool weighted_poly_eval_gpu_columnar(
        const uint64_t *weight_limbs,
        const uint64_t *const *column_ptrs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t row_count,
        const std::size_t *column_lengths = nullptr,
        std::size_t weight_rotation = 0);

    // 多列共享一次权重上传，减少 prove / verify 热路径中的重复设备传输。
    bool weighted_poly_eval_gpu_multi(
        const uint64_t *weight_limbs,
        const uint64_t *const *column_value_ptrs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t poly_count,
        const std::size_t *column_lengths = nullptr,
        std::size_t weight_rotation = 0);

    // 在统一地址空间设备上批量执行多个旋转点的同列求值，输出按
    // `[rotation][polynomial]` 排列。
    bool weighted_poly_eval_gpu_multi_rotations(
        const uint64_t *weight_limbs,
        const uint64_t *const *column_value_ptrs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t poly_count,
        const std::size_t *rotations,
        std::size_t rotation_count,
        const std::size_t *column_lengths = nullptr);

    // Cached-subset route: upload one interleaved [domain][polynomial] buffer,
    // transpose it on the device, then feed the resulting device columns to
    // the same batched multi-rotation evaluator.  This is an execution-only
    // path; the output order and field operations are identical to the
    // canonical columnar API above.
    bool weighted_poly_eval_gpu_interleaved_multi_rotations(
        const uint64_t *weight_limbs,
        const uint64_t *interleaved_value_limbs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t poly_count,
        const std::size_t *rotations,
        std::size_t rotation_count);

    // Admission for the interleaved cached-subset route.  `additional_host_bytes`
    // is the cache materialization that will happen before the CUDA call.  The
    // check accounts for the complete transient device peak and for host/device
    // availability before callers allocate that host cache.
    bool cuda_interleaved_multi_rotation_admissible(
        std::size_t domain_size,
        std::size_t poly_count,
        std::size_t rotation_count,
        std::size_t additional_host_bytes);

    // CPU 侧域元素点积，是小规模 fallback 和校验路径的通用原语。
    FieldElement dot_product(
        const std::vector<FieldElement> &lhs,
        const std::vector<FieldElement> &rhs);

    // 共享 NTT plan 缓存，供 quotient materialization 与可复用 prover-side material 转换使用。
    std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan(std::size_t size);

    void geometric_powers_native(
        mcl::Fr *output,
        std::size_t size,
        const FieldElement &base,
        std::size_t max_worker_count = 0);

    // Multiply an existing native field vector by
    // [1, base, base^2, ...] without materializing that power vector.
    void scale_by_geometric_powers_native(
        mcl::Fr *values,
        std::size_t size,
        const FieldElement &base,
        std::size_t max_worker_count = 0);

    // Copy input to output while applying
    // [1, base, base^2, ...], avoiding a persistent power table.
    void copy_scaled_by_geometric_powers_native(
        const mcl::Fr *input,
        mcl::Fr *output,
        std::size_t size,
        const FieldElement &base,
        std::size_t max_worker_count = 0);

    // Prepare the two half-size NTT lanes used to evaluate a coefficient
    // polynomial on the even and odd points of a doubled quotient coset.
    void split_coset_coefficients_native(
        const mcl::Fr *coefficients,
        const mcl::Fr *shift_powers,
        std::size_t coefficient_count,
        const FieldElement &odd_root,
        mcl::Fr *even_output,
        mcl::Fr *odd_output,
        std::size_t max_worker_count = 0);

    // Restore canonical [even_0, odd_0, ...] point order after the paired
    // quotient-coset transforms.
    void interleave_native_lanes(
        const mcl::Fr *even_values,
        const mcl::Fr *odd_values,
        std::size_t lane_size,
        mcl::Fr *output,
        std::size_t max_worker_count = 0);

    // 大型逆变换按偶/奇频点拆成两个独立半长 INTT；结果与标准
    // `ntt_plan(size)->intt` 逐元素一致，小规模保持标准实现。
    void intt_in_place(
        mcl::Fr *values,
        std::size_t size,
        std::size_t max_worker_count = 0);

    void ntt_in_place(
        mcl::Fr *values,
        std::size_t size,
        std::size_t max_worker_count = 0);

    // Execute two independent same-size forward NTTs concurrently at formal
    // transform sizes.  Small inputs retain the canonical serial plan calls.
    void ntt_pair_in_place(
        mcl::Fr *first,
        mcl::Fr *second,
        std::size_t size,
        std::size_t max_worker_count = 0);

    // Explicit CUDA quotient-transform probes.
    //
    // These functions bypass the production size/topology policy and directly
    // exercise the CUDA implementation for exact-equivalence and stage
    // benchmarking.  Production intt_in_place / ntt_pair_in_place apply their own
    // execution-routing policy before calling these probes.
    //
    // false means that the CUDA route is unavailable or cannot be admitted under
    // the current resource envelope.  Once CUDA execution starts, CUDA/ICICLE
    // failures are reported fail-closed by exception.
    bool try_cuda_intt_in_place(
        mcl::Fr *values,
        std::size_t size);

    bool try_cuda_ntt_in_place(
        mcl::Fr *values,
        std::size_t size);

    bool try_cuda_ntt_pair_in_place(
        mcl::Fr *first,
        mcl::Fr *second,
        std::size_t size);

    // 直接消费原生权重，避免 barycentric 热路径中反复包装 `FieldElement`。
    FieldElement dot_product_native_weights(
        const std::vector<FieldElement> &lhs,
        const std::vector<mcl::Fr> &rhs);
}
