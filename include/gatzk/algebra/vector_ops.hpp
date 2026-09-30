#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include <mcl/ntt.hpp>

namespace gatzk::algebra
{
    struct CudaResidentCoefficientBuffer;

    enum class AlgebraBackend
    {
        Cpu,
        Cuda,
    };

    void release_cuda_ntt_workspace();
    void release_cuda_ntt_execution_workspace();
    void release_cuda_scaled_accumulate_workspace();
    void release_cuda_scan_workspace();

    class CudaScaledAccumulateSession
    {
    public:
        virtual ~CudaScaledAccumulateSession() = default;

        virtual void accumulate_range(const mcl::Fr *source, const mcl::Fr &scale, std::size_t target_offset, std::size_t element_count,
                                      bool source_is_ephemeral = false) = 0;

        virtual void accumulate_device_range(const std::uint64_t *device_montgomery_source, int source_device, const mcl::Fr &scale, std::size_t target_offset,
                                             std::size_t element_count) = 0;

        virtual void finish() = 0;
    };

    std::unique_ptr<CudaScaledAccumulateSession> create_cuda_scaled_accumulate_session(mcl::Fr *target, std::size_t element_count, bool target_is_zero = false);

    class CudaScaledAccumulatePairSession
    {
    public:
        virtual ~CudaScaledAccumulatePairSession() = default;

        virtual void accumulate_range(const mcl::Fr *source, const mcl::Fr &first_scale, const mcl::Fr &second_scale, std::size_t target_offset,
                                      std::size_t element_count, bool source_is_ephemeral = false) = 0;

        virtual void finish() = 0;

        virtual bool finished() const noexcept = 0;
    };

    std::shared_ptr<CudaScaledAccumulatePairSession> create_cuda_scaled_accumulate_pair_session(mcl::Fr *first_target, mcl::Fr *second_target,
                                                                                                std::size_t element_count, bool first_target_is_zero = false,
                                                                                                bool second_target_is_zero = false);

    class CudaBatchInverseSession
    {
    public:
        virtual ~CudaBatchInverseSession() = default;

        virtual void *native_stream_handle() = 0;

        virtual void enqueue(const mcl::Fr *input, mcl::Fr *output, std::size_t element_count) = 0;

        virtual void wait() = 0;
    };

    std::unique_ptr<CudaBatchInverseSession> create_cuda_batch_inverse_session(std::size_t maximum_element_count);

    AlgebraBackend configured_algebra_backend();
    std::string configured_algebra_backend_name();
    bool cuda_backend_build_enabled();
    bool cuda_backend_available();
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
        std::uint64_t t_fh_calls = 0;
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
    void record_cuda_t_fh_call();
    void record_cuda_opening_fold_call();
    void record_cuda_msm_single_call();
    void record_cuda_msm_batch_call();
    void record_cuda_resident_hit();
    void record_cuda_resident_miss();

    bool cuda_field_hotspots_build_enabled();
    bool cuda_field_hotspots_available();

    bool should_use_cuda_barycentric_weights(std::size_t domain_size, bool points_precomputed);
    bool should_use_cuda_weighted_evaluation(std::size_t domain_size, std::size_t vector_count, std::size_t point_count = 1);
    bool should_use_cuda_lookup_histogram(std::size_t index_count, std::size_t bucket_count);

    bool should_use_cuda_intt(std::size_t size);

    bool should_use_cuda_ntt(std::size_t size);

    bool should_use_cuda_ntt_pair(std::size_t size);

    bool try_cuda_taylor_convolution(mcl::Fr *values, const mcl::Fr *kernel_ntt, std::size_t size);

    bool should_use_cuda_coset_transform(std::size_t target_size);

    bool try_cuda_scaled_accumulate_native(mcl::Fr *target, const mcl::Fr *source, const mcl::Fr &scale, std::size_t element_count);

    bool try_cuda_scaled_accumulate_pair_native(mcl::Fr *first_target, mcl::Fr *second_target, const mcl::Fr *source, const mcl::Fr &first_scale,
                                                const mcl::Fr &second_scale, std::size_t element_count);

    bool try_cuda_batch_inverse_native(const mcl::Fr *input, mcl::Fr *output, std::size_t element_count);

    bool try_cuda_opening_inverse_denominators_native(mcl::Fr *output, std::size_t domain_size, const FieldElement &omega, const FieldElement &point);

    bool try_cuda_opening_quotient_assembly_native(const mcl::Fr *folded_values, mcl::Fr *quotient_values, const mcl::Fr &scaled_claim_sum,
                                                   std::size_t element_count);

    bool try_cuda_opening_inverse_and_quotient_native(const mcl::Fr *folded_values, mcl::Fr *quotient_values, const mcl::Fr &scaled_claim_sum,
                                                      std::size_t domain_size, const FieldElement &omega, const FieldElement &point);

    bool try_cuda_opening_inverse_and_quotient_resident_native(const mcl::Fr *folded_values, const mcl::Fr &scaled_claim_sum, std::size_t domain_size,
                                                               const FieldElement &omega, const FieldElement &point,
                                                               std::shared_ptr<const CudaResidentCoefficientBuffer> *resident_output);

    bool try_cuda_coefficient_opening_quotient_resident_native(const mcl::Fr *coefficients, std::size_t coefficient_count, const mcl::Fr &point,
                                                               const mcl::Fr &claimed_value,
                                                               std::shared_ptr<const CudaResidentCoefficientBuffer> *resident_output);

    struct CudaLogupExecutionProfile
    {
        double inverse_ms = 0.0;
        double accumulator_ms = 0.0;
    };

    bool try_cuda_logup_inverse_and_accumulator_native(const mcl::Fr *table, const mcl::Fr *query, const mcl::Fr *q_table, const mcl::Fr *q_query,
                                                       const mcl::Fr &beta, const mcl::Fr *multiplicity, mcl::Fr *table_inverse, mcl::Fr *query_inverse,
                                                       mcl::Fr *accumulator, std::size_t active_count, std::size_t element_count,
                                                       CudaLogupExecutionProfile *profile = nullptr);

    bool try_cuda_logup_inverse_columns_native(const mcl::Fr *table, const mcl::Fr *query, const mcl::Fr *q_table, const mcl::Fr *q_query, const mcl::Fr &beta,
                                               mcl::Fr *table_inverse, mcl::Fr *query_inverse, std::size_t element_count);

    bool try_cuda_logup_accumulator_native(const mcl::Fr *table_inverse, const mcl::Fr *query_inverse, const mcl::Fr *multiplicity, const mcl::Fr *q_table,
                                           const mcl::Fr *q_query, mcl::Fr *accumulator, std::size_t active_count, std::size_t element_count);

    bool try_cuda_scaled_prefix_accumulator_native(const mcl::Fr *values, const mcl::Fr &scale, mcl::Fr *accumulator, std::size_t active_count,
                                                   std::size_t element_count);

    bool barycentric_weights_gpu(const uint64_t omega_limbs[4], const uint64_t x_limbs[4], const uint64_t zero_eval_limbs[4], const uint64_t inv_n_limbs[4],
                                 uint64_t *output_weight_limbs, std::size_t domain_size);

    bool weighted_poly_eval_gpu(const uint64_t *weight_limbs, const uint64_t *interleaved_value_limbs, uint64_t *output_limbs, std::size_t domain_size,
                                std::size_t row_count);

    bool weighted_poly_eval_gpu_columnar(const uint64_t *weight_limbs, const uint64_t *const *column_ptrs, uint64_t *output_limbs, std::size_t domain_size,
                                         std::size_t row_count, const std::size_t *column_lengths = nullptr, std::size_t weight_rotation = 0);

    bool weighted_poly_eval_gpu_multi(const uint64_t *weight_limbs, const uint64_t *const *column_value_ptrs, uint64_t *output_limbs, std::size_t domain_size,
                                      std::size_t poly_count, const std::size_t *column_lengths = nullptr, std::size_t weight_rotation = 0);

    bool weighted_poly_eval_gpu_multi_rotations(const uint64_t *weight_limbs, const uint64_t *const *column_value_ptrs, uint64_t *output_limbs,
                                                std::size_t domain_size, std::size_t poly_count, const std::size_t *rotations, std::size_t rotation_count,
                                                const std::size_t *column_lengths = nullptr);

    bool weighted_poly_eval_gpu_interleaved_multi_rotations(const uint64_t *weight_limbs, const uint64_t *interleaved_value_limbs, uint64_t *output_limbs,
                                                            std::size_t domain_size, std::size_t poly_count, const std::size_t *rotations,
                                                            std::size_t rotation_count);

    bool cuda_interleaved_multi_rotation_admissible(std::size_t domain_size, std::size_t poly_count, std::size_t rotation_count,
                                                    std::size_t additional_host_bytes);

    FieldElement dot_product(const std::vector<FieldElement> &lhs, const std::vector<FieldElement> &rhs);

    std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan(std::size_t size);

    void geometric_powers_native(mcl::Fr *output, std::size_t size, const FieldElement &base, std::size_t max_worker_count = 0);

    void scale_by_geometric_powers_native(mcl::Fr *values, std::size_t size, const FieldElement &base, std::size_t max_worker_count = 0);

    void copy_scaled_by_geometric_powers_native(const mcl::Fr *input, mcl::Fr *output, std::size_t size, const FieldElement &base,
                                                std::size_t max_worker_count = 0);

    void split_coset_coefficients_native(const mcl::Fr *coefficients, const mcl::Fr *shift_powers, std::size_t coefficient_count, const FieldElement &odd_root,
                                         mcl::Fr *even_output, mcl::Fr *odd_output, std::size_t max_worker_count = 0);

    void interleave_native_lanes(const mcl::Fr *even_values, const mcl::Fr *odd_values, std::size_t lane_size, mcl::Fr *output,
                                 std::size_t max_worker_count = 0);

    void intt_in_place(mcl::Fr *values, std::size_t size, std::size_t max_worker_count = 0);

    void ntt_in_place(mcl::Fr *values, std::size_t size, std::size_t max_worker_count = 0);

    void ntt_pair_in_place(mcl::Fr *first, mcl::Fr *second, std::size_t size, std::size_t max_worker_count = 0);

    bool try_cuda_intt_in_place(mcl::Fr *values, std::size_t size);

    bool try_cuda_ntt_in_place(mcl::Fr *values, std::size_t size);

    bool try_cuda_ntt_pair_in_place(mcl::Fr *first, mcl::Fr *second, std::size_t size);

    FieldElement dot_product_native_weights(const std::vector<FieldElement> &lhs, const std::vector<mcl::Fr> &rhs);
}
