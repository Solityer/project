#pragma once

#include <cstddef>
#include <memory>
#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
    class CudaHostTransferGuard;
    struct CudaResidentCoefficientBuffer;
    // Snapshot of the thread-local bounded NTT workspace.  Admission callers
    // use this to account for already resident input/output/scratch buffers;
    // the snapshot performs no allocation and is safe for fail-closed checks.
    struct CudaNttWorkspaceResidency
    {
        std::size_t capacity_bytes = 0;
        std::size_t resident_bytes = 0;
        int device = -1;
        bool has_input_output = false;
        bool has_staged_scratch = false;
        bool domain_initialized = false;
    };

    CudaNttWorkspaceResidency cuda_ntt_workspace_residency();

    struct CudaCosetEvaluationBatchRequest
    {
        const mcl::Fr *values = nullptr;
        std::size_t value_count = 0;

        const mcl::Fr *zero_domain_blinder = nullptr;
        std::size_t blinder_count = 0;

        mcl::Fr *device_output = nullptr;
    };

    bool cuda_coset_evaluations_to_device_exact(
        const mcl::Fr *values,
        std::size_t value_count,
        std::size_t domain_size,
        const mcl::Fr *zero_domain_blinder,
        std::size_t blinder_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size,
        bool make_default_stream_wait = false,
        bool wait_for_default_stream_before_output = false);

    bool cuda_coset_evaluations_to_device_exact_batch(
        const CudaCosetEvaluationBatchRequest *requests,
        std::size_t request_count,
        std::size_t domain_size,
        const FieldElement &shift,
        std::size_t target_size,
        bool make_default_stream_wait = false,
        std::size_t maximum_batch_size = 32);

    bool cuda_coset_coefficients_to_device_exact(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size,
        bool make_default_stream_wait = false,
        bool wait_for_default_stream_before_output = false);

    bool cuda_coset_device_to_coefficients_exact(
        const mcl::Fr *device_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients,
        bool wait_for_default_stream = false,
        mcl::Fr *device_coefficients = nullptr,
        std::shared_ptr<
            const CudaHostTransferGuard>
            *host_transfer = nullptr,
        std::size_t *logical_coefficient_count = nullptr,
        std::size_t *nonzero_coefficient_count = nullptr);

    bool cuda_coset_evaluations_to_host_exact(
        const mcl::Fr *values,
        std::size_t value_count,
        std::size_t domain_size,
        const mcl::Fr *zero_domain_blinder,
        std::size_t blinder_count,
        const FieldElement &shift,
        mcl::Fr *host_output,
        std::size_t target_size,
        bool preserve_split_lane);

    bool cuda_coset_coefficients_to_host_exact(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const FieldElement &shift,
        mcl::Fr *host_output,
        std::size_t target_size,
        bool preserve_split_lane);

    bool cuda_coset_host_to_coefficients_exact(
        const mcl::Fr *host_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients);

    bool cuda_coset_host_to_resident_coefficients_exact(
        const mcl::Fr *host_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients,
        std::shared_ptr<
            const CudaResidentCoefficientBuffer>
            *resident_coefficients);

    bool cuda_update_resident_coefficient_ranges(
        const CudaResidentCoefficientBuffer &resident,
        const FieldElement *host_coefficients,
        std::size_t host_coefficient_count,
        std::size_t first_offset,
        std::size_t first_count,
        std::size_t second_offset,
        std::size_t second_count);
}
