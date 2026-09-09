#pragma once

#include <cstddef>
#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
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
    };

    CudaNttWorkspaceResidency cuda_ntt_workspace_residency();

    // Evaluation-basis edge-domain polynomial:
    //
    // host evaluations
    //   -> device INTT
    //   -> exact Z_domain * blinder coefficient correction
    //   -> coset shift / even-odd split
    //   -> bounded (direct or staged) device NTT
    //   -> caller-owned device output.
    //
    // device_output layout:
    //   [even lane][odd lane]
    bool cuda_coset_evaluations_to_device_exact(
        const mcl::Fr *values,
        std::size_t value_count,
        std::size_t domain_size,
        const mcl::Fr *zero_domain_blinder,
        std::size_t blinder_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size);

    // Coefficient-basis polynomial:
    //
    // host coefficients
    //   -> coset shift / even-odd split
    //   -> bounded (direct or staged) device NTT
    //   -> caller-owned device output.
    //
    // Coefficient-basis Polynomial::values() already contains its formal
    // coefficients; do not apply zero_domain_blinder again here.
    bool cuda_coset_coefficients_to_device_exact(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size);

    // Canonical logical coset evaluations already resident on CUDA:
    //
    // device evaluations
    //   -> radix-2 split
    //   -> bounded (direct or staged) device INTT
    //   -> exact inverse radix-2 combine
    //   -> inverse coset shift
    //   -> one final D2H.
    bool cuda_coset_device_to_coefficients_exact(
        const mcl::Fr *device_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients);

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
}
