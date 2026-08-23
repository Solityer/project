#pragma once

#include <cstddef>
#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
    // Evaluation-basis edge-domain polynomial:
    //
    // host evaluations
    //   -> device INTT
    //   -> exact Z_domain * blinder coefficient correction
    //   -> coset shift / even-odd split
    //   -> paired device NTT
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
    //   -> paired device NTT
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

    // Materialize the first Lagrange basis polynomial L_0 of a
    // root-of-unity domain directly on the shifted quotient coset.
    //
    // L_0 has coefficient vector:
    //
    //   [1/n, 1/n, ..., 1/n]
    //
    // of length domain_size.  Constructing it directly avoids a full
    // host-side evaluation vector and source-domain INTT.
    //
    // device_output uses the same [even lane][odd lane] layout as the
    // other exact CUDA coset materializers.
    bool cuda_coset_lagrange_first_to_device_exact(
        std::size_t domain_size,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size);

    // Canonical logical coset evaluations already resident on CUDA:
    //
    // device evaluations
    //   -> radix-2 split
    //   -> paired device INTT
    //   -> exact inverse radix-2 combine
    //   -> inverse coset shift
    //   -> one final D2H.
    bool cuda_coset_device_to_coefficients_exact(
        const mcl::Fr *device_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients);
}