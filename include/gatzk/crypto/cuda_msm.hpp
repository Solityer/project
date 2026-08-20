#pragma once

#include <cstddef>

namespace gatzk::algebra
{
    class FieldElement;
}

namespace gatzk::crypto
{
    struct G1Point;

    // Execute an MSM on CUDA after the execution router has selected CUDA.
    bool try_cuda_g1_msm(
        G1Point *output,
        const G1Point *points,
        const algebra::FieldElement *scalars,
        std::size_t count,
        bool stable_srs_bases,
        std::size_t stable_srs_capacity);

    // Execute several independent scalar vectors against one immutable
    // SRS point slice while sharing its CUDA residency/upload.
    bool try_cuda_g1_msm_same_bases_batch(
        G1Point *const *outputs,
        const G1Point *points,
        const algebra::FieldElement *const *scalar_vectors,
        std::size_t count,
        std::size_t batch_size,
        std::size_t stable_srs_capacity);

    // Execution-only residency probe used by the scheduling layer.
    // A cold singleton SRS request may remain on CPU instead of paying an
    // unamortized normalization/upload/cache-preparation cost.
    bool cuda_g1_msm_resident_prefix_available(
        const G1Point *points,
        std::size_t count,
        std::size_t stable_srs_capacity);

    // Execution-only phase-boundary trim. Keep a bounded hot SRS subset while
    // returning most CUDA memory to large quotient construction.
    void trim_cuda_g1_msm_resident_cache(
        std::size_t maximum_bytes);

    // Fully release resident CUDA SRS material.
    void release_cuda_g1_msm_resident_cache();
}