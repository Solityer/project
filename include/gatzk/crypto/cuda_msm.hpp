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
        const std::size_t *counts,
        std::size_t max_count,
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
    // returning excess CUDA/unified memory to the next proving phase.
    void trim_cuda_g1_msm_resident_cache(
        std::size_t maximum_bytes);

    // True when the active CUDA MSM device uses integrated/unified physical
    // memory. This is execution-only information and never changes KZG
    // semantics or proof contents.
    bool cuda_g1_msm_integrated_memory_device();

    // Release transient scalar staging and streamed-MSM buffers while keeping
    // immutable resident SRS bases alive for later quotient commitments and
    // folded openings.
    void release_cuda_g1_msm_execution_workspace();

    // Final lifecycle release: resident SRS bases and every transient MSM
    // workspace.
    void release_cuda_g1_msm_resident_cache();
}
