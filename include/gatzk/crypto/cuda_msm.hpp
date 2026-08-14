#pragma once

#include <cstddef>

namespace gatzk::algebra
{
    class FieldElement;
}

namespace gatzk::crypto
{
    struct G1Point;

    // Executes G1 MSM on CUDA when cuda_hotspots is active.
    //
    // Immutable SRS inputs retain the resident/cache fast path whenever the
    // complete MSM fits the current resource envelope.  Oversized,
    // memory-constrained, and temporary compacted inputs use bounded streamed
    // CUDA execution.
    //
    // Returning false means CUDA MSM was not requested because the CPU backend
    // is active.  Once cuda_hotspots selects MSM execution, the MSM stays on
    // CUDA or fails closed; it does not silently fall back to a CPU MSM.
    bool try_cuda_g1_msm(
        G1Point *output,
        const G1Point *points,
        const algebra::FieldElement *scalars,
        std::size_t count,
        bool stable_srs_bases,
        std::size_t stable_srs_capacity);

    // Execution-only same-SRS batching.  Every batch member computes an
    // independent MSM with its own scalar vector while reusing exactly the
    // same immutable SRS point slice.  This changes scheduling only.
    bool try_cuda_g1_msm_same_bases_batch(
        G1Point *const *outputs,
        const G1Point *points,
        const algebra::FieldElement *const *scalar_vectors,
        std::size_t count,
        std::size_t batch_size,
        std::size_t stable_srs_capacity);
}
