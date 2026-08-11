#pragma once

#include <cstddef>

namespace gatzk::algebra
{
    class FieldElement;
}

namespace gatzk::crypto
{
    struct G1Point;

    // Executes a dense G1 MSM on CUDA only when the active route selected the
    // CUDA field backend and `points` names immutable SRS storage.  A false
    // result means that the caller must retain the canonical CPU path.  Once
    // selected, conversion, upload, launch, and validation errors fail closed.
    bool try_cuda_g1_msm(
        G1Point* output,
        const G1Point* points,
        const algebra::FieldElement* scalars,
        std::size_t count,
        bool stable_srs_bases,
        std::size_t stable_srs_capacity);
}
