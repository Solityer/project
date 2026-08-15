#include "gatzk/crypto/cuda_msm.hpp"

namespace gatzk::crypto
{
    bool try_cuda_g1_msm(
        G1Point *,
        const G1Point *,
        const algebra::FieldElement *,
        std::size_t,
        bool,
        std::size_t)
    {
        return false;
    }

    bool try_cuda_g1_msm_same_bases_batch(
        G1Point *const *,
        const G1Point *,
        const algebra::FieldElement *const *,
        std::size_t,
        std::size_t,
        std::size_t)
    {
        return false;
    }

    void release_cuda_g1_msm_resident_cache()
    {
    }
}