#include "gatzk/crypto/cuda_msm.hpp"

namespace gatzk::crypto
{
    bool try_cuda_g1_msm(G1Point *, const G1Point *, const algebra::FieldElement *, std::size_t, bool, std::size_t, const std::uint64_t *, std::size_t, int)
    {
        return false;
    }

    bool try_cuda_g1_msm_same_bases_batch(G1Point *const *, const G1Point *, const algebra::FieldElement *const *, const std::size_t *, std::size_t,
                                          std::size_t, std::size_t, const std::uint64_t *const *, const std::size_t *, const int *)
    {
        return false;
    }

    bool cuda_g1_msm_resident_prefix_available(const G1Point *, std::size_t, std::size_t) { return false; }

    void trim_cuda_g1_msm_resident_cache(std::size_t) {}

    bool cuda_g1_msm_integrated_memory_device() { return false; }

    void release_cuda_g1_msm_execution_workspace() {}

    void release_cuda_g1_msm_resident_cache() {}
}
