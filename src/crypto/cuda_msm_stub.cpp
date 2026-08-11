#include "gatzk/crypto/cuda_msm.hpp"

namespace gatzk::crypto
{
    bool try_cuda_g1_msm(
        G1Point*,
        const G1Point*,
        const algebra::FieldElement*,
        std::size_t,
        bool,
        std::size_t)
    {
        return false;
    }
}
