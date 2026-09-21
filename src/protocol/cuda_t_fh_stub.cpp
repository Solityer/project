#include "gatzk/protocol/cuda_t_fh.hpp"

#include <stdexcept>

namespace gatzk::protocol
{
    bool cuda_t_fh_executor_available()
    {
        return false;
    }

    bool cuda_t_fh_executor_can_admit(std::size_t, std::size_t, std::size_t)
    {
        return false;
    }

    std::unique_ptr<CudaTFHExecutor> create_cuda_t_fh_executor(
        std::size_t,
        std::size_t,
        const algebra::FieldElement &,
        const algebra::FieldElement &,
        const CudaTFHParameters &,
        std::size_t)
    {
        throw std::runtime_error("CUDA t_FH executor is unavailable in this build");
    }

    void release_cuda_t_fh_executor_cache()
    {
    }
}