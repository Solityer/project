#include "gatzk/protocol/cuda_fh.hpp"

namespace gatzk::protocol
{
    std::unique_ptr<CudaFHExecutor>
    create_cuda_fh_executor(
        std::size_t,
        std::size_t,
        const algebra::FieldElement &,
        const algebra::FieldElement &,
        const algebra::FieldElement &,
        const algebra::FieldElement &)
    {
        return nullptr;
    }

    bool cuda_fh_executor_available()
    {
        return false;
    }

    bool cuda_fh_executor_can_admit(
        std::size_t)
    {
        return false;
    }

    void release_cuda_fh_executor_cache()
    {
    }

} // namespace gatzk::protocol