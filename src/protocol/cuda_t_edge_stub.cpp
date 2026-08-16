#include "gatzk/protocol/cuda_t_edge.hpp"

namespace gatzk::protocol
{
    std::unique_ptr<CudaTEdgeExecutor>
    create_cuda_t_edge_executor(
        std::size_t,
        std::size_t,
        const algebra::FieldElement &,
        const algebra::FieldElement &,
        std::size_t,
        CudaTEdgeExecutionProfile *)
    {
        return nullptr;
    }

    bool cuda_t_edge_executor_available()
    {
        return false;
    }

    bool cuda_t_edge_executor_can_admit(
        std::size_t,
        std::size_t)
    {
        return false;
    }

    void release_cuda_t_edge_executor_cache()
    {
    }
}