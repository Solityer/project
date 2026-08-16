#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <mcl/bn.hpp>

namespace gatzk::algebra
{
    class CudaWeightedDotSession
    {
    public:
        virtual ~CudaWeightedDotSession() =
            default;

        virtual void accumulate(
            const mcl::Fr *values,
            std::size_t offset,
            std::size_t count) = 0;

        virtual std::vector<mcl::Fr>
        finish() = 0;
    };

    std::unique_ptr<
        CudaWeightedDotSession>
    create_cuda_weighted_dot_session(
        const std::vector<
            const mcl::Fr *> &weight_columns,
        std::size_t total_count,
        std::size_t maximum_chunk_size);

    void release_cuda_weighted_dot_cache();
}