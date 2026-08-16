#include "gatzk/protocol/cuda_t_edge.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/vector_ops.hpp"

#include "fields/snark_fields/bls12_381_scalar.cuh"
#include "gpu-utils/device_context.cuh"
#include "ntt/ntt.cuh"

namespace gatzk::protocol
{
    namespace
    {
        using GpuFr =
            algebra::gpu::GpuFr;

        using IcicleScalar =
            bls12_381::scalar_t;

        __device__ __forceinline__
            std::size_t
            split_physical_index(
                std::size_t logical_index,
                std::size_t lane_size)
        {
            return logical_index / 2 +
                   ((logical_index & 1U) != 0U
                        ? lane_size
                        : 0);
        }

        // 1. evaluation -> coefficient 的 inverse NTT
        // 2. coefficient/blinder 合并
        // 3. split + coset scale
        // 4. paired forward NTT
        // 5. linear leaf
        // 6. t_edge program interpreter
        // 7. quotient finalize
        //
        // 这里实现 CudaTEdgeExecutorImpl。
    }

    std::unique_ptr<CudaTEdgeExecutor>
    create_cuda_t_edge_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &omega,
        std::size_t cache_slot_count,
        CudaTEdgeExecutionProfile *profile)
    {
        if (!cuda_t_edge_executor_available())
        {
            return nullptr;
        }

        return std::make_unique<
            CudaTEdgeExecutorImpl>(
            domain_size,
            target_size,
            shift,
            omega,
            cache_slot_count,
            profile);
    }

    bool cuda_t_edge_executor_available()
    {
        return algebra::
            cuda_field_hotspots_available();
    }

    void release_cuda_t_edge_executor_cache()
    {
        CudaTEdgeExecutorImpl::
            release_shared_resources();
    }
}