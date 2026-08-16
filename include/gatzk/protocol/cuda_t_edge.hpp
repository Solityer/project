#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/t_edge_program.hpp"

namespace gatzk::protocol
{
    struct CudaTEdgeExecutionProfile
    {
        std::size_t term_count = 0;
        std::size_t access_count = 0;
        std::size_t unique_leaf_count = 0;
        std::size_t leaf_cache_slot_count = 0;
        std::size_t leaf_materialization_count = 0;
        std::size_t leaf_cache_hit_count = 0;
        std::size_t leaf_eviction_count = 0;
        std::size_t source_bytes_uploaded = 0;
        std::size_t quotient_bytes_downloaded = 0;

        double source_read_ms = 0.0;
        double leaf_ntt_ms = 0.0;
        double derived_leaf_ms = 0.0;
        double constraint_kernel_ms = 0.0;
        double final_download_ms = 0.0;
    };

    class CudaTEdgeExecutor
    {
    public:
        virtual ~CudaTEdgeExecutor() = default;

        virtual std::size_t cache_slot_count() const = 0;

        virtual void prepare_program(
            const TEdgeBoundProgram &program) = 0;

        virtual void upload_leaf_chunk(
            std::size_t slot,
            std::size_t logical_offset,
            const mcl::Fr *values,
            std::size_t count) = 0;

        virtual void accumulate_term(
            std::size_t term_index,
            const std::vector<std::uint32_t>
                &access_slots) = 0;

        virtual void release_leaf(
            std::size_t slot) = 0;

        virtual void finalize(
            algebra::FieldElement *output,
            std::size_t output_count) = 0;
    };

    std::unique_ptr<CudaTEdgeExecutor>
    create_cuda_t_edge_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &omega,
        std::size_t cache_slot_count,
        CudaTEdgeExecutionProfile *profile);
        
    bool cuda_t_edge_executor_available();

    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count);

    void release_cuda_t_edge_executor_cache();
}