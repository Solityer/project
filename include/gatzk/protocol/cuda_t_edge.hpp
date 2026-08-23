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

    class CudaTEdgeExecutor
    {
    public:
        virtual ~CudaTEdgeExecutor() = default;

        virtual std::size_t cache_slot_count() const = 0;

        virtual void prepare_program(
            const TEdgeBoundProgram &program) = 0;

        virtual bool materialize_evaluation_leaf(
            std::size_t slot,
            const mcl::Fr *values,
            std::size_t value_count,
            const mcl::Fr *zero_domain_blinder,
            std::size_t blinder_count) = 0;

        virtual bool materialize_coefficient_leaf(
            std::size_t slot,
            const mcl::Fr *coefficients,
            std::size_t coefficient_count) = 0;

        virtual void upload_leaf_chunk(
            std::size_t slot,
            std::size_t logical_offset,
            const mcl::Fr *values,
            std::size_t count) = 0;

        virtual void accumulate_terms(
            std::size_t term_begin,
            std::size_t term_end,
            const std::vector<std::uint32_t> &access_slots) = 0;

        virtual void release_leaf(
            std::size_t slot) = 0;

        virtual void finalize(
            algebra::FieldElement *output,
            std::size_t output_count) = 0;

        virtual void finalize_coefficients(
            algebra::FieldElement *output,
            std::size_t output_count) = 0;
    };

    std::unique_ptr<CudaTEdgeExecutor> create_cuda_t_edge_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &omega,
        std::size_t cache_slot_count);

    bool cuda_t_edge_executor_available();

    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count);

    void release_cuda_t_edge_executor_cache();

} // namespace gatzk::protocol