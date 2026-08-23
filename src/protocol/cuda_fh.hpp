#pragma once

#include <cstddef>
#include <memory>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol
{
    class CudaFHExecutor
    {
    public:
        virtual ~CudaFHExecutor() = default;

        virtual std::size_t
        cache_slot_count() const = 0;

        virtual bool
        materialize_lagrange_first_leaf(
            std::size_t slot) = 0;

        virtual bool
        materialize_evaluation_leaf(
            std::size_t slot,
            const mcl::Fr *values,
            std::size_t value_count,
            const mcl::Fr *zero_domain_blinder,
            std::size_t blinder_count) = 0;

        virtual bool
        materialize_coefficient_leaf(
            std::size_t slot,
            const mcl::Fr *coefficients,
            std::size_t coefficient_count) = 0;

        virtual void
        upload_leaf_chunk(
            std::size_t slot,
            std::size_t logical_offset,
            const mcl::Fr *values,
            std::size_t count) = 0;

        virtual void
        accumulate_accumulator(
            std::size_t boundary_slot,
            std::size_t accumulator_slot) = 0;

        virtual void
        accumulate_table_grand_product(
            std::size_t boundary_slot,
            std::size_t q_table_slot,
            std::size_t multiplicity_slot,
            std::size_t j_table_slot) = 0;

        virtual void
        accumulate_table_lookup(
            std::size_t q_table_slot,
            std::size_t table_slot,
            std::size_t j_table_slot) = 0;

        virtual void
        accumulate_query_grand_product(
            std::size_t boundary_slot,
            std::size_t q_query_slot,
            std::size_t j_query_slot) = 0;

        virtual void
        accumulate_query_lookup(
            std::size_t q_query_slot,
            std::size_t query_slot,
            std::size_t j_query_slot) = 0;

        virtual void
        finalize_coefficients(
            algebra::FieldElement *output,
            std::size_t output_count) = 0;
    };

    std::unique_ptr<CudaFHExecutor>
    create_cuda_fh_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &omega,
        const algebra::FieldElement &alpha_quot,
        const algebra::FieldElement &beta_feat);

    bool cuda_fh_executor_available();

    bool cuda_fh_executor_can_admit(
        std::size_t target_size);

    void release_cuda_fh_executor_cache();

} // namespace gatzk::protocol