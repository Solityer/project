#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/t_edge_program.hpp"

namespace gatzk::algebra
{
    struct CudaResidentCoefficientBuffer;
}

namespace gatzk::protocol
{

    enum class CudaTEdgeLeafLayout : std::uint8_t
    {
        Contiguous = 0,
        Split = 1,
    };

    struct CudaTEdgeLinearSource
    {
        std::uint32_t slot = 0;
        algebra::FieldElement scale = algebra::FieldElement::one();
        const mcl::Fr *zero_domain_blinder = nullptr;
        std::size_t blinder_count = 0;
    };

    struct CudaTEdgeEvaluationLeafRequest
    {
        std::size_t slot = 0;
        const mcl::Fr *values = nullptr;
        std::size_t value_count = 0;
        const mcl::Fr *zero_domain_blinder = nullptr;
        std::size_t blinder_count = 0;
    };

    class CudaTEdgeExecutor
    {
    public:
        virtual ~CudaTEdgeExecutor() = default;

        virtual std::size_t cache_slot_count() const = 0;

        virtual void prepare_program(const TEdgeBoundProgram &program) = 0;

        virtual bool materialize_evaluation_leaf(std::size_t slot, const mcl::Fr *values, std::size_t value_count, const mcl::Fr *zero_domain_blinder,
                                                 std::size_t blinder_count) = 0;

        virtual bool materialize_evaluation_leaf_batch(const std::vector<CudaTEdgeEvaluationLeafRequest> &requests) = 0;

        virtual bool materialize_coefficient_leaf(std::size_t slot, const mcl::Fr *coefficients, std::size_t coefficient_count) = 0;

        virtual bool materialize_linear_leaf(std::size_t target_slot, const std::vector<CudaTEdgeLinearSource> &sources,
                                             const mcl::Fr *target_zero_domain_blinder, std::size_t target_blinder_count,
                                             CudaTEdgeLeafLayout output_layout = CudaTEdgeLeafLayout::Split) = 0;

        virtual void upload_leaf_chunk(std::size_t slot, std::size_t logical_offset, const mcl::Fr *values, std::size_t count) = 0;

        virtual void accumulate_term_range(std::size_t term_begin, std::size_t term_end, const std::vector<std::uint32_t> &access_slots) = 0;

        virtual void accumulate_term(std::size_t term_index, const std::vector<std::uint32_t> &access_slots)
        {
            accumulate_term_range(term_index, term_index + 1, access_slots);
        }

        virtual void release_leaf(std::size_t slot) = 0;

        virtual void finalize(algebra::FieldElement *output, std::size_t output_count) = 0;

        virtual std::shared_ptr<const algebra::CudaResidentCoefficientBuffer> finalize_coefficients(algebra::FieldElement *output,
                                                                                                    std::size_t output_count) = 0;
    };

    std::unique_ptr<CudaTEdgeExecutor> create_cuda_t_edge_executor(std::size_t domain_size, std::size_t target_size, const algebra::FieldElement &shift,
                                                                   const algebra::FieldElement &omega, std::size_t cache_slot_count);

    bool cuda_t_edge_executor_available();

    bool cuda_t_edge_leaf_exact_ntt_supported(std::size_t target_size);

    struct CudaTEdgeExecutorAdmissionShape
    {
        std::size_t domain_size = 0;
        std::size_t instruction_count = 0;
        std::size_t term_count = 0;
        std::size_t constant_count = 0;
        std::size_t access_count = 0;
        std::size_t max_linear_source_count = 0;
        std::size_t max_linear_single_source_blinder_count = 0;
        std::size_t max_linear_source_blinder_count = 0;
        std::size_t max_linear_target_blinder_count = 0;
        std::size_t caller_host_bytes = 0;
    };

    bool cuda_t_edge_executor_can_admit(std::size_t target_size, std::size_t cache_slot_count, bool leaf_exact_ntt_supported = false);

    bool cuda_t_edge_executor_can_admit(std::size_t target_size, std::size_t cache_slot_count, bool leaf_exact_ntt_supported,
                                        const CudaTEdgeExecutorAdmissionShape &shape);

    void release_cuda_t_edge_executor_cache();

}
