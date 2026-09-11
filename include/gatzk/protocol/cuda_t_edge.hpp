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

    // Source descriptor for execution-only resident linear materialization.
    // The resident slot contains a complete target-size coset; the blinder
    // coefficients are supplied separately because they are not part of that
    // resident evaluation payload.
    struct CudaTEdgeLinearSource
    {
        std::uint32_t slot = 0;
        algebra::FieldElement scale = algebra::FieldElement::one();
        const mcl::Fr *zero_domain_blinder = nullptr;
        std::size_t blinder_count = 0;
    };

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

        // Materialize
        //
        //   target = sum_i scale_i * source_i
        //          + Z_domain * (R_target - sum_i scale_i * R_source_i)
        //
        // directly on the device. Source slots may be contiguous or split;
        // the target is written in the explicitly selected output layout and
        // is tracked by the executor for later t_edge term loads. No
        // target-sized host vector is accepted or transferred here.
        virtual bool materialize_linear_leaf(
            std::size_t target_slot,
            const std::vector<CudaTEdgeLinearSource> &sources,
            const mcl::Fr *target_zero_domain_blinder,
            std::size_t target_blinder_count,
            CudaTEdgeLeafLayout output_layout =
                CudaTEdgeLeafLayout::Split) = 0;

        virtual void upload_leaf_chunk(
            std::size_t slot,
            std::size_t logical_offset,
            const mcl::Fr *values,
            std::size_t count) = 0;

        virtual void accumulate_term(
            std::size_t term_index,
            const std::vector<std::uint32_t> &access_slots) = 0;

        virtual void release_leaf(
            std::size_t slot) = 0;

        virtual void finalize(
            algebra::FieldElement *output,
            std::size_t output_count) = 0;

        virtual std::shared_ptr<
            const algebra::CudaResidentCoefficientBuffer>
        finalize_coefficients(
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

    // Direct leaf materialization and final coefficient conversion both use a
    // bounded transform over target_size / 2.  Targets above the pinned
    // ICICLE leaf limit are decomposed into exact radix-2 stages. The executor
    // admission path requires this capability before any target-sized
    // host/device work is started.
    bool cuda_t_edge_leaf_exact_ntt_supported(
        std::size_t target_size);

    // Upper bounds for every buffer that the production t_edge executor can
    // have live before/through materialize_linear_leaf().  The prover fills
    // this from the bound program and the registered blinder descriptors so
    // admission and ensure_linear_buffers() use the same allocation shape.
    // `caller_host_bytes` covers caller-owned output/staging vectors that are
    // live while the executor is resident.  Zero fields are valid only for
    // the legacy shape-only overload below; production large-route selection
    // must pass the complete shape.
    struct CudaTEdgeExecutorAdmissionShape
    {
        std::size_t domain_size = 0;
        std::size_t instruction_count = 0;
        std::size_t constant_count = 0;
        std::size_t access_count = 0;
        std::size_t max_linear_source_count = 0;
        std::size_t max_linear_source_blinder_count = 0;
        std::size_t max_linear_target_blinder_count = 0;
        std::size_t caller_host_bytes = 0;
    };

    // CPUOnly or an unselected CUDA route owns the canonical CPU path.  Once
    // policy, capability, and the complete admission shape select this large
    // CUDA route, a false admission result is fail-closed and must not be
    // converted into a silent CPU fallback.
    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count,
        bool leaf_exact_ntt_supported = false);

    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count,
        bool leaf_exact_ntt_supported,
        const CudaTEdgeExecutorAdmissionShape &shape);

    void release_cuda_t_edge_executor_cache();

} // namespace gatzk::protocol
