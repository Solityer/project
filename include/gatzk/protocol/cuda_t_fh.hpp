#pragma once

#include <cstddef>
#include <memory>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
    struct CudaResidentCoefficientBuffer;
}

namespace gatzk::protocol
{
    struct CudaTFHParameters
    {
        algebra::FieldElement alpha_quot;
        algebra::FieldElement beta_feat;
        algebra::FieldElement eta_feat;
        bool feature_encoding_terms_enabled = false;
    };

    struct CudaTFHTile
    {
        std::size_t logical_begin = 0;
        std::size_t count = 0;

        const mcl::Fr *q_table = nullptr;
        const mcl::Fr *q_query = nullptr;
        const mcl::Fr *table = nullptr;
        const mcl::Fr *query = nullptr;
        const mcl::Fr *multiplicity = nullptr;
        const mcl::Fr *j_table = nullptr;
        const mcl::Fr *j_query = nullptr;
        const mcl::Fr *accumulator = nullptr;
        const mcl::Fr *accumulator_next = nullptr;

        const mcl::Fr *table_row = nullptr;
        const mcl::Fr *table_column = nullptr;
        const mcl::Fr *table_feature = nullptr;
        const mcl::Fr *query_row = nullptr;
        const mcl::Fr *query_column = nullptr;
        const mcl::Fr *query_feature = nullptr;
    };

    class CudaTFHExecutor
    {
    public:
        virtual ~CudaTFHExecutor() = default;

        virtual std::size_t tile_capacity() const = 0;

        virtual std::size_t pipeline_slot_count() const = 0;

        virtual void wait_for_slot(std::size_t slot) = 0;

        virtual void evaluate_tile(const CudaTFHTile &tile, std::size_t slot) = 0;

        virtual std::shared_ptr<const algebra::CudaResidentCoefficientBuffer> finalize_coefficients(algebra::FieldElement *host_coefficients,
                                                                                                    std::size_t host_capacity) = 0;
    };
    bool cuda_t_fh_executor_available();

    bool cuda_t_fh_executor_can_admit(std::size_t domain_size, std::size_t target_size, std::size_t requested_tile_size);

    std::unique_ptr<CudaTFHExecutor> create_cuda_t_fh_executor(std::size_t domain_size, std::size_t target_size, const algebra::FieldElement &shift,
                                                               const algebra::FieldElement &domain_omega, const CudaTFHParameters &parameters,
                                                               std::size_t requested_tile_size);

    void release_cuda_t_fh_executor_cache();
}