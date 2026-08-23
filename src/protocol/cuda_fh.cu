#include "gatzk/protocol/cuda_fh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/vector_ops.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using GpuFr =
            algebra::gpu::GpuFr;

        constexpr std::size_t
            kFHLeafCacheSlots = 4;

        static_assert(
            sizeof(GpuFr) ==
            sizeof(mcl::Fr));

        std::size_t checked_bytes(
            std::size_t count,
            std::size_t element_size,
            const char *label)
        {
            if (count >
                std::numeric_limits<
                    std::size_t>::max() /
                    element_size)
            {
                throw std::runtime_error(
                    std::string(
                        "CUDA FH byte overflow: ") +
                    label);
            }

            return count *
                   element_size;
        }

        std::size_t checked_add(
            std::size_t lhs,
            std::size_t rhs,
            const char *label)
        {
            if (lhs >
                std::numeric_limits<
                    std::size_t>::max() -
                    rhs)
            {
                throw std::runtime_error(
                    std::string(
                        "CUDA FH byte overflow: ") +
                    label);
            }

            return lhs + rhs;
        }

        [[noreturn]] void
        throw_cuda_fh_error(
            const char *operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string("CUDA FH ") +
                operation +
                " failed: " +
                cudaGetErrorString(status));
        }

        GpuFr gpu_field(
            const algebra::FieldElement &value)
        {
            GpuFr output{};

            const auto &native =
                value.native();

            std::memcpy(
                &output,
                &native,
                sizeof(output));

            return output;
        }

        __device__ __forceinline__
            GpuFr
            load_logical_leaf(
                const GpuFr *leaf,
                std::uint8_t split_layout,
                std::size_t logical_index,
                std::size_t target_size)
        {
            auto physical_index =
                logical_index;

            if (split_layout != 0)
            {
                const auto half_size =
                    target_size / 2;

                physical_index =
                    logical_index / 2 +
                    ((logical_index & 1U) != 0U
                         ? half_size
                         : 0);
            }

            return leaf[physical_index];
        }

        __global__ void
        fh_accumulator_kernel(
            const GpuFr *boundary,
            std::uint8_t boundary_split,
            const GpuFr *accumulator,
            std::uint8_t accumulator_split,
            std::size_t rotation_step,
            std::size_t target_size,
            GpuFr alpha,
            GpuFr alpha_squared,
            GpuFr *quotient)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= target_size)
            {
                return;
            }

            const auto rotated_index =
                (index +
                 rotation_step) &
                (target_size - 1U);

            const auto first =
                load_logical_leaf(
                    boundary,
                    boundary_split,
                    index,
                    target_size);

            const auto last =
                load_logical_leaf(
                    boundary,
                    boundary_split,
                    rotated_index,
                    target_size);

            const auto accumulator_value =
                load_logical_leaf(
                    accumulator,
                    accumulator_split,
                    index,
                    target_size);

            const auto accumulator_next =
                load_logical_leaf(
                    accumulator,
                    accumulator_split,
                    rotated_index,
                    target_size);

            const auto one =
                algebra::gpu::fr_one();

            const auto not_last =
                algebra::gpu::fr_sub(
                    one,
                    last);

            const auto accumulator_delta =
                algebra::gpu::fr_sub(
                    accumulator_next,
                    accumulator_value);

            const auto term0 =
                algebra::gpu::fr_mul(
                    first,
                    accumulator_value);

            const auto term1 =
                algebra::gpu::fr_mul(
                    not_last,
                    accumulator_delta);

            const auto term2 =
                algebra::gpu::fr_mul(
                    last,
                    accumulator_value);

            auto contribution =
                term0;

            contribution =
                algebra::gpu::fr_add(
                    contribution,
                    algebra::gpu::fr_mul(
                        alpha,
                        term1));

            contribution =
                algebra::gpu::fr_add(
                    contribution,
                    algebra::gpu::fr_mul(
                        alpha_squared,
                        term2));

            quotient[index] =
                algebra::gpu::fr_add(
                    quotient[index],
                    contribution);
        }

        __global__ void
        fh_table_grand_product_kernel(
            const GpuFr *boundary,
            std::uint8_t boundary_split,
            const GpuFr *q_table,
            std::uint8_t q_table_split,
            const GpuFr *multiplicity,
            std::uint8_t multiplicity_split,
            const GpuFr *j_table,
            std::uint8_t j_table_split,
            std::size_t rotation_step,
            std::size_t target_size,
            GpuFr alpha,
            GpuFr *quotient)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= target_size)
            {
                return;
            }

            const auto rotated_index =
                (index +
                 rotation_step) &
                (target_size - 1U);

            const auto last =
                load_logical_leaf(
                    boundary,
                    boundary_split,
                    rotated_index,
                    target_size);

            const auto q =
                load_logical_leaf(
                    q_table,
                    q_table_split,
                    index,
                    target_size);

            const auto m =
                load_logical_leaf(
                    multiplicity,
                    multiplicity_split,
                    index,
                    target_size);

            const auto j =
                load_logical_leaf(
                    j_table,
                    j_table_split,
                    index,
                    target_size);

            const auto not_last =
                algebra::gpu::fr_sub(
                    algebra::gpu::fr_one(),
                    last);

            auto product =
                algebra::gpu::fr_mul(
                    q,
                    m);

            product =
                algebra::gpu::fr_mul(
                    product,
                    j);

            product =
                algebra::gpu::fr_sub(
                    algebra::gpu::fr_zero(),
                    product);

            const auto term =
                algebra::gpu::fr_mul(
                    not_last,
                    product);

            quotient[index] =
                algebra::gpu::fr_add(
                    quotient[index],
                    algebra::gpu::fr_mul(
                        alpha,
                        term));
        }

        __global__ void
        fh_table_lookup_kernel(
            const GpuFr *q_table,
            std::uint8_t q_table_split,
            const GpuFr *table,
            std::uint8_t table_split,
            const GpuFr *j_table,
            std::uint8_t j_table_split,
            std::size_t target_size,
            GpuFr beta,
            GpuFr alpha3,
            GpuFr alpha4,
            GpuFr *quotient)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= target_size)
            {
                return;
            }

            const auto q =
                load_logical_leaf(
                    q_table,
                    q_table_split,
                    index,
                    target_size);

            const auto table_value =
                load_logical_leaf(
                    table,
                    table_split,
                    index,
                    target_size);

            const auto j =
                load_logical_leaf(
                    j_table,
                    j_table_split,
                    index,
                    target_size);

            const auto one =
                algebra::gpu::fr_one();

            auto lookup =
                algebra::gpu::fr_add(
                    table_value,
                    beta);

            lookup =
                algebra::gpu::fr_mul(
                    lookup,
                    j);

            lookup =
                algebra::gpu::fr_sub(
                    lookup,
                    one);

            const auto term3 =
                algebra::gpu::fr_mul(
                    q,
                    lookup);

            const auto term4 =
                algebra::gpu::fr_mul(
                    algebra::gpu::fr_sub(
                        one,
                        q),
                    j);

            auto contribution =
                algebra::gpu::fr_mul(
                    alpha3,
                    term3);

            contribution =
                algebra::gpu::fr_add(
                    contribution,
                    algebra::gpu::fr_mul(
                        alpha4,
                        term4));

            quotient[index] =
                algebra::gpu::fr_add(
                    quotient[index],
                    contribution);
        }

        __global__ void
        fh_query_grand_product_kernel(
            const GpuFr *boundary,
            std::uint8_t boundary_split,
            const GpuFr *q_query,
            std::uint8_t q_query_split,
            const GpuFr *j_query,
            std::uint8_t j_query_split,
            std::size_t rotation_step,
            std::size_t target_size,
            GpuFr alpha,
            GpuFr *quotient)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= target_size)
            {
                return;
            }

            const auto rotated_index =
                (index +
                 rotation_step) &
                (target_size - 1U);

            const auto last =
                load_logical_leaf(
                    boundary,
                    boundary_split,
                    rotated_index,
                    target_size);

            const auto q =
                load_logical_leaf(
                    q_query,
                    q_query_split,
                    index,
                    target_size);

            const auto j =
                load_logical_leaf(
                    j_query,
                    j_query_split,
                    index,
                    target_size);

            const auto not_last =
                algebra::gpu::fr_sub(
                    algebra::gpu::fr_one(),
                    last);

            const auto term =
                algebra::gpu::fr_mul(
                    not_last,
                    algebra::gpu::fr_mul(
                        q,
                        j));

            quotient[index] =
                algebra::gpu::fr_add(
                    quotient[index],
                    algebra::gpu::fr_mul(
                        alpha,
                        term));
        }

        __global__ void
        fh_query_lookup_kernel(
            const GpuFr *q_query,
            std::uint8_t q_query_split,
            const GpuFr *query,
            std::uint8_t query_split,
            const GpuFr *j_query,
            std::uint8_t j_query_split,
            std::size_t target_size,
            GpuFr beta,
            GpuFr alpha5,
            GpuFr alpha6,
            GpuFr *quotient)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= target_size)
            {
                return;
            }

            const auto q =
                load_logical_leaf(
                    q_query,
                    q_query_split,
                    index,
                    target_size);

            const auto query_value =
                load_logical_leaf(
                    query,
                    query_split,
                    index,
                    target_size);

            const auto j =
                load_logical_leaf(
                    j_query,
                    j_query_split,
                    index,
                    target_size);

            const auto one =
                algebra::gpu::fr_one();

            auto lookup =
                algebra::gpu::fr_add(
                    query_value,
                    beta);

            lookup =
                algebra::gpu::fr_mul(
                    lookup,
                    j);

            lookup =
                algebra::gpu::fr_sub(
                    lookup,
                    one);

            const auto term5 =
                algebra::gpu::fr_mul(
                    q,
                    lookup);

            const auto term6 =
                algebra::gpu::fr_mul(
                    algebra::gpu::fr_sub(
                        one,
                        q),
                    j);

            auto contribution =
                algebra::gpu::fr_mul(
                    alpha5,
                    term5);

            contribution =
                algebra::gpu::fr_add(
                    contribution,
                    algebra::gpu::fr_mul(
                        alpha6,
                        term6));

            quotient[index] =
                algebra::gpu::fr_add(
                    quotient[index],
                    contribution);
        }

        __global__ void
        fh_finalize_kernel(
            GpuFr *quotient,
            const GpuFr *zero_inverse_cycle,
            std::size_t target_size,
            std::size_t cycle_size)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= target_size)
            {
                return;
            }

            quotient[index] =
                algebra::gpu::fr_mul(
                    quotient[index],
                    zero_inverse_cycle[index %
                                       cycle_size]);
        }

        class CudaFHExecutorImpl final
            : public CudaFHExecutor
        {
        public:
            CudaFHExecutorImpl(
                std::size_t domain_size,
                std::size_t target_size,
                const algebra::FieldElement &shift,
                const algebra::FieldElement &omega,
                const algebra::FieldElement &alpha_quot,
                const algebra::FieldElement &beta_feat)
                : domain_size_(
                      domain_size),
                  target_size_(
                      target_size),
                  shift_(
                      shift),
                  rotation_step_(
                      target_size /
                      domain_size),
                  beta_(
                      gpu_field(
                          beta_feat))
            {
                (void)omega;

                if (domain_size_ == 0 ||
                    target_size_ == 0 ||
                    target_size_ %
                            domain_size_ !=
                        0 ||
                    (domain_size_ &
                     (domain_size_ - 1U)) != 0U ||
                    (target_size_ &
                     (target_size_ - 1U)) != 0U)
                {
                    throw std::runtime_error(
                        "invalid CUDA FH executor shape");
                }

                auto alpha_power =
                    algebra::FieldElement::one();

                for (std::size_t index = 0;
                     index <
                     alpha_powers_.size();
                     ++index)
                {
                    alpha_powers_[index] =
                        gpu_field(
                            alpha_power);

                    alpha_power *=
                        alpha_quot;
                }

                try
                {
                    const auto leaf_bytes =
                        checked_bytes(
                            target_size_,
                            sizeof(GpuFr),
                            "leaf");

                    for (auto &slot :
                         device_leaf_slots_)
                    {
                        const auto status =
                            cudaMalloc(
                                reinterpret_cast<
                                    void **>(
                                    &slot),
                                leaf_bytes);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_cuda_fh_error(
                                "leaf allocation",
                                status);
                        }
                    }

                    auto status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_quotient_),
                            leaf_bytes);

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_fh_error(
                            "quotient allocation",
                            status);
                    }

                    status =
                        cudaMemset(
                            device_quotient_,
                            0,
                            leaf_bytes);

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_fh_error(
                            "quotient clear",
                            status);
                    }

                    const auto cycle_size =
                        target_size_ /
                        domain_size_;

                    std::vector<
                        algebra::FieldElement>
                        zero_inverse_cycle(
                            cycle_size,
                            algebra::FieldElement::
                                zero());

                    const auto root =
                        algebra::FieldElement::
                            root_of_unity(
                                target_size_);

                    const auto zero_step =
                        root.pow(
                            domain_size_);

                    auto zero_power =
                        shift_.pow(
                            domain_size_);

                    const auto one =
                        algebra::FieldElement::
                            one();

                    for (std::size_t index = 0;
                         index < cycle_size;
                         ++index)
                    {
                        const auto denominator =
                            zero_power -
                            one;

                        if (denominator.is_zero())
                        {
                            throw std::runtime_error(
                                "CUDA FH coset overlaps quotient domain");
                        }

                        zero_inverse_cycle[index] =
                            denominator.inv();

                        zero_power *=
                            zero_step;
                    }

                    zero_inverse_cycle_size_ =
                        cycle_size;

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_zero_inverse_cycle_),
                            checked_bytes(
                                cycle_size,
                                sizeof(GpuFr),
                                "zero inverse cycle"));

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_fh_error(
                            "zero-cycle allocation",
                            status);
                    }

                    status =
                        cudaMemcpy(
                            device_zero_inverse_cycle_,
                            algebra::FieldElement::
                                native_array_data(
                                    zero_inverse_cycle
                                        .data()),
                            checked_bytes(
                                cycle_size,
                                sizeof(GpuFr),
                                "zero inverse cycle"),
                            cudaMemcpyHostToDevice);

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_fh_error(
                            "zero-cycle upload",
                            status);
                    }
                }
                catch (...)
                {
                    release_all_noexcept();
                    throw;
                }
            }

            ~CudaFHExecutorImpl()
                override
            {
                release_all_noexcept();
            }

            std::size_t
            cache_slot_count() const
                override
            {
                return device_leaf_slots_
                    .size();
            }

            bool
            materialize_lagrange_first_leaf(
                std::size_t slot)
                override
            {
                validate_slot(slot);

                synchronize_before_overwrite();

                const bool executed =
                    algebra::
                        cuda_coset_lagrange_first_to_device_exact(
                            domain_size_,
                            shift_,
                            reinterpret_cast<
                                mcl::Fr *>(
                                device_leaf_slots_[slot]),
                            target_size_);

                if (executed)
                {
                    split_layout_[slot] =
                        1;
                }

                return executed;
            }

            bool
            materialize_evaluation_leaf(
                std::size_t slot,
                const mcl::Fr *values,
                std::size_t value_count,
                const mcl::Fr *zero_domain_blinder,
                std::size_t blinder_count)
                override
            {
                validate_slot(slot);

                synchronize_before_overwrite();

                const bool executed =
                    algebra::
                        cuda_coset_evaluations_to_device_exact(
                            values,
                            value_count,
                            domain_size_,
                            zero_domain_blinder,
                            blinder_count,
                            shift_,
                            reinterpret_cast<
                                mcl::Fr *>(
                                device_leaf_slots_[slot]),
                            target_size_);

                if (executed)
                {
                    split_layout_[slot] =
                        1;
                }

                return executed;
            }

            bool
            materialize_coefficient_leaf(
                std::size_t slot,
                const mcl::Fr *coefficients,
                std::size_t coefficient_count)
                override
            {
                validate_slot(slot);

                synchronize_before_overwrite();

                const bool executed =
                    algebra::
                        cuda_coset_coefficients_to_device_exact(
                            coefficients,
                            coefficient_count,
                            shift_,
                            reinterpret_cast<
                                mcl::Fr *>(
                                device_leaf_slots_[slot]),
                            target_size_);

                if (executed)
                {
                    split_layout_[slot] =
                        1;
                }

                return executed;
            }

            void
            upload_leaf_chunk(
                std::size_t slot,
                std::size_t logical_offset,
                const mcl::Fr *values,
                std::size_t count)
                override
            {
                validate_slot(slot);

                if (values == nullptr ||
                    logical_offset >
                        target_size_ ||
                    count >
                        target_size_ -
                            logical_offset)
                {
                    throw std::runtime_error(
                        "invalid CUDA FH leaf upload range");
                }

                if (count == 0)
                {
                    return;
                }

                if (logical_offset == 0)
                {
                    synchronize_before_overwrite();

                    split_layout_[slot] =
                        0;
                }

                const auto status =
                    cudaMemcpy(
                        device_leaf_slots_[slot] +
                            logical_offset,
                        values,
                        checked_bytes(
                            count,
                            sizeof(GpuFr),
                            "leaf upload"),
                        cudaMemcpyHostToDevice);

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_fh_error(
                        "leaf upload",
                        status);
                }
            }

            void
            accumulate_accumulator(
                std::size_t boundary_slot,
                std::size_t accumulator_slot)
                override
            {
                validate_slot(boundary_slot);
                validate_slot(accumulator_slot);

                const auto blocks =
                    block_count();

                fh_accumulator_kernel<<<
                    blocks,
                    kBlockSize>>>(
                    device_leaf_slots_[boundary_slot],
                    split_layout_[boundary_slot],
                    device_leaf_slots_[accumulator_slot],
                    split_layout_[accumulator_slot],
                    rotation_step_,
                    target_size_,
                    alpha_powers_[1],
                    alpha_powers_[2],
                    device_quotient_);

                check_kernel(
                    "accumulator kernel");
            }

            void
            accumulate_table_grand_product(
                std::size_t boundary_slot,
                std::size_t q_table_slot,
                std::size_t multiplicity_slot,
                std::size_t j_table_slot)
                override
            {
                validate_slot(boundary_slot);
                validate_slot(q_table_slot);
                validate_slot(multiplicity_slot);
                validate_slot(j_table_slot);

                fh_table_grand_product_kernel<<<
                    block_count(),
                    kBlockSize>>>(
                    device_leaf_slots_[boundary_slot],
                    split_layout_[boundary_slot],
                    device_leaf_slots_[q_table_slot],
                    split_layout_[q_table_slot],
                    device_leaf_slots_[multiplicity_slot],
                    split_layout_[multiplicity_slot],
                    device_leaf_slots_[j_table_slot],
                    split_layout_[j_table_slot],
                    rotation_step_,
                    target_size_,
                    alpha_powers_[1],
                    device_quotient_);

                check_kernel(
                    "table grand-product kernel");
            }

            void
            accumulate_table_lookup(
                std::size_t q_table_slot,
                std::size_t table_slot,
                std::size_t j_table_slot)
                override
            {
                validate_slot(q_table_slot);
                validate_slot(table_slot);
                validate_slot(j_table_slot);

                fh_table_lookup_kernel<<<
                    block_count(),
                    kBlockSize>>>(
                    device_leaf_slots_[q_table_slot],
                    split_layout_[q_table_slot],
                    device_leaf_slots_[table_slot],
                    split_layout_[table_slot],
                    device_leaf_slots_[j_table_slot],
                    split_layout_[j_table_slot],
                    target_size_,
                    beta_,
                    alpha_powers_[3],
                    alpha_powers_[4],
                    device_quotient_);

                check_kernel(
                    "table lookup kernel");
            }

            void
            accumulate_query_grand_product(
                std::size_t boundary_slot,
                std::size_t q_query_slot,
                std::size_t j_query_slot)
                override
            {
                validate_slot(boundary_slot);
                validate_slot(q_query_slot);
                validate_slot(j_query_slot);

                fh_query_grand_product_kernel<<<
                    block_count(),
                    kBlockSize>>>(
                    device_leaf_slots_[boundary_slot],
                    split_layout_[boundary_slot],
                    device_leaf_slots_[q_query_slot],
                    split_layout_[q_query_slot],
                    device_leaf_slots_[j_query_slot],
                    split_layout_[j_query_slot],
                    rotation_step_,
                    target_size_,
                    alpha_powers_[1],
                    device_quotient_);

                check_kernel(
                    "query grand-product kernel");
            }

            void
            accumulate_query_lookup(
                std::size_t q_query_slot,
                std::size_t query_slot,
                std::size_t j_query_slot)
                override
            {
                validate_slot(q_query_slot);
                validate_slot(query_slot);
                validate_slot(j_query_slot);

                fh_query_lookup_kernel<<<
                    block_count(),
                    kBlockSize>>>(
                    device_leaf_slots_[q_query_slot],
                    split_layout_[q_query_slot],
                    device_leaf_slots_[query_slot],
                    split_layout_[query_slot],
                    device_leaf_slots_[j_query_slot],
                    split_layout_[j_query_slot],
                    target_size_,
                    beta_,
                    alpha_powers_[5],
                    alpha_powers_[6],
                    device_quotient_);

                check_kernel(
                    "query lookup kernel");
            }

            void finalize_coefficients(
                algebra::FieldElement *output,
                std::size_t output_count)
                override
            {
                if (output == nullptr ||
                    output_count !=
                        target_size_ ||
                    zero_inverse_cycle_size_ ==
                        0)
                {
                    throw std::runtime_error(
                        "invalid CUDA FH coefficient output");
                }

                fh_finalize_kernel<<<
                    block_count(),
                    kBlockSize>>>(
                    device_quotient_,
                    device_zero_inverse_cycle_,
                    target_size_,
                    zero_inverse_cycle_size_);

                check_kernel(
                    "finalize kernel");

                auto status =
                    cudaStreamSynchronize(0);

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_fh_error(
                        "finalize synchronization",
                        status);
                }

                const bool executed =
                    algebra::
                        cuda_coset_device_to_coefficients_exact(
                            reinterpret_cast<
                                const mcl::Fr *>(
                                device_quotient_),
                            target_size_,
                            shift_,
                            algebra::FieldElement::
                                mutable_native_array_data(
                                    output));

                if (!executed)
                {
                    throw std::runtime_error(
                        "CUDA FH device quotient INTT was not admitted");
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);
            }

        private:
            static constexpr unsigned
                kBlockSize = 256;

            void validate_slot(
                std::size_t slot) const
            {
                if (slot >=
                    device_leaf_slots_
                        .size())
                {
                    throw std::runtime_error(
                        "invalid CUDA FH leaf slot");
                }
            }

            void synchronize_before_overwrite()
            {
                const auto status =
                    cudaStreamSynchronize(0);

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_fh_error(
                        "leaf overwrite synchronization",
                        status);
                }
            }

            unsigned block_count() const
            {
                const auto count =
                    (target_size_ +
                     kBlockSize - 1) /
                    kBlockSize;

                if (count >
                    std::numeric_limits<
                        unsigned>::max())
                {
                    throw std::runtime_error(
                        "CUDA FH grid overflow");
                }

                return static_cast<unsigned>(
                    count);
            }

            void check_kernel(
                const char *label)
            {
                const auto status =
                    cudaGetLastError();

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_fh_error(
                        label,
                        status);
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);
            }

            void release_all_noexcept() noexcept
            {
                for (auto *slot :
                     device_leaf_slots_)
                {
                    if (slot != nullptr)
                    {
                        (void)cudaFree(
                            slot);
                    }
                }

                if (device_quotient_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_quotient_);
                }

                if (device_zero_inverse_cycle_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_zero_inverse_cycle_);
                }

                device_leaf_slots_.fill(
                    nullptr);

                device_quotient_ =
                    nullptr;

                device_zero_inverse_cycle_ =
                    nullptr;
            }

            std::size_t domain_size_ = 0;
            std::size_t target_size_ = 0;

            algebra::FieldElement
                shift_;

            std::size_t rotation_step_ = 0;

            std::array<
                GpuFr *,
                kFHLeafCacheSlots>
                device_leaf_slots_{};

            std::array<
                std::uint8_t,
                kFHLeafCacheSlots>
                split_layout_{};

            GpuFr *device_quotient_ =
                nullptr;

            GpuFr
                *device_zero_inverse_cycle_ =
                    nullptr;

            std::size_t
                zero_inverse_cycle_size_ = 0;

            std::array<GpuFr, 7>
                alpha_powers_{};

            GpuFr beta_{};
        };

        struct HostMemoryInfo
        {
            std::size_t total = 0;

            std::size_t available =
                std::numeric_limits<
                    std::size_t>::max();
        };

        HostMemoryInfo
        read_host_memory_info()
        {
            HostMemoryInfo info;

            std::ifstream input(
                "/proc/meminfo");

            if (!input)
            {
                return info;
            }

            std::string line;

            while (std::getline(
                input,
                line))
            {
                std::istringstream row(
                    line);

                std::string key;
                unsigned long long
                    value_kib = 0;

                std::string unit;

                if (!(row >>
                      key >>
                      value_kib))
                {
                    continue;
                }

                row >> unit;

                if (!unit.empty() &&
                    unit != "kB")
                {
                    continue;
                }

                const auto maximum =
                    std::numeric_limits<
                        std::size_t>::max();

                const auto bytes =
                    value_kib >
                            maximum /
                                1024ULL
                        ? maximum
                        : static_cast<
                              std::size_t>(
                              value_kib *
                              1024ULL);

                if (key ==
                    "MemTotal:")
                {
                    info.total =
                        bytes;
                }
                else if (
                    key ==
                    "MemAvailable:")
                {
                    info.available =
                        bytes;
                }
            }

            return info;
        }
    }

    std::unique_ptr<CudaFHExecutor>
    create_cuda_fh_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &omega,
        const algebra::FieldElement &alpha_quot,
        const algebra::FieldElement &beta_feat)
    {
        if (!cuda_fh_executor_available())
        {
            return nullptr;
        }

        return std::make_unique<
            CudaFHExecutorImpl>(
            domain_size,
            target_size,
            shift,
            omega,
            alpha_quot,
            beta_feat);
    }

    bool cuda_fh_executor_available()
    {
        return algebra::
            cuda_field_hotspots_available();
    }

    bool cuda_fh_executor_can_admit(
        std::size_t target_size)
    {
        if (!cuda_fh_executor_available() ||
            target_size < 2 ||
            (target_size &
             (target_size - 1U)) != 0U ||
            target_size >
                (std::size_t{1} << 27))
        {
            return false;
        }

        const auto leaf_bytes =
            checked_bytes(
                target_size,
                sizeof(GpuFr),
                "admission leaf");

        const auto persistent_bytes =
            checked_bytes(
                kFHLeafCacheSlots + 1,
                leaf_bytes,
                "persistent working set");

        const auto ntt_workspace_bytes =
            checked_bytes(
                2,
                leaf_bytes,
                "NTT workspace");

        const auto required_bytes =
            checked_add(
                persistent_bytes,
                ntt_workspace_bytes,
                "complete working set");

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status !=
            cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int integrated = 0;

        status =
            cudaDeviceGetAttribute(
                &integrated,
                cudaDevAttrIntegrated,
                device);

        if (status !=
            cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        if (integrated != 0)
        {
            const auto memory =
                read_host_memory_info();

            if (memory.available ==
                std::numeric_limits<
                    std::size_t>::max())
            {
                return false;
            }

            constexpr std::size_t
                kMinimumReserve =
                    std::size_t{16}
                    << 30;

            const auto reserve =
                std::max(
                    kMinimumReserve,
                    memory.total == 0
                        ? kMinimumReserve
                        : memory.total / 8);

            if (memory.available <=
                reserve)
            {
                return false;
            }

            return required_bytes <=
                   memory.available -
                       reserve;
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;

        status =
            cudaMemGetInfo(
                &free_bytes,
                &total_bytes);

        if (status !=
            cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        constexpr std::size_t
            kMinimumReserve =
                std::size_t{8} << 30;

        const auto reserve =
            std::max(
                kMinimumReserve,
                total_bytes / 8);

        if (free_bytes <=
            reserve)
        {
            return false;
        }

        return required_bytes <=
               free_bytes -
                   reserve;
    }

    void release_cuda_fh_executor_cache()
    {
    }

} // namespace gatzk::protocol