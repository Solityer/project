#include "gatzk/protocol/cuda_t_edge.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/cuda_ntt_device.hpp"

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/vector_ops.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using GpuFr =
            algebra::gpu::GpuFr;

        constexpr std::uint32_t
            kInvalidLeafSlot =
                std::numeric_limits<
                    std::uint32_t>::max();

        constexpr std::size_t
            kMaximumProgramStackDepth = 64;

        // The exact leaf helpers feed the paired CUDA NTT with one lane of
        // target_size / 2 elements.  Keep this capability boundary local to
        // the t_edge executor and do not conflate it with the quotient
        // program, whose pointwise kernel can still run for larger targets.
        constexpr std::size_t
            kMaximumCudaLeafNttSize =
                std::size_t{1} << 24;

        static_assert(
            sizeof(GpuFr) == sizeof(mcl::Fr));

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
                        "CUDA t_edge byte overflow: ") +
                    label);
            }

            return count * element_size;
        }

        bool checked_add_size(
            std::size_t lhs,
            std::size_t rhs,
            std::size_t *out)
        {
            if (out == nullptr ||
                lhs >
                    std::numeric_limits<std::size_t>::max() - rhs)
            {
                return false;
            }

            *out = lhs + rhs;
            return true;
        }

        bool checked_mul_size(
            std::size_t lhs,
            std::size_t rhs,
            std::size_t *out)
        {
            if (out == nullptr ||
                (lhs != 0 &&
                 rhs >
                     std::numeric_limits<std::size_t>::max() /
                         lhs))
            {
                return false;
            }

            *out = lhs * rhs;
            return true;
        }

        [[noreturn]] void
        throw_cuda_t_edge_error(
            const char *operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string("CUDA t_edge ") +
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

        __global__ void
        t_edge_term_kernel(
            const TEdgeProgramInstruction
                *instructions,
            std::size_t instruction_begin,
            std::size_t instruction_end,
            const GpuFr *constants,
            GpuFr *const *leaf_slots,
            const std::uint8_t *leaf_split_layout,
            const std::uint32_t *access_slots,
            const std::uint8_t *access_rotated,
            std::size_t rotation_step,
            std::size_t target_size,
            GpuFr alpha_scale,
            GpuFr *quotient)
        {
            const auto point =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (point >= target_size)
            {
                return;
            }

            GpuFr stack[kMaximumProgramStackDepth];

            std::size_t stack_size = 0;

            for (std::size_t index =
                     instruction_begin;
                 index < instruction_end;
                 ++index)
            {
                const auto instruction =
                    instructions[index];

                switch (instruction.opcode)
                {
                case TEdgeProgramOpcode::
                    LoadAccess:
                {
                    const auto access =
                        instruction.operand;

                    const auto slot =
                        access_slots[access];

                    const auto logical_index =
                        access_rotated[access] != 0
                            ? (point +
                               rotation_step) &
                                  (target_size - 1U)
                            : point;
                    std::size_t physical_index =
                        logical_index;

                    if (leaf_split_layout[slot] != 0)
                    {
                        const auto half_size =
                            target_size / 2;

                        physical_index =
                            logical_index / 2 +
                            ((logical_index & 1U) != 0U
                                 ? half_size
                                 : 0);
                    }

                    stack[stack_size++] =
                        leaf_slots[slot]
                                  [physical_index];

                    break;
                }

                case TEdgeProgramOpcode::
                    LoadConstant:
                {
                    stack[stack_size++] =
                        constants[instruction.operand];

                    break;
                }

                case TEdgeProgramOpcode::Add:
                {
                    const auto rhs =
                        stack[--stack_size];

                    const auto lhs =
                        stack[--stack_size];

                    stack[stack_size++] =
                        algebra::gpu::fr_add(
                            lhs,
                            rhs);

                    break;
                }

                case TEdgeProgramOpcode::
                    Subtract:
                {
                    const auto rhs =
                        stack[--stack_size];

                    const auto lhs =
                        stack[--stack_size];

                    stack[stack_size++] =
                        algebra::gpu::fr_sub(
                            lhs,
                            rhs);

                    break;
                }

                case TEdgeProgramOpcode::
                    Multiply:
                {
                    const auto rhs =
                        stack[--stack_size];

                    const auto lhs =
                        stack[--stack_size];

                    stack[stack_size++] =
                        algebra::gpu::fr_mul(
                            lhs,
                            rhs);

                    break;
                }

                case TEdgeProgramOpcode::
                    Negate:
                {
                    stack[stack_size - 1] =
                        algebra::gpu::fr_sub(
                            algebra::gpu::
                                fr_zero(),
                            stack[stack_size - 1]);

                    break;
                }
                }
            }

            const auto contribution =
                algebra::gpu::fr_mul(
                    stack[0],
                    alpha_scale);

            quotient[point] =
                algebra::gpu::fr_add(
                    quotient[point],
                    contribution);
        }

        __device__ __forceinline__ GpuFr
        t_edge_pow_u64(
            GpuFr base,
            std::size_t exponent)
        {
            auto result =
                algebra::gpu::fr_one();

            while (exponent != 0)
            {
                if ((exponent & 1U) != 0U)
                {
                    result =
                        algebra::gpu::fr_mul(
                            result,
                            base);
                }

                exponent >>= 1U;

                if (exponent != 0)
                {
                    base =
                        algebra::gpu::fr_mul(
                            base,
                            base);
                }
            }

            return result;
        }

        __device__ __forceinline__ GpuFr
        t_edge_eval_blinder(
            const GpuFr *coefficients,
            std::size_t offset,
            std::size_t count,
            GpuFr point)
        {
            auto value =
                algebra::gpu::fr_zero();

            while (count != 0)
            {
                --count;
                value =
                    algebra::gpu::fr_add(
                        algebra::gpu::fr_mul(
                            value,
                            point),
                        coefficients[offset + count]);
            }

            return value;
        }

        __global__ void
        t_edge_materialize_linear_kernel(
            GpuFr *const *leaf_slots,
            const std::uint8_t *leaf_split_layout,
            const std::uint32_t *source_slots,
            const GpuFr *source_scales,
            const GpuFr *source_blinders,
            const std::size_t *source_blinder_offsets,
            const std::size_t *source_blinder_counts,
            std::size_t source_count,
            const GpuFr *target_blinder,
            std::size_t target_blinder_count,
            std::size_t domain_size,
            std::size_t target_size,
            GpuFr shift,
            GpuFr root,
            GpuFr shift_domain_power,
            GpuFr zero_step,
            std::uint8_t output_split_layout,
            GpuFr *target)
        {
            const auto logical_index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (logical_index >= target_size)
            {
                return;
            }

            const auto half_size =
                target_size / 2U;
            const auto output_index =
                output_split_layout != 0U
                    ? logical_index / 2U +
                          ((logical_index & 1U) != 0U
                               ? half_size
                               : 0U)
                    : logical_index;

            auto value =
                algebra::gpu::fr_zero();

            auto source_blinder_sum =
                algebra::gpu::fr_zero();

            const auto point =
                algebra::gpu::fr_mul(
                    shift,
                    t_edge_pow_u64(
                        root,
                        logical_index));

            for (std::size_t source_index = 0;
                 source_index < source_count;
                 ++source_index)
            {
                const auto slot =
                    source_slots[source_index];

                const auto physical_index =
                    leaf_split_layout[slot] != 0U
                        ? logical_index / 2U +
                              ((logical_index & 1U) != 0U
                                   ? half_size
                                   : 0U)
                        : logical_index;

                value =
                    algebra::gpu::fr_add(
                        value,
                        algebra::gpu::fr_mul(
                            leaf_slots[slot]
                                      [physical_index],
                            source_scales[source_index]));

                const auto blinder_count =
                    source_blinder_counts[source_index];

                if (blinder_count != 0)
                {
                    const auto blinder_value =
                        t_edge_eval_blinder(
                            source_blinders,
                            source_blinder_offsets[
                                source_index],
                            blinder_count,
                            point);

                    source_blinder_sum =
                        algebra::gpu::fr_add(
                            source_blinder_sum,
                            algebra::gpu::fr_mul(
                                blinder_value,
                                source_scales[
                                    source_index]));
                }
            }

            const auto target_blinder_value =
                target_blinder_count == 0
                    ? algebra::gpu::fr_zero()
                    : t_edge_eval_blinder(
                          target_blinder,
                          0,
                          target_blinder_count,
                          point);

            const auto correction =
                algebra::gpu::fr_sub(
                    target_blinder_value,
                    source_blinder_sum);

            const auto zero_domain_value =
                algebra::gpu::fr_sub(
                    algebra::gpu::fr_mul(
                        shift_domain_power,
                        t_edge_pow_u64(
                            zero_step,
                            logical_index)),
                    algebra::gpu::fr_one());

            target[output_index] =
                algebra::gpu::fr_add(
                    value,
                    algebra::gpu::fr_mul(
                        zero_domain_value,
                        correction));
        }

        __global__ void
        t_edge_finalize_kernel(
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

            quotient[index] = algebra::gpu::fr_mul(quotient[index], zero_inverse_cycle[index % cycle_size]);
        }

        class CudaTEdgeExecutorImpl final
            : public CudaTEdgeExecutor
        {
        public:
            CudaTEdgeExecutorImpl(
                std::size_t domain_size,
                std::size_t target_size,
                const algebra::FieldElement &shift,
                const algebra::FieldElement &omega,
                std::size_t cache_slot_count)
                : domain_size_(domain_size),
                  target_size_(target_size),
                  shift_(shift),
                  rotation_step_(
                      target_size /
                      domain_size),
                  cache_slot_count_(
                      cache_slot_count)
            {
                (void)omega;

                if (domain_size_ == 0 ||
                    target_size_ == 0 ||
                    (domain_size_ &
                     (domain_size_ - 1U)) != 0 ||
                    target_size_ %
                            domain_size_ !=
                        0 ||
                    (target_size_ &
                     (target_size_ - 1U)) !=
                        0 ||
                    cache_slot_count_ == 0)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge "
                        "executor shape");
                }

                if (cache_slot_count_ >
                    std::numeric_limits<
                        std::uint32_t>::max())
                {
                    throw std::runtime_error(
                        "CUDA t_edge leaf-cache "
                        "slot count overflow");
                }

                host_leaf_split_layout_.assign(
                    cache_slot_count_,
                    0);

                host_leaf_bound_.assign(
                    cache_slot_count_,
                    false);

                host_leaf_uploaded_prefix_.assign(
                    cache_slot_count_,
                    0);

                try
                {
                    const auto leaf_bytes =
                        checked_bytes(
                            target_size_,
                            sizeof(GpuFr),
                            "leaf");

                    device_leaf_slots_.assign(
                        cache_slot_count_,
                        nullptr);

                    for (auto &slot :
                         device_leaf_slots_)
                    {
                        auto status =
                            cudaMalloc(
                                reinterpret_cast<
                                    void **>(
                                    &slot),
                                leaf_bytes);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_cuda_t_edge_error(
                                "leaf allocation",
                                status);
                        }
                    }

                    auto status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_leaf_table_),
                            checked_bytes(
                                cache_slot_count_,
                                sizeof(GpuFr *),
                                "leaf table"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "leaf-table allocation",
                            status);
                    }

                    status =
                        cudaMemcpy(
                            device_leaf_table_,
                            device_leaf_slots_
                                .data(),
                            checked_bytes(
                                cache_slot_count_,
                                sizeof(GpuFr *),
                                "leaf table"),
                            cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "leaf-table upload",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_leaf_split_layout_),
                            checked_bytes(
                                cache_slot_count_,
                                sizeof(std::uint8_t),
                                "leaf layout"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "leaf-layout allocation",
                            status);
                    }

                    status =
                        cudaMemset(
                            device_leaf_split_layout_,
                            0,
                            checked_bytes(
                                cache_slot_count_,
                                sizeof(std::uint8_t),
                                "leaf layout"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "leaf-layout clear",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_quotient_),
                            leaf_bytes);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "quotient allocation",
                            status);
                    }

                    status =
                        cudaMemset(
                            device_quotient_,
                            0,
                            leaf_bytes);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
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
                        root.pow(domain_size_);

                    auto zero_power =
                        shift.pow(domain_size_);

                    const auto one =
                        algebra::FieldElement::
                            one();

                    for (std::size_t index = 0;
                         index < cycle_size;
                         ++index)
                    {
                        const auto denominator =
                            zero_power - one;

                        if (denominator.is_zero())
                        {
                            throw std::runtime_error(
                                "CUDA t_edge coset "
                                "overlaps quotient "
                                "domain");
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

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
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

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
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

            ~CudaTEdgeExecutorImpl()
                override
            {
                release_all_noexcept();
            }

            std::size_t cache_slot_count()
                const override
            {
                return cache_slot_count_;
            }

            void prepare_program(
                const TEdgeBoundProgram &program)
                override
            {
                release_program_noexcept();

                if (program.instructions.empty() ||
                    program.terms.empty() ||
                    program.accesses.empty())
                {
                    throw std::runtime_error(
                        "CUDA t_edge received "
                        "an empty program");
                }

                if (program.maximum_stack_depth >
                    kMaximumProgramStackDepth)
                {
                    throw std::runtime_error(
                        "CUDA t_edge program stack "
                        "exceeds executor limit");
                }

                program_ = &program;

                try
                {
                    auto status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_instructions_),
                            checked_bytes(
                                program.instructions
                                    .size(),
                                sizeof(
                                    TEdgeProgramInstruction),
                                "instructions"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "instruction allocation",
                            status);
                    }

                    status =
                        cudaMemcpy(
                            device_instructions_,
                            program.instructions.data(),
                            checked_bytes(
                                program.instructions
                                    .size(),
                                sizeof(
                                    TEdgeProgramInstruction),
                                "instructions"),
                            cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "instruction upload",
                            status);
                    }

                    if (!program.constants.empty())
                    {
                        status =
                            cudaMalloc(
                                reinterpret_cast<
                                    void **>(
                                    &device_constants_),
                                checked_bytes(
                                    program.constants
                                        .size(),
                                    sizeof(GpuFr),
                                    "constants"));

                        if (status !=
                            cudaSuccess)
                        {
                            throw_cuda_t_edge_error(
                                "constant allocation",
                                status);
                        }

                        status =
                            cudaMemcpy(
                                device_constants_,
                                algebra::FieldElement::
                                    native_array_data(
                                        program.constants
                                            .data()),
                                checked_bytes(
                                    program.constants
                                        .size(),
                                    sizeof(GpuFr),
                                    "constants"),
                                cudaMemcpyHostToDevice);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_cuda_t_edge_error(
                                "constant upload",
                                status);
                        }
                    }

                    std::vector<std::uint8_t>
                        rotated(
                            program.accesses
                                .size(),
                            0);

                    for (std::size_t index = 0;
                         index <
                         program.accesses.size();
                         ++index)
                    {
                        rotated[index] =
                            program.accesses[index]
                                    .rotated
                                ? 1U
                                : 0U;
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_access_rotated_),
                            checked_bytes(
                                rotated.size(),
                                sizeof(std::uint8_t),
                                "access rotation"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "access-rotation "
                            "allocation",
                            status);
                    }

                    status =
                        cudaMemcpy(
                            device_access_rotated_,
                            rotated.data(),
                            rotated.size(),
                            cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "access-rotation upload",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_access_slots_),
                            checked_bytes(
                                program.accesses
                                    .size(),
                                sizeof(std::uint32_t),
                                "access slots"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "access-slot allocation",
                            status);
                    }
                }
                catch (...)
                {
                    release_program_noexcept();
                    throw;
                }
            }

            bool materialize_evaluation_leaf(
                std::size_t slot,
                const mcl::Fr *values,
                std::size_t value_count,
                const mcl::Fr *zero_domain_blinder,
                std::size_t blinder_count)
                override
            {
                if (slot >=
                    cache_slot_count_)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge evaluation leaf slot");
                }

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

                if (!executed)
                {
                    return false;
                }

                host_leaf_split_layout_[slot] =
                    1;

                const auto status =
                    cudaMemcpy(
                        device_leaf_split_layout_ +
                            slot,
                        &host_leaf_split_layout_[slot],
                        sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "leaf-layout upload",
                        status);
                }
                algebra::record_cuda_h2d(sizeof(std::uint8_t));
                host_leaf_bound_[slot] = true;
                host_leaf_uploaded_prefix_[slot] = target_size_;
                return true;
            }

            bool materialize_coefficient_leaf(
                std::size_t slot,
                const mcl::Fr *coefficients,
                std::size_t coefficient_count)
                override
            {
                if (slot >=
                    cache_slot_count_)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge coefficient leaf slot");
                }

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

                if (!executed)
                {
                    return false;
                }

                host_leaf_split_layout_[slot] =
                    1;

                const auto status =
                    cudaMemcpy(
                        device_leaf_split_layout_ +
                            slot,
                        &host_leaf_split_layout_[slot],
                        sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "leaf-layout upload",
                        status);
                }
                algebra::record_cuda_h2d(sizeof(std::uint8_t));
                host_leaf_bound_[slot] = true;
                host_leaf_uploaded_prefix_[slot] = target_size_;
                return true;
            }

            bool materialize_linear_leaf(
                std::size_t target_slot,
                const std::vector<CudaTEdgeLinearSource>
                    &sources,
                const mcl::Fr *target_zero_domain_blinder,
                std::size_t target_blinder_count,
                CudaTEdgeLeafLayout output_layout)
                override
            {
                if (target_slot >= cache_slot_count_ ||
                    target_size_ < 2 ||
                    sources.empty())
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge linear target");
                }

                if (output_layout !=
                        CudaTEdgeLeafLayout::Split &&
                    output_layout !=
                        CudaTEdgeLeafLayout::Contiguous)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge linear output layout");
                }

                if (host_leaf_bound_[target_slot])
                {
                    throw std::runtime_error(
                        "CUDA t_edge linear target slot is already bound");
                }

                if (host_leaf_uploaded_prefix_[target_slot] != 0)
                {
                    throw std::runtime_error(
                        "CUDA t_edge linear target slot has an incomplete upload");
                }

                if (target_blinder_count != 0 &&
                    target_zero_domain_blinder == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA t_edge linear target blinder is null");
                }

                const auto maximum_blinder_count =
                    target_size_ - domain_size_;

                if (target_blinder_count > maximum_blinder_count)
                {
                    throw std::runtime_error(
                        "CUDA t_edge linear target blinder exceeds shape");
                }

                std::vector<std::uint8_t> seen(
                    cache_slot_count_,
                    0);
                std::vector<std::uint32_t> source_slots;
                std::vector<GpuFr> source_scales;
                std::vector<std::size_t> source_blinder_offsets;
                std::vector<std::size_t> source_blinder_counts;

                source_slots.reserve(sources.size());
                source_scales.reserve(sources.size());
                source_blinder_offsets.reserve(sources.size());
                source_blinder_counts.reserve(sources.size());

                std::size_t packed_blinder_count = 0;
                std::vector<GpuFr> host_source_blinders;

                for (const auto &source : sources)
                {
                    if (source.slot >= cache_slot_count_ ||
                        source.slot == target_slot ||
                        seen[source.slot] != 0 ||
                        !host_leaf_bound_[source.slot])
                    {
                        throw std::runtime_error(
                            "CUDA t_edge linear source slot is missing or duplicated");
                    }

                    if (source.scale.is_zero())
                    {
                        throw std::runtime_error(
                            "CUDA t_edge linear source scale is zero");
                    }

                    if (source.blinder_count >
                        maximum_blinder_count)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge linear source blinder exceeds shape");
                    }

                    if (source.blinder_count != 0 &&
                        source.zero_domain_blinder == nullptr)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge linear source blinder is null");
                    }

                    seen[source.slot] = 1;
                    source_slots.push_back(source.slot);
                    source_scales.push_back(
                        gpu_field(source.scale));
                    source_blinder_offsets.push_back(
                        packed_blinder_count);
                    source_blinder_counts.push_back(
                        source.blinder_count);

                    std::size_t next_count = 0;
                    if (!checked_add_size(
                            packed_blinder_count,
                            source.blinder_count,
                            &next_count))
                    {
                        throw std::runtime_error(
                            "CUDA t_edge linear blinder count overflow");
                    }

                    if (next_count != packed_blinder_count)
                    {
                        host_source_blinders.resize(
                            next_count);

                        if (source.blinder_count != 0)
                        {
                            std::memcpy(
                                host_source_blinders.data() +
                                    packed_blinder_count,
                                source.zero_domain_blinder,
                                checked_bytes(
                                    source.blinder_count,
                                    sizeof(GpuFr),
                                    "linear source blinder"));
                        }
                    }

                    packed_blinder_count = next_count;
                }

                std::vector<GpuFr> host_target_blinder(
                    target_blinder_count);

                if (target_blinder_count != 0)
                {
                    std::memcpy(
                        host_target_blinder.data(),
                        target_zero_domain_blinder,
                        checked_bytes(
                            target_blinder_count,
                            sizeof(GpuFr),
                            "linear target blinder"));
                }

                ensure_linear_buffers(
                    source_slots.size(),
                    packed_blinder_count,
                    target_blinder_count);

                auto status = cudaMemcpy(
                    device_linear_source_slots_,
                    source_slots.data(),
                    checked_bytes(
                        source_slots.size(),
                        sizeof(std::uint32_t),
                        "linear source slots"),
                    cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear source-slot upload",
                        status);
                }

                algebra::record_cuda_h2d(
                    checked_bytes(
                        source_slots.size(),
                        sizeof(std::uint32_t),
                        "linear source slots"));

                status = cudaMemcpy(
                    device_linear_source_scales_,
                    source_scales.data(),
                    checked_bytes(
                        source_scales.size(),
                        sizeof(GpuFr),
                        "linear source scales"),
                    cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear source-scale upload",
                        status);
                }

                algebra::record_cuda_h2d(
                    checked_bytes(
                        source_scales.size(),
                        sizeof(GpuFr),
                        "linear source scales"));

                status = cudaMemcpy(
                    device_linear_source_blinder_offsets_,
                    source_blinder_offsets.data(),
                    checked_bytes(
                        source_blinder_offsets.size(),
                        sizeof(std::size_t),
                        "linear blinder offsets"),
                    cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear blinder-offset upload",
                        status);
                }

                algebra::record_cuda_h2d(
                    checked_bytes(
                        source_blinder_offsets.size(),
                        sizeof(std::size_t),
                        "linear blinder offsets"));

                status = cudaMemcpy(
                    device_linear_source_blinder_counts_,
                    source_blinder_counts.data(),
                    checked_bytes(
                        source_blinder_counts.size(),
                        sizeof(std::size_t),
                        "linear blinder counts"),
                    cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear blinder-count upload",
                        status);
                }

                algebra::record_cuda_h2d(
                    checked_bytes(
                        source_blinder_counts.size(),
                        sizeof(std::size_t),
                        "linear blinder counts"));

                if (packed_blinder_count != 0)
                {
                    status = cudaMemcpy(
                        device_linear_source_blinders_,
                        host_source_blinders.data(),
                        checked_bytes(
                            packed_blinder_count,
                            sizeof(GpuFr),
                            "linear source blinders"),
                        cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "linear source-blinder upload",
                            status);
                    }

                    algebra::record_cuda_h2d(
                        checked_bytes(
                            packed_blinder_count,
                            sizeof(GpuFr),
                            "linear source blinders"));
                }

                if (target_blinder_count != 0)
                {
                    status = cudaMemcpy(
                        device_linear_target_blinder_,
                        host_target_blinder.data(),
                        checked_bytes(
                            target_blinder_count,
                            sizeof(GpuFr),
                            "linear target blinder"),
                        cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "linear target-blinder upload",
                            status);
                    }

                    algebra::record_cuda_h2d(
                        checked_bytes(
                            target_blinder_count,
                            sizeof(GpuFr),
                            "linear target blinder"));
                }

                constexpr unsigned kBlockSize = 256;
                if (target_size_ >
                    std::numeric_limits<std::size_t>::max() -
                        (kBlockSize - 1U))
                {
                    throw std::runtime_error(
                        "CUDA t_edge linear grid size overflow");
                }
                const auto block_count_size =
                    (target_size_ + kBlockSize - 1U) /
                    kBlockSize;

                if (block_count_size >
                    std::numeric_limits<unsigned>::max())
                {
                    throw std::runtime_error(
                        "CUDA t_edge linear grid overflow");
                }

                const auto root =
                    algebra::FieldElement::root_of_unity(
                        target_size_);
                const auto zero_step =
                    root.pow(domain_size_);
                const auto shift_domain_power =
                    shift_.pow(domain_size_);

                const auto output_split_layout =
                    output_layout == CudaTEdgeLeafLayout::Split
                        ? std::uint8_t{1}
                        : std::uint8_t{0};

                t_edge_materialize_linear_kernel<<<
                    static_cast<unsigned>(block_count_size),
                    kBlockSize>>>(
                    device_leaf_table_,
                    device_leaf_split_layout_,
                    device_linear_source_slots_,
                    device_linear_source_scales_,
                    device_linear_source_blinders_,
                    device_linear_source_blinder_offsets_,
                    device_linear_source_blinder_counts_,
                    source_slots.size(),
                    device_linear_target_blinder_,
                    target_blinder_count,
                    domain_size_,
                    target_size_,
                    gpu_field(shift_),
                    gpu_field(root),
                    gpu_field(shift_domain_power),
                    gpu_field(zero_step),
                    output_split_layout,
                    device_leaf_slots_[target_slot]);

                status = cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear materialization launch",
                        status);
                }

                status = cudaStreamSynchronize(0);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear materialization synchronization",
                        status);
                }

                algebra::record_cuda_stream_sync();
                algebra::record_cuda_kernel_invocations(1);

                host_leaf_split_layout_[target_slot] =
                    output_split_layout;

                status = cudaMemcpy(
                    device_leaf_split_layout_ + target_slot,
                    &host_leaf_split_layout_[target_slot],
                    sizeof(std::uint8_t),
                    cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear target-layout upload",
                        status);
                }

                algebra::record_cuda_h2d(sizeof(std::uint8_t));
                host_leaf_bound_[target_slot] = true;
                return true;
            }

            void upload_leaf_chunk(
                std::size_t slot,
                std::size_t logical_offset,
                const mcl::Fr *values,
                std::size_t count)
                override
            {
                if (slot >=
                        device_leaf_slots_.size() ||
                    values == nullptr ||
                    logical_offset >
                        target_size_ ||
                    count >
                        target_size_ -
                            logical_offset)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge "
                        "leaf upload range");
                }

                if (count == 0)
                {
                    return;
                }

                if (logical_offset == 0)
                {
                    host_leaf_split_layout_[slot] =
                        0;

                    host_leaf_bound_[slot] = false;
                    host_leaf_uploaded_prefix_[slot] = 0;

                    const auto layout_status =
                        cudaMemcpy(
                            device_leaf_split_layout_ +
                                slot,
                            &host_leaf_split_layout_[slot],
                            sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice);

                    if (layout_status !=
                        cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "fallback leaf-layout upload",
                            layout_status);
                    }

                    algebra::record_cuda_h2d(sizeof(std::uint8_t));
                }
                else if (logical_offset !=
                         host_leaf_uploaded_prefix_[slot])
                {
                    throw std::runtime_error(
                        "CUDA t_edge leaf upload is not contiguous");
                }

                const auto byte_count =
                    checked_bytes(
                        count,
                        sizeof(GpuFr),
                        "leaf upload");

                const auto status =
                    cudaMemcpy(
                        device_leaf_slots_[slot] +
                            logical_offset,
                        values,
                        byte_count,
                        cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "leaf upload",
                        status);
                }

                algebra::record_cuda_h2d(byte_count);

                host_leaf_uploaded_prefix_[slot] =
                    logical_offset + count;
                host_leaf_bound_[slot] =
                    host_leaf_uploaded_prefix_[slot] ==
                    target_size_;
            }

            void accumulate_term(
                std::size_t term_index,
                const std::vector<
                    std::uint32_t>
                    &access_slots)
                override
            {
                if (program_ == nullptr ||
                    term_index >=
                        program_->terms.size() ||
                    access_slots.size() !=
                        program_->accesses.size())
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge "
                        "term invocation");
                }

                const auto &term =
                    program_->terms[term_index];

                for (const auto access :
                     term.required_accesses)
                {
                    if (access >=
                            access_slots.size() ||
                        access_slots[access] ==
                            kInvalidLeafSlot ||
                        access_slots[access] >=
                            cache_slot_count_ ||
                        !host_leaf_bound_[
                            access_slots[access]])
                    {
                        throw std::runtime_error(
                            "CUDA t_edge term has "
                            "an unbound leaf");
                    }
                }

                auto status =
                    cudaMemcpyAsync(
                        device_access_slots_,
                        access_slots.data(),
                        checked_bytes(
                            access_slots.size(),
                            sizeof(std::uint32_t),
                            "access slots"),
                        cudaMemcpyHostToDevice,
                        0);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "access-slot upload",
                        status);
                }

                constexpr unsigned
                    kBlockSize = 256;

                const auto block_count_size =
                    (target_size_ +
                     kBlockSize - 1) /
                    kBlockSize;

                if (block_count_size >
                    std::numeric_limits<
                        unsigned>::max())
                {
                    throw std::runtime_error(
                        "CUDA t_edge grid "
                        "overflow");
                }

                const auto alpha_scale =
                    gpu_field(
                        term.alpha_scale);

                t_edge_term_kernel<<<
                    static_cast<unsigned>(
                        block_count_size),
                    kBlockSize>>>(
                    device_instructions_,
                    term.instruction_begin,
                    term.instruction_end,
                    device_constants_,
                    device_leaf_table_,
                    device_leaf_split_layout_,
                    device_access_slots_,
                    device_access_rotated_,
                    rotation_step_,
                    target_size_,
                    alpha_scale,
                    device_quotient_);

                status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "term kernel launch",
                        status);
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);
            }

            void release_leaf(
                std::size_t slot)
                override
            {
                if (slot >=
                    cache_slot_count_)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge "
                        "leaf release");
                }

                host_leaf_bound_[slot] = false;
                host_leaf_split_layout_[slot] = 0;
                host_leaf_uploaded_prefix_[slot] = 0;
            }

            void finalize(
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
                        "invalid CUDA t_edge "
                        "final output");
                }

                constexpr unsigned
                    kBlockSize = 256;

                const auto block_count_size =
                    (target_size_ +
                     kBlockSize - 1) /
                    kBlockSize;

                t_edge_finalize_kernel<<<
                    static_cast<unsigned>(
                        block_count_size),
                    kBlockSize>>>(
                    device_quotient_,
                    device_zero_inverse_cycle_,
                    target_size_,
                    zero_inverse_cycle_size_);

                auto status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "finalize kernel launch",
                        status);
                }

                const auto byte_count =
                    checked_bytes(
                        target_size_,
                        sizeof(GpuFr),
                        "quotient download");

                status =
                    cudaMemcpy(
                        algebra::FieldElement::
                            mutable_native_array_data(
                                output),
                        device_quotient_,
                        byte_count,
                        cudaMemcpyDeviceToHost);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "quotient download",
                        status);
                }

                algebra::record_cuda_d2h(byte_count);

                algebra::
                    record_cuda_kernel_invocations(
                        1);
            }

            void finalize_coefficients(
                algebra::FieldElement *output,
                std::size_t output_count)
                override
            {
                if (output == nullptr ||
                    output_count != target_size_ ||
                    zero_inverse_cycle_size_ == 0)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge coefficient output");
                }

                constexpr unsigned
                    kBlockSize = 256;

                const auto block_count_size =
                    (target_size_ +
                     kBlockSize - 1) /
                    kBlockSize;

                t_edge_finalize_kernel<<<
                    static_cast<unsigned>(
                        block_count_size),
                    kBlockSize>>>(
                    device_quotient_,
                    device_zero_inverse_cycle_,
                    target_size_,
                    zero_inverse_cycle_size_);

                auto status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "finalize kernel launch",
                        status);
                }

                status =
                    cudaStreamSynchronize(0);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
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
                        "CUDA t_edge device quotient INTT was not admitted");
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);
            }

        private:
            void ensure_linear_buffers(
                std::size_t source_count,
                std::size_t source_blinder_count,
                std::size_t target_blinder_count)
            {
                const auto ensure =
                    [&](void **pointer,
                        std::size_t *capacity,
                        std::size_t required_bytes,
                        const char *label)
                {
                    if (required_bytes == 0 ||
                        (*pointer != nullptr &&
                         *capacity >= required_bytes))
                    {
                        return;
                    }

                    if (*pointer != nullptr)
                    {
                        (void)cudaFree(*pointer);
                        *pointer = nullptr;
                        *capacity = 0;
                    }

                    const auto status =
                        cudaMalloc(
                            pointer,
                            required_bytes);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            label,
                            status);
                    }

                    *capacity = required_bytes;
                };

                try
                {
                    ensure(
                        reinterpret_cast<void **>(
                            &device_linear_source_slots_),
                        &device_linear_source_slots_capacity_bytes_,
                        checked_bytes(
                            source_count,
                            sizeof(std::uint32_t),
                            "linear source slots"),
                        "linear source-slot allocation");

                    ensure(
                        reinterpret_cast<void **>(
                            &device_linear_source_scales_),
                        &device_linear_source_scales_capacity_bytes_,
                        checked_bytes(
                            source_count,
                            sizeof(GpuFr),
                            "linear source scales"),
                        "linear source-scale allocation");

                    ensure(
                        reinterpret_cast<void **>(
                            &device_linear_source_blinder_offsets_),
                        &device_linear_source_blinder_offsets_capacity_bytes_,
                        checked_bytes(
                            source_count,
                            sizeof(std::size_t),
                            "linear blinder offsets"),
                        "linear blinder-offset allocation");

                    ensure(
                        reinterpret_cast<void **>(
                            &device_linear_source_blinder_counts_),
                        &device_linear_source_blinder_counts_capacity_bytes_,
                        checked_bytes(
                            source_count,
                            sizeof(std::size_t),
                            "linear blinder counts"),
                        "linear blinder-count allocation");

                    ensure(
                        reinterpret_cast<void **>(
                            &device_linear_source_blinders_),
                        &device_linear_source_blinders_capacity_bytes_,
                        checked_bytes(
                            source_blinder_count,
                            sizeof(GpuFr),
                            "linear source blinders"),
                        "linear source-blinder allocation");

                    ensure(
                        reinterpret_cast<void **>(
                            &device_linear_target_blinder_),
                        &device_linear_target_blinder_capacity_bytes_,
                        checked_bytes(
                            target_blinder_count,
                            sizeof(GpuFr),
                            "linear target blinder"),
                        "linear target-blinder allocation");
                }
                catch (...)
                {
                    release_linear_buffers_noexcept();
                    throw;
                }
            }

            void release_linear_buffers_noexcept() noexcept
            {
                if (device_linear_source_slots_ != nullptr)
                {
                    (void)cudaFree(device_linear_source_slots_);
                }

                if (device_linear_source_scales_ != nullptr)
                {
                    (void)cudaFree(device_linear_source_scales_);
                }

                if (device_linear_source_blinder_offsets_ != nullptr)
                {
                    (void)cudaFree(
                        device_linear_source_blinder_offsets_);
                }

                if (device_linear_source_blinder_counts_ != nullptr)
                {
                    (void)cudaFree(
                        device_linear_source_blinder_counts_);
                }

                if (device_linear_source_blinders_ != nullptr)
                {
                    (void)cudaFree(
                        device_linear_source_blinders_);
                }

                if (device_linear_target_blinder_ != nullptr)
                {
                    (void)cudaFree(
                        device_linear_target_blinder_);
                }

                device_linear_source_slots_ = nullptr;
                device_linear_source_scales_ = nullptr;
                device_linear_source_blinder_offsets_ = nullptr;
                device_linear_source_blinder_counts_ = nullptr;
                device_linear_source_blinders_ = nullptr;
                device_linear_target_blinder_ = nullptr;

                device_linear_source_slots_capacity_bytes_ = 0;
                device_linear_source_scales_capacity_bytes_ = 0;
                device_linear_source_blinder_offsets_capacity_bytes_ = 0;
                device_linear_source_blinder_counts_capacity_bytes_ = 0;
                device_linear_source_blinders_capacity_bytes_ = 0;
                device_linear_target_blinder_capacity_bytes_ = 0;
            }

            void release_program_noexcept() noexcept
            {
                if (device_instructions_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_instructions_);
                }

                if (device_constants_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_constants_);
                }

                if (device_access_rotated_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_access_rotated_);
                }

                if (device_access_slots_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_access_slots_);
                }

                device_instructions_ =
                    nullptr;

                device_constants_ =
                    nullptr;

                device_access_rotated_ =
                    nullptr;

                device_access_slots_ =
                    nullptr;

                program_ = nullptr;
            }

            void release_all_noexcept() noexcept
            {
                release_program_noexcept();
                release_linear_buffers_noexcept();

                for (auto *slot :
                     device_leaf_slots_)
                {
                    if (slot != nullptr)
                    {
                        (void)cudaFree(slot);
                    }
                }

                device_leaf_slots_.clear();

                if (device_leaf_table_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_leaf_table_);
                }

                if (device_leaf_split_layout_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_leaf_split_layout_);
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

                device_leaf_table_ =
                    nullptr;

                device_leaf_split_layout_ =
                    nullptr;

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
            std::size_t cache_slot_count_ = 0;

            std::vector<GpuFr *>
                device_leaf_slots_;

            GpuFr **device_leaf_table_ =
                nullptr;

            std::vector<std::uint8_t>
                host_leaf_split_layout_;

            std::vector<bool>
                host_leaf_bound_;

            std::vector<std::size_t>
                host_leaf_uploaded_prefix_;

            std::uint8_t
                *device_leaf_split_layout_ =
                    nullptr;

            GpuFr *device_quotient_ =
                nullptr;

            GpuFr *device_zero_inverse_cycle_ =
                nullptr;

            std::uint32_t *device_linear_source_slots_ =
                nullptr;

            GpuFr *device_linear_source_scales_ =
                nullptr;

            GpuFr *device_linear_source_blinders_ =
                nullptr;

            std::size_t *device_linear_source_blinder_offsets_ =
                nullptr;

            std::size_t *device_linear_source_blinder_counts_ =
                nullptr;

            GpuFr *device_linear_target_blinder_ =
                nullptr;

            std::size_t
                device_linear_source_slots_capacity_bytes_ = 0;

            std::size_t
                device_linear_source_scales_capacity_bytes_ = 0;

            std::size_t
                device_linear_source_blinder_offsets_capacity_bytes_ = 0;

            std::size_t
                device_linear_source_blinder_counts_capacity_bytes_ = 0;

            std::size_t
                device_linear_source_blinders_capacity_bytes_ = 0;

            std::size_t
                device_linear_target_blinder_capacity_bytes_ = 0;

            std::size_t
                zero_inverse_cycle_size_ = 0;

            TEdgeProgramInstruction
                *device_instructions_ =
                    nullptr;

            GpuFr *device_constants_ =
                nullptr;

            std::uint8_t
                *device_access_rotated_ =
                    nullptr;

            std::uint32_t
                *device_access_slots_ =
                    nullptr;

            const TEdgeBoundProgram
                *program_ = nullptr;
        };
    }

    std::unique_ptr<CudaTEdgeExecutor>
    create_cuda_t_edge_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &omega,
        std::size_t cache_slot_count)
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
            cache_slot_count);
    }

    bool cuda_t_edge_executor_available()
    {
        return algebra::
            cuda_field_hotspots_available();
    }

    bool cuda_t_edge_leaf_exact_ntt_supported(
        std::size_t target_size)
    {
        return target_size >= 2 &&
               (target_size & (target_size - 1U)) == 0U &&
               target_size / 2 <=
                   kMaximumCudaLeafNttSize;
    }

    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count,
        bool leaf_exact_ntt_supported)
    {
        if (!cuda_t_edge_executor_available() ||
            target_size == 0 ||
            cache_slot_count == 0 ||
            (target_size &
             (target_size - 1U)) != 0U ||
            cache_slot_count >
                std::numeric_limits<
                    std::uint32_t>::max())
        {
            return false;
        }

        std::size_t leaf_bytes = 0;
        if (!checked_mul_size(
                target_size,
                sizeof(GpuFr),
                &leaf_bytes))
        {
            return false;
        }

        std::size_t leaf_cache_bytes = 0;
        if (!checked_mul_size(
                cache_slot_count,
                leaf_bytes,
                &leaf_cache_bytes))
        {
            return false;
        }

        std::size_t ntt_workspace_bytes = 0;
        if (leaf_exact_ntt_supported &&
            !checked_mul_size(
                leaf_bytes,
                2,
                &ntt_workspace_bytes))
        {
            return false;
        }

        // ICICLE is initialized lazily by the first CUDA NTT call.  The
        // pinned generic-domain implementation allocates one max-domain
        // managed twiddle vector plus small fixed tables.  Account for the
        // vector here so admission cannot succeed before that allocation and
        // then fail after source materialization starts.
        constexpr std::size_t kMaximumCudaNttSize =
            std::size_t{1} << 24;
        std::size_t icicle_domain_bytes = 0;
        if (!checked_mul_size(
                kMaximumCudaNttSize + 1,
                sizeof(GpuFr),
                &icicle_domain_bytes))
        {
            return false;
        }

        // The domain's small internal/basic tables and executor-side CUDA
        // bookkeeping are bounded independent of the graph workload.  Keep a
        // fixed conservative allowance rather than silently omitting them.
        constexpr std::size_t kFixedMetadataBytes =
            std::size_t{1} << 20;

        std::size_t table_bytes = 0;
        if (!checked_mul_size(
                cache_slot_count,
                sizeof(GpuFr *),
                &table_bytes))
        {
            return false;
        }

        std::size_t layout_bytes = 0;
        if (!checked_mul_size(
                cache_slot_count,
                sizeof(std::uint8_t),
                &layout_bytes))
        {
            return false;
        }

        std::size_t required_bytes = 0;
        const auto add_required =
            [&required_bytes](std::size_t value)
        {
            return checked_add_size(
                required_bytes,
                value,
                &required_bytes);
        };

        // Leaf slots, quotient, the most conservative possible inverse-cycle
        // vector (domain size >= 1), leaf pointer/layout tables, ICICLE domain
        // storage and fixed metadata are all live at the same point in the
        // production route.  The two target-sized NTT buffers are live only
        // when exact direct leaf materialization is actually supported.
        if (!add_required(leaf_cache_bytes) ||
            !add_required(leaf_bytes) ||
            (leaf_exact_ntt_supported &&
             !add_required(ntt_workspace_bytes)) ||
            !add_required(leaf_bytes) ||
            !add_required(table_bytes) ||
            !add_required(layout_bytes) ||
            !add_required(icicle_domain_bytes) ||
            !add_required(kFixedMetadataBytes))
        {
            return false;
        }

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
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

        if (status != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        if (integrated != 0)
        {
            std::size_t host_total_bytes = 0;

            std::size_t host_available_bytes =
                std::numeric_limits<
                    std::size_t>::max();

            std::ifstream input(
                "/proc/meminfo");

            if (input)
            {
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
                        host_total_bytes =
                            bytes;
                    }
                    else if (
                        key ==
                        "MemAvailable:")
                    {
                        host_available_bytes =
                            bytes;
                    }
                }
            }

            if (host_available_bytes !=
                std::numeric_limits<
                    std::size_t>::max())
            {
                constexpr std::size_t
                    kMinimumIntegratedReserve =
                        std::size_t{8}
                        << 30;

                constexpr std::size_t
                    kIntegratedTransformHeadroom =
                        std::size_t{3}
                        << 30;

                const auto reserve_bytes =
                    std::max(
                        kMinimumIntegratedReserve,
                        host_total_bytes == 0
                            ? kMinimumIntegratedReserve
                            : host_total_bytes /
                                  16);

                std::size_t minimum_available_bytes = 0;
                if (!checked_add_size(
                        reserve_bytes,
                        kIntegratedTransformHeadroom,
                        &minimum_available_bytes) ||
                    host_available_bytes <=
                        minimum_available_bytes)
                {
                    return false;
                }

                const auto safely_usable =
                    host_available_bytes -
                    reserve_bytes -
                    kIntegratedTransformHeadroom;

                return required_bytes <= safely_usable;
            }

            // UMA 上 cudaMemGetInfo() 不能代表可回收
            // page cache / shared DRAM 的真实可用量。
            // 上面的 MemAvailable 仅做预 admission；
            // cudaMalloc 仍是最终 fail-closed 检查。
            return true;
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;

        status =
            cudaMemGetInfo(
                &free_bytes,
                &total_bytes);

        if (status != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        constexpr std::size_t
            kMinimumCudaReserveBytes =
                std::size_t{4} << 30;

        const auto reserve_bytes =
            std::max(
                kMinimumCudaReserveBytes,
                total_bytes / 8);

        if (free_bytes <= reserve_bytes)
        {
            return false;
        }

        return required_bytes <=
               free_bytes -
                   reserve_bytes;
    }
    void release_cuda_t_edge_executor_cache()
    {
    }
}
