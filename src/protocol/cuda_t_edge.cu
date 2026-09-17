#include "gatzk/protocol/cuda_t_edge.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/util/effective_memory.hpp"

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

        constexpr unsigned
            kTEdgeExecutionBlockSize = 256;

        struct CudaTEdgeCompiledInstruction
        {
            TEdgeProgramOpcode opcode =
                TEdgeProgramOpcode::
                    LoadConstant;

            std::uint8_t dst = 0;
            std::uint8_t lhs = 0;
            std::uint8_t rhs = 0;
            std::uint8_t reserved = 0;

            std::uint32_t operand = 0;
        };

        struct CudaTEdgeCompiledTerm
        {
            std::size_t instruction_begin = 0;
            std::size_t instruction_end = 0;
            GpuFr alpha_scale{};
        };
        std::vector<
            CudaTEdgeCompiledInstruction>
        compile_t_edge_execution_plan(
            const TEdgeBoundProgram &program)
        {
            std::vector<
                CudaTEdgeCompiledInstruction>
                compiled(
                    program.instructions.size());

            for (const auto &term :
                 program.terms)
            {
                if (term.instruction_begin >
                        term.instruction_end ||
                    term.instruction_end >
                        program.instructions.size())
                {
                    throw std::runtime_error(
                        "CUDA t_edge term instruction range is invalid");
                }

                std::size_t depth = 0;
                std::size_t maximum_depth = 0;

                for (std::size_t index =
                         term.instruction_begin;
                     index <
                     term.instruction_end;
                     ++index)
                {
                    const auto &source =
                        program.instructions[index];

                    auto &target =
                        compiled[index];

                    target.opcode =
                        source.opcode;

                    target.operand =
                        source.operand;

                    switch (source.opcode)
                    {
                    case TEdgeProgramOpcode::
                        LoadAccess:
                    case TEdgeProgramOpcode::
                        LoadConstant:
                    {
                        if (depth >=
                            kMaximumProgramStackDepth)
                        {
                            throw std::runtime_error(
                                "CUDA t_edge compiled register depth overflow");
                        }

                        target.dst =
                            static_cast<
                                std::uint8_t>(
                                depth);

                        ++depth;

                        maximum_depth =
                            std::max(
                                maximum_depth,
                                depth);

                        break;
                    }

                    case TEdgeProgramOpcode::Add:
                    case TEdgeProgramOpcode::
                        Subtract:
                    case TEdgeProgramOpcode::
                        Multiply:
                    {
                        if (depth < 2)
                        {
                            throw std::runtime_error(
                                "CUDA t_edge binary instruction underflows stack");
                        }

                        target.lhs =
                            static_cast<
                                std::uint8_t>(
                                depth - 2U);

                        target.rhs =
                            static_cast<
                                std::uint8_t>(
                                depth - 1U);

                        target.dst =
                            target.lhs;

                        --depth;

                        break;
                    }

                    case TEdgeProgramOpcode::
                        Negate:
                    {
                        if (depth == 0)
                        {
                            throw std::runtime_error(
                                "CUDA t_edge negate instruction underflows stack");
                        }

                        target.lhs =
                            static_cast<
                                std::uint8_t>(
                                depth - 1U);

                        target.dst =
                            target.lhs;

                        break;
                    }
                    }
                }

                if (depth != 1)
                {
                    throw std::runtime_error(
                        "CUDA t_edge compiled term does not leave exactly one value");
                }

                if (maximum_depth >
                        program.maximum_stack_depth ||
                    maximum_depth >
                        kMaximumProgramStackDepth)
                {
                    throw std::runtime_error(
                        "CUDA t_edge compiled register plan exceeds declared stack depth");
                }
            }

            return compiled;
        }

        // The exact leaf helpers feed a bounded multi-stage CUDA transform
        // with one lane of target_size / 2 elements.  The leaf transform is
        // decomposed into ICICLE-supported 2^24 subtransforms plus exact
        // radix-2 combines; keep this capability boundary local to the
        // t_edge executor and do not conflate it with the quotient program.
        constexpr std::size_t
            kMaximumCudaLeafNttSize =
                std::size_t{1} << 26;

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

        template <std::size_t RegisterCount>
        struct TEdgeRegisterFile
        {
            GpuFr values[RegisterCount];

            __device__ __forceinline__ void store(
                std::uint8_t index,
                const GpuFr &value)
            {
                values[index] =
                    value;
            }

            __device__ __forceinline__
                GpuFr
                load(
                    std::uint8_t index) const
            {
                return values[index];
            }
        };

        template <>
        struct TEdgeRegisterFile<4>
        {
            GpuFr r0;
            GpuFr r1;
            GpuFr r2;
            GpuFr r3;

            __device__ __forceinline__ void store(
                std::uint8_t index,
                const GpuFr &value)
            {
                switch (index)
                {
                case 0:
                    r0 = value;
                    break;
                case 1:
                    r1 = value;
                    break;
                case 2:
                    r2 = value;
                    break;
                default:
                    r3 = value;
                    break;
                }
            }

            __device__ __forceinline__
                GpuFr
                load(
                    std::uint8_t index) const
            {
                switch (index)
                {
                case 0:
                    return r0;
                case 1:
                    return r1;
                case 2:
                    return r2;
                default:
                    return r3;
                }
            }
        };

        template <>
        struct TEdgeRegisterFile<8>
        {
            GpuFr r0;
            GpuFr r1;
            GpuFr r2;
            GpuFr r3;
            GpuFr r4;
            GpuFr r5;
            GpuFr r6;
            GpuFr r7;

            __device__ __forceinline__ void store(
                std::uint8_t index,
                const GpuFr &value)
            {
                switch (index)
                {
                case 0:
                    r0 = value;
                    break;
                case 1:
                    r1 = value;
                    break;
                case 2:
                    r2 = value;
                    break;
                case 3:
                    r3 = value;
                    break;
                case 4:
                    r4 = value;
                    break;
                case 5:
                    r5 = value;
                    break;
                case 6:
                    r6 = value;
                    break;
                default:
                    r7 = value;
                    break;
                }
            }

            __device__ __forceinline__
                GpuFr
                load(
                    std::uint8_t index) const
            {
                switch (index)
                {
                case 0:
                    return r0;
                case 1:
                    return r1;
                case 2:
                    return r2;
                case 3:
                    return r3;
                case 4:
                    return r4;
                case 5:
                    return r5;
                case 6:
                    return r6;
                default:
                    return r7;
                }
            }
        };

        template <std::size_t RegisterCount>
        __global__ void
        t_edge_term_batch_kernel(
            const CudaTEdgeCompiledInstruction
                *instructions,
            const CudaTEdgeCompiledTerm
                *terms,
            std::size_t term_begin,
            std::size_t term_end,
            const GpuFr *constants,
            GpuFr *const *leaf_slots,
            const std::uint8_t *leaf_split_layout,
            const std::uint32_t *access_slots,
            const std::uint8_t *access_rotated,
            std::size_t rotation_step,
            std::size_t target_size,
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

            auto accumulated =
                quotient[point];

            // Terms are deliberately evaluated in their original canonical
            // order.  The only change versus the previous executor is that
            // the intermediate quotient value stays in a register until the
            // end of this batch instead of being written to global memory
            // after every term.
            for (std::size_t term_index =
                     term_begin;
                 term_index < term_end;
                 ++term_index)
            {
                const auto term =
                    terms[term_index];

                TEdgeRegisterFile<
                    RegisterCount>
                    registers;

                for (std::size_t index =
                         term.instruction_begin;
                     index <
                     term.instruction_end;
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
                                      (target_size -
                                       1U)
                                : point;

                        std::size_t physical_index =
                            logical_index;

                        if (leaf_split_layout[slot] != 0)
                        {
                            const auto half_size =
                                target_size / 2;

                            physical_index =
                                logical_index / 2 +
                                ((logical_index &
                                  1U) != 0U
                                     ? half_size
                                     : 0);
                        }

                        registers.store(
                            instruction.dst,
                            leaf_slots[slot]
                                      [physical_index]);

                        break;
                    }

                    case TEdgeProgramOpcode::
                        LoadConstant:
                    {
                        registers.store(
                            instruction.dst,
                            constants[instruction
                                          .operand]);
                        break;
                    }

                    case TEdgeProgramOpcode::Add:
                    {
                        registers.store(
                            instruction.dst,
                            algebra::gpu::fr_add(
                                registers.load(
                                    instruction.lhs),
                                registers.load(
                                    instruction.rhs)));
                        break;
                    }

                    case TEdgeProgramOpcode::
                        Subtract:
                    {
                        registers.store(
                            instruction.dst,
                            algebra::gpu::fr_sub(
                                registers.load(
                                    instruction.lhs),
                                registers.load(
                                    instruction.rhs)));
                        break;
                    }

                    case TEdgeProgramOpcode::
                        Multiply:
                    {
                        registers.store(
                            instruction.dst,
                            algebra::gpu::fr_mul(
                                registers.load(
                                    instruction.lhs),
                                registers.load(
                                    instruction.rhs)));
                        break;
                    }

                    case TEdgeProgramOpcode::
                        Negate:
                    {
                        registers.store(
                            instruction.dst,
                            algebra::gpu::fr_sub(
                                algebra::gpu::
                                    fr_zero(),
                                registers.load(
                                    instruction.lhs)));
                        break;
                    }
                    }
                }

                const auto contribution =
                    algebra::gpu::fr_mul(
                        registers.load(0),
                        term.alpha_scale);

                accumulated =
                    algebra::gpu::fr_add(
                        accumulated,
                        contribution);
            }

            quotient[point] =
                accumulated;
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
            const GpuFr *lane_root_powers,
            const GpuFr *lane_zero_powers,
            const GpuFr *block_root_powers,
            const GpuFr *block_zero_powers,
            GpuFr shift_domain_power,
            std::uint8_t output_split_layout,
            GpuFr *target)
        {
            const auto block_begin =
                static_cast<std::size_t>(
                    blockIdx.x) *
                blockDim.x;

            const auto logical_index =
                block_begin +
                threadIdx.x;

            if (logical_index >= target_size)
            {
                return;
            }

            const auto block_root_power =
                block_root_powers[blockIdx.x];

            const auto block_zero_power =
                block_zero_powers[blockIdx.x];

            const auto lane_root_power =
                lane_root_powers[threadIdx.x];

            const auto lane_zero_power =
                lane_zero_powers[threadIdx.x];

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
                    algebra::gpu::fr_mul(
                        block_root_power,
                        lane_root_power));

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
                            source_blinder_offsets[source_index],
                            blinder_count,
                            point);

                    source_blinder_sum =
                        algebra::gpu::fr_add(
                            source_blinder_sum,
                            algebra::gpu::fr_mul(
                                blinder_value,
                                source_scales[source_index]));
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
                        algebra::gpu::fr_mul(
                            block_zero_power,
                            lane_zero_power)),
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

                host_leaf_requires_default_wait_.assign(
                    cache_slot_count_,
                    false);

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

                    device_leaf_arenas_.clear();

                    // Slot 0 must remain a standalone allocation because
                    // finalize_coefficients() transfers ownership of this
                    // buffer to the returned resident coefficient object.
                    auto dedicated_status =
                        cudaMalloc(
                            reinterpret_cast<void **>(
                                &device_leaf_slots_.front()),
                            leaf_bytes);

                    if (dedicated_status !=
                        cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "dedicated leaf allocation",
                            dedicated_status);
                    }

                    constexpr std::size_t
                        kLeafArenaSlotCount = 128;

                    if (cache_slot_count_ > 1)
                    {
                        device_leaf_arenas_.reserve(
                            (cache_slot_count_ - 1 +
                             kLeafArenaSlotCount - 1) /
                            kLeafArenaSlotCount);
                    }

                    for (std::size_t base_slot = 1;
                         base_slot <
                         cache_slot_count_;)
                    {
                        const auto arena_slots =
                            std::min<std::size_t>(
                                kLeafArenaSlotCount,
                                cache_slot_count_ -
                                    base_slot);

                        GpuFr *arena = nullptr;

                        const auto arena_status =
                            cudaMalloc(
                                reinterpret_cast<void **>(
                                    &arena),
                                checked_bytes(
                                    arena_slots,
                                    leaf_bytes,
                                    "leaf arena"));

                        if (arena_status !=
                            cudaSuccess)
                        {
                            throw_cuda_t_edge_error(
                                "leaf arena allocation",
                                arena_status);
                        }

                        device_leaf_arenas_
                            .push_back(arena);

                        for (std::size_t local = 0;
                             local < arena_slots;
                             ++local)
                        {
                            device_leaf_slots_[base_slot + local] =
                                arena +
                                local *
                                    target_size_;
                        }

                        base_slot +=
                            arena_slots;
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

                    const auto power_block_count =
                        (target_size_ +
                         kTEdgeExecutionBlockSize -
                         1U) /
                        kTEdgeExecutionBlockSize;

                    std::vector<
                        algebra::FieldElement>
                        lane_root_powers(
                            kTEdgeExecutionBlockSize);

                    std::vector<
                        algebra::FieldElement>
                        lane_zero_powers(
                            kTEdgeExecutionBlockSize);

                    std::vector<
                        algebra::FieldElement>
                        block_root_powers(
                            power_block_count);

                    std::vector<
                        algebra::FieldElement>
                        block_zero_powers(
                            power_block_count);

                    algebra::geometric_powers_native(
                        algebra::FieldElement::
                            mutable_native_array_data(
                                lane_root_powers.data()),
                        lane_root_powers.size(),
                        root);

                    algebra::geometric_powers_native(
                        algebra::FieldElement::
                            mutable_native_array_data(
                                lane_zero_powers.data()),
                        lane_zero_powers.size(),
                        zero_step);

                    const auto root_block_step =
                        root.pow(
                            kTEdgeExecutionBlockSize);

                    const auto zero_block_step =
                        zero_step.pow(
                            kTEdgeExecutionBlockSize);

                    algebra::geometric_powers_native(
                        algebra::FieldElement::
                            mutable_native_array_data(
                                block_root_powers.data()),
                        block_root_powers.size(),
                        root_block_step);

                    algebra::geometric_powers_native(
                        algebra::FieldElement::
                            mutable_native_array_data(
                                block_zero_powers.data()),
                        block_zero_powers.size(),
                        zero_block_step);

                    const auto lane_power_bytes =
                        checked_bytes(
                            kTEdgeExecutionBlockSize,
                            sizeof(GpuFr),
                            "lane power table");

                    const auto block_power_bytes =
                        checked_bytes(
                            power_block_count,
                            sizeof(GpuFr),
                            "block power table");

                    const auto upload_power_table =
                        [&](GpuFr **device_output,
                            const std::vector<
                                algebra::FieldElement> &values,
                            std::size_t byte_count,
                            const char *label)
                    {
                        auto upload_status =
                            cudaMalloc(
                                reinterpret_cast<
                                    void **>(
                                    device_output),
                                byte_count);

                        if (upload_status !=
                            cudaSuccess)
                        {
                            throw_cuda_t_edge_error(
                                label,
                                upload_status);
                        }

                        upload_status =
                            cudaMemcpy(
                                *device_output,
                                algebra::FieldElement::
                                    native_array_data(
                                        values.data()),
                                byte_count,
                                cudaMemcpyHostToDevice);

                        if (upload_status !=
                            cudaSuccess)
                        {
                            throw_cuda_t_edge_error(
                                label,
                                upload_status);
                        }

                        algebra::record_cuda_h2d(
                            byte_count);
                    };

                    upload_power_table(
                        &device_lane_root_powers_,
                        lane_root_powers,
                        lane_power_bytes,
                        "lane-root power upload");

                    upload_power_table(
                        &device_lane_zero_powers_,
                        lane_zero_powers,
                        lane_power_bytes,
                        "lane-zero power upload");

                    upload_power_table(
                        &device_block_root_powers_,
                        block_root_powers,
                        block_power_bytes,
                        "block-root power upload");

                    upload_power_table(
                        &device_block_zero_powers_,
                        block_zero_powers,
                        block_power_bytes,
                        "block-zero power upload");

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

                const auto compiled_program =
                    compile_t_edge_execution_plan(
                        program);
                std::vector<
                    CudaTEdgeCompiledTerm>
                    compiled_terms;

                compiled_terms.reserve(
                    program.terms.size());

                for (const auto &term :
                     program.terms)
                {
                    CudaTEdgeCompiledTerm
                        compiled_term;

                    compiled_term.instruction_begin =
                        term.instruction_begin;

                    compiled_term.instruction_end =
                        term.instruction_end;

                    compiled_term.alpha_scale =
                        gpu_field(
                            term.alpha_scale);

                    compiled_terms.push_back(
                        compiled_term);
                }

                try
                {
                    auto status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_instructions_),
                            checked_bytes(
                                compiled_program.size(),
                                sizeof(
                                    CudaTEdgeCompiledInstruction),
                                "compiled instructions"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "compiled instruction allocation",
                            status);
                    }

                    status =
                        cudaMemcpy(
                            device_instructions_,
                            compiled_program.data(),
                            checked_bytes(
                                compiled_program.size(),
                                sizeof(
                                    CudaTEdgeCompiledInstruction),
                                "compiled instructions"),
                            cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "instruction upload",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<void **>(
                                &device_terms_),
                            checked_bytes(
                                compiled_terms.size(),
                                sizeof(
                                    CudaTEdgeCompiledTerm),
                                "compiled terms"));

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "compiled term allocation",
                            status);
                    }

                    status =
                        cudaMemcpy(
                            device_terms_,
                            compiled_terms.data(),
                            checked_bytes(
                                compiled_terms.size(),
                                sizeof(
                                    CudaTEdgeCompiledTerm),
                                "compiled terms"),
                            cudaMemcpyHostToDevice);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "compiled term upload",
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
                            reinterpret_cast<mcl::Fr *>(
                                device_leaf_slots_[slot]),
                            target_size_,
                            true,
                            host_leaf_requires_default_wait_[slot]);

                if (!executed)
                {
                    return false;
                }

                host_leaf_requires_default_wait_[slot] =
                    false;

                host_leaf_split_layout_[slot] =
                    1;

                const auto status =
                    cudaMemsetAsync(
                        device_leaf_split_layout_ +
                            slot,
                        1,
                        sizeof(std::uint8_t),
                        nullptr);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "leaf-layout update",
                        status);
                }

                host_leaf_bound_[slot] = true;
                host_leaf_uploaded_prefix_[slot] = target_size_;
                return true;
            }

            bool materialize_evaluation_leaf_batch(
                const std::vector<
                    CudaTEdgeEvaluationLeafRequest>
                    &requests)
                override
            {
                if (requests.empty())
                {
                    return true;
                }

                std::vector<std::uint8_t>
                    seen(
                        cache_slot_count_,
                        0);

                bool batch_dependency_clear =
                    true;

                for (const auto &request :
                     requests)
                {
                    if (request.slot >=
                            cache_slot_count_ ||
                        seen[request.slot] != 0)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge evaluation leaf batch contains an invalid or duplicate slot");
                    }

                    if (request.value_count != 0 &&
                        request.values == nullptr)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge evaluation leaf batch contains null values");
                    }

                    if (request.blinder_count != 0 &&
                        request.zero_domain_blinder ==
                            nullptr)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge evaluation leaf batch contains a null blinder");
                    }

                    if (host_leaf_bound_[request.slot] != 0)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge evaluation leaf batch target slot is already bound");
                    }

                    if (host_leaf_requires_default_wait_[request.slot])
                    {
                        batch_dependency_clear =
                            false;
                    }

                    seen[request.slot] = 1;
                }

                int executor_device = -1;

                auto status =
                    cudaGetDevice(
                        &executor_device);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "evaluation leaf batch device lookup",
                        status);
                }

                if (batch_dependency_clear &&
                    requests.size() >= 2)
                {
                    constexpr std::size_t
                        kMaximumBatchWorkers =
                            8;

                    constexpr std::size_t
                        kMaximumInFlightLeaves =
                            32;

                    constexpr std::size_t
                        kMaximumConcurrentBatchWorkspaceBytes =
                            std::size_t{512} << 20;

                    std::size_t
                        batch_worker_count =
                            0;

                    std::size_t
                        batch_width =
                            0;

                    if (target_size_ <=
                        std::numeric_limits<
                            std::size_t>::max() /
                            sizeof(GpuFr))
                    {
                        const auto leaf_bytes =
                            target_size_ *
                            sizeof(GpuFr);

                        if (leaf_bytes != 0 &&
                            leaf_bytes <=
                                std::numeric_limits<
                                    std::size_t>::max() /
                                    2)
                        {
                            const auto
                                per_leaf_workspace_bytes =
                                    leaf_bytes *
                                    2;

                            const auto
                                memory_in_flight =
                                    per_leaf_workspace_bytes >
                                            kMaximumConcurrentBatchWorkspaceBytes
                                        ? std::size_t{0}
                                        : kMaximumConcurrentBatchWorkspaceBytes /
                                              per_leaf_workspace_bytes;

                            const auto
                                in_flight_limit =
                                    std::min<std::size_t>(
                                        {
                                            requests.size(),
                                            kMaximumInFlightLeaves,
                                            memory_in_flight,
                                        });

                            if (in_flight_limit >=
                                2)
                            {
                                const auto
                                    hardware_workers =
                                        std::max<std::size_t>(
                                            1,
                                            std::thread::
                                                hardware_concurrency());

                                batch_worker_count =
                                    std::min<std::size_t>(
                                        {
                                            kMaximumBatchWorkers,
                                            hardware_workers,
                                            std::max<std::size_t>(
                                                1,
                                                requests.size() /
                                                    2),
                                            std::max<std::size_t>(
                                                1,
                                                in_flight_limit /
                                                    2),
                                        });

                                batch_worker_count =
                                    std::max<std::size_t>(
                                        1,
                                        batch_worker_count);

                                batch_width =
                                    std::max<std::size_t>(
                                        2,
                                        in_flight_limit /
                                            batch_worker_count);

                                batch_width =
                                    std::min<std::size_t>(
                                        {
                                            batch_width,
                                            kMaximumInFlightLeaves,
                                            requests.size(),
                                        });
                            }
                        }
                    }

                    if (batch_worker_count != 0 &&
                        batch_width >= 2)
                    {
                        std::vector<
                            std::vector<
                                algebra::
                                    CudaCosetEvaluationBatchRequest>>
                            worker_requests(
                                batch_worker_count);

                        const auto
                            approximate_requests_per_worker =
                                (requests.size() +
                                 batch_worker_count -
                                 1) /
                                batch_worker_count;

                        for (auto &worker_batch :
                             worker_requests)
                        {
                            worker_batch.reserve(
                                approximate_requests_per_worker);
                        }

                        for (std::size_t index = 0;
                             index < requests.size();
                             ++index)
                        {
                            const auto &request =
                                requests[index];

                            worker_requests[index %
                                            batch_worker_count]
                                .push_back(
                                    algebra::
                                        CudaCosetEvaluationBatchRequest{
                                            .values =
                                                request.values,
                                            .value_count =
                                                request.value_count,
                                            .zero_domain_blinder =
                                                request.zero_domain_blinder,
                                            .blinder_count =
                                                request.blinder_count,
                                            .device_output =
                                                reinterpret_cast<
                                                    mcl::Fr *>(
                                                    device_leaf_slots_[request.slot]),
                                        });
                        }

                        std::atomic<bool>
                            batch_declined{
                                false};

                        std::vector<
                            std::future<void>>
                            batch_futures;

                        batch_futures.reserve(
                            batch_worker_count);

                        for (std::size_t worker = 0;
                             worker <
                             batch_worker_count;
                             ++worker)
                        {
                            if (worker_requests[worker]
                                    .empty())
                            {
                                continue;
                            }

                            batch_futures.push_back(
                                std::async(
                                    std::launch::async,
                                    [&, worker,
                                     executor_device,
                                     batch_width]()
                                    {
                                        const auto
                                            device_status =
                                                cudaSetDevice(
                                                    executor_device);

                                        if (device_status !=
                                            cudaSuccess)
                                        {
                                            throw_cuda_t_edge_error(
                                                "batched evaluation leaf device selection",
                                                device_status);
                                        }

                                        const auto &batch =
                                            worker_requests[worker];

                                        if (batch.size() <
                                            2)
                                        {
                                            batch_declined.store(
                                                true,
                                                std::memory_order_release);

                                            return;
                                        }

                                        const bool executed =
                                            algebra::
                                                cuda_coset_evaluations_to_device_exact_batch(
                                                    batch.data(),
                                                    batch.size(),
                                                    domain_size_,
                                                    shift_,
                                                    target_size_,
                                                    false,
                                                    batch_width);

                                        if (!executed)
                                        {
                                            batch_declined.store(
                                                true,
                                                std::memory_order_release);
                                        }
                                    }));
                        }

                        for (auto &future :
                             batch_futures)
                        {
                            future.get();
                        }

                        // Each joined batch call used make_default_stream_wait=false
                        // and completed its own stream before returning.
                        if (!batch_declined.load(
                                std::memory_order_acquire))
                        {
                            for (const auto &request :
                                 requests)
                            {
                                host_leaf_requires_default_wait_[request.slot] =
                                    false;

                                host_leaf_split_layout_[request.slot] =
                                    1;

                                host_leaf_bound_[request.slot] =
                                    true;

                                host_leaf_uploaded_prefix_[request.slot] =
                                    target_size_;
                            }

                            const auto layout_bytes =
                                checked_bytes(
                                    cache_slot_count_,
                                    sizeof(
                                        std::uint8_t),
                                    "batched leaf layout");

                            status =
                                cudaMemcpyAsync(
                                    device_leaf_split_layout_,
                                    host_leaf_split_layout_
                                        .data(),
                                    layout_bytes,
                                    cudaMemcpyHostToDevice,
                                    nullptr);

                            if (status !=
                                cudaSuccess)
                            {
                                throw_cuda_t_edge_error(
                                    "batched leaf-layout upload",
                                    status);
                            }

                            algebra::
                                record_cuda_h2d(
                                    layout_bytes);
                            status = cudaStreamSynchronize(nullptr);
                            if (status != cudaSuccess)
                            {
                                throw_cuda_t_edge_error("batched leaf-layout completion", status);
                            }

                            algebra::
                                record_cuda_stream_sync();

                            return true;
                        }
                    }
                }

                // The native batch route is intentionally optional for
                // staged transforms or resource-pressure cases.  Preserve the
                // already validated bounded multi-worker implementation as
                // the exact fallback rather than falling back to a different
                // mathematical path.
                const auto hardware_workers =
                    std::max<std::size_t>(
                        1,
                        std::thread::
                            hardware_concurrency());

                std::size_t worker_count =
                    std::min<std::size_t>(
                        {
                            requests.size(),
                            hardware_workers,
                            std::size_t{8},
                        });

                constexpr std::size_t
                    kMaximumConcurrentLeafWorkspaceBytes =
                        std::size_t{512} << 20;

                if (target_size_ >
                    std::numeric_limits<
                        std::size_t>::max() /
                        sizeof(GpuFr))
                {
                    worker_count = 1;
                }
                else
                {
                    const auto leaf_bytes =
                        target_size_ *
                        sizeof(GpuFr);

                    const auto workspace_bytes =
                        leaf_bytes >
                                std::numeric_limits<
                                    std::size_t>::max() /
                                    2
                            ? std::numeric_limits<
                                  std::size_t>::max()
                            : leaf_bytes * 2;

                    const auto memory_worker_count =
                        workspace_bytes == 0 ||
                                workspace_bytes >
                                    kMaximumConcurrentLeafWorkspaceBytes
                            ? std::size_t{1}
                            : std::max<std::size_t>(
                                  1,
                                  kMaximumConcurrentLeafWorkspaceBytes /
                                      workspace_bytes);

                    worker_count =
                        std::min(
                            worker_count,
                            memory_worker_count);
                }

                worker_count =
                    std::max<std::size_t>(
                        1,
                        worker_count);

                std::atomic<std::size_t>
                    next{0};

                std::atomic<bool>
                    declined{false};

                std::vector<
                    std::future<void>>
                    futures;

                futures.reserve(
                    worker_count);

                for (std::size_t worker = 0;
                     worker < worker_count;
                     ++worker)
                {
                    futures.push_back(
                        std::async(
                            std::launch::async,
                            [&, executor_device]()
                            {
                                const auto
                                    device_status =
                                        cudaSetDevice(
                                            executor_device);

                                if (device_status !=
                                    cudaSuccess)
                                {
                                    throw_cuda_t_edge_error(
                                        "evaluation leaf batch device selection",
                                        device_status);
                                }

                                while (!declined.load(
                                    std::memory_order_acquire))
                                {
                                    const auto index =
                                        next.fetch_add(
                                            1,
                                            std::memory_order_relaxed);

                                    if (index >=
                                        requests.size())
                                    {
                                        break;
                                    }

                                    const auto &request =
                                        requests[index];

                                    if (!materialize_evaluation_leaf(
                                            request.slot,
                                            request.values,
                                            request.value_count,
                                            request.zero_domain_blinder,
                                            request.blinder_count))
                                    {
                                        declined.store(
                                            true,
                                            std::memory_order_release);

                                        break;
                                    }
                                }
                            }));
                }

                for (auto &future :
                     futures)
                {
                    future.get();
                }

                status =
                    cudaDeviceSynchronize();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "evaluation leaf batch completion",
                        status);
                }

                algebra::
                    record_cuda_stream_sync();

                return !declined.load(
                    std::memory_order_acquire);
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
                            reinterpret_cast<mcl::Fr *>(
                                device_leaf_slots_[slot]),
                            target_size_,
                            true,
                            host_leaf_requires_default_wait_[slot]);

                if (!executed)
                {
                    return false;
                }

                host_leaf_requires_default_wait_[slot] =
                    false;

                host_leaf_split_layout_[slot] =
                    1;

                const auto status =
                    cudaMemsetAsync(
                        device_leaf_split_layout_ +
                            slot,
                        1,
                        sizeof(std::uint8_t),
                        nullptr);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "leaf-layout update",
                        status);
                }

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
                    device_lane_root_powers_,
                    device_lane_zero_powers_,
                    device_block_root_powers_,
                    device_block_zero_powers_,
                    gpu_field(shift_domain_power),
                    output_split_layout,
                    device_leaf_slots_[target_slot]);

                status = cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear materialization launch",
                        status);
                }

                algebra::record_cuda_kernel_invocations(1);

                host_leaf_split_layout_[target_slot] =
                    output_split_layout;

                status =
                    cudaMemsetAsync(
                        device_leaf_split_layout_ +
                            target_slot,
                        static_cast<int>(
                            output_split_layout),
                        sizeof(std::uint8_t),
                        nullptr);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "linear target-layout update",
                        status);
                }

                for (const auto &source :
                     sources)
                {
                    host_leaf_requires_default_wait_[source.slot] = true;
                }

                host_leaf_requires_default_wait_[target_slot] = true;

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

                host_leaf_requires_default_wait_[slot] =
                    false;

                host_leaf_uploaded_prefix_[slot] =
                    logical_offset + count;
                host_leaf_bound_[slot] =
                    host_leaf_uploaded_prefix_[slot] ==
                    target_size_;
            }

            void accumulate_term_range(
                std::size_t term_begin,
                std::size_t term_end,
                const std::vector<
                    std::uint32_t>
                    &access_slots)
                override
            {
                if (program_ == nullptr ||
                    device_terms_ == nullptr ||
                    term_begin >= term_end ||
                    term_end >
                        program_->terms.size() ||
                    access_slots.size() !=
                        program_->accesses.size())
                {
                    throw std::runtime_error(
                        "invalid CUDA t_edge term-range invocation");
                }

                std::size_t maximum_stack_depth = 0;

                for (std::size_t term_index =
                         term_begin;
                     term_index < term_end;
                     ++term_index)
                {
                    const auto &term =
                        program_->terms[term_index];

                    maximum_stack_depth =
                        std::max(
                            maximum_stack_depth,
                            term.maximum_stack_depth);

                    for (const auto access :
                         term.required_accesses)
                    {
                        if (access >=
                                access_slots.size() ||
                            access_slots[access] ==
                                kInvalidLeafSlot ||
                            access_slots[access] >=
                                cache_slot_count_ ||
                            !host_leaf_bound_[access_slots[access]])
                        {
                            throw std::runtime_error(
                                "CUDA t_edge term batch has an unbound leaf");
                        }
                    }
                }

                const auto access_slot_bytes =
                    checked_bytes(
                        access_slots.size(),
                        sizeof(std::uint32_t),
                        "access slots");

                auto status =
                    cudaSuccess;

                if (!access_slots_uploaded_ ||
                    host_access_slots_cache_ !=
                        access_slots)
                {
                    status =
                        cudaMemcpyAsync(
                            device_access_slots_,
                            access_slots.data(),
                            access_slot_bytes,
                            cudaMemcpyHostToDevice,
                            0);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_t_edge_error(
                            "access-slot upload",
                            status);
                    }

                    algebra::record_cuda_h2d(
                        access_slot_bytes);

                    host_access_slots_cache_ =
                        access_slots;

                    access_slots_uploaded_ =
                        true;
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
                        "CUDA t_edge grid overflow");
                }

                const auto launch =
                    [&](auto register_count)
                {
                    constexpr std::size_t
                        RegisterCount =
                            decltype(register_count)::value;

                    t_edge_term_batch_kernel<
                        RegisterCount><<<
                        static_cast<unsigned>(
                            block_count_size),
                        kBlockSize>>>(
                        device_instructions_,
                        device_terms_,
                        term_begin,
                        term_end,
                        device_constants_,
                        device_leaf_table_,
                        device_leaf_split_layout_,
                        device_access_slots_,
                        device_access_rotated_,
                        rotation_step_,
                        target_size_,
                        device_quotient_);
                };

                if (maximum_stack_depth <= 4)
                {
                    launch(
                        std::integral_constant<
                            std::size_t,
                            4>{});
                }
                else if (maximum_stack_depth <= 8)
                {
                    launch(
                        std::integral_constant<
                            std::size_t,
                            8>{});
                }
                else if (maximum_stack_depth <= 16)
                {
                    launch(
                        std::integral_constant<
                            std::size_t,
                            16>{});
                }
                else if (maximum_stack_depth <= 32)
                {
                    launch(
                        std::integral_constant<
                            std::size_t,
                            32>{});
                }
                else
                {
                    launch(
                        std::integral_constant<
                            std::size_t,
                            kMaximumProgramStackDepth>{});
                }

                status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "term-batch kernel launch",
                        status);
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);

                for (std::size_t term_index =
                         term_begin;
                     term_index < term_end;
                     ++term_index)
                {
                    for (const auto access :
                         program_->terms[term_index]
                             .required_accesses)
                    {
                        host_leaf_requires_default_wait_[access_slots[access]] = true;
                    }
                }
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

            std::shared_ptr<
                const algebra::
                    CudaResidentCoefficientBuffer>
            finalize_coefficients(
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

                if (device_leaf_slots_.empty() ||
                    device_leaf_slots_.front() ==
                        nullptr)
                {
                    throw std::runtime_error(
                        "CUDA t_edge has no reusable resident coefficient buffer");
                }

                constexpr unsigned
                    kBlockSize =
                        kTEdgeExecutionBlockSize;

                const auto block_count_size =
                    (target_size_ +
                     kBlockSize - 1U) /
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

                auto *resident_coefficients =
                    device_leaf_slots_.front();

                std::shared_ptr<
                    const algebra::
                        CudaHostTransferGuard>
                    host_transfer;

                std::size_t logical_count = 0;
                std::size_t nonzero_count = 0;

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
                                    output),
                            true,
                            reinterpret_cast<mcl::Fr *>(
                                resident_coefficients),
                            &host_transfer,
                            &logical_count,
                            &nonzero_count);

                if (!executed)
                {
                    throw std::runtime_error(
                        "CUDA t_edge device quotient INTT was not admitted");
                }

                if (logical_count == 0 ||
                    logical_count > target_size_ ||
                    nonzero_count > logical_count)
                {
                    throw std::runtime_error(
                        "CUDA t_edge resident coefficient metadata is invalid");
                }

                int resident_device = -1;

                status =
                    cudaGetDevice(
                        &resident_device);

                if (status != cudaSuccess)
                {
                    throw_cuda_t_edge_error(
                        "resident coefficient device lookup",
                        status);
                }

                device_leaf_slots_.front() =
                    nullptr;

                host_leaf_bound_.front() =
                    false;

                host_leaf_uploaded_prefix_.front() =
                    0;

                std::shared_ptr<void> allocation(
                    resident_coefficients,
                    [resident_device](
                        void *pointer) noexcept
                    {
                        if (pointer == nullptr)
                        {
                            return;
                        }

                        int previous_device = -1;

                        if (cudaGetDevice(
                                &previous_device) !=
                            cudaSuccess)
                        {
                            (void)cudaGetLastError();
                            return;
                        }

                        if (previous_device !=
                            resident_device)
                        {
                            if (cudaSetDevice(
                                    resident_device) !=
                                cudaSuccess)
                            {
                                (void)cudaGetLastError();
                                return;
                            }
                        }

                        (void)cudaFree(
                            pointer);

                        if (previous_device !=
                            resident_device)
                        {
                            (void)cudaSetDevice(
                                previous_device);
                        }
                    });

                auto descriptor =
                    std::make_shared<
                        algebra::
                            CudaResidentCoefficientBuffer>();

                descriptor->allocation =
                    std::move(allocation);

                descriptor
                    ->device_montgomery_limbs =
                    reinterpret_cast<
                        const std::uint64_t *>(
                        resident_coefficients);

                descriptor->capacity =
                    target_size_;

                descriptor->logical_count =
                    logical_count;

                descriptor->nonzero_count =
                    nonzero_count;

                descriptor->device =
                    resident_device;

                descriptor->host_transfer =
                    std::move(host_transfer);

                algebra::
                    record_cuda_kernel_invocations(
                        1);

                return descriptor;
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

                if (device_terms_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_terms_);
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

                device_terms_ =
                    nullptr;

                device_constants_ =
                    nullptr;

                device_access_rotated_ =
                    nullptr;

                device_access_slots_ =
                    nullptr;

                host_access_slots_cache_
                    .clear();

                access_slots_uploaded_ =
                    false;

                program_ = nullptr;
            }

            void release_all_noexcept() noexcept
            {
                release_program_noexcept();
                release_linear_buffers_noexcept();

                if (!device_leaf_slots_.empty() &&
                    device_leaf_slots_.front() !=
                        nullptr)
                {
                    // Slot 0 is the only standalone leaf allocation.  It can
                    // be null after finalize_coefficients() transfers its
                    // ownership to the resident coefficient buffer.
                    (void)cudaFree(
                        device_leaf_slots_
                            .front());
                }

                for (auto *arena :
                     device_leaf_arenas_)
                {
                    if (arena != nullptr)
                    {
                        (void)cudaFree(
                            arena);
                    }
                }

                device_leaf_arenas_.clear();
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

                if (device_lane_root_powers_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_lane_root_powers_);
                }

                if (device_lane_zero_powers_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_lane_zero_powers_);
                }

                if (device_block_root_powers_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_block_root_powers_);
                }

                if (device_block_zero_powers_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_block_zero_powers_);
                }

                device_leaf_table_ =
                    nullptr;

                device_leaf_split_layout_ =
                    nullptr;

                device_quotient_ =
                    nullptr;

                device_zero_inverse_cycle_ =
                    nullptr;

                device_lane_root_powers_ =
                    nullptr;

                device_lane_zero_powers_ =
                    nullptr;

                device_block_root_powers_ =
                    nullptr;

                device_block_zero_powers_ =
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

            std::vector<GpuFr *>
                device_leaf_arenas_;

            GpuFr **device_leaf_table_ =
                nullptr;

            std::vector<std::uint8_t>
                host_leaf_split_layout_;

            std::vector<std::uint8_t>
                host_leaf_bound_;

            std::vector<std::size_t>
                host_leaf_uploaded_prefix_;

            std::vector<std::uint8_t>
                host_leaf_requires_default_wait_;

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

            CudaTEdgeCompiledInstruction
                *device_instructions_ =
                    nullptr;

            CudaTEdgeCompiledTerm
                *device_terms_ =
                    nullptr;

            GpuFr *device_constants_ =
                nullptr;

            GpuFr *device_lane_root_powers_ =
                nullptr;

            GpuFr *device_lane_zero_powers_ =
                nullptr;

            GpuFr *device_block_root_powers_ =
                nullptr;

            GpuFr *device_block_zero_powers_ =
                nullptr;

            std::uint8_t
                *device_access_rotated_ =
                    nullptr;

            std::uint32_t
                *device_access_slots_ =
                    nullptr;

            std::vector<std::uint32_t>
                host_access_slots_cache_;

            bool access_slots_uploaded_ =
                false;

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
        if (!cuda_t_edge_executor_available() ||
            !cuda_t_edge_leaf_exact_ntt_supported(
                target_size))
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
        // Preserve the shape-only API for small runtime probes.  Production
        // large-route selection uses the complete overload below so program,
        // linear-buffer, caller-host, and domain geometry are admitted before
        // the executor constructor can allocate anything.
        CudaTEdgeExecutorAdmissionShape shape;
        shape.domain_size = target_size;
        return cuda_t_edge_executor_can_admit(
            target_size,
            cache_slot_count,
            leaf_exact_ntt_supported,
            shape);
    }

    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count,
        bool leaf_exact_ntt_supported,
        const CudaTEdgeExecutorAdmissionShape &shape)
    {
        if (!cuda_t_edge_executor_available() ||
            !leaf_exact_ntt_supported ||
            !cuda_t_edge_leaf_exact_ntt_supported(
                target_size) ||
            target_size < 2 ||
            cache_slot_count == 0 ||
            (target_size &
             (target_size - 1U)) != 0U ||
            cache_slot_count >
                std::numeric_limits<
                    std::uint32_t>::max() ||
            shape.domain_size == 0 ||
            shape.domain_size > target_size ||
            target_size % shape.domain_size != 0 ||
            (shape.max_linear_source_count != 0 &&
             shape.max_linear_source_count >= cache_slot_count) ||
            shape.max_linear_single_source_blinder_count >
                target_size - shape.domain_size ||
            shape.max_linear_target_blinder_count >
                target_size - shape.domain_size)
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
        if (!checked_mul_size(
                leaf_bytes,
                std::size_t{2},
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

        constexpr std::size_t kAdmissionHeadroomBytes =
            std::size_t{1} << 30;

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

        std::size_t cycle_count = 0;
        if (!checked_mul_size(
                target_size / shape.domain_size,
                sizeof(GpuFr),
                &cycle_count))
        {
            return false;
        }

        const auto checked_count_bytes =
            [](std::size_t count,
               std::size_t element_size,
               std::size_t *out)
        {
            return checked_mul_size(
                count,
                element_size,
                out);
        };

        const auto power_block_count =
            (target_size +
             kTEdgeExecutionBlockSize -
             1U) /
            kTEdgeExecutionBlockSize;

        std::size_t lane_power_bytes = 0;
        std::size_t block_power_bytes = 0;

        if (!checked_mul_size(
                2U *
                    kTEdgeExecutionBlockSize,
                sizeof(GpuFr),
                &lane_power_bytes) ||
            !checked_mul_size(
                2U *
                    power_block_count,
                sizeof(GpuFr),
                &block_power_bytes))
        {
            return false;
        }
        std::size_t device_required_bytes = 0;
        std::size_t host_required_bytes = 0;
        const auto add_device =
            [&device_required_bytes](std::size_t value)
        {
            return checked_add_size(
                device_required_bytes,
                value,
                &device_required_bytes);
        };
        const auto add_host =
            [&host_required_bytes](std::size_t value)
        {
            return checked_add_size(
                host_required_bytes,
                value,
                &host_required_bytes);
        };

        std::size_t instruction_bytes = 0;
        std::size_t term_bytes = 0;
        std::size_t constant_bytes = 0;
        std::size_t access_rotation_bytes = 0;
        std::size_t access_slot_bytes = 0;

        if (!checked_count_bytes(
                shape.instruction_count,
                sizeof(
                    CudaTEdgeCompiledInstruction),
                &instruction_bytes) ||
            !checked_count_bytes(
                shape.term_count,
                sizeof(
                    CudaTEdgeCompiledTerm),
                &term_bytes) ||
            !checked_count_bytes(
                shape.constant_count,
                sizeof(GpuFr),
                &constant_bytes) ||
            !checked_count_bytes(
                shape.access_count,
                sizeof(std::uint8_t),
                &access_rotation_bytes) ||
            !checked_count_bytes(
                shape.access_count,
                sizeof(std::uint32_t),
                &access_slot_bytes))
        {
            return false;
        }

        std::size_t linear_source_slot_bytes = 0;
        std::size_t linear_source_scale_bytes = 0;
        std::size_t linear_source_offset_bytes = 0;
        std::size_t linear_source_count_bytes = 0;
        std::size_t linear_source_blinder_bytes = 0;
        std::size_t linear_target_blinder_bytes = 0;
        if (!checked_count_bytes(
                shape.max_linear_source_count,
                sizeof(std::uint32_t),
                &linear_source_slot_bytes) ||
            !checked_count_bytes(
                shape.max_linear_source_count,
                sizeof(GpuFr),
                &linear_source_scale_bytes) ||
            !checked_count_bytes(
                shape.max_linear_source_count,
                sizeof(std::size_t),
                &linear_source_offset_bytes) ||
            !checked_count_bytes(
                shape.max_linear_source_count,
                sizeof(std::size_t),
                &linear_source_count_bytes) ||
            !checked_count_bytes(
                shape.max_linear_source_blinder_count,
                sizeof(GpuFr),
                &linear_source_blinder_bytes) ||
            !checked_count_bytes(
                shape.max_linear_target_blinder_count,
                sizeof(GpuFr),
                &linear_target_blinder_bytes))
        {
            return false;
        }

        if (!add_device(leaf_cache_bytes) ||
            !add_device(table_bytes) ||
            !add_device(layout_bytes) ||
            !add_device(leaf_bytes) ||
            !add_device(cycle_count) ||
            !add_device(instruction_bytes) ||
            !add_device(term_bytes) ||
            !add_device(constant_bytes) ||
            !add_device(access_rotation_bytes) ||
            !add_device(access_slot_bytes) ||
            !add_device(linear_source_slot_bytes) ||
            !add_device(linear_source_scale_bytes) ||
            !add_device(linear_source_offset_bytes) ||
            !add_device(linear_source_count_bytes) ||
            !add_device(linear_source_blinder_bytes) ||
            !add_device(linear_target_blinder_bytes) ||
            !add_device(lane_power_bytes) ||
            !add_device(block_power_bytes) ||
            !add_device(kFixedMetadataBytes) ||
            !add_device(kAdmissionHeadroomBytes))
        {
            return false;
        }

        // Host/UMA allocations that are live at the same peak: the
        // constructor's temporary inverse cycle, persistent slot metadata,
        // prepare_program's rotated access vector, materialize_linear_leaf's
        // packed host arrays, and caller-owned output/staging vectors.
        std::size_t host_slot_bytes = 0;
        std::size_t host_bound_bytes = 0;
        std::size_t host_prefix_bytes = 0;
        std::size_t host_linear_seen_bytes = 0;
        if (!checked_count_bytes(
                cache_slot_count,
                sizeof(std::uint8_t),
                &host_slot_bytes) ||
            !checked_count_bytes(
                cache_slot_count,
                sizeof(std::uint8_t),
                &host_bound_bytes) ||
            !checked_count_bytes(
                cache_slot_count,
                sizeof(std::size_t),
                &host_prefix_bytes) ||
            !checked_count_bytes(
                cache_slot_count,
                sizeof(std::uint8_t),
                &host_linear_seen_bytes))
        {
            return false;
        }

        if (!add_host(host_slot_bytes) ||
            !add_host(host_bound_bytes) ||
            !add_host(host_prefix_bytes) ||
            !add_host(cycle_count) ||
            !add_host(access_rotation_bytes) ||
            !add_host(host_linear_seen_bytes) ||
            !add_host(linear_source_slot_bytes) ||
            !add_host(linear_source_scale_bytes) ||
            !add_host(linear_source_offset_bytes) ||
            !add_host(linear_source_count_bytes) ||
            !add_host(linear_source_blinder_bytes) ||
            !add_host(linear_target_blinder_bytes) ||
            !add_host(lane_power_bytes) ||
            !add_host(block_power_bytes) ||
            !add_host(shape.caller_host_bytes) ||
            !add_host(kFixedMetadataBytes))
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

        const auto workspace =
            algebra::cuda_ntt_workspace_residency();
        const bool same_workspace_device =
            workspace.device == device;
        const bool workspace_reusable =
            same_workspace_device &&
            workspace.capacity_bytes >= leaf_bytes &&
            workspace.has_input_output;
        const auto resident_workspace_bytes =
            same_workspace_device
                ? workspace.resident_bytes
                : std::size_t{0};
        const auto workspace_peak_bytes =
            workspace_reusable
                ? resident_workspace_bytes
                : ntt_workspace_bytes;
        const auto icicle_domain_growth_bytes =
            workspace.domain_initialized
                ? std::size_t{0}
                : icicle_domain_bytes;

        if (!checked_add_size(
                device_required_bytes,
                workspace_peak_bytes,
                &device_required_bytes) ||
            !checked_add_size(
                device_required_bytes,
                icicle_domain_growth_bytes,
                &device_required_bytes))
        {
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
            const auto host =
                util::cached_effective_memory_snapshot();

            // UMA execution shares host DRAM with CUDA allocations.  A
            // conservative total fallback is not a known available budget;
            // reject admission until a trustworthy host snapshot exists.
            if (!host.effective_available_known)
            {
                return false;
            }

            constexpr std::size_t
                kMinimumIntegratedReserve =
                    std::size_t{8}
                    << 30;

            constexpr std::size_t
                kIntegratedTransformHeadroom =
                    std::size_t{3}
                    << 30;

            std::size_t shared_required_bytes = 0;
            if (!checked_add_size(
                    device_required_bytes,
                    host_required_bytes,
                    &shared_required_bytes))
            {
                return false;
            }

            const auto reserve_bytes =
                std::max(
                    kMinimumIntegratedReserve,
                    !host.effective_total_known
                        ? kMinimumIntegratedReserve
                        : host.effective_total_bytes /
                              16);

            std::size_t minimum_available_bytes = 0;
            if (!checked_add_size(
                    reserve_bytes,
                    kIntegratedTransformHeadroom,
                    &minimum_available_bytes) ||
                host.effective_available_bytes <=
                    minimum_available_bytes)
            {
                return false;
            }

            std::size_t available_with_resident =
                host.effective_available_bytes;
            if (resident_workspace_bytes != 0 &&
                !checked_add_size(
                    available_with_resident,
                    resident_workspace_bytes,
                    &available_with_resident))
            {
                return false;
            }

            const auto safely_usable =
                available_with_resident -
                reserve_bytes -
                kIntegratedTransformHeadroom;

            return shared_required_bytes <= safely_usable;
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

        std::size_t device_available_with_resident =
            free_bytes;
        if (resident_workspace_bytes != 0 &&
            !checked_add_size(
                device_available_with_resident,
                resident_workspace_bytes,
                &device_available_with_resident))
        {
            return false;
        }

        constexpr std::size_t
            kMinimumCudaReserveBytes =
                std::size_t{4} << 30;

        const auto reserve_bytes =
            std::max(
                kMinimumCudaReserveBytes,
                total_bytes / 8);

        if (device_available_with_resident <= reserve_bytes)
        {
            return false;
        }

        if (device_required_bytes >
            device_available_with_resident -
                reserve_bytes)
        {
            return false;
        }

        const auto host =
            util::cached_effective_memory_snapshot();
        if (!host.effective_available_known)
        {
            return false;
        }

        constexpr std::size_t kMinimumHostReserveBytes =
            std::size_t{8} << 30;
        const auto host_reserve =
            std::max(
                kMinimumHostReserveBytes,
                host.effective_total_known
                    ? host.effective_total_bytes / 16
                    : kMinimumHostReserveBytes);
        if (host.effective_available_bytes <= host_reserve)
        {
            return false;
        }

        return host_required_bytes <=
               host.effective_available_bytes -
                   host_reserve;
    }

    bool cuda_t_edge_update_resident_coefficient_ranges(
        const algebra::CudaResidentCoefficientBuffer &resident,
        const algebra::FieldElement *host_coefficients,
        std::size_t host_coefficient_count,
        std::size_t first_offset,
        std::size_t first_count,
        std::size_t second_offset,
        std::size_t second_count)
    {
        if (resident.allocation == nullptr ||
            resident.device_montgomery_limbs == nullptr ||
            resident.device < 0 ||
            host_coefficients == nullptr)
        {
            throw std::runtime_error(
                "CUDA t_edge resident coefficient update received invalid storage");
        }

        const auto range_valid =
            [&](std::size_t offset,
                std::size_t count)
        {
            return offset <=
                       host_coefficient_count &&
                   count <=
                       host_coefficient_count -
                           offset &&
                   offset <=
                       resident.capacity &&
                   count <=
                       resident.capacity -
                           offset &&
                   count <=
                       std::numeric_limits<
                           std::size_t>::max() /
                           sizeof(mcl::Fr);
        };

        if (!range_valid(
                first_offset,
                first_count) ||
            !range_valid(
                second_offset,
                second_count))
        {
            throw std::runtime_error(
                "CUDA t_edge resident coefficient update range exceeds storage");
        }

        if (first_count == 0 &&
            second_count == 0)
        {
            return true;
        }

        int previous_device = -1;

        auto status =
            cudaGetDevice(
                &previous_device);

        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(
                    "CUDA t_edge resident coefficient device lookup failed: ") +
                cudaGetErrorString(
                    status));
        }

        const bool switched_device =
            previous_device !=
            resident.device;

        if (switched_device)
        {
            status =
                cudaSetDevice(
                    resident.device);

            if (status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(
                        "CUDA t_edge resident coefficient device selection failed: ") +
                    cudaGetErrorString(
                        status));
            }
        }

        auto *device_coefficients =
            reinterpret_cast<mcl::Fr *>(
                const_cast<std::uint64_t *>(
                    resident
                        .device_montgomery_limbs));

        const auto *native_host_coefficients =
            algebra::FieldElement::
                native_array_data(
                    host_coefficients);

        const auto copy_range =
            [&](std::size_t offset,
                std::size_t count)
        {
            if (count == 0)
            {
                return cudaSuccess;
            }

            const auto byte_count =
                count *
                sizeof(mcl::Fr);

            const auto copy_status =
                cudaMemcpy(
                    device_coefficients +
                        offset,
                    native_host_coefficients +
                        offset,
                    byte_count,
                    cudaMemcpyHostToDevice);

            if (copy_status ==
                cudaSuccess)
            {
                algebra::record_cuda_h2d(
                    byte_count);
            }

            return copy_status;
        };

        auto copy_status =
            copy_range(
                first_offset,
                first_count);

        if (copy_status ==
            cudaSuccess)
        {
            copy_status =
                copy_range(
                    second_offset,
                    second_count);
        }

        cudaError_t restore_status =
            cudaSuccess;

        if (switched_device)
        {
            restore_status =
                cudaSetDevice(
                    previous_device);
        }

        if (copy_status !=
            cudaSuccess)
        {
            throw std::runtime_error(
                std::string(
                    "CUDA t_edge resident coefficient update failed: ") +
                cudaGetErrorString(
                    copy_status));
        }

        if (restore_status !=
            cudaSuccess)
        {
            throw std::runtime_error(
                std::string(
                    "CUDA t_edge resident coefficient device restore failed: ") +
                cudaGetErrorString(
                    restore_status));
        }

        return true;
    }

    void release_cuda_t_edge_executor_cache()
    {
    }
}
