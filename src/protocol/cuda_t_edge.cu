#include "gatzk/protocol/cuda_t_edge.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

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

                    stack[stack_size++] =
                        leaf_slots[slot]
                                  [logical_index];

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

            quotient[index] =
                algebra::gpu::fr_mul(
                    quotient[index],
                    zero_inverse_cycle[index % cycle_size]);
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
                std::size_t cache_slot_count,
                CudaTEdgeExecutionProfile *profile)
                : domain_size_(domain_size),
                  target_size_(target_size),
                  rotation_step_(
                      target_size /
                      domain_size),
                  cache_slot_count_(
                      cache_slot_count),
                  profile_(profile)
            {
                (void)omega;

                if (domain_size_ == 0 ||
                    target_size_ == 0 ||
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

                if (profile_ != nullptr)
                {
                    profile_->leaf_cache_slot_count =
                        cache_slot_count_;
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

                if (profile_ != nullptr)
                {
                    profile_->term_count =
                        program.terms.size();

                    profile_->access_count =
                        program.accesses.size();
                }
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

                if (profile_ != nullptr)
                {
                    profile_->source_bytes_uploaded +=
                        byte_count;

                    if (logical_offset == 0)
                    {
                        ++profile_->leaf_materialization_count;
                    }
                }
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
                            cache_slot_count_)
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

                algebra::
                    record_cuda_kernel_invocations(
                        1);

                if (profile_ != nullptr)
                {
                    profile_->quotient_bytes_downloaded =
                        byte_count;

                    std::cerr
                        << "[gatzk] cuda_t_edge"
                        << " terms="
                        << profile_->term_count
                        << " accesses="
                        << profile_->access_count
                        << " unique_leaves="
                        << profile_->unique_leaf_count
                        << " cache_slots="
                        << profile_->leaf_cache_slot_count
                        << " materializations="
                        << profile_->leaf_materialization_count
                        << " cache_hits="
                        << profile_->leaf_cache_hit_count
                        << " evictions="
                        << profile_->leaf_eviction_count
                        << " source_upload_bytes="
                        << profile_->source_bytes_uploaded
                        << " quotient_download_bytes="
                        << profile_->quotient_bytes_downloaded
                        << '\n';
                }
            }

        private:
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

                device_quotient_ =
                    nullptr;

                device_zero_inverse_cycle_ =
                    nullptr;
            }

            std::size_t domain_size_ = 0;
            std::size_t target_size_ = 0;
            std::size_t rotation_step_ = 0;
            std::size_t cache_slot_count_ = 0;

            std::vector<GpuFr *>
                device_leaf_slots_;

            GpuFr **device_leaf_table_ =
                nullptr;

            GpuFr *device_quotient_ =
                nullptr;

            GpuFr *device_zero_inverse_cycle_ =
                nullptr;

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

            CudaTEdgeExecutionProfile
                *profile_ = nullptr;
        };
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

    bool cuda_t_edge_executor_can_admit(
        std::size_t target_size,
        std::size_t cache_slot_count)
    {
        if (!cuda_t_edge_executor_available() ||
            target_size == 0 ||
            cache_slot_count == 0)
        {
            return false;
        }

        if (target_size >
            std::numeric_limits<
                std::size_t>::max() /
                sizeof(GpuFr))
        {
            return false;
        }

        const auto leaf_bytes =
            target_size *
            sizeof(GpuFr);

        if (cache_slot_count >
            std::numeric_limits<
                std::size_t>::max() /
                leaf_bytes)
        {
            return false;
        }

        const auto leaf_cache_bytes =
            cache_slot_count *
            leaf_bytes;

        if (leaf_cache_bytes >
            std::numeric_limits<
                std::size_t>::max() -
                leaf_bytes)
        {
            return false;
        }

        const auto required_bytes =
            leaf_cache_bytes +
            leaf_bytes;

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
                        std::size_t{16}
                        << 30;

                const auto reserve_bytes =
                    std::max(
                        kMinimumIntegratedReserve,
                        host_total_bytes == 0
                            ? kMinimumIntegratedReserve
                            : host_total_bytes /
                                  4);

                if (host_available_bytes <=
                    reserve_bytes)
                {
                    return false;
                }

                if (required_bytes >
                    host_available_bytes -
                        reserve_bytes)
                {
                    return false;
                }
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

        if (free_bytes <=
            reserve_bytes)
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