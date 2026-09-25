#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <sstream>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/algebra/cuda_weighted_session.hpp"
#include "gatzk/util/effective_memory.hpp"
#include "gatzk/util/route2.hpp"

// 实现四个GPU加速组件，均为工程优化:
//   1. 重心权重构造 (barycentric_weights_gpu)
//   2. 批量加权多项式求值 (weighted_poly_eval_gpu*)
//   3. Lookup 直方图统计 (lookup_histogram_indices_cuda)
//   4. 超大 opening evaluation source 的 scaled accumulate

namespace gatzk::algebra
{
    // 检查 CUDA 运行时是否有可用设备
    bool cuda_backend_runtime_available()
    {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        return status == cudaSuccess && device_count > 0;
    }

    namespace
    {
        __global__ void scaled_accumulate_kernel(
            gpu::GpuFr *target,
            const gpu::GpuFr *source,
            gpu::GpuFr scale,
            std::size_t element_count)
        {
            const auto index =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= element_count)
            {
                return;
            }
            target[index] = gpu::fr_add(
                target[index], gpu::fr_mul(source[index], scale));
        }

        __global__ void scaled_accumulate_pair_kernel(
            gpu::GpuFr *first_target,
            gpu::GpuFr *second_target,
            const gpu::GpuFr *source,
            gpu::GpuFr first_scale,
            gpu::GpuFr second_scale,
            std::size_t element_count)
        {
            const auto index =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                threadIdx.x;

            if (index >= element_count)
            {
                return;
            }

            const auto source_value =
                source[index];

            first_target[index] =
                gpu::fr_add(
                    first_target[index],
                    gpu::fr_mul(
                        source_value,
                        first_scale));

            second_target[index] =
                gpu::fr_add(
                    second_target[index],
                    gpu::fr_mul(
                        source_value,
                        second_scale));
        }

        [[noreturn]] void throw_scaled_accumulate_cuda_error(
            const char *operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string("CUDA scaled accumulate ") + operation + " failed: " + cudaGetErrorString(status));
        }

        class ScaledAccumulateCudaWorkspace
        {
        public:
            cudaStream_t stream()
            {
                if (stream_ == nullptr)
                {
                    const auto status =
                        cudaStreamCreateWithFlags(
                            &stream_,
                            cudaStreamNonBlocking);

                    if (status != cudaSuccess)
                    {
                        stream_ = nullptr;

                        throw_scaled_accumulate_cuda_error(
                            "stream creation",
                            status);
                    }
                }

                return stream_;
            }

            bool ensure(
                std::size_t element_count)
            {
                if (element_count <= capacity_ &&
                    first_target_ != nullptr &&
                    second_target_ != nullptr &&
                    source_ != nullptr)
                {
                    return true;
                }

                const auto active_stream =
                    stream();

                auto status =
                    cudaStreamSynchronize(
                        active_stream);

                if (status != cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "workspace resize synchronization",
                        status);
                }

                release_buffers();

                if (element_count == 0)
                {
                    return true;
                }

                if (element_count >
                    std::numeric_limits<std::size_t>::max() /
                        sizeof(gpu::GpuFr))
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate workspace byte-size overflow");
                }

                const auto byte_count =
                    element_count *
                    sizeof(gpu::GpuFr);

                status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &first_target_),
                        byte_count);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();
                    release_buffers();
                    return false;
                }

                if (status != cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "first target allocation",
                        status);
                }

                status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &second_target_),
                        byte_count);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();
                    release_buffers();
                    return false;
                }

                if (status != cudaSuccess)
                {
                    release_buffers();

                    throw_scaled_accumulate_cuda_error(
                        "second target allocation",
                        status);
                }

                status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &source_),
                        byte_count);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();
                    release_buffers();
                    return false;
                }

                if (status != cudaSuccess)
                {
                    release_buffers();

                    throw_scaled_accumulate_cuda_error(
                        "source allocation",
                        status);
                }

                capacity_ =
                    element_count;

                return true;
            }

            gpu::GpuFr *first_target()
            {
                return first_target_;
            }

            gpu::GpuFr *second_target()
            {
                return second_target_;
            }

            gpu::GpuFr *source()
            {
                return source_;
            }

            void release() noexcept
            {
                if (stream_ != nullptr)
                {
                    (void)cudaStreamSynchronize(
                        stream_);
                }

                release_buffers();

                if (stream_ != nullptr)
                {
                    (void)cudaStreamDestroy(
                        stream_);

                    stream_ = nullptr;
                }
            }

            ~ScaledAccumulateCudaWorkspace()
            {
                release();
            }

        private:
            void release_buffers() noexcept
            {
                if (first_target_ != nullptr)
                {
                    (void)cudaFree(
                        first_target_);
                }

                if (second_target_ != nullptr)
                {
                    (void)cudaFree(
                        second_target_);
                }

                if (source_ != nullptr)
                {
                    (void)cudaFree(
                        source_);
                }

                first_target_ = nullptr;
                second_target_ = nullptr;
                source_ = nullptr;
                capacity_ = 0;
            }

            gpu::GpuFr *first_target_ = nullptr;
            gpu::GpuFr *second_target_ = nullptr;
            gpu::GpuFr *source_ = nullptr;

            std::size_t capacity_ = 0;

            cudaStream_t stream_ = nullptr;
        };

        ScaledAccumulateCudaWorkspace &
        scaled_accumulate_workspace()
        {
            thread_local ScaledAccumulateCudaWorkspace
                workspace;

            return workspace;
        }

        class ScaledAccumulateStreamPool
        {
        public:
            cudaStream_t acquire()
            {
                const auto index =
                    next_stream_ %
                    streams_.size();

                ++next_stream_;

                if (streams_[index] ==
                    nullptr)
                {
                    const auto status =
                        cudaStreamCreateWithFlags(
                            &streams_[index],
                            cudaStreamNonBlocking);

                    if (status != cudaSuccess)
                    {
                        streams_[index] =
                            nullptr;

                        throw_scaled_accumulate_cuda_error(
                            "persistent stream creation",
                            status);
                    }
                }

                return streams_[index];
            }

            void release() noexcept
            {
                for (auto &stream :
                     streams_)
                {
                    if (stream == nullptr)
                    {
                        continue;
                    }

                    (void)cudaStreamSynchronize(
                        stream);

                    (void)cudaStreamDestroy(
                        stream);

                    stream = nullptr;
                }

                next_stream_ = 0;
            }

            ~ScaledAccumulateStreamPool()
            {
                release();
            }

        private:
            std::array<
                cudaStream_t,
                4>
                streams_{};

            std::size_t next_stream_ = 0;
        };

        ScaledAccumulateStreamPool &
        scaled_accumulate_stream_pool()
        {
            thread_local ScaledAccumulateStreamPool
                pool;

            return pool;
        }

        class EphemeralScaledAccumulateSourceStaging
        {
        public:
            EphemeralScaledAccumulateSourceStaging() =
                default;

            EphemeralScaledAccumulateSourceStaging(
                const EphemeralScaledAccumulateSourceStaging &) =
                delete;

            EphemeralScaledAccumulateSourceStaging &
            operator=(
                const EphemeralScaledAccumulateSourceStaging &) =
                delete;

            ~EphemeralScaledAccumulateSourceStaging()
            {
                release_noexcept();
            }

            const gpu::GpuFr *enqueue(
                gpu::GpuFr *device_source,
                const mcl::Fr *source,
                std::size_t element_count,
                cudaStream_t stream)
            {
                if (device_source == nullptr ||
                    source == nullptr ||
                    element_count == 0 ||
                    stream == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA ephemeral scaled-accumulate staging received invalid input");
                }

                static_assert(
                    sizeof(mcl::Fr) ==
                        sizeof(gpu::GpuFr),
                    "CUDA ephemeral scaled-accumulate staging requires matching field widths");

                if (element_count >
                    std::numeric_limits<
                        std::size_t>::max() /
                        sizeof(gpu::GpuFr))
                {
                    throw std::runtime_error(
                        "CUDA ephemeral scaled-accumulate staging byte-size overflow");
                }

                const auto byte_count =
                    element_count *
                    sizeof(gpu::GpuFr);

                auto &slot =
                    slots_[next_slot_ %
                           slots_.size()];

                ++next_slot_;

                if (slot.copy_done ==
                    nullptr)
                {
                    const auto status =
                        cudaEventCreateWithFlags(
                            &slot.copy_done,
                            cudaEventDisableTiming);

                    if (status !=
                        cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "ephemeral staging event creation",
                            status);
                    }
                }

                if (slot.copy_in_flight)
                {
                    const auto status =
                        cudaEventSynchronize(
                            slot.copy_done);

                    if (status !=
                        cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "ephemeral staging slot reuse",
                            status);
                    }

                    slot.copy_in_flight =
                        false;
                }

                if (slot.capacity_bytes <
                    byte_count)
                {
                    if (slot.host_bytes !=
                        nullptr)
                    {
                        const auto status =
                            cudaFreeHost(
                                slot.host_bytes);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_scaled_accumulate_cuda_error(
                                "ephemeral staging host release",
                                status);
                        }

                        slot.host_bytes =
                            nullptr;

                        slot.capacity_bytes =
                            0;
                    }

                    auto status =
                        cudaMallocHost(
                            &slot.host_bytes,
                            byte_count);

                    if (status ==
                        cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();

                        status =
                            cudaMemcpyAsync(
                                device_source,
                                source,
                                byte_count,
                                cudaMemcpyHostToDevice,
                                stream);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_scaled_accumulate_cuda_error(
                                "ephemeral fallback source upload",
                                status);
                        }

                        record_cuda_h2d(
                            byte_count);

                        status =
                            cudaStreamSynchronize(
                                stream);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_scaled_accumulate_cuda_error(
                                "ephemeral fallback source completion",
                                status);
                        }

                        record_cuda_stream_sync();

                        return device_source;
                    }

                    if (status !=
                        cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "ephemeral staging host allocation",
                            status);
                    }

                    slot.capacity_bytes =
                        byte_count;
                }

                std::memcpy(
                    slot.host_bytes,
                    source,
                    byte_count);

                auto status =
                    cudaMemcpyAsync(
                        device_source,
                        slot.host_bytes,
                        byte_count,
                        cudaMemcpyHostToDevice,
                        stream);

                if (status !=
                    cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "ephemeral staged source upload",
                        status);
                }

                record_cuda_h2d(
                    byte_count);

                status =
                    cudaEventRecord(
                        slot.copy_done,
                        stream);

                if (status !=
                    cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "ephemeral staging copy event",
                        status);
                }

                slot.copy_in_flight =
                    true;

                return device_source;
            }

        private:
            struct Slot
            {
                void *host_bytes =
                    nullptr;

                std::size_t capacity_bytes =
                    0;

                cudaEvent_t copy_done =
                    nullptr;

                bool copy_in_flight =
                    false;
            };

            void release_noexcept() noexcept
            {
                for (auto &slot :
                     slots_)
                {
                    if (slot.copy_done !=
                            nullptr &&
                        slot.copy_in_flight)
                    {
                        (void)cudaEventSynchronize(
                            slot.copy_done);

                        slot.copy_in_flight =
                            false;
                    }

                    if (slot.copy_done !=
                        nullptr)
                    {
                        (void)cudaEventDestroy(
                            slot.copy_done);

                        slot.copy_done =
                            nullptr;
                    }

                    if (slot.host_bytes !=
                        nullptr)
                    {
                        (void)cudaFreeHost(
                            slot.host_bytes);

                        slot.host_bytes =
                            nullptr;
                    }

                    slot.capacity_bytes =
                        0;
                }

                next_slot_ = 0;
            }

            std::array<Slot, 2>
                slots_{};

            std::size_t next_slot_ = 0;
        };

        class CudaScaledAccumulateSessionImpl final
            : public CudaScaledAccumulateSession
        {
        public:
            CudaScaledAccumulateSessionImpl(
                mcl::Fr *host_target,
                std::size_t element_count,
                bool target_is_zero)
                : host_target_(host_target),
                  element_count_(element_count),
                  zero_copy_target_(
                      cuda_backend_unified_addressing_available())
            {
                if (host_target_ == nullptr ||
                    element_count_ == 0)
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate session received invalid target storage");
                }

                if (element_count_ >
                    std::numeric_limits<std::size_t>::max() /
                        sizeof(gpu::GpuFr))
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate session byte-size overflow");
                }

                stream_ =
                    scaled_accumulate_stream_pool()
                        .acquire();

                auto status =
                    cudaSuccess;

                source_capacity_ =
                    std::min<std::size_t>(
                        element_count_,
                        std::size_t{1} << 20);

                const auto source_bytes =
                    source_capacity_ *
                    sizeof(gpu::GpuFr);

                status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &device_source_),
                        source_bytes);

                if (status !=
                    cudaSuccess)
                {
                    release_noexcept();

                    throw_scaled_accumulate_cuda_error(
                        "session source allocation",
                        status);
                }

                if (!zero_copy_target_)
                {
                    const auto target_bytes =
                        element_count_ *
                        sizeof(gpu::GpuFr);

                    status =
                        cudaMalloc(
                            reinterpret_cast<void **>(
                                &device_target_),
                            target_bytes);

                    if (status != cudaSuccess)
                    {
                        release_noexcept();

                        throw_scaled_accumulate_cuda_error(
                            "session target allocation",
                            status);
                    }

                    if (target_is_zero)
                    {
                        status =
                            cudaMemsetAsync(
                                device_target_,
                                0,
                                target_bytes,
                                stream_);

                        if (status != cudaSuccess)
                        {
                            release_noexcept();

                            throw_scaled_accumulate_cuda_error(
                                "session target initialization",
                                status);
                        }
                    }
                    else
                    {
                        status =
                            cudaMemcpyAsync(
                                device_target_,
                                host_target_,
                                target_bytes,
                                cudaMemcpyHostToDevice,
                                stream_);

                        if (status != cudaSuccess)
                        {
                            release_noexcept();

                            throw_scaled_accumulate_cuda_error(
                                "session target upload",
                                status);
                        }

                        record_cuda_h2d(
                            target_bytes);
                    }
                }
            }

            ~CudaScaledAccumulateSessionImpl() override
            {
                if (!finished_)
                {
                    try
                    {
                        finish();
                    }
                    catch (...)
                    {
                    }
                }

                release_noexcept();
            }

            void accumulate_range(
                const mcl::Fr *source,
                const mcl::Fr &scale,
                std::size_t target_offset,
                std::size_t element_count,
                bool source_is_ephemeral) override
            {
                if (finished_ ||
                    source == nullptr ||
                    target_offset > element_count_ ||
                    element_count >
                        element_count_ -
                            target_offset)
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate session received invalid range");
                }

                if (element_count == 0 ||
                    scale.isZero())
                {
                    return;
                }

                gpu::GpuFr gpu_scale{};

                std::memcpy(
                    gpu_scale.d,
                    &scale,
                    sizeof(gpu_scale.d));

                constexpr unsigned
                    kBlockSize = 256;

                std::size_t processed = 0;

                while (processed <
                       element_count)
                {
                    const bool stage_source =
                        source_is_ephemeral ||
                        !zero_copy_target_;

                    const auto current_count =
                        stage_source
                            ? std::min(
                                  source_capacity_,
                                  element_count -
                                      processed)
                            : element_count -
                                  processed;

                    const auto byte_count =
                        current_count *
                        sizeof(gpu::GpuFr);

                    const gpu::GpuFr
                        *source_values =
                            nullptr;

                    cudaError_t status =
                        cudaSuccess;

                    if (!stage_source)
                    {
                        source_values =
                            reinterpret_cast<
                                const gpu::GpuFr *>(
                                source +
                                processed);
                    }
                    else if (source_is_ephemeral)
                    {
                        source_values =
                            ephemeral_source_staging_
                                .enqueue(
                                    device_source_,
                                    source +
                                        processed,
                                    current_count,
                                    stream_);
                    }
                    else
                    {
                        status =
                            cudaMemcpyAsync(
                                device_source_,
                                source +
                                    processed,
                                byte_count,
                                cudaMemcpyHostToDevice,
                                stream_);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_scaled_accumulate_cuda_error(
                                "session source upload",
                                status);
                        }

                        record_cuda_h2d(
                            byte_count);

                        source_values =
                            device_source_;
                    }

                    const auto block_count_size =
                        (current_count +
                         kBlockSize - 1) /
                        kBlockSize;

                    if (block_count_size >
                        std::numeric_limits<
                            unsigned>::max())
                    {
                        throw std::runtime_error(
                            "CUDA scaled accumulate session grid exceeds CUDA launch limits");
                    }

                    auto *target =
                        zero_copy_target_
                            ? reinterpret_cast<
                                  gpu::GpuFr *>(
                                  host_target_ +
                                  target_offset +
                                  processed)
                            : device_target_ +
                                  target_offset +
                                  processed;

                    scaled_accumulate_kernel<<<
                        static_cast<unsigned>(
                            block_count_size),
                        kBlockSize,
                        0,
                        stream_>>>(
                        target,
                        source_values,
                        gpu_scale,
                        current_count);

                    status =
                        cudaGetLastError();

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "session kernel launch",
                            status);
                    }

                    record_cuda_kernel_invocations(
                        1);

                    processed +=
                        current_count;
                }

                record_cuda_opening_fold_call();
            }

            void accumulate_device_range(
                const std::uint64_t *device_montgomery_source,
                int source_device,
                const mcl::Fr &scale,
                std::size_t target_offset,
                std::size_t element_count) override
            {
                if (finished_)
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate session is already finished");
                }

                if (element_count == 0 ||
                    scale.isZero())
                {
                    return;
                }

                if (device_montgomery_source ==
                    nullptr)
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate resident source is null");
                }

                if (target_offset >
                        element_count_ ||
                    element_count >
                        element_count_ -
                            target_offset)
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate resident source exceeds target");
                }

                int current_device = -1;

                auto status =
                    cudaGetDevice(
                        &current_device);

                if (status !=
                    cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "resident source device lookup",
                        status);
                }

                if (source_device !=
                    current_device)
                {
                    throw std::runtime_error(
                        "CUDA scaled accumulate resident source belongs to another device");
                }

                gpu::GpuFr gpu_scale{};

                std::memcpy(
                    gpu_scale.d,
                    &scale,
                    sizeof(gpu_scale.d));

                constexpr unsigned
                    kBlockSize = 256;

                constexpr std::size_t
                    kMaximumElementsPerLaunch =
                        static_cast<std::size_t>(
                            std::numeric_limits<
                                unsigned>::max()) *
                        kBlockSize;

                std::size_t processed = 0;

                while (processed <
                       element_count)
                {
                    const auto current_count =
                        std::min(
                            element_count -
                                processed,
                            kMaximumElementsPerLaunch);

                    const auto block_count_size =
                        (current_count +
                         kBlockSize - 1) /
                        kBlockSize;

                    if (block_count_size >
                        std::numeric_limits<
                            unsigned>::max())
                    {
                        throw std::runtime_error(
                            "CUDA resident scaled accumulate grid exceeds CUDA launch limits");
                    }

                    auto *target =
                        zero_copy_target_
                            ? reinterpret_cast<
                                  gpu::GpuFr *>(
                                  host_target_ +
                                  target_offset +
                                  processed)
                            : device_target_ +
                                  target_offset +
                                  processed;

                    const auto *source =
                        reinterpret_cast<
                            const gpu::GpuFr *>(
                            device_montgomery_source) +
                        processed;

                    scaled_accumulate_kernel<<<
                        static_cast<unsigned>(
                            block_count_size),
                        kBlockSize,
                        0,
                        stream_>>>(
                        target,
                        source,
                        gpu_scale,
                        current_count);

                    status =
                        cudaGetLastError();

                    if (status !=
                        cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "resident session kernel launch",
                            status);
                    }

                    record_cuda_kernel_invocations(
                        1);

                    processed +=
                        current_count;
                }

                record_cuda_opening_fold_call();
            }

            void finish() override
            {
                if (finished_)
                {
                    return;
                }

                if (!zero_copy_target_)
                {
                    const auto target_bytes =
                        element_count_ *
                        sizeof(gpu::GpuFr);

                    const auto status =
                        cudaMemcpyAsync(
                            host_target_,
                            device_target_,
                            target_bytes,
                            cudaMemcpyDeviceToHost,
                            stream_);

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "session target download",
                            status);
                    }

                    record_cuda_d2h(
                        target_bytes);
                }

                const auto status =
                    cudaStreamSynchronize(
                        stream_);

                if (status != cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "session synchronization",
                        status);
                }
                record_cuda_stream_sync();

                finished_ = true;
            }

        private:
            void release_noexcept() noexcept
            {
                if (stream_ != nullptr &&
                    !finished_)
                {
                    (void)cudaStreamSynchronize(
                        stream_);
                }

                if (device_source_ != nullptr)
                {
                    (void)cudaFree(
                        device_source_);

                    device_source_ =
                        nullptr;
                }

                if (device_target_ != nullptr)
                {
                    (void)cudaFree(
                        device_target_);

                    device_target_ =
                        nullptr;
                }

                stream_ = nullptr;
            }

            mcl::Fr *host_target_ = nullptr;
            std::size_t element_count_ = 0;
            std::size_t source_capacity_ = 0;

            bool zero_copy_target_ = false;
            bool finished_ = false;

            gpu::GpuFr *device_target_ =
                nullptr;

            gpu::GpuFr *device_source_ =
                nullptr;

            EphemeralScaledAccumulateSourceStaging
                ephemeral_source_staging_;

            cudaStream_t stream_ =
                nullptr;
        };

        class CudaScaledAccumulatePairSessionImpl final
            : public CudaScaledAccumulatePairSession
        {
        public:
            CudaScaledAccumulatePairSessionImpl(
                mcl::Fr *first_host_target,
                mcl::Fr *second_host_target,
                std::size_t element_count,
                bool first_target_is_zero,
                bool second_target_is_zero)
                : first_host_target_(first_host_target),
                  second_host_target_(second_host_target),
                  element_count_(element_count),
                  zero_copy_targets_(
                      cuda_backend_unified_addressing_available())
            {
                if (first_host_target_ == nullptr ||
                    second_host_target_ == nullptr ||
                    first_host_target_ == second_host_target_ ||
                    element_count_ == 0)
                {
                    throw std::runtime_error(
                        "CUDA paired scaled accumulate session received invalid target storage");
                }

                if (element_count_ >
                    std::numeric_limits<std::size_t>::max() /
                        sizeof(gpu::GpuFr))
                {
                    throw std::runtime_error(
                        "CUDA paired scaled accumulate session byte-size overflow");
                }

                target_bytes_ =
                    element_count_ *
                    sizeof(gpu::GpuFr);

                source_capacity_ =
                    std::min<std::size_t>(
                        element_count_,
                        std::size_t{1} << 20);

                stream_ =
                    scaled_accumulate_stream_pool()
                        .acquire();

                auto status =
                    cudaSuccess;

                try
                {
                    status =
                        cudaMalloc(
                            reinterpret_cast<void **>(
                                &device_source_),
                            source_capacity_ *
                                sizeof(gpu::GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session source allocation",
                            status);
                    }

                    if (zero_copy_targets_)
                    {
                        first_device_target_ =
                            reinterpret_cast<
                                gpu::GpuFr *>(
                                first_host_target_);

                        second_device_target_ =
                            reinterpret_cast<
                                gpu::GpuFr *>(
                                second_host_target_);

                        return;
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<void **>(
                                &first_device_target_),
                            target_bytes_);

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session first target allocation",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<void **>(
                                &second_device_target_),
                            target_bytes_);

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session second target allocation",
                            status);
                    }

                    if (first_target_is_zero)
                    {
                        status =
                            cudaMemsetAsync(
                                first_device_target_,
                                0,
                                target_bytes_,
                                stream_);
                    }
                    else
                    {
                        status =
                            cudaMemcpyAsync(
                                first_device_target_,
                                first_host_target_,
                                target_bytes_,
                                cudaMemcpyHostToDevice,
                                stream_);

                        if (status == cudaSuccess)
                        {
                            record_cuda_h2d(
                                target_bytes_);
                        }
                    }

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session first target initialization",
                            status);
                    }

                    if (second_target_is_zero)
                    {
                        status =
                            cudaMemsetAsync(
                                second_device_target_,
                                0,
                                target_bytes_,
                                stream_);
                    }
                    else
                    {
                        status =
                            cudaMemcpyAsync(
                                second_device_target_,
                                second_host_target_,
                                target_bytes_,
                                cudaMemcpyHostToDevice,
                                stream_);

                        if (status == cudaSuccess)
                        {
                            record_cuda_h2d(
                                target_bytes_);
                        }
                    }

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session second target initialization",
                            status);
                    }
                }
                catch (...)
                {
                    release_noexcept();
                    throw;
                }
            }

            ~CudaScaledAccumulatePairSessionImpl() override
            {
                if (!finished_)
                {
                    try
                    {
                        finish();
                    }
                    catch (...)
                    {
                    }
                }

                release_noexcept();
            }

            void accumulate_range(
                const mcl::Fr *source,
                const mcl::Fr &first_scale,
                const mcl::Fr &second_scale,
                std::size_t target_offset,
                std::size_t element_count,
                bool source_is_ephemeral) override
            {
                if (finished_ ||
                    source == nullptr ||
                    target_offset > element_count_ ||
                    element_count >
                        element_count_ -
                            target_offset)
                {
                    throw std::runtime_error(
                        "CUDA paired scaled accumulate session received invalid range");
                }

                if (element_count == 0 ||
                    (first_scale.isZero() &&
                     second_scale.isZero()))
                {
                    return;
                }

                gpu::GpuFr first_gpu_scale{};
                gpu::GpuFr second_gpu_scale{};

                std::memcpy(
                    first_gpu_scale.d,
                    &first_scale,
                    sizeof(first_gpu_scale.d));

                std::memcpy(
                    second_gpu_scale.d,
                    &second_scale,
                    sizeof(second_gpu_scale.d));

                constexpr unsigned
                    kBlockSize = 256;

                std::size_t processed = 0;

                while (processed <
                       element_count)
                {
                    const bool stage_source =
                        source_is_ephemeral ||
                        !zero_copy_targets_;

                    const auto current_count =
                        stage_source
                            ? std::min(
                                  source_capacity_,
                                  element_count -
                                      processed)
                            : element_count -
                                  processed;

                    const auto byte_count =
                        current_count *
                        sizeof(gpu::GpuFr);

                    const gpu::GpuFr
                        *source_values =
                            nullptr;

                    cudaError_t status =
                        cudaSuccess;

                    if (!stage_source)
                    {
                        source_values =
                            reinterpret_cast<
                                const gpu::GpuFr *>(
                                source +
                                processed);
                    }
                    else if (source_is_ephemeral)
                    {
                        source_values =
                            ephemeral_source_staging_
                                .enqueue(
                                    device_source_,
                                    source +
                                        processed,
                                    current_count,
                                    stream_);
                    }
                    else
                    {
                        status =
                            cudaMemcpyAsync(
                                device_source_,
                                source +
                                    processed,
                                byte_count,
                                cudaMemcpyHostToDevice,
                                stream_);

                        if (status !=
                            cudaSuccess)
                        {
                            throw_scaled_accumulate_cuda_error(
                                "paired session source upload",
                                status);
                        }

                        record_cuda_h2d(
                            byte_count);

                        source_values =
                            device_source_;
                    }

                    const auto block_count_size =
                        (current_count +
                         kBlockSize - 1) /
                        kBlockSize;

                    if (block_count_size >
                        std::numeric_limits<
                            unsigned>::max())
                    {
                        throw std::runtime_error(
                            "CUDA paired scaled accumulate session grid exceeds CUDA launch limits");
                    }

                    scaled_accumulate_pair_kernel<<<
                        static_cast<unsigned>(
                            block_count_size),
                        kBlockSize,
                        0,
                        stream_>>>(
                        first_device_target_ +
                            target_offset +
                            processed,
                        second_device_target_ +
                            target_offset +
                            processed,
                        source_values,
                        first_gpu_scale,
                        second_gpu_scale,
                        current_count);

                    status =
                        cudaGetLastError();

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session kernel launch",
                            status);
                    }

                    record_cuda_kernel_invocations(
                        1);

                    processed +=
                        current_count;
                }

                record_cuda_opening_fold_call();
            }

            void finish() override
            {
                if (finished_)
                {
                    return;
                }

                if (!zero_copy_targets_)
                {
                    auto status =
                        cudaMemcpyAsync(
                            first_host_target_,
                            first_device_target_,
                            target_bytes_,
                            cudaMemcpyDeviceToHost,
                            stream_);

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session first target download",
                            status);
                    }

                    record_cuda_d2h(
                        target_bytes_);

                    status =
                        cudaMemcpyAsync(
                            second_host_target_,
                            second_device_target_,
                            target_bytes_,
                            cudaMemcpyDeviceToHost,
                            stream_);

                    if (status != cudaSuccess)
                    {
                        throw_scaled_accumulate_cuda_error(
                            "paired session second target download",
                            status);
                    }

                    record_cuda_d2h(
                        target_bytes_);
                }

                const auto status =
                    cudaStreamSynchronize(
                        stream_);

                if (status != cudaSuccess)
                {
                    throw_scaled_accumulate_cuda_error(
                        "paired session synchronization",
                        status);
                }

                record_cuda_stream_sync();

                finished_ = true;
            }

            bool finished() const noexcept override
            {
                return finished_;
            }

        private:
            void release_noexcept() noexcept
            {
                if (stream_ != nullptr &&
                    !finished_)
                {
                    (void)cudaStreamSynchronize(
                        stream_);
                }

                if (!zero_copy_targets_)
                {
                    if (first_device_target_ !=
                        nullptr)
                    {
                        (void)cudaFree(
                            first_device_target_);
                    }

                    if (second_device_target_ !=
                        nullptr)
                    {
                        (void)cudaFree(
                            second_device_target_);
                    }
                }

                if (device_source_ != nullptr)
                {
                    (void)cudaFree(
                        device_source_);
                }

                first_device_target_ =
                    nullptr;

                second_device_target_ =
                    nullptr;

                device_source_ =
                    nullptr;

                stream_ =
                    nullptr;
            }

            mcl::Fr *first_host_target_ =
                nullptr;

            mcl::Fr *second_host_target_ =
                nullptr;

            gpu::GpuFr *first_device_target_ =
                nullptr;

            gpu::GpuFr *second_device_target_ =
                nullptr;

            gpu::GpuFr *device_source_ =
                nullptr;

            std::size_t element_count_ = 0;
            std::size_t target_bytes_ = 0;
            std::size_t source_capacity_ = 0;

            bool zero_copy_targets_ = false;
            bool finished_ = false;

            EphemeralScaledAccumulateSourceStaging
                ephemeral_source_staging_;

            cudaStream_t stream_ =
                nullptr;
        };
    }

    std::unique_ptr<CudaScaledAccumulateSession>
    create_cuda_scaled_accumulate_session(
        mcl::Fr *target,
        std::size_t element_count,
        bool target_is_zero)
    {
        if (element_count == 0 ||
            !util::route2_options()
                 .cuda_field_hotspots)
        {
            return nullptr;
        }

        const auto minimum_size =
            cuda_backend_unified_addressing_available()
                ? (std::size_t{1} << 18)
                : (std::size_t{1} << 22);

        if (element_count <
            minimum_size)
        {
            return nullptr;
        }

        if (target == nullptr)
        {
            throw std::runtime_error(
                "CUDA scaled accumulate session received a null target");
        }

        return std::make_unique<
            CudaScaledAccumulateSessionImpl>(
            target,
            element_count,
            target_is_zero);
    }

    std::shared_ptr<CudaScaledAccumulatePairSession>
    create_cuda_scaled_accumulate_pair_session(
        mcl::Fr *first_target,
        mcl::Fr *second_target,
        std::size_t element_count,
        bool first_target_is_zero,
        bool second_target_is_zero)
    {
        if (element_count == 0 ||
            !util::route2_options()
                 .cuda_field_hotspots)
        {
            return nullptr;
        }

        const auto minimum_size =
            cuda_backend_unified_addressing_available()
                ? (std::size_t{1} << 18)
                : (std::size_t{1} << 22);

        if (element_count <
            minimum_size)
        {
            return nullptr;
        }

        if (first_target == nullptr ||
            second_target == nullptr ||
            first_target == second_target)
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate session received invalid targets");
        }

        return std::make_shared<
            CudaScaledAccumulatePairSessionImpl>(
            first_target,
            second_target,
            element_count,
            first_target_is_zero,
            second_target_is_zero);
    }

    void release_cuda_scaled_accumulate_workspace()
    {
        scaled_accumulate_workspace()
            .release();

        scaled_accumulate_stream_pool()
            .release();
    }

    void scaled_accumulate_cuda(
        uint64_t *target_limbs,
        const uint64_t *source_limbs,
        const uint64_t scale_limbs[4],
        std::size_t element_count)
    {
        if (element_count == 0)
        {
            return;
        }
        if (target_limbs == nullptr || source_limbs == nullptr || scale_limbs == nullptr)
        {
            throw std::runtime_error(
                "CUDA scaled accumulate received a null limb buffer");
        }
        static_assert(
            sizeof(gpu::GpuFr) == 4 * sizeof(uint64_t),
            "CUDA scaled accumulate requires four-limb GpuFr storage");

        gpu::GpuFr scale{};
        std::memcpy(scale.d, scale_limbs, sizeof(scale.d));

        auto &workspace =
            scaled_accumulate_workspace();

        const auto stream =
            workspace.stream();

        if (cuda_backend_unified_addressing_available())
        {
            constexpr unsigned
                kUvaBlockSize = 256;

            const auto block_count_size =
                (element_count +
                 kUvaBlockSize - 1U) /
                kUvaBlockSize;

            if (block_count_size >
                std::numeric_limits<
                    unsigned>::max())
            {
                throw std::runtime_error(
                    "CUDA scaled accumulate grid exceeds CUDA launch limits");
            }

            scaled_accumulate_kernel<<<
                static_cast<unsigned>(
                    block_count_size),
                kUvaBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    gpu::GpuFr *>(
                    target_limbs),
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    source_limbs),
                scale,
                element_count);

            auto status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_scaled_accumulate_cuda_error(
                    "UVA kernel launch",
                    status);
            }

            record_cuda_kernel_invocations(
                1);

            record_cuda_opening_fold_call();

            status =
                cudaStreamSynchronize(
                    stream);

            if (status != cudaSuccess)
            {
                throw_scaled_accumulate_cuda_error(
                    "UVA kernel synchronization",
                    status);
            }

            record_cuda_stream_sync();

            return;
        }

        if (!workspace.ensure(
                element_count))
        {
            throw std::runtime_error(
                "CUDA scaled accumulate cannot admit workspace");
        }

        const auto byte_count =
            element_count *
            sizeof(gpu::GpuFr);

        auto status =
            cudaMemcpyAsync(
                workspace.first_target(),
                target_limbs,
                byte_count,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "target upload",
                status);
        }

        record_cuda_h2d(
            byte_count);

        status =
            cudaMemcpyAsync(
                workspace.source(),
                source_limbs,
                byte_count,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "source upload",
                status);
        }

        record_cuda_h2d(
            byte_count);

        constexpr unsigned kBlockSize = 256;

        const auto block_count_size =
            (element_count + kBlockSize - 1) /
            kBlockSize;

        if (block_count_size >
            std::numeric_limits<unsigned>::max())
        {
            throw std::runtime_error(
                "CUDA scaled accumulate grid exceeds CUDA launch limits");
        }

        const auto block_count =
            static_cast<unsigned>(
                block_count_size);

        scaled_accumulate_kernel<<<
            block_count,
            kBlockSize,
            0,
            stream>>>(
            workspace.first_target(),
            workspace.source(),
            scale,
            element_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "kernel launch",
                status);
        }

        record_cuda_kernel_invocations(
            1);

        record_cuda_opening_fold_call();

        status =
            cudaMemcpyAsync(
                target_limbs,
                workspace.first_target(),
                byte_count,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "target download",
                status);
        }

        record_cuda_d2h(
            byte_count);

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "kernel synchronization",
                status);
        }

        record_cuda_stream_sync();
    }

    void scaled_accumulate_pair_cuda(
        uint64_t *first_target_limbs,
        uint64_t *second_target_limbs,
        const uint64_t *source_limbs,
        const uint64_t first_scale_limbs[4],
        const uint64_t second_scale_limbs[4],
        std::size_t element_count)
    {
        if (element_count == 0)
        {
            return;
        }

        if (first_target_limbs == nullptr ||
            second_target_limbs == nullptr ||
            source_limbs == nullptr ||
            first_scale_limbs == nullptr ||
            second_scale_limbs == nullptr)
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate received a null limb buffer");
        }

        if (first_target_limbs == second_target_limbs ||
            first_target_limbs == source_limbs ||
            second_target_limbs == source_limbs)
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate requires distinct source and targets");
        }

        static_assert(
            sizeof(gpu::GpuFr) == 4 * sizeof(uint64_t),
            "CUDA paired scaled accumulate requires four-limb GpuFr storage");

        gpu::GpuFr first_scale{};
        gpu::GpuFr second_scale{};

        std::memcpy(
            first_scale.d,
            first_scale_limbs,
            sizeof(first_scale.d));

        std::memcpy(
            second_scale.d,
            second_scale_limbs,
            sizeof(second_scale.d));

        auto &workspace =
            scaled_accumulate_workspace();

        const auto stream =
            workspace.stream();

        if (cuda_backend_unified_addressing_available())
        {
            constexpr unsigned
                kUvaBlockSize = 256;

            const auto block_count_size =
                (element_count +
                 kUvaBlockSize - 1U) /
                kUvaBlockSize;

            if (block_count_size >
                std::numeric_limits<
                    unsigned>::max())
            {
                throw std::runtime_error(
                    "CUDA paired scaled accumulate grid exceeds CUDA launch limits");
            }

            scaled_accumulate_pair_kernel<<<
                static_cast<unsigned>(
                    block_count_size),
                kUvaBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    gpu::GpuFr *>(
                    first_target_limbs),
                reinterpret_cast<
                    gpu::GpuFr *>(
                    second_target_limbs),
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    source_limbs),
                first_scale,
                second_scale,
                element_count);

            auto status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_scaled_accumulate_cuda_error(
                    "paired UVA kernel launch",
                    status);
            }

            record_cuda_kernel_invocations(
                1);

            record_cuda_opening_fold_call();

            status =
                cudaStreamSynchronize(
                    stream);

            if (status != cudaSuccess)
            {
                throw_scaled_accumulate_cuda_error(
                    "paired UVA kernel synchronization",
                    status);
            }

            record_cuda_stream_sync();

            return;
        }

        if (!workspace.ensure(
                element_count))
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate cannot admit workspace");
        }

        const auto byte_count =
            element_count *
            sizeof(gpu::GpuFr);

        auto status =
            cudaMemcpyAsync(
                workspace.first_target(),
                first_target_limbs,
                byte_count,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "first target upload",
                status);
        }

        record_cuda_h2d(
            byte_count);

        status =
            cudaMemcpyAsync(
                workspace.second_target(),
                second_target_limbs,
                byte_count,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "second target upload",
                status);
        }

        record_cuda_h2d(
            byte_count);

        status =
            cudaMemcpyAsync(
                workspace.source(),
                source_limbs,
                byte_count,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "source upload",
                status);
        }

        record_cuda_h2d(
            byte_count);

        constexpr unsigned kBlockSize = 256;

        const auto block_count_size =
            (element_count + kBlockSize - 1) /
            kBlockSize;

        if (block_count_size >
            std::numeric_limits<unsigned>::max())
        {
            throw std::runtime_error(
                "CUDA paired scaled accumulate grid exceeds CUDA launch limits");
        }

        const auto block_count =
            static_cast<unsigned>(
                block_count_size);

        scaled_accumulate_pair_kernel<<<
            block_count,
            kBlockSize,
            0,
            stream>>>(
            workspace.first_target(),
            workspace.second_target(),
            workspace.source(),
            first_scale,
            second_scale,
            element_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "paired kernel launch",
                status);
        }

        record_cuda_kernel_invocations(
            1);

        record_cuda_opening_fold_call();

        status =
            cudaMemcpyAsync(
                first_target_limbs,
                workspace.first_target(),
                byte_count,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "first target download",
                status);
        }

        record_cuda_d2h(
            byte_count);

        status =
            cudaMemcpyAsync(
                second_target_limbs,
                workspace.second_target(),
                byte_count,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "second target download",
                status);
        }

        record_cuda_d2h(
            byte_count);

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scaled_accumulate_cuda_error(
                "paired kernel synchronization",
                status);
        }

        record_cuda_stream_sync();
    }

    // 重心插值权重计算(用于在单个挑战点x上快速评估多项式)
    namespace
    {
        constexpr unsigned
            kBarycentricScanBlockSize = 256;

        __device__ __forceinline__
            gpu::GpuFr
            barycentric_pow_u64(
                gpu::GpuFr base,
                std::size_t exponent)
        {
            auto result =
                gpu::fr_one();

            while (exponent != 0)
            {
                if ((exponent & 1U) != 0U)
                {
                    result =
                        gpu::fr_mul(
                            result,
                            base);
                }

                base =
                    gpu::fr_sqr(
                        base);

                exponent >>=
                    1U;
            }

            return result;
        }

        __global__ void
        barycentric_prepare_parallel_kernel(
            const gpu::GpuFr omega,
            const gpu::GpuFr x,
            gpu::GpuFr *__restrict__ denominators,
            std::size_t domain_size)
        {
            const auto lane =
                static_cast<unsigned>(
                    threadIdx.x);

            const auto block_begin =
                static_cast<std::size_t>(
                    blockIdx.x) *
                blockDim.x;

            const auto index =
                block_begin +
                lane;

            __shared__ gpu::GpuFr
                lane_powers[kBarycentricScanBlockSize];

            __shared__ gpu::GpuFr
                block_power;

            lane_powers[lane] =
                lane == 0
                    ? gpu::fr_one()
                    : omega;

            __syncthreads();

            for (unsigned offset = 1;
                 offset <
                 kBarycentricScanBlockSize;
                 offset <<= 1U)
            {
                auto value =
                    lane_powers[lane];

                if (lane >= offset)
                {
                    value =
                        gpu::fr_mul(
                            lane_powers[lane - offset],
                            value);
                }

                __syncthreads();

                lane_powers[lane] =
                    value;

                __syncthreads();
            }

            if (lane == 0)
            {
                block_power =
                    barycentric_pow_u64(
                        omega,
                        block_begin);
            }

            __syncthreads();

            if (index >= domain_size)
            {
                return;
            }

            const auto omega_power =
                gpu::fr_mul(
                    block_power,
                    lane_powers[lane]);

            denominators[index] =
                gpu::fr_sub(
                    x,
                    omega_power);
        }

        __global__ void
        product_exclusive_scan_block_kernel(
            const gpu::GpuFr *__restrict__ input,
            gpu::GpuFr *__restrict__ output,
            gpu::GpuFr *__restrict__ block_products,
            std::size_t count,
            bool reverse)
        {
            const auto lane =
                static_cast<unsigned>(
                    threadIdx.x);

            const auto traversal_begin =
                static_cast<std::size_t>(
                    blockIdx.x) *
                blockDim.x;

            const auto traversal_index =
                traversal_begin +
                lane;

            const bool active =
                traversal_index < count;

            const auto input_index =
                reverse
                    ? count - 1U -
                          traversal_index
                    : traversal_index;

            __shared__ gpu::GpuFr
                scan[kBarycentricScanBlockSize];

            scan[lane] =
                active
                    ? input[input_index]
                    : gpu::fr_one();

            __syncthreads();

            for (unsigned offset = 1;
                 offset <
                 kBarycentricScanBlockSize;
                 offset <<= 1U)
            {
                auto value =
                    scan[lane];

                if (lane >= offset)
                {
                    value =
                        gpu::fr_mul(
                            scan[lane - offset],
                            value);
                }

                __syncthreads();

                scan[lane] =
                    value;

                __syncthreads();
            }

            if (active)
            {
                output[input_index] =
                    lane == 0
                        ? gpu::fr_one()
                        : scan[lane - 1U];
            }

            if (lane == 0 &&
                block_products != nullptr)
            {
                const auto remaining =
                    count - traversal_begin;

                const auto valid_count =
                    static_cast<unsigned>(
                        remaining <
                                blockDim.x
                            ? remaining
                            : blockDim.x);

                block_products[blockIdx.x] =
                    scan[valid_count - 1U];
            }
        }

        __global__ void
        apply_product_block_prefix_kernel(
            gpu::GpuFr *__restrict__ output,
            const gpu::GpuFr *__restrict__ block_prefixes,
            std::size_t count,
            bool reverse)
        {
            const auto traversal_index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (traversal_index >= count)
            {
                return;
            }

            const auto block_index =
                traversal_index /
                blockDim.x;

            const auto output_index =
                reverse
                    ? count - 1U -
                          traversal_index
                    : traversal_index;

            output[output_index] =
                gpu::fr_mul(
                    block_prefixes[block_index],
                    output[output_index]);
        }

        __global__ void
        barycentric_invert_product_kernel(
            const gpu::GpuFr *__restrict__ denominators,
            const gpu::GpuFr *__restrict__ prefixes,
            gpu::GpuFr *__restrict__ global_inverse,
            std::size_t domain_size)
        {
            if (blockIdx.x != 0 ||
                threadIdx.x != 0)
            {
                return;
            }

            const auto last =
                domain_size - 1U;

            const auto total_product =
                gpu::fr_mul(
                    prefixes[last],
                    denominators[last]);

            global_inverse[0] =
                gpu::fr_inv(
                    total_product);
        }

        __global__ void
        barycentric_finalize_parallel_kernel(
            const gpu::GpuFr *__restrict__ denominators,
            const gpu::GpuFr *__restrict__ prefixes,
            const gpu::GpuFr *__restrict__ suffixes,
            const gpu::GpuFr *__restrict__ global_inverse,
            const gpu::GpuFr x,
            const gpu::GpuFr zero_eval,
            const gpu::GpuFr inv_n,
            gpu::GpuFr *__restrict__ weights,
            std::size_t domain_size)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= domain_size)
            {
                return;
            }

            const auto denominator_inverse =
                gpu::fr_mul(
                    gpu::fr_mul(
                        prefixes[index],
                        suffixes[index]),
                    global_inverse[0]);

            const auto omega_power =
                gpu::fr_sub(
                    x,
                    denominators[index]);

            const auto scaled_point =
                gpu::fr_mul(
                    omega_power,
                    inv_n);

            weights[index] =
                gpu::fr_mul(
                    gpu::fr_mul(
                        zero_eval,
                        scaled_point),
                    denominator_inverse);
        }

        struct BarycentricScanLevel
        {
            gpu::GpuFr *block_products =
                nullptr;

            gpu::GpuFr *block_prefixes =
                nullptr;

            std::size_t block_count = 0;
        };

        void release_barycentric_scan_levels(
            std::vector<
                BarycentricScanLevel> *levels)
        {
            if (levels == nullptr)
            {
                return;
            }

            for (auto &level : *levels)
            {
                if (level.block_products !=
                    nullptr)
                {
                    (void)cudaFree(
                        level.block_products);

                    level.block_products =
                        nullptr;
                }

                if (level.block_prefixes !=
                    nullptr)
                {
                    (void)cudaFree(
                        level.block_prefixes);

                    level.block_prefixes =
                        nullptr;
                }
            }

            levels->clear();
        }

        bool allocate_barycentric_scan_levels(
            std::size_t count,
            std::vector<
                BarycentricScanLevel> *levels)
        {
            if (levels == nullptr)
            {
                return false;
            }

            levels->clear();

            auto current_count =
                count;

            while (current_count >
                   kBarycentricScanBlockSize)
            {
                const auto block_count =
                    (current_count +
                     kBarycentricScanBlockSize -
                     1U) /
                    kBarycentricScanBlockSize;

                BarycentricScanLevel level;
                level.block_count =
                    block_count;

                const auto byte_count =
                    block_count *
                    sizeof(gpu::GpuFr);

                auto status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &level.block_products),
                        byte_count);

                if (status != cudaSuccess)
                {
                    if (status ==
                        cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();
                    }

                    release_barycentric_scan_levels(
                        levels);

                    return false;
                }

                status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &level.block_prefixes),
                        byte_count);

                if (status != cudaSuccess)
                {
                    if (status ==
                        cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();
                    }

                    (void)cudaFree(
                        level.block_products);

                    release_barycentric_scan_levels(
                        levels);

                    return false;
                }

                levels->push_back(
                    level);

                current_count =
                    block_count;
            }

            return true;
        }

        [[noreturn]] void
        throw_scan_cuda_error(
            const char *operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string(
                    "CUDA scan workspace ") +
                operation +
                " failed: " +
                cudaGetErrorString(
                    status));
        }

        class PersistentScanWorkspace
        {
        public:
            cudaStream_t stream()
            {
                if (stream_ ==
                    nullptr)
                {
                    const auto status =
                        cudaStreamCreateWithFlags(
                            &stream_,
                            cudaStreamNonBlocking);

                    if (status !=
                        cudaSuccess)
                    {
                        stream_ =
                            nullptr;

                        throw_scan_cuda_error(
                            "stream creation",
                            status);
                    }
                }

                return stream_;
            }

            bool ensure_product(
                std::size_t count)
            {
                return ensure(
                    count,
                    true,
                    false,
                    false,
                    true,
                    true);
            }

            bool ensure_sum(
                std::size_t count,
                bool need_error)
            {
                return ensure(
                    count,
                    false,
                    true,
                    false,
                    false,
                    need_error);
            }

            bool ensure_logup(
                std::size_t count)
            {
                return ensure(
                    count,
                    true,
                    true,
                    false,
                    true,
                    true);
            }

            bool ensure_barycentric(
                std::size_t count)
            {
                return ensure(
                    count,
                    true,
                    true,
                    true,
                    true,
                    false);
            }

            gpu::GpuFr *prefixes()
            {
                return prefixes_;
            }

            gpu::GpuFr *suffixes()
            {
                return suffixes_;
            }

            gpu::GpuFr *scratch_a()
            {
                return scratch_a_;
            }

            gpu::GpuFr *scratch_b()
            {
                return scratch_b_;
            }

            gpu::GpuFr *global_inverse()
            {
                return global_inverse_;
            }

            unsigned int *error()
            {
                return error_;
            }

            std::vector<
                BarycentricScanLevel> &
            scan_levels()
            {
                return scan_levels_;
            }

            void release() noexcept
            {
                if (stream_ !=
                    nullptr)
                {
                    (void)cudaStreamSynchronize(
                        stream_);
                }

                release_field_buffer(
                    &prefixes_,
                    &prefix_capacity_);

                release_field_buffer(
                    &suffixes_,
                    &suffix_capacity_);

                release_field_buffer(
                    &scratch_a_,
                    &scratch_a_capacity_);

                release_field_buffer(
                    &scratch_b_,
                    &scratch_b_capacity_);

                release_field_buffer(
                    &global_inverse_,
                    &global_inverse_capacity_);

                if (error_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        error_);

                    error_ = nullptr;
                }

                release_barycentric_scan_levels(
                    &scan_levels_);

                scan_capacity_ = 0;

                if (stream_ !=
                    nullptr)
                {
                    (void)cudaStreamDestroy(
                        stream_);

                    stream_ = nullptr;
                }
            }

            ~PersistentScanWorkspace()
            {
                release();
            }

        private:
            static bool needs_growth(
                const gpu::GpuFr *pointer,
                std::size_t capacity,
                std::size_t required)
            {
                return pointer ==
                           nullptr ||
                       capacity <
                           required;
            }

            bool ensure_field_buffer(
                gpu::GpuFr **pointer,
                std::size_t *capacity,
                std::size_t required,
                const char *label)
            {
                if (!needs_growth(
                        *pointer,
                        *capacity,
                        required))
                {
                    return true;
                }

                release_field_buffer(
                    pointer,
                    capacity);

                if (required == 0)
                {
                    return true;
                }

                if (required >
                    std::numeric_limits<
                        std::size_t>::max() /
                        sizeof(
                            gpu::GpuFr))
                {
                    throw std::runtime_error(
                        std::string(
                            "CUDA scan workspace byte-size overflow: ") +
                        label);
                }

                const auto status =
                    cudaMalloc(
                        reinterpret_cast<
                            void **>(
                            pointer),
                        required *
                            sizeof(
                                gpu::GpuFr));

                if (status ==
                    cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    *pointer = nullptr;
                    *capacity = 0;

                    return false;
                }

                if (status !=
                    cudaSuccess)
                {
                    throw_scan_cuda_error(
                        label,
                        status);
                }

                *capacity = required;

                return true;
            }

            static void
            release_field_buffer(
                gpu::GpuFr **pointer,
                std::size_t *capacity) noexcept
            {
                if (*pointer !=
                    nullptr)
                {
                    (void)cudaFree(
                        *pointer);

                    *pointer =
                        nullptr;
                }

                *capacity = 0;
            }

            bool ensure(
                std::size_t count,
                bool need_suffixes,
                bool need_scratch_a,
                bool need_scratch_b,
                bool need_global_inverse,
                bool need_error)
            {
                if (count == 0)
                {
                    return true;
                }

                const bool grow =
                    needs_growth(
                        prefixes_,
                        prefix_capacity_,
                        count) ||
                    (need_suffixes &&
                     needs_growth(
                         suffixes_,
                         suffix_capacity_,
                         count)) ||
                    (need_scratch_a &&
                     needs_growth(
                         scratch_a_,
                         scratch_a_capacity_,
                         count)) ||
                    (need_scratch_b &&
                     needs_growth(
                         scratch_b_,
                         scratch_b_capacity_,
                         count)) ||
                    (need_global_inverse &&
                     needs_growth(
                         global_inverse_,
                         global_inverse_capacity_,
                         1)) ||
                    (need_error &&
                     error_ == nullptr) ||
                    scan_capacity_ <
                        count;

                const auto active_stream =
                    stream();

                if (!grow)
                {
                    return true;
                }

                const auto sync_status =
                    cudaStreamSynchronize(
                        active_stream);

                if (sync_status !=
                    cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "growth synchronization",
                        sync_status);
                }

                record_cuda_stream_sync();

                if (!ensure_field_buffer(
                        &prefixes_,
                        &prefix_capacity_,
                        count,
                        "prefix allocation"))
                {
                    return false;
                }

                if (need_suffixes &&
                    !ensure_field_buffer(
                        &suffixes_,
                        &suffix_capacity_,
                        count,
                        "suffix allocation"))
                {
                    return false;
                }

                if (need_scratch_a &&
                    !ensure_field_buffer(
                        &scratch_a_,
                        &scratch_a_capacity_,
                        count,
                        "scratch-a allocation"))
                {
                    return false;
                }

                if (need_scratch_b &&
                    !ensure_field_buffer(
                        &scratch_b_,
                        &scratch_b_capacity_,
                        count,
                        "scratch-b allocation"))
                {
                    return false;
                }

                if (need_global_inverse &&
                    !ensure_field_buffer(
                        &global_inverse_,
                        &global_inverse_capacity_,
                        1,
                        "global inverse allocation"))
                {
                    return false;
                }

                if (need_error &&
                    error_ == nullptr)
                {
                    const auto status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &error_),
                            sizeof(
                                unsigned int));

                    if (status ==
                        cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();

                        error_ =
                            nullptr;

                        return false;
                    }

                    if (status !=
                        cudaSuccess)
                    {
                        throw_scan_cuda_error(
                            "error flag allocation",
                            status);
                    }
                }

                if (scan_capacity_ <
                    count)
                {
                    release_barycentric_scan_levels(
                        &scan_levels_);

                    if (!allocate_barycentric_scan_levels(
                            count,
                            &scan_levels_))
                    {
                        scan_capacity_ =
                            0;

                        return false;
                    }

                    scan_capacity_ =
                        count;
                }

                return true;
            }

            gpu::GpuFr *prefixes_ =
                nullptr;

            gpu::GpuFr *suffixes_ =
                nullptr;

            gpu::GpuFr *scratch_a_ =
                nullptr;

            gpu::GpuFr *scratch_b_ =
                nullptr;

            gpu::GpuFr *global_inverse_ =
                nullptr;

            unsigned int *error_ =
                nullptr;

            std::size_t prefix_capacity_ = 0;
            std::size_t suffix_capacity_ = 0;
            std::size_t scratch_a_capacity_ = 0;
            std::size_t scratch_b_capacity_ = 0;
            std::size_t global_inverse_capacity_ = 0;
            std::size_t scan_capacity_ = 0;

            std::vector<
                BarycentricScanLevel>
                scan_levels_;

            cudaStream_t stream_ =
                nullptr;
        };

        PersistentScanWorkspace &
        persistent_scan_workspace()
        {
            thread_local PersistentScanWorkspace
                workspace;

            return workspace;
        }

        bool launch_exclusive_product_scan(
            const gpu::GpuFr *input,
            gpu::GpuFr *output,
            std::size_t count,
            bool reverse,
            std::vector<
                BarycentricScanLevel> &levels,
            std::size_t level_index,
            std::size_t *kernel_count,
            cudaStream_t stream =
                nullptr)
        {
            if (count == 0)
            {
                return true;
            }

            const auto block_count_size =
                (count +
                 kBarycentricScanBlockSize -
                 1U) /
                kBarycentricScanBlockSize;

            if (block_count_size >
                std::numeric_limits<
                    unsigned>::max())
            {
                return false;
            }

            gpu::GpuFr *block_products =
                nullptr;

            if (block_count_size > 1)
            {
                if (level_index >=
                        levels.size() ||
                    levels[level_index]
                            .block_count <
                        block_count_size)
                {
                    return false;
                }

                block_products =
                    levels[level_index]
                        .block_products;
            }

            product_exclusive_scan_block_kernel<<<
                static_cast<unsigned>(
                    block_count_size),
                kBarycentricScanBlockSize,
                0,
                stream>>>(
                input,
                output,
                block_products,
                count,
                reverse);

            if (cudaGetLastError() !=
                cudaSuccess)
            {
                return false;
            }

            if (kernel_count != nullptr)
            {
                ++(*kernel_count);
            }

            if (block_count_size <= 1)
            {
                return true;
            }

            auto &level =
                levels[level_index];

            if (!launch_exclusive_product_scan(
                    level.block_products,
                    level.block_prefixes,
                    block_count_size,
                    false,
                    levels,
                    level_index + 1U,
                    kernel_count,
                    stream))
            {
                return false;
            }

            apply_product_block_prefix_kernel<<<
                static_cast<unsigned>(
                    block_count_size),
                kBarycentricScanBlockSize,
                0,
                stream>>>(
                output,
                level.block_prefixes,
                count,
                reverse);

            if (cudaGetLastError() !=
                cudaSuccess)
            {
                return false;
            }

            if (kernel_count != nullptr)
            {
                ++(*kernel_count);
            }

            return true;
        }
        __global__ void
        validate_nonzero_kernel(
            const gpu::GpuFr *values,
            unsigned int *error,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            if (gpu::fr_is_zero(
                    values[index]))
            {
                atomicCAS(
                    error,
                    0U,
                    1U);
            }
        }

        __global__ void
        inverse_finalize_kernel(
            const gpu::GpuFr *prefixes,
            const gpu::GpuFr *suffixes,
            const gpu::GpuFr *global_inverse,
            gpu::GpuFr *output,
            std::size_t count,
            bool negate)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            auto inverse =
                gpu::fr_mul(
                    gpu::fr_mul(
                        prefixes[index],
                        suffixes[index]),
                    global_inverse[0]);

            if (negate)
            {
                inverse =
                    gpu::fr_sub(
                        gpu::fr_zero(),
                        inverse);
            }

            output[index] =
                inverse;
        }

        __global__ void
        opening_quotient_assembly_kernel(
            const gpu::GpuFr *folded_values,
            gpu::GpuFr *quotient_values,
            gpu::GpuFr scaled_claim_sum,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const auto numerator =
                gpu::fr_sub(
                    folded_values[index],
                    scaled_claim_sum);

            quotient_values[index] =
                gpu::fr_mul(
                    numerator,
                    quotient_values[index]);
        }

        constexpr std::size_t
            kCoefficientOpeningRowsPerThread =
                1024;

        __global__ void
        coefficient_opening_block_summary_kernel(
            const gpu::GpuFr *coefficients,
            gpu::GpuFr point,
            gpu::GpuFr *block_constants,
            std::size_t quotient_size)
        {
            const auto logical_block =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto begin =
                logical_block *
                kCoefficientOpeningRowsPerThread;

            if (begin >= quotient_size)
            {
                return;
            }

            auto end =
                begin +
                kCoefficientOpeningRowsPerThread;

            if (end > quotient_size)
            {
                end = quotient_size;
            }

            auto accumulated =
                gpu::fr_zero();

            for (std::size_t index = end;
                 index-- > begin;)
            {
                accumulated =
                    gpu::fr_add(
                        coefficients[index + 1],
                        gpu::fr_mul(
                            point,
                            accumulated));
            }

            block_constants[logical_block] =
                accumulated;
        }

        __global__ void
        coefficient_opening_emit_kernel(
            const gpu::GpuFr *coefficients,
            gpu::GpuFr point,
            const gpu::GpuFr *block_incoming,
            gpu::GpuFr *quotient,
            std::size_t quotient_size)
        {
            const auto logical_block =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto begin =
                logical_block *
                kCoefficientOpeningRowsPerThread;

            if (begin >= quotient_size)
            {
                return;
            }

            auto end =
                begin +
                kCoefficientOpeningRowsPerThread;

            if (end > quotient_size)
            {
                end = quotient_size;
            }

            auto accumulated =
                block_incoming[logical_block];

            for (std::size_t index = end;
                 index-- > begin;)
            {
                accumulated =
                    gpu::fr_add(
                        coefficients[index + 1],
                        gpu::fr_mul(
                            point,
                            accumulated));

                quotient[index] =
                    accumulated;
            }
        }

        __global__ void
        sum_exclusive_scan_block_kernel(
            const gpu::GpuFr *input,
            gpu::GpuFr *output,
            gpu::GpuFr *block_sums,
            std::size_t count)
        {
            const auto lane =
                static_cast<unsigned>(
                    threadIdx.x);

            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                lane;

            const bool active =
                index < count;

            __shared__ gpu::GpuFr
                scan[kBarycentricScanBlockSize];

            scan[lane] =
                active
                    ? input[index]
                    : gpu::fr_zero();

            __syncthreads();

            for (unsigned offset = 1;
                 offset <
                 kBarycentricScanBlockSize;
                 offset <<= 1U)
            {
                auto value =
                    scan[lane];

                if (lane >= offset)
                {
                    value =
                        gpu::fr_add(
                            scan[lane -
                                 offset],
                            value);
                }

                __syncthreads();

                scan[lane] =
                    value;

                __syncthreads();
            }

            if (active)
            {
                output[index] =
                    lane == 0
                        ? gpu::fr_zero()
                        : scan[lane - 1U];
            }

            if (lane == 0 &&
                block_sums != nullptr)
            {
                const auto remaining =
                    count -
                    static_cast<std::size_t>(
                        blockIdx.x) *
                        blockDim.x;

                const auto valid_count =
                    static_cast<unsigned>(
                        remaining <
                                static_cast<std::size_t>(
                                    blockDim.x)
                            ? remaining
                            : static_cast<std::size_t>(
                                  blockDim.x));

                block_sums[blockIdx.x] =
                    scan[valid_count - 1U];
            }
        }

        __global__ void
        apply_sum_block_prefix_kernel(
            gpu::GpuFr *output,
            const gpu::GpuFr *block_prefixes,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const auto block_index =
                index /
                blockDim.x;

            output[index] =
                gpu::fr_add(
                    block_prefixes[block_index],
                    output[index]);
        }

        bool launch_exclusive_sum_scan(
            const gpu::GpuFr *input,
            gpu::GpuFr *output,
            std::size_t count,
            std::vector<
                BarycentricScanLevel> &levels,
            std::size_t level_index,
            std::size_t *kernel_count,
            cudaStream_t stream =
                nullptr)
        {
            if (count == 0)
            {
                return true;
            }

            const auto block_count =
                (count +
                 kBarycentricScanBlockSize -
                 1U) /
                kBarycentricScanBlockSize;

            if (block_count >
                std::numeric_limits<
                    unsigned>::max())
            {
                return false;
            }

            gpu::GpuFr *block_sums =
                nullptr;

            if (block_count > 1)
            {
                if (level_index >=
                        levels.size() ||
                    levels[level_index]
                            .block_count <
                        block_count)
                {
                    return false;
                }

                block_sums =
                    levels[level_index]
                        .block_products;
            }

            sum_exclusive_scan_block_kernel<<<
                static_cast<unsigned>(
                    block_count),
                kBarycentricScanBlockSize,
                0,
                stream>>>(
                input,
                output,
                block_sums,
                count);

            if (cudaGetLastError() !=
                cudaSuccess)
            {
                return false;
            }

            if (kernel_count != nullptr)
            {
                ++(*kernel_count);
            }

            if (block_count <= 1)
            {
                return true;
            }

            auto &level =
                levels[level_index];

            if (!launch_exclusive_sum_scan(
                    level.block_products,
                    level.block_prefixes,
                    block_count,
                    levels,
                    level_index + 1U,
                    kernel_count,
                    stream))
            {
                return false;
            }

            apply_sum_block_prefix_kernel<<<
                static_cast<unsigned>(
                    block_count),
                kBarycentricScanBlockSize,
                0,
                stream>>>(
                output,
                level.block_prefixes,
                count);

            if (cudaGetLastError() !=
                cudaSuccess)
            {
                return false;
            }

            if (kernel_count != nullptr)
            {
                ++(*kernel_count);
            }

            return true;
        }
        __global__ void
        logup_prepare_denominator_kernel(
            const gpu::GpuFr *values,
            const gpu::GpuFr *selectors,
            gpu::GpuFr beta,
            gpu::GpuFr *denominators,
            unsigned int *error,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const auto selector =
                selectors[index];

            const bool inactive =
                gpu::fr_is_zero(
                    selector);

            const bool active =
                gpu::fr_is_zero(
                    gpu::fr_sub(
                        selector,
                        gpu::fr_one()));

            if (!inactive &&
                !active)
            {
                atomicCAS(
                    error,
                    0U,
                    2U);

                denominators[index] =
                    gpu::fr_one();

                return;
            }

            if (inactive)
            {
                denominators[index] =
                    gpu::fr_one();

                return;
            }

            const auto denominator =
                gpu::fr_add(
                    values[index],
                    beta);

            if (gpu::fr_is_zero(
                    denominator))
            {
                atomicCAS(
                    error,
                    0U,
                    1U);

                denominators[index] =
                    gpu::fr_one();

                return;
            }

            denominators[index] =
                denominator;
        }

        __global__ void
        logup_finalize_inverse_kernel(
            const gpu::GpuFr *selectors,
            const gpu::GpuFr *prefixes,
            const gpu::GpuFr *suffixes,
            const gpu::GpuFr *global_inverse,
            gpu::GpuFr *output,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            if (gpu::fr_is_zero(
                    selectors[index]))
            {
                output[index] =
                    gpu::fr_zero();

                return;
            }

            output[index] =
                gpu::fr_mul(
                    gpu::fr_mul(
                        prefixes[index],
                        suffixes[index]),
                    global_inverse[0]);
        }

        __global__ void
        logup_term_kernel(
            const gpu::GpuFr *table_inverse,
            const gpu::GpuFr *query_inverse,
            const gpu::GpuFr *multiplicity,
            const gpu::GpuFr *q_table,
            const gpu::GpuFr *q_query,
            gpu::GpuFr *terms,
            unsigned int *error,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            const auto table_selector =
                q_table[index];

            const auto query_selector =
                q_query[index];

            const bool table_zero =
                gpu::fr_is_zero(
                    table_selector);

            const bool table_one =
                gpu::fr_is_zero(
                    gpu::fr_sub(
                        table_selector,
                        gpu::fr_one()));

            const bool query_zero =
                gpu::fr_is_zero(
                    query_selector);

            const bool query_one =
                gpu::fr_is_zero(
                    gpu::fr_sub(
                        query_selector,
                        gpu::fr_one()));

            if ((!table_zero &&
                 !table_one) ||
                (!query_zero &&
                 !query_one))
            {
                atomicCAS(
                    error,
                    0U,
                    2U);

                terms[index] =
                    gpu::fr_zero();

                return;
            }

            auto term =
                gpu::fr_zero();

            if (table_one)
            {
                term =
                    gpu::fr_mul(
                        multiplicity[index],
                        table_inverse[index]);
            }

            if (query_one)
            {
                term =
                    gpu::fr_sub(
                        term,
                        query_inverse[index]);
            }

            terms[index] =
                term;
        }

        __global__ void
        scaled_prefix_term_kernel(
            const gpu::GpuFr *values,
            gpu::GpuFr scale,
            gpu::GpuFr *terms,
            std::size_t count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= count)
            {
                return;
            }

            terms[index] =
                gpu::fr_mul(
                    values[index],
                    scale);
        }

        __global__ void
        prefix_accumulator_finalize_kernel(
            const gpu::GpuFr *terms,
            const gpu::GpuFr *prefixes,
            gpu::GpuFr *output,
            std::size_t active_count,
            std::size_t element_count)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= element_count)
            {
                return;
            }

            if (index == 0 ||
                active_count == 0)
            {
                output[index] =
                    gpu::fr_zero();

                return;
            }

            const auto final_value =
                gpu::fr_add(
                    prefixes[active_count - 1U],
                    terms[active_count - 1U]);

            if (index >
                active_count)
            {
                output[index] =
                    final_value;

                return;
            }

            const auto source =
                index - 1U;

            output[index] =
                gpu::fr_add(
                    prefixes[source],
                    terms[source]);
        }

        class CudaBatchInverseSessionImpl final
            : public CudaBatchInverseSession
        {
        public:
            explicit CudaBatchInverseSessionImpl(
                std::size_t maximum_element_count)
                : maximum_element_count_(
                      maximum_element_count)
            {
                if (maximum_element_count_ == 0)
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session requires a non-zero capacity");
                }

                if (!workspace_.ensure_product(
                        maximum_element_count_))
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session workspace allocation failed");
                }

                const auto status =
                    cudaHostAlloc(
                        reinterpret_cast<
                            void **>(
                            &host_error_),
                        sizeof(
                            unsigned int),
                        cudaHostAllocPortable);

                if (status != cudaSuccess)
                {
                    host_error_ = nullptr;

                    throw_scan_cuda_error(
                        "batch inverse session host error allocation",
                        status);
                }

                *host_error_ = 0;
            }

            ~CudaBatchInverseSessionImpl()
                override
            {
                if (pending_)
                {
                    (void)cudaStreamSynchronize(
                        workspace_.stream());
                }

                if (host_error_ != nullptr)
                {
                    (void)cudaFreeHost(
                        host_error_);

                    host_error_ = nullptr;
                }
            }

            void *
            native_stream_handle()
                override
            {
                return reinterpret_cast<
                    void *>(
                    workspace_.stream());
            }

            void enqueue(
                const mcl::Fr *input,
                mcl::Fr *output,
                std::size_t element_count)
                override
            {
                if (pending_)
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session already has an in-flight operation");
                }

                if (element_count == 0)
                {
                    return;
                }

                if (input == nullptr ||
                    output == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session received null storage");
                }

                if (element_count >
                    maximum_element_count_)
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session input exceeds reserved capacity");
                }

                const auto block_count_size =
                    (element_count +
                     kBarycentricScanBlockSize -
                     1U) /
                    kBarycentricScanBlockSize;

                if (block_count_size >
                    std::numeric_limits<
                        unsigned>::max())
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session grid exceeds CUDA limits");
                }

                const auto stream =
                    workspace_.stream();

                auto *prefixes =
                    workspace_.prefixes();

                auto *suffixes =
                    workspace_.suffixes();

                auto *global_inverse =
                    workspace_.global_inverse();

                auto *error =
                    workspace_.error();

                auto &scan_levels =
                    workspace_.scan_levels();

                *host_error_ = 0;

                auto status =
                    cudaMemsetAsync(
                        error,
                        0,
                        sizeof(
                            unsigned int),
                        stream);

                if (status != cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "batch inverse session error reset",
                        status);
                }

                const auto *gpu_input =
                    reinterpret_cast<
                        const gpu::GpuFr *>(
                        input);

                auto *gpu_output =
                    reinterpret_cast<
                        gpu::GpuFr *>(
                        output);

                validate_nonzero_kernel<<<
                    static_cast<unsigned>(
                        block_count_size),
                    kBarycentricScanBlockSize,
                    0,
                    stream>>>(
                    gpu_input,
                    error,
                    element_count);

                status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "batch inverse session validation launch",
                        status);
                }

                std::size_t kernel_count = 1;

                if (!launch_exclusive_product_scan(
                        gpu_input,
                        prefixes,
                        element_count,
                        false,
                        scan_levels,
                        0,
                        &kernel_count,
                        stream) ||
                    !launch_exclusive_product_scan(
                        gpu_input,
                        suffixes,
                        element_count,
                        true,
                        scan_levels,
                        0,
                        &kernel_count,
                        stream))
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session scan failed");
                }

                barycentric_invert_product_kernel<<<
                    1,
                    1,
                    0,
                    stream>>>(
                    gpu_input,
                    prefixes,
                    global_inverse,
                    element_count);

                status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "batch inverse session product inversion launch",
                        status);
                }

                ++kernel_count;

                inverse_finalize_kernel<<<
                    static_cast<unsigned>(
                        block_count_size),
                    kBarycentricScanBlockSize,
                    0,
                    stream>>>(
                    prefixes,
                    suffixes,
                    global_inverse,
                    gpu_output,
                    element_count,
                    false);

                status =
                    cudaGetLastError();

                if (status != cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "batch inverse session finalize launch",
                        status);
                }

                ++kernel_count;

                status =
                    cudaMemcpyAsync(
                        host_error_,
                        error,
                        sizeof(
                            unsigned int),
                        cudaMemcpyDeviceToHost,
                        stream);

                if (status != cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "batch inverse session error download",
                        status);
                }

                record_cuda_kernel_invocations(
                    kernel_count);

                record_cuda_d2h(
                    sizeof(
                        unsigned int));

                pending_ = true;
            }

            void wait()
                override
            {
                if (!pending_)
                {
                    return;
                }

                const auto status =
                    cudaStreamSynchronize(
                        workspace_.stream());

                pending_ = false;

                if (status != cudaSuccess)
                {
                    throw_scan_cuda_error(
                        "batch inverse session synchronization",
                        status);
                }

                record_cuda_stream_sync();

                if (*host_error_ != 0)
                {
                    throw std::runtime_error(
                        "CUDA batch inverse session encountered zero");
                }
            }

        private:
            std::size_t maximum_element_count_ = 0;

            PersistentScanWorkspace workspace_;

            unsigned int *host_error_ = nullptr;

            bool pending_ = false;
        };

    } // namespace

    void release_cuda_scan_workspace()
    {
        persistent_scan_workspace()
            .release();
    }

    std::unique_ptr<CudaBatchInverseSession>
    create_cuda_batch_inverse_session(
        std::size_t maximum_element_count)
    {
        if (maximum_element_count == 0 ||
            !cuda_backend_unified_addressing_available())
        {
            return nullptr;
        }

        return std::make_unique<
            CudaBatchInverseSessionImpl>(
            maximum_element_count);
    }

    bool barycentric_weights_gpu(
        const uint64_t omega_limbs[4],
        const uint64_t x_limbs[4],
        const uint64_t zero_eval_limbs[4],
        const uint64_t inv_n_limbs[4],
        uint64_t *output_weight_limbs,
        std::size_t domain_size)
    {
        if (domain_size == 0)
        {
            return true;
        }

        if (output_weight_limbs == nullptr)
        {
            return false;
        }

        gpu::GpuFr omega{};
        gpu::GpuFr x{};
        gpu::GpuFr zero_eval{};
        gpu::GpuFr inv_n{};

        std::memcpy(
            omega.d,
            omega_limbs,
            sizeof(omega.d));

        std::memcpy(
            x.d,
            x_limbs,
            sizeof(x.d));

        std::memcpy(
            zero_eval.d,
            zero_eval_limbs,
            sizeof(zero_eval.d));

        std::memcpy(
            inv_n.d,
            inv_n_limbs,
            sizeof(inv_n.d));

        if ((zero_eval.d[0] |
             zero_eval.d[1] |
             zero_eval.d[2] |
             zero_eval.d[3]) == 0ULL)
        {
            return false;
        }

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_barycentric(
                domain_size))
        {
            return false;
        }

        const auto stream =
            workspace.stream();

        auto *d_denominators =
            workspace.scratch_a();

        auto *d_prefixes =
            workspace.prefixes();

        auto *d_suffixes =
            workspace.suffixes();

        auto *d_global_inverse =
            workspace.global_inverse();

        auto *d_weights =
            workspace.scratch_b();

        auto &scan_levels =
            workspace.scan_levels();

        const auto block_count_size =
            (domain_size +
             kBarycentricScanBlockSize -
             1U) /
            kBarycentricScanBlockSize;

        if (block_count_size >
            std::numeric_limits<
                unsigned>::max())
        {
            return false;
        }

        std::size_t kernel_count = 0;

        barycentric_prepare_parallel_kernel<<<
            static_cast<unsigned>(
                block_count_size),
            kBarycentricScanBlockSize,
            0,
            stream>>>(
            omega,
            x,
            d_denominators,
            domain_size);

        auto status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            return false;
        }

        ++kernel_count;

        if (!launch_exclusive_product_scan(
                d_denominators,
                d_prefixes,
                domain_size,
                false,
                scan_levels,
                0,
                &kernel_count,
                stream) ||
            !launch_exclusive_product_scan(
                d_denominators,
                d_suffixes,
                domain_size,
                true,
                scan_levels,
                0,
                &kernel_count,
                stream))
        {
            return false;
        }

        barycentric_invert_product_kernel<<<
            1,
            1,
            0,
            stream>>>(
            d_denominators,
            d_prefixes,
            d_global_inverse,
            domain_size);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            return false;
        }

        ++kernel_count;

        barycentric_finalize_parallel_kernel<<<
            static_cast<unsigned>(
                block_count_size),
            kBarycentricScanBlockSize,
            0,
            stream>>>(
            d_denominators,
            d_prefixes,
            d_suffixes,
            d_global_inverse,
            x,
            zero_eval,
            inv_n,
            d_weights,
            domain_size);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            return false;
        }

        ++kernel_count;

        const auto element_bytes =
            domain_size *
            sizeof(gpu::GpuFr);

        status =
            cudaMemcpyAsync(
                output_weight_limbs,
                d_weights,
                element_bytes,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            return false;
        }

        record_cuda_d2h(
            element_bytes);

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            return false;
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        return true;
    }
    bool batch_inverse_cuda(
        const mcl::Fr *input,
        mcl::Fr *output,
        std::size_t element_count)
    {
        if (element_count == 0)
        {
            return true;
        }

        if (!cuda_backend_unified_addressing_available())
        {
            return false;
        }

        if (input == nullptr ||
            output == nullptr)
        {
            throw std::runtime_error(
                "CUDA batch inverse received null storage");
        }

        static_assert(
            sizeof(mcl::Fr) ==
            sizeof(gpu::GpuFr));

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_product(
                element_count))
        {
            throw std::runtime_error(
                "CUDA batch inverse workspace allocation failed");
        }

        const auto stream =
            workspace.stream();

        auto *prefixes =
            workspace.prefixes();

        auto *suffixes =
            workspace.suffixes();

        auto *global_inverse =
            workspace.global_inverse();

        auto *error =
            workspace.error();

        auto &scan_levels =
            workspace.scan_levels();

        auto status =
            cudaMemsetAsync(
                error,
                0,
                sizeof(unsigned int),
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "batch inverse error reset",
                status);
        }

        constexpr unsigned block_size =
            256;

        const auto blocks =
            static_cast<unsigned>(
                (element_count +
                 block_size - 1U) /
                block_size);

        const auto *gpu_input =
            reinterpret_cast<
                const gpu::GpuFr *>(
                input);

        auto *gpu_output =
            reinterpret_cast<
                gpu::GpuFr *>(
                output);

        validate_nonzero_kernel<<<
            blocks,
            block_size,
            0,
            stream>>>(
            gpu_input,
            error,
            element_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "batch inverse validation launch",
                status);
        }

        std::size_t kernel_count = 1;

        if (!launch_exclusive_product_scan(
                gpu_input,
                prefixes,
                element_count,
                false,
                scan_levels,
                0,
                &kernel_count,
                stream) ||
            !launch_exclusive_product_scan(
                gpu_input,
                suffixes,
                element_count,
                true,
                scan_levels,
                0,
                &kernel_count,
                stream))
        {
            throw std::runtime_error(
                "CUDA batch inverse scan failed");
        }

        barycentric_invert_product_kernel<<<
            1,
            1,
            0,
            stream>>>(
            gpu_input,
            prefixes,
            global_inverse,
            element_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "batch inverse product inversion launch",
                status);
        }

        ++kernel_count;

        inverse_finalize_kernel<<<
            blocks,
            block_size,
            0,
            stream>>>(
            prefixes,
            suffixes,
            global_inverse,
            gpu_output,
            element_count,
            false);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "batch inverse finalize launch",
                status);
        }

        ++kernel_count;

        unsigned int host_error = 0;

        status =
            cudaMemcpyAsync(
                &host_error,
                error,
                sizeof(host_error),
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "batch inverse error download",
                status);
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "batch inverse synchronization",
                status);
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        if (host_error != 0)
        {
            throw std::runtime_error(
                "CUDA batch inverse encountered zero");
        }

        return true;
    }

    bool opening_inverse_denominators_cuda(
        mcl::Fr *output,
        std::size_t domain_size,
        const mcl::Fr &omega,
        const mcl::Fr &point)
    {
        if (domain_size == 0)
        {
            return true;
        }

        if (!cuda_backend_unified_addressing_available())
        {
            return false;
        }

        gpu::GpuFr gpu_omega{};
        gpu::GpuFr gpu_point{};

        std::memcpy(
            gpu_omega.d,
            &omega,
            sizeof(gpu_omega.d));

        std::memcpy(
            gpu_point.d,
            &point,
            sizeof(gpu_point.d));

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_product(
                domain_size))
        {
            throw std::runtime_error(
                "CUDA opening inverse workspace allocation failed");
        }

        const auto stream =
            workspace.stream();

        auto *prefixes =
            workspace.prefixes();

        auto *suffixes =
            workspace.suffixes();

        auto *global_inverse =
            workspace.global_inverse();

        auto *error =
            workspace.error();

        auto &scan_levels =
            workspace.scan_levels();

        auto status =
            cudaMemsetAsync(
                error,
                0,
                sizeof(unsigned int),
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening inverse error reset",
                status);
        }

        constexpr unsigned block_size =
            kBarycentricScanBlockSize;

        const auto blocks =
            static_cast<unsigned>(
                (domain_size +
                 block_size - 1U) /
                block_size);

        auto *denominators =
            reinterpret_cast<
                gpu::GpuFr *>(
                output);

        barycentric_prepare_parallel_kernel<<<
            blocks,
            block_size,
            0,
            stream>>>(
            gpu_omega,
            gpu_point,
            denominators,
            domain_size);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening denominator preparation launch",
                status);
        }

        validate_nonzero_kernel<<<
            blocks,
            block_size,
            0,
            stream>>>(
            denominators,
            error,
            domain_size);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening denominator validation launch",
                status);
        }

        std::size_t kernel_count = 2;

        if (!launch_exclusive_product_scan(
                denominators,
                prefixes,
                domain_size,
                false,
                scan_levels,
                0,
                &kernel_count,
                stream) ||
            !launch_exclusive_product_scan(
                denominators,
                suffixes,
                domain_size,
                true,
                scan_levels,
                0,
                &kernel_count,
                stream))
        {
            throw std::runtime_error(
                "CUDA opening inverse product scan failed");
        }

        barycentric_invert_product_kernel<<<
            1,
            1,
            0,
            stream>>>(
            denominators,
            prefixes,
            global_inverse,
            domain_size);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening product inversion launch",
                status);
        }

        ++kernel_count;

        inverse_finalize_kernel<<<
            blocks,
            block_size,
            0,
            stream>>>(
            prefixes,
            suffixes,
            global_inverse,
            denominators,
            domain_size,
            true);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening inverse finalize launch",
                status);
        }

        ++kernel_count;

        unsigned int host_error = 0;

        status =
            cudaMemcpyAsync(
                &host_error,
                error,
                sizeof(host_error),
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening inverse error download",
                status);
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "opening inverse synchronization",
                status);
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        if (host_error != 0)
        {
            throw std::runtime_error(
                "opening point overlaps evaluation domain");
        }

        return true;
    }
    bool opening_quotient_assembly_cuda(
        const mcl::Fr *folded_values,
        mcl::Fr *quotient_values,
        const mcl::Fr &scaled_claim_sum,
        std::size_t element_count)
    {
        if (element_count == 0)
            return true;
        if (!cuda_backend_unified_addressing_available())
            return false;

        gpu::GpuFr gpu_claim{};
        std::memcpy(gpu_claim.d, &scaled_claim_sum, sizeof(gpu_claim.d));
        constexpr unsigned block_size = 256;
        const auto blocks = static_cast<unsigned>((element_count + block_size - 1U) / block_size);
        auto &workspace = persistent_scan_workspace();
        const auto stream = workspace.stream();

        opening_quotient_assembly_kernel<<<blocks, block_size, 0, stream>>>(
            reinterpret_cast<const gpu::GpuFr *>(folded_values),
            reinterpret_cast<gpu::GpuFr *>(quotient_values),
            gpu_claim,
            element_count);

        auto status = cudaGetLastError();
        if (status != cudaSuccess)
            throw_scan_cuda_error("opening quotient assembly launch", status);
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw_scan_cuda_error("opening quotient assembly synchronization", status);
        record_cuda_kernel_invocations(1);
        record_cuda_stream_sync();
        return true;
    }

    bool opening_inverse_and_quotient_cuda(
        const mcl::Fr *folded_values,
        mcl::Fr *quotient_values,
        const mcl::Fr &scaled_claim_sum,
        std::size_t domain_size,
        const mcl::Fr &omega,
        const mcl::Fr &point)
    {
        if (domain_size == 0)
            return true;
        if (!cuda_backend_unified_addressing_available())
            return false;

        gpu::GpuFr gpu_omega{};
        gpu::GpuFr gpu_point{};
        gpu::GpuFr gpu_claim{};
        std::memcpy(gpu_omega.d, &omega, sizeof(gpu_omega.d));
        std::memcpy(gpu_point.d, &point, sizeof(gpu_point.d));
        std::memcpy(gpu_claim.d, &scaled_claim_sum, sizeof(gpu_claim.d));

        auto &workspace = persistent_scan_workspace();
        if (!workspace.ensure_product(domain_size))
            throw std::runtime_error("CUDA fused opening workspace allocation failed");

        const auto stream = workspace.stream();
        auto *prefixes = workspace.prefixes();
        auto *suffixes = workspace.suffixes();
        auto *global_inverse = workspace.global_inverse();
        auto *error = workspace.error();
        auto &scan_levels = workspace.scan_levels();
        auto *denominators = reinterpret_cast<gpu::GpuFr *>(quotient_values);

        auto status = cudaMemsetAsync(error, 0, sizeof(unsigned int), stream);
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening error reset", status);

        constexpr unsigned block_size = kBarycentricScanBlockSize;
        const auto blocks = static_cast<unsigned>((domain_size + block_size - 1U) / block_size);

        barycentric_prepare_parallel_kernel<<<blocks, block_size, 0, stream>>>(gpu_omega, gpu_point, denominators, domain_size);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening denominator launch", status);

        validate_nonzero_kernel<<<blocks, block_size, 0, stream>>>(denominators, error, domain_size);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening denominator validation", status);

        std::size_t kernel_count = 2;
        if (!launch_exclusive_product_scan(denominators, prefixes, domain_size, false, scan_levels, 0, &kernel_count, stream) ||
            !launch_exclusive_product_scan(denominators, suffixes, domain_size, true, scan_levels, 0, &kernel_count, stream))
            throw std::runtime_error("CUDA fused opening product scan failed");

        barycentric_invert_product_kernel<<<1, 1, 0, stream>>>(denominators, prefixes, global_inverse, domain_size);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening product inversion", status);
        ++kernel_count;

        inverse_finalize_kernel<<<blocks, block_size, 0, stream>>>(prefixes, suffixes, global_inverse, denominators, domain_size, true);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening inverse finalize", status);
        ++kernel_count;

        opening_quotient_assembly_kernel<<<blocks, block_size, 0, stream>>>(
            reinterpret_cast<const gpu::GpuFr *>(folded_values),
            reinterpret_cast<gpu::GpuFr *>(quotient_values),
            gpu_claim,
            domain_size);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening quotient assembly", status);
        ++kernel_count;

        unsigned int host_error = 0;
        status = cudaMemcpyAsync(&host_error, error, sizeof(host_error), cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening error download", status);
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw_scan_cuda_error("fused opening synchronization", status);

        record_cuda_kernel_invocations(kernel_count);
        record_cuda_stream_sync();
        if (host_error != 0)
            throw std::runtime_error("opening point overlaps evaluation domain");
        return true;
    }

    bool opening_inverse_and_quotient_resident_cuda(
        const mcl::Fr *folded_values,
        const mcl::Fr &scaled_claim_sum,
        std::size_t domain_size,
        const mcl::Fr &omega,
        const mcl::Fr &point,
        std::shared_ptr<
            const CudaResidentCoefficientBuffer>
            *resident_output)
    {
        if (resident_output == nullptr)
        {
            throw std::runtime_error(
                "CUDA resident opening descriptor is null");
        }

        resident_output->reset();

        if (domain_size == 0 ||
            folded_values == nullptr)
        {
            return false;
        }

        int device = -1;

        auto status =
            cudaGetDevice(
                &device);

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "resident opening device lookup",
                status);
        }

        mcl::Fr *device_quotient =
            nullptr;

        const auto byte_count =
            domain_size *
            sizeof(mcl::Fr);

        status =
            cudaMalloc(
                reinterpret_cast<
                    void **>(
                    &device_quotient),
                byte_count);

        if (status ==
            cudaErrorMemoryAllocation)
        {
            (void)cudaGetLastError();
            return false;
        }

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "resident opening quotient allocation",
                status);
        }

        try
        {
            if (!opening_inverse_and_quotient_cuda(
                    folded_values,
                    device_quotient,
                    scaled_claim_sum,
                    domain_size,
                    omega,
                    point))
            {
                cudaFree(
                    device_quotient);

                return false;
            }
        }
        catch (...)
        {
            (void)cudaFree(
                device_quotient);

            throw;
        }

        auto *owned =
            device_quotient;

        std::shared_ptr<void>
            allocation(
                owned,
                [device](
                    void *pointer) noexcept
                {
                    if (pointer == nullptr)
                    {
                        return;
                    }

                    int previous = -1;

                    if (cudaGetDevice(
                            &previous) !=
                        cudaSuccess)
                    {
                        return;
                    }

                    if (previous !=
                            device &&
                        cudaSetDevice(
                            device) !=
                            cudaSuccess)
                    {
                        return;
                    }

                    (void)cudaFree(
                        pointer);

                    if (previous !=
                        device)
                    {
                        (void)cudaSetDevice(
                            previous);
                    }
                });

        auto descriptor =
            std::make_shared<
                CudaResidentCoefficientBuffer>();

        descriptor->allocation =
            std::move(
                allocation);

        descriptor
            ->device_montgomery_limbs =
            reinterpret_cast<
                const std::uint64_t *>(
                owned);

        descriptor->capacity =
            domain_size;

        descriptor->logical_count =
            domain_size;

        descriptor->nonzero_count =
            domain_size;

        descriptor->nonzero_count_known =
            false;

        descriptor->device =
            device;

        descriptor->host_transfer.reset();

        *resident_output =
            std::move(
                descriptor);

        record_cuda_opening_fold_call();

        return true;
    }

    bool coefficient_opening_quotient_resident_cuda(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const mcl::Fr &point,
        const mcl::Fr &claimed_value,
        std::shared_ptr<
            const CudaResidentCoefficientBuffer>
            *resident_output)
    {
        if (resident_output == nullptr)
        {
            throw std::runtime_error(
                "CUDA resident coefficient opening descriptor is null");
        }

        resident_output->reset();

        if (coefficients == nullptr ||
            coefficient_count <= 1 ||
            !cuda_backend_unified_addressing_available())
        {
            return false;
        }

        const auto quotient_size =
            coefficient_count - 1;

        const auto block_count =
            (quotient_size +
             kCoefficientOpeningRowsPerThread -
             1) /
            kCoefficientOpeningRowsPerThread;

        if (block_count == 0)
        {
            return false;
        }

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_sum(
                block_count,
                false))
        {
            return false;
        }

        const auto stream =
            workspace.stream();

        auto *block_constants =
            workspace.scratch_a();

        auto *block_incoming =
            workspace.prefixes();

        mcl::Fr *device_quotient =
            nullptr;

        auto status =
            cudaMalloc(
                reinterpret_cast<
                    void **>(
                    &device_quotient),
                quotient_size *
                    sizeof(mcl::Fr));

        if (status ==
            cudaErrorMemoryAllocation)
        {
            (void)cudaGetLastError();
            return false;
        }

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "coefficient opening quotient allocation",
                status);
        }

        gpu::GpuFr gpu_point{};

        std::memcpy(
            gpu_point.d,
            &point,
            sizeof(gpu_point.d));

        constexpr unsigned
            kBlockSize = 256;

        const auto grid_size =
            (block_count +
             kBlockSize - 1) /
            kBlockSize;

        if (grid_size >
            std::numeric_limits<
                unsigned>::max())
        {
            cudaFree(
                device_quotient);

            throw std::runtime_error(
                "CUDA coefficient opening grid exceeds CUDA limits");
        }

        try
        {
            coefficient_opening_block_summary_kernel<<<
                static_cast<unsigned>(
                    grid_size),
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    coefficients),
                gpu_point,
                block_constants,
                quotient_size);

            status =
                cudaGetLastError();

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening summary launch",
                    status);
            }

            std::vector<mcl::Fr>
                host_constants(
                    block_count);

            const auto block_bytes =
                block_count *
                sizeof(mcl::Fr);

            status =
                cudaMemcpyAsync(
                    host_constants.data(),
                    block_constants,
                    block_bytes,
                    cudaMemcpyDeviceToHost,
                    stream);

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening summary download",
                    status);
            }

            record_cuda_d2h(
                block_bytes);

            status =
                cudaStreamSynchronize(
                    stream);

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening summary synchronization",
                    status);
            }

            record_cuda_stream_sync();

            std::vector<mcl::Fr>
                host_incoming(
                    block_count);

            mcl::Fr boundary;
            boundary.clear();

            const auto point_field =
                FieldElement::
                    from_native(
                        point);

            const auto full_multiplier =
                point_field
                    .pow(
                        kCoefficientOpeningRowsPerThread)
                    .native();

            for (std::size_t block =
                     block_count;
                 block-- > 0;)
            {
                host_incoming[block] =
                    boundary;

                const auto begin =
                    block *
                    kCoefficientOpeningRowsPerThread;

                const auto block_length =
                    std::min(
                        kCoefficientOpeningRowsPerThread,
                        quotient_size -
                            begin);

                const auto multiplier =
                    block_length ==
                            kCoefficientOpeningRowsPerThread
                        ? full_multiplier
                        : point_field
                              .pow(
                                  block_length)
                              .native();

                mcl::Fr scaled;

                mcl::Fr::mul(
                    scaled,
                    boundary,
                    multiplier);

                mcl::Fr::add(
                    boundary,
                    host_constants[block],
                    scaled);
            }

            status =
                cudaMemcpyAsync(
                    block_incoming,
                    host_incoming.data(),
                    block_bytes,
                    cudaMemcpyHostToDevice,
                    stream);

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening incoming upload",
                    status);
            }

            record_cuda_h2d(
                block_bytes);

            coefficient_opening_emit_kernel<<<
                static_cast<unsigned>(
                    grid_size),
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    coefficients),
                gpu_point,
                block_incoming,
                reinterpret_cast<
                    gpu::GpuFr *>(
                    device_quotient),
                quotient_size);

            status =
                cudaGetLastError();

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening emit launch",
                    status);
            }

            mcl::Fr quotient_zero;

            status =
                cudaMemcpyAsync(
                    &quotient_zero,
                    device_quotient,
                    sizeof(mcl::Fr),
                    cudaMemcpyDeviceToHost,
                    stream);

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening remainder download",
                    status);
            }

            record_cuda_d2h(
                sizeof(mcl::Fr));

            status =
                cudaStreamSynchronize(
                    stream);

            if (status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "coefficient opening emit synchronization",
                    status);
            }

            record_cuda_stream_sync();

            record_cuda_kernel_invocations(
                2);

            mcl::Fr remainder;
            mcl::Fr product;

            mcl::Fr::sub(
                remainder,
                coefficients[0],
                claimed_value);

            mcl::Fr::mul(
                product,
                point,
                quotient_zero);

            mcl::Fr::add(
                remainder,
                remainder,
                product);

            if (!remainder.isZero())
            {
                throw std::runtime_error(
                    "claimed value does not match committed polynomial");
            }
        }
        catch (...)
        {
            (void)cudaFree(
                device_quotient);

            throw;
        }

        int device = -1;

        status =
            cudaGetDevice(
                &device);

        if (status !=
            cudaSuccess)
        {
            cudaFree(
                device_quotient);

            throw_scan_cuda_error(
                "coefficient opening resident device lookup",
                status);
        }

        auto *owned =
            device_quotient;

        std::shared_ptr<void>
            allocation(
                owned,
                [device](
                    void *pointer) noexcept
                {
                    if (pointer == nullptr)
                    {
                        return;
                    }

                    int previous = -1;

                    if (cudaGetDevice(
                            &previous) !=
                        cudaSuccess)
                    {
                        return;
                    }

                    if (previous !=
                            device &&
                        cudaSetDevice(
                            device) !=
                            cudaSuccess)
                    {
                        return;
                    }

                    (void)cudaFree(
                        pointer);

                    if (previous !=
                        device)
                    {
                        (void)cudaSetDevice(
                            previous);
                    }
                });

        auto descriptor =
            std::make_shared<
                CudaResidentCoefficientBuffer>();

        descriptor->allocation =
            std::move(
                allocation);

        descriptor
            ->device_montgomery_limbs =
            reinterpret_cast<
                const std::uint64_t *>(
                owned);

        descriptor->capacity =
            quotient_size;

        descriptor->logical_count =
            quotient_size;

        descriptor->nonzero_count =
            quotient_size;

        descriptor->nonzero_count_known =
            false;

        descriptor->device =
            device;

        descriptor->host_transfer.reset();

        *resident_output =
            std::move(
                descriptor);

        record_cuda_opening_fold_call();

        return true;
    }

    bool logup_inverse_columns_cuda(
        const mcl::Fr *table,
        const mcl::Fr *query,
        const mcl::Fr *q_table,
        const mcl::Fr *q_query,
        const mcl::Fr &beta,
        mcl::Fr *table_inverse,
        mcl::Fr *query_inverse,
        std::size_t element_count)
    {
        if (element_count == 0)
        {
            return true;
        }

        if (!cuda_backend_unified_addressing_available())
        {
            return false;
        }

        gpu::GpuFr gpu_beta{};

        std::memcpy(
            gpu_beta.d,
            &beta,
            sizeof(gpu_beta.d));

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_product(
                element_count))
        {
            throw std::runtime_error(
                "CUDA LogUp inverse workspace allocation failed");
        }

        const auto stream =
            workspace.stream();

        auto *prefixes =
            workspace.prefixes();

        auto *suffixes =
            workspace.suffixes();

        auto *global_inverse =
            workspace.global_inverse();

        auto *error =
            workspace.error();

        auto &scan_levels =
            workspace.scan_levels();

        auto status =
            cudaMemsetAsync(
                error,
                0,
                sizeof(unsigned int),
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp inverse error reset",
                status);
        }

        constexpr unsigned block_size =
            256;

        const auto blocks =
            static_cast<unsigned>(
                (element_count +
                 block_size - 1U) /
                block_size);

        std::size_t kernel_count = 0;

        const auto run_column =
            [&](const mcl::Fr *values,
                const mcl::Fr *selectors,
                mcl::Fr *output)
        {
            auto *gpu_output =
                reinterpret_cast<
                    gpu::GpuFr *>(
                    output);

            logup_prepare_denominator_kernel<<<
                blocks,
                block_size,
                0,
                stream>>>(
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    values),
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    selectors),
                gpu_beta,
                gpu_output,
                error,
                element_count);

            auto launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "LogUp denominator preparation launch",
                    launch_status);
            }

            ++kernel_count;

            if (!launch_exclusive_product_scan(
                    gpu_output,
                    prefixes,
                    element_count,
                    false,
                    scan_levels,
                    0,
                    &kernel_count,
                    stream) ||
                !launch_exclusive_product_scan(
                    gpu_output,
                    suffixes,
                    element_count,
                    true,
                    scan_levels,
                    0,
                    &kernel_count,
                    stream))
            {
                throw std::runtime_error(
                    "CUDA LogUp inverse scan failed");
            }

            barycentric_invert_product_kernel<<<
                1,
                1,
                0,
                stream>>>(
                gpu_output,
                prefixes,
                global_inverse,
                element_count);

            launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "LogUp product inversion launch",
                    launch_status);
            }

            ++kernel_count;

            logup_finalize_inverse_kernel<<<
                blocks,
                block_size,
                0,
                stream>>>(
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    selectors),
                prefixes,
                suffixes,
                global_inverse,
                gpu_output,
                element_count);

            launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "LogUp inverse finalize launch",
                    launch_status);
            }

            ++kernel_count;
        };

        run_column(
            table,
            q_table,
            table_inverse);

        run_column(
            query,
            q_query,
            query_inverse);

        unsigned int host_error = 0;

        status =
            cudaMemcpyAsync(
                &host_error,
                error,
                sizeof(host_error),
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp inverse error download",
                status);
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp inverse synchronization",
                status);
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        if (host_error == 1)
        {
            throw std::runtime_error(
                "cannot invert zero in lookup inverse columns");
        }

        if (host_error == 2)
        {
            throw std::runtime_error(
                "lookup inverse selector is not boolean");
        }

        return true;
    }
    bool logup_accumulator_cuda(
        const mcl::Fr *table_inverse,
        const mcl::Fr *query_inverse,
        const mcl::Fr *multiplicity,
        const mcl::Fr *q_table,
        const mcl::Fr *q_query,
        mcl::Fr *accumulator,
        std::size_t active_count,
        std::size_t element_count)
    {
        if (element_count == 0 ||
            active_count == 0)
        {
            return true;
        }

        if (!cuda_backend_unified_addressing_available())
        {
            return false;
        }

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_sum(
                active_count,
                true))
        {
            throw std::runtime_error(
                "CUDA LogUp accumulator workspace allocation failed");
        }

        const auto stream =
            workspace.stream();

        auto *terms =
            workspace.scratch_a();

        auto *prefixes =
            workspace.prefixes();

        auto *error =
            workspace.error();

        auto &scan_levels =
            workspace.scan_levels();

        auto status =
            cudaMemsetAsync(
                error,
                0,
                sizeof(unsigned int),
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp accumulator error reset",
                status);
        }

        constexpr unsigned block_size =
            256;

        const auto active_blocks =
            static_cast<unsigned>(
                (active_count +
                 block_size - 1U) /
                block_size);

        logup_term_kernel<<<
            active_blocks,
            block_size,
            0,
            stream>>>(
            reinterpret_cast<
                const gpu::GpuFr *>(
                table_inverse),
            reinterpret_cast<
                const gpu::GpuFr *>(
                query_inverse),
            reinterpret_cast<
                const gpu::GpuFr *>(
                multiplicity),
            reinterpret_cast<
                const gpu::GpuFr *>(
                q_table),
            reinterpret_cast<
                const gpu::GpuFr *>(
                q_query),
            terms,
            error,
            active_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp term launch",
                status);
        }

        std::size_t kernel_count = 1;

        if (!launch_exclusive_sum_scan(
                terms,
                prefixes,
                active_count,
                scan_levels,
                0,
                &kernel_count,
                stream))
        {
            throw std::runtime_error(
                "CUDA LogUp accumulator scan failed");
        }

        const auto output_blocks =
            static_cast<unsigned>(
                (element_count +
                 block_size - 1U) /
                block_size);

        prefix_accumulator_finalize_kernel<<<
            output_blocks,
            block_size,
            0,
            stream>>>(
            terms,
            prefixes,
            reinterpret_cast<
                gpu::GpuFr *>(
                accumulator),
            active_count,
            element_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp accumulator finalize launch",
                status);
        }

        ++kernel_count;

        unsigned int host_error = 0;

        status =
            cudaMemcpyAsync(
                &host_error,
                error,
                sizeof(host_error),
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp accumulator error download",
                status);
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "LogUp accumulator synchronization",
                status);
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        if (host_error != 0)
        {
            throw std::runtime_error(
                "lookup accumulator selector is not boolean");
        }

        return true;
    }
    bool logup_inverse_and_accumulator_cuda(
        const mcl::Fr *table,
        const mcl::Fr *query,
        const mcl::Fr *q_table,
        const mcl::Fr *q_query,
        const mcl::Fr &beta,
        const mcl::Fr *multiplicity,
        mcl::Fr *table_inverse,
        mcl::Fr *query_inverse,
        mcl::Fr *accumulator,
        std::size_t active_count,
        std::size_t element_count,
        CudaLogupExecutionProfile *profile)
    {
        if (profile != nullptr)
        {
            *profile =
                CudaLogupExecutionProfile{};
        }

        if (element_count == 0 ||
            active_count == 0)
        {
            return true;
        }

        if (!cuda_backend_unified_addressing_available())
        {
            return false;
        }

        if (active_count >=
            element_count)
        {
            throw std::runtime_error(
                "CUDA fused LogUp active count exceeds domain");
        }

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_logup(
                element_count))
        {
            throw std::runtime_error(
                "CUDA fused LogUp workspace allocation failed");
        }

        const auto stream =
            workspace.stream();

        auto *prefixes =
            workspace.prefixes();

        auto *suffixes =
            workspace.suffixes();

        auto *terms =
            workspace.scratch_a();

        auto *global_inverse =
            workspace.global_inverse();

        auto *error =
            workspace.error();

        auto &scan_levels =
            workspace.scan_levels();

        gpu::GpuFr gpu_beta{};

        std::memcpy(
            gpu_beta.d,
            &beta,
            sizeof(gpu_beta.d));

        auto status =
            cudaMemsetAsync(
                error,
                0,
                sizeof(unsigned int),
                stream);

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "fused LogUp error reset",
                status);
        }

        struct TimingEvents
        {
            cudaEvent_t start = nullptr;
            cudaEvent_t inverse_done = nullptr;
            cudaEvent_t accumulator_done = nullptr;
            bool ready = false;

            ~TimingEvents()
            {
                if (start != nullptr)
                    (void)cudaEventDestroy(start);

                if (inverse_done != nullptr)
                    (void)cudaEventDestroy(inverse_done);

                if (accumulator_done != nullptr)
                    (void)cudaEventDestroy(accumulator_done);
            }
        };

        TimingEvents timing;

        if (profile != nullptr)
        {
            if (cudaEventCreate(
                    &timing.start) ==
                    cudaSuccess &&
                cudaEventCreate(
                    &timing.inverse_done) ==
                    cudaSuccess &&
                cudaEventCreate(
                    &timing.accumulator_done) ==
                    cudaSuccess)
            {
                if (cudaEventRecord(
                        timing.start,
                        stream) ==
                    cudaSuccess)
                {
                    timing.ready =
                        true;
                }
            }

            if (!timing.ready)
            {
                (void)cudaGetLastError();
            }
        }

        constexpr unsigned
            block_size = 256;

        const auto blocks =
            static_cast<unsigned>(
                (element_count +
                 block_size - 1U) /
                block_size);

        std::size_t kernel_count = 0;

        const auto run_column =
            [&](const mcl::Fr *values,
                const mcl::Fr *selectors,
                mcl::Fr *output)
        {
            auto *gpu_output =
                reinterpret_cast<
                    gpu::GpuFr *>(
                    output);

            logup_prepare_denominator_kernel<<<
                blocks,
                block_size,
                0,
                stream>>>(
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    values),
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    selectors),
                gpu_beta,
                gpu_output,
                error,
                element_count);

            auto launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "fused LogUp denominator preparation",
                    launch_status);
            }

            ++kernel_count;

            if (!launch_exclusive_product_scan(
                    gpu_output,
                    prefixes,
                    element_count,
                    false,
                    scan_levels,
                    0,
                    &kernel_count,
                    stream) ||
                !launch_exclusive_product_scan(
                    gpu_output,
                    suffixes,
                    element_count,
                    true,
                    scan_levels,
                    0,
                    &kernel_count,
                    stream))
            {
                throw std::runtime_error(
                    "CUDA fused LogUp inverse scan failed");
            }

            barycentric_invert_product_kernel<<<
                1,
                1,
                0,
                stream>>>(
                gpu_output,
                prefixes,
                global_inverse,
                element_count);

            launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "fused LogUp product inversion",
                    launch_status);
            }

            ++kernel_count;

            logup_finalize_inverse_kernel<<<
                blocks,
                block_size,
                0,
                stream>>>(
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    selectors),
                prefixes,
                suffixes,
                global_inverse,
                gpu_output,
                element_count);

            launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw_scan_cuda_error(
                    "fused LogUp inverse finalize",
                    launch_status);
            }

            ++kernel_count;
        };

        run_column(
            table,
            q_table,
            table_inverse);

        run_column(
            query,
            q_query,
            query_inverse);

        if (timing.ready)
        {
            if (cudaEventRecord(
                    timing.inverse_done,
                    stream) !=
                cudaSuccess)
            {
                timing.ready =
                    false;

                (void)cudaGetLastError();
            }
        }

        const auto active_blocks =
            static_cast<unsigned>(
                (active_count +
                 block_size - 1U) /
                block_size);

        logup_term_kernel<<<
            active_blocks,
            block_size,
            0,
            stream>>>(
            reinterpret_cast<
                const gpu::GpuFr *>(
                table_inverse),
            reinterpret_cast<
                const gpu::GpuFr *>(
                query_inverse),
            reinterpret_cast<
                const gpu::GpuFr *>(
                multiplicity),
            reinterpret_cast<
                const gpu::GpuFr *>(
                q_table),
            reinterpret_cast<
                const gpu::GpuFr *>(
                q_query),
            terms,
            error,
            active_count);

        status =
            cudaGetLastError();

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "fused LogUp term launch",
                status);
        }

        ++kernel_count;

        if (!launch_exclusive_sum_scan(
                terms,
                prefixes,
                active_count,
                scan_levels,
                0,
                &kernel_count,
                stream))
        {
            throw std::runtime_error(
                "CUDA fused LogUp accumulator scan failed");
        }

        const auto output_blocks =
            static_cast<unsigned>(
                (element_count +
                 block_size - 1U) /
                block_size);

        prefix_accumulator_finalize_kernel<<<
            output_blocks,
            block_size,
            0,
            stream>>>(
            terms,
            prefixes,
            reinterpret_cast<
                gpu::GpuFr *>(
                accumulator),
            active_count,
            element_count);

        status =
            cudaGetLastError();

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "fused LogUp accumulator finalize",
                status);
        }

        ++kernel_count;

        if (timing.ready)
        {
            if (cudaEventRecord(
                    timing.accumulator_done,
                    stream) !=
                cudaSuccess)
            {
                timing.ready =
                    false;

                (void)cudaGetLastError();
            }
        }

        unsigned int host_error = 0;

        status =
            cudaMemcpyAsync(
                &host_error,
                error,
                sizeof(host_error),
                cudaMemcpyDeviceToHost,
                stream);

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "fused LogUp error download",
                status);
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status !=
            cudaSuccess)
        {
            throw_scan_cuda_error(
                "fused LogUp synchronization",
                status);
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        if (profile != nullptr &&
            timing.ready)
        {
            float inverse_ms = 0.0F;
            float accumulator_ms = 0.0F;

            if (cudaEventElapsedTime(
                    &inverse_ms,
                    timing.start,
                    timing.inverse_done) ==
                    cudaSuccess &&
                cudaEventElapsedTime(
                    &accumulator_ms,
                    timing.inverse_done,
                    timing.accumulator_done) ==
                    cudaSuccess)
            {
                profile->inverse_ms =
                    static_cast<double>(
                        inverse_ms);

                profile->accumulator_ms =
                    static_cast<double>(
                        accumulator_ms);
            }
        }

        if (host_error == 1)
        {
            throw std::runtime_error(
                "cannot invert zero in lookup inverse columns");
        }

        if (host_error == 2)
        {
            throw std::runtime_error(
                "lookup selector is not boolean");
        }

        return true;
    }

    bool scaled_prefix_accumulator_cuda(
        const mcl::Fr *values,
        const mcl::Fr &scale,
        mcl::Fr *accumulator,
        std::size_t active_count,
        std::size_t element_count)
    {
        if (element_count == 0 ||
            active_count == 0)
        {
            return true;
        }

        if (!cuda_backend_unified_addressing_available())
        {
            return false;
        }

        gpu::GpuFr gpu_scale{};

        std::memcpy(
            gpu_scale.d,
            &scale,
            sizeof(gpu_scale.d));

        auto &workspace =
            persistent_scan_workspace();

        if (!workspace.ensure_sum(
                active_count,
                false))
        {
            throw std::runtime_error(
                "CUDA prefix accumulator workspace allocation failed");
        }

        const auto stream =
            workspace.stream();

        auto *terms =
            workspace.scratch_a();

        auto *prefixes =
            workspace.prefixes();

        auto &scan_levels =
            workspace.scan_levels();

        constexpr unsigned block_size =
            256;

        const auto active_blocks =
            static_cast<unsigned>(
                (active_count +
                 block_size - 1U) /
                block_size);

        scaled_prefix_term_kernel<<<
            active_blocks,
            block_size,
            0,
            stream>>>(
            reinterpret_cast<
                const gpu::GpuFr *>(
                values),
            gpu_scale,
            terms,
            active_count);

        auto status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "scaled prefix term launch",
                status);
        }

        std::size_t kernel_count = 1;

        if (!launch_exclusive_sum_scan(
                terms,
                prefixes,
                active_count,
                scan_levels,
                0,
                &kernel_count,
                stream))
        {
            throw std::runtime_error(
                "CUDA prefix accumulator scan failed");
        }

        const auto output_blocks =
            static_cast<unsigned>(
                (element_count +
                 block_size - 1U) /
                block_size);

        prefix_accumulator_finalize_kernel<<<
            output_blocks,
            block_size,
            0,
            stream>>>(
            terms,
            prefixes,
            reinterpret_cast<
                gpu::GpuFr *>(
                accumulator),
            active_count,
            element_count);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "scaled prefix finalize launch",
                status);
        }

        ++kernel_count;

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_scan_cuda_error(
                "scaled prefix synchronization",
                status);
        }

        record_cuda_kernel_invocations(
            kernel_count);

        record_cuda_stream_sync();

        return true;
    }
    // =========================================================================
    // CUDA 侧批量加权求值
    // =========================================================================
    // 这里计算 `result[j] = sum_i weights[i] * values[i * row_count + j]`，
    // 对应 proving / verifying 在固定点评估多列 evaluation 多项式的热路径。

    namespace
    {

        constexpr std::size_t kMaxGpuValueBytes =
            2ULL * 1024ULL * 1024ULL * 1024ULL;
        constexpr std::size_t kMaxRotationPartialBytes =
            256ULL * 1024ULL * 1024ULL;

        bool checked_size_mul(
            std::size_t lhs,
            std::size_t rhs,
            std::size_t *result)
        {
            if (result == nullptr ||
                (rhs != 0 &&
                 lhs > std::numeric_limits<std::size_t>::max() / rhs))
            {
                return false;
            }
            *result = lhs * rhs;
            return true;
        }

        bool checked_size_add(
            std::size_t lhs,
            std::size_t rhs,
            std::size_t *result)
        {
            if (result == nullptr ||
                lhs > std::numeric_limits<std::size_t>::max() - rhs)
            {
                return false;
            }
            *result = lhs + rhs;
            return true;
        }

        bool checked_ceil_div(
            std::size_t value,
            std::size_t divisor,
            std::size_t *result)
        {
            if (result == nullptr || divisor == 0)
                return false;

            const auto quotient = value / divisor;
            const auto remainder = value % divisor;
            if (remainder != 0 &&
                quotient == std::numeric_limits<std::size_t>::max())
            {
                return false;
            }
            *result = quotient + (remainder == 0 ? 0 : 1);
            return true;
        }

        struct InterleavedMultiRotationWorkspace
        {
            std::size_t field_bytes = 0;
            std::size_t subset_bytes = 0;
            std::size_t partial_bytes = 0;
            std::size_t output_bytes = 0;
            std::size_t pointer_bytes = 0;
            std::size_t rotation_bytes = 0;
            std::size_t device_peak_bytes = 0;
        };

        bool make_interleaved_multi_rotation_workspace(
            std::size_t domain_size,
            std::size_t poly_count,
            std::size_t rotation_count,
            InterleavedMultiRotationWorkspace *workspace)
        {
            if (workspace == nullptr ||
                domain_size == 0 ||
                poly_count == 0 ||
                rotation_count == 0 ||
                poly_count > 65535 ||
                rotation_count > 65535)
            {
                return false;
            }

            InterleavedMultiRotationWorkspace result;
            std::size_t interleaved_elements = 0;
            std::size_t partial_blocks = 0;
            std::size_t partial_elements = 0;
            std::size_t output_count = 0;
            if (!checked_size_mul(
                    domain_size,
                    sizeof(gpu::GpuFr),
                    &result.field_bytes) ||
                result.field_bytes == 0 ||
                result.field_bytes > kMaxGpuValueBytes ||
                !checked_size_mul(
                    domain_size,
                    poly_count,
                    &interleaved_elements) ||
                !checked_size_mul(
                    interleaved_elements,
                    sizeof(gpu::GpuFr),
                    &result.subset_bytes) ||
                result.subset_bytes == 0 ||
                result.subset_bytes > kMaxGpuValueBytes ||
                !checked_size_mul(
                    poly_count,
                    rotation_count,
                    &output_count) ||
                !checked_size_mul(
                    output_count,
                    sizeof(gpu::GpuFr),
                    &result.output_bytes) ||
                result.output_bytes == 0 ||
                result.output_bytes > kMaxGpuValueBytes ||
                !checked_ceil_div(
                    domain_size,
                    256,
                    &partial_blocks) ||
                partial_blocks > std::numeric_limits<unsigned>::max() ||
                !checked_size_mul(
                    partial_blocks,
                    poly_count,
                    &partial_elements) ||
                !checked_size_mul(
                    partial_elements,
                    rotation_count,
                    &partial_elements) ||
                !checked_size_mul(
                    partial_elements,
                    sizeof(gpu::GpuFr),
                    &result.partial_bytes) ||
                result.partial_bytes == 0 ||
                result.partial_bytes > kMaxRotationPartialBytes ||
                !checked_size_mul(
                    poly_count,
                    sizeof(const gpu::GpuFr *),
                    &result.pointer_bytes) ||
                !checked_size_mul(
                    rotation_count,
                    sizeof(std::size_t),
                    &result.rotation_bytes))
            {
                return false;
            }

            constexpr std::size_t kAdmissionHeadroomBytes =
                64ULL * 1024ULL * 1024ULL;
            auto add_peak = [&result](std::size_t value)
            {
                return checked_size_add(
                    result.device_peak_bytes,
                    value,
                    &result.device_peak_bytes);
            };
            if (!add_peak(result.field_bytes) ||
                !add_peak(result.subset_bytes) ||
                !add_peak(result.subset_bytes) ||
                !add_peak(result.partial_bytes) ||
                !add_peak(result.output_bytes) ||
                !add_peak(result.pointer_bytes) ||
                !add_peak(result.rotation_bytes) ||
                !add_peak(kAdmissionHeadroomBytes))
            {
                return false;
            }

            *workspace = result;
            return true;
        }

        struct WeightedEvalChunkPlan
        {
            std::size_t weight_bytes = 0;
            std::size_t result_bytes = 0;
            std::size_t bytes_per_column = 0;
            std::size_t cols_per_chunk = 0;
        };

        bool make_weighted_eval_chunk_plan(
            std::size_t domain_size,
            std::size_t row_count,
            WeightedEvalChunkPlan *plan)
        {
            if (plan == nullptr ||
                !checked_size_mul(
                    domain_size,
                    sizeof(gpu::GpuFr),
                    &plan->weight_bytes) ||
                !checked_size_mul(
                    row_count,
                    sizeof(gpu::GpuFr),
                    &plan->result_bytes))
            {
                return false;
            }

            // A single column must fit under the hard per-chunk cap.  Keep
            // this check explicit so a future row-count/grid limit cannot
            // silently turn an overflowing denominator into one oversized
            // allocation.
            plan->bytes_per_column = plan->result_bytes;
            if (plan->bytes_per_column == 0 ||
                plan->bytes_per_column > kMaxGpuValueBytes)
            {
                return false;
            }

            const auto max_cols_per_chunk =
                kMaxGpuValueBytes / plan->bytes_per_column;
            if (max_cols_per_chunk == 0)
                return false;

            plan->cols_per_chunk =
                std::min(domain_size, max_cols_per_chunk);
            return plan->cols_per_chunk != 0;
        }

        // ---- Columnar kernel: reads polynomial data via array of column pointers ----
        // 每个线程处理一个域元素，对每个多项式累加 `w[col] * values[row][col]`，
        // 再在 block 内用共享内存做归约。
        __global__ void weighted_eval_columnar_kernel(
            const gpu::GpuFr *__restrict__ weights,         // 主机/统一地址空间中的 `[domain_size]`
            const gpu::GpuFr *const *__restrict__ columns,  // 每列一个指针，均指向 `[domain_size]`
            const std::size_t *__restrict__ column_lengths, // 可选的每列真实长度，尾部按零处理
            gpu::GpuFr *__restrict__ partial_results,       // `[num_blocks * row_count]`
            std::size_t domain_size,
            std::size_t row_count,
            std::size_t weight_rotation)
        {

            const auto tid = threadIdx.x;
            const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
            const auto row = blockIdx.y;

            if (row >= row_count)
                return;

            gpu::GpuFr acc = gpu::fr_zero();
            if (col < domain_size && (column_lengths == nullptr || col < column_lengths[row]))
            {
                const auto weight_index =
                    col >= weight_rotation
                        ? col - weight_rotation
                        : domain_size - (weight_rotation - col);
                acc = gpu::fr_mul(
                    weights[weight_index], columns[row][col]);
            }

            extern __shared__ gpu::GpuFr smem[];
            smem[tid] = acc;
            __syncthreads();

            for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                {
                    smem[tid] = gpu::fr_add(smem[tid], smem[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                partial_results[static_cast<std::size_t>(blockIdx.x) * row_count + row] = smem[0];
            }
        }

        __global__ void weighted_eval_columnar_rotations_kernel(
            const gpu::GpuFr *__restrict__ weights,
            const gpu::GpuFr *const *__restrict__ columns,
            const std::size_t *__restrict__ column_lengths,
            const std::size_t *__restrict__ rotations,
            gpu::GpuFr *__restrict__ partial_results,
            std::size_t domain_size,
            std::size_t row_count,
            std::size_t num_blocks)
        {
            const auto tid = threadIdx.x;
            const auto col =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
            const auto row = static_cast<std::size_t>(blockIdx.y);
            const auto point = static_cast<std::size_t>(blockIdx.z);
            if (row >= row_count)
                return;

            gpu::GpuFr acc = gpu::fr_zero();
            if (col < domain_size && (column_lengths == nullptr || col < column_lengths[row]))
            {
                const auto rotation = rotations[point];
                const auto weight_index =
                    col >= rotation
                        ? col - rotation
                        : domain_size - (rotation - col);
                acc = gpu::fr_mul(
                    weights[weight_index], columns[row][col]);
            }

            extern __shared__ gpu::GpuFr smem[];
            smem[tid] = acc;
            __syncthreads();
            for (unsigned stride = blockDim.x / 2;
                 stride > 0;
                 stride >>= 1)
            {
                if (tid < stride)
                {
                    smem[tid] = gpu::fr_add(
                        smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }
            if (tid == 0)
            {
                partial_results[(point * num_blocks + static_cast<std::size_t>(blockIdx.x)) * row_count + row] = smem[0];
            }
        }

        // Transpose the cached subset from its canonical interleaved layout
        // `[domain][polynomial]` to the column-major device layout consumed
        // by weighted_eval_columnar_rotations_kernel.  The 32x8 tile gives
        // coalesced reads from the interleaved rows and coalesced writes to
        // each output polynomial column without creating a host-side copy.
        __global__ void transpose_interleaved_to_columnar_kernel(
            const gpu::GpuFr *__restrict__ interleaved,
            gpu::GpuFr *__restrict__ columnar,
            std::size_t domain_size,
            std::size_t poly_count)
        {
            constexpr unsigned kTileDim = 32;
            constexpr unsigned kBlockRows = 8;
            __shared__ gpu::GpuFr tile[kTileDim][kTileDim + 1];

            const auto tx = threadIdx.x;
            const auto ty = threadIdx.y;
            const auto input_x =
                static_cast<std::size_t>(blockIdx.x) * kTileDim + tx;
            const auto input_y_base =
                static_cast<std::size_t>(blockIdx.y) * kTileDim;

            for (unsigned offset = 0;
                 offset < kTileDim;
                 offset += kBlockRows)
            {
                const auto input_y = input_y_base + ty + offset;
                if (input_x < poly_count && input_y < domain_size)
                {
                    tile[ty + offset][tx] =
                        interleaved[input_y * poly_count + input_x];
                }
            }
            __syncthreads();

            const auto output_x_base =
                static_cast<std::size_t>(blockIdx.y) * kTileDim;
            const auto output_y_base =
                static_cast<std::size_t>(blockIdx.x) * kTileDim;
            for (unsigned offset = 0;
                 offset < kTileDim;
                 offset += kBlockRows)
            {
                const auto output_x = output_x_base + tx;
                const auto output_y = output_y_base + ty + offset;
                if (output_x < domain_size && output_y < poly_count)
                {
                    columnar[output_y * domain_size + output_x] =
                        tile[tx][ty + offset];
                }
            }
        }

        __global__ void reduce_rotation_partials_kernel(
            const gpu::GpuFr *__restrict__ partial_results,
            gpu::GpuFr *__restrict__ final_results,
            std::size_t num_blocks,
            std::size_t row_count,
            std::size_t rotation_count)
        {
            const auto output_index =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto output_count = row_count * rotation_count;
            if (output_index >= output_count)
                return;
            const auto point = output_index / row_count;
            const auto row = output_index % row_count;
            gpu::GpuFr sum = gpu::fr_zero();
            for (std::size_t block = 0; block < num_blocks; ++block)
            {
                sum = gpu::fr_add(
                    sum,
                    partial_results[(point * num_blocks + block) * row_count + row]);
            }
            final_results[output_index] = sum;
        }

        __global__ void accumulate_rotation_results_kernel(
            const gpu::GpuFr *__restrict__ chunk_results,
            gpu::GpuFr *__restrict__ final_results,
            std::size_t poly_count,
            std::size_t poly_index,
            std::size_t rotation_count)
        {
            const auto point =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                threadIdx.x;
            if (point >= rotation_count)
                return;

            const auto output_index = point * poly_count + poly_index;
            final_results[output_index] = gpu::fr_add(
                final_results[output_index],
                chunk_results[point]);
        }

        // Shared device-column implementation for both the existing
        // columnar multi-rotation API and the cached-subset transpose route.
        // `d_columns` and `d_weights` are already device-resident; only the
        // compact rotation metadata and final result cross the host boundary.
        bool weighted_poly_eval_gpu_multi_rotations_device_columns(
            const gpu::GpuFr *d_weights,
            const gpu::GpuFr *const *d_columns,
            uint64_t *output_limbs,
            std::size_t domain_size,
            std::size_t poly_count,
            const std::size_t *rotations,
            std::size_t rotation_count)
        {
            if (domain_size == 0 || poly_count == 0 || rotation_count == 0)
            {
                return true;
            }
            if (d_weights == nullptr || d_columns == nullptr ||
                output_limbs == nullptr || rotations == nullptr ||
                poly_count > 65535 || rotation_count > 65535)
            {
                return false;
            }

            std::size_t num_blocks_size = 0;
            std::size_t partial_elements = 0;
            std::size_t partial_bytes = 0;
            std::size_t output_count = 0;
            std::size_t output_bytes = 0;
            std::size_t rotation_bytes = 0;
            if (!checked_ceil_div(
                    domain_size,
                    256,
                    &num_blocks_size) ||
                num_blocks_size > std::numeric_limits<unsigned>::max() ||
                !checked_size_mul(
                    num_blocks_size,
                    poly_count,
                    &partial_elements) ||
                !checked_size_mul(
                    partial_elements,
                    rotation_count,
                    &partial_elements) ||
                !checked_size_mul(
                    partial_elements,
                    sizeof(gpu::GpuFr),
                    &partial_bytes) ||
                partial_bytes == 0 ||
                partial_bytes > kMaxRotationPartialBytes ||
                !checked_size_mul(
                    poly_count,
                    rotation_count,
                    &output_count) ||
                !checked_size_mul(
                    output_count,
                    sizeof(gpu::GpuFr),
                    &output_bytes) ||
                output_bytes == 0 ||
                output_bytes > kMaxGpuValueBytes ||
                !checked_size_mul(
                    rotation_count,
                    sizeof(std::size_t),
                    &rotation_bytes))
            {
                return false;
            }

            std::size_t reduce_grid_size = 0;
            if (!checked_ceil_div(
                    output_count,
                    256,
                    &reduce_grid_size) ||
                reduce_grid_size > std::numeric_limits<unsigned>::max())
            {
                return false;
            }

            std::size_t common_domain_bytes = 0;
            if (!checked_size_mul(
                    domain_size,
                    sizeof(gpu::GpuFr),
                    &common_domain_bytes) ||
                common_domain_bytes > kMaxGpuValueBytes)
            {
                return false;
            }

            gpu::GpuFr *d_partials = nullptr;
            gpu::GpuFr *d_final = nullptr;
            std::size_t *d_rotations = nullptr;
            const auto cleanup = [&]()
            {
                cudaFree(d_final);
                cudaFree(d_partials);
                cudaFree(d_rotations);
            };

            if (cudaMalloc(&d_partials, partial_bytes) != cudaSuccess ||
                cudaMalloc(&d_final, output_bytes) != cudaSuccess ||
                cudaMalloc(&d_rotations, rotation_bytes) != cudaSuccess)
            {
                cleanup();
                return false;
            }

            std::vector<std::size_t> normalized_rotations(rotation_count);
            for (std::size_t index = 0;
                 index < rotation_count;
                 ++index)
            {
                normalized_rotations[index] = rotations[index] % domain_size;
            }
            if (cudaMemcpy(
                    d_rotations,
                    normalized_rotations.data(),
                    rotation_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemset(d_partials, 0, partial_bytes) != cudaSuccess ||
                cudaMemset(d_final, 0, output_bytes) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            record_cuda_h2d(rotation_bytes);

            const auto num_blocks =
                static_cast<unsigned>(num_blocks_size);
            const auto reduce_grid =
                static_cast<unsigned>(reduce_grid_size);
            constexpr unsigned kBlockDim = 256;
            const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);
            dim3 grid(
                num_blocks,
                static_cast<unsigned>(poly_count),
                static_cast<unsigned>(rotation_count));
            weighted_eval_columnar_rotations_kernel<<<
                grid,
                kBlockDim,
                smem_bytes>>>(
                d_weights,
                d_columns,
                nullptr,
                d_rotations,
                d_partials,
                domain_size,
                poly_count,
                num_blocks);
            if (cudaGetLastError() != cudaSuccess)
            {
                cleanup();
                return false;
            }

            reduce_rotation_partials_kernel<<<
                reduce_grid,
                kBlockDim>>>(
                d_partials,
                d_final,
                num_blocks,
                poly_count,
                rotation_count);
            if (cudaGetLastError() != cudaSuccess)
            {
                cleanup();
                return false;
            }

            if (cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(
                    output_limbs,
                    d_final,
                    output_bytes,
                    cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            record_cuda_stream_sync();
            record_cuda_d2h(output_bytes);
            cleanup();
            record_cuda_kernel_invocations(2);
            return true;
        }

        // 分块加权求值 kernel。
        // 每个 block 负责域上的一个连续片段，并为全部多项式写出局部和。
        __global__ void weighted_eval_kernel(
            const gpu::GpuFr *__restrict__ weights,   // `[domain_size]`
            const gpu::GpuFr *__restrict__ values,    // 交织布局 `[domain_size * row_count]`
            gpu::GpuFr *__restrict__ partial_results, // `[num_chunks * row_count]`
            std::size_t domain_size,
            std::size_t row_count,
            std::size_t chunk_start,
            std::size_t chunk_end)
        {

            const auto tid = threadIdx.x;
            const auto chunk_id = blockIdx.x;
            const auto row = blockIdx.y;

            if (row >= row_count)
                return;

            const auto real_chunk_start = chunk_start + chunk_id * blockDim.x;
            const auto col = real_chunk_start + tid;

            gpu::GpuFr acc = gpu::fr_zero();
            if (col < chunk_end && col < domain_size)
            {
                // 本次上传的数据从 `chunk_start` 开始，因此这里使用 chunk 内偏移。
                const auto local_col = col - chunk_start;
                acc = gpu::fr_mul(weights[col], values[local_col * row_count + row]);
            }

            // block 内共享内存归约。
            extern __shared__ gpu::GpuFr smem[];
            smem[tid] = acc;
            __syncthreads();

            for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                {
                    smem[tid] = gpu::fr_add(smem[tid], smem[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                partial_results[chunk_id * row_count + row] = smem[0];
            }
        }

        // 对所有 chunk 的局部结果做最终归约。
        __global__ void reduce_partials_kernel(
            const gpu::GpuFr *__restrict__ partial_results, // [num_chunks * row_count]
            gpu::GpuFr *__restrict__ final_results,         // [row_count]
            std::size_t num_chunks,
            std::size_t row_count)
        {
            const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= row_count)
                return;
            gpu::GpuFr acc = gpu::fr_zero();
            for (std::size_t c = 0; c < num_chunks; ++c)
            {
                acc = gpu::fr_add(acc, partial_results[c * row_count + row]);
            }
            final_results[row] = acc;
        }

        __global__ void accumulate_results_kernel(
            const gpu::GpuFr *__restrict__ chunk_results,
            gpu::GpuFr *__restrict__ final_results,
            std::size_t row_count)
        {
            const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= row_count)
                return;
            final_results[row] = gpu::fr_add(final_results[row], chunk_results[row]);
        }

    } // namespace

    // 主机侧包装：以 `mcl::Fr` 兼容布局触发批量加权求值。
    bool weighted_poly_eval_gpu(
        const uint64_t *weight_limbs,
        const uint64_t *interleaved_value_limbs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t row_count)
    {

        if (domain_size == 0 || row_count == 0)
            return true;

        WeightedEvalChunkPlan chunk_plan;
        if (!make_weighted_eval_chunk_plan(
                domain_size,
                row_count,
                &chunk_plan))
        {
            return false;
        }
        if (row_count > 65535)
            return false;
        if (weight_limbs == nullptr ||
            interleaved_value_limbs == nullptr ||
            output_limbs == nullptr)
        {
            return false;
        }

        const auto weight_bytes = chunk_plan.weight_bytes;
        const auto result_bytes = chunk_plan.result_bytes;
        const auto cols_per_chunk = chunk_plan.cols_per_chunk;

        const bool zero_copy = cuda_backend_unified_addressing_available();

        gpu::GpuFr *d_weights = nullptr;
        gpu::GpuFr *d_values = nullptr;
        gpu::GpuFr *d_final = nullptr;

        const auto cleanup = [&]()
        {
            if (d_values != nullptr)
                cudaFree(d_values);
            if (d_weights != nullptr)
                cudaFree(d_weights);
            if (d_final != nullptr)
                cudaFree(d_final);
        };

        if (!zero_copy && cudaMalloc(&d_weights, weight_bytes) != cudaSuccess)
            return false;
        if (cudaMalloc(&d_final, result_bytes) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        const auto *weight_ptr =
            zero_copy ? reinterpret_cast<const gpu::GpuFr *>(weight_limbs) : nullptr;
        if (!zero_copy)
        {
            if (cudaMemcpy(d_weights, weight_limbs, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            weight_ptr = d_weights;
        }
        if (cudaMemset(d_final, 0, result_bytes) != cudaSuccess)
        {
            cleanup();
            return false;
        }
        std::size_t kernel_launches = 0;

        // 分块处理域元素，统一管理设备内存占用。
        for (std::size_t chunk_start = 0;
             chunk_start < domain_size;)
        {
            const auto chunk_cols =
                std::min(domain_size - chunk_start, cols_per_chunk);
            if (chunk_cols == 0)
            {
                cleanup();
                return false;
            }
            const auto chunk_end = chunk_start + chunk_cols;

            std::size_t chunk_value_elements = 0;
            std::size_t chunk_value_bytes = 0;
            std::size_t chunk_offset_elements = 0;
            if (!checked_size_mul(
                    chunk_cols,
                    row_count,
                    &chunk_value_elements) ||
                !checked_size_mul(
                    chunk_value_elements,
                    sizeof(gpu::GpuFr),
                    &chunk_value_bytes) ||
                chunk_value_bytes > kMaxGpuValueBytes ||
                !checked_size_mul(
                    chunk_start,
                    row_count,
                    &chunk_offset_elements))
            {
                cleanup();
                return false;
            }

            const auto *chunk_values =
                reinterpret_cast<const gpu::GpuFr *>(interleaved_value_limbs) +
                chunk_offset_elements;
            if (!zero_copy)
            {
                if (d_values != nullptr)
                {
                    cudaFree(d_values);
                    d_values = nullptr;
                }
                if (cudaMalloc(&d_values, chunk_value_bytes) != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                if (cudaMemcpy(d_values, chunk_values, chunk_value_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                chunk_values = d_values;
            }

            constexpr unsigned kBlockDim = 128;
            std::size_t num_domain_blocks_size = 0;
            if (!checked_ceil_div(
                    chunk_cols,
                    kBlockDim,
                    &num_domain_blocks_size) ||
                num_domain_blocks_size >
                    std::numeric_limits<unsigned>::max())
            {
                cudaFree(d_values);
                d_values = nullptr;
                cleanup();
                return false;
            }
            const auto num_domain_blocks =
                static_cast<unsigned>(num_domain_blocks_size);

            // 为当前 chunk 准备局部结果缓冲区。
            gpu::GpuFr *d_partials = nullptr;
            std::size_t partial_elements = 0;
            std::size_t partial_bytes = 0;
            if (!checked_size_mul(
                    num_domain_blocks_size,
                    row_count,
                    &partial_elements) ||
                !checked_size_mul(
                    partial_elements,
                    sizeof(gpu::GpuFr),
                    &partial_bytes))
            {
                cudaFree(d_values);
                d_values = nullptr;
                cleanup();
                return false;
            }
            if (cudaMalloc(&d_partials, partial_bytes) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            if (cudaMemset(d_partials, 0, partial_bytes) != cudaSuccess)
            {
                cudaFree(d_partials);
                cleanup();
                return false;
            }

            const auto grid_y = static_cast<unsigned>(row_count);
            dim3 grid(num_domain_blocks, grid_y);
            const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);

            // kernel 读取的是 chunk 相对索引，因为本轮上传的数据从 `d_values[0]` 开始。
            weighted_eval_kernel<<<grid, kBlockDim, smem_bytes>>>(
                weight_ptr, chunk_values, d_partials,
                domain_size, row_count, chunk_start, chunk_end);
            ++kernel_launches;

            // 把当前 chunk 的局部结果并入全局输出。
            {
                constexpr unsigned rBlock = 256;
                std::size_t r_grid_size = 0;
                if (!checked_ceil_div(
                        row_count,
                        rBlock,
                        &r_grid_size) ||
                    r_grid_size > std::numeric_limits<unsigned>::max())
                {
                    cudaFree(d_partials);
                    cleanup();
                    return false;
                }
                const auto rGrid = static_cast<unsigned>(r_grid_size);
                // 先把本 chunk 内的局部结果折叠成单列结果。
                gpu::GpuFr *d_chunk_result = nullptr;
                if (cudaMalloc(&d_chunk_result, result_bytes) != cudaSuccess)
                {
                    cudaFree(d_partials);
                    cleanup();
                    return false;
                }
                reduce_partials_kernel<<<rGrid, rBlock>>>(d_partials, d_chunk_result, num_domain_blocks, row_count);
                accumulate_results_kernel<<<rGrid, rBlock>>>(d_chunk_result, d_final, row_count);
                kernel_launches += 2;
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    cudaFree(d_chunk_result);
                    cudaFree(d_partials);
                    cleanup();
                    return false;
                }
                cudaFree(d_chunk_result);
            }
            cudaFree(d_partials);
            chunk_start = chunk_end;
        }

        // 下载最终结果。
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            cleanup();
            return false;
        }
        if (cudaMemcpy(output_limbs, d_final, result_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        cleanup();
        record_cuda_kernel_invocations(kernel_launches);
        return true;
    }

    bool weighted_poly_eval_gpu_columnar(
        const uint64_t *weight_limbs,
        const uint64_t *const *column_ptrs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t row_count,
        const std::size_t *column_lengths,
        std::size_t weight_rotation)
    {
        if (domain_size == 0 ||
            row_count == 0)
        {
            return true;
        }

        if (weight_limbs == nullptr ||
            column_ptrs == nullptr ||
            output_limbs == nullptr ||
            row_count > 65535)
        {
            return false;
        }

        constexpr unsigned
            kBlockDim = 256;

        const auto num_blocks =
            static_cast<std::size_t>(
                (domain_size +
                 kBlockDim - 1) /
                kBlockDim);

        if (num_blocks >
            std::numeric_limits<
                unsigned>::max())
        {
            return false;
        }

        if (row_count >
            std::numeric_limits<
                std::size_t>::max() /
                num_blocks)
        {
            return false;
        }

        const auto partial_count =
            num_blocks *
            row_count;

        struct ColumnarWorkspace
        {
            cudaStream_t stream = nullptr;
            const gpu::GpuFr **columns = nullptr;
            std::size_t *lengths = nullptr;
            gpu::GpuFr *partials = nullptr;
            gpu::GpuFr *final_values = nullptr;
            std::size_t column_capacity = 0;
            std::size_t length_capacity = 0;
            std::size_t partial_capacity = 0;
            std::size_t final_capacity = 0;

            bool has_allocations() const
            {
                return columns != nullptr ||
                       lengths != nullptr ||
                       partials != nullptr ||
                       final_values != nullptr;
            }

            bool ensure_stream()
            {
                if (stream != nullptr)
                    return true;

                const auto status =
                    cudaStreamCreateWithFlags(
                        &stream,
                        cudaStreamNonBlocking);

                if (status != cudaSuccess)
                {
                    stream = nullptr;
                    return false;
                }
                return true;
            }

            bool synchronize_for_growth()
            {
                if (!has_allocations())
                    return true;

                const auto status =
                    cudaStreamSynchronize(stream);

                if (status != cudaSuccess)
                    return false;

                record_cuda_stream_sync();
                return true;
            }

            bool ensure(
                std::size_t rows,
                std::size_t partials_required,
                bool need_lengths)
            {
                if (!ensure_stream())
                    return false;

                const bool grow =
                    column_capacity < rows ||
                    partial_capacity < partials_required ||
                    final_capacity < rows ||
                    (need_lengths &&
                     length_capacity < rows);

                if (!grow)
                    return true;

                if (!synchronize_for_growth())
                    return false;

                if (column_capacity < rows)
                {
                    if (columns != nullptr)
                    {
                        (void)cudaFree(columns);
                        columns = nullptr;
                        column_capacity = 0;
                    }

                    if (cudaMalloc(
                            reinterpret_cast<void **>(
                                &columns),
                            rows *
                                sizeof(
                                    const gpu::GpuFr *)) !=
                        cudaSuccess)
                    {
                        return false;
                    }

                    column_capacity = rows;
                }

                if (need_lengths &&
                    length_capacity < rows)
                {
                    if (lengths != nullptr)
                    {
                        (void)cudaFree(lengths);
                        lengths = nullptr;
                        length_capacity = 0;
                    }

                    if (cudaMalloc(
                            reinterpret_cast<void **>(
                                &lengths),
                            rows *
                                sizeof(
                                    std::size_t)) !=
                        cudaSuccess)
                    {
                        return false;
                    }

                    length_capacity = rows;
                }

                if (partial_capacity <
                    partials_required)
                {
                    if (partials != nullptr)
                    {
                        (void)cudaFree(partials);
                        partials = nullptr;
                        partial_capacity = 0;
                    }

                    if (cudaMalloc(
                            reinterpret_cast<void **>(
                                &partials),
                            partials_required *
                                sizeof(
                                    gpu::GpuFr)) !=
                        cudaSuccess)
                    {
                        return false;
                    }

                    partial_capacity =
                        partials_required;
                }

                if (final_capacity < rows)
                {
                    if (final_values != nullptr)
                    {
                        (void)cudaFree(final_values);
                        final_values = nullptr;
                        final_capacity = 0;
                    }

                    if (cudaMalloc(
                            reinterpret_cast<void **>(
                                &final_values),
                            rows *
                                sizeof(
                                    gpu::GpuFr)) !=
                        cudaSuccess)
                    {
                        return false;
                    }

                    final_capacity = rows;
                }

                return true;
            }

            ~ColumnarWorkspace()
            {
                if (stream != nullptr)
                    (void)cudaStreamSynchronize(stream);
                if (columns != nullptr)
                    (void)cudaFree(columns);
                if (lengths != nullptr)
                    (void)cudaFree(lengths);
                if (partials != nullptr)
                    (void)cudaFree(partials);
                if (final_values != nullptr)
                    (void)cudaFree(final_values);
                if (stream != nullptr)
                    (void)cudaStreamDestroy(stream);
            }
        };

        thread_local ColumnarWorkspace
            workspace;

        if (!workspace.ensure(
                row_count,
                partial_count,
                column_lengths != nullptr))
        {
            return false;
        }

        std::vector<
            const gpu::GpuFr *>
            host_columns(
                row_count);

        for (std::size_t index = 0;
             index < row_count;
             ++index)
        {
            if (column_ptrs[index] == nullptr)
                return false;

            host_columns[index] =
                reinterpret_cast<
                    const gpu::GpuFr *>(
                    column_ptrs[index]);
        }

        const auto pointer_bytes =
            row_count *
            sizeof(
                const gpu::GpuFr *);

        auto status =
            cudaMemcpyAsync(
                workspace.columns,
                host_columns.data(),
                pointer_bytes,
                cudaMemcpyHostToDevice,
                workspace.stream);

        if (status != cudaSuccess)
            return false;

        record_cuda_h2d(pointer_bytes);

        const std::size_t *device_lengths =
            nullptr;

        if (column_lengths != nullptr)
        {
            const auto length_bytes =
                row_count *
                sizeof(std::size_t);

            status =
                cudaMemcpyAsync(
                    workspace.lengths,
                    column_lengths,
                    length_bytes,
                    cudaMemcpyHostToDevice,
                    workspace.stream);

            if (status != cudaSuccess)
                return false;

            record_cuda_h2d(length_bytes);
            device_lengths = workspace.lengths;
        }

        const auto partial_bytes =
            partial_count *
            sizeof(gpu::GpuFr);

        status =
            cudaMemsetAsync(
                workspace.partials,
                0,
                partial_bytes,
                workspace.stream);

        if (status != cudaSuccess)
            return false;

        dim3 grid(
            static_cast<unsigned>(
                num_blocks),
            static_cast<unsigned>(
                row_count));

        const auto shared_bytes =
            kBlockDim *
            sizeof(gpu::GpuFr);

        weighted_eval_columnar_kernel<<<
            grid,
            kBlockDim,
            shared_bytes,
            workspace.stream>>>(
            reinterpret_cast<
                const gpu::GpuFr *>(
                weight_limbs),
            workspace.columns,
            device_lengths,
            workspace.partials,
            domain_size,
            row_count,
            weight_rotation %
                domain_size);

        status = cudaGetLastError();

        if (status != cudaSuccess)
            return false;

        const auto reduce_grid =
            static_cast<unsigned>(
                (row_count +
                 kBlockDim - 1) /
                kBlockDim);

        reduce_partials_kernel<<<
            reduce_grid,
            kBlockDim,
            0,
            workspace.stream>>>(
            workspace.partials,
            workspace.final_values,
            static_cast<unsigned>(
                num_blocks),
            row_count);

        status = cudaGetLastError();

        if (status != cudaSuccess)
            return false;

        const auto result_bytes =
            row_count *
            sizeof(gpu::GpuFr);

        status =
            cudaMemcpyAsync(
                output_limbs,
                workspace.final_values,
                result_bytes,
                cudaMemcpyDeviceToHost,
                workspace.stream);

        if (status != cudaSuccess)
            return false;

        record_cuda_d2h(result_bytes);

        status =
            cudaStreamSynchronize(
                workspace.stream);

        if (status != cudaSuccess)
            return false;

        record_cuda_stream_sync();
        record_cuda_kernel_invocations(2);
        return true;
    }
    // ---- 共享权重的多列求值入口 ----
    // 统一地址空间设备走单次列式 kernel，离散显卡则复用权重并顺序处理每一列。
    bool weighted_poly_eval_gpu_multi(
        const uint64_t *weight_limbs,
        const uint64_t *const *column_value_ptrs,
        uint64_t *output_limbs, // `poly_count × 4 uint64`
        std::size_t domain_size,
        std::size_t poly_count,
        const std::size_t *column_lengths,
        std::size_t weight_rotation)
    {

        if (domain_size == 0 || poly_count == 0)
            return true;

        if (weight_limbs == nullptr || column_value_ptrs == nullptr ||
            output_limbs == nullptr)
        {
            return false;
        }

        // 统一地址空间可用时，优先走零拷贝列式 kernel，一次处理全部多项式。
        if (cuda_backend_unified_addressing_available())
        {
            return weighted_poly_eval_gpu_columnar(
                weight_limbs, column_value_ptrs, output_limbs,
                domain_size, poly_count, column_lengths, weight_rotation);
        }

        // ---- Discrete GPU fallback: sequential per-polynomial with weight reuse ----
        std::size_t weight_bytes = 0;
        std::size_t single_val_bytes = 0;
        std::size_t num_blocks_size = 0;
        std::size_t partial_bytes = 0;
        if (!checked_size_mul(
                domain_size,
                sizeof(gpu::GpuFr),
                &weight_bytes) ||
            !checked_size_mul(
                domain_size,
                sizeof(gpu::GpuFr),
                &single_val_bytes) ||
            weight_bytes > kMaxGpuValueBytes ||
            single_val_bytes > kMaxGpuValueBytes ||
            !checked_ceil_div(
                domain_size,
                128,
                &num_blocks_size) ||
            num_blocks_size > std::numeric_limits<unsigned>::max() ||
            !checked_size_mul(
                num_blocks_size,
                sizeof(gpu::GpuFr),
                &partial_bytes) ||
            partial_bytes == 0 ||
            partial_bytes > kMaxRotationPartialBytes)
        {
            return false;
        }
        std::size_t working_set_bytes = 0;
        constexpr std::size_t kAdmissionHeadroomBytes =
            64ULL * 1024ULL * 1024ULL;
        if (!checked_size_add(
                weight_bytes,
                single_val_bytes,
                &working_set_bytes) ||
            !checked_size_add(
                working_set_bytes,
                partial_bytes,
                &working_set_bytes) ||
            !checked_size_add(
                working_set_bytes,
                sizeof(gpu::GpuFr),
                &working_set_bytes) ||
            !checked_size_add(
                working_set_bytes,
                kAdmissionHeadroomBytes,
                &working_set_bytes))
        {
            return false;
        }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
            free_bytes < working_set_bytes)
        {
            return false;
        }
        (void)total_bytes;
        const auto result_bytes_single = sizeof(gpu::GpuFr);

        gpu::GpuFr *d_weights = nullptr;
        if (cudaMalloc(&d_weights, weight_bytes) != cudaSuccess)
            return false;
        if (cudaMemcpy(d_weights, weight_limbs, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(d_weights);
            return false;
        }
        record_cuda_h2d(weight_bytes);

        gpu::GpuFr *d_values = nullptr;
        if (cudaMalloc(&d_values, single_val_bytes) != cudaSuccess)
        {
            cudaFree(d_weights);
            return false;
        }

        constexpr unsigned kBlockDim = 128;
        const auto num_blocks = static_cast<unsigned>(num_blocks_size);

        gpu::GpuFr *d_partials = nullptr;
        if (cudaMalloc(&d_partials, partial_bytes) != cudaSuccess)
        {
            cudaFree(d_values);
            cudaFree(d_weights);
            return false;
        }

        gpu::GpuFr *d_result = nullptr;
        if (cudaMalloc(&d_result, result_bytes_single) != cudaSuccess)
        {
            cudaFree(d_partials);
            cudaFree(d_values);
            cudaFree(d_weights);
            return false;
        }

        const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);

        for (std::size_t p = 0; p < poly_count; ++p)
        {
            const auto column_size = column_lengths == nullptr
                                         ? domain_size
                                         : std::min(domain_size, column_lengths[p]);
            if (column_size > 0 && column_value_ptrs[p] == nullptr)
            {
                cudaFree(d_result);
                cudaFree(d_partials);
                cudaFree(d_values);
                cudaFree(d_weights);
                return false;
            }
            if (column_size < domain_size)
            {
                if (cudaMemset(d_values, 0, single_val_bytes) != cudaSuccess)
                {
                    cudaFree(d_result);
                    cudaFree(d_partials);
                    cudaFree(d_values);
                    cudaFree(d_weights);
                    return false;
                }
            }
            if (column_size > 0 &&
                cudaMemcpy(
                    d_values,
                    column_value_ptrs[p],
                    column_size * sizeof(gpu::GpuFr),
                    cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cudaFree(d_result);
                cudaFree(d_partials);
                cudaFree(d_values);
                cudaFree(d_weights);
                return false;
            }
            if (column_size > 0)
            {
                record_cuda_h2d(column_size * sizeof(gpu::GpuFr));
            }
            if (cudaMemset(d_partials, 0, partial_bytes) != cudaSuccess)
            {
                cudaFree(d_result);
                cudaFree(d_partials);
                cudaFree(d_values);
                cudaFree(d_weights);
                return false;
            }

            dim3 grid(num_blocks, 1);
            weighted_eval_kernel<<<grid, kBlockDim, smem_bytes>>>(
                d_weights, d_values, d_partials,
                domain_size, 1, 0, domain_size);

            constexpr unsigned rBlock = 256;
            reduce_partials_kernel<<<1, rBlock>>>(d_partials, d_result, num_blocks, 1);

            auto *output_ptr = reinterpret_cast<gpu::GpuFr *>(output_limbs) + p;
            if (cudaMemcpy(output_ptr, d_result, result_bytes_single, cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                cudaFree(d_result);
                cudaFree(d_partials);
                cudaFree(d_values);
                cudaFree(d_weights);
                return false;
            }
            record_cuda_d2h(result_bytes_single);
        }

        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            cudaFree(d_result);
            cudaFree(d_partials);
            cudaFree(d_values);
            cudaFree(d_weights);
            return false;
        }
        record_cuda_stream_sync();

        cudaFree(d_result);
        cudaFree(d_partials);
        cudaFree(d_values);
        cudaFree(d_weights);
        record_cuda_kernel_invocations(poly_count * 2);
        return true;
    }

    bool weighted_poly_eval_gpu_multi_rotations(
        const uint64_t *weight_limbs,
        const uint64_t *const *column_value_ptrs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t poly_count,
        const std::size_t *rotations,
        std::size_t rotation_count,
        const std::size_t *column_lengths)
    {
        if (domain_size == 0 || poly_count == 0 || rotation_count == 0)
            return true;
        if (poly_count > 65535 || rotation_count > 65535 ||
            weight_limbs == nullptr || column_value_ptrs == nullptr ||
            output_limbs == nullptr || rotations == nullptr)
            return false;

        std::size_t common_domain_bytes = 0;
        std::size_t common_num_blocks = 0;
        std::size_t common_partial_elements = 0;
        std::size_t common_partial_bytes = 0;
        std::size_t common_output_count = 0;
        std::size_t common_output_bytes = 0;
        if (!checked_size_mul(
                domain_size,
                sizeof(gpu::GpuFr),
                &common_domain_bytes) ||
            common_domain_bytes > kMaxGpuValueBytes ||
            !checked_ceil_div(
                domain_size,
                256,
                &common_num_blocks) ||
            common_num_blocks > std::numeric_limits<unsigned>::max() ||
            !checked_size_mul(
                common_num_blocks,
                poly_count,
                &common_partial_elements) ||
            !checked_size_mul(
                common_partial_elements,
                rotation_count,
                &common_partial_elements) ||
            !checked_size_mul(
                common_partial_elements,
                sizeof(gpu::GpuFr),
                &common_partial_bytes) ||
            common_partial_bytes == 0 ||
            common_partial_bytes > kMaxRotationPartialBytes ||
            !checked_size_mul(
                poly_count,
                rotation_count,
                &common_output_count) ||
            !checked_size_mul(
                common_output_count,
                sizeof(gpu::GpuFr),
                &common_output_bytes) ||
            common_output_bytes == 0 ||
            common_output_bytes > kMaxGpuValueBytes)
        {
            return false;
        }

        // Discrete devices cannot dereference the host column pointers.  Keep
        // one device column buffer and accumulate every column into the
        // rotation-major output, so the batch size is bounded by the same
        // partial-buffer cap as the unified path without multiplying device
        // column storage by the number of rotations.
        if (!cuda_backend_unified_addressing_available())
        {
            constexpr unsigned kBlockDim = 256;
            constexpr unsigned kReduceBlockDim = 256;
            std::size_t weight_bytes = 0;
            std::size_t value_bytes = 0;
            std::size_t num_blocks_size = 0;
            std::size_t partial_elements = 0;
            std::size_t partial_bytes = 0;
            std::size_t output_count = 0;
            std::size_t output_bytes = 0;
            std::size_t chunk_result_bytes = 0;
            std::size_t rotation_bytes = 0;
            if (!checked_size_mul(
                    domain_size,
                    sizeof(gpu::GpuFr),
                    &weight_bytes) ||
                !checked_size_mul(
                    domain_size,
                    sizeof(gpu::GpuFr),
                    &value_bytes) ||
                weight_bytes > kMaxGpuValueBytes ||
                value_bytes > kMaxGpuValueBytes ||
                !checked_ceil_div(
                    domain_size,
                    kBlockDim,
                    &num_blocks_size) ||
                num_blocks_size > std::numeric_limits<unsigned>::max() ||
                !checked_size_mul(
                    num_blocks_size,
                    rotation_count,
                    &partial_elements) ||
                !checked_size_mul(
                    partial_elements,
                    sizeof(gpu::GpuFr),
                    &partial_bytes) ||
                partial_bytes == 0 ||
                partial_bytes > kMaxRotationPartialBytes ||
                !checked_size_mul(
                    poly_count,
                    rotation_count,
                    &output_count) ||
                !checked_size_mul(
                    output_count,
                    sizeof(gpu::GpuFr),
                    &output_bytes) ||
                output_bytes == 0 ||
                output_bytes > kMaxGpuValueBytes ||
                !checked_size_mul(
                    rotation_count,
                    sizeof(gpu::GpuFr),
                    &chunk_result_bytes) ||
                !checked_size_mul(
                    rotation_count,
                    sizeof(std::size_t),
                    &rotation_bytes))
            {
                return false;
            }

            std::size_t working_set_bytes = 0;
            constexpr std::size_t kAdmissionHeadroomBytes =
                64ULL * 1024ULL * 1024ULL;
            if (!checked_size_add(
                    weight_bytes,
                    value_bytes,
                    &working_set_bytes) ||
                !checked_size_add(
                    working_set_bytes,
                    partial_bytes,
                    &working_set_bytes) ||
                !checked_size_add(
                    working_set_bytes,
                    chunk_result_bytes,
                    &working_set_bytes) ||
                !checked_size_add(
                    working_set_bytes,
                    output_bytes,
                    &working_set_bytes) ||
                !checked_size_add(
                    working_set_bytes,
                    sizeof(const gpu::GpuFr *),
                    &working_set_bytes) ||
                !checked_size_add(
                    working_set_bytes,
                    sizeof(std::size_t),
                    &working_set_bytes) ||
                !checked_size_add(
                    working_set_bytes,
                    rotation_bytes,
                    &working_set_bytes))
            {
                return false;
            }
            std::size_t required_bytes = 0;
            if (!checked_size_add(
                    working_set_bytes,
                    kAdmissionHeadroomBytes,
                    &required_bytes))
            {
                return false;
            }
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
                free_bytes < required_bytes)
            {
                return false;
            }
            (void)total_bytes;

            std::size_t reduce_grid_size = 0;
            if (!checked_ceil_div(
                    rotation_count,
                    kReduceBlockDim,
                    &reduce_grid_size) ||
                reduce_grid_size > std::numeric_limits<unsigned>::max())
            {
                return false;
            }

            gpu::GpuFr *d_weights = nullptr;
            gpu::GpuFr *d_values = nullptr;
            gpu::GpuFr *d_partials = nullptr;
            gpu::GpuFr *d_chunk_result = nullptr;
            gpu::GpuFr *d_final = nullptr;
            const gpu::GpuFr **d_columns = nullptr;
            std::size_t *d_column_lengths = nullptr;
            std::size_t *d_rotations = nullptr;
            const auto cleanup = [&]()
            {
                cudaFree(d_final);
                cudaFree(d_chunk_result);
                cudaFree(d_partials);
                cudaFree(d_rotations);
                cudaFree(d_column_lengths);
                cudaFree(d_columns);
                cudaFree(d_values);
                cudaFree(d_weights);
            };

            if (cudaMalloc(&d_weights, weight_bytes) != cudaSuccess ||
                cudaMalloc(&d_values, value_bytes) != cudaSuccess ||
                cudaMalloc(&d_partials, partial_bytes) != cudaSuccess ||
                cudaMalloc(
                    &d_chunk_result,
                    chunk_result_bytes) != cudaSuccess ||
                cudaMalloc(&d_final, output_bytes) != cudaSuccess ||
                cudaMalloc(&d_columns, sizeof(const gpu::GpuFr *)) != cudaSuccess ||
                cudaMalloc(&d_column_lengths, sizeof(std::size_t)) != cudaSuccess ||
                cudaMalloc(
                    &d_rotations,
                    rotation_bytes) != cudaSuccess)
            {
                cleanup();
                return false;
            }

            if (cudaMemcpy(
                    d_weights,
                    weight_limbs,
                    weight_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemset(d_final, 0, output_bytes) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            record_cuda_h2d(weight_bytes);

            std::vector<std::size_t> normalized_rotations(rotation_count);
            for (std::size_t index = 0; index < rotation_count; ++index)
            {
                normalized_rotations[index] = rotations[index] % domain_size;
            }
            if (cudaMemcpy(
                    d_rotations,
                    normalized_rotations.data(),
                    rotation_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            record_cuda_h2d(rotation_bytes);

            const auto device_column = d_values;
            if (cudaMemcpy(
                    d_columns,
                    &device_column,
                    sizeof(device_column),
                    cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            record_cuda_h2d(sizeof(device_column));

            const auto num_blocks = static_cast<unsigned>(num_blocks_size);
            const auto reduce_grid = static_cast<unsigned>(reduce_grid_size);
            const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);
            std::size_t kernel_launches = 0;
            for (std::size_t poly_index = 0;
                 poly_index < poly_count;
                 ++poly_index)
            {
                const auto column_size = column_lengths == nullptr
                                             ? domain_size
                                             : std::min(
                                                   domain_size,
                                                   column_lengths[poly_index]);
                if (column_size > 0 &&
                    column_value_ptrs[poly_index] == nullptr)
                {
                    cleanup();
                    return false;
                }

                if (column_size < domain_size &&
                    cudaMemset(d_values, 0, value_bytes) != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                if (column_size > 0)
                {
                    std::size_t column_bytes = 0;
                    if (!checked_size_mul(
                            column_size,
                            sizeof(gpu::GpuFr),
                            &column_bytes) ||
                        cudaMemcpy(
                            d_values,
                            column_value_ptrs[poly_index],
                            column_bytes,
                            cudaMemcpyHostToDevice) != cudaSuccess)
                    {
                        cleanup();
                        return false;
                    }
                    record_cuda_h2d(column_bytes);
                }

                if (cudaMemcpy(
                        d_column_lengths,
                        &column_size,
                        sizeof(column_size),
                        cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemset(d_partials, 0, partial_bytes) != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                record_cuda_h2d(sizeof(column_size));

                dim3 grid(
                    num_blocks,
                    1,
                    static_cast<unsigned>(rotation_count));
                weighted_eval_columnar_rotations_kernel<<<
                    grid,
                    kBlockDim,
                    smem_bytes>>>(
                    d_weights,
                    d_columns,
                    d_column_lengths,
                    d_rotations,
                    d_partials,
                    domain_size,
                    1,
                    num_blocks);
                if (cudaGetLastError() != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                reduce_rotation_partials_kernel<<<
                    reduce_grid,
                    kReduceBlockDim>>>(
                    d_partials,
                    d_chunk_result,
                    num_blocks,
                    1,
                    rotation_count);
                if (cudaGetLastError() != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                accumulate_rotation_results_kernel<<<
                    reduce_grid,
                    kReduceBlockDim>>>(
                    d_chunk_result,
                    d_final,
                    poly_count,
                    poly_index,
                    rotation_count);
                if (cudaGetLastError() != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                kernel_launches += 3;
            }

            if (cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(
                    output_limbs,
                    d_final,
                    output_bytes,
                    cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                cleanup();
                return false;
            }
            record_cuda_stream_sync();
            record_cuda_d2h(output_bytes);
            cleanup();
            record_cuda_kernel_invocations(kernel_launches);
            return true;
        }

        constexpr unsigned kBlockDim = 256;
        const auto num_blocks = static_cast<unsigned>(common_num_blocks);
        const auto grid_y = static_cast<unsigned>(
            std::min(poly_count, static_cast<std::size_t>(65535)));

        const gpu::GpuFr **d_columns = nullptr;
        const auto ptr_bytes = poly_count * sizeof(const gpu::GpuFr *);
        if (cudaMalloc(&d_columns, ptr_bytes) != cudaSuccess)
            return false;
        auto cleanup_columns = [&]()
        {
            cudaFree(d_columns);
        };
        std::vector<const gpu::GpuFr *> h_columns(poly_count);
        for (std::size_t index = 0; index < poly_count; ++index)
        {
            h_columns[index] = reinterpret_cast<const gpu::GpuFr *>(
                column_value_ptrs[index]);
        }
        if (cudaMemcpy(
                d_columns,
                h_columns.data(),
                ptr_bytes,
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cleanup_columns();
            return false;
        }
        record_cuda_h2d(ptr_bytes);

        std::size_t *d_column_lengths = nullptr;
        if (column_lengths != nullptr)
        {
            const auto length_bytes = poly_count * sizeof(std::size_t);
            if (cudaMalloc(&d_column_lengths, length_bytes) != cudaSuccess || cudaMemcpy(
                                                                                  d_column_lengths,
                                                                                  column_lengths,
                                                                                  length_bytes,
                                                                                  cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cudaFree(d_column_lengths);
                cleanup_columns();
                return false;
            }
            record_cuda_h2d(length_bytes);
        }

        std::vector<std::size_t> normalized_rotations(rotation_count);
        for (std::size_t index = 0; index < rotation_count; ++index)
        {
            normalized_rotations[index] = rotations[index] % domain_size;
        }
        std::size_t *d_rotations = nullptr;
        const auto rotation_bytes = rotation_count * sizeof(std::size_t);
        if (cudaMalloc(&d_rotations, rotation_bytes) != cudaSuccess || cudaMemcpy(
                                                                           d_rotations,
                                                                           normalized_rotations.data(),
                                                                           rotation_bytes,
                                                                           cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(d_rotations);
            cudaFree(d_column_lengths);
            cleanup_columns();
            return false;
        }
        record_cuda_h2d(rotation_bytes);

        const auto partial_count = common_partial_elements;
        gpu::GpuFr *d_partials = nullptr;
        if (cudaMalloc(
                &d_partials,
                partial_count * sizeof(gpu::GpuFr)) != cudaSuccess)
        {
            cudaFree(d_rotations);
            cudaFree(d_column_lengths);
            cleanup_columns();
            return false;
        }
        const auto output_count = common_output_count;
        gpu::GpuFr *d_final = nullptr;
        if (cudaMalloc(
                &d_final,
                output_count * sizeof(gpu::GpuFr)) != cudaSuccess)
        {
            cudaFree(d_partials);
            cudaFree(d_rotations);
            cudaFree(d_column_lengths);
            cleanup_columns();
            return false;
        }

        dim3 grid(num_blocks, grid_y, static_cast<unsigned>(rotation_count));
        const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);
        weighted_eval_columnar_rotations_kernel<<<
            grid, kBlockDim, smem_bytes>>>(
            reinterpret_cast<const gpu::GpuFr *>(weight_limbs),
            d_columns,
            d_column_lengths,
            d_rotations,
            d_partials,
            domain_size,
            poly_count,
            num_blocks);
        constexpr unsigned kReduceBlockDim = 256;
        const auto reduce_grid = static_cast<unsigned>(
            (output_count + kReduceBlockDim - 1) / kReduceBlockDim);
        reduce_rotation_partials_kernel<<<
            reduce_grid, kReduceBlockDim>>>(
            d_partials,
            d_final,
            num_blocks,
            poly_count,
            rotation_count);

        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            cudaFree(d_final);
            cudaFree(d_partials);
            cudaFree(d_rotations);
            cudaFree(d_column_lengths);
            cleanup_columns();
            return false;
        }
        record_cuda_stream_sync();
        const auto output_bytes = common_output_bytes;
        if (cudaMemcpy(
                output_limbs,
                d_final,
                output_bytes,
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(d_final);
            cudaFree(d_partials);
            cudaFree(d_rotations);
            cudaFree(d_column_lengths);
            cleanup_columns();
            return false;
        }
        record_cuda_d2h(output_bytes);
        cudaFree(d_final);
        cudaFree(d_partials);
        cudaFree(d_rotations);
        cudaFree(d_column_lengths);
        cleanup_columns();
        record_cuda_kernel_invocations(2);
        return true;
    }

    bool cuda_interleaved_multi_rotation_admissible(
        std::size_t domain_size,
        std::size_t poly_count,
        std::size_t rotation_count,
        std::size_t additional_host_bytes)
    {
        InterleavedMultiRotationWorkspace workspace;
        if (!make_interleaved_multi_rotation_workspace(
                domain_size,
                poly_count,
                rotation_count,
                &workspace))
        {
            return false;
        }

        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }
        int integrated = 0;
        if (cudaDeviceGetAttribute(
                &integrated,
                cudaDevAttrIntegrated,
                device) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        const auto host = util::cached_effective_memory_snapshot();
        if (!host.effective_available_known)
        {
            return false;
        }

        constexpr std::size_t kHostSafetyReserveBytes =
            8ULL << 30;
        constexpr std::size_t kIntegratedHostSafetyReserveBytes =
            16ULL << 30;
        const auto host_reserve = std::max(
            integrated != 0
                ? kIntegratedHostSafetyReserveBytes
                : kHostSafetyReserveBytes,
            host.effective_total_known
                ? host.effective_total_bytes / 8
                : std::size_t{0});

        std::size_t required_host = additional_host_bytes;
        if (integrated != 0 &&
            !checked_size_add(
                required_host,
                workspace.device_peak_bytes,
                &required_host))
        {
            return false;
        }
        if (!checked_size_add(
                required_host,
                host_reserve,
                &required_host) ||
            host.effective_available_bytes < required_host)
        {
            return false;
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }
        const auto device_reserve = std::max<std::size_t>(
            64ULL * 1024ULL * 1024ULL,
            total_bytes / 8);
        std::size_t required_device = workspace.device_peak_bytes;
        if (!checked_size_add(
                required_device,
                device_reserve,
                &required_device))
        {
            return false;
        }
        return free_bytes >= required_device;
    }

    bool weighted_poly_eval_gpu_interleaved_multi_rotations(
        const uint64_t *weight_limbs,
        const uint64_t *interleaved_value_limbs,
        uint64_t *output_limbs,
        std::size_t domain_size,
        std::size_t poly_count,
        const std::size_t *rotations,
        std::size_t rotation_count)
    {
        if (domain_size == 0 || poly_count == 0 || rotation_count == 0)
        {
            return true;
        }
        if (weight_limbs == nullptr ||
            interleaved_value_limbs == nullptr ||
            output_limbs == nullptr ||
            rotations == nullptr ||
            poly_count > 65535 ||
            rotation_count > 65535)
        {
            return false;
        }

        InterleavedMultiRotationWorkspace workspace;
        if (!make_interleaved_multi_rotation_workspace(
                domain_size,
                poly_count,
                rotation_count,
                &workspace) ||
            !cuda_interleaved_multi_rotation_admissible(
                domain_size,
                poly_count,
                rotation_count,
                0))
        {
            return false;
        }

        const auto field_bytes = workspace.field_bytes;
        const auto interleaved_bytes = workspace.subset_bytes;
        const auto ptr_bytes = workspace.pointer_bytes;

        gpu::GpuFr *d_weights = nullptr;
        gpu::GpuFr *d_interleaved = nullptr;
        gpu::GpuFr *d_columnar = nullptr;
        const gpu::GpuFr **d_columns = nullptr;
        const auto cleanup = [&]()
        {
            cudaFree(d_columns);
            cudaFree(d_columnar);
            cudaFree(d_interleaved);
            cudaFree(d_weights);
        };

        if (cudaMalloc(&d_weights, field_bytes) != cudaSuccess ||
            cudaMalloc(&d_interleaved, interleaved_bytes) != cudaSuccess ||
            cudaMalloc(&d_columnar, interleaved_bytes) != cudaSuccess ||
            cudaMalloc(&d_columns, ptr_bytes) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        if (cudaMemcpy(
                d_weights,
                weight_limbs,
                field_bytes,
                cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(
                d_interleaved,
                interleaved_value_limbs,
                interleaved_bytes,
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cleanup();
            return false;
        }
        record_cuda_h2d(field_bytes);
        record_cuda_h2d(interleaved_bytes);

        constexpr unsigned kTileDim = 32;
        std::size_t grid_x_size = 0;
        std::size_t grid_y_size = 0;
        if (!checked_ceil_div(
                poly_count,
                kTileDim,
                &grid_x_size) ||
            !checked_ceil_div(
                domain_size,
                kTileDim,
                &grid_y_size) ||
            grid_x_size > std::numeric_limits<unsigned>::max() ||
            grid_y_size > std::numeric_limits<unsigned>::max())
        {
            cleanup();
            return false;
        }

        transpose_interleaved_to_columnar_kernel<<<
            dim3(
                static_cast<unsigned>(grid_x_size),
                static_cast<unsigned>(grid_y_size)),
            dim3(kTileDim, 8)>>>(
            d_interleaved,
            d_columnar,
            domain_size,
            poly_count);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }

        std::vector<const gpu::GpuFr *> h_columns(poly_count);
        for (std::size_t row = 0; row < poly_count; ++row)
        {
            h_columns[row] = d_columnar + row * domain_size;
        }
        if (cudaMemcpy(
                d_columns,
                h_columns.data(),
                ptr_bytes,
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cleanup();
            return false;
        }
        record_cuda_h2d(ptr_bytes);

        // No second full source copy is retained while the multi-rotation
        // kernel runs.  The shared helper below uses the existing optimized
        // columnar rotation/reduction kernels over d_columnar.
        cudaFree(d_interleaved);
        d_interleaved = nullptr;
        const bool ok =
            weighted_poly_eval_gpu_multi_rotations_device_columns(
                d_weights,
                d_columns,
                output_limbs,
                domain_size,
                poly_count,
                rotations,
                rotation_count);
        if (!ok)
        {
            cleanup();
            return false;
        }
        cleanup();
        record_cuda_kernel_invocations(1);
        return true;
    }

    namespace
    {
        struct WeightedDotWeightCacheEntry
        {
            gpu::GpuFr *device_weights =
                nullptr;

            std::size_t domain_size = 0;
            std::size_t column_count = 0;
            std::size_t byte_count = 0;

            ~WeightedDotWeightCacheEntry()
            {
                if (device_weights != nullptr)
                {
                    (void)cudaFree(
                        device_weights);
                }
            }
        };

        std::mutex &
        weighted_dot_cache_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        std::unordered_map<
            std::string,
            std::shared_ptr<
                WeightedDotWeightCacheEntry>> &
        weighted_dot_cache()
        {
            static std::unordered_map<
                std::string,
                std::shared_ptr<
                    WeightedDotWeightCacheEntry>>
                cache;

            return cache;
        }

        __global__ void
        weighted_dot_chunk_kernel(
            const gpu::GpuFr *weights,
            const gpu::GpuFr *values,
            gpu::GpuFr *partials,
            std::size_t total_count,
            std::size_t offset,
            std::size_t count,
            std::size_t column_count)
        {
            extern __shared__
                gpu::GpuFr shared[];

            const auto column =
                static_cast<std::size_t>(
                    blockIdx.y);

            const auto local =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            auto value =
                gpu::fr_zero();

            if (column < column_count &&
                local < count)
            {
                value =
                    gpu::fr_mul(
                        values[local],
                        weights[column *
                                    total_count +
                                offset +
                                local]);
            }

            shared[threadIdx.x] =
                value;

            __syncthreads();

            for (unsigned stride =
                     blockDim.x / 2;
                 stride != 0;
                 stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    shared[threadIdx.x] =
                        gpu::fr_add(
                            shared[threadIdx.x],
                            shared[threadIdx.x +
                                   stride]);
                }

                __syncthreads();
            }

            if (threadIdx.x == 0)
            {
                partials[column * gridDim.x +
                         blockIdx.x] =
                    shared[0];
            }
        }

        __global__ void
        weighted_dot_reduce_kernel(
            const gpu::GpuFr *partials,
            gpu::GpuFr *accumulators,
            std::size_t partial_count,
            std::size_t column_count)
        {
            const auto column =
                static_cast<std::size_t>(
                    blockIdx.x);

            if (column >= column_count ||
                threadIdx.x != 0)
            {
                return;
            }

            auto sum =
                gpu::fr_zero();

            for (std::size_t index = 0;
                 index < partial_count;
                 ++index)
            {
                sum =
                    gpu::fr_add(
                        sum,
                        partials[column *
                                     partial_count +
                                 index]);
            }

            accumulators[column] =
                gpu::fr_add(
                    accumulators[column],
                    sum);
        }

        std::string weighted_dot_cache_key(
            const std::vector<
                const mcl::Fr *> &columns,
            std::size_t total_count)
        {
            std::ostringstream out;

            out << total_count
                << ':'
                << columns.size();

            for (const auto *column :
                 columns)
            {
                out << ':'
                    << static_cast<
                           unsigned long long>(
                           reinterpret_cast<
                               std::uintptr_t>(
                               column));
            }

            return out.str();
        }

        std::shared_ptr<
            WeightedDotWeightCacheEntry>
        acquire_weighted_dot_weights(
            const std::vector<
                const mcl::Fr *> &columns,
            std::size_t total_count)
        {
            for (const auto *column :
                 columns)
            {
                if (column == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA weighted-dot "
                        "received a null "
                        "weight column");
                }
            }

            const auto key =
                weighted_dot_cache_key(
                    columns,
                    total_count);

            std::lock_guard<std::mutex> lock(
                weighted_dot_cache_mutex());

            auto &cache =
                weighted_dot_cache();

            if (const auto found =
                    cache.find(key);
                found != cache.end())
            {
                return found->second;
            }

            if (columns.size() >
                std::numeric_limits<
                    std::size_t>::max() /
                    total_count /
                    sizeof(gpu::GpuFr))
            {
                throw std::runtime_error(
                    "CUDA weighted-dot "
                    "weight byte overflow");
            }

            auto entry =
                std::make_shared<
                    WeightedDotWeightCacheEntry>();

            entry->domain_size =
                total_count;

            entry->column_count =
                columns.size();

            entry->byte_count =
                total_count *
                columns.size() *
                sizeof(gpu::GpuFr);

            auto status =
                cudaMalloc(
                    reinterpret_cast<void **>(
                        &entry->device_weights),
                    entry->byte_count);

            if (status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(
                        "CUDA weighted-dot "
                        "weight allocation "
                        "failed: ") +
                    cudaGetErrorString(
                        status));
            }

            for (std::size_t column = 0;
                 column < columns.size();
                 ++column)
            {
                status =
                    cudaMemcpy(
                        entry->device_weights +
                            column *
                                total_count,
                        columns[column],
                        total_count *
                            sizeof(
                                gpu::GpuFr),
                        cudaMemcpyHostToDevice);

                if (status !=
                    cudaSuccess)
                {
                    throw std::runtime_error(
                        std::string(
                            "CUDA weighted-dot "
                            "weight upload "
                            "failed: ") +
                        cudaGetErrorString(
                            status));
                }
            }

            cache.emplace(
                key,
                entry);

            return entry;
        }

        class CudaWeightedDotSessionImpl final
            : public CudaWeightedDotSession
        {
        public:
            CudaWeightedDotSessionImpl(
                const std::vector<
                    const mcl::Fr *>
                    &weight_columns,
                std::size_t total_count,
                std::size_t
                    maximum_chunk_size)
                : total_count_(
                      total_count),
                  column_count_(
                      weight_columns.size()),
                  maximum_chunk_size_(
                      std::min(
                          total_count,
                          maximum_chunk_size)),
                  weights_(
                      acquire_weighted_dot_weights(
                          weight_columns,
                          total_count))
            {
                static_assert(
                    sizeof(gpu::GpuFr) ==
                    sizeof(mcl::Fr));

                if (column_count_ == 0 ||
                    maximum_chunk_size_ == 0)
                {
                    throw std::runtime_error(
                        "invalid CUDA weighted-dot "
                        "session shape");
                }

                constexpr std::size_t
                    kBlockSize = 256;

                partial_capacity_ =
                    (maximum_chunk_size_ +
                     kBlockSize - 1) /
                    kBlockSize;

                try
                {
                    auto status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_values_),
                            maximum_chunk_size_ *
                                sizeof(
                                    gpu::GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw std::runtime_error(
                            std::string(
                                "CUDA weighted-dot "
                                "value allocation "
                                "failed: ") +
                            cudaGetErrorString(
                                status));
                    }

                    if (column_count_ >
                        std::numeric_limits<
                            std::size_t>::max() /
                            partial_capacity_ /
                            sizeof(
                                gpu::GpuFr))
                    {
                        throw std::runtime_error(
                            "CUDA weighted-dot "
                            "partial byte overflow");
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_partials_),
                            column_count_ *
                                partial_capacity_ *
                                sizeof(
                                    gpu::GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw std::runtime_error(
                            std::string(
                                "CUDA weighted-dot "
                                "partial allocation "
                                "failed: ") +
                            cudaGetErrorString(
                                status));
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_accumulators_),
                            column_count_ *
                                sizeof(
                                    gpu::GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw std::runtime_error(
                            std::string(
                                "CUDA weighted-dot "
                                "accumulator "
                                "allocation failed: ") +
                            cudaGetErrorString(
                                status));
                    }

                    status =
                        cudaMemset(
                            device_accumulators_,
                            0,
                            column_count_ *
                                sizeof(
                                    gpu::GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw std::runtime_error(
                            std::string(
                                "CUDA weighted-dot "
                                "accumulator clear "
                                "failed: ") +
                            cudaGetErrorString(
                                status));
                    }
                }
                catch (...)
                {
                    release_noexcept();
                    throw;
                }
            }

            ~CudaWeightedDotSessionImpl()
                override
            {
                release_noexcept();
            }

            void accumulate(
                const mcl::Fr *values,
                std::size_t offset,
                std::size_t count)
                override
            {
                if (finished_)
                {
                    throw std::runtime_error(
                        "CUDA weighted-dot "
                        "session was already "
                        "finished");
                }

                if (count == 0)
                {
                    return;
                }

                if (values == nullptr ||
                    offset >
                        total_count_ ||
                    count >
                        total_count_ -
                            offset ||
                    count >
                        maximum_chunk_size_)
                {
                    throw std::runtime_error(
                        "invalid CUDA weighted-dot "
                        "chunk");
                }

                auto status =
                    cudaMemcpy(
                        device_values_,
                        values,
                        count *
                            sizeof(
                                gpu::GpuFr),
                        cudaMemcpyHostToDevice);

                if (status != cudaSuccess)
                {
                    throw std::runtime_error(
                        std::string(
                            "CUDA weighted-dot "
                            "value upload failed: ") +
                        cudaGetErrorString(
                            status));
                }

                constexpr unsigned
                    kBlockSize = 256;

                const auto partial_count =
                    (count +
                     kBlockSize - 1) /
                    kBlockSize;

                if (partial_count >
                        std::numeric_limits<
                            unsigned>::max() ||
                    column_count_ >
                        65535)
                {
                    throw std::runtime_error(
                        "CUDA weighted-dot grid "
                        "overflow");
                }

                const dim3 grid(
                    static_cast<unsigned>(
                        partial_count),
                    static_cast<unsigned>(
                        column_count_),
                    1);

                weighted_dot_chunk_kernel<<<
                    grid,
                    kBlockSize,
                    kBlockSize *
                        sizeof(
                            gpu::GpuFr)>>>(
                    weights_->device_weights,
                    device_values_,
                    device_partials_,
                    total_count_,
                    offset,
                    count,
                    column_count_);

                status =
                    cudaGetLastError();

                if (status !=
                    cudaSuccess)
                {
                    throw std::runtime_error(
                        std::string(
                            "CUDA weighted-dot "
                            "chunk kernel failed: ") +
                        cudaGetErrorString(
                            status));
                }

                weighted_dot_reduce_kernel<<<
                    static_cast<unsigned>(
                        column_count_),
                    1>>>(
                    device_partials_,
                    device_accumulators_,
                    partial_count,
                    column_count_);

                status =
                    cudaGetLastError();

                if (status !=
                    cudaSuccess)
                {
                    throw std::runtime_error(
                        std::string(
                            "CUDA weighted-dot "
                            "reduce kernel failed: ") +
                        cudaGetErrorString(
                            status));
                }

                record_cuda_kernel_invocations(
                    2);
            }

            std::vector<mcl::Fr>
            finish() override
            {
                if (finished_)
                {
                    throw std::runtime_error(
                        "CUDA weighted-dot "
                        "session was already "
                        "finished");
                }

                std::vector<mcl::Fr>
                    output(
                        column_count_);

                const auto status =
                    cudaMemcpy(
                        output.data(),
                        device_accumulators_,
                        column_count_ *
                            sizeof(
                                gpu::GpuFr),
                        cudaMemcpyDeviceToHost);

                if (status != cudaSuccess)
                {
                    throw std::runtime_error(
                        std::string(
                            "CUDA weighted-dot "
                            "result download "
                            "failed: ") +
                        cudaGetErrorString(
                            status));
                }

                finished_ = true;

                return output;
            }

        private:
            void release_noexcept() noexcept
            {
                if (device_values_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_values_);
                }

                if (device_partials_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_partials_);
                }

                if (device_accumulators_ !=
                    nullptr)
                {
                    (void)cudaFree(
                        device_accumulators_);
                }

                device_values_ =
                    nullptr;

                device_partials_ =
                    nullptr;

                device_accumulators_ =
                    nullptr;
            }

            std::size_t total_count_ = 0;
            std::size_t column_count_ = 0;

            std::size_t
                maximum_chunk_size_ = 0;

            std::size_t
                partial_capacity_ = 0;

            std::shared_ptr<
                WeightedDotWeightCacheEntry>
                weights_;

            gpu::GpuFr *device_values_ =
                nullptr;

            gpu::GpuFr *device_partials_ =
                nullptr;

            gpu::GpuFr
                *device_accumulators_ =
                    nullptr;

            bool finished_ = false;
        };
    }

    std::unique_ptr<CudaWeightedDotSession>
    create_cuda_weighted_dot_session(
        const std::vector<
            const mcl::Fr *> &weight_columns,
        std::size_t total_count,
        std::size_t maximum_chunk_size)
    {
        if (weight_columns.empty() ||
            total_count == 0 ||
            maximum_chunk_size == 0)
        {
            return nullptr;
        }

        return std::make_unique<
            CudaWeightedDotSessionImpl>(
            weight_columns,
            total_count,
            maximum_chunk_size);
    }

    void release_cuda_weighted_dot_cache()
    {
        std::lock_guard<std::mutex> lock(
            weighted_dot_cache_mutex());

        weighted_dot_cache().clear();
    }

} // namespace gatzk::algebra

namespace gatzk::protocol
{
    namespace
    {

        __global__ void histogram_indices_kernel(
            const unsigned long long *indices,
            std::size_t count,
            unsigned long long *buckets,
            std::size_t bucket_count)
        {
            const auto tid =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
            if (tid >= count)
            {
                return;
            }
            const auto bucket = static_cast<std::size_t>(indices[tid]);
            if (bucket < bucket_count)
            {
                atomicAdd(&buckets[bucket], 1ULL);
            }
        }

    } // namespace

    std::vector<std::size_t> lookup_histogram_indices_cuda(
        const std::vector<std::size_t> &indices,
        std::size_t domain_size)
    {
        std::vector<std::size_t> out(
            domain_size,
            0);

        if (indices.empty() ||
            domain_size == 0)
        {
            return out;
        }

        static_assert(
            sizeof(std::size_t) ==
                sizeof(unsigned long long),
            "CUDA lookup histogram requires 64-bit size_t");

        constexpr unsigned int
            block_size = 256;

        const auto grid_size =
            static_cast<unsigned int>(
                (indices.size() +
                 block_size - 1U) /
                block_size);

        if (gatzk::algebra::
                cuda_backend_unified_addressing_available())
        {
            std::vector<unsigned long long>
                host_counts(
                    domain_size,
                    0ULL);

            histogram_indices_kernel<<<
                grid_size,
                block_size>>>(
                reinterpret_cast<
                    const unsigned long long *>(
                    indices.data()),
                indices.size(),
                host_counts.data(),
                domain_size);

            const auto launch_status =
                cudaGetLastError();

            if (launch_status !=
                cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(
                        "cuda zero-copy histogram "
                        "kernel launch failed: ") +
                    cudaGetErrorString(
                        launch_status));
            }

            const auto sync_status =
                cudaDeviceSynchronize();

            if (sync_status !=
                cudaSuccess)
            {
                throw std::runtime_error(
                    std::string(
                        "cuda zero-copy histogram "
                        "synchronization failed: ") +
                    cudaGetErrorString(
                        sync_status));
            }

            for (std::size_t i = 0;
                 i < domain_size;
                 ++i)
            {
                out[i] =
                    static_cast<std::size_t>(
                        host_counts[i]);
            }

            gatzk::algebra::
                record_cuda_kernel_invocations(
                    1);

            gatzk::algebra::
                record_cuda_stream_sync();

            return out;
        }

        std::vector<unsigned long long>
            host_indices(
                indices.begin(),
                indices.end());

        std::vector<unsigned long long>
            host_counts(
                domain_size,
                0ULL);

        unsigned long long
            *device_indices = nullptr;

        unsigned long long
            *device_counts = nullptr;

        const auto indices_bytes =
            sizeof(unsigned long long) *
            host_indices.size();

        const auto counts_bytes =
            sizeof(unsigned long long) *
            host_counts.size();

        if (cudaMalloc(
                &device_indices,
                indices_bytes) !=
            cudaSuccess)
        {
            return {};
        }

        if (cudaMalloc(
                &device_counts,
                counts_bytes) !=
            cudaSuccess)
        {
            cudaFree(device_indices);
            return {};
        }

        const auto cleanup = [&]()
        {
            cudaFree(device_indices);
            cudaFree(device_counts);
        };

        if (cudaMemcpy(
                device_indices,
                host_indices.data(),
                indices_bytes,
                cudaMemcpyHostToDevice) !=
            cudaSuccess)
        {
            cleanup();

            throw std::runtime_error(
                "cuda histogram failed to upload indices");
        }

        gatzk::algebra::
            record_cuda_h2d(
                indices_bytes);

        if (cudaMemset(
                device_counts,
                0,
                counts_bytes) !=
            cudaSuccess)
        {
            cleanup();

            throw std::runtime_error(
                "cuda histogram failed to reset bucket buffer");
        }

        histogram_indices_kernel<<<
            grid_size,
            block_size>>>(
            device_indices,
            host_indices.size(),
            device_counts,
            host_counts.size());

        const auto launch_status =
            cudaGetLastError();

        if (launch_status !=
            cudaSuccess)
        {
            cleanup();

            throw std::runtime_error(
                std::string(
                    "cuda histogram kernel "
                    "launch failed: ") +
                cudaGetErrorString(
                    launch_status));
        }

        const auto sync_status =
            cudaDeviceSynchronize();

        if (sync_status !=
            cudaSuccess)
        {
            cleanup();

            throw std::runtime_error(
                std::string(
                    "cuda histogram synchronization failed: ") +
                cudaGetErrorString(
                    sync_status));
        }

        gatzk::algebra::
            record_cuda_stream_sync();

        if (cudaMemcpy(
                host_counts.data(),
                device_counts,
                counts_bytes,
                cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        {
            cleanup();

            throw std::runtime_error(
                "cuda histogram failed to download counts");
        }

        gatzk::algebra::
            record_cuda_d2h(
                counts_bytes);

        cleanup();

        for (std::size_t i = 0;
             i < domain_size;
             ++i)
        {
            out[i] =
                static_cast<std::size_t>(
                    host_counts[i]);
        }

        gatzk::algebra::
            record_cuda_kernel_invocations(
                1);

        return out;
    }
} // namespace gatzk::protocol
