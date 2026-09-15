#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <chrono>

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/effective_memory.hpp"

#include "fields/snark_fields/bls12_381_scalar.cuh"
#include "gpu-utils/device_context.cuh"
#include "ntt/ntt.cuh"

extern "C" cudaError_t bls12_381_initialize_domain(
    bls12_381::scalar_t *primitive_root,
    device_context::DeviceContext &ctx,
    bool fast_twiddles_mode);

extern "C" cudaError_t bls12_381_release_domain(
    device_context::DeviceContext &ctx);

extern "C" cudaError_t bls12_381_ntt_cuda(
    const bls12_381::scalar_t *input,
    int size,
    ntt::NTTDir dir,
    ntt::NTTConfig<bls12_381::scalar_t> &config,
    bls12_381::scalar_t *output);

namespace gatzk::algebra
{
    namespace
    {
        using IcicleScalar = bls12_381::scalar_t;

        // OGBN t_edge needs:
        //   2^23 inverse NTT
        //   two 2^24 forward NTTs.
        //
        // Deliberately do not initialize a 2^32 ICICLE domain: the additional
        // twiddle storage would be unnecessary for the production quotient path.
        constexpr std::size_t kMaximumCudaNttSize =
            std::size_t{1} << 24;

        // Larger quotient cosets are decomposed into radix-2 stages whose
        // leaf transforms stay at kMaximumCudaNttSize.  Keep the extension
        // bounded at a 2^26 lane (2^27 total coset) so admission can account
        // for a fixed, auditable workspace shape.
        constexpr std::size_t kMaximumCudaStagedNttSize =
            std::size_t{1} << 26;
        constexpr std::size_t kMaximumCudaCosetTargetSize =
            std::size_t{1} << 27;

        // Preserve workspace/headroom after persistent ICICLE twiddles and the
        // explicit input/output buffers have been accounted for.
        constexpr std::size_t kCudaNttReserveBytes =
            std::size_t{4} << 30;

        std::size_t checked_ntt_bytes(
            std::size_t size,
            std::size_t batch_size);

        struct CudaNttWorkspace;

        bool valid_bounded_coset_target(
            std::size_t target_size)
        {
            return target_size >= 2 &&
                   (target_size & (target_size - 1U)) == 0U &&
                   target_size <= kMaximumCudaCosetTargetSize &&
                   target_size / 2 <= kMaximumCudaStagedNttSize;
        }

        bool admit_large_coset_workspace(
            std::size_t target_size,
            int device,
            const CudaNttWorkspace &workspace,
            std::size_t caller_buffer_bytes = 0);

        static_assert(
            sizeof(IcicleScalar) == sizeof(mcl::Fr),
            "ICICLE BLS12-381 scalar and mcl::Fr must have identical width");

        static_assert(
            sizeof(mcl::Fr) == 4 * sizeof(std::uint64_t),
            "mcl::Fr must use four 64-bit limbs");

        std::mutex &cuda_ntt_domain_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }
        std::array<bool, device_context::MAX_DEVICES> &
        cuda_ntt_domain_initialized()
        {
            static std::array<
                bool,
                device_context::MAX_DEVICES>
                initialized{};
            return initialized;
        }

        [[noreturn]] void throw_cuda_ntt_error(
            const char *operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string("CUDA quotient NTT ") + operation + " failed: " + cudaGetErrorString(status));
        }

        class CudaRegisteredHostTransferGuard final
            : public CudaHostTransferGuard
        {
        public:
            CudaRegisteredHostTransferGuard(
                void *host_pointer,
                cudaEvent_t event,
                int device,
                bool unregister_on_wait)
                : host_pointer_(host_pointer),
                  event_(event),
                  device_(device),
                  unregister_on_wait_(unregister_on_wait)
            {
            }

            ~CudaRegisteredHostTransferGuard() override
            {
                complete_noexcept();
            }

            void wait() const override
            {
                std::lock_guard<std::mutex>
                    lock(mutex_);

                if (completed_)
                {
                    return;
                }

                const auto status =
                    complete_locked();

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "asynchronous coefficient transfer completion",
                        status);
                }
            }

        private:
            cudaError_t complete_locked() const noexcept
            {
                int previous_device = -1;

                auto first_error =
                    cudaGetDevice(
                        &previous_device);

                if (first_error != cudaSuccess)
                {
                    return first_error;
                }

                bool switched_device = false;

                if (previous_device != device_)
                {
                    const auto status =
                        cudaSetDevice(
                            device_);

                    if (status != cudaSuccess)
                    {
                        return status;
                    }

                    switched_device = true;
                }

                first_error = cudaSuccess;

                if (event_ != nullptr)
                {
                    const auto status =
                        cudaEventSynchronize(
                            event_);

                    if (first_error == cudaSuccess &&
                        status != cudaSuccess)
                    {
                        first_error = status;
                    }
                }

                if (unregister_on_wait_ &&
                    host_pointer_ != nullptr)
                {
                    const auto status =
                        cudaHostUnregister(
                            host_pointer_);

                    if (first_error == cudaSuccess &&
                        status != cudaSuccess)
                    {
                        first_error = status;
                    }

                    unregister_on_wait_ = false;
                }

                if (event_ != nullptr)
                {
                    const auto status =
                        cudaEventDestroy(
                            event_);

                    if (first_error == cudaSuccess &&
                        status != cudaSuccess)
                    {
                        first_error = status;
                    }

                    event_ = nullptr;
                }

                completed_ = true;

                if (switched_device)
                {
                    const auto status =
                        cudaSetDevice(
                            previous_device);

                    if (first_error == cudaSuccess &&
                        status != cudaSuccess)
                    {
                        first_error = status;
                    }
                }

                return first_error;
            }

            void complete_noexcept() const noexcept
            {
                std::lock_guard<std::mutex>
                    lock(mutex_);

                if (!completed_)
                {
                    (void)complete_locked();
                }
            }

            void *host_pointer_ = nullptr;

            mutable cudaEvent_t event_ =
                nullptr;

            int device_ = -1;

            mutable bool unregister_on_wait_ =
                false;

            mutable bool completed_ =
                false;

            mutable std::mutex mutex_;
        };

        void enqueue_default_stream_wait(
            cudaStream_t producer_stream,
            const char *operation)
        {
            cudaEvent_t event = nullptr;

            auto status =
                cudaEventCreateWithFlags(
                    &event,
                    cudaEventDisableTiming);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    operation,
                    status);
            }

            status =
                cudaEventRecord(
                    event,
                    producer_stream);

            if (status == cudaSuccess)
            {
                status =
                    cudaStreamWaitEvent(
                        nullptr,
                        event,
                        0);
            }

            const auto destroy_status =
                cudaEventDestroy(
                    event);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    operation,
                    status);
            }

            if (destroy_status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "dependency event destruction",
                    destroy_status);
            }
        }

        void enqueue_stream_wait_for_default(
            cudaStream_t consumer_stream,
            const char *operation)
        {
            cudaEvent_t event = nullptr;

            auto status =
                cudaEventCreateWithFlags(
                    &event,
                    cudaEventDisableTiming);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    operation,
                    status);
            }

            status =
                cudaEventRecord(
                    event,
                    nullptr);

            if (status == cudaSuccess)
            {
                status =
                    cudaStreamWaitEvent(
                        consumer_stream,
                        event,
                        0);
            }

            const auto destroy_status =
                cudaEventDestroy(
                    event);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    operation,
                    status);
            }

            if (destroy_status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "dependency event destruction",
                    destroy_status);
            }
        }

        std::size_t checked_ntt_bytes(
            std::size_t size,
            std::size_t batch_size)
        {
            if (batch_size == 0 || size >
                                       std::numeric_limits<std::size_t>::max() / batch_size / sizeof(mcl::Fr))
            {
                throw std::runtime_error(
                    "CUDA quotient NTT byte-size overflow");
            }

            return size * batch_size * sizeof(mcl::Fr);
        }

        IcicleScalar protocol_maximum_root()
        {
            // Use the protocol's exact MCL root rather than assuming that a
            // backend-specific primitive-root convention is identical.
            const auto root =
                FieldElement::root_of_unity(
                    kMaximumCudaNttSize);

            std::array<std::uint8_t, 64> bytes{};

            if (root.native().getLittleEndian(
                    bytes.data(),
                    bytes.size()) == 0)
            {
                throw std::runtime_error(
                    "failed to export protocol NTT root");
            }

            IcicleScalar output{};

            static_assert(
                sizeof(output.limbs_storage.limbs) == sizeof(mcl::Fr));

            std::memcpy(
                output.limbs_storage.limbs,
                bytes.data(),
                sizeof(output.limbs_storage.limbs));

            return output;
        }

        void ensure_cuda_ntt_domain(
            int device,
            device_context::DeviceContext &ctx)
        {
            if (device < 0 ||
                static_cast<std::size_t>(device) >=
                    device_context::MAX_DEVICES)
            {
                throw std::runtime_error(
                    "CUDA quotient NTT received an invalid device id");
            }

            std::lock_guard<std::mutex> lock(
                cuda_ntt_domain_mutex());

            auto &initialized =
                cuda_ntt_domain_initialized();

            if (initialized[static_cast<std::size_t>(device)])
            {
                return;
            }

            auto primitive_root =
                protocol_maximum_root();

            const auto status =
                bls12_381_initialize_domain(
                    &primitive_root,
                    ctx,
                    false);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "domain initialization",
                    status);
            }

            const auto sync_status =
                cudaStreamSynchronize(ctx.stream);

            if (sync_status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "domain initialization synchronization",
                    sync_status);
            }

            initialized[static_cast<std::size_t>(device)] = true;
        }
        struct CudaNttWorkspace
        {
            IcicleScalar *input = nullptr;
            IcicleScalar *output = nullptr;
            IcicleScalar *scratch_a = nullptr;
            IcicleScalar *scratch_b = nullptr;

            unsigned long long
                *coefficient_shape = nullptr;

            std::size_t capacity_bytes = 0;
            int device = -1;
            cudaStream_t stream = nullptr;

            ~CudaNttWorkspace()
            {
                release();
            }

            bool ensure(
                std::size_t required_bytes,
                int current_device,
                bool staged = false)
            {
                if (stream == nullptr ||
                    device != current_device)
                {
                    release();

                    device = current_device;

                    auto status =
                        cudaStreamCreateWithFlags(
                            &stream,
                            cudaStreamNonBlocking);

                    if (status != cudaSuccess)
                    {
                        stream = nullptr;
                        device = -1;

                        throw_cuda_ntt_error(
                            "stream creation",
                            status);
                    }
                }

                if (capacity_bytes >= required_bytes &&
                    input != nullptr &&
                    output != nullptr &&
                    (!staged ||
                     (scratch_a != nullptr &&
                      scratch_b != nullptr)))
                {
                    return true;
                }

                const auto release_buffer = [&](IcicleScalar **buffer)
                {
                    if (*buffer != nullptr)
                    {
                        (void)cudaFreeAsync(
                            *buffer,
                            stream);
                        *buffer = nullptr;
                    }
                };

                release_buffer(&input);
                release_buffer(&output);
                release_buffer(&scratch_a);
                release_buffer(&scratch_b);

                (void)cudaStreamSynchronize(stream);

                capacity_bytes = 0;

                auto status =
                    cudaMallocAsync(
                        reinterpret_cast<void **>(&input),
                        required_bytes,
                        stream);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();
                    input = nullptr;
                    return false;
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "persistent input allocation",
                        status);
                }

                status =
                    cudaMallocAsync(
                        reinterpret_cast<void **>(&output),
                        required_bytes,
                        stream);

                if (status != cudaSuccess)
                {
                    (void)cudaFreeAsync(
                        input,
                        stream);

                    (void)cudaStreamSynchronize(
                        stream);

                    input = nullptr;
                    output = nullptr;

                    if (status == cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();
                        return false;
                    }

                    throw_cuda_ntt_error(
                        "persistent output allocation",
                        status);
                }

                if (staged)
                {
                    status =
                        cudaMallocAsync(
                            reinterpret_cast<void **>(&scratch_a),
                            required_bytes,
                            stream);

                    if (status != cudaSuccess)
                    {
                        release_buffer(&input);
                        release_buffer(&output);
                        (void)cudaStreamSynchronize(stream);
                        scratch_a = nullptr;
                        scratch_b = nullptr;

                        if (status == cudaErrorMemoryAllocation)
                        {
                            (void)cudaGetLastError();
                            return false;
                        }

                        throw_cuda_ntt_error(
                            "persistent staged scratch-a allocation",
                            status);
                    }

                    status =
                        cudaMallocAsync(
                            reinterpret_cast<void **>(&scratch_b),
                            required_bytes,
                            stream);

                    if (status != cudaSuccess)
                    {
                        release_buffer(&input);
                        release_buffer(&output);
                        release_buffer(&scratch_a);
                        (void)cudaStreamSynchronize(stream);
                        scratch_b = nullptr;

                        if (status == cudaErrorMemoryAllocation)
                        {
                            (void)cudaGetLastError();
                            return false;
                        }

                        throw_cuda_ntt_error(
                            "persistent staged scratch-b allocation",
                            status);
                    }
                }

                capacity_bytes = required_bytes;
                return true;
            }

            std::size_t resident_bytes() const noexcept
            {
                if (capacity_bytes == 0)
                {
                    return 0;
                }

                std::size_t buffer_count = 0;
                if (input != nullptr)
                {
                    ++buffer_count;
                }
                if (output != nullptr)
                {
                    ++buffer_count;
                }
                if (scratch_a != nullptr)
                {
                    ++buffer_count;
                }
                if (scratch_b != nullptr)
                {
                    ++buffer_count;
                }
                if (buffer_count == 0 ||
                    capacity_bytes >
                        std::numeric_limits<std::size_t>::max() /
                            buffer_count)
                {
                    return std::numeric_limits<std::size_t>::max();
                }
                return buffer_count * capacity_bytes;
            }

            unsigned long long *
            coefficient_shape_buffer()
            {
                if (coefficient_shape !=
                    nullptr)
                {
                    return coefficient_shape;
                }

                if (stream == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA NTT coefficient-shape workspace has no stream");
                }

                const auto status =
                    cudaMallocAsync(
                        reinterpret_cast<
                            void **>(
                            &coefficient_shape),
                        2U *
                            sizeof(
                                unsigned long long),
                        stream);

                if (status != cudaSuccess)
                {
                    coefficient_shape =
                        nullptr;

                    throw_cuda_ntt_error(
                        "persistent coefficient-shape allocation",
                        status);
                }

                return coefficient_shape;
            }

            bool can_reuse(
                std::size_t required_bytes,
                int current_device,
                bool staged) const noexcept
            {
                return stream != nullptr &&
                       device == current_device &&
                       capacity_bytes >= required_bytes &&
                       input != nullptr &&
                       output != nullptr &&
                       (!staged ||
                        (scratch_a != nullptr &&
                         scratch_b != nullptr));
            }

            void release() noexcept
            {
                if (stream != nullptr)
                {
                    if (input != nullptr)
                    {
                        (void)cudaFreeAsync(
                            input,
                            stream);
                    }

                    if (output != nullptr)
                    {
                        (void)cudaFreeAsync(
                            output,
                            stream);
                    }

                    if (scratch_a != nullptr)
                    {
                        (void)cudaFreeAsync(
                            scratch_a,
                            stream);
                    }

                    if (scratch_b != nullptr)
                    {
                        (void)cudaFreeAsync(
                            scratch_b,
                            stream);
                    }

                    if (coefficient_shape !=
                        nullptr)
                    {
                        (void)cudaFreeAsync(
                            coefficient_shape,
                            stream);
                    }

                    (void)cudaStreamSynchronize(
                        stream);

                    (void)cudaStreamDestroy(
                        stream);
                }

                input = nullptr;
                output = nullptr;
                scratch_a = nullptr;
                scratch_b = nullptr;
                coefficient_shape = nullptr;
                capacity_bytes = 0;
                stream = nullptr;
                device = -1;
            }
        };

        CudaNttWorkspace &cuda_ntt_workspace()
        {
            thread_local CudaNttWorkspace workspace;
            return workspace;
        }

        // ICICLE allocates the pinned 2^24 domain lazily.  The exact allocator
        // footprint includes implementation tables in addition to the field
        // vector, so keep a separate fixed allowance in every admission.  It
        // is deliberately conservative and remains bounded independently of
        // the caller's graph fan-out.
        constexpr std::size_t kIcicleDomainHeadroomBytes =
            std::size_t{1} << 30;

        bool admit_large_coset_workspace(
            std::size_t target_size,
            int device,
            const CudaNttWorkspace &workspace,
            std::size_t caller_buffer_bytes)
        {
            if (target_size < 2 ||
                (target_size & (target_size - 1U)) != 0U ||
                target_size > kMaximumCudaCosetTargetSize)
            {
                return false;
            }

            std::size_t one_buffer_bytes = 0;
            std::size_t icicle_domain_bytes = 0;
            try
            {
                one_buffer_bytes =
                    checked_ntt_bytes(target_size, 1);
                icicle_domain_bytes =
                    checked_ntt_bytes(
                        kMaximumCudaNttSize + 1,
                        1);
            }
            catch (...)
            {
                return false;
            }

            if (one_buffer_bytes >
                std::numeric_limits<std::size_t>::max() / 2)
            {
                return false;
            }

            const auto persistent_workspace_bytes =
                2 * one_buffer_bytes;
            const bool reusable =
                workspace.can_reuse(
                    one_buffer_bytes,
                    device,
                    false);
            const auto resident_workspace_bytes =
                workspace.device == device
                    ? workspace.resident_bytes()
                    : 0;

            // ensure() releases an undersized workspace before allocating the
            // replacement.  When capacity is reusable, all existing buffers
            // (including staged scratch) remain live and are already reflected
            // in the free-memory query.  Otherwise the replacement is the
            // conservative peak to admit before any allocation occurs.
            const auto live_workspace_bytes =
                reusable
                    ? resident_workspace_bytes
                    : persistent_workspace_bytes;

            std::size_t required_bytes = live_workspace_bytes;
            const auto add_required =
                [&required_bytes](std::size_t value)
            {
                if (value >
                    std::numeric_limits<std::size_t>::max() -
                        required_bytes)
                {
                    return false;
                }
                required_bytes += value;
                return true;
            };

            // host_output / source buffers are caller-owned but are live for
            // the complete transform.  Include them in the UMA and discrete
            // conservative budget so a new large caller buffer cannot bypass
            // the CUDA admission gate.
            if (!add_required(caller_buffer_bytes) ||
                !add_required(icicle_domain_bytes) ||
                !add_required(kIcicleDomainHeadroomBytes))
            {
                return false;
            }

            int integrated = 0;
            auto status =
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
                if (!host.effective_available_known)
                {
                    return false;
                }

                constexpr std::size_t kMinimumIntegratedReserve =
                    std::size_t{8} << 30;
                constexpr std::size_t kIntegratedTransformHeadroom =
                    std::size_t{3} << 30;
                const auto reserve_bytes =
                    std::max(
                        kMinimumIntegratedReserve,
                        !host.effective_total_known
                            ? kMinimumIntegratedReserve
                            : host.effective_total_bytes / 16);
                if (reserve_bytes >
                    std::numeric_limits<std::size_t>::max() -
                        kIntegratedTransformHeadroom)
                {
                    return false;
                }
                const auto fixed_reserve =
                    reserve_bytes +
                    kIntegratedTransformHeadroom;
                if (host.effective_available_bytes <= fixed_reserve)
                {
                    return false;
                }
                return required_bytes <=
                       host.effective_available_bytes -
                           fixed_reserve;
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

            const auto reserve_bytes =
                std::max(
                    kCudaNttReserveBytes,
                    total_bytes / 8);
            if (free_bytes <= reserve_bytes)
            {
                return false;
            }
            return required_bytes <=
                   free_bytes - reserve_bytes;
        }

        using GpuFr =
            algebra::gpu::GpuFr;

        constexpr std::size_t
            kCosetPowerChunkElements = 256;

        GpuFr gpu_field(
            const FieldElement &value)
        {
            GpuFr out{};
            const auto &native = value.native();

            std::memcpy(
                &out,
                &native,
                sizeof(out));

            return out;
        }

        __device__ __forceinline__
            GpuFr
            fr_pow_u64(
                GpuFr base,
                std::uint64_t exponent)
        {
            auto result =
                algebra::gpu::fr_one();

            while (exponent != 0)
            {
                if ((exponent & 1ULL) != 0)
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
        __global__ void
        prepare_coset_lanes_kernel(
            const GpuFr *coefficients,
            std::size_t coefficient_count,
            std::size_t source_domain_size,
            const GpuFr *blinder,
            std::size_t blinder_count,
            std::size_t active_count,
            GpuFr shift,
            GpuFr odd_shift,
            GpuFr *even_output,
            GpuFr *odd_output)
        {
            const auto chunk =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto begin =
                chunk *
                kCosetPowerChunkElements;

            if (begin >= active_count)
            {
                return;
            }

            const auto end =
                min(
                    active_count,
                    begin +
                        kCosetPowerChunkElements);

            auto even_power =
                fr_pow_u64(
                    shift,
                    begin);

            auto odd_power =
                fr_pow_u64(
                    odd_shift,
                    begin);

            for (std::size_t index = begin;
                 index < end;
                 ++index)
            {
                auto coefficient =
                    index < coefficient_count
                        ? coefficients[index]
                        : algebra::gpu::fr_zero();

                // Exactly reproduce:
                //
                // c[i] -= R[i]
                // c[n+i] += R[i]
                //
                // from coset_evaluate_polynomial_into().
                if (index < blinder_count)
                {
                    coefficient =
                        algebra::gpu::fr_sub(
                            coefficient,
                            blinder[index]);
                }

                if (index >= source_domain_size)
                {
                    const auto blinder_index =
                        index -
                        source_domain_size;

                    if (blinder_index <
                        blinder_count)
                    {
                        coefficient =
                            algebra::gpu::fr_add(
                                coefficient,
                                blinder[blinder_index]);
                    }
                }

                even_output[index] =
                    algebra::gpu::fr_mul(
                        coefficient,
                        even_power);

                odd_output[index] =
                    algebra::gpu::fr_mul(
                        coefficient,
                        odd_power);

                even_power =
                    algebra::gpu::fr_mul(
                        even_power,
                        shift);

                odd_power =
                    algebra::gpu::fr_mul(
                        odd_power,
                        odd_shift);
            }
        }
        __global__ void
        split_even_odd_kernel(
            const GpuFr *input,
            GpuFr *even_output,
            GpuFr *odd_output,
            std::size_t half_size)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= half_size)
            {
                return;
            }

            even_output[index] =
                input[2 * index];

            odd_output[index] =
                input[2 * index + 1];
        }

        __global__ void
        interleave_split_lanes_kernel(
            const GpuFr *input,
            GpuFr *output,
            std::size_t half_size)
        {
            const auto index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            if (index >= half_size)
            {
                return;
            }

            output[2 * index] =
                input[index];

            output[2 * index + 1] =
                input[index + half_size];
        }
        template <bool TrackCoefficientShape>
        __global__ void
        combine_inverse_coset_kernel(
            const GpuFr *even_values,
            const GpuFr *odd_values,
            GpuFr *output,
            std::size_t half_size,
            GpuFr omega_inverse,
            GpuFr inverse_two,
            GpuFr shift_inverse,
            GpuFr shift_inverse_half,
            unsigned long long *coefficient_shape)
        {
            const auto chunk =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto begin =
                chunk *
                kCosetPowerChunkElements;

            if constexpr (!TrackCoefficientShape)
            {
                if (begin >= half_size)
                {
                    return;
                }
            }

            unsigned long long local_nonzero_count = 0;
            unsigned long long local_last_nonzero_exclusive = 0;

            if (begin < half_size)
            {
                const auto end =
                    min(
                        half_size,
                        begin +
                            kCosetPowerChunkElements);

                auto omega_power =
                    fr_pow_u64(
                        omega_inverse,
                        begin);

                auto shift_power =
                    fr_pow_u64(
                        shift_inverse,
                        begin);

                for (std::size_t index = begin;
                     index < end;
                     ++index)
                {
                    const auto odd_scaled =
                        algebra::gpu::fr_mul(
                            odd_values[index],
                            omega_power);

                    const auto first =
                        algebra::gpu::fr_mul(
                            algebra::gpu::fr_add(
                                even_values[index],
                                odd_scaled),
                            inverse_two);

                    const auto second =
                        algebra::gpu::fr_mul(
                            algebra::gpu::fr_sub(
                                even_values[index],
                                odd_scaled),
                            inverse_two);

                    const auto first_output =
                        algebra::gpu::fr_mul(
                            first,
                            shift_power);

                    const auto second_output =
                        algebra::gpu::fr_mul(
                            second,
                            algebra::gpu::fr_mul(
                                shift_power,
                                shift_inverse_half));

                    output[index] =
                        first_output;

                    output[index + half_size] =
                        second_output;

                    if constexpr (TrackCoefficientShape)
                    {
                        if (!algebra::gpu::fr_is_zero(
                                first_output))
                        {
                            ++local_nonzero_count;

                            const auto last =
                                static_cast<unsigned long long>(
                                    index + 1U);

                            if (last >
                                local_last_nonzero_exclusive)
                            {
                                local_last_nonzero_exclusive =
                                    last;
                            }
                        }

                        if (!algebra::gpu::fr_is_zero(
                                second_output))
                        {
                            ++local_nonzero_count;

                            const auto last =
                                static_cast<unsigned long long>(
                                    index +
                                    half_size +
                                    1U);

                            if (last >
                                local_last_nonzero_exclusive)
                            {
                                local_last_nonzero_exclusive =
                                    last;
                            }
                        }
                    }

                    omega_power =
                        algebra::gpu::fr_mul(
                            omega_power,
                            omega_inverse);

                    shift_power =
                        algebra::gpu::fr_mul(
                            shift_power,
                            shift_inverse);
                }
            }

            if constexpr (TrackCoefficientShape)
            {
                extern __shared__ unsigned long long
                    shared_shape[];

                auto *shared_nonzero =
                    shared_shape;

                auto *shared_last =
                    shared_shape +
                    blockDim.x;

                shared_nonzero[threadIdx.x] =
                    local_nonzero_count;

                shared_last[threadIdx.x] =
                    local_last_nonzero_exclusive;

                __syncthreads();

                for (unsigned stride =
                         blockDim.x / 2U;
                     stride != 0;
                     stride >>= 1U)
                {
                    if (threadIdx.x < stride)
                    {
                        shared_nonzero[threadIdx.x] +=
                            shared_nonzero[threadIdx.x +
                                           stride];

                        const auto candidate =
                            shared_last[threadIdx.x +
                                        stride];

                        if (candidate >
                            shared_last[threadIdx.x])
                        {
                            shared_last[threadIdx.x] =
                                candidate;
                        }
                    }

                    __syncthreads();
                }

                if (threadIdx.x == 0 &&
                    shared_nonzero[0] != 0)
                {
                    atomicAdd(
                        coefficient_shape,
                        shared_nonzero[0]);

                    atomicMax(
                        coefficient_shape + 1,
                        shared_last[0]);
                }
            }
        }

        // The pinned ICICLE build is intentionally limited to a 2^24
        // transform. Quotient cosets can be one or two powers larger. The
        // kernels below provide the exact radix-2 decomposition needed to
        // keep every ICICLE call within that bound while retaining the
        // canonical natural-order transform.
        __global__ void
        split_ntt_residues_kernel(
            const GpuFr *input,
            GpuFr *output,
            std::size_t vector_count,
            std::size_t transform_size,
            std::size_t base_size,
            std::size_t radix)
        {
            const auto flat_index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto total =
                vector_count * transform_size;

            if (flat_index >= total)
            {
                return;
            }

            const auto vector =
                flat_index / transform_size;
            const auto logical =
                flat_index % transform_size;
            const auto residue =
                logical % radix;
            const auto index =
                logical / radix;

            output[vector * transform_size +
                   residue * base_size + index] =
                input[flat_index];
        }

        __global__ void
        combine_radix2_groups_kernel(
            const GpuFr *input,
            GpuFr *output,
            std::size_t vector_count,
            std::size_t groups_per_vector,
            std::size_t half_size,
            std::size_t input_group_stride,
            std::size_t second_input_offset,
            std::size_t output_group_stride,
            std::size_t vector_stride,
            GpuFr root,
            GpuFr combine_scale)
        {
            const auto chunk_index =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto group_items =
                groups_per_vector *
                ((half_size + kCosetPowerChunkElements - 1) /
                 kCosetPowerChunkElements);
            const auto total =
                vector_count * group_items;

            if (chunk_index >= total)
            {
                return;
            }

            const auto vector =
                chunk_index / group_items;
            const auto group_index =
                (chunk_index % group_items) /
                ((half_size + kCosetPowerChunkElements - 1) /
                 kCosetPowerChunkElements);
            const auto chunk =
                chunk_index %
                ((half_size + kCosetPowerChunkElements - 1) /
                 kCosetPowerChunkElements);
            const auto begin =
                chunk * kCosetPowerChunkElements;
            const auto end =
                min(
                    half_size,
                    begin + kCosetPowerChunkElements);

            const auto input_base =
                vector * vector_stride +
                group_index * input_group_stride;
            const auto output_base =
                vector * vector_stride +
                group_index * output_group_stride;

            auto twiddle =
                fr_pow_u64(root, begin);
            for (std::size_t index = begin;
                 index < end;
                 ++index)
            {
                const auto scaled_odd =
                    algebra::gpu::fr_mul(
                        input[input_base + second_input_offset + index],
                        twiddle);
                const auto even =
                    input[input_base + index];

                // An in-place combine is safe: each index owns exactly the two
                // outputs at index and index+half_size, and no other thread
                // writes either location.
                output[output_base + index] =
                    algebra::gpu::fr_mul(
                        algebra::gpu::fr_add(even, scaled_odd),
                        combine_scale);
                output[output_base + half_size + index] =
                    algebra::gpu::fr_mul(
                        algebra::gpu::fr_sub(even, scaled_odd),
                        combine_scale);
                twiddle =
                    algebra::gpu::fr_mul(
                        twiddle,
                        root);
            }
        }

        bool run_cuda_ntt_device(
            const IcicleScalar *device_input,
            IcicleScalar *device_output,
            std::size_t size,
            std::size_t batch_size,
            ntt::NTTDir direction,
            int device,
            cudaStream_t stream)
        {
            if (device_input == nullptr ||
                device_output == nullptr ||
                size == 0 ||
                (size & (size - 1U)) != 0U ||
                size > kMaximumCudaNttSize ||
                batch_size == 0 ||
                batch_size >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()) ||
                size >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()))
            {
                return false;
            }

            // DeviceContext::stream is a reference.  Do not obtain a
            // context from get_default_device_context() and then assign
            // through ctx.stream here: that reference aliases ICICLE's
            // process-global static default_stream and therefore races when
            // independent CUDA NTT workers execute concurrently.
            //
            // Bind the context reference to per-call storage instead.  The
            // storage lives for the complete synchronous host-side NTT
            // submission and each worker therefore keeps its own CUDA stream.
            cudaStream_t context_stream =
                stream;

            device_context::DeviceContext
                ctx{
                    context_stream,
                    static_cast<std::size_t>(
                        device),
                    cudaMemPool_t{}};

            ensure_cuda_ntt_domain(
                device,
                ctx);

            auto config =
                ntt::default_ntt_config<
                    IcicleScalar>(ctx);
            config.batch_size =
                static_cast<int>(
                    batch_size);

            config.columns_batch = false;

            config.ordering =
                ntt::Ordering::kNN;

            config.are_inputs_on_device =
                true;

            config.are_outputs_on_device =
                true;

            config.is_async = true;

            config.ntt_algorithm =
                ntt::NttAlgorithm::Auto;

            const auto status =
                bls12_381_ntt_cuda(
                    device_input,
                    static_cast<int>(size),
                    direction,
                    config,
                    device_output);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "execution",
                    status);
            }

            if (direction ==
                ntt::NTTDir::kForward)
            {
                record_cuda_ntt_call();
            }
            else
            {
                record_cuda_intt_call();
            }

            return true;
        }
        bool run_bounded_ntt_device(
            const mcl::Fr *device_input,
            mcl::Fr *device_output,
            std::size_t size,
            std::size_t batch_size,
            ntt::NTTDir direction,
            int device,
            cudaStream_t stream,
            mcl::Fr *scratch_a,
            mcl::Fr *scratch_b)
        {
            if (device_input == nullptr ||
                device_output == nullptr ||
                size < 2 ||
                (size & (size - 1U)) != 0U ||
                size > kMaximumCudaStagedNttSize ||
                batch_size == 0 ||
                batch_size > 2)
            {
                return false;
            }

            if (size <= kMaximumCudaNttSize)
            {
                return run_cuda_ntt_device(
                    reinterpret_cast<const IcicleScalar *>(
                        device_input),
                    reinterpret_cast<IcicleScalar *>(
                        device_output),
                    size,
                    batch_size,
                    direction,
                    device,
                    stream);
            }

            if (scratch_a == nullptr ||
                scratch_b == nullptr ||
                scratch_a == scratch_b)
            {
                return false;
            }

            const auto radix =
                size / kMaximumCudaNttSize;
            if (radix != 2 && radix != 4)
            {
                return false;
            }

            if (batch_size >
                std::numeric_limits<std::size_t>::max() / size)
            {
                return false;
            }

            const auto total_elements =
                batch_size * size;
            const auto total_bytes =
                checked_ntt_bytes(size, batch_size);

            constexpr unsigned kBlockSize = 256;
            const auto split_blocks_size =
                (total_elements + kBlockSize - 1) / kBlockSize;
            if (split_blocks_size >
                std::numeric_limits<unsigned>::max())
            {
                return false;
            }

            split_ntt_residues_kernel<<<
                static_cast<unsigned>(split_blocks_size),
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<const GpuFr *>(device_input),
                reinterpret_cast<GpuFr *>(scratch_a),
                batch_size,
                size,
                kMaximumCudaNttSize,
                radix);

            auto status = cudaGetLastError();
            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "staged transform residue split",
                    status);
            }
            record_cuda_kernel_invocations(1);

            if (batch_size >
                std::numeric_limits<std::size_t>::max() / radix)
            {
                return false;
            }
            const auto subtransform_batch =
                batch_size * radix;

            if (!run_cuda_ntt_device(
                    reinterpret_cast<const IcicleScalar *>(scratch_a),
                    reinterpret_cast<IcicleScalar *>(scratch_b),
                    kMaximumCudaNttSize,
                    subtransform_batch,
                    direction,
                    device,
                    stream))
            {
                return false;
            }

            const auto combine_root_for =
                [&](std::size_t transform_size)
            {
                auto root =
                    FieldElement::root_of_unity(transform_size);
                if (direction == ntt::NTTDir::kInverse)
                {
                    root = root.inv();
                }
                return gpu_field(root);
            };

            const auto combine_scale =
                direction == ntt::NTTDir::kInverse
                    ? gpu_field(FieldElement(2).inv())
                    : gpu_field(FieldElement(1));

            if (radix == 2)
            {
                const auto combine_items =
                    batch_size *
                    ((kMaximumCudaNttSize +
                      kCosetPowerChunkElements - 1) /
                     kCosetPowerChunkElements);
                const auto combine_blocks_size =
                    (combine_items + kBlockSize - 1) /
                    kBlockSize;
                if (combine_blocks_size >
                    std::numeric_limits<unsigned>::max())
                {
                    return false;
                }

                combine_radix2_groups_kernel<<<
                    static_cast<unsigned>(combine_blocks_size),
                    kBlockSize,
                    0,
                    stream>>>(
                    reinterpret_cast<const GpuFr *>(scratch_b),
                    reinterpret_cast<GpuFr *>(scratch_b),
                    batch_size,
                    1,
                    kMaximumCudaNttSize,
                    kMaximumCudaNttSize,
                    kMaximumCudaNttSize,
                    size,
                    size,
                    combine_root_for(size),
                    combine_scale);

                status = cudaGetLastError();
                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "staged transform radix-2 combine",
                        status);
                }
                record_cuda_kernel_invocations(1);
            }
            else
            {
                const auto first_stage_items =
                    batch_size * 2 *
                    ((kMaximumCudaNttSize +
                      kCosetPowerChunkElements - 1) /
                     kCosetPowerChunkElements);
                const auto first_stage_blocks_size =
                    (first_stage_items + kBlockSize - 1) /
                    kBlockSize;
                if (first_stage_blocks_size >
                    std::numeric_limits<unsigned>::max())
                {
                    return false;
                }

                // Pair residues (0,2) and (1,3) in each vector to form the
                // two length-2^25 intermediate transforms.
                combine_radix2_groups_kernel<<<
                    static_cast<unsigned>(first_stage_blocks_size),
                    kBlockSize,
                    0,
                    stream>>>(
                    reinterpret_cast<const GpuFr *>(scratch_b),
                    reinterpret_cast<GpuFr *>(scratch_a),
                    batch_size,
                    2,
                    kMaximumCudaNttSize,
                    kMaximumCudaNttSize,
                    2 * kMaximumCudaNttSize,
                    2 * kMaximumCudaNttSize,
                    size,
                    combine_root_for(2 * kMaximumCudaNttSize),
                    combine_scale);

                status = cudaGetLastError();
                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "staged transform first radix-2 combine",
                        status);
                }
                record_cuda_kernel_invocations(1);

                const auto final_stage_items =
                    batch_size * 2 *
                    ((kMaximumCudaNttSize +
                      kCosetPowerChunkElements - 1) /
                     kCosetPowerChunkElements);
                const auto final_stage_blocks_size =
                    (final_stage_items + kBlockSize - 1) /
                    kBlockSize;
                if (final_stage_blocks_size >
                    std::numeric_limits<unsigned>::max())
                {
                    return false;
                }

                combine_radix2_groups_kernel<<<
                    static_cast<unsigned>(final_stage_blocks_size),
                    kBlockSize,
                    0,
                    stream>>>(
                    reinterpret_cast<const GpuFr *>(scratch_a),
                    reinterpret_cast<GpuFr *>(scratch_a),
                    batch_size,
                    1,
                    2 * kMaximumCudaNttSize,
                    2 * kMaximumCudaNttSize,
                    2 * kMaximumCudaNttSize,
                    size,
                    size,
                    combine_root_for(size),
                    combine_scale);

                status = cudaGetLastError();
                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "staged transform final radix-2 combine",
                        status);
                }
                record_cuda_kernel_invocations(1);
            }

            const auto staged_output =
                radix == 2 ? scratch_b : scratch_a;
            if (device_output != staged_output)
            {
                status = cudaMemcpyAsync(
                    device_output,
                    staged_output,
                    total_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream);
                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "staged transform output copy",
                        status);
                }
            }

            return true;
        }

        // Generic production NTTs use the same two persistent buffers as the
        // direct path.  A staged transform additionally needs two full-size
        // scratch buffers for residue decomposition and radix combines.  Do
        // this admission before CudaNttWorkspace::ensure() so an accepted
        // large route cannot fail halfway through allocation.
        bool admit_generic_ntt_workspace(
            std::size_t size,
            std::size_t batch_size,
            int device,
            const CudaNttWorkspace &workspace,
            std::size_t caller_buffer_bytes)
        {
            if (size < 2 ||
                (size & (size - 1U)) != 0U ||
                size > kMaximumCudaStagedNttSize ||
                batch_size == 0 ||
                batch_size > 2)
            {
                return false;
            }

            std::size_t total_bytes = 0;
            std::size_t icicle_domain_bytes = 0;
            try
            {
                total_bytes = checked_ntt_bytes(size, batch_size);
                icicle_domain_bytes =
                    checked_ntt_bytes(
                        kMaximumCudaNttSize + 1,
                        1);
            }
            catch (...)
            {
                return false;
            }

            const auto buffer_count =
                size > kMaximumCudaNttSize
                    ? std::size_t{4}
                    : std::size_t{2};
            if (total_bytes >
                std::numeric_limits<std::size_t>::max() /
                    buffer_count)
            {
                return false;
            }
            const auto required_workspace_bytes =
                buffer_count * total_bytes;
            const bool reusable =
                workspace.can_reuse(
                    total_bytes,
                    device,
                    size > kMaximumCudaNttSize);
            const auto resident_workspace_bytes =
                workspace.device == device
                    ? workspace.resident_bytes()
                    : 0;
            const auto live_workspace_bytes =
                reusable
                    ? resident_workspace_bytes
                    : required_workspace_bytes;

            std::size_t required_bytes = live_workspace_bytes;
            const auto add_required =
                [&required_bytes](std::size_t value)
            {
                if (value >
                    std::numeric_limits<std::size_t>::max() -
                        required_bytes)
                {
                    return false;
                }
                required_bytes += value;
                return true;
            };
            if (!add_required(caller_buffer_bytes) ||
                !add_required(icicle_domain_bytes) ||
                !add_required(kIcicleDomainHeadroomBytes))
            {
                return false;
            }

            int integrated = 0;
            auto status =
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
                if (!host.effective_available_known)
                {
                    return false;
                }

                constexpr std::size_t kMinimumIntegratedReserve =
                    std::size_t{8} << 30;
                constexpr std::size_t kIntegratedTransformHeadroom =
                    std::size_t{3} << 30;
                const auto reserve_bytes =
                    std::max(
                        kMinimumIntegratedReserve,
                        !host.effective_total_known
                            ? kMinimumIntegratedReserve
                            : host.effective_total_bytes / 16);
                if (reserve_bytes >
                    std::numeric_limits<std::size_t>::max() -
                        kIntegratedTransformHeadroom)
                {
                    return false;
                }
                const auto fixed_reserve =
                    reserve_bytes +
                    kIntegratedTransformHeadroom;
                if (host.effective_available_bytes <= fixed_reserve)
                {
                    return false;
                }
                return required_bytes <=
                       host.effective_available_bytes -
                           fixed_reserve;
            }

            std::size_t free_bytes = 0;
            std::size_t total_device_bytes = 0;
            status =
                cudaMemGetInfo(
                    &free_bytes,
                    &total_device_bytes);
            if (status != cudaSuccess)
            {
                (void)cudaGetLastError();
                return false;
            }
            const auto reserve_bytes =
                std::max(
                    kCudaNttReserveBytes,
                    total_device_bytes / 8);
            if (required_bytes >
                std::numeric_limits<std::size_t>::max() -
                    reserve_bytes)
            {
                return false;
            }
            return free_bytes >=
                   required_bytes + reserve_bytes;
        }

        bool run_cuda_ntt_batch(
            mcl::Fr *first,
            mcl::Fr *second,
            std::size_t size,
            std::size_t batch_size,
            ntt::NTTDir direction)
        {
            if (size < 2 ||
                (size & (size - 1U)) != 0U ||
                size > kMaximumCudaStagedNttSize)
            {
                return false;
            }

            if (first == nullptr ||
                (batch_size == 2 &&
                 second == nullptr))
            {
                throw std::runtime_error(
                    "CUDA quotient NTT received a null input");
            }

            if (batch_size != 1 &&
                batch_size != 2)
            {
                throw std::runtime_error(
                    "CUDA quotient NTT supports only batch size 1 or 2");
            }

            int device = 0;

            auto status =
                cudaGetDevice(&device);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "device lookup",
                    status);
            }

            const auto one_vector_bytes =
                checked_ntt_bytes(
                    size,
                    1);

            const auto total_bytes =
                checked_ntt_bytes(
                    size,
                    batch_size);

            const bool staged =
                size > kMaximumCudaNttSize;

            auto &workspace =
                cuda_ntt_workspace();

            if (!admit_generic_ntt_workspace(
                    size,
                    batch_size,
                    device,
                    workspace,
                    total_bytes))
            {
                return false;
            }

            if (!workspace.ensure(
                    total_bytes,
                    device,
                    staged))
            {
                return false;
            }

            const auto stream =
                workspace.stream;

            status =
                cudaMemcpyAsync(
                    workspace.input,
                    first,
                    one_vector_bytes,
                    cudaMemcpyHostToDevice,
                    stream);

            if (status != cudaSuccess)
            {
                workspace.release();

                throw_cuda_ntt_error(
                    "first input upload",
                    status);
            }

            record_cuda_h2d(
                one_vector_bytes);

            if (batch_size == 2)
            {
                status =
                    cudaMemcpyAsync(
                        workspace.input + size,
                        second,
                        one_vector_bytes,
                        cudaMemcpyHostToDevice,
                        stream);

                if (status != cudaSuccess)
                {
                    workspace.release();

                    throw_cuda_ntt_error(
                        "second input upload",
                        status);
                }

                record_cuda_h2d(
                    one_vector_bytes);
            }

            if (!run_bounded_ntt_device(
                    reinterpret_cast<const mcl::Fr *>(
                        workspace.input),
                    reinterpret_cast<mcl::Fr *>(
                        workspace.output),
                    size,
                    batch_size,
                    direction,
                    device,
                    stream,
                    reinterpret_cast<mcl::Fr *>(
                        workspace.scratch_a),
                    reinterpret_cast<mcl::Fr *>(
                        workspace.scratch_b)))
            {
                return false;
            }

            status =
                cudaMemcpyAsync(
                    first,
                    workspace.output,
                    one_vector_bytes,
                    cudaMemcpyDeviceToHost,
                    stream);

            if (status != cudaSuccess)
            {
                workspace.release();

                throw_cuda_ntt_error(
                    "first output download",
                    status);
            }

            record_cuda_d2h(
                one_vector_bytes);

            if (batch_size == 2)
            {
                status =
                    cudaMemcpyAsync(
                        second,
                        workspace.output + size,
                        one_vector_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);

                if (status != cudaSuccess)
                {
                    workspace.release();

                    throw_cuda_ntt_error(
                        "second output download",
                        status);
                }

                record_cuda_d2h(
                    one_vector_bytes);
            }

            status =
                cudaStreamSynchronize(
                    stream);

            if (status != cudaSuccess)
            {
                workspace.release();

                throw_cuda_ntt_error(
                    "synchronization",
                    status);
            }

            record_cuda_stream_sync();

            return true;
        }

    }

    CudaNttWorkspaceResidency
    cuda_ntt_workspace_residency()
    {
        const auto &workspace =
            cuda_ntt_workspace();

        return CudaNttWorkspaceResidency{
            .capacity_bytes = workspace.capacity_bytes,
            .resident_bytes = workspace.resident_bytes(),
            .device = workspace.device,
            .has_input_output =
                workspace.input != nullptr &&
                workspace.output != nullptr,
            .has_staged_scratch =
                workspace.scratch_a != nullptr &&
                workspace.scratch_b != nullptr};
    }

    bool cuda_coset_evaluations_to_device_exact(
        const mcl::Fr *values,
        std::size_t value_count,
        std::size_t domain_size,
        const mcl::Fr *zero_domain_blinder,
        std::size_t blinder_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size,
        bool make_default_stream_wait,
        bool wait_for_default_stream_before_output)
    {
        if (device_output == nullptr ||
            domain_size == 0 ||
            target_size < 2 ||
            value_count > domain_size ||
            (domain_size &
             (domain_size - 1U)) != 0U ||
            (target_size &
             (target_size - 1U)) != 0U)
        {
            return false;
        }

        if (value_count != 0 &&
            values == nullptr)
        {
            throw std::runtime_error(
                "CUDA coset evaluation received null values");
        }

        if (blinder_count != 0 &&
            zero_domain_blinder == nullptr)
        {
            throw std::runtime_error(
                "CUDA coset evaluation received null blinder");
        }

        if (domain_size >
            std::numeric_limits<
                std::size_t>::max() -
                blinder_count)
        {
            return false;
        }

        const auto half_size =
            target_size / 2;

        const auto active_count =
            domain_size +
            blinder_count;

        if (domain_size >
                kMaximumCudaStagedNttSize ||
            half_size >
                kMaximumCudaStagedNttSize ||
            target_size >
                kMaximumCudaCosetTargetSize ||
            active_count >
                half_size)
        {
            return false;
        }

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        const auto total_bytes =
            checked_ntt_bytes(
                target_size,
                1);

        if (!admit_large_coset_workspace(
                target_size,
                device,
                workspace))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        status =
            cudaMemsetAsync(
                workspace.input,
                0,
                total_bytes,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "evaluation workspace clear",
                status);
        }

        if (value_count != 0)
        {
            status =
                cudaMemcpyAsync(
                    workspace.input,
                    values,
                    checked_ntt_bytes(
                        value_count,
                        1),
                    cudaMemcpyHostToDevice,
                    stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "evaluation upload",
                    status);
            }

            record_cuda_h2d(
                checked_ntt_bytes(
                    value_count,
                    1));
        }

        if (!run_bounded_ntt_device(
                reinterpret_cast<const mcl::Fr *>(workspace.input),
                reinterpret_cast<mcl::Fr *>(workspace.output),
                domain_size,
                1,
                ntt::NTTDir::kInverse,
                device,
                stream,
                reinterpret_cast<mcl::Fr *>(workspace.output),
                reinterpret_cast<mcl::Fr *>(workspace.input)))
        {
            return false;
        }

        if (blinder_count != 0)
        {
            status =
                cudaMemcpyAsync(
                    workspace.output +
                        domain_size,
                    zero_domain_blinder,
                    checked_ntt_bytes(
                        blinder_count,
                        1),
                    cudaMemcpyHostToDevice,
                    stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "blinder upload",
                    status);
            }

            record_cuda_h2d(
                checked_ntt_bytes(
                    blinder_count,
                    1));
        }

        status =
            cudaMemsetAsync(
                workspace.input,
                0,
                total_bytes,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coset lane clear",
                status);
        }

        const auto root =
            FieldElement::
                root_of_unity(
                    target_size);

        const auto odd_shift =
            shift * root;

        const auto chunk_count =
            (active_count +
             kCosetPowerChunkElements -
             1) /
            kCosetPowerChunkElements;

        constexpr unsigned
            kBlockSize = 256;

        const auto block_count =
            static_cast<unsigned>(
                (chunk_count +
                 kBlockSize - 1) /
                kBlockSize);

        if (block_count != 0)
        {
            prepare_coset_lanes_kernel<<<
                block_count,
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output),
                domain_size,
                domain_size,
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output +
                    domain_size),
                blinder_count,
                active_count,
                gpu_field(
                    shift),
                gpu_field(
                    odd_shift),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input +
                    half_size));

            status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coset lane preparation",
                    status);
            }

            record_cuda_kernel_invocations(
                1);
        }

        if (wait_for_default_stream_before_output)
        {
            enqueue_stream_wait_for_default(
                stream,
                "coset evaluation output dependency");
        }

        if (!run_bounded_ntt_device(
                reinterpret_cast<const mcl::Fr *>(workspace.input),
                reinterpret_cast<
                    mcl::Fr *>(
                    device_output),
                half_size,
                2,
                ntt::NTTDir::kForward,
                device,
                stream,
                reinterpret_cast<mcl::Fr *>(workspace.output),
                reinterpret_cast<mcl::Fr *>(workspace.input)))
        {
            return false;
        }
        if (make_default_stream_wait)
        {
            enqueue_default_stream_wait(
                stream,
                "coset evaluation default-stream dependency");

            return true;
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coset evaluation synchronization",
                status);
        }

        record_cuda_stream_sync();

        return true;
    }

    bool cuda_coset_coefficients_to_device_exact(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size,
        bool make_default_stream_wait,
        bool wait_for_default_stream_before_output)
    {
        if (device_output == nullptr ||
            target_size < 2 ||
            (target_size &
             (target_size - 1U)) != 0U)
        {
            return false;
        }

        if (coefficient_count != 0 &&
            coefficients == nullptr)
        {
            throw std::runtime_error(
                "CUDA coefficient coset received null coefficients");
        }

        const auto half_size =
            target_size / 2;

        if (half_size >
                kMaximumCudaStagedNttSize ||
            target_size >
                kMaximumCudaCosetTargetSize ||
            coefficient_count >
                half_size)
        {
            return false;
        }

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        const auto total_bytes =
            checked_ntt_bytes(
                target_size,
                1);

        if (!admit_large_coset_workspace(
                target_size,
                device,
                workspace))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        status =
            cudaMemsetAsync(
                workspace.input,
                0,
                total_bytes,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient lane clear",
                status);
        }

        if (coefficient_count != 0)
        {
            status =
                cudaMemcpyAsync(
                    workspace.output,
                    coefficients,
                    checked_ntt_bytes(
                        coefficient_count,
                        1),
                    cudaMemcpyHostToDevice,
                    stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient upload",
                    status);
            }

            record_cuda_h2d(
                checked_ntt_bytes(
                    coefficient_count,
                    1));
        }

        const auto root =
            FieldElement::
                root_of_unity(
                    target_size);

        const auto odd_shift =
            shift * root;

        const auto chunk_count =
            (coefficient_count +
             kCosetPowerChunkElements -
             1) /
            kCosetPowerChunkElements;

        constexpr unsigned
            kBlockSize = 256;

        const auto block_count =
            static_cast<unsigned>(
                (chunk_count +
                 kBlockSize - 1) /
                kBlockSize);

        if (block_count != 0)
        {
            prepare_coset_lanes_kernel<<<
                block_count,
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output),
                coefficient_count,
                coefficient_count,
                nullptr,
                0,
                coefficient_count,
                gpu_field(
                    shift),
                gpu_field(
                    odd_shift),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input +
                    half_size));

            status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient coset preparation",
                    status);
            }

            record_cuda_kernel_invocations(
                1);
        }

        if (wait_for_default_stream_before_output)
        {
            enqueue_stream_wait_for_default(
                stream,
                "coefficient coset output dependency");
        }

        if (!run_bounded_ntt_device(
                reinterpret_cast<const mcl::Fr *>(workspace.input),
                reinterpret_cast<
                    mcl::Fr *>(
                    device_output),
                half_size,
                2,
                ntt::NTTDir::kForward,
                device,
                stream,
                reinterpret_cast<mcl::Fr *>(workspace.output),
                reinterpret_cast<mcl::Fr *>(workspace.input)))
        {
            return false;
        }

        if (make_default_stream_wait)
        {
            enqueue_default_stream_wait(
                stream,
                "coefficient coset default-stream dependency");

            return true;
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient coset synchronization",
                status);
        }

        record_cuda_stream_sync();

        return true;
    }

    bool cuda_coset_device_to_coefficients_exact(
        const mcl::Fr *device_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients,
        bool wait_for_default_stream,
        mcl::Fr *device_coefficients,
        std::shared_ptr<
            const CudaHostTransferGuard>
            *host_transfer,
        std::size_t *logical_coefficient_count,
        std::size_t *nonzero_coefficient_count)
    {
        if (device_values == nullptr ||
            host_coefficients == nullptr ||
            size < 2 ||
            (size &
             (size - 1U)) != 0U)
        {
            return false;
        }

        if (host_transfer != nullptr)
        {
            host_transfer->reset();
        }

        const bool track_coefficient_shape =
            host_transfer != nullptr ||
            logical_coefficient_count != nullptr ||
            nonzero_coefficient_count != nullptr;

        const auto half_size =
            size / 2;

        if (half_size >
                kMaximumCudaStagedNttSize ||
            size >
                kMaximumCudaCosetTargetSize)
        {
            return false;
        }

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        const auto total_bytes =
            checked_ntt_bytes(
                size,
                1);

        if (!admit_large_coset_workspace(
                size,
                device,
                workspace))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        if (wait_for_default_stream)
        {
            enqueue_stream_wait_for_default(
                stream,
                "inverse coset producer dependency");
        }

        constexpr unsigned
            kBlockSize = 256;

        const auto split_blocks =
            static_cast<unsigned>(
                (half_size +
                 kBlockSize - 1) /
                kBlockSize);

        split_even_odd_kernel<<<
            split_blocks,
            kBlockSize,
            0,
            stream>>>(
            reinterpret_cast<
                const GpuFr *>(
                device_values),
            reinterpret_cast<
                GpuFr *>(
                workspace.input),
            reinterpret_cast<
                GpuFr *>(
                workspace.input +
                half_size),
            half_size);

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "inverse coset split",
                status);
        }

        record_cuda_kernel_invocations(
            1);

        if (!run_bounded_ntt_device(
                reinterpret_cast<const mcl::Fr *>(workspace.input),
                reinterpret_cast<mcl::Fr *>(workspace.output),
                half_size,
                2,
                ntt::NTTDir::kInverse,
                device,
                stream,
                reinterpret_cast<mcl::Fr *>(workspace.output),
                reinterpret_cast<mcl::Fr *>(workspace.input)))
        {
            return false;
        }

        const auto omega_inverse =
            FieldElement::
                root_of_unity(
                    size)
                    .inv();

        const auto inverse_two =
            FieldElement(2).inv();

        const auto shift_inverse =
            shift.inv();

        const auto
            shift_inverse_half =
                shift_inverse.pow(
                    half_size);

        const auto chunk_count =
            (half_size +
             kCosetPowerChunkElements -
             1) /
            kCosetPowerChunkElements;

        const auto combine_blocks =
            static_cast<unsigned>(
                (chunk_count +
                 kBlockSize - 1) /
                kBlockSize);

        auto *device_coefficient_shape =
            track_coefficient_shape
                ? workspace
                      .coefficient_shape_buffer()
                : nullptr;

        if (track_coefficient_shape)
        {
            status =
                cudaMemsetAsync(
                    device_coefficient_shape,
                    0,
                    2U *
                        sizeof(
                            unsigned long long),
                    stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient shape initialization",
                    status);
            }

            combine_inverse_coset_kernel<
                true><<<
                combine_blocks,
                kBlockSize,
                2U *
                    kBlockSize *
                    sizeof(
                        unsigned long long),
                stream>>>(
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output),
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output +
                    half_size),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input),
                half_size,
                gpu_field(
                    omega_inverse),
                gpu_field(
                    inverse_two),
                gpu_field(
                    shift_inverse),
                gpu_field(
                    shift_inverse_half),
                device_coefficient_shape);
        }
        else
        {
            combine_inverse_coset_kernel<
                false><<<
                combine_blocks,
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output),
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output +
                    half_size),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input),
                half_size,
                gpu_field(
                    omega_inverse),
                gpu_field(
                    inverse_two),
                gpu_field(
                    shift_inverse),
                gpu_field(
                    shift_inverse_half),
                nullptr);
        }

        status =
            cudaGetLastError();

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "inverse coset combine",
                status);
        }

        record_cuda_kernel_invocations(
            1);

        if (device_coefficients != nullptr &&
            device_coefficients !=
                reinterpret_cast<mcl::Fr *>(
                    workspace.input))
        {
            status =
                cudaMemcpyAsync(
                    device_coefficients,
                    workspace.input,
                    total_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "resident coefficient copy",
                    status);
            }
        }

        if (device_coefficients != nullptr)
        {
            enqueue_default_stream_wait(
                stream,
                "resident coefficient consumer dependency");
        }

        std::size_t resolved_logical_count =
            size;

        std::size_t resolved_nonzero_count =
            size;

        if (track_coefficient_shape)
        {
            std::array<
                unsigned long long,
                2>
                host_coefficient_shape{0, 0};

            status =
                cudaMemcpyAsync(
                    host_coefficient_shape.data(),
                    device_coefficient_shape,
                    sizeof(
                        host_coefficient_shape),
                    cudaMemcpyDeviceToHost,
                    stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient shape download",
                    status);
            }

            record_cuda_d2h(
                sizeof(
                    host_coefficient_shape));

            cudaEvent_t shape_ready =
                nullptr;

            status =
                cudaEventCreateWithFlags(
                    &shape_ready,
                    cudaEventDisableTiming);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient shape event creation",
                    status);
            }

            status =
                cudaEventRecord(
                    shape_ready,
                    stream);

            if (status == cudaSuccess)
            {
                status =
                    cudaEventSynchronize(
                        shape_ready);
            }

            const auto destroy_status =
                cudaEventDestroy(
                    shape_ready);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient shape dependency",
                    status);
            }

            if (destroy_status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient shape event destruction",
                    destroy_status);
            }

            if (host_coefficient_shape[1] >
                static_cast<
                    unsigned long long>(
                    size))
            {
                throw std::runtime_error(
                    "CUDA coefficient shape exceeds transform size");
            }

            resolved_nonzero_count =
                static_cast<std::size_t>(
                    host_coefficient_shape[0]);

            resolved_logical_count =
                host_coefficient_shape[1] == 0
                    ? std::size_t{1}
                    : static_cast<std::size_t>(
                          host_coefficient_shape[1]);

            if (resolved_nonzero_count >
                resolved_logical_count)
            {
                throw std::runtime_error(
                    "CUDA coefficient nonzero count exceeds logical coefficient count");
            }
        }

        if (logical_coefficient_count != nullptr)
        {
            *logical_coefficient_count =
                resolved_logical_count;
        }

        if (nonzero_coefficient_count != nullptr)
        {
            *nonzero_coefficient_count =
                resolved_nonzero_count;
        }

        if (host_transfer != nullptr)
        {
            const auto host_bytes =
                checked_ntt_bytes(
                    resolved_logical_count,
                    1);

            bool unregister_on_wait =
                false;

            status =
                cudaHostRegister(
                    host_coefficients,
                    host_bytes,
                    cudaHostRegisterDefault);

            if (status == cudaSuccess)
            {
                unregister_on_wait =
                    true;
            }
            else if (
                status ==
                cudaErrorHostMemoryAlreadyRegistered)
            {
                (void)cudaGetLastError();
            }
            else
            {
                (void)cudaGetLastError();

                status =
                    cudaMemcpyAsync(
                        host_coefficients,
                        workspace.input,
                        host_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "coefficient download",
                        status);
                }

                record_cuda_d2h(
                    host_bytes);

                status =
                    cudaStreamSynchronize(
                        stream);

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "coefficient synchronization",
                        status);
                }

                record_cuda_stream_sync();

                return true;
            }

            cudaEvent_t transfer_done =
                nullptr;

            status =
                cudaEventCreateWithFlags(
                    &transfer_done,
                    cudaEventDisableTiming);

            if (status != cudaSuccess)
            {
                if (unregister_on_wait)
                {
                    (void)cudaHostUnregister(
                        host_coefficients);
                }

                throw_cuda_ntt_error(
                    "coefficient transfer event creation",
                    status);
            }

            status =
                cudaMemcpyAsync(
                    host_coefficients,
                    workspace.input,
                    host_bytes,
                    cudaMemcpyDeviceToHost,
                    stream);

            if (status != cudaSuccess)
            {
                (void)cudaEventDestroy(
                    transfer_done);

                if (unregister_on_wait)
                {
                    (void)cudaHostUnregister(
                        host_coefficients);
                }

                throw_cuda_ntt_error(
                    "coefficient download",
                    status);
            }

            record_cuda_d2h(
                host_bytes);

            status =
                cudaEventRecord(
                    transfer_done,
                    stream);

            if (status != cudaSuccess)
            {
                const auto synchronize_status =
                    cudaStreamSynchronize(
                        stream);

                if (unregister_on_wait)
                {
                    (void)cudaHostUnregister(
                        host_coefficients);
                }

                (void)cudaEventDestroy(
                    transfer_done);

                if (synchronize_status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "coefficient transfer recovery synchronization",
                        synchronize_status);
                }

                throw_cuda_ntt_error(
                    "coefficient transfer event recording",
                    status);
            }

            try
            {
                *host_transfer =
                    std::make_shared<
                        CudaRegisteredHostTransferGuard>(
                        host_coefficients,
                        transfer_done,
                        device,
                        unregister_on_wait);
            }
            catch (...)
            {
                (void)cudaEventSynchronize(
                    transfer_done);

                if (unregister_on_wait)
                {
                    (void)cudaHostUnregister(
                        host_coefficients);
                }

                (void)cudaEventDestroy(
                    transfer_done);

                throw;
            }

            return true;
        }

        status =
            cudaMemcpyAsync(
                host_coefficients,
                workspace.input,
                total_bytes,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient download",
                status);
        }

        record_cuda_d2h(
            total_bytes);

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient synchronization",
                status);
        }

        record_cuda_stream_sync();

        return true;
    }

    bool cuda_coset_evaluations_to_host_exact(
        const mcl::Fr *values,
        std::size_t value_count,
        std::size_t domain_size,
        const mcl::Fr *zero_domain_blinder,
        std::size_t blinder_count,
        const FieldElement &shift,
        mcl::Fr *host_output,
        std::size_t target_size,
        bool preserve_split_lane)
    {
        if (host_output == nullptr ||
            !valid_bounded_coset_target(target_size) ||
            domain_size == 0 ||
            (domain_size & (domain_size - 1U)) != 0U ||
            domain_size > kMaximumCudaStagedNttSize ||
            value_count > domain_size ||
            domain_size >
                std::numeric_limits<std::size_t>::max() -
                    blinder_count ||
            domain_size + blinder_count > target_size / 2)
        {
            return false;
        }

        if (value_count != 0 && values == nullptr)
        {
            throw std::runtime_error(
                "CUDA coset evaluation received null values");
        }

        if (blinder_count != 0 && zero_domain_blinder == nullptr)
        {
            throw std::runtime_error(
                "CUDA coset evaluation received null blinder");
        }

        const auto total_bytes =
            checked_ntt_bytes(
                target_size,
                1);

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        if (!admit_large_coset_workspace(
                target_size,
                device,
                workspace,
                total_bytes))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        if (!cuda_coset_evaluations_to_device_exact(
                values,
                value_count,
                domain_size,
                zero_domain_blinder,
                blinder_count,
                shift,
                reinterpret_cast<mcl::Fr *>(
                    workspace.output),
                target_size))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        const auto half_size =
            target_size / 2;

        const mcl::Fr *device_source =
            reinterpret_cast<const mcl::Fr *>(
                workspace.output);

        if (!preserve_split_lane)
        {
            constexpr unsigned
                kBlockSize = 256;

            const auto block_count =
                static_cast<unsigned>(
                    (half_size +
                     kBlockSize - 1) /
                    kBlockSize);

            interleave_split_lanes_kernel<<<
                block_count,
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input),
                half_size);

            status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coset interleave",
                    status);
            }

            record_cuda_kernel_invocations(
                1);

            device_source =
                reinterpret_cast<const mcl::Fr *>(
                    workspace.input);
        }

        status =
            cudaMemcpyAsync(
                host_output,
                device_source,
                total_bytes,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coset output download",
                status);
        }

        record_cuda_d2h(
            total_bytes);

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coset output synchronization",
                status);
        }

        record_cuda_stream_sync();

        return true;
    }

    bool cuda_coset_coefficients_to_host_exact(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const FieldElement &shift,
        mcl::Fr *host_output,
        std::size_t target_size,
        bool preserve_split_lane)
    {
        if (host_output == nullptr ||
            !valid_bounded_coset_target(target_size) ||
            coefficient_count > target_size / 2)
        {
            return false;
        }

        if (coefficient_count != 0 && coefficients == nullptr)
        {
            throw std::runtime_error(
                "CUDA coefficient coset received null coefficients");
        }

        const auto total_bytes =
            checked_ntt_bytes(
                target_size,
                1);

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        if (!admit_large_coset_workspace(
                target_size,
                device,
                workspace,
                total_bytes))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        if (!cuda_coset_coefficients_to_device_exact(
                coefficients,
                coefficient_count,
                shift,
                reinterpret_cast<mcl::Fr *>(
                    workspace.output),
                target_size))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        const auto half_size =
            target_size / 2;

        const mcl::Fr *device_source =
            reinterpret_cast<const mcl::Fr *>(
                workspace.output);

        if (!preserve_split_lane)
        {
            constexpr unsigned
                kBlockSize = 256;

            const auto block_count =
                static_cast<unsigned>(
                    (half_size +
                     kBlockSize - 1) /
                    kBlockSize);

            interleave_split_lanes_kernel<<<
                block_count,
                kBlockSize,
                0,
                stream>>>(
                reinterpret_cast<
                    const GpuFr *>(
                    workspace.output),
                reinterpret_cast<
                    GpuFr *>(
                    workspace.input),
                half_size);

            status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "coefficient coset interleave",
                    status);
            }

            record_cuda_kernel_invocations(
                1);

            device_source =
                reinterpret_cast<const mcl::Fr *>(
                    workspace.input);
        }

        status =
            cudaMemcpyAsync(
                host_output,
                device_source,
                total_bytes,
                cudaMemcpyDeviceToHost,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient coset output download",
                status);
        }

        record_cuda_d2h(
            total_bytes);

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient coset output synchronization",
                status);
        }

        record_cuda_stream_sync();

        return true;
    }

    bool cuda_coset_host_to_coefficients_exact(
        const mcl::Fr *host_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients)
    {
        if (host_values == nullptr ||
            host_coefficients == nullptr ||
            !valid_bounded_coset_target(size))
        {
            return false;
        }

        const auto total_bytes =
            checked_ntt_bytes(
                size,
                1);

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        if (!admit_large_coset_workspace(
                size,
                device,
                workspace,
                total_bytes))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        status =
            cudaMemcpyAsync(
                workspace.output,
                host_values,
                total_bytes,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "quotient coset upload",
                status);
        }

        record_cuda_h2d(
            total_bytes);

        return cuda_coset_device_to_coefficients_exact(
            reinterpret_cast<const mcl::Fr *>(
                workspace.output),
            size,
            shift,
            host_coefficients);
    }

    bool cuda_coset_host_to_resident_coefficients_exact(
        const mcl::Fr *host_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients,
        std::shared_ptr<
            const CudaResidentCoefficientBuffer>
            *resident_coefficients)
    {
        if (resident_coefficients ==
            nullptr)
        {
            throw std::runtime_error(
                "CUDA resident coefficient output descriptor is null");
        }

        resident_coefficients->reset();

        if (host_values == nullptr ||
            host_coefficients == nullptr ||
            !valid_bounded_coset_target(
                size))
        {
            return false;
        }

        const auto total_bytes =
            checked_ntt_bytes(
                size,
                1);

        int device = 0;

        auto status =
            cudaGetDevice(
                &device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "resident coefficient device lookup",
                status);
        }

        auto &workspace =
            cuda_ntt_workspace();

        if (!admit_large_coset_workspace(
                size,
                device,
                workspace,
                total_bytes))
        {
            return false;
        }

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

        mcl::Fr *device_coefficients =
            nullptr;

        status =
            cudaMallocAsync(
                reinterpret_cast<
                    void **>(
                    &device_coefficients),
                total_bytes,
                stream);

        if (status ==
            cudaErrorMemoryAllocation)
        {
            (void)cudaGetLastError();

            return false;
        }

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "resident coefficient allocation",
                status);
        }

        const auto cleanup_device =
            [&]() noexcept
        {
            if (device_coefficients !=
                nullptr)
            {
                (void)cudaFreeAsync(
                    device_coefficients,
                    stream);

                (void)cudaStreamSynchronize(
                    stream);

                device_coefficients =
                    nullptr;
            }
        };

        status =
            cudaMemcpyAsync(
                workspace.output,
                host_values,
                total_bytes,
                cudaMemcpyHostToDevice,
                stream);

        if (status != cudaSuccess)
        {
            cleanup_device();

            throw_cuda_ntt_error(
                "resident quotient upload",
                status);
        }

        record_cuda_h2d(
            total_bytes);

        std::shared_ptr<
            const CudaHostTransferGuard>
            host_transfer;

        std::size_t logical_count = 0;
        std::size_t nonzero_count = 0;

        bool executed = false;

        try
        {
            executed =
                cuda_coset_device_to_coefficients_exact(
                    reinterpret_cast<
                        const mcl::Fr *>(
                        workspace.output),
                    size,
                    shift,
                    host_coefficients,
                    false,
                    device_coefficients,
                    &host_transfer,
                    &logical_count,
                    &nonzero_count);
        }
        catch (...)
        {
            cleanup_device();
            throw;
        }

        if (!executed)
        {
            cleanup_device();
            return false;
        }

        if (logical_count == 0 ||
            logical_count > size ||
            nonzero_count >
                logical_count)
        {
            cleanup_device();

            throw std::runtime_error(
                "CUDA resident coefficient metadata is invalid");
        }

        auto *owned_device_coefficients =
            device_coefficients;

        device_coefficients =
            nullptr;

        std::shared_ptr<void>
            allocation(
                owned_device_coefficients,
                [device](
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
                        device)
                    {
                        if (cudaSetDevice(
                                device) !=
                            cudaSuccess)
                        {
                            (void)cudaGetLastError();
                            return;
                        }
                    }

                    (void)cudaFree(
                        pointer);

                    if (previous_device !=
                        device)
                    {
                        (void)cudaSetDevice(
                            previous_device);
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
                owned_device_coefficients);

        descriptor->capacity =
            size;

        descriptor->logical_count =
            logical_count;

        descriptor->nonzero_count =
            nonzero_count;

        descriptor->device =
            device;

        descriptor->host_transfer =
            std::move(
                host_transfer);

        *resident_coefficients =
            std::move(
                descriptor);

        return true;
    }

    bool cuda_ntt_in_place_exact(
        mcl::Fr *values,
        std::size_t size)
    {
        return run_cuda_ntt_batch(
            values,
            nullptr,
            size,
            1,
            ntt::NTTDir::kForward);
    }

    bool cuda_intt_in_place_exact(
        mcl::Fr *values,
        std::size_t size)
    {
        return run_cuda_ntt_batch(
            values,
            nullptr,
            size,
            1,
            ntt::NTTDir::kInverse);
    }

    bool cuda_ntt_pair_in_place_exact(
        mcl::Fr *first,
        mcl::Fr *second,
        std::size_t size)
    {
        if (first == second)
        {
            throw std::runtime_error(
                "CUDA paired NTT requires distinct input vectors");
        }

        return run_cuda_ntt_batch(
            first,
            second,
            size,
            2,
            ntt::NTTDir::kForward);
    }
    void release_cuda_ntt_workspace()
    {
        cuda_ntt_workspace().release();

        int device = 0;

        auto status =
            cudaGetDevice(&device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "workspace release device lookup",
                status);
        }

        if (device < 0 ||
            static_cast<std::size_t>(device) >=
                device_context::MAX_DEVICES)
        {
            throw std::runtime_error(
                "CUDA quotient NTT workspace release "
                "received an invalid device id");
        }

        {
            std::lock_guard<std::mutex> lock(
                cuda_ntt_domain_mutex());

            auto &initialized =
                cuda_ntt_domain_initialized();

            const auto device_index =
                static_cast<std::size_t>(device);

            if (initialized[device_index])
            {
                // release_domain frees twiddle material shared by every NTT on
                // this device.  All quotient transforms must be complete before
                // this stage boundary is entered.
                status =
                    cudaDeviceSynchronize();

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "domain release pre-synchronization",
                        status);
                }
                // Keep release-domain stream storage local for the same
                // reason as the execution context above: DeviceContext
                // stores its stream by reference.
                cudaStream_t release_stream =
                    nullptr;

                device_context::DeviceContext
                    ctx{
                        release_stream,
                        static_cast<std::size_t>(
                            device_index),
                        cudaMemPool_t{}};

                status =
                    bls12_381_release_domain(
                        ctx);

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "domain release",
                        status);
                }

                status =
                    cudaDeviceSynchronize();

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "domain release synchronization",
                        status);
                }

                initialized[device_index] =
                    false;
            }
        }

        cudaMemPool_t pool = nullptr;

        if (cudaDeviceGetDefaultMemPool(
                &pool,
                device) == cudaSuccess &&
            pool != nullptr)
        {
            (void)cudaMemPoolTrimTo(
                pool,
                0);
        }
    }

} // namespace gatzk::algebra
