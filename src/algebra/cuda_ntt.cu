#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <chrono>

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/vector_ops.hpp"

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
            std::size_t{1} << 26;

        // Preserve workspace/headroom after persistent ICICLE twiddles and the
        // explicit input/output buffers have been accounted for.
        constexpr std::size_t kCudaNttReserveBytes =
            std::size_t{4} << 30;

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

            std::size_t capacity_bytes = 0;
            int device = -1;
            cudaStream_t stream = nullptr;

            ~CudaNttWorkspace()
            {
                release();
            }

            void ensure_stream(
                int current_device)
            {
                if (stream != nullptr &&
                    device == current_device)
                {
                    return;
                }

                release();

                device = current_device;

                const auto status =
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

            bool ensure_input_only(
                std::size_t required_bytes,
                int current_device)
            {
                ensure_stream(
                    current_device);

                // Device-output exact transforms write their final NTT result
                // directly into the caller-owned leaf.  They therefore need only
                // one persistent target-sized staging vector.
                if (output != nullptr)
                {
                    auto status =
                        cudaFreeAsync(
                            output,
                            stream);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_ntt_error(
                            "unused output workspace release",
                            status);
                    }

                    output = nullptr;

                    status =
                        cudaStreamSynchronize(
                            stream);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_ntt_error(
                            "unused output workspace release synchronization",
                            status);
                    }
                }

                if (capacity_bytes >=
                        required_bytes &&
                    input != nullptr)
                {
                    return true;
                }

                if (input != nullptr)
                {
                    const auto status =
                        cudaFreeAsync(
                            input,
                            stream);

                    if (status != cudaSuccess)
                    {
                        throw_cuda_ntt_error(
                            "input workspace resize release",
                            status);
                    }

                    input = nullptr;

                    const auto sync_status =
                        cudaStreamSynchronize(
                            stream);

                    if (sync_status !=
                        cudaSuccess)
                    {
                        throw_cuda_ntt_error(
                            "input workspace resize synchronization",
                            sync_status);
                    }
                }

                capacity_bytes = 0;

                auto status =
                    cudaMallocAsync(
                        reinterpret_cast<void **>(
                            &input),
                        required_bytes,
                        stream);

                if (status ==
                    cudaErrorMemoryAllocation)
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

                capacity_bytes =
                    required_bytes;

                return true;
            }

            bool ensure(
                std::size_t required_bytes,
                int current_device)
            {
                ensure_stream(
                    current_device);

                if (capacity_bytes <
                        required_bytes ||
                    input == nullptr)
                {
                    if (input != nullptr)
                    {
                        (void)cudaFreeAsync(
                            input,
                            stream);

                        input = nullptr;
                    }

                    if (output != nullptr)
                    {
                        (void)cudaFreeAsync(
                            output,
                            stream);

                        output = nullptr;
                    }

                    (void)cudaStreamSynchronize(
                        stream);

                    capacity_bytes = 0;

                    auto status =
                        cudaMallocAsync(
                            reinterpret_cast<void **>(
                                &input),
                            required_bytes,
                            stream);

                    if (status ==
                        cudaErrorMemoryAllocation)
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

                    capacity_bytes =
                        required_bytes;
                }

                if (output != nullptr)
                {
                    return true;
                }

                const auto status =
                    cudaMallocAsync(
                        reinterpret_cast<void **>(
                            &output),
                        required_bytes,
                        stream);

                if (status ==
                    cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    output = nullptr;
                    return false;
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "persistent output allocation",
                        status);
                }

                return true;
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

                    (void)cudaStreamSynchronize(
                        stream);

                    (void)cudaStreamDestroy(
                        stream);
                }

                input = nullptr;
                output = nullptr;
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
        prepare_constant_prefix_coset_lanes_kernel(
            GpuFr coefficient,
            std::size_t coefficient_count,
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

            if (begin >= coefficient_count)
            {
                return;
            }

            const auto end =
                min(
                    coefficient_count,
                    begin +
                        kCosetPowerChunkElements);

            auto even_power =
                fr_pow_u64(
                    shift,
                    static_cast<std::uint64_t>(
                        begin));

            auto odd_power =
                fr_pow_u64(
                    odd_shift,
                    static_cast<std::uint64_t>(
                        begin));

            for (std::size_t index = begin;
                 index < end;
                 ++index)
            {
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
        combine_inverse_coset_kernel(
            const GpuFr *even_values,
            const GpuFr *odd_values,
            GpuFr *output,
            std::size_t half_size,
            GpuFr omega_inverse,
            GpuFr inverse_two,
            GpuFr shift_inverse,
            GpuFr shift_inverse_half)
        {
            const auto chunk =
                static_cast<std::size_t>(
                    blockIdx.x) *
                    blockDim.x +
                threadIdx.x;

            const auto begin =
                chunk *
                kCosetPowerChunkElements;

            if (begin >= half_size)
            {
                return;
            }

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

                output[index] =
                    algebra::gpu::fr_mul(
                        first,
                        shift_power);

                output[index + half_size] =
                    algebra::gpu::fr_mul(
                        second,
                        algebra::gpu::fr_mul(
                            shift_power,
                            shift_inverse_half));

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

            auto ctx =
                device_context::
                    get_default_device_context();

            ctx.device_id =
                static_cast<std::size_t>(
                    device);

            ctx.stream = stream;

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

            return true;
        }
        bool run_cuda_ntt_batch(
            mcl::Fr *first,
            mcl::Fr *second,
            std::size_t size,
            std::size_t batch_size,
            ntt::NTTDir direction)
        {
            if (size == 0 ||
                (size & (size - 1U)) != 0U ||
                size > kMaximumCudaNttSize)
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

            int integrated_device = 0;

            status =
                cudaDeviceGetAttribute(
                    &integrated_device,
                    cudaDevAttrIntegrated,
                    device);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "integrated-memory capability query",
                    status);
            }

            if (integrated_device == 0)
            {
                std::size_t free_bytes = 0;
                std::size_t total_device_bytes = 0;

                status =
                    cudaMemGetInfo(
                        &free_bytes,
                        &total_device_bytes);

                if (status != cudaSuccess)
                {
                    throw_cuda_ntt_error(
                        "memory query",
                        status);
                }

                (void)total_device_bytes;

                if (total_bytes >
                    (std::numeric_limits<
                         std::size_t>::max() -
                     kCudaNttReserveBytes) /
                        2)
                {
                    return false;
                }

                const auto required_free_bytes =
                    2 * total_bytes +
                    kCudaNttReserveBytes;

                if (free_bytes <
                    required_free_bytes)
                {
                    return false;
                }
            }

            auto &workspace =
                cuda_ntt_workspace();

            if (!workspace.ensure_input_only(
                    total_bytes,
                    device))
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
            }

            if (!run_cuda_ntt_device(
                    workspace.input,
                    workspace.output,
                    size,
                    batch_size,
                    direction,
                    device,
                    stream))
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

            return true;
        }

    }

    bool cuda_coset_evaluations_to_device_exact(
        const mcl::Fr *values,
        std::size_t value_count,
        std::size_t domain_size,
        const mcl::Fr *zero_domain_blinder,
        std::size_t blinder_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size)
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
                kMaximumCudaNttSize ||
            half_size >
                kMaximumCudaNttSize ||
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
        }

        // The caller-owned leaf is dead until the final forward transform.
        // Reuse its prefix as the source-domain coefficient scratch instead
        // of retaining a second target-sized CUDA workspace.
        if (!run_cuda_ntt_device(
                workspace.input,
                reinterpret_cast<
                    IcicleScalar *>(
                    device_output),
                domain_size,
                1,
                ntt::NTTDir::kInverse,
                device,
                stream))
        {
            return false;
        }

        if (blinder_count != 0)
        {
            status =
                cudaMemcpyAsync(
                    device_output +
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
                    device_output),
                domain_size,
                domain_size,
                reinterpret_cast<
                    const GpuFr *>(
                    device_output +
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

        if (!run_cuda_ntt_device(
                workspace.input,
                reinterpret_cast<
                    IcicleScalar *>(
                    device_output),
                half_size,
                2,
                ntt::NTTDir::kForward,
                device,
                stream))
        {
            return false;
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

        return true;
    }

    bool cuda_coset_coefficients_to_device_exact(
        const mcl::Fr *coefficients,
        std::size_t coefficient_count,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size)
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
                kMaximumCudaNttSize ||
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

        if (!workspace.ensure_input_only(
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
                    device_output,
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
                    device_output),
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

        if (!run_cuda_ntt_device(
                workspace.input,
                reinterpret_cast<
                    IcicleScalar *>(
                    device_output),
                half_size,
                2,
                ntt::NTTDir::kForward,
                device,
                stream))
        {
            return false;
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

        return true;
    }

    bool cuda_coset_lagrange_first_to_device_exact(
        std::size_t domain_size,
        const FieldElement &shift,
        mcl::Fr *device_output,
        std::size_t target_size)
    {
        if (device_output == nullptr ||
            domain_size == 0 ||
            target_size < 2 ||
            (domain_size &
             (domain_size - 1U)) != 0U ||
            (target_size &
             (target_size - 1U)) != 0U ||
            target_size % domain_size != 0)
        {
            return false;
        }

        const auto half_size =
            target_size / 2;

        if (domain_size > half_size ||
            half_size >
                kMaximumCudaNttSize)
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

        if (!workspace.ensure_input_only(
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
                "L0 coset lane clear",
                status);
        }

        const auto root =
            FieldElement::
                root_of_unity(
                    target_size);

        const auto odd_shift =
            shift * root;

        const auto inverse_domain_size =
            FieldElement(
                static_cast<std::uint64_t>(
                    domain_size))
                .inv();

        const auto chunk_count =
            (domain_size +
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
            prepare_constant_prefix_coset_lanes_kernel<<<
                block_count,
                kBlockSize,
                0,
                stream>>>(
                gpu_field(
                    inverse_domain_size),
                domain_size,
                gpu_field(
                    shift),
                gpu_field(
                    odd_shift),
                reinterpret_cast<GpuFr *>(
                    workspace.input),
                reinterpret_cast<GpuFr *>(
                    workspace.input +
                    half_size));

            status =
                cudaGetLastError();

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "L0 coset lane preparation",
                    status);
            }

            record_cuda_kernel_invocations(
                1);
        }

        if (!run_cuda_ntt_device(
                workspace.input,
                reinterpret_cast<
                    IcicleScalar *>(
                    device_output),
                half_size,
                2,
                ntt::NTTDir::kForward,
                device,
                stream))
        {
            return false;
        }

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "L0 coset synchronization",
                status);
        }

        return true;
    }

    bool cuda_coset_device_to_coefficients_exact(
        const mcl::Fr *device_values,
        std::size_t size,
        const FieldElement &shift,
        mcl::Fr *host_coefficients)
    {
        if (device_values == nullptr ||
            host_coefficients == nullptr ||
            size < 2 ||
            (size &
             (size - 1U)) != 0U)
        {
            return false;
        }

        const auto half_size =
            size / 2;

        if (half_size >
            kMaximumCudaNttSize)
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

        if (!workspace.ensure(
                total_bytes,
                device))
        {
            return false;
        }

        const auto stream =
            workspace.stream;

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

        if (!run_cuda_ntt_device(
                workspace.input,
                workspace.output,
                half_size,
                2,
                ntt::NTTDir::kInverse,
                device,
                stream))
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

        combine_inverse_coset_kernel<<<
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
                shift_inverse_half));

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

        status =
            cudaStreamSynchronize(
                stream);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "coefficient synchronization",
                status);
        }

        return true;
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

                auto ctx =
                    device_context::
                        get_default_device_context();

                ctx.device_id =
                    device_index;

                ctx.stream = nullptr;

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