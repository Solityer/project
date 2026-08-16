#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include "gatzk/algebra/vector_ops.hpp"

#include "fields/snark_fields/bls12_381_scalar.cuh"
#include "gpu-utils/device_context.cuh"
#include "ntt/ntt.cuh"

extern "C" cudaError_t bls12_381_initialize_domain(
    bls12_381::scalar_t *primitive_root,
    device_context::DeviceContext &ctx,
    bool fast_twiddles_mode);

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

        std::mutex &cuda_ntt_mutex()
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
            if (device < 0 || static_cast<std::size_t>(device) >= device_context::MAX_DEVICES)
            {
                throw std::runtime_error(
                    "CUDA quotient NTT received an invalid device id");
            }

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

            initialized[static_cast<std::size_t>(device)] =
                true;
        }
        struct CudaNttWorkspace
        {
            IcicleScalar *input = nullptr;
            IcicleScalar *output = nullptr;

            std::size_t capacity_bytes = 0;
            int device = -1;

            bool ensure(
                std::size_t required_bytes,
                int current_device,
                cudaStream_t stream)
            {
                if (device == current_device &&
                    capacity_bytes >= required_bytes &&
                    input != nullptr &&
                    output != nullptr)
                {
                    return true;
                }

                release(stream);

                device = current_device;

                auto status =
                    cudaMallocAsync(
                        reinterpret_cast<void **>(
                            &input),
                        required_bytes,
                        stream);

                if (status != cudaSuccess)
                {
                    input = nullptr;
                    output = nullptr;
                    capacity_bytes = 0;

                    if (status ==
                        cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();
                        return false;
                    }

                    throw_cuda_ntt_error(
                        "persistent input allocation",
                        status);
                }

                status =
                    cudaMallocAsync(
                        reinterpret_cast<void **>(
                            &output),
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
                    capacity_bytes = 0;

                    if (status ==
                        cudaErrorMemoryAllocation)
                    {
                        (void)cudaGetLastError();
                        return false;
                    }

                    throw_cuda_ntt_error(
                        "persistent output allocation",
                        status);
                }

                capacity_bytes =
                    required_bytes;

                return true;
            }

            void release(
                cudaStream_t stream) noexcept
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

                input = nullptr;
                output = nullptr;
                capacity_bytes = 0;
                device = -1;
            }
        };

        CudaNttWorkspace &
        cuda_ntt_workspace()
        {
            static CudaNttWorkspace workspace;
            return workspace;
        }

        bool run_cuda_ntt_batch(
            mcl::Fr *first,
            mcl::Fr *second,
            std::size_t size,
            std::size_t batch_size,
            ntt::NTTDir direction)
        {
            if (size == 0 || (size & (size - 1U)) != 0U || size > kMaximumCudaNttSize)
            {
                return false;
            }

            if (first == nullptr || (batch_size == 2 && second == nullptr))
            {
                throw std::runtime_error(
                    "CUDA quotient NTT received a null input");
            }

            if (batch_size != 1 && batch_size != 2)
            {
                throw std::runtime_error(
                    "CUDA quotient NTT supports only batch size 1 or 2");
            }

            if (size >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
            {
                return false;
            }

            std::lock_guard<std::mutex> lock(
                cuda_ntt_mutex());

            int device = 0;

            auto status =
                cudaGetDevice(&device);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "device lookup",
                    status);
            }

            auto ctx =
                device_context::get_default_device_context();

            ctx.device_id =
                static_cast<std::size_t>(device);

            ensure_cuda_ntt_domain(
                device,
                ctx);

            const auto one_vector_bytes =
                checked_ntt_bytes(size, 1);

            const auto total_bytes =
                checked_ntt_bytes(
                    size,
                    batch_size);

            int integrated_device = 0;

            status = cudaDeviceGetAttribute(
                &integrated_device,
                cudaDevAttrIntegrated,
                device);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "integrated-memory capability query",
                    status);
            }

            // On discrete GPUs, cudaMemGetInfo() describes the dedicated device-memory
            // pool closely enough for conservative admission.
            //
            // DGX Spark / GB10 is an integrated UMA platform.  NVIDIA documents that
            // cudaMemGetInfo() does not include DRAM that the CPU can reclaim from page
            // cache / swap, so treating it as a hard allocation ceiling can reject valid
            // allocations.  On UMA, let the actual CUDA allocations below be the
            // authoritative fail-closed resource check.
            if (integrated_device == 0)
            {
                std::size_t free_bytes = 0;
                std::size_t total_device_bytes = 0;

                status = cudaMemGetInfo(
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
                    (std::numeric_limits<std::size_t>::max() - kCudaNttReserveBytes) / 2)
                {
                    return false;
                }

                const auto required_free_bytes =
                    2 * total_bytes + kCudaNttReserveBytes;

                if (free_bytes < required_free_bytes)
                {
                    return false;
                }
            }

            IcicleScalar *device_input = nullptr;
            IcicleScalar *device_output = nullptr;

            status = cudaMallocAsync(
                reinterpret_cast<void **>(&device_input),
                total_bytes,
                ctx.stream);

            if (status != cudaSuccess)
            {
                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();
                    return false;
                }

                throw_cuda_ntt_error(
                    "input allocation",
                    status);
            }

            status = cudaMallocAsync(
                reinterpret_cast<void **>(&device_output),
                total_bytes,
                ctx.stream);

            if (status != cudaSuccess)
            {
                cleanup_device_buffers(
                    device_input,
                    nullptr,
                    ctx.stream);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();
                    return false;
                }

                throw_cuda_ntt_error(
                    "output allocation",
                    status);
            }

            status = cudaMemcpyAsync(
                device_input,
                first,
                one_vector_bytes,
                cudaMemcpyHostToDevice,
                ctx.stream);

            if (status != cudaSuccess)
            {
                cleanup_device_buffers(
                    device_input,
                    device_output,
                    ctx.stream);

                throw_cuda_ntt_error(
                    "first input upload",
                    status);
            }

            if (batch_size == 2)
            {
                status = cudaMemcpyAsync(
                    device_input + size,
                    second,
                    one_vector_bytes,
                    cudaMemcpyHostToDevice,
                    ctx.stream);

                if (status != cudaSuccess)
                {
                    cleanup_device_buffers(
                        device_input,
                        device_output,
                        ctx.stream);

                    throw_cuda_ntt_error(
                        "second input upload",
                        status);
                }
            }

            auto config =
                ntt::default_ntt_config<IcicleScalar>(
                    ctx);

            config.batch_size =
                static_cast<int>(batch_size);
            config.columns_batch = false;
            config.ordering =
                ntt::Ordering::kNN;
            config.are_inputs_on_device = true;
            config.are_outputs_on_device = true;
            config.is_async = false;
            config.ntt_algorithm =
                ntt::NttAlgorithm::Auto;

            status = bls12_381_ntt_cuda(
                device_input,
                static_cast<int>(size),
                direction,
                config,
                device_output);

            if (status != cudaSuccess)
            {
                cleanup_device_buffers(
                    device_input,
                    device_output,
                    ctx.stream);

                throw_cuda_ntt_error(
                    "execution",
                    status);
            }

            status = cudaMemcpyAsync(
                first,
                device_output,
                one_vector_bytes,
                cudaMemcpyDeviceToHost,
                ctx.stream);

            if (status != cudaSuccess)
            {
                cleanup_device_buffers(
                    device_input,
                    device_output,
                    ctx.stream);

                throw_cuda_ntt_error(
                    "first output download",
                    status);
            }

            if (batch_size == 2)
            {
                status = cudaMemcpyAsync(
                    second,
                    device_output + size,
                    one_vector_bytes,
                    cudaMemcpyDeviceToHost,
                    ctx.stream);

                if (status != cudaSuccess)
                {
                    cleanup_device_buffers(
                        device_input,
                        device_output,
                        ctx.stream);

                    throw_cuda_ntt_error(
                        "second output download",
                        status);
                }
            }

            status = cudaFreeAsync(
                device_input,
                ctx.stream);

            if (status != cudaSuccess)
            {
                cleanup_device_buffers(
                    nullptr,
                    device_output,
                    ctx.stream);

                throw_cuda_ntt_error(
                    "input release",
                    status);
            }

            device_input = nullptr;

            status = cudaFreeAsync(
                device_output,
                ctx.stream);

            if (status != cudaSuccess)
            {
                cleanup_device_buffers(
                    nullptr,
                    device_output,
                    ctx.stream);

                throw_cuda_ntt_error(
                    "output release",
                    status);
            }

            device_output = nullptr;

            status = cudaStreamSynchronize(
                ctx.stream);

            if (status != cudaSuccess)
            {
                throw_cuda_ntt_error(
                    "synchronization",
                    status);
            }

            return true;
        }
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
        std::lock_guard<std::mutex> lock(
            cuda_ntt_mutex());

        int device = 0;

        auto status =
            cudaGetDevice(&device);

        if (status != cudaSuccess)
        {
            throw_cuda_ntt_error(
                "workspace release device lookup",
                status);
        }

        auto ctx =
            device_context::
                get_default_device_context();

        ctx.device_id =
            static_cast<std::size_t>(
                device);

        cuda_ntt_workspace().release(
            ctx.stream);

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