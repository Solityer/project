#include "gatzk/crypto/cuda_msm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

#include "curves/params/bls12_381.cuh"
#include "msm/msm.cuh"

extern "C" cudaError_t bls12_381_msm_cuda(
    const bls12_381::scalar_t *,
    const bls12_381::affine_t *,
    int,
    msm::MSMConfig &,
    bls12_381::projective_t *);

namespace gatzk::crypto
{
    namespace
    {
        constexpr std::size_t kBaseConversionChunkSize = 1ULL << 20;

        // Existing resident CUDA MSM remains the preferred fast path when the
        // complete MSM fits safely in the current resource envelope.
        // Dedicated-memory GPUs retain the existing conservative resident ceiling.
        constexpr std::size_t kMaxDiscreteResidentCudaMsmCount =
            1ULL << 25;

        // Integrated UMA devices can admit a larger resident prefix.  Allocation
        // failure remains fail-closed and falls back to bounded streamed CUDA.
        constexpr std::size_t kMaxIntegratedResidentCudaMsmCount =
            1ULL << 26;

        // ICICLE natively supports several MSMs sharing one point vector.
        constexpr std::size_t kMaxResidentCudaMsmBatchSize = 8;

        // Bound aggregate batch working-set growth.  Large individual MSMs retain
        // the ordinary resident/streamed path rather than forcing an oversized batch.
        constexpr std::size_t kMaxDiscreteResidentBatchScalarCount =
            1ULL << 24;

        constexpr std::size_t kMaxIntegratedResidentBatchScalarCount =
            1ULL << 25;

        constexpr std::size_t kPreferredStreamMsmChunkSize =
            1ULL << 20;

        constexpr std::size_t kMinimumStreamMsmChunkSize =
            1ULL << 16;

        constexpr std::size_t kDeviceBaseCacheLimitBytes =
            24ULL << 30;

        constexpr std::size_t kCudaSafetyReserveBytes =
            8ULL << 30;

        constexpr std::size_t kHostSafetyReserveBytes =
            8ULL << 30;

        constexpr std::size_t kIntegratedHostSafetyReserveBytes =
            16ULL << 30;

        [[noreturn]] void throw_cuda_error(
            const char *operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string("CUDA G1 MSM ") + operation + " failed: " + cudaGetErrorString(status));
        }

        class CudaMsmResourcePressure : public std::runtime_error
        {
        public:
            using std::runtime_error::runtime_error;
        };

        struct HostMemoryInfo
        {
            std::size_t total_bytes = 0;
            std::size_t available_bytes =
                std::numeric_limits<std::size_t>::max();
        };

        HostMemoryInfo read_host_memory_info()
        {
            HostMemoryInfo info;

            std::ifstream input("/proc/meminfo");
            if (!input)
            {
                return info;
            }

            std::string line;
            while (std::getline(input, line))
            {
                std::istringstream row(line);

                std::string key;
                unsigned long long value_kib = 0;
                std::string unit;

                if (!(row >> key >> value_kib))
                {
                    continue;
                }

                row >> unit;

                const auto max_bytes =
                    std::numeric_limits<std::size_t>::max();

                const std::size_t bytes =
                    value_kib > max_bytes / 1024ULL
                        ? max_bytes
                        : static_cast<std::size_t>(value_kib * 1024ULL);

                if (key == "MemTotal:")
                {
                    info.total_bytes = bytes;
                }
                else if (key == "MemAvailable:")
                {
                    info.available_bytes = bytes;
                }
            }

            return info;
        }

        std::size_t checked_byte_count(
            std::size_t count,
            std::size_t element_size,
            const char *label)
        {
            if (count >
                std::numeric_limits<std::size_t>::max() / element_size)
            {
                throw std::runtime_error(
                    std::string("CUDA G1 MSM byte-size overflow for ") + label);
            }

            return count * element_size;
        }

        std::size_t checked_add(
            std::size_t lhs,
            std::size_t rhs,
            const char *label)
        {
            if (lhs >
                std::numeric_limits<std::size_t>::max() - rhs)
            {
                throw std::runtime_error(
                    std::string("CUDA G1 MSM byte-size overflow for ") + label);
            }

            return lhs + rhs;
        }

        bool cuda_msm_device_is_integrated()
        {
            int device = 0;

            auto status =
                cudaGetDevice(&device);

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "device lookup",
                    status);
            }

            int integrated = 0;

            status =
                cudaDeviceGetAttribute(
                    &integrated,
                    cudaDevAttrIntegrated,
                    device);

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "integrated-memory capability query",
                    status);
            }

            return integrated != 0;
        }

        template <class IcicleField, class MclField>
        IcicleField to_icicle_field(const MclField &input)
        {
            std::array<std::uint8_t, 64> bytes{};
            if (input.getLittleEndian(bytes.data(), bytes.size()) == 0)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM failed to export a canonical field value");
            }
            IcicleField output{};
            constexpr std::size_t byte_count =
                sizeof(output.limbs_storage.limbs);
            static_assert(byte_count <= 64);
            std::memcpy(
                output.limbs_storage.limbs,
                bytes.data(),
                byte_count);
            return output;
        }

        void copy_mcl_montgomery_scalars_to_icicle(
            bls12_381::scalar_t *output,
            const algebra::FieldElement *input,
            std::size_t count)
        {
            static_assert(
                sizeof(bls12_381::scalar_t) == sizeof(mcl::Fr),
                "ICICLE BLS12-381 scalar and mcl::Fr must have identical width");

            static_assert(
                sizeof(mcl::Fr) == 4 * sizeof(std::uint64_t),
                "mcl::Fr must use four 64-bit limbs");

            if (count == 0)
            {
                return;
            }

            if (output == nullptr || input == nullptr)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM Montgomery scalar staging received a null buffer");
            }

            const auto *native =
                algebra::FieldElement::native_array_data(input);

            std::memcpy(
                output,
                native,
                checked_byte_count(
                    count,
                    sizeof(mcl::Fr),
                    "Montgomery scalar staging"));
        }

        template <class MclField, class IcicleField>
        MclField to_mcl_field(const IcicleField &input)
        {
            MclField output;
            output.setLittleEndianMod(
                reinterpret_cast<const std::uint8_t *>(
                    input.limbs_storage.limbs),
                sizeof(input.limbs_storage.limbs));
            return output;
        }

        struct BaseStorageFingerprint
        {
            std::size_t capacity = 0;
            std::array<std::string, 3> canonical_anchors;

            bool operator==(const BaseStorageFingerprint &other) const
            {
                return capacity == other.capacity && canonical_anchors == other.canonical_anchors;
            }
        };

        BaseStorageFingerprint fingerprint_storage(
            const G1Point *points,
            std::size_t capacity)
        {
            if (points == nullptr || capacity == 0)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM cannot fingerprint empty SRS storage");
            }
            return BaseStorageFingerprint{
                capacity,
                {
                    points[0].to_string(),
                    points[capacity / 2].to_string(),
                    points[capacity - 1].to_string(),
                },
            };
        }

        struct BaseCacheEntry
        {
            bls12_381::affine_t *device_points = nullptr;
            std::size_t uploaded_count = 0;
            std::size_t byte_count = 0;
            std::uint64_t last_use = 0;
            BaseStorageFingerprint fingerprint;
        };

        class DeviceBaseCache
        {
        public:
            ~DeviceBaseCache()
            {
                for (auto &[storage, entry] : entries_)
                {
                    (void)storage;
                    if (entry.device_points != nullptr)
                    {
                        (void)cudaFree(entry.device_points);
                    }
                }
            }

            DeviceBaseCache(const DeviceBaseCache &) = delete;
            DeviceBaseCache &operator=(const DeviceBaseCache &) = delete;

            static DeviceBaseCache &instance()
            {
                static DeviceBaseCache cache;
                return cache;
            }

            bool contains_usable_prefix(
                const G1Point *points,
                std::size_t count,
                std::size_t storage_capacity)
            {
                const auto found = entries_.find(points);

                if (found == entries_.end())
                {
                    return false;
                }

                if (!(found->second.fingerprint == fingerprint_storage(
                                                       points,
                                                       storage_capacity)))
                {
                    return false;
                }

                return found->second.uploaded_count >= count;
            }

            void clear()
            {
                while (!entries_.empty())
                {
                    erase_entry(
                        entries_.begin(),
                        "base-cache clear");
                }
            }

            // The caller holds the single CUDA-MSM execution lock, so lookup,
            // preparation, eviction, and use cannot race with another MSM.
            bls12_381::affine_t *get_or_prepare(
                const G1Point *points,
                std::size_t count,
                std::size_t storage_capacity)
            {
                const auto fingerprint =
                    fingerprint_storage(points, storage_capacity);
                auto found = entries_.find(points);
                if (found != entries_.end())
                {
                    if (!(found->second.fingerprint == fingerprint))
                    {
                        // A later context reused the same allocator address.
                        // Retire the stale device material before admitting
                        // the new immutable SRS identity.
                        erase_entry(found, "stale base replacement");
                        found = entries_.end();
                    }
                    else if (found->second.uploaded_count >= count)
                    {
                        found->second.last_use = ++use_clock_;
                        return found->second.device_points;
                    }
                    else
                    {
                        // Replace a smaller prefix.  This avoids storing many
                        // overlapping copies for the same powers/Lagrange SRS.
                        erase_entry(found, "base-prefix upgrade");
                        found = entries_.end();
                    }
                }

                const auto byte_count =
                    checked_byte_count(
                        count,
                        sizeof(bls12_381::affine_t),
                        "resident base cache");

                if (byte_count > kDeviceBaseCacheLimitBytes)
                {
                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM immutable SRS slice exceeds "
                        "the resident device base cache");
                }

                evict_to_fit(byte_count);

                bls12_381::affine_t *device_points = nullptr;

                auto status = cudaMalloc(
                    reinterpret_cast<void **>(&device_points),
                    byte_count);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM resident base allocation "
                        "exhausted available memory");
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_error(
                        "base allocation",
                        status);
                }

                try
                {
                    upload_normalized_points(
                        device_points, points, count);
                }
                catch (...)
                {
                    (void)cudaFree(device_points);
                    throw;
                }

                entries_.emplace(
                    points,
                    BaseCacheEntry{
                        device_points,
                        count,
                        byte_count,
                        ++use_clock_,
                        fingerprint,
                    });
                cached_bytes_ += byte_count;
                return device_points;
            }

        private:
            DeviceBaseCache() = default;

            static void upload_normalized_points(
                bls12_381::affine_t *device_points,
                const G1Point *points,
                std::size_t count)
            {
                static_assert(
                    sizeof(G1Point) == sizeof(mcl::G1),
                    "G1Point must wrap exactly one mcl::G1");
                std::vector<mcl::G1> normalized;
                std::vector<bls12_381::affine_t> converted;
                const auto chunk_capacity =
                    std::min(count, kBaseConversionChunkSize);
                normalized.resize(chunk_capacity);
                converted.resize(chunk_capacity);

                for (std::size_t offset = 0; offset < count;
                     offset += chunk_capacity)
                {
                    const auto chunk_size =
                        std::min(chunk_capacity, count - offset);
                    auto *native_points = const_cast<mcl::G1 *>(
                        reinterpret_cast<const mcl::G1 *>(points + offset));
                    mcl::G1::normalizeVec(
                        normalized.data(), native_points, chunk_size);
                    for (std::size_t index = 0; index < chunk_size; ++index)
                    {
                        if (normalized[index].isZero())
                        {
                            converted[index] =
                                bls12_381::affine_t::zero();
                            continue;
                        }
                        converted[index] = {
                            to_icicle_field<bls12_381::point_field_t>(
                                normalized[index].x),
                            to_icicle_field<bls12_381::point_field_t>(
                                normalized[index].y),
                        };
                    }
                    const auto status = cudaMemcpy(
                        device_points + offset,
                        converted.data(),
                        chunk_size * sizeof(converted[0]),
                        cudaMemcpyHostToDevice);
                    if (status != cudaSuccess)
                    {
                        throw_cuda_error("base upload", status);
                    }
                }
            }

            void evict_to_fit(std::size_t incoming_bytes)
            {
                while (cached_bytes_ >
                       kDeviceBaseCacheLimitBytes - incoming_bytes)
                {
                    if (entries_.empty())
                    {
                        throw std::runtime_error(
                            "CUDA G1 MSM base-cache accounting underflow");
                    }
                    auto victim = std::min_element(
                        entries_.begin(), entries_.end(),
                        [](const auto &lhs, const auto &rhs)
                        {
                            return lhs.second.last_use < rhs.second.last_use;
                        });
                    erase_entry(victim, "base-cache eviction");
                }
            }

            using EntryMap = std::unordered_map<
                const G1Point *, BaseCacheEntry>;

            void erase_entry(
                EntryMap::iterator entry,
                const char *operation)
            {
                const auto status = cudaFree(
                    entry->second.device_points);
                if (status != cudaSuccess)
                {
                    throw_cuda_error(operation, status);
                }
                cached_bytes_ -= entry->second.byte_count;
                entries_.erase(entry);
            }

            EntryMap entries_;
            std::size_t cached_bytes_ = 0;
            std::uint64_t use_clock_ = 0;
        };

        std::mutex &cuda_msm_execution_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        G1Point to_mcl_point(const bls12_381::projective_t &input)
        {
            const auto affine =
                bls12_381::projective_t::to_affine(input);
            G1Point output;
            if (affine == bls12_381::affine_t::zero())
            {
                output.value.clear();
                return output;
            }
            output.value.x = to_mcl_field<mcl::Fp>(affine.x);
            output.value.y = to_mcl_field<mcl::Fp>(affine.y);
            output.value.z = 1;
            if (!output.value.isValid() || !mcl::bn::isValidOrderBLS12(output.value))
            {
                throw std::runtime_error(
                    "CUDA G1 MSM returned an invalid or non-subgroup point");
            }
            return output;
        }

        bool resident_cuda_msm_is_safe(
            DeviceBaseCache &cache,
            const G1Point *points,
            std::size_t count,
            std::size_t stable_srs_capacity,
            std::size_t batch_size = 1)
        {
            if (batch_size == 0)
            {
                return false;
            }

            const bool integrated =
                cuda_msm_device_is_integrated();

            const auto max_resident_count =
                integrated
                    ? kMaxIntegratedResidentCudaMsmCount
                    : kMaxDiscreteResidentCudaMsmCount;

            if (count > max_resident_count)
            {
                return false;
            }

            if (batch_size > 1)
            {
                if (batch_size >
                    kMaxResidentCudaMsmBatchSize)
                {
                    return false;
                }

                const auto max_batch_scalar_count =
                    integrated
                        ? kMaxIntegratedResidentBatchScalarCount
                        : kMaxDiscreteResidentBatchScalarCount;

                if (count >
                    max_batch_scalar_count / batch_size)
                {
                    return false;
                }
            }

            if (count >
                std::numeric_limits<std::size_t>::max() / batch_size)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM batch scalar-count overflow");
            }

            const auto total_scalar_count =
                count * batch_size;

            const auto base_bytes =
                checked_byte_count(
                    count,
                    sizeof(bls12_381::affine_t),
                    "resident bases");

            const auto scalar_bytes =
                checked_byte_count(
                    total_scalar_count,
                    sizeof(bls12_381::scalar_t),
                    "resident scalars");

            const bool already_resident =
                cache.contains_usable_prefix(
                    points,
                    count,
                    stable_srs_capacity);

            const auto new_base_bytes =
                already_resident
                    ? std::size_t{0}
                    : base_bytes;

            // On integrated UMA, cudaMemGetInfo() is not used as a hard admission
            // ceiling.  Host MemAvailable plus a large reserve provides the
            // pre-admission guard; actual CUDA/ICICLE allocations remain the
            // authoritative fail-closed resource check.
            if (integrated)
            {
                const auto host =
                    read_host_memory_info();

                if (host.available_bytes !=
                    std::numeric_limits<std::size_t>::max())
                {
                    auto required_host =
                        checked_add(
                            new_base_bytes,
                            scalar_bytes,
                            "integrated resident working set");

                    const auto reserve =
                        std::max<std::size_t>(
                            kIntegratedHostSafetyReserveBytes,
                            host.total_bytes == 0
                                ? kIntegratedHostSafetyReserveBytes
                                : host.total_bytes / 4);

                    required_host =
                        checked_add(
                            required_host,
                            reserve,
                            "integrated resident reserve");

                    if (host.available_bytes <
                        required_host)
                    {
                        return false;
                    }
                }

                return true;
            }

            std::size_t cuda_free = 0;
            std::size_t cuda_total = 0;

            const auto status =
                cudaMemGetInfo(
                    &cuda_free,
                    &cuda_total);

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "memory query",
                    status);
            }

            auto required_cuda =
                checked_add(
                    new_base_bytes,
                    scalar_bytes,
                    "resident CUDA requirement");

            required_cuda =
                checked_add(
                    required_cuda,
                    std::max<std::size_t>(
                        kCudaSafetyReserveBytes,
                        cuda_total / 8),
                    "resident CUDA reserve");

            if (cuda_free < required_cuda)
            {
                return false;
            }

            const auto host =
                read_host_memory_info();

            if (host.available_bytes !=
                std::numeric_limits<std::size_t>::max())
            {
                auto required_host =
                    checked_add(
                        scalar_bytes,
                        std::max<std::size_t>(
                            kHostSafetyReserveBytes,
                            host.total_bytes == 0
                                ? kHostSafetyReserveBytes
                                : host.total_bytes / 8),
                        "resident host reserve");

                if (host.available_bytes <
                    required_host)
                {
                    return false;
                }
            }

            return true;
        }

        void execute_resident_cuda_msm(
            G1Point *output,
            const G1Point *points,
            const algebra::FieldElement *scalars,
            std::size_t count,
            std::size_t stable_srs_capacity,
            DeviceBaseCache &cache)
        {
            auto *device_points =
                cache.get_or_prepare(
                    points,
                    count,
                    stable_srs_capacity);

            std::vector<bls12_381::scalar_t>
                converted_scalars;

            try
            {
                converted_scalars.resize(count);
            }
            catch (const std::bad_alloc &)
            {
                throw CudaMsmResourcePressure(
                    "CUDA G1 MSM resident scalar staging "
                    "exhausted host memory");
            }

            copy_mcl_montgomery_scalars_to_icicle(
                converted_scalars.data(),
                scalars,
                count);

            auto config =
                msm::default_msm_config();

            config.are_points_on_device = true;
            config.are_points_montgomery_form = false;

            config.are_scalars_on_device = false;
            config.are_scalars_montgomery_form = true;

            config.are_results_on_device = false;

            bls12_381::projective_t result;

            const auto status =
                bls12_381_msm_cuda(
                    converted_scalars.data(),
                    device_points,
                    static_cast<int>(count),
                    config,
                    &result);

            if (status == cudaErrorMemoryAllocation)
            {
                (void)cudaGetLastError();

                throw CudaMsmResourcePressure(
                    "CUDA G1 MSM resident ICICLE execution "
                    "exhausted available memory");
            }

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "ICICLE execution",
                    status);
            }

            *output =
                to_mcl_point(result);

            algebra::record_cuda_kernel_invocations(1);
        }

        void execute_resident_cuda_msm_batch(
            G1Point *const *outputs,
            const G1Point *points,
            const algebra::FieldElement *const *scalar_vectors,
            std::size_t count,
            std::size_t batch_size,
            std::size_t stable_srs_capacity,
            DeviceBaseCache &cache)
        {
            auto *device_points =
                cache.get_or_prepare(
                    points,
                    count,
                    stable_srs_capacity);

            if (count >
                std::numeric_limits<std::size_t>::max() / batch_size)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM batch scalar-count overflow");
            }

            const auto total_scalar_count =
                count * batch_size;

            std::vector<bls12_381::scalar_t>
                batched_scalars;

            std::vector<bls12_381::projective_t>
                results;

            try
            {
                batched_scalars.resize(
                    total_scalar_count);

                results.resize(
                    batch_size);
            }
            catch (const std::bad_alloc &)
            {
                throw CudaMsmResourcePressure(
                    "CUDA G1 MSM resident batch staging "
                    "exhausted host memory");
            }

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                copy_mcl_montgomery_scalars_to_icicle(
                    batched_scalars.data() + batch * count,
                    scalar_vectors[batch],
                    count);
            }

            auto config =
                msm::default_msm_config();

            config.points_size =
                static_cast<int>(count);

            config.batch_size =
                static_cast<int>(batch_size);

            config.are_points_on_device = true;
            config.are_points_montgomery_form = false;

            config.are_scalars_on_device = false;
            config.are_scalars_montgomery_form = true;

            config.are_results_on_device = false;

            const auto status =
                bls12_381_msm_cuda(
                    batched_scalars.data(),
                    device_points,
                    static_cast<int>(count),
                    config,
                    results.data());

            if (status == cudaErrorMemoryAllocation)
            {
                (void)cudaGetLastError();

                throw CudaMsmResourcePressure(
                    "CUDA G1 MSM resident batch execution "
                    "exhausted available memory");
            }

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "batched ICICLE execution",
                    status);
            }

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                *outputs[batch] =
                    to_mcl_point(
                        results[batch]);
            }

            algebra::record_cuda_kernel_invocations(1);
        }

        struct StreamMsmBuffers
        {
            bls12_381::affine_t *device_points = nullptr;

            std::vector<mcl::G1> normalized_points;

            std::vector<bls12_381::affine_t>
                converted_points;

            std::vector<bls12_381::scalar_t>
                converted_scalars;

            ~StreamMsmBuffers()
            {
                if (device_points != nullptr)
                {
                    (void)cudaFree(device_points);
                }
            }

            void release()
            {
                if (device_points != nullptr)
                {
                    const auto status =
                        cudaFree(device_points);

                    device_points = nullptr;

                    if (status != cudaSuccess)
                    {
                        throw_cuda_error(
                            "stream base release",
                            status);
                    }
                }

                std::vector<mcl::G1>().swap(
                    normalized_points);

                std::vector<
                    bls12_381::affine_t>()
                    .swap(
                        converted_points);

                std::vector<
                    bls12_381::scalar_t>()
                    .swap(
                        converted_scalars);
            }

            bool prepare(
                std::size_t chunk_size,
                std::size_t scalar_batch_size = 1)
            {
                release();
                if (scalar_batch_size == 0 ||
                    chunk_size >
                        std::numeric_limits<
                            std::size_t>::max() /
                            scalar_batch_size)
                {
                    return false;
                }
                try
                {
                    normalized_points.resize(
                        chunk_size);

                    converted_points.resize(
                        chunk_size);

                    converted_scalars.resize(
                        chunk_size *
                        scalar_batch_size);
                }
                catch (const std::bad_alloc &)
                {
                    release();
                    return false;
                }

                const auto base_bytes =
                    checked_byte_count(
                        chunk_size,
                        sizeof(bls12_381::affine_t),
                        "stream bases");

                const auto status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &device_points),
                        base_bytes);

                if (status ==
                    cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    release();
                    return false;
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_error(
                        "stream base allocation",
                        status);
                }

                return true;
            }
        };

        void upload_stream_msm_points(
            StreamMsmBuffers &buffers,
            const G1Point *points,
            std::size_t count)
        {
            static_assert(
                sizeof(G1Point) == sizeof(mcl::G1),
                "G1Point must wrap exactly one mcl::G1");

            auto *native_points =
                const_cast<mcl::G1 *>(
                    reinterpret_cast<const mcl::G1 *>(
                        points));

            mcl::G1::normalizeVec(
                buffers.normalized_points.data(),
                native_points,
                count);

            for (std::size_t index = 0;
                 index < count;
                 ++index)
            {
                if (buffers.normalized_points[index].isZero())
                {
                    buffers.converted_points[index] =
                        bls12_381::affine_t::zero();

                    continue;
                }

                buffers.converted_points[index] = {
                    to_icicle_field<bls12_381::point_field_t>(
                        buffers.normalized_points[index].x),

                    to_icicle_field<bls12_381::point_field_t>(
                        buffers.normalized_points[index].y),
                };
            }

            const auto status =
                cudaMemcpy(
                    buffers.device_points,
                    buffers.converted_points.data(),
                    count * sizeof(bls12_381::affine_t),
                    cudaMemcpyHostToDevice);

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "stream base upload",
                    status);
            }
        }

        void execute_streamed_cuda_msm(
            G1Point *output,
            const G1Point *points,
            const algebra::FieldElement *scalars,
            std::size_t count)
        {
            if (count == 0)
            {
                *output = g1_zero();
                return;
            }

            const auto minimum_chunk =
                std::min(
                    count,
                    kMinimumStreamMsmChunkSize);

            auto chunk_size =
                std::min(
                    count,
                    kPreferredStreamMsmChunkSize);

            StreamMsmBuffers buffers;

            while (!buffers.prepare(chunk_size))
            {
                if (chunk_size <= minimum_chunk)
                {
                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM cannot establish the minimum "
                        "streamed GPU working set");
                }

                chunk_size =
                    std::max(
                        minimum_chunk,
                        chunk_size / 2);
            }

            util::info(
                "cuda_msm route=streamed count=" + std::to_string(count) + " chunk=" + std::to_string(chunk_size));

            G1Point total = g1_zero();

            std::size_t offset = 0;

            while (offset < count)
            {
                const auto current_count =
                    std::min(
                        chunk_size,
                        count - offset);

                upload_stream_msm_points(
                    buffers,
                    points + offset,
                    current_count);

                copy_mcl_montgomery_scalars_to_icicle(
                    buffers.converted_scalars.data(),
                    scalars + offset,
                    current_count);

                auto config =
                    msm::default_msm_config();

                config.are_points_on_device = true;
                config.are_points_montgomery_form = false;

                config.are_scalars_on_device = false;
                config.are_scalars_montgomery_form = true;

                config.are_results_on_device = false;

                bls12_381::projective_t result;
                const auto status =
                    bls12_381_msm_cuda(
                        buffers.converted_scalars.data(),
                        buffers.device_points,
                        static_cast<int>(current_count),
                        config,
                        &result);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    while (true)
                    {
                        if (chunk_size <= minimum_chunk)
                        {
                            throw CudaMsmResourcePressure(
                                "CUDA G1 MSM minimum streamed chunk "
                                "still exceeds available GPU resources");
                        }

                        chunk_size =
                            std::max(
                                minimum_chunk,
                                chunk_size / 2);

                        if (buffers.prepare(chunk_size))
                        {
                            break;
                        }
                    }

                    util::info(
                        "cuda_msm route=streamed_retry count=" + std::to_string(count) + " chunk=" + std::to_string(chunk_size));

                    // Retry the same mathematical range using the smaller valid buffer.
                    continue;
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_error(
                        "streamed ICICLE execution",
                        status);
                }

                const auto partial =
                    to_mcl_point(result);

                mcl::G1::add(
                    total.value,
                    total.value,
                    partial.value);

                algebra::record_cuda_kernel_invocations(1);

                offset += current_count;
            }

            *output = total;
        }

        void execute_streamed_cuda_msm_same_bases_batch(
            G1Point *const *outputs,
            const G1Point *points,
            const algebra::FieldElement *const *scalar_vectors,
            std::size_t count,
            std::size_t batch_size)
        {
            if (outputs == nullptr ||
                points == nullptr ||
                scalar_vectors == nullptr ||
                batch_size == 0)
            {
                throw std::runtime_error(
                    "CUDA streamed MSM batch "
                    "received invalid input");
            }

            if (batch_size >
                static_cast<std::size_t>(
                    std::numeric_limits<
                        int>::max()))
            {
                throw std::runtime_error(
                    "CUDA streamed MSM batch "
                    "size exceeds int");
            }

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                if (outputs[batch] ==
                        nullptr ||
                    scalar_vectors[batch] ==
                        nullptr)
                {
                    throw std::runtime_error(
                        "CUDA streamed MSM batch "
                        "contains null member");
                }
            }

            if (count == 0)
            {
                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    *outputs[batch] =
                        g1_zero();
                }

                return;
            }

            const auto minimum_chunk =
                std::min(
                    count,
                    kMinimumStreamMsmChunkSize);

            auto chunk_size =
                std::min(
                    count,
                    kPreferredStreamMsmChunkSize);

            StreamMsmBuffers buffers;

            while (!buffers.prepare(
                chunk_size,
                batch_size))
            {
                if (chunk_size <=
                    minimum_chunk)
                {
                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM cannot "
                        "establish minimum "
                        "streamed batch "
                        "working set");
                }

                chunk_size =
                    std::max(
                        minimum_chunk,
                        chunk_size / 2);
            }

            util::info(
                "cuda_msm route=streamed_batch"
                " count=" +
                std::to_string(count) +
                " batch=" +
                std::to_string(batch_size) +
                " chunk=" +
                std::to_string(chunk_size));

            std::vector<G1Point>
                totals(
                    batch_size,
                    g1_zero());

            std::size_t offset = 0;

            while (offset < count)
            {
                const auto current_count =
                    std::min(
                        chunk_size,
                        count - offset);

                upload_stream_msm_points(
                    buffers,
                    points + offset,
                    current_count);

                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    copy_mcl_montgomery_scalars_to_icicle(
                        buffers
                                .converted_scalars
                                .data() +
                            batch *
                                current_count,
                        scalar_vectors[batch] +
                            offset,
                        current_count);
                }

                auto config =
                    msm::default_msm_config();

                config.points_size =
                    static_cast<int>(
                        current_count);

                config.batch_size =
                    static_cast<int>(
                        batch_size);

                config.are_points_on_device =
                    true;

                config.are_points_montgomery_form =
                    false;

                config.are_scalars_on_device =
                    false;

                config.are_scalars_montgomery_form =
                    true;

                config.are_results_on_device =
                    false;

                std::vector<
                    bls12_381::projective_t>
                    results(
                        batch_size);

                const auto status =
                    bls12_381_msm_cuda(
                        buffers
                            .converted_scalars
                            .data(),
                        buffers.device_points,
                        static_cast<int>(
                            current_count),
                        config,
                        results.data());

                if (status ==
                    cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    while (true)
                    {
                        if (chunk_size <=
                            minimum_chunk)
                        {
                            throw CudaMsmResourcePressure(
                                "CUDA G1 MSM "
                                "minimum streamed "
                                "batch chunk still "
                                "exceeds resources");
                        }

                        chunk_size =
                            std::max(
                                minimum_chunk,
                                chunk_size / 2);

                        if (buffers.prepare(
                                chunk_size,
                                batch_size))
                        {
                            break;
                        }
                    }

                    util::info(
                        "cuda_msm "
                        "route=streamed_batch_retry"
                        " count=" +
                        std::to_string(count) +
                        " batch=" +
                        std::to_string(
                            batch_size) +
                        " chunk=" +
                        std::to_string(
                            chunk_size));

                    continue;
                }

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_error(
                        "streamed batch ICICLE "
                        "execution",
                        status);
                }

                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    const auto partial =
                        to_mcl_point(
                            results[batch]);

                    mcl::G1::add(
                        totals[batch].value,
                        totals[batch].value,
                        partial.value);
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);

                offset += current_count;
            }

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                *outputs[batch] =
                    totals[batch];
            }
        }

    } // namespace
    void release_cuda_g1_msm_resident_cache()
    {
        if (!util::route2_options().cuda_field_hotspots)
        {
            return;
        }

        std::lock_guard<std::mutex> execution_lock(
            cuda_msm_execution_mutex());

        DeviceBaseCache::instance().clear();

        util::info(
            "cuda_msm resident base cache released before large quotient");
    }

    bool try_cuda_g1_msm(
        G1Point *output,
        const G1Point *points,
        const algebra::FieldElement *scalars,
        std::size_t count,
        bool stable_srs_bases,
        std::size_t stable_srs_capacity)
    {
        // CPU backend retains the existing canonical CPU MSM.
        if (!util::route2_options().cuda_field_hotspots)
        {
            return false;
        }

        // In cuda_hotspots mode a genuine MSM is GPU-required.
        if (!algebra::cuda_backend_available())
        {
            throw std::runtime_error(
                "CUDA G1 MSM requested but CUDA runtime "
                "is unavailable");
        }

        if (output == nullptr || (count != 0 && (points == nullptr || scalars == nullptr)))
        {
            throw std::runtime_error(
                "CUDA G1 MSM selected with null input");
        }

        if (count == 0)
        {
            *output = g1_zero();
            return true;
        }

        if (stable_srs_bases && stable_srs_capacity < count)
        {
            throw std::runtime_error(
                "CUDA G1 MSM immutable SRS capacity "
                "is smaller than MSM size");
        }

        std::lock_guard<std::mutex> execution_lock(
            cuda_msm_execution_mutex());

        auto &cache =
            DeviceBaseCache::instance();

        if (stable_srs_bases && resident_cuda_msm_is_safe(
                                    cache,
                                    points,
                                    count,
                                    stable_srs_capacity,
                                    1))
        {
            try
            {
                execute_resident_cuda_msm(
                    output,
                    points,
                    scalars,
                    count,
                    stable_srs_capacity,
                    cache);

                if (count >= (1ULL << 20))
                {
                    util::info(
                        "cuda_msm route=resident count=" + std::to_string(count));
                }

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                cache.clear();
            }
            catch (const std::bad_alloc &)
            {
                cache.clear();
            }
        }

        // First try bounded streaming while preserving useful resident SRS
        // material.  Only genuine resource pressure is allowed to evict the
        // resident cache and retry the same CUDA MSM.
        try
        {
            execute_streamed_cuda_msm(
                output,
                points,
                scalars,
                count);

            return true;
        }
        catch (const CudaMsmResourcePressure &)
        {
            cache.clear();

            util::info(
                "cuda_msm streamed resource pressure: "
                "cleared resident base cache and retrying");

            execute_streamed_cuda_msm(
                output,
                points,
                scalars,
                count);

            return true;
        }
    }

    bool try_cuda_g1_msm_same_bases_batch(
        G1Point *const *outputs,
        const G1Point *points,
        const algebra::FieldElement *const *scalar_vectors,
        std::size_t count,
        std::size_t batch_size,
        std::size_t stable_srs_capacity)
    {
        if (!util::route2_options()
                 .cuda_field_hotspots)
        {
            return false;
        }

        if (!algebra::cuda_backend_available())
        {
            throw std::runtime_error(
                "CUDA G1 MSM batch requested but CUDA "
                "runtime is unavailable");
        }

        if (batch_size == 0)
        {
            return true;
        }

        if (outputs == nullptr ||
            scalar_vectors == nullptr ||
            points == nullptr)
        {
            throw std::runtime_error(
                "CUDA G1 MSM batch received null input");
        }

        if (stable_srs_capacity < count)
        {
            throw std::runtime_error(
                "CUDA G1 MSM batch immutable SRS capacity "
                "is smaller than MSM size");
        }

        for (std::size_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            if (outputs[batch] == nullptr ||
                scalar_vectors[batch] == nullptr)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM batch contains null member");
            }
        }

        if (count == 0)
        {
            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                *outputs[batch] =
                    g1_zero();
            }

            return true;
        }

        if (batch_size == 1)
        {
            return try_cuda_g1_msm(
                outputs[0],
                points,
                scalar_vectors[0],
                count,
                true,
                stable_srs_capacity);
        }

        std::unique_lock<std::mutex>
            execution_lock(
                cuda_msm_execution_mutex());

        auto &cache =
            DeviceBaseCache::instance();

        if (resident_cuda_msm_is_safe(
                cache,
                points,
                count,
                stable_srs_capacity,
                batch_size))
        {
            try
            {
                execute_resident_cuda_msm_batch(
                    outputs,
                    points,
                    scalar_vectors,
                    count,
                    batch_size,
                    stable_srs_capacity,
                    cache);

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                cache.clear();
            }
            catch (const std::bad_alloc &)
            {
                cache.clear();
            }
        }

        // A batch that exceeds the safe shared working set is not a protocol
        // failure.  Release the CUDA execution lock and execute the exact same
        // MSMs through the existing resident/streamed CUDA route one by one.
        execute_streamed_cuda_msm_same_bases_batch(
            outputs,
            points,
            scalar_vectors,
            count,
            batch_size);

        return true;
    }
}