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
#include "gatzk/util/effective_memory.hpp"
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

        // A direct resident batch is only worthwhile while the complete scalar
        // working set remains bounded.  Larger batches keep the resident SRS
        // bases but fall through to the bounded chunked/streamed routes.
        constexpr std::size_t kMaxDiscreteResidentBatchScalarCount =
            1ULL << 24;

        constexpr std::size_t kMaxIntegratedResidentBatchScalarCount =
            1ULL << 25;

        constexpr std::size_t kPreferredStreamMsmChunkSize =
            1ULL << 20;

        constexpr std::size_t kMinimumStreamMsmChunkSize =
            1ULL << 16;
        // ICICLE v2.8.0 direct resident BLS12-381 MSM is kept within the
        // validated <= 2^23 execution range.  Larger MSMs remain on CUDA but
        // are routed through the existing bounded chunked/streamed paths,
        // avoiding one oversized ICICLE bucket-method invocation.
        constexpr std::size_t
            kIcicleV28MaximumDirectResidentMsmCount =
                1ULL << 23;

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

                void copy_mcl_canonical_scalars_to_icicle(
            bls12_381::scalar_t *output,
            const algebra::FieldElement *input,
            std::size_t count)
        {
            if (count == 0)
            {
                return;
            }

            if (output == nullptr || input == nullptr)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM canonical scalar staging received a null buffer");
            }

            for (std::size_t index = 0;
                 index < count;
                 ++index)
            {
                output[index] =
                    to_icicle_field<bls12_381::scalar_t>(
                        input[index].native());
            }
        }

        void copy_icicle_scalars_to_device(
            bls12_381::scalar_t *output,
            const bls12_381::scalar_t *input,
            std::size_t count)
        {
            if (count == 0)
            {
                return;
            }

            if (output == nullptr || input == nullptr)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM device scalar staging received a null buffer");
            }

            const auto status =
                cudaMemcpy(
                    output,
                    input,
                    checked_byte_count(
                        count,
                        sizeof(bls12_381::scalar_t),
                        "device canonical scalar staging"),
                    cudaMemcpyHostToDevice);

            if (status != cudaSuccess)
            {
                throw_cuda_error(
                    "scalar upload",
                    status);
            }
        }

        class ScopedDeviceScalarBuffer
        {
        public:
            explicit ScopedDeviceScalarBuffer(
                std::size_t count)
            {
                if (count == 0)
                {
                    return;
                }

                const auto byte_count =
                    checked_byte_count(
                        count,
                        sizeof(bls12_381::scalar_t),
                        "resident device scalar buffer");

                const auto status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &device_scalars_),
                        byte_count);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM resident device scalar allocation "
                        "exhausted available memory");
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_error(
                        "resident scalar allocation",
                        status);
                }
            }

            ~ScopedDeviceScalarBuffer()
            {
                if (device_scalars_ != nullptr)
                {
                    (void)cudaFree(
                        device_scalars_);
                }
            }

            ScopedDeviceScalarBuffer(
                const ScopedDeviceScalarBuffer &) = delete;

            ScopedDeviceScalarBuffer &
            operator=(
                const ScopedDeviceScalarBuffer &) = delete;

            bls12_381::scalar_t *data()
            {
                return device_scalars_;
            }

        private:
            bls12_381::scalar_t *device_scalars_ =
                nullptr;
        };

        class ReusableScalarStaging
        {
        public:
            ReusableScalarStaging() = default;

            ~ReusableScalarStaging()
            {
                if (device_scalars_ != nullptr)
                {
                    (void)cudaFree(device_scalars_);
                }
            }

            ReusableScalarStaging(
                const ReusableScalarStaging &) = delete;

            ReusableScalarStaging &
            operator=(
                const ReusableScalarStaging &) = delete;
            std::size_t additional_host_bytes(
                std::size_t count) const
            {
                if (count <= host_scalars_.capacity())
                {
                    return 0;
                }

                return checked_byte_count(
                    count,
                    sizeof(bls12_381::scalar_t),
                    "reusable host scalar staging growth");
            }

            std::size_t additional_device_bytes(
                std::size_t count) const
            {
                if (count <= device_capacity_)
                {
                    return 0;
                }

                return checked_byte_count(
                    count - device_capacity_,
                    sizeof(bls12_381::scalar_t),
                    "reusable device scalar staging growth");
            }

            bls12_381::scalar_t *host(
                std::size_t count)
            {
                try
                {
                    host_scalars_.resize(count);
                }
                catch (const std::bad_alloc &)
                {
                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM reusable host scalar staging "
                        "exhausted host memory");
                }

                return host_scalars_.data();
            }

            bls12_381::scalar_t *device(
                std::size_t count)
            {
                if (count <= device_capacity_)
                {
                    return device_scalars_;
                }

                if (device_scalars_ != nullptr)
                {
                    const auto release_status =
                        cudaFree(device_scalars_);

                    device_scalars_ = nullptr;
                    device_capacity_ = 0;

                    if (release_status != cudaSuccess)
                    {
                        throw_cuda_error(
                            "reusable device scalar release",
                            release_status);
                    }
                }

                if (count == 0)
                {
                    return nullptr;
                }

                const auto byte_count =
                    checked_byte_count(
                        count,
                        sizeof(bls12_381::scalar_t),
                        "reusable device scalar staging");

                const auto status =
                    cudaMalloc(
                        reinterpret_cast<void **>(
                            &device_scalars_),
                        byte_count);

                if (status == cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    device_scalars_ = nullptr;
                    device_capacity_ = 0;

                    throw CudaMsmResourcePressure(
                        "CUDA G1 MSM reusable device scalar staging "
                        "exhausted device memory");
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_error(
                        "reusable device scalar allocation",
                        status);
                }

                device_capacity_ = count;
                return device_scalars_;
            }

            void release()
            {
                std::vector<
                    bls12_381::scalar_t>()
                    .swap(host_scalars_);

                if (device_scalars_ != nullptr)
                {
                    const auto status =
                        cudaFree(device_scalars_);

                    device_scalars_ = nullptr;
                    device_capacity_ = 0;

                    if (status != cudaSuccess)
                    {
                        throw_cuda_error(
                            "reusable device scalar release",
                            status);
                    }
                }
            }

        private:
            std::vector<
                bls12_381::scalar_t>
                host_scalars_;

            bls12_381::scalar_t
                *device_scalars_ = nullptr;

            std::size_t device_capacity_ = 0;
        };

        ReusableScalarStaging &
        reusable_scalar_staging()
        {
            static ReusableScalarStaging staging;
            return staging;
        }

        void release_stream_msm_buffers();

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

            void trim_to_bytes(
                std::size_t maximum_bytes)
            {
                while (cached_bytes_ >
                       maximum_bytes)
                {
                    if (entries_.empty())
                    {
                        throw std::runtime_error(
                            "CUDA G1 MSM base-cache trim "
                            "accounting underflow");
                    }

                    auto victim =
                        std::min_element(
                            entries_.begin(),
                            entries_.end(),
                            [](const auto &lhs,
                               const auto &rhs)
                            {
                                return lhs.second.last_use <
                                       rhs.second.last_use;
                            });

                    erase_entry(
                        victim,
                        "base-cache trim");
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

        G1Point to_mcl_point(
            const bls12_381::projective_t &input,
            const char *route,
            std::size_t count,
            int configured_window_bits,
            std::size_t batch_index =
                std::numeric_limits<std::size_t>::max())
        {
            G1Point output;

            if (bls12_381::projective_t::is_zero(input))
            {
                output.value.clear();
                return output;
            }

            if (!bls12_381::projective_t::is_on_curve(input))
            {
                throw std::runtime_error(
                    "CUDA G1 MSM returned an off-curve ICICLE "
                    "projective point: route=" +
                    std::string(route) +
                    " count=" +
                    std::to_string(count) +
                    " configured_c=" +
                    std::to_string(configured_window_bits) +
                    (batch_index ==
                             std::numeric_limits<std::size_t>::max()
                         ? std::string{}
                         : " batch_index=" +
                               std::to_string(batch_index)));
            }

            const auto affine =
                bls12_381::projective_t::to_affine(input);

            if (affine == bls12_381::affine_t::zero())
            {
                output.value.clear();
                return output;
            }

            output.value.x =
                to_mcl_field<mcl::Fp>(affine.x);
            output.value.y =
                to_mcl_field<mcl::Fp>(affine.y);
            output.value.z = 1;

            const bool mcl_valid =
                output.value.isValid();

            const bool subgroup_valid =
                mcl_valid &&
                mcl::bn::isValidOrderBLS12(
                    output.value);

            if (!mcl_valid || !subgroup_valid)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM returned an invalid converted "
                    "point: route=" +
                    std::string(route) +
                    " count=" +
                    std::to_string(count) +
                    " configured_c=" +
                    std::to_string(configured_window_bits) +
                    " mcl_valid=" +
                    std::string(
                        mcl_valid ? "true" : "false") +
                    " subgroup_valid=" +
                    std::string(
                        subgroup_valid
                            ? "true"
                            : "false") +
                    (batch_index ==
                             std::numeric_limits<std::size_t>::max()
                         ? std::string{}
                         : " batch_index=" +
                               std::to_string(batch_index)));
            }

            return output;
        }

        bool resident_cuda_msm_is_safe(
            DeviceBaseCache &cache,
            const G1Point *points,
            std::size_t count,
            std::size_t stable_srs_capacity,
            std::size_t batch_size = 1,
            bool bounded_integrated_scalar_staging = false)
        {
            if (batch_size == 0)
            {
                return false;
            }

            const bool integrated =
                cuda_msm_device_is_integrated();

            // Direct full-width ICICLE resident MSMs above 2^23 have produced
            // invalid projective results on the pinned v2.8.0 backend.
            // Keep the immutable base prefix resident on UMA when possible,
            // but execute the scalar dimension through the bounded chunked
            // route instead of issuing one oversized MSM.
            if (!bounded_integrated_scalar_staging &&
                count >
                    kIcicleV28MaximumDirectResidentMsmCount)
            {
                return false;
            }

            const auto max_resident_count =
                integrated
                    ? kMaxIntegratedResidentCudaMsmCount
                    : kMaxDiscreteResidentCudaMsmCount;

            if (count > max_resident_count)
            {
                return false;
            }

            if (batch_size >
                kMaxResidentCudaMsmBatchSize)
            {
                return false;
            }

            if (count >
                std::numeric_limits<std::size_t>::max() /
                    batch_size)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM batch scalar-count overflow");
            }

            const auto total_scalar_count =
                count * batch_size;

            if (!bounded_integrated_scalar_staging &&
                batch_size > 1)
            {
                const auto max_batch_scalar_count =
                    integrated
                        ? kMaxIntegratedResidentBatchScalarCount
                        : kMaxDiscreteResidentBatchScalarCount;

                if (total_scalar_count >
                    max_batch_scalar_count)
                {
                    return false;
                }
            }

            const auto staged_scalar_count =
                integrated &&
                        bounded_integrated_scalar_staging
                    ? std::min(
                          count,
                          kPreferredStreamMsmChunkSize) *
                          batch_size
                    : total_scalar_count;

            const auto base_bytes =
                checked_byte_count(
                    count,
                    sizeof(bls12_381::affine_t),
                    "resident bases");

            auto &scalar_staging =
                reusable_scalar_staging();

            const auto incremental_host_scalar_bytes =
                scalar_staging
                    .additional_host_bytes(
                        staged_scalar_count);

            const auto incremental_device_scalar_bytes =
                integrated
                    ? std::size_t{0}
                    : scalar_staging
                          .additional_device_bytes(
                              staged_scalar_count);

            const bool already_resident =
                cache.contains_usable_prefix(
                    points,
                    count,
                    stable_srs_capacity);

            const auto new_base_bytes =
                already_resident
                    ? std::size_t{0}
                    : base_bytes;

            if (integrated)
            {
                const auto host =
                    util::cached_effective_memory_snapshot();

                // UMA device memory is the host working set.  A non-zero
                // fallback byte count is not evidence that MemAvailable (or
                // a cgroup limit/current pair) was trustworthy; do not let
                // an unknown host snapshot fail open while cudaMemGetInfo()
                // is unable to distinguish reclaimable DRAM from page cache.
                if (!host.effective_available_known)
                {
                    return false;
                }

                auto required_host =
                    checked_add(
                        new_base_bytes,
                        incremental_host_scalar_bytes,
                        "integrated resident working set");

                const auto reserve =
                    std::max<std::size_t>(
                        kIntegratedHostSafetyReserveBytes,
                        !host.effective_total_known
                            ? kIntegratedHostSafetyReserveBytes
                            : host.effective_total_bytes / 4);

                required_host =
                    checked_add(
                        required_host,
                        reserve,
                        "integrated resident reserve");

                if (host.effective_available_bytes <
                    required_host)
                {
                    return false;
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
                    incremental_device_scalar_bytes,
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
                util::cached_effective_memory_snapshot();

            // Device free bytes above are deliberately checked separately
            // from host staging/reserve admission.  If the host snapshot is
            // unknown, fail closed for the host portion rather than treating
            // device free memory as a proxy for host RAM.
            if (!host.effective_available_known)
            {
                return false;
            }

            {
                const auto reserve =
                    std::max<std::size_t>(
                        kHostSafetyReserveBytes,
                        !host.effective_total_known
                            ? kHostSafetyReserveBytes
                            : host.effective_total_bytes / 8);

                const auto required_host =
                    checked_add(
                        incremental_host_scalar_bytes,
                        reserve,
                        "resident host requirement");

                if (host.effective_available_bytes <
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
            release_stream_msm_buffers();

            auto *device_points =
                cache.get_or_prepare(
                    points,
                    count,
                    stable_srs_capacity);

            auto config =
                msm::default_msm_config();

            config.are_points_on_device = true;
            config.are_points_montgomery_form = false;
            config.are_scalars_montgomery_form = false;
            config.are_results_on_device = false;
            bls12_381::projective_t result;
            cudaError_t status = cudaSuccess;

            auto &scalar_staging =
                reusable_scalar_staging();

            auto *converted_scalars =
                scalar_staging.host(count);

            copy_mcl_canonical_scalars_to_icicle(
                converted_scalars,
                scalars,
                count);

            if (cuda_msm_device_is_integrated())
            {
                config.are_scalars_on_device = false;

                status =
                    bls12_381_msm_cuda(
                        converted_scalars,
                        device_points,
                        static_cast<int>(count),
                        config,
                        &result);
            }
            else
            {
                auto *device_scalars =
                    scalar_staging.device(count);

                copy_icicle_scalars_to_device(
                    device_scalars,
                    converted_scalars,
                    count);

                config.are_scalars_on_device = true;

                status =
                    bls12_381_msm_cuda(
                        device_scalars,
                        device_points,
                        static_cast<int>(count),
                        config,
                        &result);
            }

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
                to_mcl_point(
                    result,
                    "resident",
                    count,
                    config.c);

            algebra::record_cuda_kernel_invocations(1);
        }
        void execute_resident_cuda_msm_batch(
            G1Point *const *outputs,
            const G1Point *points,
            const algebra::FieldElement *const *scalar_vectors,
            const std::size_t *counts,
            std::size_t max_count,
            std::size_t batch_size,
            std::size_t stable_srs_capacity,
            DeviceBaseCache &cache)
        {
            if (counts == nullptr || max_count == 0)
            {
                throw std::runtime_error(
                    "CUDA resident MSM batch received invalid count metadata");
            }

            release_stream_msm_buffers();

            auto *device_points =
                cache.get_or_prepare(
                    points,
                    max_count,
                    stable_srs_capacity);

            if (max_count >
                std::numeric_limits<std::size_t>::max() /
                    batch_size)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM batch scalar-count overflow");
            }

            const auto total_scalar_count =
                max_count * batch_size;

            std::vector<
                bls12_381::projective_t>
                results;

            try
            {
                results.resize(batch_size);
            }
            catch (const std::bad_alloc &)
            {
                throw CudaMsmResourcePressure(
                    "CUDA G1 MSM resident batch result staging "
                    "exhausted host memory");
            }
            auto config =
                msm::default_msm_config();

            config.points_size =
                static_cast<int>(max_count);

            config.batch_size =
                static_cast<int>(batch_size);
            config.are_points_on_device = true;
            config.are_points_montgomery_form = false;
            config.are_scalars_montgomery_form = false;
            config.are_results_on_device = false;

            cudaError_t status = cudaSuccess;

            auto &scalar_staging =
                reusable_scalar_staging();

            auto *batched_scalars =
                scalar_staging.host(
                    total_scalar_count);

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                if (counts[batch] > max_count)
                {
                    throw std::runtime_error(
                        "CUDA resident MSM batch request exceeds max count");
                }

                copy_mcl_canonical_scalars_to_icicle(
                    batched_scalars +
                        batch * max_count,
                    scalar_vectors[batch],
                    counts[batch]);

                std::fill(
                    batched_scalars +
                        batch * max_count +
                        counts[batch],
                    batched_scalars +
                        (batch + 1) * max_count,
                    bls12_381::scalar_t{});
            }

            if (cuda_msm_device_is_integrated())
            {
                config.are_scalars_on_device = false;

                status =
                    bls12_381_msm_cuda(
                        batched_scalars,
                        device_points,
                        static_cast<int>(max_count),
                        config,
                        results.data());
            }
            else
            {
                auto *device_scalars =
                    scalar_staging.device(
                        total_scalar_count);

                copy_icicle_scalars_to_device(
                    device_scalars,
                    batched_scalars,
                    total_scalar_count);

                config.are_scalars_on_device = true;

                status =
                    bls12_381_msm_cuda(
                        device_scalars,
                        device_points,
                        static_cast<int>(max_count),
                        config,
                        results.data());
            }

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
                        results[batch],
                        "resident_batch",
                        max_count,
                        config.c,
                        batch);
            }

            algebra::record_cuda_kernel_invocations(1);
        }
        void execute_cached_base_chunked_cuda_msm_batch(
            G1Point *const *outputs,
            const G1Point *points,
            const algebra::FieldElement *const *scalar_vectors,
            const std::size_t *counts,
            std::size_t max_count,
            std::size_t batch_size,
            std::size_t stable_srs_capacity,
            DeviceBaseCache &cache)
        {
            if (outputs == nullptr ||
                points == nullptr ||
                scalar_vectors == nullptr ||
                batch_size == 0)
            {
                throw std::runtime_error(
                    "CUDA cached-base chunked MSM received invalid input");
            }

            if (counts == nullptr || max_count == 0)
            {
                throw std::runtime_error(
                    "CUDA cached-base chunked MSM received invalid count metadata");
            }

            if (batch_size >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
            {
                throw std::runtime_error(
                    "CUDA cached-base chunked MSM batch size exceeds int");
            }

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                if (outputs[batch] == nullptr ||
                    scalar_vectors[batch] == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA cached-base chunked MSM contains null member");
                }
                if (counts[batch] > max_count)
                {
                    throw std::runtime_error(
                        "CUDA cached-base chunked MSM request exceeds max count");
                }
            }

            auto *device_points =
                cache.get_or_prepare(
                    points,
                    max_count,
                    stable_srs_capacity);

            const auto minimum_chunk =
                std::min(
                    max_count,
                    kMinimumStreamMsmChunkSize);

            auto chunk_size =
                std::min(
                    max_count,
                    kPreferredStreamMsmChunkSize);

            std::vector<bls12_381::scalar_t>
                converted_scalars;

            auto resize_scalar_staging =
                [&]()
            {
                while (true)
                {
                    if (chunk_size >
                        std::numeric_limits<std::size_t>::max() /
                            batch_size)
                    {
                        throw std::runtime_error(
                            "CUDA cached-base chunked MSM scalar-count overflow");
                    }

                    try
                    {
                        converted_scalars.resize(
                            chunk_size *
                            batch_size);

                        return;
                    }
                    catch (const std::bad_alloc &)
                    {
                        if (chunk_size <=
                            minimum_chunk)
                        {
                            throw CudaMsmResourcePressure(
                                "CUDA cached-base chunked MSM cannot "
                                "establish minimum scalar staging");
                        }

                        chunk_size =
                            std::max(
                                minimum_chunk,
                                chunk_size / 2);
                    }
                }
            };

            resize_scalar_staging();

            std::vector<bls12_381::projective_t>
                results;

            std::vector<G1Point>
                totals(
                    batch_size,
                    g1_zero());

            try
            {
                results.resize(
                    batch_size);
            }
            catch (const std::bad_alloc &)
            {
                throw CudaMsmResourcePressure(
                    "CUDA cached-base chunked MSM result staging "
                    "exhausted host memory");
            }

            util::info(
                "cuda_msm route=resident_chunked_batch"
                " max_count=" +
                std::to_string(max_count) +
                " batch=" +
                std::to_string(batch_size) +
                " chunk=" +
                std::to_string(chunk_size));

            std::size_t offset = 0;

            while (offset < max_count)
            {
                const auto current_count =
                    std::min(
                        chunk_size,
                        max_count - offset);

                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    const auto valid_count =
                        offset < counts[batch]
                            ? std::min(
                                  current_count,
                                  counts[batch] - offset)
                            : std::size_t{0};

                    copy_mcl_canonical_scalars_to_icicle(
                        converted_scalars.data() +
                            batch * current_count,
                        valid_count == 0
                            ? nullptr
                            : scalar_vectors[batch] + offset,
                        valid_count);

                    std::fill(
                        converted_scalars.data() +
                            batch * current_count +
                            valid_count,
                        converted_scalars.data() +
                            (batch + 1) * current_count,
                        bls12_381::scalar_t{});
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
                    false;

                config.are_results_on_device =
                    false;

                const auto status =
                    bls12_381_msm_cuda(
                        converted_scalars.data(),
                        device_points + offset,
                        static_cast<int>(
                            current_count),
                        config,
                        results.data());

                if (status ==
                    cudaErrorMemoryAllocation)
                {
                    (void)cudaGetLastError();

                    if (chunk_size <=
                        minimum_chunk)
                    {
                        throw CudaMsmResourcePressure(
                            "CUDA cached-base chunked MSM minimum "
                            "chunk still exceeds available resources");
                    }

                    chunk_size =
                        std::max(
                            minimum_chunk,
                            chunk_size / 2);

                    resize_scalar_staging();

                    util::info(
                        "cuda_msm "
                        "route=resident_chunked_batch_retry"
                        " max_count=" +
                        std::to_string(max_count) +
                        " batch=" +
                        std::to_string(batch_size) +
                        " chunk=" +
                        std::to_string(chunk_size));

                    continue;
                }

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_error(
                        "cached-base chunked batch ICICLE execution",
                        status);
                }

                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    const auto partial =
                        to_mcl_point(
                            results[batch],
                            "resident_chunked_batch",
                            current_count,
                            config.c,
                            batch);

                    mcl::G1::add(
                        totals[batch].value,
                        totals[batch].value,
                        partial.value);
                }

                algebra::record_cuda_kernel_invocations(
                    1);

                offset +=
                    current_count;
            }

            for (std::size_t batch = 0;
                 batch < batch_size;
                 ++batch)
            {
                *outputs[batch] =
                    totals[batch];
            }
        }
        struct StreamMsmBuffers
        {
            bls12_381::affine_t *device_points =
                nullptr;

            std::size_t device_point_capacity = 0;

            std::vector<mcl::G1>
                normalized_points;

            std::vector<
                bls12_381::affine_t>
                converted_points;

            std::vector<
                bls12_381::scalar_t>
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
                    device_point_capacity = 0;

                    if (status != cudaSuccess)
                    {
                        throw_cuda_error(
                            "stream base release",
                            status);
                    }
                }

                std::vector<mcl::G1>()
                    .swap(normalized_points);

                std::vector<
                    bls12_381::affine_t>()
                    .swap(converted_points);

                std::vector<
                    bls12_381::scalar_t>()
                    .swap(converted_scalars);
            }

            bool prepare(
                std::size_t chunk_size,
                std::size_t scalar_batch_size = 1)
            {
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
                    return false;
                }

                if (device_point_capacity >=
                        chunk_size &&
                    device_points != nullptr)
                {
                    return true;
                }

                if (device_points != nullptr)
                {
                    const auto release_status =
                        cudaFree(device_points);

                    device_points = nullptr;
                    device_point_capacity = 0;

                    if (release_status != cudaSuccess)
                    {
                        throw_cuda_error(
                            "stream base resize release",
                            release_status);
                    }
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

                    device_points = nullptr;
                    device_point_capacity = 0;

                    return false;
                }

                if (status != cudaSuccess)
                {
                    throw_cuda_error(
                        "stream base allocation",
                        status);
                }

                device_point_capacity =
                    chunk_size;

                return true;
            }
        };

        StreamMsmBuffers &
        stream_msm_buffers()
        {
            static StreamMsmBuffers buffers;
            return buffers;
        }

        void release_stream_msm_buffers()
        {
            stream_msm_buffers().release();
        }
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

            reusable_scalar_staging().release();

            auto &buffers =
                stream_msm_buffers();

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

                copy_mcl_canonical_scalars_to_icicle(
                    buffers.converted_scalars.data(),
                    scalars + offset,
                    current_count);

                auto config =
                    msm::default_msm_config();

                config.are_points_on_device = true;
                config.are_points_montgomery_form = false;

                config.are_scalars_on_device = false;
                config.are_scalars_montgomery_form = false;

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
                    to_mcl_point(
                        result,
                        "streamed",
                        current_count,
                        config.c);

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
            const std::size_t *counts,
            std::size_t max_count,
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

            if (counts == nullptr || max_count == 0)
            {
                throw std::runtime_error(
                    "CUDA streamed MSM batch received invalid count metadata");
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
                if (counts[batch] > max_count)
                {
                    throw std::runtime_error(
                        "CUDA streamed MSM batch request exceeds max count");
                }
            }

            const auto minimum_chunk =
                std::min(
                    max_count,
                    kMinimumStreamMsmChunkSize);

            auto chunk_size =
                std::min(
                    max_count,
                    kPreferredStreamMsmChunkSize);
            reusable_scalar_staging().release();

            auto &buffers =
                stream_msm_buffers();

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
                " max_count=" +
                std::to_string(max_count) +
                " batch=" +
                std::to_string(batch_size) +
                " chunk=" +
                std::to_string(chunk_size));

            std::vector<G1Point>
                totals(
                    batch_size,
                    g1_zero());

            std::size_t offset = 0;

            while (offset < max_count)
            {
                const auto current_count =
                    std::min(
                        chunk_size,
                        max_count - offset);

                upload_stream_msm_points(
                    buffers,
                    points + offset,
                    current_count);

                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    const auto valid_count =
                        offset < counts[batch]
                            ? std::min(
                                  current_count,
                                  counts[batch] - offset)
                            : std::size_t{0};

                    copy_mcl_canonical_scalars_to_icicle(
                        buffers
                                .converted_scalars
                                .data() +
                            batch *
                                current_count,
                        valid_count == 0
                            ? nullptr
                            : scalar_vectors[batch] + offset,
                        valid_count);

                    std::fill(
                        buffers
                                .converted_scalars
                                .data() +
                            batch * current_count +
                            valid_count,
                        buffers
                                .converted_scalars
                                .data() +
                            (batch + 1) * current_count,
                        bls12_381::scalar_t{});
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
                    false;

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
                        " max_count=" +
                        std::to_string(max_count) +
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
                            results[batch],
                            "streamed_batch",
                            current_count,
                            config.c,
                            batch);

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

    bool cuda_g1_msm_resident_prefix_available(
        const G1Point *points,
        std::size_t count,
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
                "CUDA G1 MSM residency probe requested "
                "without an available CUDA runtime");
        }

        if (count == 0)
        {
            return true;
        }

        if (points == nullptr)
        {
            throw std::runtime_error(
                "CUDA G1 MSM residency probe received "
                "a null SRS pointer");
        }

        if (stable_srs_capacity < count)
        {
            throw std::runtime_error(
                "CUDA G1 MSM residency probe received "
                "an invalid SRS capacity");
        }

        std::lock_guard<std::mutex>
            execution_lock(
                cuda_msm_execution_mutex());

        return DeviceBaseCache::instance()
            .contains_usable_prefix(
                points,
                count,
                stable_srs_capacity);
    }
    void trim_cuda_g1_msm_resident_cache(
        std::size_t maximum_bytes)
    {
        if (!util::route2_options()
                 .cuda_field_hotspots)
        {
            return;
        }

        std::lock_guard<std::mutex>
            execution_lock(
                cuda_msm_execution_mutex());

        DeviceBaseCache::instance()
            .trim_to_bytes(
                maximum_bytes);
    }

    bool cuda_g1_msm_integrated_memory_device()
    {
        if (!util::route2_options()
                 .cuda_field_hotspots)
        {
            return false;
        }

        if (!algebra::cuda_backend_available())
        {
            return false;
        }

        return cuda_msm_device_is_integrated();
    }

    void release_cuda_g1_msm_execution_workspace()
    {
        if (!util::route2_options()
                 .cuda_field_hotspots)
        {
            return;
        }

        std::lock_guard<std::mutex>
            execution_lock(
                cuda_msm_execution_mutex());

        reusable_scalar_staging().release();
        release_stream_msm_buffers();

        util::info(
            "cuda_msm transient execution workspace released; "
            "resident SRS bases preserved");
    }

    void release_cuda_g1_msm_resident_cache()
    {
        if (!util::route2_options()
                 .cuda_field_hotspots)
        {
            return;
        }

        std::lock_guard<std::mutex>
            execution_lock(
                cuda_msm_execution_mutex());

        DeviceBaseCache::instance().clear();
        reusable_scalar_staging().release();
        release_stream_msm_buffers();

        util::info(
            "cuda_msm resident SRS cache and execution workspace released");
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

        const bool integrated =
            cuda_msm_device_is_integrated();

        // Primary route: one complete resident MSM.
        //
        // Resource pressure in scalar staging must not invalidate an already
        // prepared immutable SRS base cache.  Base residency and scalar
        // workspace have independent lifetimes: preserving the former lets
        // the bounded fallback reuse the expensive normalized/uploaded SRS
        // instead of paying that cost a second time.
        if (stable_srs_bases &&
            resident_cuda_msm_is_safe(
                cache,
                points,
                count,
                stable_srs_capacity,
                1,
                false))
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

                if (count >=
                    (1ULL << 20))
                {
                    util::info(
                        "cuda_msm route=resident count=" +
                        std::to_string(count));
                }

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                // The resident base preparation may already have completed.
                // Drop only scalar workspace and preserve immutable SRS
                // residency for the bounded fallback below.
                reusable_scalar_staging().release();
            }
            catch (const std::bad_alloc &)
            {
                reusable_scalar_staging().release();
            }
        }

        // UMA-only fallback: retain the immutable SRS bases while bounding
        // scalar staging to one chunk.  If this path also encounters scalar
        // pressure, keep the resident base cache alive; the final streamed
        // route will evict it only if that memory is genuinely required.
        if (stable_srs_bases &&
            integrated &&
            resident_cuda_msm_is_safe(
                cache,
                points,
                count,
                stable_srs_capacity,
                1,
                true))
        {
            try
            {
                reusable_scalar_staging().release();
                release_stream_msm_buffers();

                G1Point *outputs[1] = {
                    output};

                const algebra::FieldElement *
                    scalar_vectors[1] = {
                        scalars};

                const std::size_t counts[1] = {
                    count};

                execute_cached_base_chunked_cuda_msm_batch(
                    outputs,
                    points,
                    scalar_vectors,
                    counts,
                    count,
                    1,
                    stable_srs_capacity,
                    cache);

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                reusable_scalar_staging().release();
            }
            catch (const std::bad_alloc &)
            {
                reusable_scalar_staging().release();
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
        const std::size_t *counts,
        std::size_t max_count,
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
            counts == nullptr ||
            points == nullptr)
        {
            throw std::runtime_error(
                "CUDA G1 MSM batch received null input");
        }

        if (max_count >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "CUDA G1 MSM batch max count exceeds int");
        }

        if (stable_srs_capacity < max_count)
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

            if (counts[batch] > max_count)
            {
                throw std::runtime_error(
                    "CUDA G1 MSM batch request exceeds max count");
            }
        }

        if (max_count == 0)
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
                counts[0],
                true,
                stable_srs_capacity);
        }

        std::unique_lock<std::mutex>
            execution_lock(
                cuda_msm_execution_mutex());

        auto &cache =
            DeviceBaseCache::instance();
        const bool integrated =
            cuda_msm_device_is_integrated();

        // Route 1: one native ICICLE same-base batch while the complete
        // scalar working set fits the direct resident envelope.
        if (resident_cuda_msm_is_safe(
                cache,
                points,
                max_count,
                stable_srs_capacity,
                batch_size,
                false))
        {
            try
            {
                execute_resident_cuda_msm_batch(
                    outputs,
                    points,
                    scalar_vectors,
                    counts,
                    max_count,
                    batch_size,
                    stable_srs_capacity,
                    cache);

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                // Do not destroy a successfully prepared base prefix merely
                // because the complete batched scalar staging was too large.
                reusable_scalar_staging().release();
            }
            catch (const std::bad_alloc &)
            {
                reusable_scalar_staging().release();
            }
        }

        // Route 2: full-width warm-resident wave.
        //
        // A large same-base batch can exceed the *aggregate* scalar envelope
        // even though every individual MSM fits comfortably.  Direct scalar
        // chunking splits the dimension into 2^20 chunks, turning,
        // for example, a 50M x 2 batch into roughly fifty ICICLE launches.
        //
        // When the SRS prefix is already resident, execute the independent
        // scalar vectors as full-width MSMs one after another.  The expensive
        // base normalization/upload is paid zero additional times and scalar
        // staging is reused between members.
        if (batch_size > 1 &&
            cache.contains_usable_prefix(
                points,
                max_count,
                stable_srs_capacity) &&
            resident_cuda_msm_is_safe(
                cache,
                points,
                max_count,
                stable_srs_capacity,
                1,
                false))
        {
            try
            {
                for (std::size_t batch = 0;
                     batch < batch_size;
                     ++batch)
                {
                    execute_resident_cuda_msm(
                        outputs[batch],
                        points,
                        scalar_vectors[batch],
                        counts[batch],
                        stable_srs_capacity,
                        cache);
                }

                util::info(
                    "cuda_msm route=resident_serial_batch max_count=" +
                    std::to_string(max_count) +
                    " batch=" +
                    std::to_string(batch_size));

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                // Preserve the immutable bases.  Only the scalar workspace
                // needs to be retired before the bounded route is attempted.
                reusable_scalar_staging().release();
            }
            catch (const std::bad_alloc &)
            {
                reusable_scalar_staging().release();
            }
        }

        // Route 3: on UMA, retain the cached SRS and bound only the scalar
        // dimension when a full-width resident wave is not admissible.
        if (integrated &&
            resident_cuda_msm_is_safe(
                cache,
                points,
                max_count,
                stable_srs_capacity,
                batch_size,
                true))
        {
            try
            {
                reusable_scalar_staging().release();
                release_stream_msm_buffers();

                execute_cached_base_chunked_cuda_msm_batch(
                    outputs,
                    points,
                    scalar_vectors,
                    counts,
                    max_count,
                    batch_size,
                    stable_srs_capacity,
                    cache);

                return true;
            }
            catch (const CudaMsmResourcePressure &)
            {
                reusable_scalar_staging().release();
            }
            catch (const std::bad_alloc &)
            {
                reusable_scalar_staging().release();
            }
        }

        // Keep the CUDA execution lock while the shared-base
        // streamed batch is active.  The point vector is uploaded
        // once per chunk for the whole batch.
        try
        {
            execute_streamed_cuda_msm_same_bases_batch(
                outputs,
                points,
                scalar_vectors,
                counts,
                max_count,
                batch_size);

            return true;
        }
        catch (const CudaMsmResourcePressure &)
        {
            cache.clear();

            util::info(
                "cuda_msm streamed_batch "
                "resource pressure: cleared "
                "resident base cache and retrying");

            execute_streamed_cuda_msm_same_bases_batch(
                outputs,
                points,
                scalar_vectors,
                counts,
                max_count,
                batch_size);

            return true;
        }
        catch (const std::bad_alloc &)
        {
            cache.clear();

            util::info(
                "cuda_msm streamed_batch "
                "host staging pressure: cleared "
                "resident base cache and retrying");

            execute_streamed_cuda_msm_same_bases_batch(
                outputs,
                points,
                scalar_vectors,
                counts,
                max_count,
                batch_size);

            return true;
        }
    }

#if defined(GATZK_BUILD_TESTS) && GATZK_BUILD_TESTS
    bool try_cuda_g1_msm_same_bases_batch_for_tests(
        G1Point *const *outputs,
        const G1Point *points,
        const algebra::FieldElement *const *scalar_vectors,
        const std::size_t *counts,
        std::size_t max_count,
        std::size_t batch_size,
        std::size_t stable_srs_capacity,
        CudaG1MsmBatchTestRoute route)
    {
        if (!util::route2_options().cuda_field_hotspots)
        {
            return false;
        }
        if (!algebra::cuda_backend_available())
        {
            throw std::runtime_error(
                "CUDA mixed-count test route requested but CUDA runtime "
                "is unavailable");
        }
        if (batch_size == 0)
        {
            return true;
        }
        if (outputs == nullptr ||
            points == nullptr ||
            scalar_vectors == nullptr ||
            counts == nullptr)
        {
            throw std::runtime_error(
                "CUDA mixed-count test route received null input");
        }
        if (max_count == 0 ||
            max_count >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "CUDA mixed-count test route received invalid max count");
        }
        if (stable_srs_capacity < max_count)
        {
            throw std::runtime_error(
                "CUDA mixed-count test route SRS capacity is too small");
        }
        for (std::size_t batch = 0;
             batch < batch_size;
             ++batch)
        {
            if (outputs[batch] == nullptr ||
                scalar_vectors[batch] == nullptr ||
                counts[batch] == 0 ||
                counts[batch] > max_count)
            {
                throw std::runtime_error(
                    "CUDA mixed-count test route has invalid member count");
            }
        }

        std::unique_lock<std::mutex> execution_lock(
            cuda_msm_execution_mutex());
        auto &cache = DeviceBaseCache::instance();

        switch (route)
        {
        case CudaG1MsmBatchTestRoute::Resident:
            execute_resident_cuda_msm_batch(
                outputs,
                points,
                scalar_vectors,
                counts,
                max_count,
                batch_size,
                stable_srs_capacity,
                cache);
            return true;
        case CudaG1MsmBatchTestRoute::CachedChunked:
            execute_cached_base_chunked_cuda_msm_batch(
                outputs,
                points,
                scalar_vectors,
                counts,
                max_count,
                batch_size,
                stable_srs_capacity,
                cache);
            return true;
        case CudaG1MsmBatchTestRoute::Streamed:
            execute_streamed_cuda_msm_same_bases_batch(
                outputs,
                points,
                scalar_vectors,
                counts,
                max_count,
                batch_size);
            return true;
        }

        throw std::runtime_error(
            "CUDA mixed-count test route is unknown");
    }
#endif
}
