#include "gatzk/crypto/cuda_msm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/util/route2.hpp"

#include "curves/params/bls12_381.cuh"
#include "msm/msm.cuh"

extern "C" cudaError_t bls12_381_msm_cuda(
    const bls12_381::scalar_t*,
    const bls12_381::affine_t*,
    int,
    msm::MSMConfig&,
    bls12_381::projective_t*);

namespace gatzk::crypto
{
    namespace
    {
        constexpr std::size_t kMinCudaMsmSize = 1ULL << 13;
        constexpr std::size_t kBaseConversionChunkSize = 1ULL << 20;
        // The largest formal coefficient slice is 2^27 BLS12-381 affine
        // points (12 GiB).  Keep the cache bounded while allowing that slice
        // plus common FH/edge material on the unified-memory target.
        constexpr std::size_t kDeviceBaseCacheLimitBytes = 24ULL << 30;

        [[noreturn]] void throw_cuda_error(
            const char* operation,
            cudaError_t status)
        {
            throw std::runtime_error(
                std::string("CUDA G1 MSM ") + operation + " failed: "
                + cudaGetErrorString(status));
        }

        template <class IcicleField, class MclField>
        IcicleField to_icicle_field(const MclField& input)
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

        template <class MclField, class IcicleField>
        MclField to_mcl_field(const IcicleField& input)
        {
            MclField output;
            output.setLittleEndianMod(
                reinterpret_cast<const std::uint8_t*>(
                    input.limbs_storage.limbs),
                sizeof(input.limbs_storage.limbs));
            return output;
        }

        struct BaseStorageFingerprint
        {
            std::size_t capacity = 0;
            std::array<std::string, 3> canonical_anchors;

            bool operator==(const BaseStorageFingerprint& other) const
            {
                return capacity == other.capacity
                    && canonical_anchors == other.canonical_anchors;
            }
        };

        BaseStorageFingerprint fingerprint_storage(
            const G1Point* points,
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
            bls12_381::affine_t* device_points = nullptr;
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
                for (auto& [storage, entry] : entries_)
                {
                    (void)storage;
                    if (entry.device_points != nullptr)
                    {
                        (void)cudaFree(entry.device_points);
                    }
                }
            }

            DeviceBaseCache(const DeviceBaseCache&) = delete;
            DeviceBaseCache& operator=(const DeviceBaseCache&) = delete;

            static DeviceBaseCache& instance()
            {
                static DeviceBaseCache cache;
                return cache;
            }

            // The caller holds the single CUDA-MSM execution lock, so lookup,
            // preparation, eviction, and use cannot race with another MSM.
            bls12_381::affine_t* get_or_prepare(
                const G1Point* points,
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

                if (count > std::numeric_limits<std::size_t>::max()
                        / sizeof(bls12_381::affine_t))
                {
                    throw std::runtime_error(
                        "CUDA G1 MSM base-cache size overflow");
                }
                const auto byte_count =
                    count * sizeof(bls12_381::affine_t);
                if (byte_count > kDeviceBaseCacheLimitBytes)
                {
                    throw std::runtime_error(
                        "CUDA G1 MSM immutable SRS slice exceeds the bounded "
                        "device base cache");
                }
                evict_to_fit(byte_count);

                bls12_381::affine_t* device_points = nullptr;
                auto status = cudaMalloc(
                    reinterpret_cast<void**>(&device_points), byte_count);
                if (status != cudaSuccess)
                {
                    throw_cuda_error("base allocation", status);
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
                bls12_381::affine_t* device_points,
                const G1Point* points,
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
                    auto* native_points = const_cast<mcl::G1*>(
                        reinterpret_cast<const mcl::G1*>(points + offset));
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
                        [](const auto& lhs, const auto& rhs) {
                            return lhs.second.last_use
                                < rhs.second.last_use;
                        });
                    erase_entry(victim, "base-cache eviction");
                }
            }

            using EntryMap = std::unordered_map<
                const G1Point*, BaseCacheEntry>;

            void erase_entry(
                EntryMap::iterator entry,
                const char* operation)
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

        std::mutex& cuda_msm_execution_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        G1Point to_mcl_point(const bls12_381::projective_t& input)
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
            if (!output.value.isValid()
                || !mcl::bn::isValidOrderBLS12(output.value))
            {
                throw std::runtime_error(
                    "CUDA G1 MSM returned an invalid or non-subgroup point");
            }
            return output;
        }
    }

    bool try_cuda_g1_msm(
        G1Point* output,
        const G1Point* points,
        const algebra::FieldElement* scalars,
        std::size_t count,
        bool stable_srs_bases,
        std::size_t stable_srs_capacity)
    {
        const bool policy_size_supported =
            count <= kDeviceBaseCacheLimitBytes
                / sizeof(bls12_381::affine_t);
        if (!stable_srs_bases
            || stable_srs_capacity < count
            || count < kMinCudaMsmSize
            || !policy_size_supported
            || !util::route2_options().cuda_field_hotspots
            || !algebra::cuda_backend_available())
        {
            return false;
        }
        if (output == nullptr || points == nullptr || scalars == nullptr)
        {
            throw std::runtime_error(
                "CUDA G1 MSM selected with a null input");
        }
        if (count > static_cast<std::size_t>(
                        std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "CUDA G1 MSM input exceeds the ICICLE size limit");
        }

        std::lock_guard<std::mutex> execution_lock(
            cuda_msm_execution_mutex());
        auto* device_points =
            DeviceBaseCache::instance().get_or_prepare(
                points, count, stable_srs_capacity);

        std::vector<bls12_381::scalar_t> converted_scalars(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            converted_scalars[index] =
                to_icicle_field<bls12_381::scalar_t>(
                    scalars[index].native());
        }

        auto config = msm::default_msm_config();
        config.are_points_on_device = true;
        config.are_points_montgomery_form = false;
        config.are_scalars_on_device = false;
        config.are_scalars_montgomery_form = false;
        config.are_results_on_device = false;
        bls12_381::projective_t result;
        const auto status = bls12_381_msm_cuda(
            converted_scalars.data(),
            device_points,
            static_cast<int>(count),
            config,
            &result);
        if (status != cudaSuccess)
        {
            throw_cuda_error("ICICLE execution", status);
        }
        *output = to_mcl_point(result);
        algebra::record_cuda_kernel_invocations(1);
        return true;
    }
}
