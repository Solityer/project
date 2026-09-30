#pragma once

#include <cstddef>
#include <cstdint>

namespace gatzk::algebra
{
    class FieldElement;
}

namespace gatzk::crypto
{
    struct G1Point;

    bool try_cuda_g1_msm(G1Point *output, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, bool stable_srs_bases,
                         std::size_t stable_srs_capacity, const std::uint64_t *device_montgomery_scalars = nullptr,
                         std::size_t device_montgomery_scalar_capacity = 0, int device_montgomery_scalar_device = -1);

    bool try_cuda_g1_msm_same_bases_batch(G1Point *const *outputs, const G1Point *points, const algebra::FieldElement *const *scalar_vectors,
                                          const std::size_t *counts, std::size_t max_count, std::size_t batch_size, std::size_t stable_srs_capacity,
                                          const std::uint64_t *const *device_montgomery_scalar_vectors = nullptr,
                                          const std::size_t *device_montgomery_scalar_capacities = nullptr,
                                          const int *device_montgomery_scalar_devices = nullptr);

    bool cuda_g1_msm_resident_prefix_available(const G1Point *points, std::size_t count, std::size_t stable_srs_capacity);
    void trim_cuda_g1_msm_resident_cache(std::size_t maximum_bytes);

    bool cuda_g1_msm_integrated_memory_device();

    void release_cuda_g1_msm_execution_workspace();

    void release_cuda_g1_msm_resident_cache();
}
