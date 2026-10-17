#pragma once

#include <cstdint>
#include <string>

namespace zkdl {

struct CpuHyraxGpuCapacityInfo {
    std::uint64_t cuda_free_bytes = 0U;
    std::uint64_t cuda_total_bytes = 0U;
    std::uint64_t workspace_required_bytes = 0U;
    std::uint64_t authenticated_basis_size = 0U;
    std::uint64_t max_opening_padded_generator_size = 0U;
    int device_ordinal = -1;
    std::string device_name;
    std::string device_uuid;
};

bool query_cpu_hyrax_gpu_capacity(
    std::uint64_t authenticated_basis_size,
    std::uint64_t max_opening_padded_generator_size,
    CpuHyraxGpuCapacityInfo* info, std::string* error = nullptr);
bool query_cpu_hyrax_gpu_device_info(CpuHyraxGpuCapacityInfo* info,
                                     std::string* error = nullptr);

}  // namespace zkdl
