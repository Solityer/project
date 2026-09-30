#pragma once

#include <cstddef>
#include <string_view>

namespace gatzk::util
{

    struct EffectiveMemoryTextSources
    {
        std::string_view meminfo;
        std::string_view cgroup_v2_limit;
        std::string_view cgroup_v2_current;
        std::string_view cgroup_v1_limit;
        std::string_view cgroup_v1_current;
        std::string_view cgroup_v2_membership_limit;
        std::string_view cgroup_v2_membership_current;
        std::string_view cgroup_v2_root_limit;
        std::string_view cgroup_v2_root_current;
        std::string_view cgroup_v1_membership_limit;
        std::string_view cgroup_v1_membership_current;
        std::string_view cgroup_v1_root_limit;
        std::string_view cgroup_v1_root_current;
    };

    struct EffectiveMemorySnapshot
    {
        std::size_t host_total_bytes = 0;
        std::size_t host_available_bytes = 0;
        std::size_t effective_total_bytes = 0;
        std::size_t effective_available_bytes = 0;
        std::size_t cgroup_limit_bytes = 0;
        std::size_t cgroup_current_bytes = 0;

        bool host_total_known = false;
        bool host_available_known = false;
        bool effective_total_known = false;
        bool effective_available_known = false;
        bool cgroup_v2 = false;
        bool cgroup_v1 = false;
        bool cgroup_limit_known = false;
        bool cgroup_unlimited = false;
        bool cgroup_current_known = false;
        bool cgroup_malformed = false;
        bool conservative = false;
    };

    struct EffectiveMemoryAdmission
    {
        std::size_t available_bytes = 0;
        std::size_t reserve_bytes = 0;
        std::size_t transient_bytes = 0;
        std::size_t geometry_bytes = 0;
        std::size_t budget_bytes = 0;
        bool conservative = false;
    };

    struct OpeningMemoryWorkset
    {
        std::size_t folded_bytes = 0;
        std::size_t scratch_bytes = 0;
        std::size_t source_bytes = 0;
    };

    struct OpeningMemoryAdmission
    {
        std::size_t budget_bytes = 0;
        std::size_t largest_workset_bytes = 0;
        std::size_t maximum_parallel_groups = 1;
        bool memory_known = false;
        bool overflow = false;
        bool conservative = false;
    };

    EffectiveMemorySnapshot parse_effective_memory_snapshot(const EffectiveMemoryTextSources &sources);

    EffectiveMemorySnapshot read_effective_memory_snapshot();

    EffectiveMemorySnapshot cached_effective_memory_snapshot();

    std::size_t recommended_memory_reserve_bytes(const EffectiveMemorySnapshot &snapshot, std::size_t minimum_bytes, std::size_t numerator,
                                                 std::size_t denominator);

    EffectiveMemoryAdmission admit_effective_memory(const EffectiveMemorySnapshot &snapshot, std::size_t transient_bytes, std::size_t geometry_bytes,
                                                    std::size_t reserve_bytes, std::size_t divisor, std::size_t maximum_budget_bytes);

    OpeningMemoryAdmission admit_opening_memory(const EffectiveMemorySnapshot &snapshot, const OpeningMemoryWorkset *worksets, std::size_t workset_count,
                                                std::size_t reserve_bytes, std::size_t maximum_parallel_groups);

}
