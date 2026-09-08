#pragma once

#include <cstddef>
#include <string_view>

namespace gatzk::util
{

    // Text sources are deliberately injectable so the production parser can
    // be exercised against malformed, limited and unlimited cgroup values
    // without changing the process' resource environment.  Empty sources
    // mean that the corresponding controller/file was not available.
    struct EffectiveMemoryTextSources
    {
        std::string_view meminfo;
        // The membership-specific controller files take precedence over the
        // cgroup mount-root files.  The latter are retained as an injectable
        // fallback so parser tests can model a root `max` plus a finite
        // nested controller without consulting the host's cgroup tree.
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
        // `available_bytes` is the effective snapshot value before the
        // reservation/transient/geometry deductions below.
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

    EffectiveMemorySnapshot parse_effective_memory_snapshot(
        const EffectiveMemoryTextSources &sources);

    // Read the host and the current process' cgroup controllers.  This is a
    // low-frequency snapshot operation; callers on hot paths should use the
    // cached variant below.
    EffectiveMemorySnapshot read_effective_memory_snapshot();

    // The cache is process-local and expires periodically, so it cannot carry
    // admission state across independent formal runs in separate processes.
    EffectiveMemorySnapshot cached_effective_memory_snapshot();

    std::size_t recommended_memory_reserve_bytes(
        const EffectiveMemorySnapshot &snapshot,
        std::size_t minimum_bytes,
        std::size_t numerator,
        std::size_t denominator);

    // Unknown effective availability is never converted into a heuristic
    // budget.  Admission returns zero until a trustworthy available value is
    // observed (for example, MemAvailable, a finite cgroup current value, or
    // the bare-metal sysconf available-page fallback).

    // Return the safe budget left after accounting for transient and geometry
    // working sets.  A zero budget is intentional: callers must reduce
    // concurrency or select a bounded/streamed route rather than bypassing
    // admission.
    EffectiveMemoryAdmission admit_effective_memory(
        const EffectiveMemorySnapshot &snapshot,
        std::size_t transient_bytes,
        std::size_t geometry_bytes,
        std::size_t reserve_bytes,
        std::size_t divisor,
        std::size_t maximum_budget_bytes);

    // Admission for folded opening groups.  A group is counted by the actual
    // folded vector, denominator scratch vector and its source working set.
    // Unknown memory or arithmetic overflow never opens an unbounded
    // parallel route: the returned cap is serial (one group at a time).
    OpeningMemoryAdmission admit_opening_memory(
        const EffectiveMemorySnapshot &snapshot,
        const OpeningMemoryWorkset *worksets,
        std::size_t workset_count,
        std::size_t reserve_bytes,
        std::size_t maximum_parallel_groups);

} // namespace gatzk::util
