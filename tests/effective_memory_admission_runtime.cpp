#include "gatzk/util/effective_memory.hpp"
#include "gatzk/util/spill_arena_accounting.hpp"
#include "gatzk/algebra/field.hpp"

#include <array>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    using gatzk::util::EffectiveMemoryTextSources;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename Fn>
    void expect_failure(Fn &&fn, const std::string &message)
    {
        bool failed = false;
        try
        {
            fn();
        }
        catch (const std::exception &)
        {
            failed = true;
        }
        expect(failed, message);
    }
}

int main()
{
    try
    {
        constexpr std::string_view host_meminfo =
            "MemTotal:       33554432 kB\n"
            "MemAvailable:   25165824 kB\n";

        const auto unrestricted =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                });
        expect(
            unrestricted.host_total_known &&
                unrestricted.host_available_known &&
                unrestricted.effective_total_bytes == (std::size_t{32} << 30),
            "unrestricted host snapshot was not parsed");
        expect(
            !unrestricted.cgroup_limit_known &&
                unrestricted.effective_available_bytes ==
                    (std::size_t{24} << 30),
            "unrestricted bare-metal snapshot was incorrectly constrained");

        const auto host_total_without_available =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo =
                        "MemTotal:       33554432 kB\n",
                });
        expect(
            host_total_without_available.host_total_known &&
                !host_total_without_available.host_available_known &&
                !host_total_without_available.effective_available_known &&
                host_total_without_available.effective_available_bytes == 0 &&
                host_total_without_available.conservative,
            "MemTotal without MemAvailable synthesized an unsafe budget");
        const auto missing_available_admission =
            gatzk::util::admit_effective_memory(
                host_total_without_available,
                0,
                0,
                0,
                1,
                std::numeric_limits<std::size_t>::max());
        expect(
            missing_available_admission.available_bytes == 0 &&
                missing_available_admission.budget_bytes == 0,
            "unknown host availability was admitted from total/8");

        const auto missing_meminfo_and_cgroup =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{});
        expect(
            !missing_meminfo_and_cgroup.host_total_known &&
                !missing_meminfo_and_cgroup.host_available_known &&
                !missing_meminfo_and_cgroup.cgroup_v2 &&
                !missing_meminfo_and_cgroup.cgroup_v1 &&
                !missing_meminfo_and_cgroup.effective_available_known &&
                missing_meminfo_and_cgroup.effective_available_bytes == 0,
            "missing meminfo and cgroup did not fail closed");

        const auto missing_meminfo_with_finite_cgroup =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .cgroup_v2_limit = std::to_string(std::size_t{16} << 30),
                    .cgroup_v2_current = std::to_string(std::size_t{4} << 30),
                });
        expect(
            missing_meminfo_with_finite_cgroup.cgroup_limit_known &&
                missing_meminfo_with_finite_cgroup.cgroup_current_known &&
                missing_meminfo_with_finite_cgroup.effective_available_known &&
                missing_meminfo_with_finite_cgroup.effective_available_bytes ==
                    (std::size_t{12} << 30),
            "known finite cgroup availability was lost with missing meminfo");

        const auto live_snapshot =
            gatzk::util::read_effective_memory_snapshot();
        expect(
            live_snapshot.effective_total_known ||
                live_snapshot.conservative,
            "live effective-memory snapshot was neither known nor conservative");

        const auto limited_v2 =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = std::to_string(std::size_t{16} << 30),
                    .cgroup_v2_current = std::to_string(std::size_t{4} << 30),
                });
        expect(
            limited_v2.cgroup_v2 &&
                limited_v2.cgroup_limit_known &&
                limited_v2.cgroup_current_known &&
                limited_v2.effective_total_bytes == (std::size_t{16} << 30) &&
                limited_v2.effective_available_bytes == (std::size_t{12} << 30),
            "finite cgroup v2 limit/current was not applied");

        const auto nested_v2_overrides_root =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_membership_limit = std::to_string(std::size_t{16} << 30),
                    .cgroup_v2_membership_current = std::to_string(std::size_t{12} << 30),
                    .cgroup_v2_root_limit = "max",
                    .cgroup_v2_root_current = "0",
                });
        expect(
            nested_v2_overrides_root.cgroup_v2 &&
                nested_v2_overrides_root.cgroup_limit_known &&
                nested_v2_overrides_root.cgroup_limit_bytes ==
                    (std::size_t{16} << 30) &&
                nested_v2_overrides_root.effective_available_bytes ==
                    (std::size_t{4} << 30),
            "nested v2 membership did not override an unlimited root");

        const auto nested_v1_overrides_root =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v1_membership_limit = std::to_string(std::size_t{16} << 30),
                    .cgroup_v1_membership_current = std::to_string(std::size_t{8} << 30),
                    .cgroup_v1_root_limit = "9223372036854771712",
                    .cgroup_v1_root_current = "0",
                });
        expect(
            nested_v1_overrides_root.cgroup_v1 &&
                nested_v1_overrides_root.cgroup_limit_known &&
                nested_v1_overrides_root.cgroup_limit_bytes ==
                    (std::size_t{16} << 30) &&
                nested_v1_overrides_root.effective_available_bytes ==
                    (std::size_t{8} << 30),
            "nested v1 membership did not override an unlimited root");

        const auto unlimited_v1 =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v1_limit = "9223372036854771712",
                    .cgroup_v1_current = "1048576",
                });
        expect(
            unlimited_v1.cgroup_v1 &&
                unlimited_v1.cgroup_unlimited &&
                unlimited_v1.effective_available_bytes ==
                    unrestricted.effective_available_bytes,
            "v1 huge unlimited sentinel changed bare-metal admission");

        const auto malformed =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = "not-a-limit",
                    .cgroup_v2_current = "1048576",
                });
        expect(
            malformed.cgroup_malformed && malformed.conservative &&
                !malformed.effective_available_known &&
                malformed.effective_available_bytes == 0 &&
                gatzk::util::admit_effective_memory(
                    malformed,
                    0,
                    0,
                    0,
                    1,
                    std::numeric_limits<std::size_t>::max())
                        .budget_bytes == 0,
            "malformed cgroup data was not rejected conservatively");

        const auto overflow_controller =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = "18446744073709551616",
                    .cgroup_v2_current = "1",
                });
        expect(
            overflow_controller.cgroup_malformed &&
                overflow_controller.conservative,
            "overflowed cgroup controller value was admitted");

        const auto whitespace_controller =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = " \t\n",
                    .cgroup_v2_current = " \t\n",
                });
        expect(
            whitespace_controller.cgroup_malformed &&
                whitespace_controller.conservative &&
                !whitespace_controller.effective_available_known &&
                whitespace_controller.effective_available_bytes == 0,
            "whitespace cgroup controller value was not rejected");

        const auto empty_marker_controller =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = "__gatzk_empty_cgroup_value__",
                    .cgroup_v2_current = "__gatzk_empty_cgroup_value__",
                });
        expect(
            empty_marker_controller.cgroup_malformed &&
                empty_marker_controller.conservative,
            "empty cgroup controller marker was treated as unrestricted");

        expect(
            gatzk::util::recommended_memory_reserve_bytes(
                unrestricted,
                std::size_t{1} << 20,
                0,
                4) ==
                (std::size_t{1} << 20),
            "zero reserve numerator caused invalid arithmetic");
        expect_failure(
            [&]()
            {
                (void)gatzk::util::recommended_memory_reserve_bytes(
                    unrestricted,
                    std::size_t{1} << 20,
                    1,
                    0);
            },
            "zero reserve denominator was not rejected safely");

        gatzk::util::EffectiveMemorySnapshot unknown_memory{};
        unknown_memory.effective_available_bytes = std::size_t{64} << 30;
        expect(
            gatzk::util::admit_effective_memory(
                unknown_memory,
                0,
                0,
                0,
                1,
                std::numeric_limits<std::size_t>::max())
                .budget_bytes == 0 &&
                gatzk::util::admit_effective_memory(
                    unknown_memory,
                    0,
                    0,
                    0,
                    1,
                    std::numeric_limits<std::size_t>::max())
                    .available_bytes == 0,
            "unknown effective memory was admitted from a stale byte value");

        const auto unrestricted_admission =
            gatzk::util::admit_effective_memory(
                unrestricted,
                std::size_t{1} << 30,
                std::size_t{1} << 30,
                std::size_t{4} << 30,
                1,
                std::numeric_limits<std::size_t>::max());
        const auto limited_admission =
            gatzk::util::admit_effective_memory(
                limited_v2,
                std::size_t{1} << 30,
                std::size_t{1} << 30,
                std::size_t{4} << 30,
                1,
                std::numeric_limits<std::size_t>::max());
        expect(
            unrestricted_admission.budget_bytes == (std::size_t{18} << 30),
            "unrestricted admission arithmetic changed");
        expect(
            limited_admission.budget_bytes == (std::size_t{6} << 30) &&
                limited_admission.budget_bytes <
                    unrestricted_admission.budget_bytes,
            "finite cgroup admission did not lower the safe budget");

        const std::array<gatzk::util::OpeningMemoryWorkset, 4>
            opening_worksets{
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = std::size_t{1} << 30,
                    .scratch_bytes = std::size_t{1} << 30,
                    .source_bytes = std::size_t{1} << 30},
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = std::size_t{1} << 30,
                    .scratch_bytes = std::size_t{1} << 30,
                    .source_bytes = std::size_t{1} << 30},
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = std::size_t{1} << 30,
                    .scratch_bytes = std::size_t{1} << 30,
                    .source_bytes = std::size_t{1} << 30},
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = std::size_t{1} << 30,
                    .scratch_bytes = std::size_t{1} << 30,
                    .source_bytes = std::size_t{1} << 30},
            };
        const auto high_opening_admission =
            gatzk::util::admit_opening_memory(
                unrestricted,
                opening_worksets.data(),
                opening_worksets.size(),
                std::size_t{1} << 30,
                opening_worksets.size());
        expect(
            high_opening_admission.maximum_parallel_groups == 4 &&
                high_opening_admission.largest_workset_bytes ==
                    (std::size_t{3} << 30),
            "high-budget opening admission did not retain safe parallelism");

        const auto low_memory =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = std::to_string(std::size_t{4} << 30),
                    .cgroup_v2_current = std::to_string(std::size_t{3} << 30),
                });
        const auto low_opening_admission =
            gatzk::util::admit_opening_memory(
                low_memory,
                opening_worksets.data(),
                opening_worksets.size(),
                std::size_t{1} << 30,
                opening_worksets.size());
        expect(
            low_opening_admission.maximum_parallel_groups == 1 &&
                low_opening_admission.budget_bytes <
                    high_opening_admission.budget_bytes,
            "low-budget opening admission did not serialize groups");

        const auto field_bytes = sizeof(gatzk::algebra::FieldElement);
        const auto ppi_domain_bytes =
            (std::size_t{1} << 23) * field_bytes;
        const auto ogbn_domain_bytes =
            (std::size_t{1} << 24) * field_bytes;
        const std::array<gatzk::util::OpeningMemoryWorkset, 2>
            large_dataset_worksets{
                // PPI-scale edge-domain folded + denominator scratch.
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = ppi_domain_bytes,
                    .scratch_bytes = ppi_domain_bytes,
                    .source_bytes = (std::size_t{1} << 20) * field_bytes},
                // OGBN-Arxiv-scale edge-domain folded + denominator scratch.
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = ogbn_domain_bytes,
                    .scratch_bytes = ogbn_domain_bytes,
                    .source_bytes = (std::size_t{1} << 20) * field_bytes},
            };
        const auto large_dataset_high_admission =
            gatzk::util::admit_opening_memory(
                unrestricted,
                large_dataset_worksets.data(),
                large_dataset_worksets.size(),
                std::size_t{1} << 30,
                large_dataset_worksets.size());
        expect(
            large_dataset_high_admission.largest_workset_bytes ==
                    2 * ogbn_domain_bytes +
                        ((std::size_t{1} << 20) * field_bytes) &&
                large_dataset_high_admission.maximum_parallel_groups >= 2,
            "PPI/OGBN opening-scale estimate lost safe high-budget overlap");
        const auto large_dataset_low_admission =
            gatzk::util::admit_opening_memory(
                low_memory,
                large_dataset_worksets.data(),
                large_dataset_worksets.size(),
                std::size_t{1} << 30,
                large_dataset_worksets.size());
        expect(
            large_dataset_low_admission.maximum_parallel_groups == 1 &&
                large_dataset_low_admission.largest_workset_bytes ==
                    large_dataset_high_admission.largest_workset_bytes,
            "PPI/OGBN opening-scale estimate was not fail-closed under pressure");

        expect(
            gatzk::util::admit_opening_memory(
                unknown_memory,
                opening_worksets.data(),
                opening_worksets.size(),
                0,
                opening_worksets.size())
                    .maximum_parallel_groups == 1,
            "unknown opening memory was not fail-closed");

        const std::array<gatzk::util::OpeningMemoryWorkset, 1>
            overflowing_workset{
                gatzk::util::OpeningMemoryWorkset{
                    .folded_bytes = std::numeric_limits<std::size_t>::max(),
                    .scratch_bytes = 1,
                    .source_bytes = 1}};
        const auto overflowing_opening_admission =
            gatzk::util::admit_opening_memory(
                unrestricted,
                overflowing_workset.data(),
                overflowing_workset.size(),
                0,
                2);
        expect(
            overflowing_opening_admission.overflow &&
                overflowing_opening_admission.maximum_parallel_groups == 1,
            "opening working-set overflow was not fail-closed");

        const auto limited_v2_32g =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = std::to_string(std::size_t{32} << 30),
                    .cgroup_v2_current = std::to_string(std::size_t{16} << 30),
                });
        const auto limited_32g_admission =
            gatzk::util::admit_effective_memory(
                limited_v2_32g,
                std::size_t{1} << 30,
                std::size_t{1} << 30,
                std::size_t{4} << 30,
                1,
                std::numeric_limits<std::size_t>::max());
        expect(
            limited_32g_admission.budget_bytes == (std::size_t{10} << 30) &&
                limited_32g_admission.budget_bytes <
                    unrestricted_admission.budget_bytes,
            "simulated 32 GiB cgroup did not lower admission");

        const auto exhausted_16g =
            gatzk::util::parse_effective_memory_snapshot(
                EffectiveMemoryTextSources{
                    .meminfo = host_meminfo,
                    .cgroup_v2_limit = std::to_string(std::size_t{16} << 30),
                    .cgroup_v2_current = std::to_string(std::size_t{15} << 30),
                });
        const auto exhausted_admission =
            gatzk::util::admit_effective_memory(
                exhausted_16g,
                std::size_t{1} << 30,
                std::size_t{1} << 30,
                std::size_t{4} << 30,
                1,
                std::numeric_limits<std::size_t>::max());
        expect(
            exhausted_admission.budget_bytes == 0,
            "over-budget 16 GiB cgroup was not rejected");

        expect(
            gatzk::util::checked_spill_bytes(3, 4, "test") == 12,
            "spill byte accounting changed");
        expect_failure(
            []()
            {
                (void)gatzk::util::checked_spill_bytes(
                    std::numeric_limits<std::size_t>::max(),
                    2,
                    "overflow");
            },
            "spill byte overflow was accepted");
        expect(
            gatzk::util::spill_align_up(4097, 4096, "alignment") == 8192,
            "spill alignment boundary changed");
        expect_failure(
            []()
            {
                (void)gatzk::util::spill_align_up(
                    std::numeric_limits<std::size_t>::max(),
                    4096,
                    "alignment overflow");
            },
            "spill alignment overflow was accepted");
        expect(
            gatzk::util::spill_arena_can_grow(8, 2, 10) &&
                !gatzk::util::spill_arena_can_grow(8, 3, 10) &&
                !gatzk::util::spill_arena_can_grow(11, 0, 10),
            "spill arena cap boundary changed");

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "effective_memory_admission_runtime: " << error.what() << '\n';
        return error.what() == nullptr ? 1 : 2;
    }
}
