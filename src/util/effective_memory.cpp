#include "gatzk/util/effective_memory.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace gatzk::util
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        constexpr std::uint64_t kKiB = 1024;
        constexpr std::uint64_t kHugeUnlimitedSentinel = 1ULL << 60;
        constexpr auto kSnapshotCacheLifetime = std::chrono::milliseconds(250);

        struct ParsedBytes
        {
            bool present = false;
            bool valid = false;
            bool unlimited = false;
            std::size_t value = 0;
        };

        struct ParsedMeminfo
        {
            ParsedBytes total;
            ParsedBytes available;
        };

        std::string_view trim(std::string_view value)
        {
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t' || value.front() == '\r' || value.front() == '\n'))
            {
                value.remove_prefix(1);
            }
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\r' || value.back() == '\n'))
            {
                value.remove_suffix(1);
            }
            return value;
        }

        std::optional<std::uint64_t> parse_uint64(std::string_view value)
        {
            value = trim(value);
            if (value.empty())
            {
                return std::nullopt;
            }

            std::uint64_t parsed = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
            {
                return std::nullopt;
            }
            return parsed;
        }

        ParsedBytes parse_cgroup_value(std::string_view raw)
        {
            ParsedBytes parsed;
            raw = trim(raw);
            if (raw.empty())
            {
                return parsed;
            }
            parsed.present = true;

            if (raw == "max")
            {
                parsed.valid = true;
                parsed.unlimited = true;
                return parsed;
            }

            const auto value = parse_uint64(raw);
            if (!value.has_value())
            {
                return parsed;
            }

            if (*value >= kHugeUnlimitedSentinel)
            {
                parsed.valid = true;
                parsed.unlimited = true;
                return parsed;
            }

            if (*value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                return parsed;
            }

            parsed.valid = true;
            parsed.value = static_cast<std::size_t>(*value);
            return parsed;
        }

        ParsedBytes parse_meminfo_bytes(std::string_view raw, bool *malformed)
        {
            ParsedBytes parsed;
            const auto separator = raw.find_first_of(" \t");
            if (separator == std::string_view::npos)
            {
                if (!raw.empty() && malformed != nullptr)
                {
                    *malformed = true;
                }
                return parsed;
            }

            const auto number_begin = raw.find_first_not_of(" \t", separator);
            if (number_begin == std::string_view::npos)
            {
                if (malformed != nullptr)
                {
                    *malformed = true;
                }
                return parsed;
            }
            const auto number_end = raw.find_first_of(" \t", number_begin);
            const auto number = raw.substr(number_begin, number_end == std::string_view::npos ? std::string_view::npos : number_end - number_begin);
            const auto value_kib = parse_uint64(number);
            if (!value_kib.has_value() || *value_kib > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / kKiB)
            {
                if (malformed != nullptr)
                {
                    *malformed = true;
                }
                return parsed;
            }

            if (number_end != std::string_view::npos)
            {
                const auto unit_begin = raw.find_first_not_of(" \t", number_end);
                if (unit_begin != std::string_view::npos)
                {
                    const auto unit_end = raw.find_first_of(" \t", unit_begin);
                    const auto unit = raw.substr(unit_begin, unit_end == std::string_view::npos ? std::string_view::npos : unit_end - unit_begin);
                    if (unit != "kB")
                    {
                        if (malformed != nullptr)
                        {
                            *malformed = true;
                        }
                        return parsed;
                    }
                }
            }

            parsed.present = true;
            parsed.valid = true;
            parsed.value = static_cast<std::size_t>(*value_kib * kKiB);
            return parsed;
        }

        ParsedMeminfo parse_meminfo(std::string_view text, bool *malformed)
        {
            ParsedMeminfo parsed;
            std::size_t begin = 0;
            while (begin <= text.size())
            {
                const auto end = text.find('\n', begin);
                const auto line = text.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
                const auto colon = line.find(':');
                if (colon != std::string_view::npos)
                {
                    const auto key = line.substr(0, colon);
                    const auto payload = line.substr(colon + 1);
                    if (key == "MemTotal")
                    {
                        parsed.total = parse_meminfo_bytes(payload, malformed);
                    }
                    else if (key == "MemAvailable")
                    {
                        parsed.available = parse_meminfo_bytes(payload, malformed);
                    }
                }
                if (end == std::string_view::npos)
                {
                    break;
                }
                begin = end + 1;
            }
            return parsed;
        }

        bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t *out)
        {
            if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
            {
                return false;
            }
            *out = lhs + rhs;
            return true;
        }

        std::optional<std::string> read_text_file(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input)
            {
                return std::nullopt;
            }
            std::ostringstream contents;
            contents << input.rdbuf();
            if (!input.good() && !input.eof())
            {
                return std::nullopt;
            }
            return contents.str();
        }

        bool controller_list_contains(std::string_view controllers, std::string_view wanted)
        {
            std::size_t begin = 0;
            while (begin <= controllers.size())
            {
                const auto end = controllers.find(',', begin);
                const auto controller = controllers.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
                if (controller == wanted)
                {
                    return true;
                }
                if (end == std::string_view::npos)
                {
                    break;
                }
                begin = end + 1;
            }
            return false;
        }

        std::vector<std::filesystem::path> cgroup_paths(bool v2, std::string_view file)
        {
            std::vector<std::filesystem::path> membership_paths;
            const auto root = v2 ? std::filesystem::path("/sys/fs/cgroup") : std::filesystem::path("/sys/fs/cgroup/memory");

            const auto self_cgroup = read_text_file("/proc/self/cgroup");
            if (!self_cgroup.has_value())
            {
                return {root / file};
            }

            std::size_t begin = 0;
            while (begin <= self_cgroup->size())
            {
                const auto end = self_cgroup->find('\n', begin);
                const auto line = std::string_view(*self_cgroup).substr(begin, end == std::string::npos ? std::string_view::npos : end - begin);
                const auto first_colon = line.find(':');
                const auto second_colon = first_colon == std::string_view::npos ? std::string_view::npos : line.find(':', first_colon + 1);
                if (first_colon != std::string_view::npos && second_colon != std::string_view::npos)
                {
                    const auto controllers = line.substr(first_colon + 1, second_colon - first_colon - 1);
                    const auto relative = line.substr(second_colon + 1);
                    const bool matches = v2 ? controllers.empty() : controller_list_contains(controllers, "memory");
                    if (matches && !relative.empty() && relative.front() == '/')
                    {
                        membership_paths.emplace_back(root / relative.substr(1) / file);
                    }
                }
                if (end == std::string::npos)
                {
                    break;
                }
                begin = end + 1;
            }
            if (!membership_paths.empty())
            {
                return membership_paths;
            }
            return {root / file};
        }

        std::optional<std::string> read_first_cgroup_file(bool v2, std::string_view file)
        {
            for (const auto &path : cgroup_paths(v2, file))
            {
                std::error_code error;
                if (!std::filesystem::is_regular_file(path, error) || error)
                {
                    continue;
                }
                if (const auto contents = read_text_file(path); contents.has_value())
                {
                    if (trim(*contents).empty())
                    {
                        return std::string("__gatzk_empty_cgroup_value__");
                    }
                    return contents;
                }
                return std::string("__gatzk_unreadable_cgroup_value__");
            }
            return std::nullopt;
        }

    }

    EffectiveMemorySnapshot parse_effective_memory_snapshot(const EffectiveMemoryTextSources &sources)
    {
        EffectiveMemorySnapshot snapshot;
        bool host_malformed = false;
        const auto meminfo = parse_meminfo(sources.meminfo, &host_malformed);
        snapshot.host_total_known = meminfo.total.valid;
        snapshot.host_available_known = meminfo.available.valid;
        snapshot.host_total_bytes = meminfo.total.value;
        snapshot.host_available_bytes = meminfo.available.value;
        snapshot.conservative = host_malformed;

        const auto select_limit = [](std::string_view legacy, std::string_view membership, std::string_view root)
        { return !membership.empty() ? membership : (!root.empty() ? root : legacy); };
        const auto select_current = [](std::string_view legacy, std::string_view membership, std::string_view root)
        { return !membership.empty() ? membership : (!root.empty() ? root : legacy); };

        const auto v2_limit = select_limit(sources.cgroup_v2_limit, sources.cgroup_v2_membership_limit, sources.cgroup_v2_root_limit);
        const auto v2_current = select_current(sources.cgroup_v2_current, sources.cgroup_v2_membership_current, sources.cgroup_v2_root_current);
        const auto v1_limit = select_limit(sources.cgroup_v1_limit, sources.cgroup_v1_membership_limit, sources.cgroup_v1_root_limit);
        const auto v1_current = select_current(sources.cgroup_v1_current, sources.cgroup_v1_membership_current, sources.cgroup_v1_root_current);

        const bool use_v2 = !v2_limit.empty() || !v2_current.empty();
        const bool use_v1 = !use_v2 && (!v1_limit.empty() || !v1_current.empty());
        snapshot.cgroup_v2 = use_v2;
        snapshot.cgroup_v1 = use_v1;

        const auto limit = parse_cgroup_value(use_v2 ? v2_limit : v1_limit);
        const auto current = parse_cgroup_value(use_v2 ? v2_current : v1_current);

        if (use_v2 || use_v1)
        {
            if (!limit.present || !limit.valid || (!current.present && !limit.unlimited))
            {
                snapshot.cgroup_malformed = true;
                snapshot.conservative = true;
            }
            if (limit.valid && limit.unlimited)
            {
                snapshot.cgroup_unlimited = true;
            }
            else if (limit.valid)
            {
                snapshot.cgroup_limit_known = true;
                snapshot.cgroup_limit_bytes = limit.value;
            }
            if (current.valid && !current.unlimited)
            {
                snapshot.cgroup_current_known = true;
                snapshot.cgroup_current_bytes = current.value;
            }
            else if (current.present && !snapshot.cgroup_unlimited)
            {
                snapshot.cgroup_malformed = true;
                snapshot.conservative = true;
            }
        }

        if (snapshot.host_total_known && snapshot.host_available_known && snapshot.host_available_bytes > snapshot.host_total_bytes)
        {
            snapshot.host_available_known = false;
            snapshot.host_available_bytes = 0;
            snapshot.conservative = true;
        }

        if (snapshot.host_total_known)
        {
            snapshot.effective_total_bytes = snapshot.host_total_bytes;
            snapshot.effective_total_known = true;
        }
        if (snapshot.cgroup_limit_known && (!snapshot.effective_total_known || snapshot.cgroup_limit_bytes < snapshot.effective_total_bytes))
        {
            snapshot.effective_total_bytes = snapshot.cgroup_limit_bytes;
            snapshot.effective_total_known = true;
        }

        std::size_t available = 0;
        bool available_known = false;
        if (snapshot.host_available_known)
        {
            available = snapshot.host_available_bytes;
            available_known = true;
        }
        else
        {
            available = 0;
            snapshot.conservative = true;
        }

        if (snapshot.cgroup_limit_known)
        {
            if (snapshot.cgroup_current_known)
            {
                const auto cgroup_available =
                    snapshot.cgroup_current_bytes >= snapshot.cgroup_limit_bytes ? std::size_t{0} : snapshot.cgroup_limit_bytes - snapshot.cgroup_current_bytes;
                if (!available_known || cgroup_available < available)
                {
                    available = cgroup_available;
                }
                available_known = true;
            }
            else
            {
                available = 0;
                available_known = false;
                snapshot.conservative = true;
            }
        }

        if ((snapshot.cgroup_v2 || snapshot.cgroup_v1) && snapshot.cgroup_malformed)
        {
            available = 0;
            available_known = false;
            snapshot.conservative = true;
        }

        snapshot.effective_available_bytes = available;
        snapshot.effective_available_known = available_known;
        if (snapshot.effective_total_known && snapshot.effective_available_bytes > snapshot.effective_total_bytes)
        {
            snapshot.effective_available_bytes = snapshot.effective_total_bytes;
            snapshot.conservative = true;
        }
        return snapshot;
    }

    EffectiveMemorySnapshot read_effective_memory_snapshot()
    {
        std::string meminfo;
        if (const auto contents = read_text_file("/proc/meminfo"); contents.has_value())
        {
            meminfo = *contents;
        }

        std::string cgroup_v2_limit;
        std::string cgroup_v2_current;
        if (const auto contents = read_first_cgroup_file(true, "memory.max"); contents.has_value())
        {
            cgroup_v2_limit = *contents;
        }
        if (const auto contents = read_first_cgroup_file(true, "memory.current"); contents.has_value())
        {
            cgroup_v2_current = *contents;
        }

        std::string cgroup_v1_limit;
        std::string cgroup_v1_current;
        if (const auto contents = read_first_cgroup_file(false, "memory.limit_in_bytes"); contents.has_value())
        {
            cgroup_v1_limit = *contents;
        }
        if (const auto contents = read_first_cgroup_file(false, "memory.usage_in_bytes"); contents.has_value())
        {
            cgroup_v1_current = *contents;
        }

        auto snapshot = parse_effective_memory_snapshot(EffectiveMemoryTextSources{
            .meminfo = meminfo,
            .cgroup_v2_limit = cgroup_v2_limit,
            .cgroup_v2_current = cgroup_v2_current,
            .cgroup_v1_limit = cgroup_v1_limit,
            .cgroup_v1_current = cgroup_v1_current,
            .cgroup_v2_membership_limit = {},
            .cgroup_v2_membership_current = {},
            .cgroup_v2_root_limit = {},
            .cgroup_v2_root_current = {},
            .cgroup_v1_membership_limit = {},
            .cgroup_v1_membership_current = {},
            .cgroup_v1_root_limit = {},
            .cgroup_v1_root_current = {},
        });

        const auto bytes_from_pages = [](long pages_raw, long page_size_raw) -> std::optional<std::size_t>
        {
            if (pages_raw <= 0 || page_size_raw <= 0)
            {
                return std::nullopt;
            }
            const auto pages = static_cast<std::uintmax_t>(pages_raw);
            const auto page_size = static_cast<std::uintmax_t>(page_size_raw);
            if (pages > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) / page_size)
            {
                return std::nullopt;
            }
            return static_cast<std::size_t>(pages * page_size);
        };

        const auto page_size_raw = ::sysconf(_SC_PAGESIZE);
        const auto sysconf_total = bytes_from_pages(::sysconf(_SC_PHYS_PAGES), page_size_raw);
        const auto sysconf_available = bytes_from_pages(::sysconf(_SC_AVPHYS_PAGES), page_size_raw);

        if (!snapshot.host_total_known && sysconf_total.has_value())
        {
            snapshot.host_total_known = true;
            snapshot.host_total_bytes = *sysconf_total;
            if (!snapshot.effective_total_known || snapshot.host_total_bytes < snapshot.effective_total_bytes)
            {
                snapshot.effective_total_known = true;
                snapshot.effective_total_bytes = snapshot.host_total_bytes;
            }
            snapshot.conservative = true;
        }

        if (!snapshot.host_available_known && sysconf_available.has_value() && !snapshot.cgroup_limit_known && !snapshot.cgroup_malformed)
        {
            snapshot.host_available_known = true;
            snapshot.host_available_bytes = *sysconf_available;
            if (snapshot.host_total_known && snapshot.host_available_bytes > snapshot.host_total_bytes)
            {
                snapshot.host_available_known = false;
                snapshot.host_available_bytes = 0;
                snapshot.conservative = true;
            }
            else if (!snapshot.effective_available_known)
            {
                snapshot.effective_available_bytes = snapshot.host_available_bytes;
                snapshot.effective_available_known = true;
                snapshot.conservative = true;
            }
        }

        return snapshot;
    }

    EffectiveMemorySnapshot cached_effective_memory_snapshot()
    {
        static std::mutex mutex;
        static EffectiveMemorySnapshot snapshot;
        static Clock::time_point refreshed_at{};
        static bool initialized = false;

        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized || now - refreshed_at >= kSnapshotCacheLifetime)
        {
            snapshot = read_effective_memory_snapshot();
            refreshed_at = now;
            initialized = true;
        }
        return snapshot;
    }

    std::size_t recommended_memory_reserve_bytes(const EffectiveMemorySnapshot &snapshot, std::size_t minimum_bytes, std::size_t numerator,
                                                 std::size_t denominator)
    {
        if (denominator == 0)
        {
            throw std::invalid_argument("memory reserve denominator must be non-zero");
        }
        if (numerator == 0)
        {
            return minimum_bytes;
        }
        if (!snapshot.effective_total_known)
        {
            return minimum_bytes;
        }
        const auto scaled = snapshot.effective_total_bytes <= std::numeric_limits<std::size_t>::max() / numerator
                                ? (snapshot.effective_total_bytes * numerator) / denominator
                                : std::numeric_limits<std::size_t>::max();
        return std::max(minimum_bytes, scaled);
    }

    EffectiveMemoryAdmission admit_effective_memory(const EffectiveMemorySnapshot &snapshot, std::size_t transient_bytes, std::size_t geometry_bytes,
                                                    std::size_t reserve_bytes, std::size_t divisor, std::size_t maximum_budget_bytes)
    {
        EffectiveMemoryAdmission admission{
            .available_bytes = snapshot.effective_available_known ? snapshot.effective_available_bytes : std::size_t{0},
            .reserve_bytes = reserve_bytes,
            .transient_bytes = transient_bytes,
            .geometry_bytes = geometry_bytes,
            .budget_bytes = 0,
            .conservative = snapshot.conservative,
        };

        std::size_t overhead = 0;
        if (!snapshot.effective_available_known || !checked_add(reserve_bytes, transient_bytes, &overhead) ||
            !checked_add(overhead, geometry_bytes, &overhead) || divisor == 0 || admission.available_bytes <= overhead)
        {
            return admission;
        }

        auto budget = (admission.available_bytes - overhead) / divisor;
        if (snapshot.conservative)
        {
            budget /= 2;
        }
        admission.budget_bytes = std::min(budget, maximum_budget_bytes);
        return admission;
    }

    OpeningMemoryAdmission admit_opening_memory(const EffectiveMemorySnapshot &snapshot, const OpeningMemoryWorkset *worksets, std::size_t workset_count,
                                                std::size_t reserve_bytes, std::size_t maximum_parallel_groups)
    {
        OpeningMemoryAdmission result{
            .budget_bytes = 0,
            .largest_workset_bytes = 0,
            .maximum_parallel_groups = workset_count == 0 ? 0 : std::size_t{1},
            .memory_known = snapshot.effective_available_known,
            .overflow = false,
            .conservative = snapshot.conservative,
        };

        if (worksets == nullptr || workset_count == 0 || maximum_parallel_groups == 0)
        {
            result.maximum_parallel_groups = 0;
            return result;
        }

        for (std::size_t index = 0; index < workset_count; ++index)
        {
            const auto &workset = worksets[index];
            std::size_t total = 0;
            if (!checked_add(total, workset.folded_bytes, &total) || !checked_add(total, workset.scratch_bytes, &total) ||
                !checked_add(total, workset.source_bytes, &total))
            {
                result.overflow = true;
                result.maximum_parallel_groups = 1;
                return result;
            }
            result.largest_workset_bytes = std::max(result.largest_workset_bytes, total);
        }

        const auto base = admit_effective_memory(snapshot, 0, 0, reserve_bytes, 1, std::numeric_limits<std::size_t>::max());
        result.budget_bytes = base.budget_bytes;

        if (!result.memory_known || result.largest_workset_bytes == 0 || result.budget_bytes == 0)
        {
            result.maximum_parallel_groups = 1;
            return result;
        }

        const auto by_budget = result.budget_bytes / result.largest_workset_bytes;
        result.maximum_parallel_groups = std::min(maximum_parallel_groups, std::max<std::size_t>(1, by_budget));
        return result;
    }

}
