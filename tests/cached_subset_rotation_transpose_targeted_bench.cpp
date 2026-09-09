#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    using gatzk::algebra::EvaluationExecutionPolicy;
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::PackedEvaluationBackend;
    using gatzk::algebra::Polynomial;
    using gatzk::algebra::RootOfUnityDomain;

    struct Options
    {
        std::string route = "cuda";
        bool candidate = true;
        std::size_t domain_log2 = 18;
        std::size_t repeats = 5;
    };

    std::size_t parse_size(const std::string &value, const char *label)
    {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size())
        {
            throw std::runtime_error(std::string("invalid ") + label);
        }
        return static_cast<std::size_t>(parsed);
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            const auto require_value = [&](const char *name)
            {
                if (index + 1 >= argc)
                {
                    throw std::runtime_error(
                        std::string("missing value for ") + name);
                }
                return std::string(argv[++index]);
            };

            if (argument == "--route")
            {
                options.route = require_value("--route");
                if (options.route != "cpu" && options.route != "cuda")
                {
                    throw std::runtime_error("--route must be cpu or cuda");
                }
            }
            else if (argument == "--candidate")
            {
                const auto value = require_value("--candidate");
                if (value == "on")
                {
                    options.candidate = true;
                }
                else if (value == "off")
                {
                    options.candidate = false;
                }
                else
                {
                    throw std::runtime_error(
                        "--candidate must be on or off");
                }
            }
            else if (argument == "--domain-log2")
            {
                options.domain_log2 = parse_size(
                    require_value("--domain-log2"),
                    "domain-log2");
            }
            else if (argument == "--repeats")
            {
                options.repeats = parse_size(
                    require_value("--repeats"),
                    "repeats");
            }
            else
            {
                throw std::runtime_error(
                    "unknown argument: " + argument);
            }
        }
        if (options.domain_log2 < 18 || options.domain_log2 >=
                std::numeric_limits<std::size_t>::digits)
        {
            throw std::runtime_error(
                "domain-log2 must be in [18, size_t_bits)");
        }
        if (options.repeats == 0)
        {
            throw std::runtime_error("repeats must be positive");
        }
        return options;
    }

    std::uint64_t read_proc_kb(const char *field)
    {
        std::ifstream input("/proc/self/status");
        const std::string requested(field);
        std::string line;
        while (std::getline(input, line))
        {
            if (line.compare(0, requested.size(), requested) == 0)
            {
                std::istringstream parser(line.substr(requested.size()));
                std::uint64_t value = 0;
                parser >> value;
                return value;
            }
        }
        return 0;
    }

    std::uint64_t peak_rss_kb()
    {
        struct rusage usage
        {
        };
        if (getrusage(RUSAGE_SELF, &usage) != 0)
        {
            return 0;
        }
        return static_cast<std::uint64_t>(usage.ru_maxrss);
    }

    double elapsed_ms(
        const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::uint64_t digest_native(const std::vector<FieldElement> &values)
    {
        static_assert(
            sizeof(mcl::Fr) == 4 * sizeof(std::uint64_t),
            "benchmark expects four-limb mcl::Fr storage");
        std::uint64_t digest = 1469598103934665603ULL;
        for (const auto &value : values)
        {
            const auto *limbs = reinterpret_cast<const std::uint64_t *>(
                &value.native());
            for (std::size_t limb = 0; limb < 4; ++limb)
            {
                digest ^= limbs[limb];
                digest *= 1099511628211ULL;
            }
        }
        return digest;
    }

    std::uint64_t digest_native(const std::vector<mcl::Fr> &values)
    {
        static_assert(
            sizeof(mcl::Fr) == 4 * sizeof(std::uint64_t),
            "benchmark expects four-limb mcl::Fr storage");
        std::uint64_t digest = 1469598103934665603ULL;
        for (const auto &value : values)
        {
            const auto *limbs = reinterpret_cast<const std::uint64_t *>(
                &value);
            for (std::size_t limb = 0; limb < 4; ++limb)
            {
                digest ^= limbs[limb];
                digest *= 1099511628211ULL;
            }
        }
        return digest;
    }

    std::vector<mcl::Fr> expected_results(
        const std::vector<mcl::Fr> &weights,
        const std::vector<Polynomial> &polynomials,
        const std::vector<std::size_t> &rotations)
    {
        const auto domain_size = weights.size();
        const auto domain_mask = domain_size - 1U;
        std::vector<mcl::Fr> expected(
            rotations.size() * polynomials.size());
        for (auto &value : expected)
        {
            value.clear();
        }
        for (std::size_t point = 0; point < rotations.size(); ++point)
        {
            const auto rotation = rotations[point] & domain_mask;
            for (std::size_t row = 0; row < polynomials.size(); ++row)
            {
                const auto &values = polynomials[row].values();
                for (std::size_t column = 0; column < domain_size; ++column)
                {
                    const auto rotated_column =
                        (column + rotation) & domain_mask;
                    mcl::Fr term;
                    mcl::Fr::mul(
                        term,
                        weights[column],
                        values[rotated_column].native());
                    mcl::Fr::add(
                        expected[point * polynomials.size() + row],
                        expected[point * polynomials.size() + row],
                        term);
                }
            }
        }
        return expected;
    }

    std::vector<std::pair<std::string, const Polynomial *>> make_entries(
        const std::vector<Polynomial> &polynomials)
    {
        std::vector<std::pair<std::string, const Polynomial *>> entries;
        entries.reserve(polynomials.size());
        for (std::size_t row = 0; row < polynomials.size(); ++row)
        {
            entries.emplace_back(
                "cached-subset-column-" + std::to_string(row),
                &polynomials[row]);
        }
        return entries;
    }

    void require_exact(
        const std::vector<FieldElement> &actual,
        const std::vector<mcl::Fr> &expected)
    {
        if (actual.size() != expected.size())
        {
            throw std::runtime_error("cached subset output size mismatch");
        }
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            if (actual[index].native() != expected[index])
            {
                throw std::runtime_error(
                    "cached subset output differs from CPU reference at " +
                    std::to_string(index));
            }
        }
    }

    void require_exact(
        const std::vector<std::vector<FieldElement>> &actual,
        const std::vector<mcl::Fr> &expected)
    {
        std::size_t flat_index = 0;
        for (const auto &row : actual)
        {
            for (const auto &value : row)
            {
                if (flat_index >= expected.size() ||
                    value.native() != expected[flat_index])
                {
                    throw std::runtime_error(
                        "cached subset output differs from CPU reference at " +
                        std::to_string(flat_index));
                }
                ++flat_index;
            }
        }
        if (flat_index != expected.size())
        {
            throw std::runtime_error("cached subset output size mismatch");
        }
    }

    std::uint64_t digest_native(
        const std::vector<std::vector<FieldElement>> &values)
    {
        std::uint64_t digest = 1469598103934665603ULL;
        for (const auto &row : values)
        {
            for (const auto &value : row)
            {
                const auto *limbs = reinterpret_cast<const std::uint64_t *>(
                    &value.native());
                for (std::size_t limb = 0; limb < 4; ++limb)
                {
                    digest ^= limbs[limb];
                    digest *= 1099511628211ULL;
                }
            }
        }
        return digest;
    }

    std::uint64_t delta_u64(std::uint64_t after, std::uint64_t before)
    {
        return after >= before ? after - before : 0;
    }

    void run_case(
        const Options &options,
        std::size_t domain_size,
        std::size_t row_count)
    {
        const auto domain = RootOfUnityDomain::create("edge", domain_size);
        std::vector<Polynomial> polynomials;
        polynomials.reserve(row_count);
        for (std::size_t row = 0; row < row_count; ++row)
        {
            auto values = std::make_shared<std::vector<FieldElement>>(
                domain_size);
            for (std::size_t column = 0; column < domain_size; ++column)
            {
                const auto value = static_cast<std::uint64_t>(
                    17 + ((column * (row + 5) + row * 97) % 1000003));
                (*values)[column] = FieldElement(value);
            }
            polynomials.push_back(Polynomial::from_shared_evaluations(
                "cached-subset-column-" + std::to_string(row),
                std::move(values),
                domain));
        }

        const auto entries = make_entries(polynomials);
        PackedEvaluationBackend backend(domain, entries);
        std::vector<std::string> labels;
        labels.reserve(row_count);
        for (std::size_t row = 0; row < row_count; ++row)
        {
            labels.push_back(
                "cached-subset-column-" + std::to_string(row));
        }

        std::vector<mcl::Fr> weights(domain_size);
        const auto weights_start = std::chrono::steady_clock::now();
        for (std::size_t column = 0; column < domain_size; ++column)
        {
            weights[column] = FieldElement(
                static_cast<std::uint64_t>(
                    29 + ((column * 19 + 11) % 1000003)))
                                 .native();
        }
        const auto weights_end = std::chrono::steady_clock::now();

        const std::vector<std::size_t> rotations = {0, 1, 2};
        const auto expected = expected_results(
            weights,
            polynomials,
            rotations);
        const auto expected_digest = digest_native(expected);

        gatzk::algebra::set_cached_subset_multi_rotation_candidate_for_tests(
            options.candidate);
        const auto candidate_invocations_before =
            gatzk::algebra::cached_subset_multi_rotation_candidate_invocations_for_tests();

        const auto policy = options.route == "cpu"
                                ? EvaluationExecutionPolicy::CpuOnly
                                : EvaluationExecutionPolicy::Automatic;
        // This is the production direct caller used by
        // EvaluationMemoization::precompute_named. Its first invocation
        // includes natural subset materialization and backend evaluation.
        const auto precompute_start = std::chrono::steady_clock::now();
        const auto warmup_start = std::chrono::steady_clock::now();
        const auto warmup = backend.evaluate_with_native_weight_rotations(
            labels,
            weights,
            rotations,
            policy);
        const auto warmup_end = std::chrono::steady_clock::now();
        const auto precompute_end = warmup_end;
        require_exact(warmup, expected);
        const auto expected_subset_bytes =
            domain_size * row_count * sizeof(mcl::Fr);
        const auto cached_subset_bytes =
            backend.cached_subset_bytes_for_tests(labels);
        const bool cached_subset_natural =
            cached_subset_bytes == expected_subset_bytes;

        const auto before_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_kernel_invocations =
            gatzk::algebra::cuda_kernel_invocation_count();
        std::vector<double> samples;
        samples.reserve(options.repeats);
        std::vector<std::vector<FieldElement>> last;
        std::uint64_t rss_kb = 0;
        for (std::size_t repeat = 0; repeat < options.repeats; ++repeat)
        {
            const auto start = std::chrono::steady_clock::now();
            last = backend.evaluate_with_native_weight_rotations(
                labels,
                weights,
                rotations,
                policy);
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(elapsed_ms(start, end));
            rss_kb = std::max(rss_kb, read_proc_kb("VmRSS:"));
            require_exact(last, expected);
        }
        const auto after_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto candidate_invocations_after =
            gatzk::algebra::cached_subset_multi_rotation_candidate_invocations_for_tests();

        auto sorted_samples = samples;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        const auto median = sorted_samples[sorted_samples.size() / 2];
        const auto mean = std::accumulate(
            samples.begin(),
            samples.end(),
            0.0) /
                          static_cast<double>(samples.size());
        const auto workspace_field_bytes =
            domain_size * sizeof(mcl::Fr);
        const auto workspace_subset_bytes =
            workspace_field_bytes * row_count;
        const auto workspace_partial_bytes =
            ((domain_size + 255) / 256) * row_count * rotations.size() *
            sizeof(mcl::Fr);
        const auto legacy_workspace_estimate_bytes =
            workspace_field_bytes +
            workspace_subset_bytes +
            workspace_partial_bytes +
            row_count * rotations.size() * sizeof(mcl::Fr);
        const auto candidate_workspace_estimate_bytes =
            workspace_field_bytes +
            workspace_subset_bytes * 2 +
            workspace_partial_bytes +
            row_count * rotations.size() * sizeof(mcl::Fr);
        const auto device_peak_estimate_bytes =
            workspace_field_bytes +
            workspace_subset_bytes * 2 +
            workspace_partial_bytes +
            row_count * rotations.size() * sizeof(mcl::Fr) +
            row_count * sizeof(const mcl::Fr *) +
            rotations.size() * sizeof(std::size_t) +
            64ULL * 1024ULL * 1024ULL;
        const bool candidate_workload_eligible =
            static_cast<unsigned __int128>(domain_size) * row_count >=
            (static_cast<unsigned __int128>(1) << 24);

        std::cout << std::fixed << std::setprecision(3)
                  << "route=" << options.route
                  << " candidate=" << (options.candidate ? "on" : "off")
                  << " domain=" << domain_size
                  << " columns=" << row_count
                  << " rotations=0,1,2"
                  << " production_entry=EvaluationMemoization::precompute_named->PackedEvaluationBackend::evaluate_with_native_weight_rotations"
                  << " cached_subset_expected_bytes=" << expected_subset_bytes
                  << " cached_subset_bytes=" << cached_subset_bytes
                  << " cached_subset_natural="
                  << (cached_subset_natural ? "true" : "false")
                  << " candidate_workload_eligible="
                  << (candidate_workload_eligible ? "true" : "false")
                  << " gpu_device=\""
                  << gatzk::algebra::cuda_device_summary() << "\"\n"
                  << "weights_build_ms="
                  << elapsed_ms(weights_start, weights_end)
                  << " precompute_wall_ms="
                  << elapsed_ms(precompute_start, precompute_end)
                  << " warmup_parent_ms="
                  << elapsed_ms(warmup_start, warmup_end)
                  << " parent_wall_ms_median=" << median
                  << " parent_wall_ms_mean=" << mean
                  << " digest_expected=" << expected_digest
                  << " digest=" << digest_native(last)
                  << " candidate_invocations="
                  << (candidate_invocations_after - candidate_invocations_before)
                  << " kernel_invocations="
                  << delta_u64(
                         gatzk::algebra::cuda_kernel_invocation_count(),
                         before_kernel_invocations)
                  << " h2d_bytes="
                  << delta_u64(after_stats.h2d_bytes, before_stats.h2d_bytes)
                  << " d2h_bytes="
                  << delta_u64(after_stats.d2h_bytes, before_stats.d2h_bytes)
                  << " h2d_operations="
                  << delta_u64(
                         after_stats.h2d_operations,
                         before_stats.h2d_operations)
                  << " d2h_operations="
                  << delta_u64(
                         after_stats.d2h_operations,
                         before_stats.d2h_operations)
                  << " syncs="
                  << delta_u64(
                         after_stats.stream_synchronizations,
                         before_stats.stream_synchronizations)
                  << " rss_kb=" << read_proc_kb("VmRSS:")
                  << " peak_rss_kb=" << peak_rss_kb()
                  << " legacy_workspace_estimate_bytes="
                  << legacy_workspace_estimate_bytes
                  << " candidate_workspace_estimate_bytes="
                  << candidate_workspace_estimate_bytes
                  << " device_peak_estimate_bytes="
                  << device_peak_estimate_bytes
                  << " status=EXACT\n";
        (void)rss_kb;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        if (options.route == "cuda" &&
            !gatzk::algebra::cuda_backend_available())
        {
            std::cerr << "CUDA unavailable; targeted benchmark skipped\n";
            return 77;
        }
        auto route = gatzk::util::route2_options();
        route.cuda_field_hotspots = options.route == "cuda";
        gatzk::util::set_route2_options(route);
        const auto domain_size = std::size_t{1} << options.domain_log2;
        const bool reduced_memory_admission_rejected =
            !gatzk::algebra::cuda_interleaved_multi_rotation_admissible(
                domain_size,
                64,
                3,
                std::numeric_limits<std::size_t>::max());
        if (!reduced_memory_admission_rejected)
        {
            throw std::runtime_error(
                "reduced-memory admission regression unexpectedly accepted");
        }
        std::cout << "reduced_memory_admission=REJECTED"
                  << " production_cache_policy=unchanged"
                  << " source_entry=EvaluationMemoization::precompute_named->PackedEvaluationBackend::evaluate_with_native_weight_rotations\n";
        for (const auto row_count : {
                 std::size_t{8},
                 std::size_t{32},
                 std::size_t{64}})
        {
            run_case(options, domain_size, row_count);
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "cached subset benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
