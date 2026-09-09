#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/resource.h>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::Polynomial;
    using gatzk::crypto::CommitBatchProfile;
    using gatzk::crypto::Commitment;
    using gatzk::crypto::KZG;
    using gatzk::crypto::KZGKeyPair;

    constexpr std::size_t kMinimumCount = 8193;
    constexpr std::size_t kMaximumCount = 16384;
    constexpr std::size_t kMaximumBatchSize = 8;
    constexpr std::size_t kDefaultWarmups = 1;
    constexpr std::size_t kDefaultRepeats = 5;

    enum class Mode
    {
        Candidate,
        Legacy,
        Equal,
    };

    struct Options
    {
        std::string route = "cuda";
        std::filesystem::path srs_path =
            "artifacts/srs/ppi_formal_srs";
        std::filesystem::path output_path;
        std::size_t warmups = kDefaultWarmups;
        std::size_t repeats = kDefaultRepeats;
        std::vector<std::size_t> batch_sizes{
            2,
            4,
            8};
        bool run_candidate = true;
        bool run_legacy = true;
        bool run_equal = true;
    };

    [[noreturn]] void usage_error(const std::string &message)
    {
        throw std::runtime_error(
            message +
            "\nusage: gatzk_kzg_msm_mixed_prefix_targeted_bench "
            "[--route cpu|cuda] [--srs-path PATH] [--output PATH] "
            "[--batch N] [--warmups N] [--repeats N] "
            "[--mode candidate|legacy|equal|all]");
    }

    std::size_t parse_size(const char *value, const char *label)
    {
        if (value == nullptr || *value == '\0' || *value == '-')
        {
            usage_error(std::string("invalid ") + label);
        }

        std::size_t consumed = 0;
        unsigned long long parsed = 0;
        try
        {
            parsed = std::stoull(value, &consumed, 0);
        }
        catch (const std::exception &)
        {
            usage_error(std::string("invalid ") + label + ": " + value);
        }

        if (consumed != std::string(value).size() || parsed == 0)
        {
            usage_error(std::string("invalid ") + label + ": " + value);
        }

        return static_cast<std::size_t>(parsed);
    }

    Mode parse_mode(const std::string &value)
    {
        if (value == "candidate")
        {
            return Mode::Candidate;
        }
        if (value == "legacy")
        {
            return Mode::Legacy;
        }
        if (value == "equal")
        {
            return Mode::Equal;
        }
        usage_error("invalid mode: " + value);
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument(argv[index]);
            if (argument == "--route")
            {
                if (++index >= argc ||
                    (std::string(argv[index]) != "cpu" &&
                     std::string(argv[index]) != "cuda"))
                {
                    usage_error("--route requires cpu or cuda");
                }
                options.route = argv[index];
            }
            else if (argument == "--srs-path")
            {
                if (++index >= argc)
                {
                    usage_error("--srs-path requires PATH");
                }
                options.srs_path = argv[index];
            }
            else if (argument == "--output")
            {
                if (++index >= argc)
                {
                    usage_error("--output requires PATH");
                }
                options.output_path = argv[index];
            }
            else if (argument == "--batch")
            {
                if (++index >= argc)
                {
                    usage_error("--batch requires N");
                }
                const auto batch = parse_size(argv[index], "batch");
                if (batch > kMaximumBatchSize)
                {
                    usage_error("--batch exceeds the production wave size");
                }
                options.batch_sizes = {batch};
            }
            else if (argument == "--warmups")
            {
                if (++index >= argc)
                {
                    usage_error("--warmups requires N");
                }
                options.warmups = parse_size(argv[index], "warmups");
            }
            else if (argument == "--repeats")
            {
                if (++index >= argc)
                {
                    usage_error("--repeats requires N");
                }
                options.repeats = parse_size(argv[index], "repeats");
            }
            else if (argument == "--mode")
            {
                if (++index >= argc)
                {
                    usage_error("--mode requires candidate, legacy, equal, or all");
                }
                options.run_candidate = false;
                options.run_legacy = false;
                options.run_equal = false;
                const std::string mode(argv[index]);
                if (mode == "all")
                {
                    options.run_candidate = true;
                    options.run_legacy = true;
                    options.run_equal = true;
                }
                else
                {
                    switch (parse_mode(mode))
                    {
                    case Mode::Candidate:
                        options.run_candidate = true;
                        break;
                    case Mode::Legacy:
                        options.run_legacy = true;
                        break;
                    case Mode::Equal:
                        options.run_equal = true;
                        break;
                    }
                }
            }
            else if (argument == "--help" || argument == "-h")
            {
                std::cout
                    << "usage: gatzk_kzg_msm_mixed_prefix_targeted_bench "
                       "[--route cpu|cuda] [--srs-path PATH] [--output PATH] "
                       "[--batch N] [--warmups N] [--repeats N] "
                       "[--mode candidate|legacy|equal|all]\n";
                std::exit(0);
            }
            else
            {
                usage_error("unknown option: " + argument);
            }
        }

        return options;
    }

    void configure_route(const std::string &route)
    {
        auto options = gatzk::util::route2_options();
        options.fast_msm = true;
        options.cuda_field_hotspots = route == "cuda";
        gatzk::util::set_route2_options(options);
    }

    std::uint64_t current_rss_bytes()
    {
        struct rusage usage
        {
        };
        if (getrusage(RUSAGE_SELF, &usage) != 0)
        {
            return 0;
        }
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
    }

    struct PayloadSet
    {
        std::vector<std::size_t> counts;
        std::vector<std::string> names;
        std::vector<std::vector<FieldElement>> coefficients;
    };

    PayloadSet build_payload(
        std::size_t lane_batch_size,
        bool equal_length)
    {
        if (lane_batch_size == 0 ||
            lane_batch_size > kMaximumBatchSize)
        {
            throw std::runtime_error("invalid mixed MSM lane batch size");
        }

        const std::vector<std::size_t> lengths{
            kMinimumCount,
            12289,
            kMaximumCount};

        PayloadSet payload;
        payload.counts.reserve(lane_batch_size * lengths.size());
        payload.names.reserve(lane_batch_size * lengths.size());
        payload.coefficients.reserve(lane_batch_size * lengths.size());

        for (std::size_t lane = 0; lane < lane_batch_size; ++lane)
        {
            for (std::size_t length_index = 0;
                 length_index < lengths.size();
                 ++length_index)
            {
                const auto count = equal_length
                                       ? kMaximumCount
                                       : lengths[length_index];
                const auto object_index = payload.counts.size();
                payload.counts.push_back(count);
                payload.names.push_back(
                    std::string(equal_length ? "equal_" : "mixed_") +
                    std::to_string(lane) + "_" +
                    std::to_string(length_index));

                auto &values = payload.coefficients.emplace_back();
                values.resize(count);
                for (std::size_t index = 0; index < count; ++index)
                {
                    const auto raw =
                        (static_cast<std::uint64_t>(index % 104729U) + 1U) *
                            (static_cast<std::uint64_t>(object_index) + 3U) +
                        static_cast<std::uint64_t>(17U +
                                                   (index % 31U));
                    values[index] = FieldElement(raw);
                    if (values[index].is_zero())
                    {
                        throw std::runtime_error(
                            "deterministic MSM payload unexpectedly contains zero");
                    }
                }
            }
        }

        return payload;
    }

    std::vector<std::pair<std::string, Polynomial>> materialize(
        const PayloadSet &payload,
        const std::vector<std::size_t> &indices)
    {
        std::vector<std::pair<std::string, Polynomial>> result;
        result.reserve(indices.size());
        for (const auto index : indices)
        {
            result.emplace_back(
                payload.names[index],
                Polynomial::from_coefficients(
                    payload.names[index],
                    payload.coefficients[index]));
        }
        return result;
    }

    std::vector<std::size_t> all_indices(const PayloadSet &payload)
    {
        std::vector<std::size_t> indices(payload.counts.size());
        for (std::size_t index = 0; index < indices.size(); ++index)
        {
            indices[index] = index;
        }
        return indices;
    }

    struct CommitResult
    {
        std::vector<Commitment> commitments;
        CommitBatchProfile profile;
        double outer_wall_ms = 0.0;
        std::uint64_t kernel_invocations = 0;
        gatzk::algebra::CudaExecutionStats cuda_stats{};
        std::uint64_t rss_before = 0;
        std::uint64_t rss_after = 0;
    };

    void accumulate_profile(
        CommitBatchProfile *target,
        const CommitBatchProfile &source)
    {
        target->evaluation_ms += source.evaluation_ms;
        target->msm_ms += source.msm_ms;
        target->used_same_srs_mixed_prefix =
            target->used_same_srs_mixed_prefix ||
            source.used_same_srs_mixed_prefix;
        target->used_same_srs_exact_count =
            target->used_same_srs_exact_count ||
            source.used_same_srs_exact_count;
        target->objects.insert(
            target->objects.end(),
            source.objects.begin(),
            source.objects.end());
    }

    bool expected_mixed_prefix_guard(
        const PayloadSet &payload,
        bool equal_length)
    {
        if (equal_length || payload.counts.size() < 2)
        {
            return false;
        }

        const auto max_count = *std::max_element(
            payload.counts.begin(),
            payload.counts.end());
        std::size_t total_count = 0;
        for (const auto count : payload.counts)
        {
            if (count > std::numeric_limits<std::size_t>::max() - total_count)
            {
                throw std::runtime_error("test guard scalar count overflows");
            }
            total_count += count;
        }

        std::vector<std::size_t> exact_counts = payload.counts;
        std::sort(exact_counts.begin(), exact_counts.end());
        std::size_t legacy_calls = 0;
        std::size_t offset = 0;
        while (offset < exact_counts.size())
        {
            std::size_t end = offset + 1;
            while (end < exact_counts.size() &&
                   exact_counts[end] == exact_counts[offset])
            {
                ++end;
            }
            const auto group_size = end - offset;
            legacy_calls +=
                group_size / kMaximumBatchSize +
                (group_size % kMaximumBatchSize != 0 ? 1 : 0);
            offset = end;
        }

        if (payload.counts.size() >
            std::numeric_limits<std::size_t>::max() / max_count)
        {
            throw std::runtime_error("test guard padded count overflows");
        }
        const auto padded_count = payload.counts.size() * max_count;
        if (total_count > padded_count)
        {
            throw std::runtime_error("test guard padding accounting mismatch");
        }
        const auto padding_count = padded_count - total_count;
        const bool padding_within_half =
            padding_count <= std::numeric_limits<std::size_t>::max() / 2 &&
            padding_count * 2 <= padded_count;
        const auto candidate_calls =
            payload.counts.size() / kMaximumBatchSize +
            (payload.counts.size() % kMaximumBatchSize != 0 ? 1 : 0);
        return candidate_calls < legacy_calls && padding_within_half;
    }

    CommitResult run_once(
        const PayloadSet &payload,
        const KZGKeyPair &key,
        Mode mode,
        bool equal_length)
    {
        const auto indices = all_indices(payload);
        auto commitments = std::vector<Commitment>(payload.counts.size());
        CommitBatchProfile aggregate_profile;
        const auto before_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_kernel =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto rss_before = current_rss_bytes();
        const auto start = std::chrono::steady_clock::now();

        if (mode == Mode::Candidate || mode == Mode::Equal)
        {
            auto named = materialize(payload, indices);
            auto result = KZG::commit_owned_batch(
                std::move(named),
                key,
                &aggregate_profile,
                "msm_mixed_prefix_bench");
            if (result.size() != commitments.size())
            {
                throw std::runtime_error("candidate commitment count mismatch");
            }
            commitments = std::move(result);
        }
        else
        {
            std::vector<std::size_t> length_order = indices;
            std::stable_sort(
                length_order.begin(),
                length_order.end(),
                [&](std::size_t lhs, std::size_t rhs)
                {
                    if (payload.counts[lhs] != payload.counts[rhs])
                    {
                        return payload.counts[lhs] > payload.counts[rhs];
                    }
                    return lhs < rhs;
                });

            std::size_t offset = 0;
            while (offset < length_order.size())
            {
                const auto count = payload.counts[length_order[offset]];
                std::size_t end = offset + 1;
                while (end < length_order.size() &&
                       payload.counts[length_order[end]] == count)
                {
                    ++end;
                }

                std::vector<std::size_t> group_indices(
                    length_order.begin() +
                        static_cast<std::ptrdiff_t>(offset),
                    length_order.begin() +
                        static_cast<std::ptrdiff_t>(end));
                auto named = materialize(payload, group_indices);
                CommitBatchProfile group_profile;
                auto result = KZG::commit_owned_batch(
                    std::move(named),
                    key,
                    &group_profile,
                    "msm_mixed_prefix_bench_legacy");
                if (result.size() != group_indices.size())
                {
                    throw std::runtime_error("legacy commitment count mismatch");
                }
                for (std::size_t local = 0;
                     local < group_indices.size();
                     ++local)
                {
                    commitments[group_indices[local]] =
                        std::move(result[local]);
                }
                accumulate_profile(&aggregate_profile, group_profile);

                offset = end;
            }
        }

        // The production guard is intentionally observable in this targeted
        // regression.  In particular, the mixed batch8 shape has three
        // candidate and three exact-count calls, so it must take the exact
        // path even though both routes expose the same CUDA call count.
        if (gatzk::util::route2_options().cuda_field_hotspots &&
            (mode == Mode::Candidate || mode == Mode::Equal))
        {
            const auto expected_mixed =
                expected_mixed_prefix_guard(payload, equal_length);
            if (aggregate_profile.used_same_srs_mixed_prefix != expected_mixed)
            {
                throw std::runtime_error(
                    "same-SRS mixed-prefix guard decision mismatch");
            }
            if (expected_mixed && aggregate_profile.used_same_srs_exact_count)
            {
                throw std::runtime_error(
                    "same-SRS mixed-prefix candidate also reported exact path");
            }
            if (!expected_mixed && payload.counts.size() >= 2 &&
                !aggregate_profile.used_same_srs_exact_count)
            {
                throw std::runtime_error(
                    "same-SRS exact-count guard path was not observed");
            }
        }

        const auto stop = std::chrono::steady_clock::now();
        const auto after_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto after_kernel =
            gatzk::algebra::cuda_kernel_invocation_count();

        CommitResult result;
        result.commitments = std::move(commitments);
        result.profile = std::move(aggregate_profile);
        result.outer_wall_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        result.kernel_invocations = after_kernel - before_kernel;
        result.cuda_stats = {
            after_stats.h2d_bytes - before_stats.h2d_bytes,
            after_stats.d2h_bytes - before_stats.d2h_bytes,
            after_stats.h2d_operations - before_stats.h2d_operations,
            after_stats.d2h_operations - before_stats.d2h_operations,
            after_stats.stream_synchronizations - before_stats.stream_synchronizations,
            after_stats.ntt_calls - before_stats.ntt_calls,
            after_stats.intt_calls - before_stats.intt_calls,
            after_stats.quotient_calls - before_stats.quotient_calls,
            after_stats.t_edge_calls - before_stats.t_edge_calls,
            after_stats.opening_fold_calls - before_stats.opening_fold_calls,
            after_stats.msm_single_calls - before_stats.msm_single_calls,
            after_stats.msm_batch_calls - before_stats.msm_batch_calls,
            after_stats.resident_hits - before_stats.resident_hits,
            after_stats.resident_misses - before_stats.resident_misses,
        };
        result.rss_before = rss_before;
        result.rss_after = current_rss_bytes();
        return result;
    }

    std::uint64_t commitment_digest(
        const std::vector<Commitment> &commitments)
    {
        constexpr std::uint64_t kOffset = 1469598103934665603ULL;
        constexpr std::uint64_t kPrime = 1099511628211ULL;
        std::uint64_t digest = kOffset;
        for (const auto &commitment : commitments)
        {
            const auto bytes = gatzk::crypto::serialize(commitment.point);
            for (const auto byte : bytes)
            {
                digest ^= byte;
                digest *= kPrime;
            }
        }
        return digest;
    }

    void require_equal(
        const std::vector<Commitment> &lhs,
        const std::vector<Commitment> &rhs,
        std::string_view label)
    {
        if (lhs.size() != rhs.size())
        {
            throw std::runtime_error(
                std::string(label) + " commitment count mismatch");
        }
        for (std::size_t index = 0; index < lhs.size(); ++index)
        {
            if (!(lhs[index].point == rhs[index].point))
            {
                throw std::runtime_error(
                    std::string(label) +
                    " commitment differs at object " +
                    std::to_string(index));
            }
        }
    }

    std::string mode_name(Mode mode)
    {
        switch (mode)
        {
        case Mode::Candidate:
            return "candidate";
        case Mode::Legacy:
            return "legacy";
        case Mode::Equal:
            return "equal";
        }
        return "unknown";
    }

    void write_measurement(
        std::ostream &out,
        std::size_t lane_batch_size,
        bool equal_length,
        Mode mode,
        std::size_t repeat,
        const CommitResult &result,
        std::uint64_t digest)
    {
        out << "route=" << gatzk::util::route2_feature_label(
                            gatzk::util::route2_options())
            << " family=" << (equal_length ? "equal" : "mixed")
            << " lane_batch=" << lane_batch_size
            << " objects=" << result.commitments.size()
            << " mode=" << mode_name(mode)
            << " repeat=" << repeat
            << " outer_batch_wall_ms=" << std::fixed << std::setprecision(3)
            << result.outer_wall_ms
            << " scoped_commitment_wall_ms=" << result.outer_wall_ms
            << " profile_msm_ms=" << result.profile.msm_ms
            << " profile_objects=" << result.profile.objects.size()
            << " same_srs_mixed_prefix="
            << (result.profile.used_same_srs_mixed_prefix ? 1 : 0)
            << " same_srs_exact_count="
            << (result.profile.used_same_srs_exact_count ? 1 : 0)
            << " digest=" << digest
            << " kernel_invocations=" << result.kernel_invocations
            << " h2d_bytes=" << result.cuda_stats.h2d_bytes
            << " d2h_bytes=" << result.cuda_stats.d2h_bytes
            << " h2d_operations=" << result.cuda_stats.h2d_operations
            << " d2h_operations=" << result.cuda_stats.d2h_operations
            << " syncs=" << result.cuda_stats.stream_synchronizations
            << " msm_single_calls=" << result.cuda_stats.msm_single_calls
            << " msm_batch_calls=" << result.cuda_stats.msm_batch_calls
            << " resident_hits=" << result.cuda_stats.resident_hits
            << " resident_misses=" << result.cuda_stats.resident_misses
            << " rss_before=" << result.rss_before
            << " rss_after=" << result.rss_after
            << '\n';
    }

    void run_series(
        std::ostream &out,
        const Options &options,
        const KZGKeyPair &key,
        const PayloadSet &payload,
        bool equal_length,
        Mode mode,
        std::vector<Commitment> *reference)
    {
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
        for (std::size_t warmup = 0;
             warmup < options.warmups;
             ++warmup)
        {
            const auto warmup_result = run_once(payload, key, mode, equal_length);
            if (reference != nullptr && warmup == options.warmups - 1)
            {
                *reference = warmup_result.commitments;
            }
        }

        for (std::size_t repeat = 0;
             repeat < options.repeats;
             ++repeat)
        {
            const auto result = run_once(payload, key, mode, equal_length);
            const auto digest = commitment_digest(result.commitments);
            write_measurement(
                out,
                payload.counts.size() / 3,
                equal_length,
                mode,
                repeat,
                result,
                digest);
            if (reference != nullptr && repeat == 0)
            {
                *reference = result.commitments;
            }
        }
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
    }

    void run_shape(
        std::ostream &out,
        const Options &options,
        const KZGKeyPair &key,
        std::size_t lane_batch_size,
        bool equal_length)
    {
        const auto payload = build_payload(lane_batch_size, equal_length);
        std::vector<Commitment> candidate_reference;
        std::vector<Commitment> legacy_reference;
        std::vector<Commitment> equal_reference;

        if (options.run_candidate || options.run_legacy)
        {
            configure_route(options.route);
            if (options.run_candidate)
            {
                run_series(
                    out,
                    options,
                    key,
                    payload,
                    equal_length,
                    Mode::Candidate,
                    &candidate_reference);
            }
            if (options.run_legacy)
            {
                run_series(
                    out,
                    options,
                    key,
                    payload,
                    equal_length,
                    Mode::Legacy,
                    &legacy_reference);
            }
        }

        if (options.run_candidate && options.run_legacy)
        {
            require_equal(
                candidate_reference,
                legacy_reference,
                "candidate/legacy");
            out << "exact_check= candidate_vs_legacy"
                << " family=" << (equal_length ? "equal" : "mixed")
                << " lane_batch=" << lane_batch_size
                << " digest=" << commitment_digest(candidate_reference)
                << " status=EXACT\n";
        }

        if (equal_length && options.run_equal)
        {
            configure_route(options.route);
            run_series(
                out,
                options,
                key,
                payload,
                true,
                Mode::Equal,
                &equal_reference);
        }

        if (options.route == "cuda" &&
            (!candidate_reference.empty() || !equal_reference.empty()))
        {
            configure_route("cpu");
            const auto cpu_result = run_once(
                payload,
                key,
                equal_length ? Mode::Equal : Mode::Candidate,
                equal_length);
            const auto &gpu_reference =
                equal_length ? equal_reference : candidate_reference;
            if (!gpu_reference.empty())
            {
                require_equal(
                    gpu_reference,
                    cpu_result.commitments,
                    "CUDA/CPU");
                out << "exact_check= cuda_vs_cpu"
                    << " family=" << (equal_length ? "equal" : "mixed")
                    << " lane_batch=" << lane_batch_size
                    << " digest=" << commitment_digest(cpu_result.commitments)
                    << " status=EXACT\n";
            }
            configure_route(options.route);
        }
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
            std::cout << "status=unavailable route=cuda\n";
            return 77;
        }

        configure_route(options.route);
        const auto max_count = kMaximumCount;
        const auto key = KZG::load_keys_for_domains(
            options.srs_path.string(),
            "trusted_srs",
            "ppi_formal_srs",
            // This benchmark commits coefficient-basis objects only.  Keep
            // one small authenticated Lagrange domain in the production
            // loader/audit contract; requesting the default all-domain set
            // would require materializing the full FH coefficient prefix,
            // which is unrelated to this MSM path.
            {{"d", 16}},
            max_count);

        std::ofstream output_file;
        std::ostream *output = &std::cout;
        if (!options.output_path.empty())
        {
            if (!options.output_path.parent_path().empty())
            {
                std::filesystem::create_directories(
                    options.output_path.parent_path());
            }
            output_file.open(options.output_path);
            if (!output_file)
            {
                throw std::runtime_error(
                    "failed to open benchmark output: " +
                    options.output_path.string());
            }
            output = &output_file;
        }

        *output << "benchmark=KZG_commit_owned_batch_mixed_prefix"
                << " srs_path=" << options.srs_path.string()
                << " srs_id=" << key.pk.srs_id
                << " srs_type=" << key.pk.srs_type
                << " loaded_coefficient_capacity=" << key.pk.powers_g1->size()
                << " authenticated_manifest_hash=" << key.manifest.manifest_hash
                << " warmups=" << options.warmups
                << " repeats=" << options.repeats
                << "\n";

        for (const auto lane_batch_size : options.batch_sizes)
        {
            if (options.run_candidate || options.run_legacy)
            {
                run_shape(
                    *output,
                    options,
                    key,
                    lane_batch_size,
                    false);
            }
            if (options.run_equal)
            {
                run_shape(
                    *output,
                    options,
                    key,
                    lane_batch_size,
                    true);
            }
        }

        *output << "status=PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
