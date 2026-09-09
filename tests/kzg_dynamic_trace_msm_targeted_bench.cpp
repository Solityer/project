#include <algorithm>
#include <array>
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
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    using gatzk::algebra::CudaExecutionStats;
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::Polynomial;
    using gatzk::crypto::CommitBatchProfile;
    using gatzk::crypto::Commitment;
    using gatzk::crypto::KZG;
    using gatzk::crypto::KZGKeyPair;
    using gatzk::protocol::DynamicCommitmentBatchTestInput;
    using gatzk::protocol::DynamicCommitmentBatchTestResult;
    using gatzk::protocol::ProtocolCompilerArtifact;
    using gatzk::protocol::TraceArtifacts;

    constexpr std::size_t kDomainSize = 16;
    constexpr std::size_t kMaximumBatchSize = 4;
    constexpr std::size_t kWarmups = 1;
    constexpr std::size_t kRepeats = 5;
    constexpr std::string_view kRegistryPath =
        "artifacts/verifier_registry/ppi/verifier_registry.txt";
    constexpr std::string_view kRegistryDigest =
        "45fb77629c03df20453cc7baff5417f3cc023da3a9a015c1338ff8136e582212";

    struct Options
    {
        std::string route = "cuda";
        std::filesystem::path srs_path =
            "artifacts/srs/ppi_formal_srs";
        std::filesystem::path registry_path =
            std::string(kRegistryPath);
        std::string registry_digest = std::string(kRegistryDigest);
        std::filesystem::path output_path;
        std::size_t warmups = kWarmups;
        std::size_t repeats = kRepeats;
        std::vector<std::size_t> batch_sizes{2, 4};
    };

    [[noreturn]] void usage_error(const std::string &message)
    {
        throw std::runtime_error(
            message +
            "\nusage: gatzk_kzg_dynamic_trace_msm_targeted_bench "
            "[--route cpu|cuda] [--srs-path PATH] "
            "[--registry PATH] [--registry-digest HEX] "
            "[--output PATH] [--batch N] [--warmups N] [--repeats N]");
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
            else if (argument == "--registry")
            {
                if (++index >= argc)
                {
                    usage_error("--registry requires PATH");
                }
                options.registry_path = argv[index];
            }
            else if (argument == "--registry-digest")
            {
                if (++index >= argc)
                {
                    usage_error("--registry-digest requires HEX");
                }
                options.registry_digest = argv[index];
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
                    usage_error("--batch exceeds the dynamic test envelope");
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
            else if (argument == "--help" || argument == "-h")
            {
                std::cout
                    << "usage: gatzk_kzg_dynamic_trace_msm_targeted_bench "
                       "[--route cpu|cuda] [--srs-path PATH] "
                       "[--registry PATH] [--registry-digest HEX] "
                       "[--output PATH] [--batch N] [--warmups N] "
                       "[--repeats N]\n";
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

    struct PrivateLedgerEntry
    {
        std::string name;
        std::string domain;
        std::size_t domain_size = 0;
        std::size_t query_bound = 0;
    };

    std::vector<PrivateLedgerEntry> select_private_entries(
        const ProtocolCompilerArtifact &artifact,
        std::string_view required_domain,
        std::size_t required_domain_size,
        std::size_t required)
    {
        if (required_domain_size == 0)
        {
            throw std::runtime_error(
                "dynamic benchmark requires a nonzero authenticated domain size");
        }

        std::vector<PrivateLedgerEntry> entries;
        entries.reserve(required);
        std::vector<std::string> available;
        for (const auto &entry : artifact.observation_ledger)
        {
            if (entry.object_or_challenge.starts_with("P_h_l0_h0_") &&
                entry.domain == required_domain)
            {
                available.push_back(
                    entry.object_or_challenge + ":q=" +
                    std::to_string(entry.query_bound));
            }
            if (entry.object_or_challenge.starts_with("P_h_l0_h0_") &&
                entry.domain == required_domain &&
                entry.query_bound != 0 &&
                std::find_if(
                    entries.begin(),
                    entries.end(),
                    [&](const PrivateLedgerEntry &candidate)
                    { return candidate.name == entry.object_or_challenge; }) ==
                    entries.end())
            {
                entries.push_back({
                    .name = entry.object_or_challenge,
                    .domain = entry.domain,
                    .domain_size = required_domain_size,
                    .query_bound = entry.query_bound});
                if (entries.size() == required)
                {
                    break;
                }
            }
        }
        if (entries.size() != required)
        {
            throw std::runtime_error(
                "authenticated PPI compiler artifact lacks enough private "
                    "dynamic observation entries for domain " +
                std::string(required_domain) + "; found " +
                std::to_string(entries.size()) + ", required " +
                std::to_string(required) + "; available=" +
                [&]()
                {
                    std::string listed;
                    for (std::size_t index = 0;
                         index < available.size();
                         ++index)
                    {
                        if (index != 0)
                        {
                            listed += ',';
                        }
                        listed += available[index];
                    }
                    return listed;
                }());
        }
        return entries;
    }

    struct Payload
    {
        std::vector<DynamicCommitmentBatchTestInput> inputs;
        std::vector<std::size_t> counts;
    };

    Payload build_payload(
        std::size_t batch_size,
        const std::shared_ptr<gatzk::algebra::RootOfUnityDomain> &domain,
        const std::vector<PrivateLedgerEntry> &entries)
    {
        if (batch_size == 0 || batch_size > kMaximumBatchSize)
        {
            throw std::runtime_error("invalid dynamic test batch size");
        }
        const std::array<std::size_t, 3> lengths{8, 12, 16};
        Payload payload;
        if (domain == nullptr || domain->name != "d" ||
            domain->size != kDomainSize)
        {
            throw std::runtime_error(
                "dynamic benchmark requires the authenticated d:16 domain");
        }
        if (entries.size() < batch_size * lengths.size())
        {
            throw std::runtime_error(
                "dynamic benchmark payload lacks enough authenticated ledger "
                "entries");
        }
        payload.inputs.reserve(batch_size * lengths.size());
        payload.counts.reserve(batch_size * lengths.size());
        for (std::size_t index = 0;
             index < batch_size * lengths.size();
             ++index)
        {
            const auto length = lengths[index % lengths.size()];
            const auto &entry = entries[index];
            if (entry.domain != domain->name ||
                entry.domain_size != domain->size ||
                entry.query_bound == 0 ||
                length > entry.domain_size)
            {
                throw std::runtime_error(
                    "dynamic benchmark attempted to reinterpret an "
                    "authenticated ledger entry across domains: " +
                    entry.name);
            }
            std::vector<FieldElement> values(length);
            for (std::size_t value_index = 0;
                 value_index < values.size();
                 ++value_index)
            {
                const auto raw =
                    (static_cast<std::uint64_t>(value_index) + 3U) *
                        static_cast<std::uint64_t>(index + 7U) +
                    static_cast<std::uint64_t>(19U +
                                               (value_index % 5U));
                values[value_index] = FieldElement(raw);
                if (values[value_index].is_zero())
                {
                    throw std::runtime_error(
                        "dynamic test payload unexpectedly contains zero");
                }
            }
            payload.inputs.push_back({
                .name = entry.name,
                .values = std::move(values),
                .domain = domain,
            });
            payload.counts.push_back(length);
        }
        return payload;
    }

    void accumulate_profile(
        CommitBatchProfile *target,
        const CommitBatchProfile &source)
    {
        target->evaluation_ms += source.evaluation_ms;
        target->msm_ms += source.msm_ms;
        target->objects.insert(
            target->objects.end(),
            source.objects.begin(),
            source.objects.end());
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
                    std::string(label) + " commitment differs at object " +
                    std::to_string(index));
            }
        }
    }

    struct RouteDelta
    {
        CudaExecutionStats stats{};
        std::uint64_t kernel_invocations = 0;
    };

    RouteDelta route_delta(
        const CudaExecutionStats &before_stats,
        const CudaExecutionStats &after_stats,
        std::uint64_t before_kernel,
        std::uint64_t after_kernel)
    {
        return RouteDelta{
            .stats = {
                after_stats.h2d_bytes - before_stats.h2d_bytes,
                after_stats.d2h_bytes - before_stats.d2h_bytes,
                after_stats.h2d_operations - before_stats.h2d_operations,
                after_stats.d2h_operations - before_stats.d2h_operations,
                after_stats.stream_synchronizations -
                    before_stats.stream_synchronizations,
                after_stats.ntt_calls - before_stats.ntt_calls,
                after_stats.intt_calls - before_stats.intt_calls,
                after_stats.quotient_calls - before_stats.quotient_calls,
                after_stats.t_edge_calls - before_stats.t_edge_calls,
                after_stats.opening_fold_calls -
                    before_stats.opening_fold_calls,
                after_stats.msm_single_calls - before_stats.msm_single_calls,
                after_stats.msm_batch_calls - before_stats.msm_batch_calls,
                after_stats.resident_hits - before_stats.resident_hits,
                after_stats.resident_misses - before_stats.resident_misses,
            },
            .kernel_invocations = after_kernel - before_kernel,
        };
    }

    struct CandidateRun
    {
        TraceArtifacts trace;
        DynamicCommitmentBatchTestResult dynamic;
        RouteDelta route;
        std::uint64_t digest = 0;
        std::uint64_t rss_before = 0;
        std::uint64_t rss_after = 0;
    };

    CandidateRun run_candidate(
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact)
    {
        TraceArtifacts trace;
        trace.compiler_artifact = artifact;
        const auto before_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_kernel =
            gatzk::algebra::cuda_kernel_invocation_count();
        CandidateRun result;
        result.rss_before = current_rss_bytes();
        result.dynamic =
            gatzk::protocol::add_dynamic_commitment_batch_for_tests(
                trace,
                payload.inputs,
                key,
                false);
        const auto after_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto after_kernel =
            gatzk::algebra::cuda_kernel_invocation_count();
        result.route = route_delta(
            before_stats,
            after_stats,
            before_kernel,
            after_kernel);
        result.digest = commitment_digest(result.dynamic.commitments);
        result.rss_after = current_rss_bytes();
        result.trace = std::move(trace);
        return result;
    }

    struct LegacyRun
    {
        std::vector<Commitment> commitments;
        CommitBatchProfile profile;
        double wall_ms = 0.0;
        double scoped_commitment_ms = 0.0;
        RouteDelta route;
    };

    LegacyRun run_legacy(
        const Payload &payload,
        const CandidateRun &candidate,
        const KZGKeyPair &key)
    {
        std::vector<Polynomial> polynomials;
        polynomials.reserve(payload.inputs.size());
        for (const auto &input : payload.inputs)
        {
            auto polynomial = Polynomial::from_evaluations(
                input.name,
                input.values,
                input.domain);
            const auto candidate_it =
                candidate.trace.polynomials.find(input.name);
            if (candidate_it == candidate.trace.polynomials.end())
            {
                throw std::runtime_error(
                    "candidate trace omitted polynomial: " + input.name);
            }
            // Capture the exact candidate blinder from the same run.  This
            // makes legacy/candidate/CPU point equality meaningful without
            // replacing the production random blinder mechanism.
            polynomial.zero_domain_blinder =
                candidate_it->second.zero_domain_blinder;
            polynomials.push_back(std::move(polynomial));
        }

        std::vector<std::size_t> order(payload.inputs.size());
        for (std::size_t index = 0; index < order.size(); ++index)
        {
            order[index] = index;
        }
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](std::size_t lhs, std::size_t rhs)
            {
                if (payload.counts[lhs] != payload.counts[rhs])
                {
                    return payload.counts[lhs] > payload.counts[rhs];
                }
                return lhs < rhs;
            });

        LegacyRun result;
        result.commitments.resize(payload.inputs.size());
        const auto before_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_kernel =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto start = std::chrono::steady_clock::now();
        std::size_t offset = 0;
        while (offset < order.size())
        {
            const auto length = payload.counts[order[offset]];
            std::size_t end = offset + 1;
            while (end < order.size() &&
                   payload.counts[order[end]] == length)
            {
                ++end;
            }

            std::vector<std::pair<std::string, const Polynomial *>> named;
            named.reserve(end - offset);
            for (std::size_t position = offset; position < end; ++position)
            {
                const auto index = order[position];
                named.push_back({
                    payload.inputs[index].name,
                    &polynomials[index]});
            }

            CommitBatchProfile group_profile;
            std::vector<Commitment> group_commitments;
            {
                gatzk::protocol::timing::ScopedCommitment commitment_scope(
                    nullptr);
                group_commitments = KZG::commit_batch(
                    named,
                    key,
                    &group_profile,
                    nullptr);
            }
            // The wall/profile values are the production KZG baseline.  A
            // null scope above avoids attributing this diagnostic baseline to
            // formal RunMetrics; the candidate seam itself records the real
            // ScopedCommitment interval.
            accumulate_profile(&result.profile, group_profile);
            for (std::size_t local = 0;
                 local < group_commitments.size();
                 ++local)
            {
                result.commitments[order[offset + local]] =
                    std::move(group_commitments[local]);
            }
            offset = end;
        }
        const auto stop = std::chrono::steady_clock::now();
        result.wall_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        result.scoped_commitment_ms = 0.0;
        const auto after_stats =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto after_kernel =
            gatzk::algebra::cuda_kernel_invocation_count();
        result.route = route_delta(
            before_stats,
            after_stats,
            before_kernel,
            after_kernel);
        return result;
    }

    struct Trial
    {
        CandidateRun candidate;
        LegacyRun legacy;
        LegacyRun cpu;
    };

    Trial run_trial(
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact,
        const std::string &route)
    {
        configure_route(route);
        Trial result;
        result.candidate = run_candidate(payload, key, artifact);

        result.legacy = run_legacy(payload, result.candidate, key);
        require_equal(
            result.candidate.dynamic.commitments,
            result.legacy.commitments,
            "trace candidate/legacy");

        // Re-run the exact captured blinded payload through the canonical CPU
        // KZG baseline.  No second random blinder is sampled.
        configure_route("cpu");
        result.cpu = run_legacy(payload, result.candidate, key);
        require_equal(
            result.candidate.dynamic.commitments,
            result.cpu.commitments,
            "trace candidate/CPU");
        configure_route(route);
        return result;
    }

    void write_trial(
        std::ostream &out,
        std::size_t batch_size,
        std::size_t repeat,
        std::string_view phase,
        const Trial &trial)
    {
        const auto &candidate = trial.candidate;
        const auto &stats = candidate.route.stats;
        out << "family=dynamic_trace"
            << " lane_batch=" << batch_size
            << " objects=" << candidate.dynamic.commitments.size()
            << " phase=" << phase
            << " repeat=" << repeat
            << " candidate_dynamic_batch_wall_ms="
            << std::fixed << std::setprecision(3)
            << candidate.dynamic.wall_ms
            << " candidate_scoped_commitment_ms="
            << candidate.dynamic.scoped_commitment_ms
            << " candidate_dynamic_commit_ms="
            << candidate.dynamic.dynamic_commit_ms
            << " candidate_dynamic_commit_msm_ms="
            << candidate.dynamic.dynamic_commit_msm_ms
            << " legacy_kzg_wall_ms=" << trial.legacy.wall_ms
            << " legacy_profile_msm_ms=" << trial.legacy.profile.msm_ms
            << " cpu_kzg_wall_ms=" << trial.cpu.wall_ms
            << " cpu_profile_msm_ms=" << trial.cpu.profile.msm_ms
            << " digest=" << candidate.digest
            << " legacy_digest=" << commitment_digest(trial.legacy.commitments)
            << " candidate_kernel_invocations="
            << candidate.route.kernel_invocations
            << " candidate_msm_batch_calls=" << stats.msm_batch_calls
            << " candidate_msm_single_calls=" << stats.msm_single_calls
            << " candidate_resident_hits=" << stats.resident_hits
            << " candidate_resident_misses=" << stats.resident_misses
            << " candidate_h2d_bytes=" << stats.h2d_bytes
            << " candidate_d2h_bytes=" << stats.d2h_bytes
            << " candidate_h2d_operations=" << stats.h2d_operations
            << " candidate_d2h_operations=" << stats.d2h_operations
            << " candidate_syncs=" << stats.stream_synchronizations
            << " rss_before=" << candidate.rss_before
            << " rss_after=" << candidate.rss_after
            << " exact=candidate==legacy==cpu\n";
    }

    void run_series(
        std::ostream &out,
        const Options &options,
        std::size_t batch_size,
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact)
    {
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
        for (std::size_t warmup = 0;
             warmup < options.warmups;
             ++warmup)
        {
            const auto trial = run_trial(
                payload,
                key,
                artifact,
                options.route);
            write_trial(out, batch_size, warmup, "warmup", trial);
        }
        for (std::size_t repeat = 0;
             repeat < options.repeats;
             ++repeat)
        {
            const auto trial = run_trial(
                payload,
                key,
                artifact,
                options.route);
            write_trial(out, batch_size, repeat, "measured", trial);
        }
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
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

        const auto verifier =
            gatzk::protocol::load_verifier_context_from_registry(
                options.registry_path.string(),
                options.registry_digest);
        if (verifier.compiler_artifact == nullptr)
        {
            throw std::runtime_error(
                "authenticated PPI registry omitted compiler artifact");
        }
        if (verifier.domains.d == nullptr ||
            verifier.domains.d->size != kDomainSize)
        {
            throw std::runtime_error(
                "authenticated PPI registry does not expose d:16");
        }
        // The d:16 domain is part of this targeted workload contract, but the
        // names and blinder lengths still come only from the authenticated
        // compiler ledger.  Do not reinterpret an `in`/`edge` entry as d.
        const auto ledger_entries = select_private_entries(
            *verifier.compiler_artifact,
            "d",
            verifier.domains.d->size,
            kMaximumBatchSize * 3);
        std::size_t required_coefficient_count = 0;
        for (const auto &entry : ledger_entries)
        {
            if (entry.domain_size != kDomainSize ||
                entry.query_bound == 0 ||
                entry.domain_size >
                    std::numeric_limits<std::size_t>::max() -
                        entry.query_bound)
            {
                throw std::runtime_error(
                    "invalid authenticated d-domain ledger bounds: " +
                    entry.name);
            }
            required_coefficient_count = std::max(
                required_coefficient_count,
                entry.domain_size + entry.query_bound);
        }
        if (required_coefficient_count == 0)
        {
            throw std::runtime_error(
                "authenticated d-domain ledger has no blinder capacity");
        }

        configure_route(options.route);
        const auto key = KZG::load_keys_for_domains(
            options.srs_path.string(),
            "trusted_srs",
            "ppi_formal_srs",
            { {"d", kDomainSize} },
            // The zero-polynomial audit needs g1^(tau^d) plus every selected
            // ledger blinder coefficient.  Request exactly that authenticated
            // prefix instead of materializing the full coefficient SRS.
            required_coefficient_count);
        const auto manifest_domain_it =
            key.manifest.domain_sizes.find("d");
        if (manifest_domain_it == key.manifest.domain_sizes.end() ||
            manifest_domain_it->second != kDomainSize ||
            key.pk.powers_g1 == nullptr ||
            key.pk.powers_g1->size() < required_coefficient_count)
        {
            throw std::runtime_error(
                "authenticated SRS manifest/domain does not cover d:16 "
                "dynamic blinder prefix");
        }
        const auto domain = verifier.domains.d;

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
                    "failed to open dynamic benchmark output: " +
                    options.output_path.string());
            }
            output = &output_file;
        }

        *output << "benchmark=trace_add_dynamic_commitment_batch"
                << " route=" << options.route
                << " srs_id=" << key.pk.srs_id
                << " srs_type=" << key.pk.srs_type
                << " loaded_coefficient_capacity="
                << key.pk.powers_g1->size()
                << " loaded_lagrange_domain=d:16"
                << " authenticated_manifest_hash="
                << key.manifest.manifest_hash
                << " authenticated_registry_digest="
                << options.registry_digest
                << " warmups=" << options.warmups
                << " repeats=" << options.repeats
                << " required_coefficient_count="
                << required_coefficient_count
                << " ledger_entries=";
        for (std::size_t index = 0;
             index < ledger_entries.size();
             ++index)
        {
            if (index != 0)
            {
                *output << ',';
            }
            const auto &entry = ledger_entries[index];
            *output << entry.name << ":domain=" << entry.domain
                    << ":size=" << entry.domain_size
                    << ":query_bound=" << entry.query_bound;
        }
        *output << '\n';

        for (const auto batch_size : options.batch_sizes)
        {
            const auto payload = build_payload(
                batch_size,
                domain,
                ledger_entries);
            run_series(
                *output,
                options,
                batch_size,
                payload,
                key,
                verifier.compiler_artifact);
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
