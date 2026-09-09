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
#include <unistd.h>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/protocol/schema.hpp"
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
    using gatzk::protocol::DynamicCommitmentBatchTestResult;
    using gatzk::protocol::DynamicCommitmentMatrixBatchTestInput;
    using gatzk::protocol::ProtocolCompilerArtifact;
    using gatzk::protocol::TraceArtifacts;
    using gatzk::protocol::VerifierContext;

    constexpr std::string_view kRegistryPath =
        "artifacts/verifier_registry/ppi/verifier_registry.txt";
    constexpr std::string_view kRegistryDigest =
        "45fb77629c03df20453cc7baff5417f3cc023da3a9a015c1338ff8136e582212";
    constexpr std::size_t kTargetCount = 3;
    constexpr std::size_t kDefaultWarmups = 1;
    constexpr std::size_t kDefaultRepeats = 3;
    constexpr double kSmokeRssLimit = 0.60;

    struct Options
    {
        std::string route = "cuda";
        std::filesystem::path srs_path = "artifacts/srs/ppi_formal_srs";
        std::filesystem::path registry_path = std::string(kRegistryPath);
        std::string registry_digest = std::string(kRegistryDigest);
        std::filesystem::path output_path;
        std::size_t warmups = kDefaultWarmups;
        std::size_t repeats = kDefaultRepeats;
        bool smoke = false;
    };

    struct MatrixTarget
    {
        std::string name;
        std::size_t rows = 0;
        std::size_t columns = 0;
        std::size_t count = 0;
        std::size_t degree_bound = 0;
        std::size_t query_bound = 0;
    };

    struct Payload
    {
        std::vector<DynamicCommitmentMatrixBatchTestInput> inputs;
        std::vector<MatrixTarget> targets;
    };

    [[noreturn]] void usage_error(const std::string &message)
    {
        throw std::runtime_error(
            message +
            "\nusage: gatzk_kzg_dynamic_trace_matrix_msm_targeted_bench "
            "[--route cpu|cuda] [--srs-path PATH] [--registry PATH] "
            "[--registry-digest HEX] [--output PATH] [--smoke] "
            "[--warmups N] [--repeats N]");
    }

    std::size_t parse_size(const char *value, const char *label, bool allow_zero = false)
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
        if (consumed != std::string(value).size() || (!allow_zero && parsed == 0))
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
                    usage_error("--srs-path requires PATH");
                options.srs_path = argv[index];
            }
            else if (argument == "--registry")
            {
                if (++index >= argc)
                    usage_error("--registry requires PATH");
                options.registry_path = argv[index];
            }
            else if (argument == "--registry-digest")
            {
                if (++index >= argc)
                    usage_error("--registry-digest requires HEX");
                options.registry_digest = argv[index];
            }
            else if (argument == "--output")
            {
                if (++index >= argc)
                    usage_error("--output requires PATH");
                options.output_path = argv[index];
            }
            else if (argument == "--warmups")
            {
                if (++index >= argc)
                    usage_error("--warmups requires N");
                options.warmups = parse_size(argv[index], "warmups", true);
            }
            else if (argument == "--repeats")
            {
                if (++index >= argc)
                    usage_error("--repeats requires N");
                options.repeats = parse_size(argv[index], "repeats", true);
            }
            else if (argument == "--smoke")
            {
                options.smoke = true;
            }
            else if (argument == "--help" || argument == "-h")
            {
                std::cout
                    << "usage: gatzk_kzg_dynamic_trace_matrix_msm_targeted_bench "
                       "[--route cpu|cuda] [--srs-path PATH] [--registry PATH] "
                       "[--registry-digest HEX] [--output PATH] [--smoke] "
                       "[--warmups N] [--repeats N]\n";
                std::exit(0);
            }
            else
            {
                usage_error("unknown option: " + argument);
            }
        }
        if (options.smoke)
        {
            options.warmups = 0;
            options.repeats = 1;
        }
        if (options.warmups == 0 && options.repeats == 0)
            usage_error("warmups and repeats cannot both be zero");
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
            return 0;
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
    }

    std::uint64_t physical_memory_bytes()
    {
        const auto pages = sysconf(_SC_PHYS_PAGES);
        const auto page_size = sysconf(_SC_PAGESIZE);
        if (pages <= 0 || page_size <= 0)
            return 0;
        const auto page_count = static_cast<std::uint64_t>(pages);
        const auto bytes_per_page = static_cast<std::uint64_t>(page_size);
        if (page_count > std::numeric_limits<std::uint64_t>::max() / bytes_per_page)
            return 0;
        return page_count * bytes_per_page;
    }

    std::string available_private_coef_entries(
        const ProtocolCompilerArtifact &artifact)
    {
        std::string listed;
        for (const auto &entry : artifact.proof_schema.entries)
        {
            if (entry.domain != "coef" ||
                entry.visibility != gatzk::protocol::PolynomialVisibility::PrivateWitness ||
                !entry.enters_commitment)
                continue;
            const auto ledger_it = std::find_if(
                artifact.observation_ledger.begin(),
                artifact.observation_ledger.end(),
                [&](const auto &candidate)
                {
                    return candidate.object_or_challenge == entry.tag &&
                           candidate.domain == "coef";
                });
            if (ledger_it == artifact.observation_ledger.end())
                continue;
            if (!listed.empty())
                listed.push_back(',');
            listed += entry.tag + ":degree=" +
                      std::to_string(ledger_it->degree_bound) +
                      ":query=" + std::to_string(ledger_it->query_bound);
        }
        return listed;
    }

    MatrixTarget authenticate_target(
        const ProtocolCompilerArtifact &artifact,
        std::string name,
        std::size_t rows,
        std::size_t columns)
    {
        if (name.empty() || rows == 0 || columns == 0)
            throw std::runtime_error("invalid production matrix target shape: " + name);
        if (columns > std::numeric_limits<std::size_t>::max() / rows)
            throw std::runtime_error("production matrix target shape overflows: " + name);
        const auto count = rows * columns;

        const auto schema_it = std::find_if(
            artifact.proof_schema.entries.begin(),
            artifact.proof_schema.entries.end(),
            [&](const auto &entry)
            { return entry.tag == name; });
        if (schema_it == artifact.proof_schema.entries.end() ||
            schema_it->domain != "coef" ||
            schema_it->visibility != gatzk::protocol::PolynomialVisibility::PrivateWitness ||
            !schema_it->enters_commitment)
        {
            throw std::runtime_error(
                "production matrix target is not an authenticated private coef commitment: " +
                name + "; available_private_coef_entries=" +
                available_private_coef_entries(artifact));
        }

        const auto ledger_it = std::find_if(
            artifact.observation_ledger.begin(),
            artifact.observation_ledger.end(),
            [&](const auto &entry)
            {
                return entry.object_or_challenge == name &&
                       entry.domain == "coef";
            });
        if (ledger_it == artifact.observation_ledger.end())
        {
            throw std::runtime_error(
                "production matrix target is missing an authenticated coef observation ledger entry: " +
                name + "; available_private_coef_entries=" +
                available_private_coef_entries(artifact));
        }
        if (ledger_it->degree_bound == std::numeric_limits<std::size_t>::max() ||
            count > ledger_it->degree_bound + 1)
        {
            throw std::runtime_error(
                "production matrix target exceeds authenticated coefficient degree bound: " +
                name + " count=" + std::to_string(count) +
                " degree_bound=" + std::to_string(ledger_it->degree_bound));
        }
        return MatrixTarget{
            .name = std::move(name),
            .rows = rows,
            .columns = columns,
            .count = count,
            .degree_bound = ledger_it->degree_bound,
            .query_bound = ledger_it->query_bound,
        };
    }

    std::vector<MatrixTarget> select_production_targets(
        const VerifierContext &verifier)
    {
        if (verifier.compiler_artifact == nullptr)
            throw std::runtime_error("authenticated PPI registry omitted compiler artifact");
        if (verifier.local.num_nodes == 0 ||
            verifier.model.hidden_layers.empty() ||
            verifier.model.hidden_layers.front().shape.head_dim == 0 ||
            verifier.model.hidden_layers.front().shape.head_count == 0 ||
            verifier.model.output_layer.head_count == 0 ||
            verifier.model.C == 0)
        {
            throw std::runtime_error("authenticated PPI registry has incomplete production model shape");
        }

        const auto &hidden = verifier.model.hidden_layers.front();
        const auto hidden_width = hidden.shape.head_dim;
        if (hidden.shape.head_count >
            std::numeric_limits<std::size_t>::max() / hidden_width)
            throw std::runtime_error("authenticated hidden concat width overflows");
        const auto concat_width = hidden.shape.head_count * hidden_width;
        const auto rows = verifier.local.num_nodes;
        return {
            authenticate_target(
                *verifier.compiler_artifact,
                gatzk::protocol::hidden_head_prefix(0, 0) + "_H_prime",
                rows,
                hidden_width),
            authenticate_target(
                *verifier.compiler_artifact,
                gatzk::protocol::hidden_layer_concat_label(0),
                rows,
                concat_width),
            authenticate_target(
                *verifier.compiler_artifact,
                gatzk::protocol::output_y_lin_label(0),
                rows,
                verifier.model.C),
        };
    }

    Payload build_payload(const std::vector<MatrixTarget> &targets)
    {
        Payload payload;
        payload.targets = targets;
        payload.inputs.reserve(targets.size());
        for (std::size_t target_index = 0; target_index < targets.size(); ++target_index)
        {
            const auto &target = targets[target_index];
            gatzk::model::Matrix matrix;
            matrix.resize(target.rows);
            for (std::size_t row = 0; row < target.rows; ++row)
            {
                matrix[row].resize(target.columns);
                for (std::size_t column = 0; column < target.columns; ++column)
                {
                    const auto raw =
                        17ULL +
                        ((static_cast<std::uint64_t>(row) * 1315423911ULL +
                          static_cast<std::uint64_t>(column) * 2654435761ULL +
                          static_cast<std::uint64_t>(target_index + 1) * 889523592379ULL) %
                         1000000007ULL);
                    matrix[row][column] = FieldElement(raw);
                    if (matrix[row][column].is_zero())
                        throw std::runtime_error("deterministic matrix payload contains zero");
                }
            }
            payload.inputs.push_back({
                .name = target.name,
                .matrix = std::move(matrix),
            });
        }
        return payload;
    }

    std::uint64_t commitment_digest(const std::vector<Commitment> &commitments)
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
            throw std::runtime_error(std::string(label) + " commitment count mismatch");
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
        const auto before_stats = gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_kernel = gatzk::algebra::cuda_kernel_invocation_count();
        CandidateRun result;
        result.rss_before = current_rss_bytes();
        result.dynamic =
            gatzk::protocol::add_dynamic_commitment_matrix_batch_for_tests(
                trace,
                payload.inputs,
                key,
                false);
        const auto after_stats = gatzk::algebra::cuda_execution_stats_snapshot();
        const auto after_kernel = gatzk::algebra::cuda_kernel_invocation_count();
        result.route = route_delta(before_stats, after_stats, before_kernel, after_kernel);
        result.digest = commitment_digest(result.dynamic.commitments);
        result.rss_after = current_rss_bytes();
        for (const auto &input : payload.inputs)
        {
            const auto polynomial_it = trace.polynomials.find(input.name);
            if (polynomial_it == trace.polynomials.end())
                throw std::runtime_error("matrix candidate omitted polynomial: " + input.name);
            const auto expected_count =
                input.matrix.size() * input.matrix.front().size();
            if (polynomial_it->second.basis != gatzk::algebra::PolynomialBasis::Coefficient ||
                !polynomial_it->second.zero_domain_blinder.empty() ||
                polynomial_it->second.size() != expected_count)
            {
                throw std::runtime_error(
                    "matrix candidate changed coefficient payload/blinder semantics: " +
                    input.name);
            }
        }
        result.trace = std::move(trace);
        return result;
    }

    void accumulate_profile(CommitBatchProfile *target, const CommitBatchProfile &source)
    {
        target->evaluation_ms += source.evaluation_ms;
        target->msm_ms += source.msm_ms;
        target->objects.insert(target->objects.end(), source.objects.begin(), source.objects.end());
    }

    struct LegacyRun
    {
        std::vector<Commitment> commitments;
        CommitBatchProfile profile;
        double wall_ms = 0.0;
        double scoped_commitment_ms = 0.0;
        RouteDelta route;
        std::uint64_t rss_before = 0;
        std::uint64_t rss_after = 0;
    };

    LegacyRun run_legacy(const Payload &payload, const CandidateRun &candidate, const KZGKeyPair &key)
    {
        LegacyRun result;
        result.commitments.resize(payload.inputs.size());
        const auto before_stats = gatzk::algebra::cuda_execution_stats_snapshot();
        const auto before_kernel = gatzk::algebra::cuda_kernel_invocation_count();
        result.rss_before = current_rss_bytes();
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < payload.inputs.size(); ++index)
        {
            const auto polynomial_it = candidate.trace.polynomials.find(payload.inputs[index].name);
            if (polynomial_it == candidate.trace.polynomials.end())
                throw std::runtime_error("legacy baseline omitted polynomial: " + payload.inputs[index].name);
            std::vector<std::pair<std::string, const Polynomial *>> named{
                {payload.inputs[index].name, &polynomial_it->second}};
            CommitBatchProfile profile;
            gatzk::protocol::RunMetrics metrics;
            std::vector<Commitment> commitments;
            {
                gatzk::protocol::timing::ScopedCommitment scope(&metrics);
                commitments = KZG::commit_batch(named, key, &profile, nullptr);
            }
            if (commitments.size() != 1)
                throw std::runtime_error("legacy exact wave returned wrong object count");
            result.commitments[index] = std::move(commitments.front());
            result.scoped_commitment_ms += metrics.commitment_time_ms;
            accumulate_profile(&result.profile, profile);
        }
        result.wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        const auto after_stats = gatzk::algebra::cuda_execution_stats_snapshot();
        const auto after_kernel = gatzk::algebra::cuda_kernel_invocation_count();
        result.route = route_delta(before_stats, after_stats, before_kernel, after_kernel);
        result.rss_after = current_rss_bytes();
        return result;
    }

    struct Trial
    {
        CandidateRun candidate;
        LegacyRun legacy;
    };

    Trial run_trial(
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact,
        const std::string &route)
    {
        configure_route(route);
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
        Trial result;
        result.candidate = run_candidate(payload, key, artifact);

        // Reset the resident cache before the exact-wave baseline so both
        // lanes report a comparable cold first invocation.  The two later
        // legacy objects still exercise the resident steady path.
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
        result.legacy = run_legacy(payload, result.candidate, key);
        require_equal(result.candidate.dynamic.commitments, result.legacy.commitments, "matrix candidate/legacy");
        return result;
    }

    void write_profiles(
        std::ostream &out,
        std::string_view lane,
        const std::vector<gatzk::protocol::DynamicCommitmentObjectProfile> &profiles)
    {
        for (const auto &profile : profiles)
        {
            out << "profile lane=" << lane
                << " label=" << profile.label
                << " vector_length=" << profile.vector_length
                << " domain_size=" << profile.domain_size
                << " basis=" << profile.basis
                << " msm_ms=" << std::fixed << std::setprecision(3)
                << profile.msm_ms
                << " total_commitment_ms=" << profile.total_commitment_ms
                << " effective_scalar_count=" << profile.effective_scalar_count
                << " resident_profile=" << (profile.shared_or_borrowed ? "shared" : "owned")
                << '\n';
        }
    }

    void write_trial(
        std::ostream &out,
        const std::vector<MatrixTarget> &targets,
        std::string_view phase,
        std::size_t repeat,
        const Trial &trial,
        std::uint64_t physical_memory)
    {
        const auto &candidate = trial.candidate;
        const auto &legacy = trial.legacy;
        const auto &candidate_stats = candidate.route.stats;
        const auto &legacy_stats = legacy.route.stats;
        const auto rss_ratio = [&](std::uint64_t value)
        {
            return physical_memory == 0
                       ? 0.0
                       : static_cast<double>(value) /
                             static_cast<double>(physical_memory);
        };
        out << "family=dynamic_trace_matrix"
            << " phase=" << phase
            << " repeat=" << repeat
            << " objects=" << targets.size()
            << " candidate_dynamic_batch_wall_ms=" << std::fixed << std::setprecision(3)
            << candidate.dynamic.wall_ms
            << " candidate_scoped_commitment_ms=" << candidate.dynamic.scoped_commitment_ms
            << " candidate_dynamic_commit_ms=" << candidate.dynamic.dynamic_commit_ms
            << " candidate_dynamic_commit_msm_ms=" << candidate.dynamic.dynamic_commit_msm_ms
            << " legacy_exact_waves_wall_ms=" << legacy.wall_ms
            << " legacy_scoped_commitment_ms=" << legacy.scoped_commitment_ms
            << " legacy_profile_msm_ms=" << legacy.profile.msm_ms
            << " candidate_digest=" << candidate.digest
            << " legacy_digest=" << commitment_digest(legacy.commitments)
            << " candidate_kernel_invocations=" << candidate.route.kernel_invocations
            << " candidate_msm_batch_calls=" << candidate_stats.msm_batch_calls
            << " candidate_msm_single_calls=" << candidate_stats.msm_single_calls
            << " candidate_resident_hits=" << candidate_stats.resident_hits
            << " candidate_resident_misses=" << candidate_stats.resident_misses
            << " candidate_h2d_bytes=" << candidate_stats.h2d_bytes
            << " candidate_d2h_bytes=" << candidate_stats.d2h_bytes
            << " candidate_h2d_operations=" << candidate_stats.h2d_operations
            << " candidate_d2h_operations=" << candidate_stats.d2h_operations
            << " candidate_syncs=" << candidate_stats.stream_synchronizations
            << " legacy_kernel_invocations=" << legacy.route.kernel_invocations
            << " legacy_msm_batch_calls=" << legacy_stats.msm_batch_calls
            << " legacy_msm_single_calls=" << legacy_stats.msm_single_calls
            << " legacy_resident_hits=" << legacy_stats.resident_hits
            << " legacy_resident_misses=" << legacy_stats.resident_misses
            << " legacy_h2d_bytes=" << legacy_stats.h2d_bytes
            << " legacy_d2h_bytes=" << legacy_stats.d2h_bytes
            << " candidate_rss_before=" << candidate.rss_before
            << " candidate_rss_after=" << candidate.rss_after
            << " candidate_rss_fraction=" << rss_ratio(candidate.rss_after)
            << " legacy_rss_before=" << legacy.rss_before
            << " legacy_rss_after=" << legacy.rss_after
            << " legacy_rss_fraction=" << rss_ratio(legacy.rss_after)
            << " semantic_reference=legacy_exact_cuda_or_cpu"
            << " cpu_reference=skipped_large_production_shape_existing_d16_cpu_exact_retained"
            << " exact=candidate==legacy\n";
        for (const auto &target : targets)
        {
            out << "shape name=" << target.name
                << " rows=" << target.rows
                << " columns=" << target.columns
                << " count=" << target.count
                << " ledger_degree_bound=" << target.degree_bound
                << " ledger_query_bound=" << target.query_bound
                << '\n';
        }
        write_profiles(out, "candidate", candidate.dynamic.object_profiles);
        std::vector<gatzk::protocol::DynamicCommitmentObjectProfile> legacy_profiles;
        for (const auto &profile : legacy.profile.objects)
        {
            gatzk::protocol::DynamicCommitmentObjectProfile converted;
            converted.label = profile.name;
            converted.basis = profile.basis;
            converted.domain = profile.domain;
            converted.vector_length = profile.vector_length;
            converted.effective_scalar_count = profile.effective_scalar_count;
            converted.msm_ms = profile.msm_ms;
            converted.total_commitment_ms = profile.msm_ms;
            legacy_profiles.push_back(std::move(converted));
        }
        write_profiles(out, "legacy", legacy_profiles);
    }

    void run_series(
        std::ostream &out,
        const Options &options,
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact,
        std::uint64_t physical_memory)
    {
        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
        {
            const auto trial = run_trial(payload, key, artifact, options.route);
            write_trial(out, payload.targets, "warmup", warmup, trial, physical_memory);
        }
        for (std::size_t repeat = 0; repeat < options.repeats; ++repeat)
        {
            const auto trial = run_trial(payload, key, artifact, options.route);
            write_trial(out, payload.targets, "measured", repeat, trial, physical_memory);
        }
        gatzk::crypto::release_cuda_g1_msm_resident_cache();
    }
}

int main(int argc, char **argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        if (options.route == "cuda" && !gatzk::algebra::cuda_backend_available())
        {
            std::cout << "status=unavailable route=cuda\n";
            return 77;
        }
        // The production trace seam emits object profiles only when detailed
        // profiling is explicitly requested.  This is a test-process setting;
        // it does not alter the protocol payload or commitment path.
        setenv("GATZK_DETAILED_PROFILE", "1", 1);

        const auto verifier = gatzk::protocol::load_verifier_context_from_registry(
            options.registry_path.string(), options.registry_digest);
        if (verifier.compiler_artifact == nullptr)
            throw std::runtime_error("authenticated PPI registry omitted compiler artifact");
        const auto targets = select_production_targets(verifier);
        if (targets.size() != kTargetCount)
            throw std::runtime_error("production matrix target selection changed target count");

        std::size_t required_coefficient_count = 0;
        for (const auto &target : targets)
            required_coefficient_count = std::max(required_coefficient_count, target.count);
        if (required_coefficient_count == 0)
            throw std::runtime_error("production matrix target coefficient prefix is empty");

        configure_route(options.route);
        const auto d_domain = verifier.domains.d;
        if (d_domain == nullptr || d_domain->size == 0)
            throw std::runtime_error("authenticated PPI registry omitted d evaluation domain");
        const auto key = KZG::load_keys_for_domains(
            options.srs_path.string(),
            "trusted_srs",
            "ppi_formal_srs",
            {{"d", d_domain->size}},
            required_coefficient_count);
        const auto manifest_d_it = key.manifest.domain_sizes.find("d");
        if (manifest_d_it == key.manifest.domain_sizes.end() ||
            manifest_d_it->second != d_domain->size ||
            key.pk.powers_g1 == nullptr ||
            key.pk.powers_g1->size() < required_coefficient_count)
        {
            throw std::runtime_error("authenticated PPI SRS does not cover d evaluation and matrix coefficient prefix");
        }

        std::ofstream output_file;
        std::ostream *output = &std::cout;
        if (!options.output_path.empty())
        {
            if (!options.output_path.parent_path().empty())
                std::filesystem::create_directories(options.output_path.parent_path());
            output_file.open(options.output_path);
            if (!output_file)
                throw std::runtime_error("failed to open matrix benchmark output: " + options.output_path.string());
            output = &output_file;
        }

        const auto physical_memory = physical_memory_bytes();
        *output << "benchmark=trace_add_dynamic_commitment_matrix_batch"
                << " route=" << options.route
                << " smoke=" << (options.smoke ? "true" : "false")
                << " warmups=" << options.warmups
                << " repeats=" << options.repeats
                << " srs_id=" << key.pk.srs_id
                << " srs_type=" << key.pk.srs_type
                << " authenticated_manifest_hash=" << key.manifest.manifest_hash
                << " authenticated_registry_digest=" << options.registry_digest
                << " loaded_coefficient_capacity=" << key.pk.powers_g1->size()
                << " loaded_lagrange_domain=d:" << d_domain->size
                << " required_coefficient_count=" << required_coefficient_count
                << " physical_memory_bytes=" << physical_memory
                << " candidate_lane=one_authenticated_production_batch"
                << " legacy_lane=three_exact_single_object_waves"
                << '\n';
        for (const auto &target : targets)
        {
            *output << "target name=" << target.name
                    << " rows=" << target.rows
                    << " columns=" << target.columns
                    << " count=" << target.count
                    << " ledger_degree_bound=" << target.degree_bound
                    << " ledger_query_bound=" << target.query_bound
                    << '\n';
        }

        const auto payload = build_payload(targets);
        run_series(*output, options, payload, key, verifier.compiler_artifact, physical_memory);
        if (options.smoke && physical_memory != 0)
        {
            // The smoke path is the admission check for this production-shaped
            // probe.  A resident/streamed implementation that exceeds the
            // agreed process-memory envelope is a failed smoke, not a usable
            // performance datapoint.
            const auto rss_fraction = static_cast<double>(current_rss_bytes()) /
                                       static_cast<double>(physical_memory);
            if (rss_fraction >= kSmokeRssLimit)
            {
                throw std::runtime_error(
                    "matrix production-shape smoke exceeded RSS limit: fraction=" +
                    std::to_string(rss_fraction));
            }
            *output << "smoke_rss_limit_fraction=" << kSmokeRssLimit
                    << " smoke_rss_limit_pass=true\n";
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
