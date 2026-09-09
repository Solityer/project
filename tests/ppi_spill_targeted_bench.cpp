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
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/resource.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    using gatzk::algebra::DomainEvaluationWeights;
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::RootOfUnityDomain;
    using gatzk::crypto::BatchOpeningProfile;
    using gatzk::crypto::Commitment;
    using gatzk::crypto::KZG;
    using gatzk::crypto::KZGKeyPair;
    using gatzk::crypto::OpeningSemanticSourceInfo;
    using gatzk::protocol::DynamicCommitmentBatchTestInput;
    using gatzk::protocol::DynamicCommitmentBatchTestResult;
    using gatzk::protocol::ProtocolCompilerArtifact;
    using gatzk::protocol::TraceArtifacts;

    constexpr std::size_t kMinimumDomainSize = 1U << 18U;
    constexpr std::size_t kMaximumPairs = 8;
    constexpr std::size_t kDefaultPairs = 2;
    constexpr std::size_t kDefaultWarmups = 1;
    constexpr std::size_t kDefaultRepeats = 5;
    constexpr std::string_view kRegistryPath =
        "artifacts/verifier_registry/ppi/verifier_registry.txt";
    constexpr std::string_view kRegistryDigest =
        "45fb77629c03df20453cc7baff5417f3cc023da3a9a015c1338ff8136e582212";

    struct Options
    {
        std::string route = "cpu";
        std::size_t pairs = kDefaultPairs;
        std::size_t warmups = kDefaultWarmups;
        std::size_t repeats = kDefaultRepeats;
        std::filesystem::path srs_path = "artifacts/srs/ppi_formal_srs";
        std::filesystem::path registry_path = std::string(kRegistryPath);
        std::string registry_digest = std::string(kRegistryDigest);
        std::filesystem::path output_path = "tests/ppi_spill_results";
        std::filesystem::path provenance_path;
        bool inspect_registry = false;
    };

    struct RunProvenance
    {
        std::string run_id = "unspecified";
        std::string git_head = "unspecified";
        std::string non_comparison_diff_sha256 = "unspecified";
        std::string benchmark_source_sha256 = "unspecified";
        std::string binary_sha256 = "unspecified";
    };

    [[noreturn]] void usage_error(const std::string &message)
    {
        throw std::runtime_error(
            message +
            "\nusage: gatzk_ppi_spill_targeted_bench "
            "[--route cpu|cuda] [--pairs 1|2|4|8] "
            "[--warmups N] [--repeats N] [--srs-path PATH] "
            "[--registry PATH] [--registry-digest HEX] [--output PATH] "
            "[--provenance PATH]");
    }

    std::size_t parse_size(
        const char *value,
        const char *label,
        bool allow_zero = false)
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
        if (consumed != std::string(value).size() ||
            (!allow_zero && parsed == 0))
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
            else if (argument == "--pairs")
            {
                if (++index >= argc)
                {
                    usage_error("--pairs requires N");
                }
                options.pairs = parse_size(argv[index], "pairs");
                if (options.pairs != 1 && options.pairs != 2 &&
                    options.pairs != 4 && options.pairs != 8)
                {
                    usage_error("--pairs must be 1, 2, 4, or 8");
                }
            }
            else if (argument == "--warmups")
            {
                if (++index >= argc)
                {
                    usage_error("--warmups requires N");
                }
                options.warmups = parse_size(argv[index], "warmups", true);
            }
            else if (argument == "--repeats")
            {
                if (++index >= argc)
                {
                    usage_error("--repeats requires N");
                }
                options.repeats = parse_size(argv[index], "repeats", true);
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
            else if (argument == "--provenance")
            {
                if (++index >= argc)
                {
                    usage_error("--provenance requires PATH");
                }
                options.provenance_path = argv[index];
            }
            else if (argument == "--inspect-registry")
            {
                options.inspect_registry = true;
            }
            else if (argument == "--help" || argument == "-h")
            {
                std::cout
                    << "usage: gatzk_ppi_spill_targeted_bench "
                       "[--route cpu|cuda] [--pairs 1|2|4|8] "
                       "[--warmups N] [--repeats N] [--srs-path PATH] "
                       "[--registry PATH] [--registry-digest HEX] "
                       "[--output PATH] [--provenance PATH] "
                       "[--inspect-registry]\n";
                std::exit(0);
            }
            else
            {
                usage_error("unknown option: " + argument);
            }
        }
        return options;
    }

    RunProvenance load_run_provenance(const std::filesystem::path &path)
    {
        RunProvenance provenance;
        if (path.empty())
        {
            return provenance;
        }
        std::ifstream input(path);
        if (!input)
        {
            throw std::runtime_error(
                "unable to open PPI spill benchmark provenance: " +
                path.string());
        }
        std::string line;
        while (std::getline(input, line))
        {
            const auto separator = line.find('=');
            if (separator == std::string::npos)
            {
                continue;
            }
            const auto key = line.substr(0, separator);
            auto value = line.substr(separator + 1);
            if (!value.empty() && value.back() == '\r')
            {
                value.pop_back();
            }
            if (key == "run_id")
            {
                provenance.run_id = std::move(value);
            }
            else if (key == "git_head")
            {
                provenance.git_head = std::move(value);
            }
            else if (key == "non_comparison_diff_sha256")
            {
                provenance.non_comparison_diff_sha256 = std::move(value);
            }
            else if (key == "benchmark_source_sha256")
            {
                provenance.benchmark_source_sha256 = std::move(value);
            }
            else if (key == "binary_sha256")
            {
                provenance.binary_sha256 = std::move(value);
            }
        }
        return provenance;
    }

    void configure_route(const std::string &route)
    {
        auto route_options = gatzk::util::route2_options();
        route_options.fast_msm = true;
        route_options.cuda_field_hotspots = route == "cuda";
        gatzk::util::set_route2_options(route_options);
    }

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    std::uint64_t current_rss_kb()
    {
        std::ifstream input("/proc/self/status");
        std::string line;
        while (std::getline(input, line))
        {
            if (line.rfind("VmRSS:", 0) == 0)
            {
                std::istringstream fields(line.substr(6));
                std::uint64_t value = 0;
                fields >> value;
                return value;
            }
        }
        return 0;
    }

    std::uint64_t max_rss_kb()
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

    struct PrivateEntry
    {
        std::string name;
        std::size_t query_bound = 0;
    };

    std::vector<PrivateEntry> select_private_entries(
        const ProtocolCompilerArtifact &artifact,
        const std::string &domain_name,
        std::size_t pair_count)
    {
        std::vector<PrivateEntry> result;
        result.reserve(pair_count * 2);

        // These are the exact names produced at trace.cpp:8024-8040 for one
        // signed-division limb.  The producer passes the same
        // `limb_column` view to both limb_label and query_label; selecting
        // those two labels as a pair preserves that production relation while
        // allowing the tests-only seam to fan out several relation instances.
        const std::array<std::string_view, 5> value_names{
            "A", "DMinus1", "Q", "R", "Delta"};
        const auto find_ledger = [&](const std::string &name)
        -> const gatzk::protocol::CompilerLedgerEntry *
        {
            const auto it = std::find_if(
                artifact.observation_ledger.begin(),
                artifact.observation_ledger.end(),
                [&](const auto &entry)
                {
                    return entry.object_or_challenge == name &&
                           entry.domain == domain_name &&
                           entry.query_bound != 0;
                });
            return it == artifact.observation_ledger.end() ? nullptr : &*it;
        };
        const auto require_schema = [&](const std::string &name)
        {
            const auto it = std::find_if(
                artifact.proof_schema.entries.begin(),
                artifact.proof_schema.entries.end(),
                [&](const auto &entry)
                { return entry.tag == name; });
            if (it == artifact.proof_schema.entries.end() ||
                it->domain != domain_name ||
                it->visibility !=
                    gatzk::protocol::PolynomialVisibility::PrivateWitness ||
                !it->enters_commitment || !it->enters_opening ||
                !it->enters_quotient)
            {
                throw std::runtime_error(
                    "authenticated PPI edge schema state is not private/commit/open/quotient: " +
                    name);
            }
        };

        for (const auto value_name : value_names)
        {
            for (std::size_t limb = 0; limb < 5 &&
                                         result.size() < pair_count * 2;
                 ++limb)
            {
                const auto prefix = std::string("P_h_l0_h0_Div_");
                const auto limb_name =
                    prefix + "Limb_" + std::string(value_name) + "_" +
                    std::to_string(limb);
                const auto query_name =
                    prefix + "Range_" + std::string(value_name) + "_" +
                    std::to_string(limb) + "_Query";
                const auto *limb_ledger = find_ledger(limb_name);
                const auto *query_ledger = find_ledger(query_name);
                if (limb_ledger == nullptr || query_ledger == nullptr)
                {
                    continue;
                }
                require_schema(limb_name);
                require_schema(query_name);
                result.push_back({limb_name, limb_ledger->query_bound});
                result.push_back({query_name, query_ledger->query_bound});
            }
            if (result.size() == pair_count * 2)
            {
                break;
            }
        }
        if (result.size() != pair_count * 2)
        {
            std::ostringstream error;
            error << "authenticated " << domain_name
                  << " private PPI signed-division alias pairs has only "
                  << result.size() / 2 << " suitable pairs; required "
                  << pair_count
                  << "; ledger candidates=";
            std::size_t listed = 0;
            for (const auto &entry : artifact.observation_ledger)
            {
                if (entry.domain != domain_name || entry.query_bound == 0)
                {
                    continue;
                }
                if (listed++ != 0)
                {
                    error << ',';
                }
                error << entry.object_or_challenge << ":q=" << entry.query_bound;
                if (listed == 32)
                {
                    error << ",...";
                    break;
                }
            }
            throw std::runtime_error(error.str());
        }
        return result;
    }

    std::uint64_t digest_bytes(
        std::uint64_t digest,
        const std::uint8_t *bytes,
        std::size_t count)
    {
        constexpr std::uint64_t kPrime = 1099511628211ULL;
        for (std::size_t index = 0; index < count; ++index)
        {
            digest ^= bytes[index];
            digest *= kPrime;
        }
        return digest;
    }

    std::uint64_t digest_text(
        std::uint64_t digest,
        std::string_view text)
    {
        return digest_bytes(
            digest,
            reinterpret_cast<const std::uint8_t *>(text.data()),
            text.size());
    }

    std::uint64_t digest_field(
        std::uint64_t digest,
        const FieldElement &value)
    {
        return digest_text(digest, value.to_string());
    }

    std::uint64_t commitment_digest(
        const std::vector<Commitment> &commitments)
    {
        constexpr std::uint64_t kOffset = 1469598103934665603ULL;
        std::uint64_t digest = kOffset;
        for (const auto &commitment : commitments)
        {
            const auto bytes = gatzk::crypto::serialize(commitment.point);
            digest = digest_bytes(digest, bytes.data(), bytes.size());
        }
        return digest;
    }

    std::uint64_t point_digest(const std::array<gatzk::crypto::G1Point, 2> &points)
    {
        constexpr std::uint64_t kOffset = 1469598103934665603ULL;
        std::uint64_t digest = kOffset;
        for (const auto &point : points)
        {
            const auto bytes = gatzk::crypto::serialize(point);
            digest = digest_bytes(digest, bytes.data(), bytes.size());
        }
        return digest;
    }

    FieldElement non_domain_point(
        const std::shared_ptr<RootOfUnityDomain> &domain,
        std::uint64_t seed)
    {
        auto point = FieldElement(seed);
        while (domain->rotation_shift(FieldElement::one(), point).has_value())
        {
            point += FieldElement::one();
        }
        return point;
    }

    struct SourceOwner
    {
        std::vector<std::vector<FieldElement>> columns;
    };

    struct Payload
    {
        std::vector<DynamicCommitmentBatchTestInput> inputs;
        std::uint64_t digest = 1469598103934665603ULL;
    };

    Payload build_payload(
        std::size_t pairs,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        const std::vector<PrivateEntry> &entries)
    {
        expect(domain != nullptr && domain->name == "edge" &&
                   domain->size >= kMinimumDomainSize,
               "targeted PPI spill workload requires authenticated edge >= 2^18");
        expect(entries.size() >= pairs * 2,
               "targeted PPI spill workload lacks authenticated entries");

        auto owner = std::make_shared<SourceOwner>();
        owner->columns.resize(pairs);
        Payload payload;
        payload.inputs.reserve(pairs * 2);

        for (std::size_t pair = 0; pair < pairs; ++pair)
        {
            auto &values = owner->columns[pair];
            values.resize(domain->size);
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                // Keep every physical source dense so the production spill
                // path cannot hide work behind zero-tail compaction.
                values[index] = FieldElement(
                    static_cast<std::uint64_t>(
                        17U + pair * 1000003U + index));
            }

            for (std::size_t member = 0; member < 2; ++member)
            {
                const auto entry_index = pair * 2 + member;
                const auto &entry = entries[entry_index];
                std::vector<FieldElement> blinder(entry.query_bound);
                for (std::size_t coefficient = 0;
                     coefficient < blinder.size();
                     ++coefficient)
                {
                    blinder[coefficient] = FieldElement(
                        static_cast<std::uint64_t>(
                            101U + pair * 1009U + member * 313U +
                            coefficient));
                }
                const auto view =
                    std::shared_ptr<const std::vector<FieldElement>>(
                        owner,
                        &values);
                payload.inputs.push_back({
                    .name = entry.name,
                    .values = {},
                    .shared_values = view,
                    .zero_domain_blinder_for_tests = std::move(blinder),
                    .domain = domain});

                payload.digest = digest_text(payload.digest, entry.name);
                payload.digest = digest_field(payload.digest, values.front());
                payload.digest = digest_field(payload.digest, values.back());
            }
        }
        return payload;
    }

    std::vector<FieldElement> evaluate_descriptor(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        const FieldElement &point,
        const DomainEvaluationWeights &weights)
    {
        return {
            gatzk::protocol::evaluate_spilled_evaluation_polynomial(
                spilled,
                domain,
                point,
                weights)};
    }

    struct LaneResult
    {
        bool reuse_enabled = false;
        TraceArtifacts trace;
        DynamicCommitmentBatchTestResult dynamic;
        std::vector<TraceArtifacts::SpilledEvaluationPolynomial> descriptors;
        std::array<std::vector<Commitment>, 2> groups;
        std::array<std::vector<FieldElement>, 2> claims;
        std::array<FieldElement, 2> points;
        std::array<FieldElement, 2> challenges;
        std::array<BatchOpeningProfile, 2> opening_profiles{};
        std::array<gatzk::crypto::G1Point, 2> witnesses{
            gatzk::crypto::g1_zero(),
            gatzk::crypto::g1_zero()};
        double claim_ms = 0.0;
        double opening_ms = 0.0;
        bool verification_passed = false;
        bool tamper_detected = false;
        std::size_t unique_spill_files = 0;
        std::size_t spill_bytes = 0;
        std::uint64_t rss_kb = 0;
    };

    std::string semantic_key(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled)
    {
        std::ostringstream key;
        key << (spilled.semantic_source_id == 0 ? "descriptor" : "authenticated")
            << '|' << spilled.semantic_source_id << '|' << spilled.domain_name
            << '|' << spilled.size << '|'
            << gatzk::protocol::spilled_file_value_count(spilled) << '|'
            << (spilled.zero_tail_compacted ? "compact" : "full") << '|'
            << spilled.path;
        return key.str();
    }

    gatzk::crypto::OpeningSemanticSourceResolver make_semantic_resolver(
        const std::unordered_map<std::string, const TraceArtifacts::SpilledEvaluationPolynomial *> &descriptors,
        const std::shared_ptr<RootOfUnityDomain> &domain)
    {
        return [&descriptors, domain](const Commitment &commitment)
                   -> std::optional<OpeningSemanticSourceInfo>
        {
            const auto descriptor_it = descriptors.find(commitment.name);
            if (descriptor_it == descriptors.end() ||
                descriptor_it->second == nullptr)
            {
                return std::nullopt;
            }
            const auto &spilled = *descriptor_it->second;
            if (spilled.lease == nullptr || spilled.path.empty())
            {
                throw std::runtime_error(
                    "targeted spill resolver received incomplete descriptor: " +
                    commitment.name);
            }
            return OpeningSemanticSourceInfo{
                .key = semantic_key(spilled),
                .domain = domain,
                .value_count = spilled.size,
                .stored_value_count =
                    gatzk::protocol::spilled_file_value_count(spilled),
                .zero_tail_compacted = spilled.zero_tail_compacted,
                .zero_domain_blinder = spilled.zero_domain_blinder,
                .file_path = spilled.path,
                .file_backed = true,
                .lease = spilled.lease};
        };
    }

    LaneResult run_lane(
        bool reuse_enabled,
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact,
        const std::shared_ptr<RootOfUnityDomain> &domain)
    {
        gatzk::protocol::set_exact_shared_evaluation_view_reuse_for_tests(
            reuse_enabled);
        LaneResult result;
        result.reuse_enabled = reuse_enabled;
        result.trace.compiler_artifact = artifact;
        const auto rss_before = current_rss_kb();

        const auto commit_start = std::chrono::steady_clock::now();
        result.dynamic =
            gatzk::protocol::add_dynamic_commitment_batch_for_tests(
                result.trace,
                payload.inputs,
                key,
                false);
        const auto commit_wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() -
                                         commit_start)
                                         .count();
        result.dynamic.wall_ms = std::max(result.dynamic.wall_ms, commit_wall_ms);

        expect(result.trace.spilled_polynomials.size() == payload.inputs.size(),
               "production dynamic seam did not spill every edge evaluation column");
        result.descriptors.reserve(payload.inputs.size());
        for (const auto &input : payload.inputs)
        {
            const auto it = result.trace.spilled_polynomials.find(input.name);
            expect(it != result.trace.spilled_polynomials.end(),
                   "production spill descriptor missing: " + input.name);
            result.descriptors.push_back(it->second);
        }

        std::unordered_set<std::string> spill_paths;
        for (const auto &descriptor : result.descriptors)
        {
            expect(descriptor.domain_name == domain->name &&
                       descriptor.size == domain->size &&
                       descriptor.physical_size == domain->size &&
                       descriptor.zero_tail_compacted &&
                       descriptor.lease != nullptr,
                   "production spill descriptor lost authenticated edge metadata");
            std::error_code error;
            const auto bytes = std::filesystem::file_size(descriptor.path, error);
            expect(!error, "unable to stat production spill file: " + descriptor.path);
            if (spill_paths.insert(descriptor.path).second)
            {
                result.spill_bytes += static_cast<std::size_t>(bytes);
            }
        }
        result.unique_spill_files = spill_paths.size();
        const auto expected_unique = reuse_enabled
                                         ? payload.inputs.size() / 2
                                         : payload.inputs.size();
        expect(result.unique_spill_files == expected_unique,
               "exact-view spill reuse toggle produced unexpected file count");
        for (std::size_t pair = 0; pair < payload.inputs.size() / 2; ++pair)
        {
            const auto &first = result.descriptors[pair * 2];
            const auto &second = result.descriptors[pair * 2 + 1];
            if (reuse_enabled)
            {
                expect(first.path == second.path &&
                           first.semantic_source_id == second.semantic_source_id,
                       "exact alias pair did not share spill identity");
            }
            else
            {
                expect(first.path != second.path &&
                           first.semantic_source_id != second.semantic_source_id,
                       "off lane unexpectedly reused spill identity");
            }
        }

        result.groups[0].reserve(payload.inputs.size() / 2);
        result.groups[1].reserve(payload.inputs.size() / 2);
        result.claims[0].reserve(payload.inputs.size() / 2);
        result.claims[1].reserve(payload.inputs.size() / 2);
        result.points = {
            non_domain_point(domain, 7919),
            non_domain_point(domain, 104729)};
        result.challenges = {FieldElement(37), FieldElement(53)};

        const auto claim_start = std::chrono::steady_clock::now();
        std::array<DomainEvaluationWeights, 2> weights{
            gatzk::algebra::build_domain_evaluation_weights(
                domain,
                result.points[0]),
            gatzk::algebra::build_domain_evaluation_weights(
                domain,
                result.points[1])};
        for (std::size_t index = 0; index < result.descriptors.size(); ++index)
        {
            result.groups[index % 2].push_back(result.dynamic.commitments[index]);
            const auto values = evaluate_descriptor(
                result.descriptors[index],
                domain,
                result.points[index % 2],
                weights[index % 2]);
            result.claims[index % 2].push_back(values.front());
        }
        result.claim_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - claim_start)
                              .count();

        std::unordered_map<
            std::string,
            const TraceArtifacts::SpilledEvaluationPolynomial *>
            descriptor_map;
        descriptor_map.reserve(result.descriptors.size());
        for (std::size_t index = 0; index < result.descriptors.size(); ++index)
        {
            descriptor_map.emplace(
                result.dynamic.commitments[index].name,
                &result.descriptors[index]);
        }
        const auto semantic_resolver = make_semantic_resolver(
            descriptor_map,
            domain);
        const gatzk::crypto::OpeningPolynomialResolver opening_resolver =
            [](const Commitment &)
            -> std::shared_ptr<const gatzk::algebra::Polynomial>
        {
            // Every selected edge commitment is file-backed.  Returning null
            // makes the scheduler use its authenticated semantic-source seam.
            return nullptr;
        };

        const auto opening_start = std::chrono::steady_clock::now();
        result.witnesses =
            KZG::open_point_group_pair_shared_stream(
                result.groups,
                result.points,
                result.claims,
                result.challenges,
                key,
                &result.opening_profiles,
                nullptr,
                &opening_resolver,
                &semantic_resolver);
        result.opening_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - opening_start)
                                .count();

        result.verification_passed =
            KZG::verify_point_group(
                result.groups[0],
                result.points[0],
                result.claims[0],
                result.challenges[0],
                result.witnesses[0],
                key.vk) &&
            KZG::verify_point_group(
                result.groups[1],
                result.points[1],
                result.claims[1],
                result.challenges[1],
                result.witnesses[1],
                key.vk);
        expect(result.verification_passed,
               "paired streamed opening failed KZG verification");

        // Tamper the first group's physical source with another valid spill
        // path.  Claims and commitments stay unchanged; the verifier must not
        // accept the witness produced from the wrong authenticated payload.
        if (result.descriptors.size() >= 4)
        {
            auto tampered_descriptors = result.descriptors;
            tampered_descriptors[0].path = result.descriptors[2].path;
            std::unordered_map<
                std::string,
                const TraceArtifacts::SpilledEvaluationPolynomial *>
                tampered_map;
            tampered_map.reserve(tampered_descriptors.size());
            for (std::size_t index = 0; index < tampered_descriptors.size(); ++index)
            {
                tampered_map.emplace(
                    result.dynamic.commitments[index].name,
                    &tampered_descriptors[index]);
            }
            const auto tampered_resolver = make_semantic_resolver(
                tampered_map,
                domain);
            std::array<BatchOpeningProfile, 2> ignored_profiles{};
            const auto tampered_witnesses =
                KZG::open_point_group_pair_shared_stream(
                    result.groups,
                    result.points,
                    result.claims,
                    result.challenges,
                    key,
                    &ignored_profiles,
                    nullptr,
                    &opening_resolver,
                    &tampered_resolver);
            result.tamper_detected =
                !KZG::verify_point_group(
                    result.groups[0],
                    result.points[0],
                    result.claims[0],
                    result.challenges[0],
                    tampered_witnesses[0],
                    key.vk);
            expect(result.tamper_detected,
                   "tampered spill payload unexpectedly verified");
        }
        else
        {
            // A single alias pair has no second physical source to swap in;
            // still exercise the same verifier binding with a tampered claim.
            auto tampered_claims = result.claims[0];
            tampered_claims.front() += FieldElement::one();
            result.tamper_detected =
                !KZG::verify_point_group(
                    result.groups[0],
                    result.points[0],
                    tampered_claims,
                    result.challenges[0],
                    result.witnesses[0],
                    key.vk);
            expect(result.tamper_detected,
                   "tampered single-pair claim unexpectedly verified");
        }

        result.rss_kb = std::max(rss_before, current_rss_kb());
        return result;
    }

    void write_lane(
        std::ostream &output,
        std::size_t pairs,
        std::string_view phase,
        std::size_t iteration,
        const LaneResult &lane,
        std::uint64_t payload_digest)
    {
        std::size_t scan_count = 0;
        std::size_t scan_bytes = 0;
        double witness_msm_ms = 0.0;
        for (const auto &profile : lane.opening_profiles)
        {
            scan_count += profile.streamed_semantic_source_scan_count;
            scan_bytes += profile.streamed_semantic_source_scan_bytes;
            witness_msm_ms += profile.witness_msm_ms;
        }
        const auto pair_hits = lane.opening_profiles[1]
                                   .streamed_semantic_source_scan_count == 0
                               ? pairs
                               : 0;
        output << "benchmark=ppi_spill_targeted"
               << " phase=" << phase
               << " iteration=" << iteration
               << " reuse=" << (lane.reuse_enabled ? "on" : "off")
               << " pairs=" << pairs
               << " payload_digest=" << payload_digest
               << " commitment_digest="
               << commitment_digest(lane.dynamic.commitments)
               << " opening_digest=" << point_digest(lane.witnesses)
               << " verification=" << (lane.verification_passed ? "PASS" : "FAIL")
               << " tamper=" << (lane.tamper_detected ? "PASS" : "FAIL")
               << " commit_wall_ms=" << std::fixed << std::setprecision(3)
               << lane.dynamic.wall_ms
               << " dynamic_commit_ms=" << lane.dynamic.dynamic_commit_ms
               << " dynamic_msm_ms=" << lane.dynamic.dynamic_commit_msm_ms
               << " claim_ms=" << lane.claim_ms
               << " opening_wall_ms=" << lane.opening_ms
               << " opening_witness_msm_ms=" << witness_msm_ms
               << " source_scan_count=" << scan_count
               << " source_scan_bytes=" << scan_bytes
               << " pair_hits=" << pair_hits
               << " spill_files=" << lane.unique_spill_files
               << " spill_bytes=" << lane.spill_bytes
               << " rss_kb=" << lane.rss_kb << '\n';
        output.flush();
    }

    void run_phase(
        std::ostream &output,
        const Options &options,
        const Payload &payload,
        const KZGKeyPair &key,
        const std::shared_ptr<const ProtocolCompilerArtifact> &artifact,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        bool reuse_enabled,
        std::string_view phase)
    {
        for (std::size_t iteration = 0;
             iteration < (phase == "warmup" ? options.warmups : options.repeats);
             ++iteration)
        {
            const auto lane = run_lane(
                reuse_enabled,
                payload,
                key,
                artifact,
                domain);
            write_lane(
                output,
                options.pairs,
                phase,
                iteration,
                lane,
                payload.digest);
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
        const auto verifier =
            gatzk::protocol::load_verifier_context_from_registry(
                options.registry_path.string(),
                options.registry_digest);
        expect(verifier.compiler_artifact != nullptr,
               "authenticated PPI registry omitted compiler artifact");

        if (options.inspect_registry)
        {
            std::cout << "registry_inspect=ppi\n";
            for (const auto &entry : verifier.compiler_artifact->proof_schema.entries)
            {
                if (entry.tag.find("Div_") == std::string::npos &&
                    entry.tag.find("Limb_") == std::string::npos &&
                    entry.tag.find("Range_") == std::string::npos)
                {
                    continue;
                }
                std::cout << "schema tag=" << entry.tag
                          << " domain=" << entry.domain
                          << " private="
                          << (entry.visibility ==
                                      gatzk::protocol::PolynomialVisibility::PrivateWitness
                                  ? "true"
                                  : "false")
                          << " commit=" << (entry.enters_commitment ? "true" : "false")
                          << " opening=" << (entry.enters_opening ? "true" : "false")
                          << " quotient=" << (entry.enters_quotient ? "true" : "false")
                          << '\n';
            }
            for (const auto &entry : verifier.compiler_artifact->observation_ledger)
            {
                if (entry.object_or_challenge.find("Div_") == std::string::npos &&
                    entry.object_or_challenge.find("Limb_") == std::string::npos &&
                    entry.object_or_challenge.find("Range_") == std::string::npos)
                {
                    continue;
                }
                std::cout << "ledger object=" << entry.object_or_challenge
                          << " domain=" << entry.domain
                          << " query_bound=" << entry.query_bound
                          << " degree_bound=" << entry.degree_bound << '\n';
            }
            return 0;
        }
        const auto domain = verifier.domains.edge;
        expect(domain != nullptr && domain->name == "edge" &&
                   domain->size >= kMinimumDomainSize,
               "authenticated PPI registry has no edge domain >= 2^18");

        const auto entries = select_private_entries(
            *verifier.compiler_artifact,
            "edge",
            options.pairs);
        std::size_t max_query_bound = 0;
        for (const auto &entry : entries)
        {
            max_query_bound = std::max(max_query_bound, entry.query_bound);
        }
        expect(max_query_bound != 0 &&
                   domain->size <=
                       std::numeric_limits<std::size_t>::max() -
                           max_query_bound,
               "authenticated edge query bound overflows coefficient prefix");

        const auto key = KZG::load_keys_for_domains(
            options.srs_path.string(),
            "trusted_srs",
            "ppi_formal_srs",
            {{"edge", domain->size}},
            domain->size + max_query_bound);
        const auto manifest_domain = key.manifest.domain_sizes.find("edge");
        expect(manifest_domain != key.manifest.domain_sizes.end() &&
                   manifest_domain->second == domain->size &&
                   key.pk.lagrange_srs_by_domain.contains("edge") &&
                   key.pk.lagrange_srs_by_domain.at("edge") != nullptr &&
                   key.pk.lagrange_srs_by_domain.at("edge")->size() == domain->size,
               "authenticated PPI SRS does not exactly cover edge");

        const auto payload = build_payload(options.pairs, domain, entries);
        const auto provenance = load_run_provenance(options.provenance_path);
        std::ofstream output_file;
        std::ostream *output = &std::cout;
        if (!options.output_path.empty())
        {
            if (!options.output_path.parent_path().empty())
            {
                std::filesystem::create_directories(options.output_path.parent_path());
            }
            output_file.open(
                options.output_path,
                std::ios::out | std::ios::trunc);
            if (!output_file)
            {
                throw std::runtime_error(
                    "unable to open PPI spill benchmark output: " +
                    options.output_path.string());
            }
            output = &output_file;
        }

        *output << "benchmark=ppi_spill_targeted_header"
                << " run_id=" << provenance.run_id
                << " git_head=" << provenance.git_head
                << " non_comparison_diff_sha256="
                << provenance.non_comparison_diff_sha256
                << " benchmark_source_sha256="
                << provenance.benchmark_source_sha256
                << " binary_sha256=" << provenance.binary_sha256
                << " route=" << options.route
                << " pairs=" << options.pairs
                << " domain=" << domain->name << ':' << domain->size
                << " srs_id=" << key.pk.srs_id
                << " srs_type=" << key.pk.srs_type
                << " manifest_hash=" << key.manifest.manifest_hash
                << " registry_digest=" << options.registry_digest
                << " warmups=" << options.warmups
                << " repeats=" << options.repeats
                << " max_query_bound=" << max_query_bound << '\n';
        output->flush();

        run_phase(
            *output,
            options,
            payload,
            key,
            verifier.compiler_artifact,
            domain,
            true,
            "warmup");
        run_phase(
            *output,
            options,
            payload,
            key,
            verifier.compiler_artifact,
            domain,
            false,
            "warmup");
        run_phase(
            *output,
            options,
            payload,
            key,
            verifier.compiler_artifact,
            domain,
            true,
            "measured");
        run_phase(
            *output,
            options,
            payload,
            key,
            verifier.compiler_artifact,
            domain,
            false,
            "measured");
        *output << "status=PASS max_rss_kb=" << max_rss_kb() << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "status=FAIL error=" << error.what() << '\n';
        return 1;
    }
}
