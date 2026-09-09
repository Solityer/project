#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::Polynomial;
    using gatzk::algebra::RootOfUnityDomain;
    using gatzk::protocol::ProtocolContext;
    using gatzk::protocol::RunMetrics;
    using gatzk::protocol::TraceArtifacts;

    using Clock = std::chrono::steady_clock;

    struct Options
    {
        std::string mode = "materializer";
        std::string route = "cpu";
        std::size_t domain_log2 = 12;
        std::size_t target_log2 = 0;
        std::size_t repeats = 1;
        std::size_t dependency_count = 15;
        bool generic_staged = true;
    };

    struct StreamCapture
    {
        explicit StreamCapture(std::ostream &stream)
            : stream_(stream), previous_(stream.rdbuf(buffer_.rdbuf()))
        {
        }

        ~StreamCapture()
        {
            stream_.rdbuf(previous_);
        }

        std::string str() const
        {
            return buffer_.str();
        }

    private:
        std::ostream &stream_;
        std::ostringstream buffer_;
        std::streambuf *previous_ = nullptr;
    };

    [[noreturn]] void fail(const std::string &message)
    {
        throw std::runtime_error(message);
    }

    std::size_t parse_size(const std::string &text, const char *name)
    {
        if (text.empty() || text.front() == '-')
        {
            fail(std::string("invalid ") + name + ": " + text);
        }
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed, 10);
        if (consumed != text.size() || value == 0)
        {
            fail(std::string("invalid ") + name + ": " + text);
        }
        return static_cast<std::size_t>(value);
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
                    fail(std::string("missing value for ") + name);
                }
                return std::string(argv[++index]);
            };

            if (argument == "--mode")
            {
                options.mode = require_value("--mode");
            }
            else if (argument == "--route")
            {
                options.route = require_value("--route");
            }
            else if (argument == "--domain-log2")
            {
                options.domain_log2 = parse_size(
                    require_value("--domain-log2"),
                    "--domain-log2");
            }
            else if (argument == "--target-log2")
            {
                options.target_log2 = parse_size(
                    require_value("--target-log2"),
                    "--target-log2");
            }
            else if (argument == "--repeats")
            {
                options.repeats = parse_size(
                    require_value("--repeats"),
                    "--repeats");
            }
            else if (argument == "--dependency-count")
            {
                options.dependency_count = parse_size(
                    require_value("--dependency-count"),
                    "--dependency-count");
            }
            else if (argument == "--generic-staged")
            {
                const auto value = require_value("--generic-staged");
                if (value == "on")
                {
                    options.generic_staged = true;
                }
                else if (value == "off")
                {
                    options.generic_staged = false;
                }
                else
                {
                    fail("--generic-staged must be on or off");
                }
            }
            else if (argument == "--help")
            {
                std::cout
                    << "--mode materializer|coset "
                    << "--route cpu|cuda "
                    << "--domain-log2 N --target-log2 N --repeats N "
                    << "--dependency-count N "
                    << "--generic-staged on|off\n";
                std::exit(0);
            }
            else
            {
                fail("unknown argument: " + argument);
            }
        }

        if (options.route != "cpu" && options.route != "cuda")
        {
            fail("--route must be cpu or cuda");
        }
        if (options.mode != "materializer" && options.mode != "coset")
        {
            fail("--mode must be materializer or coset");
        }
        if (options.mode == "coset" && options.target_log2 == 0)
        {
            fail("--target-log2 is required for --mode coset");
        }
        return options;
    }

    std::size_t current_rss_bytes()
    {
        std::ifstream input("/proc/self/statm");
        std::size_t pages = 0;
        std::size_t resident_pages = 0;
        if (!(input >> pages >> resident_pages))
        {
            return 0;
        }
        (void)pages;
        const auto page_size = ::sysconf(_SC_PAGESIZE);
        return page_size > 0
                   ? resident_pages * static_cast<std::size_t>(page_size)
                   : 0;
    }

    std::size_t peak_rss_bytes()
    {
        std::ifstream input("/proc/self/status");
        std::string key;
        while (input >> key)
        {
            if (key != "VmHWM:")
            {
                std::string ignored;
                std::getline(input, ignored);
                continue;
            }
            std::size_t kib = 0;
            input >> kib;
            return kib * 1024;
        }
        return 0;
    }

    std::string sha256_hex(
        const std::vector<std::pair<std::string, Polynomial>> &polynomials)
    {
        SHA256_CTX context;
        SHA256_Init(&context);
        for (const auto &[name, polynomial] : polynomials)
        {
            SHA256_Update(
                &context,
                name.data(),
                name.size());
            const auto basis = static_cast<std::uint8_t>(polynomial.basis);
            SHA256_Update(&context, &basis, sizeof(basis));
            for (const auto &value : polynomial.values())
            {
                const auto &native = value.native();
                SHA256_Update(
                    &context,
                    &native,
                    sizeof(native));
            }
        }
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256_Final(digest.data(), &context);
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const auto byte : digest)
        {
            output << std::setw(2) << static_cast<unsigned>(byte);
        }
        return output.str();
    }

    std::string sha256_hex_values(const std::vector<mcl::Fr> &values)
    {
        SHA256_CTX context;
        SHA256_Init(&context);
        if (!values.empty())
        {
            SHA256_Update(
                &context,
                values.data(),
                values.size() * sizeof(mcl::Fr));
        }
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256_Final(digest.data(), &context);
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const auto byte : digest)
        {
            output << std::setw(2) << static_cast<unsigned>(byte);
        }
        return output.str();
    }

    std::size_t power_of_two(std::size_t log2)
    {
        if (log2 >= sizeof(std::size_t) * 8)
        {
            fail("power-of-two exponent overflows size_t");
        }
        return std::size_t{1} << log2;
    }

    std::size_t checked_workspace_bytes(std::size_t target_size)
    {
        constexpr auto multiplier = 2 * sizeof(mcl::Fr);
        if (target_size >
            std::numeric_limits<std::size_t>::max() / multiplier)
        {
            fail("workspace byte count overflows size_t");
        }
        return target_size * multiplier;
    }

    ProtocolContext make_context(
        std::size_t domain_size,
        TraceArtifacts *trace,
        std::size_t dependency_count)
    {
        if (trace == nullptr || domain_size == 0)
        {
            fail("invalid materializer fixture");
        }

        ProtocolContext context;
        // Deliberately use a shape with different feature counts.  This keeps
        // the FH relation on its full production dependency registry, so the
        // benchmark exercises prewarm/evaluate/coefficients rather than the
        // full-graph identity shortcut.
        context.dataset.num_nodes = 0;
        context.dataset.num_features = 1;
        context.local.num_nodes = 0;
        context.local.num_features = 0;
        context.domains.fh = RootOfUnityDomain::create("FH", domain_size);

        const auto target_size = domain_size * 4;
        context.kzg.pk.degree_bound = target_size - 1;

        static const std::vector<std::string> dependencies = {
            "P_H",
            "P_R_feat",
            "P_Table_feat",
            "P_Query_feat",
            "P_T_H",
            "P_Row_feat_tbl",
            "P_Col_feat_tbl",
            "P_Row_feat_qry",
            "P_Col_feat_qry",
            "P_I_feat_qry",
            "P_Q_tbl_feat",
            "P_m_feat",
            "P_Q_qry_feat",
            "P_J_tbl_feat",
            "P_J_qry_feat",
        };
        if (dependency_count == 0 ||
            dependency_count > dependencies.size())
        {
            fail("dependency count must be in [1,15]");
        }
        std::vector<std::shared_ptr<const std::vector<FieldElement>>>
            independent_sources;
        independent_sources.reserve(dependency_count);
        for (std::size_t source_index = 0;
             source_index < dependency_count;
             ++source_index)
        {
            auto values = std::make_shared<std::vector<FieldElement>>();
            values->resize(domain_size);
            for (std::size_t index = 0; index < domain_size; ++index)
            {
                // Each selected source is an independent formal dependency;
                // labels beyond the requested fanout reuse the deterministic
                // source pool so the production registry remains complete.
                (*values)[index] = FieldElement(
                    static_cast<std::uint64_t>(
                        17 + source_index * 1000003ULL +
                        ((index * 1315423911ULL) % 1000003ULL)));
            }
            independent_sources.push_back(std::move(values));
        }
        for (std::size_t index = 0;
             index < dependency_count;
             ++index)
        {
            const auto &label = dependencies[index];
            trace->polynomials.emplace(
                label,
                Polynomial::from_shared_evaluations(
                    label,
                    independent_sources[index],
                    context.domains.fh));
        }
        for (std::size_t index = dependency_count;
             index < dependencies.size();
             ++index)
        {
            const auto &label = dependencies[index];
            trace->polynomials.emplace(
                label,
                Polynomial::from_shared_evaluations(
                    label,
                    independent_sources[index % dependency_count],
                    context.domains.fh));
        }
        return context;
    }

    std::string progress_value(
        const std::string &progress,
        std::string_view marker)
    {
        const auto position = progress.find(marker);
        if (position == std::string::npos)
        {
            return "NA";
        }
        const auto begin = position + marker.size();
        const auto end = progress.find('\n', begin);
        return progress.substr(begin, end == std::string::npos ? end : end - begin);
    }

    void configure_route(
        const std::string &route,
        bool generic_staged)
    {
        auto options = gatzk::util::route2_options();
        options.cuda_field_hotspots = route == "cuda";
        gatzk::util::set_route2_options(options);
#if defined(GATZK_BUILD_TESTS) && GATZK_BUILD_TESTS
        gatzk::algebra::set_cuda_generic_staged_route_for_tests(
            generic_staged);
#else
        (void)generic_staged;
#endif
    }

    int run_materializer(const Options &options)
    {
        if (options.domain_log2 >= sizeof(std::size_t) * 8 - 2)
        {
            fail("domain exponent is too large");
        }
        const auto domain_size = power_of_two(options.domain_log2);
        const auto target_size = domain_size * 4;
        if (options.route == "cuda" &&
            !gatzk::algebra::cuda_backend_available())
        {
            std::cout << "status=unavailable route=cuda target_size="
                      << target_size << '\n';
            return 77;
        }

        configure_route(options.route, options.generic_staged);
        std::map<std::string, FieldElement> challenges = {
            {"alpha_quot", FieldElement(3)},
            {"beta_feat", FieldElement(5)},
            {"eta_feat", FieldElement(7)},
        };

        for (std::size_t repeat = 0; repeat < options.repeats; ++repeat)
        {
            TraceArtifacts trace;
            auto context = make_context(
                domain_size,
                &trace,
                options.dependency_count);
            RunMetrics metrics;
            const auto stats_before =
                gatzk::algebra::cuda_execution_stats_snapshot();
            const auto kernels_before =
                gatzk::algebra::cuda_kernel_invocation_count();
            const auto rss_before = current_rss_bytes();
            const auto wall_start = Clock::now();
            setenv("GATZK_RUN_PROGRESS", "1", 1);
            std::string progress;
            std::vector<std::pair<std::string, Polynomial>> output;
            {
                StreamCapture capture(std::cerr);
                output =
                    gatzk::protocol::
                        materialize_quotient_polynomials_for_targeted_benchmark(
                            context,
                            trace,
                            challenges,
                            {"t_FH"},
                            &metrics);
                progress = capture.str();
            }
            const auto wall_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - wall_start)
                    .count();
            const auto stats_after =
                gatzk::algebra::cuda_execution_stats_snapshot();
            const auto kernels_after =
                gatzk::algebra::cuda_kernel_invocation_count();

            const auto delta = [](std::uint64_t after, std::uint64_t before)
            {
                return after >= before ? after - before : 0;
            };
            const auto leaf_size = target_size / 2;
            const auto radix = leaf_size >= (std::size_t{1} << 24)
                                   ? leaf_size / (std::size_t{1} << 24)
                                   : 1;
            const auto quotient_calls =
                delta(stats_after.quotient_calls, stats_before.quotient_calls);
            const auto ntt_calls =
                delta(stats_after.ntt_calls, stats_before.ntt_calls);
            const auto intt_calls =
                delta(stats_after.intt_calls, stats_before.intt_calls);
            std::cout << "mode=materializer"
                      << " repeat=" << repeat
                      << " requested_route=" << options.route
                      << " dependency_count=" << options.dependency_count
                      << " scaled_fanout=" << options.dependency_count
                      << "/15"
                      << " generic_staged="
                      << (options.generic_staged ? "on" : "off")
                      << " route_label="
                      << (options.generic_staged && options.route == "cuda"
                              ? "candidate_generic_cuda"
                              : "legacy_dependency_cpu")
                      << " actual_route="
                      << (quotient_calls != 0
                              ? (options.generic_staged
                                     ? "candidate_generic_cuda"
                                     : "legacy_dependency_cpu")
                              : (options.route == "cuda"
                                     ? "legacy_dependency_cpu"
                                     : "legacy_dependency_cpu"))
                      << " route_diagnostic=generic_staged_test_hook"
                      << " domain_size=" << domain_size
                      << " target_size=" << target_size
                      << " leaf_size=" << leaf_size
                      << " radix=" << radix
                      << " batch=2"
                      << " workspace_bytes="
                      << checked_workspace_bytes(target_size)
                      << " production_parent_wall_ms=" << wall_ms
                      << " quotient_t_fh_ms=" << metrics.quotient_t_fh_ms
                      << " stage_timing=diagnostic_progress_only"
                      << " prewarm="
                      << progress_value(
                             progress,
                             "prove:quotient:t_FH:coset_prewarm:done:ms=")
                      << " evaluate="
                      << progress_value(
                             progress,
                             "prove:quotient:t_FH:evaluate:done:ms=")
                      << " coefficients="
                      << progress_value(
                             progress,
                             "prove:quotient:t_FH:coefficients:done:ms=")
                      << " kernel_invocations="
                      << (kernels_after - kernels_before)
                      << " ntt="
                      << ntt_calls
                      << " intt="
                      << intt_calls
                      << " quotient_calls=" << quotient_calls
                      << " h2d_bytes="
                      << delta(stats_after.h2d_bytes, stats_before.h2d_bytes)
                      << " d2h_bytes="
                      << delta(stats_after.d2h_bytes, stats_before.d2h_bytes)
                      << " sync="
                      << delta(
                             stats_after.stream_synchronizations,
                             stats_before.stream_synchronizations)
                      << " rss_before=" << rss_before
                      << " rss_after=" << current_rss_bytes()
                      << " rss_hwm=" << peak_rss_bytes()
                      << " output_size="
                      << (output.empty() ? 0 : output.front().second.size())
                      << " digest=" << sha256_hex(output)
                      << '\n';
        }
        return 0;
    }

    int run_coset(const Options &options)
    {
        if (options.target_log2 < 2 ||
            options.target_log2 >= sizeof(std::size_t) * 8)
        {
            fail("invalid target exponent");
        }
        const auto target_size = power_of_two(options.target_log2);
        const auto required_workspace =
            checked_workspace_bytes(target_size);
        if (!gatzk::algebra::cuda_backend_available())
        {
            std::cout << "status=unavailable target_size=" << target_size
                      << " required_workspace_bytes=" << required_workspace
                      << '\n';
            return 77;
        }

        const FieldElement shift(7);
        const std::vector<FieldElement> expected_coefficients = {
            FieldElement(1),
            FieldElement(3),
            FieldElement(5),
            FieldElement(7),
            FieldElement(11),
        };
        std::vector<mcl::Fr> coefficients;
        coefficients.reserve(expected_coefficients.size());
        for (const auto &coefficient : expected_coefficients)
        {
            coefficients.push_back(coefficient.native());
        }
        std::vector<mcl::Fr> values(target_size);
        const auto stats_before =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto kernels_before =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto rss_before = current_rss_bytes();
        const auto start = Clock::now();
        bool materialized = false;
        try
        {
            materialized =
                gatzk::algebra::cuda_coset_coefficients_to_host_exact(
                    coefficients.data(),
                    coefficients.size(),
                    shift,
                    values.data(),
                    target_size,
                    false);
        }
        catch (const std::exception &error)
        {
            std::cout << "status=error target_size=" << target_size
                      << " required_workspace_bytes=" << required_workspace
                      << " error=" << error.what() << '\n';
            return 1;
        }
        if (!materialized)
        {
            std::cout << "status=declined target_size=" << target_size
                      << " required_workspace_bytes=" << required_workspace
                      << " rss_before=" << rss_before << '\n';
            return 77;
        }

        bool exact = true;
        const auto root = FieldElement::root_of_unity(target_size);
        auto point = shift;
        for (std::size_t logical = 0; logical < target_size; ++logical)
        {
            FieldElement expected = FieldElement::zero();
            for (auto coefficient = expected_coefficients.rbegin();
                 coefficient != expected_coefficients.rend();
                 ++coefficient)
            {
                expected *= point;
                expected += *coefficient;
            }
            if (!(values[logical] == expected.native()))
            {
                exact = false;
                break;
            }
            point *= root;
        }
        const auto forward_digest = sha256_hex_values(values);
        bool roundtrip = false;
        try
        {
            roundtrip =
                gatzk::algebra::cuda_coset_host_to_coefficients_exact(
                    values.data(),
                    target_size,
                    shift,
                    values.data());
        }
        catch (const std::exception &error)
        {
            std::cout << "status=roundtrip_error target_size=" << target_size
                      << " required_workspace_bytes=" << required_workspace
                      << " error=" << error.what() << '\n';
            return 1;
        }
        bool roundtrip_exact = roundtrip;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const auto expected = index < coefficients.size()
                                      ? coefficients[index]
                                      : mcl::Fr(0);
            if (!(values[index] == expected))
            {
                roundtrip_exact = false;
                break;
            }
        }
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        const auto stats_after =
            gatzk::algebra::cuda_execution_stats_snapshot();
        const auto kernels_after =
            gatzk::algebra::cuda_kernel_invocation_count();
        const auto delta = [](std::uint64_t after, std::uint64_t before)
        {
            return after >= before ? after - before : 0;
        };
        const auto leaf_size = target_size / 2;
        const auto radix = leaf_size >= (std::size_t{1} << 24)
                               ? leaf_size / (std::size_t{1} << 24)
                               : 1;
        std::cout << "mode=coset"
                  << " target_size=" << target_size
                  << " leaf_size=" << leaf_size
                  << " radix=" << radix
                  << " layout=interleaved"
                  << " batch=2"
                  << " workspace_bytes=" << required_workspace
                  << " exact=" << (exact ? 1 : 0)
                  << " roundtrip_exact=" << (roundtrip_exact ? 1 : 0)
                  << " wall_ms=" << elapsed_ms
                  << " kernel_invocations=" << (kernels_after - kernels_before)
                  << " ntt=" << delta(stats_after.ntt_calls, stats_before.ntt_calls)
                  << " intt=" << delta(stats_after.intt_calls, stats_before.intt_calls)
                  << " h2d_bytes=" << delta(stats_after.h2d_bytes, stats_before.h2d_bytes)
                  << " d2h_bytes=" << delta(stats_after.d2h_bytes, stats_before.d2h_bytes)
                  << " sync=" << delta(stats_after.stream_synchronizations, stats_before.stream_synchronizations)
                  << " rss_before=" << rss_before
                  << " rss_after=" << current_rss_bytes()
                  << " rss_hwm=" << peak_rss_bytes()
                  << " forward_digest=" << forward_digest
                  << " roundtrip_digest=" << sha256_hex_values(values)
                  << '\n';
        return exact && roundtrip_exact ? 0 : 1;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        if (options.mode == "materializer")
        {
            return run_materializer(options);
        }
        return run_coset(options);
    }
    catch (const std::exception &error)
    {
        std::cerr << "quotient_materializer_targeted_bench: "
                  << error.what() << '\n';
        return 1;
    }
}
