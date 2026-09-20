#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/util/route2.hpp"
#include "gatzk/util/trace_spill.hpp"

namespace
{
    namespace algebra = gatzk::algebra;
    namespace protocol = gatzk::protocol;
    namespace util = gatzk::util;
    using F = algebra::FieldElement;
    using P = algebra::Polynomial;

    void require(bool condition, const std::string &label)
    {
        if (!condition)
            throw std::runtime_error(label);
    }

    template <typename Fn>
    void reject(Fn &&fn, const std::string &label)
    {
        bool rejected = false;
        try
        {
            fn();
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        require(rejected, "expected rejection: " + label);
    }

    std::vector<F> sample(std::size_t n)
    {
        std::vector<F> out(n, F::zero());
        for (std::size_t i = 0; i < n; ++i)
            out[i] = F::from_signed(static_cast<std::int64_t>((i * 137 + 19) % 1021) - 510);
        return out;
    }

    F reference_horner(const std::vector<F> &values, const F &x)
    {
        auto result = F::zero();
        for (std::size_t i = values.size(); i-- > 0;)
            result = result * x + values[i];
        return result;
    }

    std::vector<F> reference_shift(const std::vector<F> &a, const F &delta)
    {
        std::vector<F> result(a.size(), F::zero());
        for (std::size_t j = 0; j < a.size(); ++j)
        {
            auto binomial = F::one();
            for (std::size_t k = 0; k <= j; ++k)
            {
                result[k] += a[j] * binomial * delta.pow(j - k);
                if (k < j)
                    binomial = binomial * F(j - k) / F(k + 1);
            }
        }
        return result;
    }

    void test_horner_and_sharing()
    {
        for (const auto n : std::array<std::size_t, 7>{0, 1, 31, 65536, 65537, 524288, 524305})
        {
            const auto values = sample(n);
            auto polynomial = P::from_coefficients("test_coefficients", {});
            polynomial.shared_data = std::make_shared<const std::vector<F>>(values);
            for (const auto &x : {F::zero(), F::one(), F(7)})
            {
                const auto expected = reference_horner(values, x);
                require(algebra::horner(values, x) == expected, "block Horner");
                require(polynomial.semantic_evaluate(x) == expected, "shared coefficient evaluation");
            }
            const auto material = algebra::materialize_blinded_coefficients(polynomial, n + 2);
            require(material.size() == n + 3, "configured coefficient length");
            require(std::equal(values.begin(), values.end(), material.begin()), "shared coefficient material");
            require(std::all_of(material.begin() + static_cast<std::ptrdiff_t>(n), material.end(),
                                [](const auto &v)
                                { return v.is_zero(); }),
                    "coefficient zero padding");
        }
    }

    void test_evaluation_and_blinder()
    {
        for (const auto n : std::array<std::size_t, 5>{2, 32, 16384, 32768, 65536})
        {
            const auto domain = algebra::RootOfUnityDomain::create("execution_test", n);
            auto coefficients = sample(n);
            auto evaluations = coefficients;
            algebra::ntt_in_place(F::mutable_native_array_data(evaluations.data()), n);
            auto p = P::from_evaluations("test_evaluation", evaluations, domain);
            for (const auto &x : {F::zero(), F::one(), F(7)})
                require(p.semantic_evaluate(x) == reference_horner(coefficients, x), "batched barycentric evaluation");
            p.set_zero_domain_blinder({F(3), F(5), F(11)});
            const auto material = algebra::materialize_blinded_coefficients(p, n + 2);
            for (const auto &x : {F::zero(), F(7), domain->point_at(n - 1)})
            {
                const auto expected = reference_horner(coefficients, x) +
                                      domain->zero_polynomial_eval(x) * reference_horner(p.zero_domain_blinder, x);
                require(reference_horner(material, x) == expected, "blinded coefficient material");
            }
            auto compact = P::from_evaluations(
                "test_compact", std::vector<F>(evaluations.begin(), evaluations.begin() + n / 2), domain);
            require(compact.semantic_evaluate(domain->point_at(n - 1)).is_zero(), "root in omitted zero tail");
            auto empty = P::from_evaluations("test_empty", {}, domain);
            require(empty.semantic_evaluate(F(7)).is_zero(), "empty evaluation polynomial");
            reject([&]
                   { (void)algebra::materialize_blinded_coefficients(p, n); }, "blinder extent");
        }
    }

    void test_taylor()
    {
        {
            std::vector<F> singleton{F(37)};
            const auto original = singleton;
            algebra::ntt_in_place(
                F::mutable_native_array_data(singleton.data()),
                singleton.size());
            require(singleton == original, "size-one NTT identity");
            algebra::intt_in_place(
                F::mutable_native_array_data(singleton.data()),
                singleton.size());
            require(singleton == original, "size-one INTT identity");
        }

        for (const auto degree : std::array<std::size_t, 10>{0, 1, 2, 8, 17, 33, 65, 129, 257, 300})
        {
            const auto a = sample(degree + 1);
            for (const auto &delta : {F::zero(), F::one(), F(19)})
            {
                const auto shifted = algebra::coefficient_taylor_shift(a, degree, delta);
                require(shifted == reference_shift(a, delta), "Taylor full coefficients");
            }
        }
        const auto a = sample(65539);
        const auto shifted = algebra::coefficient_taylor_shift(a, a.size() - 1, F(19));
        for (const auto &x : {F::zero(), F(7), F(113)})
            require(reference_horner(shifted, x) == reference_horner(a, x + F(19)), "large Taylor identity");
        std::vector<std::future<std::vector<F>>> futures;
        for (int i = 0; i < 4; ++i)
            futures.push_back(std::async(std::launch::async, [&]
                                         { return algebra::coefficient_taylor_shift(a, a.size() - 1, F(19)); }));
        for (auto &future : futures)
            require(future.get() == shifted, "concurrent Taylor cache reuse");
        algebra::release_taylor_shift_cache();
        require(algebra::coefficient_taylor_shift(a, a.size() - 1, F(19)) == shifted, "Taylor cache rebuild");
        reject([&]
               { (void)algebra::coefficient_taylor_shift(a, a.size(), F(19)); }, "configured Taylor length");
        algebra::release_taylor_shift_cache();
    }

    void test_coefficient_reader()
    {
        const std::filesystem::path root(GATZK_EXECUTION_TEST_ROOT);
        auto lease = util::TraceSpillLease::create(root);
        const auto values = sample(1024);
        {
            auto output = lease->create_payload_stream("coefficients.bin");
            algebra::write_field_elements_binary(*output, std::span<const F>(values));
            output->flush();
            require(static_cast<bool>(*output), "test spill flush");
        }
        protocol::TraceArtifacts trace;
        trace.commitments.emplace("test_spill", gatzk::crypto::Commitment{
                                                    .name = "test_spill", .point = gatzk::crypto::g1_zero()});
        trace.polynomial_domains["test_spill"] = "coef";
        trace.spilled_coefficient_polynomials.emplace("test_spill",
                                                      protocol::TraceArtifacts::SpilledCoefficientPolynomial{
                                                          .path = (lease->path() / "coefficients.bin").string(),
                                                          .size = values.size(),
                                                          .lease = lease});
        const auto decoded = protocol::resolve_trace_coefficient_polynomial(trace, "test_spill");
        require(decoded->values() == values, "coefficient spill round trip");
        auto &descriptor = trace.spilled_coefficient_polynomials.at("test_spill");
        ++descriptor.size;
        reject([&]
               { (void)protocol::resolve_trace_coefficient_polynomial(trace, "test_spill"); }, "spill length");
        --descriptor.size;
        const auto real_path = descriptor.path;
        descriptor.path = (root / "outside.bin").string();
        reject([&]
               { (void)protocol::resolve_trace_coefficient_polynomial(trace, "test_spill"); }, "spill path escape");
        descriptor.path = real_path;
        descriptor.lease.reset();
        reject([&]
               { (void)protocol::resolve_trace_coefficient_polynomial(trace, "test_spill"); }, "missing lease");
        descriptor.lease = lease;
        trace.polynomial_domains["test_spill"] = "N";
        reject([&]
               { (void)protocol::resolve_trace_coefficient_polynomial(trace, "test_spill"); }, "spill basis registration");
        trace.polynomial_domains["test_spill"] = "coef";
        {
            auto bad = lease->create_payload_stream("noncanonical.bin");
            const std::array<unsigned char, 32> bytes = []
            { std::array<unsigned char, 32> out{}; out.fill(255); return out; }();
            bad->write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            bad->flush();
            require(static_cast<bool>(*bad), "negative spill flush");
        }
        descriptor.path = (lease->path() / "noncanonical.bin").string();
        descriptor.size = 1;
        reject([&]
               { (void)protocol::resolve_trace_coefficient_polynomial(trace, "test_spill"); }, "noncanonical coefficient");
        // This test checks the source resolver, not a KZG verification claim.
    }

    void test_evaluation_spill_registry()
    {
        const std::filesystem::path root(
            GATZK_EXECUTION_TEST_ROOT);
        auto lease =
            util::TraceSpillLease::create(root);
        const std::string payload_name =
            "evaluation.bin";
        {
            auto output =
                lease->create_payload_stream(
                    payload_name);
            output->flush();
            require(
                static_cast<bool>(*output),
                "evaluation spill flush");
        }

        const auto domain =
            algebra::RootOfUnityDomain::create(
                "N", 1024);
        const std::string name =
            "test_spilled_evaluation";
        const std::vector<F> blinder{
            F(7), F(11)};

        protocol::TraceArtifacts trace;
        trace.commitments.emplace(
            name,
            gatzk::crypto::Commitment{
                .name = name,
                .point =
                    gatzk::crypto::g1_zero(),
            });
        trace.polynomial_domains[name] =
            domain->name;
        trace.spilled_polynomials.emplace(
            name,
            protocol::TraceArtifacts::
                SpilledEvaluationPolynomial{
                    .path =
                        (lease->path() /
                         payload_name)
                            .string(),
                    .domain_name =
                        domain->name,
                    .size =
                        domain->size,
                    .physical_size = 0,
                    .zero_tail_compacted = true,
                    .semantic_source_id = 1,
                    .compact_semantic_source_all_zero = true,
                    .zero_domain_blinder =
                        blinder,
                    .lease =
                        lease,
                });

        const auto &resolved_blinder =
            protocol::
                resolve_trace_zero_domain_blinder(
                    trace,
                    name,
                    domain);
        require(
            resolved_blinder == blinder,
            "spilled evaluation blinder resolution");

        const auto restored =
            protocol::
                resolve_trace_opening_polynomial(
                    trace,
                    name,
                    domain);
        require(
            restored->size() ==
                domain->size,
            "spilled evaluation logical length");
        require(
            restored->zero_domain_blinder ==
                blinder,
            "spilled evaluation blinder round trip");
        require(
            std::all_of(
                restored->values().begin(),
                restored->values().end(),
                [](const F &value)
                {
                    return value.is_zero();
                }),
            "spilled evaluation zero-tail reconstruction");

        auto &descriptor =
            trace.spilled_polynomials.at(name);
        const auto real_path =
            descriptor.path;
        descriptor.path =
            (root /
             "outside_evaluation.bin")
                .string();
        reject(
            [&]
            {
                (void)protocol::
                    resolve_trace_zero_domain_blinder(
                        trace,
                        name,
                        domain);
            },
            "evaluation spill path escape");
        descriptor.path =
            real_path;

        descriptor.lease.reset();
        reject(
            [&]
            {
                (void)protocol::
                    resolve_trace_zero_domain_blinder(
                        trace,
                        name,
                        domain);
            },
            "evaluation spill missing lease");
        descriptor.lease =
            lease;

        ++descriptor.size;
        reject(
            [&]
            {
                (void)protocol::
                    resolve_trace_zero_domain_blinder(
                        trace,
                        name,
                        domain);
            },
            "evaluation spill domain bound");
        --descriptor.size;

        trace.polynomial_domains[name] =
            "edge";
        reject(
            [&]
            {
                (void)protocol::
                    resolve_trace_zero_domain_blinder(
                        trace,
                        name,
                        domain);
            },
            "evaluation spill domain registration");
        trace.polynomial_domains[name] =
            domain->name;
    }

    void test_coefficient_publication()
    {
        protocol::TraceArtifacts trace;
        trace.spill_directory_lease = util::TraceSpillLease::create(
            std::filesystem::path(GATZK_EXECUTION_TEST_ROOT));
        trace.pi_bind = std::make_shared<protocol::PiBindProof>();
        trace.pi_bind->subproofs.emplace_back();
        const std::string name = "test_completed_coefficient";
        const std::size_t count = (std::size_t{1} << 20) + 3;
        auto material = std::make_shared<P>(P::from_coefficients(name, sample(count)));
        material->shared_data =
            std::make_shared<const std::vector<F>>(
                std::move(material->data));
        material->data.clear();
        require(
            material->values().size() == count,
            "shared coefficient publication source");
        std::weak_ptr<const P> weak = material;
        gatzk::crypto::Commitment commitment{
            .name = name, .point = gatzk::crypto::g1_zero(), .opening_polynomial = material};
        trace.commitments.emplace(name, commitment);
        trace.polynomials.emplace(name, *material);
        trace.polynomial_domains[name] = "coef";
        auto &binding = trace.pi_bind->subproofs.front();
        binding.aggregation_commitments.push_back({name, commitment});
        binding.relation_commitments.push_back({name, commitment});
        material.reset();
        commitment.opening_polynomial.reset();
        protocol::persist_completed_trace_coefficient(trace, name, true);
        require(!weak.expired(), "dump trace retains coefficient material");
        protocol::persist_completed_trace_coefficient(trace, name, false);
        require(weak.expired(), "every Commitment copy releases its material");
        require(!trace.polynomials.contains(name), "resident coefficient registry erased");
        require(trace.spilled_coefficient_polynomials.at(name).size == count, "fixed spill length");
        require(trace.commitments.at(name).point == gatzk::crypto::g1_zero(), "commitment unchanged");
        require(!binding.aggregation_commitments.front().second.opening_polynomial &&
                    !binding.relation_commitments.front().second.opening_polynomial,
                "binding owners released");
        protocol::persist_completed_trace_coefficient(trace, name, false);
        const auto restored = protocol::resolve_trace_coefficient_polynomial(trace, name);
        require(restored->values() == sample(count), "publisher/reader byte-equivalent coefficients");
    }

    void test_cuda()
    {
        require(algebra::cuda_field_hotspots_available(), "CUDA requested but unavailable");
        auto route = util::route2_options();
        route.cuda_field_hotspots = false;
        route.cuda_trace_hotspots = false;
        util::set_route2_options(route);
        const std::size_t degree = (std::size_t{1} << 17) + 2;
        const auto a = sample(degree + 1);
        algebra::release_taylor_shift_cache();
        const auto expected = algebra::coefficient_taylor_shift(a, degree, F(19));
        route.cuda_field_hotspots = true;
        route.cuda_trace_hotspots = true;
        util::set_route2_options(route);
        require(algebra::should_use_cuda_ntt(std::size_t{1} << 18), "target CUDA size policy");
        const auto before = algebra::cuda_execution_stats_snapshot();
        const auto got = algebra::coefficient_taylor_shift(a, degree, F(19));
        const auto after = algebra::cuda_execution_stats_snapshot();
        require(got == expected, "CPU/GPU Taylor coefficient equality");
        require(after.ntt_calls > before.ntt_calls && after.intt_calls > before.intt_calls,
                "fused Taylor must execute both GPU transforms");
        algebra::release_taylor_shift_cache();
        require(algebra::coefficient_taylor_shift(a, degree, F(19)) == expected,
                "cold GPU kernel-precompute and convolution parity");
        algebra::release_taylor_shift_cache();
        algebra::release_cuda_ntt_workspace();
        {
            std::vector<F> target(std::size_t{1} << 20, F::zero());
            auto session = algebra::create_cuda_scaled_accumulate_session(
                F::mutable_native_array_data(target.data()), target.size(), true);
            require(session != nullptr, "GPU scaled-accumulate session required for lifetime test");
            constexpr std::size_t chunk = 4096;
            for (std::size_t block = 0; block < 8; ++block)
            {
                auto temporary = sample(chunk);
                session->accumulate_range(
                    F::native_array_data(temporary.data()), F(3).native(),
                    block * chunk, temporary.size(), true);
            }
            session->finish();
            session.reset();
            const auto expected_chunk = sample(chunk);
            for (std::size_t i = 0; i < target.size(); ++i)
                require(target[i] == (i < 8 * chunk ? expected_chunk[i % chunk] * F(3) : F::zero()),
                        "ephemeral CUDA source lifetime");
            algebra::release_cuda_scaled_accumulate_workspace();
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool gpu = false;
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--gpu")
                gpu = true;
            else
                throw std::runtime_error("unknown argument");
        }
        auto route = util::route2_options();
        route.cuda_field_hotspots = false;
        route.cuda_trace_hotspots = false;
        route.parallel_fft = true;
        util::set_route2_options(route);
        test_horner_and_sharing();
        test_evaluation_and_blinder();
        test_taylor();
        test_coefficient_reader();
        test_evaluation_spill_registry();
        test_coefficient_publication();
        if (gpu)
            test_cuda();
        std::cout << "CORE_EXECUTION_REGRESSION_OK\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CORE_EXECUTION_REGRESSION_FAIL: " << error.what() << '\n';
        return 1;
    }
}