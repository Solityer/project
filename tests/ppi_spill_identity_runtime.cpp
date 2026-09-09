#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::RootOfUnityDomain;
    using gatzk::protocol::SpillEvaluationTestInput;
    using gatzk::protocol::TraceArtifacts;

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

    struct SharedColumns
    {
        std::vector<FieldElement> first;
        std::vector<FieldElement> second;
    };

    std::vector<FieldElement> evaluate(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<RootOfUnityDomain> &domain,
        const std::vector<FieldElement> &points)
    {
        std::vector<const gatzk::algebra::DomainEvaluationWeights *> weights;
        std::vector<gatzk::algebra::DomainEvaluationWeights> storage;
        storage.reserve(points.size());
        weights.reserve(points.size());
        for (const auto &point : points)
        {
            storage.push_back(
                gatzk::algebra::build_domain_evaluation_weights(domain, point));
        }
        for (const auto &entry : storage)
        {
            weights.push_back(&entry);
        }
        return gatzk::protocol::evaluate_spilled_evaluation_polynomial_points(
            spilled,
            domain,
            points,
            weights);
    }

    FieldElement non_domain_point(
        const std::shared_ptr<RootOfUnityDomain> &domain)
    {
        auto point = FieldElement(7);
        while (domain->rotation_shift(FieldElement::one(), point).has_value())
        {
            point += FieldElement::one();
        }
        return point;
    }
}

int main()
{
    try
    {
        constexpr std::size_t logical_size = 1U << 12U;
        const auto domain = RootOfUnityDomain::create(
            "ppi_identity",
            logical_size);
        auto owner = std::make_shared<SharedColumns>();
        owner->first.resize(logical_size);
        owner->second.resize(logical_size);
        for (std::size_t index = 0; index < logical_size; ++index)
        {
            owner->first[index] = FieldElement(index + 3U);
            owner->second[index] = FieldElement(index + 10003U);
        }

        // The aliasing pointers share one control block.  The exact-view pair
        // also has the same get() pointer; the member-view pair deliberately
        // differs only in get(), which must fail closed for semantic reuse.
        const auto first_view =
            std::shared_ptr<const std::vector<FieldElement>>(
                owner,
                &owner->first);
        const auto second_view =
            std::shared_ptr<const std::vector<FieldElement>>(
                owner,
                &owner->second);

        const std::vector<FieldElement> blinder{
            FieldElement(11),
            FieldElement(13),
            FieldElement(17)};
        TraceArtifacts trace;
        const auto exact_a =
            gatzk::protocol::spill_evaluation_polynomial_for_tests(
                trace,
                SpillEvaluationTestInput{
                    .name = "ppi_identity_exact_a",
                    .shared_values = first_view,
                    .zero_domain_blinder = blinder,
                    .domain = domain});
        const auto exact_b =
            gatzk::protocol::spill_evaluation_polynomial_for_tests(
                trace,
                SpillEvaluationTestInput{
                    .name = "ppi_identity_exact_b",
                    .shared_values = first_view,
                    .zero_domain_blinder = blinder,
                    .domain = domain});
        expect(exact_a.semantic_source_id != 0, "exact view lacks source token");
        expect(exact_a.semantic_source_id == exact_b.semantic_source_id,
               "exact shared view did not retain its token");
        expect(exact_a.path == exact_b.path,
               "exact shared view did not reuse its spill path");
        expect(exact_a.domain_name == domain->name &&
                   exact_a.size == logical_size &&
                   exact_a.physical_size == logical_size &&
                   exact_a.zero_tail_compacted && exact_a.lease != nullptr,
               "exact spill descriptor lost domain/physical/lease metadata");

        const auto member_view =
            gatzk::protocol::spill_evaluation_polynomial_for_tests(
                trace,
                SpillEvaluationTestInput{
                    .name = "ppi_identity_member",
                    .shared_values = second_view,
                    .zero_domain_blinder = blinder,
                    .domain = domain});
        expect(member_view.semantic_source_id != exact_a.semantic_source_id,
               "different member view reused exact source token");
        expect(member_view.path != exact_a.path,
               "different member view reused exact spill path");

        const auto point = non_domain_point(domain);
        const auto exact_values = evaluate(exact_a, domain, {point});
        const auto member_values = evaluate(member_view, domain, {point});
        expect(exact_values.size() == 1 && member_values.size() == 1,
               "spill evaluation returned wrong point count");
        expect(exact_values.front() != member_values.front(),
               "different member views unexpectedly evaluated identically");

        auto wrong_domain = exact_a;
        wrong_domain.domain_name = "other_domain";
        expect_failure(
            [&]()
            {
                (void)evaluate(wrong_domain, domain, {point});
            },
            "domain-name mismatch was not fail-closed");

        auto wrong_size = exact_a;
        wrong_size.size = logical_size + 1;
        expect_failure(
            [&]()
            {
                (void)evaluate(wrong_size, domain, {point});
            },
            "logical-size mismatch was not fail-closed");

        auto wrong_path = exact_a;
        wrong_path.path = (std::filesystem::path(exact_a.path).parent_path() /
                           "missing-payload.bin")
                              .string();
        expect_failure(
            [&]()
            {
                (void)evaluate(wrong_path, domain, {point});
            },
            "lease/path mismatch was not fail-closed");

        std::cout << "status=PASS exact_token=" << exact_a.semantic_source_id
                  << " member_token=" << member_view.semantic_source_id
                  << " exact_path=" << exact_a.path
                  << " member_path=" << member_view.path << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "status=FAIL error=" << error.what() << '\n';
        return 1;
    }
}
