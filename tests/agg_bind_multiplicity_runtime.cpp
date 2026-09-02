#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/protocol/proof.hpp"
#include "protocol/internal/verifier_public_polynomial.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::algebra::Polynomial;
    using gatzk::algebra::RootOfUnityDomain;
    using gatzk::data::Edge;
    using gatzk::data::LocalGraph;
    using gatzk::protocol::ProtocolContext;
    using gatzk::protocol::VerifierPublicPolynomialDescriptor;
    using gatzk::protocol::internal::AggregationMultiplicityCache;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    void expect_rejected(
        const LocalGraph &local,
        std::size_t n_domain_size,
        const std::string &label)
    {
        bool rejected = false;
        try
        {
            (void)gatzk::protocol::internal::build_aggregation_multiplicity_cache(
                local,
                n_domain_size);
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        expect(rejected, label + " was accepted");
    }

    FieldElement first_non_subgroup_point(const RootOfUnityDomain &domain)
    {
        auto candidate = FieldElement(2);
        while (domain.rotation_shift(FieldElement::one(), candidate).has_value())
            candidate += FieldElement::one();
        return candidate;
    }

    ProtocolContext small_context(
        const std::shared_ptr<RootOfUnityDomain> &n_domain,
        const std::shared_ptr<RootOfUnityDomain> &edge_domain,
        const LocalGraph &local)
    {
        ProtocolContext context;
        context.local = local;
        context.domains.fh = RootOfUnityDomain::create("FH", 8);
        context.domains.edge = edge_domain;
        context.domains.in = RootOfUnityDomain::create("in", 8);
        context.domains.d = RootOfUnityDomain::create("d", 8);
        context.domains.cat = RootOfUnityDomain::create("cat", 8);
        context.domains.c = RootOfUnityDomain::create("C", 8);
        context.domains.n = n_domain;
        return context;
    }

    void check_polynomial(
        const ProtocolContext &context,
        const AggregationMultiplicityCache &multiplicity,
        const std::string &name,
        const std::vector<FieldElement> &canonical_values)
    {
        const auto domain = context.domains.n;
        const VerifierPublicPolynomialDescriptor descriptor{
            .name = name,
            .basis = gatzk::algebra::PolynomialBasis::Evaluation,
            .domain = "N",
            .size = domain->size};
        const auto canonical = Polynomial::from_evaluations(
            name,
            canonical_values,
            domain);

        // A point on the subgroup must select the canonical row directly.
        const auto subgroup_point = domain->point_at(2);
        expect(
            gatzk::protocol::internal::evaluate_deterministic_public_polynomial(
                context,
                descriptor,
                subgroup_point,
                multiplicity)
                == canonical.evaluate(subgroup_point),
            name + " disagrees at subgroup point");

        // An off-subgroup point exercises the full barycentric reconstruction.
        const auto non_subgroup_point = first_non_subgroup_point(*domain);
        expect(
            gatzk::protocol::internal::evaluate_deterministic_public_polynomial(
                context,
                descriptor,
                non_subgroup_point,
                multiplicity)
                == canonical.evaluate(non_subgroup_point),
            name + " disagrees at non-subgroup point");

        // The final three rows are domain padding and must remain zero.
        const auto padding_point = domain->point_at(domain->size - 1);
        expect(
            gatzk::protocol::internal::evaluate_deterministic_public_polynomial(
                context,
                descriptor,
                padding_point,
                multiplicity)
                .is_zero(),
            name + " did not preserve zero padding");
    }
}

int main()
{
    try
    {
        constexpr std::size_t n_domain_size = 8;
        constexpr std::size_t edge_domain_size = 8;
        LocalGraph local;
        local.num_nodes = 5;
        local.num_features = 1;
        local.num_classes = 2;
        local.absolute_ids = {0, 1, 2, 3, 4};
        // Duplicate edges and self-loops are deliberate. Nodes 3 and 4 are
        // isolated, while rows 5..7 below are N-domain padding.
        local.edges = {
            Edge{.src = 0, .dst = 0, .graph_id = 0, .stable_index = 0},
            Edge{.src = 0, .dst = 0, .graph_id = 0, .stable_index = 1},
            Edge{.src = 1, .dst = 0, .graph_id = 0, .stable_index = 2},
            Edge{.src = 2, .dst = 1, .graph_id = 0, .stable_index = 3},
            Edge{.src = 2, .dst = 2, .graph_id = 0, .stable_index = 4},
            Edge{.src = 2, .dst = 2, .graph_id = 0, .stable_index = 5},
        };

        const auto n_domain = RootOfUnityDomain::create("N", n_domain_size);
        const auto edge_domain = RootOfUnityDomain::create("edge", edge_domain_size);
        const auto context = small_context(n_domain, edge_domain, local);
        const auto multiplicity =
            gatzk::protocol::internal::build_aggregation_multiplicity_cache(
                local,
                n_domain_size);

        std::vector<FieldElement> canonical_src(n_domain_size, FieldElement::zero());
        std::vector<FieldElement> canonical_dst(n_domain_size, FieldElement::zero());
        for (const auto &edge : local.edges)
        {
            canonical_src[edge.src] += FieldElement::one();
            canonical_dst[edge.dst] += FieldElement::one();
        }
        expect(multiplicity.src == canonical_src,
               "cached src multiplicity differs from canonical edge scan");
        expect(multiplicity.dst == canonical_dst,
               "cached dst multiplicity differs from canonical edge scan");
        expect(multiplicity.src[3].is_zero() && multiplicity.src[4].is_zero()
                   && multiplicity.dst[3].is_zero() && multiplicity.dst[4].is_zero()
                   && multiplicity.src[5].is_zero() && multiplicity.dst[5].is_zero()
                   && multiplicity.src[7].is_zero() && multiplicity.dst[7].is_zero(),
               "isolated or padded multiplicity rows are not zero");

        check_polynomial(context, multiplicity, "P_AggBind_m_src", canonical_src);
        check_polynomial(context, multiplicity, "P_AggBind_m_dst", canonical_dst);

        auto invalid_src = local;
        invalid_src.edges.push_back(
            Edge{.src = invalid_src.num_nodes, .dst = 0, .graph_id = 0, .stable_index = 6});
        expect_rejected(
            invalid_src,
            n_domain_size,
            "out-of-range src edge");

        auto invalid_dst = local;
        invalid_dst.edges.push_back(
            Edge{.src = 0, .dst = invalid_dst.num_nodes, .graph_id = 0, .stable_index = 6});
        expect_rejected(
            invalid_dst,
            n_domain_size,
            "out-of-range dst edge");

        auto oversized_nodes = local;
        oversized_nodes.num_nodes = n_domain_size + 1;
        expect_rejected(
            oversized_nodes,
            n_domain_size,
            "node count larger than N domain");

        std::cout << "agg_bind_multiplicity_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "agg_bind_multiplicity_runtime: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
