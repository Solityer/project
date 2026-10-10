#include "gatzk/mechanism/internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gatzk::mechanism::internal
{
    namespace
    {
        constexpr const char *kFoldedNode = "ff/H_star_node";
        constexpr const char *kFoldedEdge = "ff/H_star_edge";
        constexpr const char *kFoldChallenge = "ff/width_fold_xi";

        std::string node_label(std::size_t lane)
        {
            return "ff/H_node/" + std::to_string(lane);
        }

        std::string edge_label(std::size_t lane)
        {
            return "ff/H_edge/" + std::to_string(lane);
        }

        std::string route_stem(const PublicStatement &statement, std::size_t lane)
        {
            return statement.variant == Variant::Original ? "ff/route/folded" : "ff/route/coord/" + std::to_string(lane);
        }

        void require(bool condition, const std::string &message)
        {
            if (!condition)
                throw std::runtime_error(message);
        }

        void validate_statement(const PublicStatement &s)
        {
            require(s.mechanism == Mechanism::FeatureFolding, "Feature Folding received a different mechanism statement");
            require(s.nodes > 0 && s.edges >= s.nodes && s.width > 0 && s.width <= 32 && s.graphs > 0 && s.graphs <= s.nodes &&
                        s.node_domain_size > s.nodes && s.edge_domain_size > s.edges,
                    "Feature Folding statement has an illegal workload shape");
            require(s.node_ptr.size() == s.graphs + 1 && s.edge_ptr.size() == s.graphs + 1 && s.topology.size() == s.edges &&
                        s.node_ptr.front() == 0 && s.node_ptr.back() == s.nodes && s.edge_ptr.front() == 0 && s.edge_ptr.back() == s.edges,
                    "Feature Folding graph pointer or topology shape mismatch");
            for (std::size_t graph = 0; graph < s.graphs; ++graph)
            {
                require(s.node_ptr[graph] < s.node_ptr[graph + 1] && s.node_ptr[graph + 1] <= s.nodes &&
                            s.edge_ptr[graph] < s.edge_ptr[graph + 1] && s.edge_ptr[graph + 1] <= s.edges,
                        "Feature Folding graph has an empty node or edge range");
                require(s.edge_ptr[graph + 1] - s.edge_ptr[graph] >= s.node_ptr[graph + 1] - s.node_ptr[graph],
                        "Feature Folding graph does not provide at least one outgoing edge per node");
                for (std::size_t edge_index = s.edge_ptr[graph]; edge_index < s.edge_ptr[graph + 1]; ++edge_index)
                    require(s.topology[edge_index].graph == graph,
                            "Feature Folding edge pointer disagrees with the edge graph id");
            }

            std::vector<bool> self_edge(s.nodes, false);
            std::vector<std::size_t> outdegree(s.nodes, 0);
            std::tuple<std::uint64_t, std::uint64_t> previous{};
            for (std::size_t edge_index = 0; edge_index < s.topology.size(); ++edge_index)
            {
                const auto &edge = s.topology[edge_index];
                require(edge.graph < s.graphs && edge.src >= s.node_ptr[edge.graph] && edge.src < s.node_ptr[edge.graph + 1] &&
                            edge.dst >= s.node_ptr[edge.graph] && edge.dst < s.node_ptr[edge.graph + 1],
                        "Feature Folding topology contains a cross-graph or out-of-range edge");
                const auto current = std::make_tuple(edge.graph, edge.dst);
                require(edge_index == 0 || previous <= current, "Feature Folding topology is not in canonical graph/destination group order");
                previous = current;
                ++outdegree[static_cast<std::size_t>(edge.src)];
                if (edge.src == edge.dst)
                    self_edge[static_cast<std::size_t>(edge.src)] = true;
            }
            for (std::size_t node = 0; node < s.nodes; ++node)
                require(outdegree[node] != 0 && self_edge[node], "Feature Folding node lacks its required positive outdegree/self-edge");
        }

        std::vector<std::string> source_labels(const PublicStatement &s)
        {
            std::vector<std::string> labels;
            labels.reserve(2 * s.width);
            for (std::size_t lane = 0; lane < s.width; ++lane)
            {
                labels.push_back(node_label(lane));
                labels.push_back(edge_label(lane));
            }
            return labels;
        }

        struct RawWitness
        {
            std::vector<std::vector<FieldElement>> node;
            std::vector<std::vector<FieldElement>> edge;
        };

        RawWitness make_raw_witness(const ProofBuildContext &context)
        {
            const auto &s = context.statement;
            RawWitness raw;
            raw.node.resize(s.width);
            raw.edge.resize(s.width);
            if (context.real_input != nullptr)
            {
                const auto &w = context.real_input->witness;
                require(w.width == 8 && w.node_count == s.nodes && w.edge_count == s.edges && w.node_ptr == s.node_ptr &&
                            w.edge_ptr == s.edge_ptr && s.width <= w.width,
                        "Feature Folding real input does not match the statement or formal first-head width");
                for (std::size_t edge = 0; edge < s.edges; ++edge)
                    require(w.edge_src_indices[edge] == s.topology[edge].src && w.edge_dst_indices[edge] == s.topology[edge].dst,
                            "Feature Folding real input topology differs from the public statement");
                for (std::size_t lane = 0; lane < s.width; ++lane)
                {
                    raw.node[lane].reserve(s.nodes);
                    for (std::size_t node = 0; node < s.nodes; ++node)
                        raw.node[lane].push_back(w.h_prime[node][lane]);
                    raw.edge[lane].resize(s.edges, FieldElement::zero());
                    for (std::size_t edge = 0; edge < s.edges; ++edge)
                        raw.edge[lane][edge] = raw.node[lane].at(static_cast<std::size_t>(s.topology[edge].src));
                }
                return raw;
            }
            for (std::size_t lane = 0; lane < s.width; ++lane)
            {
                raw.node[lane] = derive_witness_fields(context.witness_seed, node_label(lane), s.nodes);
                raw.edge[lane].resize(s.edges, FieldElement::zero());
                for (std::size_t edge_index = 0; edge_index < s.edges; ++edge_index)
                {
                    const auto source = static_cast<std::size_t>(s.topology[edge_index].src);
                    raw.edge[lane][edge_index] = raw.node[lane].at(source);
                }
            }
            return raw;
        }

        std::vector<FieldElement> pad(const std::vector<FieldElement> &values, std::size_t domain_size)
        {
            require(values.size() <= domain_size, "Feature Folding evaluation vector exceeds its physical domain");
            auto padded = values;
            padded.resize(domain_size, FieldElement::zero());
            return padded;
        }

        void commit_raw_witness(ProofBuildContext &context, const RawWitness &raw)
        {
            const auto &s = context.statement;
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            for (std::size_t lane = 0; lane < s.width; ++lane)
            {
                commit_masked_evaluations(context, node_label(lane), pad(raw.node[lane], s.node_domain_size), node_domain, 8);
                commit_masked_evaluations(context, edge_label(lane), pad(raw.edge[lane], s.edge_domain_size), edge_domain, 8);
            }
        }

        std::vector<std::pair<std::string, FieldElement>> folding_terms(const PublicStatement &s, const FieldElement &xi, bool edge)
        {
            std::vector<std::pair<std::string, FieldElement>> terms;
            terms.reserve(s.width);
            FieldElement power = FieldElement::one();
            for (std::size_t lane = 0; lane < s.width; ++lane)
            {
                terms.emplace_back(edge ? edge_label(lane) : node_label(lane), power);
                power *= xi;
            }
            return terms;
        }

        void require_rank_eight_lineage(const ProofBuildContext &context, const std::string &derived_label,
                                        const std::vector<std::pair<std::string, FieldElement>> &terms)
        {
            const auto derived = context.objects.find(derived_label);
            require(derived != context.objects.end() && derived->second.opening_polynomial != nullptr,
                    "Feature Folding derived polynomial is missing: " + derived_label);
            std::vector<FieldElement> expected(8, FieldElement::zero());
            for (const auto &[label, scalar] : terms)
            {
                const auto source = context.objects.find(label);
                require(source != context.objects.end() && source->second.opening_polynomial != nullptr &&
                            source->second.opening_polynomial->zero_domain_blinder.size() == 8,
                        "Feature Folding source does not carry its rank-eight privacy mask: " + label);
                for (std::size_t i = 0; i < expected.size(); ++i)
                    expected[i] += scalar * source->second.opening_polynomial->zero_domain_blinder[i];
            }
            require(derived->second.opening_polynomial->zero_domain_blinder == expected,
                    "Feature Folding derived polynomial did not preserve its full zero-domain mask lineage: " + derived_label);
        }

        std::vector<FieldElement> fold_values(const std::vector<std::vector<FieldElement>> &columns, const FieldElement &xi,
                                              std::size_t domain_size)
        {
            std::vector<FieldElement> output(domain_size, FieldElement::zero());
            FieldElement power = FieldElement::one();
            for (const auto &column : columns)
            {
                require(column.size() <= domain_size, "Feature Folding source column exceeds its physical domain");
                for (std::size_t i = 0; i < column.size(); ++i)
                    output[i] += power * column[i];
                power *= xi;
            }
            return output;
        }

        std::vector<FieldElement> outdegree_multiplicity(const PublicStatement &s)
        {
            std::vector<FieldElement> multiplicity(s.node_domain_size, FieldElement::zero());
            for (const auto &edge : s.topology)
                multiplicity.at(static_cast<std::size_t>(edge.src)) += FieldElement::one();
            return multiplicity;
        }

        std::vector<FieldElement> public_node_ids(const PublicStatement &s)
        {
            std::vector<FieldElement> values(s.node_domain_size, FieldElement::zero());
            for (std::size_t node = 0; node < s.nodes; ++node)
                values[node] = FieldElement(static_cast<std::uint64_t>(node));
            return values;
        }

        std::vector<FieldElement> public_edge_sources(const PublicStatement &s)
        {
            std::vector<FieldElement> values(s.edge_domain_size, FieldElement::zero());
            for (std::size_t edge = 0; edge < s.edges; ++edge)
                values[edge] = FieldElement(s.topology[edge].src);
            return values;
        }

        Polynomial tuple_polynomial(const std::string &label, const std::vector<FieldElement> &public_ids,
                                     const Commitment &feature, const FieldElement &eta,
                                     const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            require(feature.opening_polynomial != nullptr && feature.opening_polynomial->basis == algebra::PolynomialBasis::Evaluation &&
                        feature.opening_polynomial->domain != nullptr && feature.opening_polynomial->domain->name == domain->name &&
                        feature.opening_polynomial->domain->size == domain->size && feature.opening_polynomial->size() == domain->size &&
                        feature.opening_polynomial->zero_domain_blinder.size() == 8 && public_ids.size() == domain->size,
                    "Feature Folding tuple source lacks matching domain or rank-eight mask: " + label);
            std::vector<FieldElement> values(domain->size, FieldElement::zero());
            for (std::size_t i = 0; i < domain->size; ++i)
                values[i] = public_ids[i] + eta * feature.opening_polynomial->value_at(i);
            auto polynomial = Polynomial::from_evaluations(label, std::move(values), domain);
            auto tuple_mask = feature.opening_polynomial->zero_domain_blinder;
            for (auto &coefficient : tuple_mask)
                coefficient *= eta;
            polynomial.set_zero_domain_blinder(std::move(tuple_mask));
            return polynomial;
        }

        void absorb_recorded_commitment(ProofVerifyContext &context, const std::string &label)
        {
            const auto &record = required_commitment(context.proof, label);
            context.transcript.absorb_commitment(label, record.point);
            context.objects.emplace(label, Commitment{label, record.point, nullptr});
        }

        bool verify_linear_commitment_point(const MechanismProof &proof, const std::string &derived_label,
                                            const std::vector<std::pair<std::string, FieldElement>> &terms)
        {
            std::vector<crypto::G1Point> points;
            std::vector<FieldElement> scalars;
            points.reserve(terms.size());
            scalars.reserve(terms.size());
            for (const auto &[label, scalar] : terms)
            {
                points.push_back(required_commitment(proof, label).point);
                scalars.push_back(scalar);
            }
            return crypto::g1_multi_scalar_mul(points, scalars) == required_commitment(proof, derived_label).point;
        }

        FieldElement evaluate_public(const std::string &label, const std::vector<FieldElement> &values,
                                     const std::shared_ptr<algebra::RootOfUnityDomain> &domain, const FieldElement &point)
        {
            return Polynomial::from_evaluations(label, values, domain).evaluate(point);
        }

        bool check_route_tuple(const ProofVerifyContext &context, const std::string &stem,
                               const std::string &node_feature_label, const std::string &edge_feature_label,
                               const FieldElement &eta, const FieldElement &zeta)
        {
            const auto group = required_opening_group(context.proof, stem + "/zeta");
            const auto table = required_opening_value(group, stem + "/Table");
            const auto query = required_opening_value(group, stem + "/Query");
            const auto node_feature = required_opening_value(group, node_feature_label);
            const auto edge_feature = required_opening_value(group, edge_feature_label);
            const auto &s = context.statement;
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            return table == evaluate_public("ff/public/node_id", public_node_ids(s), node_domain, zeta) + eta * node_feature &&
                   query == evaluate_public("ff/public/edge_src", public_edge_sources(s), edge_domain, zeta) + eta * edge_feature;
        }

        void prove_one_route(ProofBuildContext &context, const std::string &stem,
                             const std::string &node_feature_label, const std::string &edge_feature_label)
        {
            const auto &s = context.statement;
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto eta = context.transcript.challenge_excluding(stem + "/eta", [](const FieldElement &value) { return value.is_zero(); });
            const auto table_label = stem + "/Table";
            const auto query_label = stem + "/Query";
            auto table_poly = tuple_polynomial(table_label, public_node_ids(s), context.objects.at(node_feature_label), eta, node_domain);
            auto query_poly = tuple_polynomial(query_label, public_edge_sources(s), context.objects.at(edge_feature_label), eta, edge_domain);
            const auto table_values = table_poly.values();
            const auto query_values = query_poly.values();
            commit_polynomial(context, table_label, std::move(table_poly));
            commit_polynomial(context, query_label, std::move(query_poly));
            const auto multiplicity = outdegree_multiplicity(s);
            const auto zeta = prove_route_argument(context, stem, table_values, query_values, multiplicity, s.nodes, s.edges,
                                                   node_domain, edge_domain, {node_feature_label, edge_feature_label});
            require(evaluate_public("ff/public/node_id", public_node_ids(s), node_domain, zeta) +
                            eta * context.objects.at(node_feature_label).opening_polynomial->evaluate(zeta) ==
                        context.objects.at(table_label).opening_polynomial->evaluate(zeta),
                    "Feature Folding prover tuple does not bind node feature to its public id");
            require(evaluate_public("ff/public/edge_src", public_edge_sources(s), edge_domain, zeta) +
                            eta * context.objects.at(edge_feature_label).opening_polynomial->evaluate(zeta) ==
                        context.objects.at(query_label).opening_polynomial->evaluate(zeta),
                    "Feature Folding prover tuple does not bind edge feature to its public source id");
        }

        bool verify_one_route(ProofVerifyContext &context, const std::string &stem,
                              const std::string &node_feature_label, const std::string &edge_feature_label)
        {
            const auto &s = context.statement;
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto eta = context.transcript.challenge_excluding(stem + "/eta", [](const FieldElement &value) { return value.is_zero(); });
            absorb_recorded_commitment(context, stem + "/Table");
            absorb_recorded_commitment(context, stem + "/Query");
            FieldElement zeta = FieldElement::zero();
            if (!verify_route_argument(context, stem, outdegree_multiplicity(s), s.nodes, s.edges, node_domain, edge_domain,
                                       {node_feature_label, edge_feature_label}, &zeta))
                return false;
            return check_route_tuple(context, stem, node_feature_label, edge_feature_label, eta, zeta);
        }
    }

    std::vector<std::string> feature_folding_commitment_labels(const PublicStatement &s)
    {
        validate_statement(s);
        auto labels = source_labels(s);
        if (s.variant == Variant::Original)
        {
            labels.emplace_back(kFoldedNode);
            labels.emplace_back(kFoldedEdge);
            const auto route = route_commitment_labels(route_stem(s, 0));
            labels.insert(labels.end(), route.begin(), route.end());
        }
        else
        {
            for (std::size_t lane = 0; lane < s.width; ++lane)
            {
                const auto route = route_commitment_labels(route_stem(s, lane));
                labels.insert(labels.end(), route.begin(), route.end());
            }
        }
        return labels;
    }

    std::vector<std::string> feature_folding_opening_group_labels(const PublicStatement &s)
    {
        validate_statement(s);
        std::vector<std::string> labels;
        const auto route_count = s.variant == Variant::Original ? std::size_t(1) : s.width;
        for (std::size_t lane = 0; lane < route_count; ++lane)
        {
            const auto route = route_opening_group_labels(route_stem(s, lane));
            labels.insert(labels.end(), route.begin(), route.end());
        }
        return labels;
    }

    void prove_feature_folding(ProofBuildContext &context)
    {
        const auto &s = context.statement;
        validate_statement(s);
        const auto raw = make_raw_witness(context);
        commit_raw_witness(context, raw);

        if (s.variant == Variant::Original)
        {
            // xi is sampled only after all source columns are fixed, so a prover cannot choose
            // a favorable projection after seeing the folding challenge.
            const auto xi = context.transcript.challenge_excluding(kFoldChallenge,
                                                                    [](const FieldElement &value) { return value.is_zero(); });
            const auto node_terms = folding_terms(s, xi, false);
            const auto edge_terms = folding_terms(s, xi, true);
            derive_linear_commitment(context, kFoldedNode, node_terms);
            derive_linear_commitment(context, kFoldedEdge, edge_terms);
            require_rank_eight_lineage(context, kFoldedNode, node_terms);
            require_rank_eight_lineage(context, kFoldedEdge, edge_terms);
            prove_one_route(context, route_stem(s, 0), kFoldedNode, kFoldedEdge);
        }
        else
        {
            // Each coordinate receives a fresh tuple-compression challenge and a complete
            // route proof, replacing width folding with one independently checked route per lane.
            for (std::size_t lane = 0; lane < s.width; ++lane)
                prove_one_route(context, route_stem(s, lane), node_label(lane), edge_label(lane));
        }
    }

    bool verify_feature_folding(ProofVerifyContext &context)
    {
        const auto &s = context.statement;
        validate_statement(s);
        for (const auto &label : source_labels(s))
            absorb_recorded_commitment(context, label);

        if (s.variant == Variant::Original)
        {
            const auto xi = context.transcript.challenge_excluding(kFoldChallenge,
                                                                    [](const FieldElement &value) { return value.is_zero(); });
            const auto node_terms = folding_terms(s, xi, false);
            const auto edge_terms = folding_terms(s, xi, true);
            if (!verify_linear_commitment_point(context.proof, kFoldedNode, node_terms) ||
                !verify_linear_commitment_point(context.proof, kFoldedEdge, edge_terms))
                return false;
            absorb_recorded_commitment(context, kFoldedNode);
            absorb_recorded_commitment(context, kFoldedEdge);
            return verify_one_route(context, route_stem(s, 0), kFoldedNode, kFoldedEdge);
        }

        for (std::size_t lane = 0; lane < s.width; ++lane)
            if (!verify_one_route(context, route_stem(s, lane), node_label(lane), edge_label(lane)))
                return false;
        return true;
    }
}
