#include "gatzk/mechanism/internal.hpp"

#include "gatzk/protocol/psq.hpp"
#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace gatzk::mechanism::internal
{
    namespace
    {
        constexpr const char *kU = "sn/U_edge";
        constexpr const char *kVStar = "sn/V_star_edge";
        constexpr const char *kSum = "sn/Sum_node";
        constexpr const char *kHaggStar = "sn/Hagg_star_node";
        constexpr const char *kW = "sn/w_edge";
        constexpr const char *kT = "sn/T_node";

        std::string v_label(std::size_t lane) { return "sn/V_edge/" + std::to_string(lane); }
        std::string hagg_label(std::size_t lane) { return "sn/Hagg_node/" + std::to_string(lane); }

        std::vector<std::string> channels(const PublicStatement &statement)
        {
            return statement.variant == Variant::Original ? std::vector<std::string>{"shared"}
                                                            : std::vector<std::string>{"den", "agg"};
        }

        std::vector<std::string> route_labels(const std::string &stem)
        {
            return route_commitment_labels(stem);
        }

        void require(bool condition, const char *message)
        {
            if (!condition)
                throw std::runtime_error(message);
        }

        void validate_topology(const PublicStatement &s)
        {
            require(s.mechanism == Mechanism::SharedNeighborhood, "Shared Neighborhood received a different mechanism statement");
            require(s.nodes > 0 && s.edges >= s.nodes && s.graphs > 0 && s.graphs <= s.nodes && s.width > 0 && s.width <= 32 &&
                        s.node_domain_size > s.nodes && s.edge_domain_size > s.edges,
                    "Shared Neighborhood statement has an illegal workload shape");
            require(s.node_ptr.size() == s.graphs + 1 && s.edge_ptr.size() == s.graphs + 1 && s.topology.size() == s.edges,
                    "Shared Neighborhood graph pointer/topology shape mismatch");
            require(s.node_ptr.front() == 0 && s.node_ptr.back() == s.nodes && s.edge_ptr.front() == 0 && s.edge_ptr.back() == s.edges,
                    "Shared Neighborhood graph pointers have invalid endpoints");
            for (std::size_t graph = 0; graph < s.graphs; ++graph)
                require(s.node_ptr[graph] < s.node_ptr[graph + 1] && s.node_ptr[graph + 1] <= s.nodes &&
                            s.edge_ptr[graph] < s.edge_ptr[graph + 1] && s.edge_ptr[graph + 1] <= s.edges,
                        "Shared Neighborhood graph pointer range is empty or out of bounds");
            std::vector<std::size_t> indegree(s.nodes, 0);
            std::vector<bool> self_edge(s.nodes, false);
            for (std::size_t graph = 0; graph < s.graphs; ++graph)
            {
                require(s.node_ptr[graph] < s.node_ptr[graph + 1] && s.edge_ptr[graph] < s.edge_ptr[graph + 1],
                        "Shared Neighborhood graph has an empty node or edge range");
                for (std::size_t edge_index = s.edge_ptr[graph]; edge_index < s.edge_ptr[graph + 1]; ++edge_index)
                    require(s.topology[edge_index].graph == graph, "Shared Neighborhood edge pointer disagrees with edge graph id");
            }
            std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> previous{};
            for (std::size_t i = 0; i < s.topology.size(); ++i)
            {
                const auto &edge = s.topology[i];
                require(edge.graph < s.graphs && edge.src >= s.node_ptr[edge.graph] && edge.src < s.node_ptr[edge.graph + 1] &&
                            edge.dst >= s.node_ptr[edge.graph] && edge.dst < s.node_ptr[edge.graph + 1],
                        "Shared Neighborhood topology contains a cross-graph or out-of-range edge");
                const auto current = std::make_tuple(edge.graph, edge.dst, edge.src);
                require(i == 0 || previous <= current, "Shared Neighborhood topology is not in canonical graph/destination/source order");
                previous = current;
                ++indegree[edge.dst];
                if (edge.src == edge.dst)
                    self_edge[edge.dst] = true;
            }
            for (std::size_t node = 0; node < s.nodes; ++node)
            {
                // The current statement contract includes a self-edge for every node, so every target has
                // positive public multiplicity. Keep the zero-indegree case fail-closed if that contract changes.
                require(indegree[node] == 0 || self_edge[node], "Shared Neighborhood positive-indegree node lacks its required self-edge");
                require(indegree[node] != 0, "Shared Neighborhood statement leaves a zero-indegree target unconstrained");
            }
        }

        std::vector<FieldElement> pad_to_domain(const std::vector<FieldElement> &values, std::size_t size)
        {
            if (values.size() > size)
                throw std::runtime_error("Shared Neighborhood witness exceeds its authenticated evaluation domain");
            auto result = values;
            result.resize(size, FieldElement::zero());
            return result;
        }

        struct RawWitness
        {
            std::vector<FieldElement> u;
            std::vector<std::vector<FieldElement>> v;
            std::vector<FieldElement> sum;
            std::vector<std::vector<FieldElement>> hagg;
        };

        RawWitness make_raw_witness(const ProofBuildContext &context)
        {
            const auto &s = context.statement;
            RawWitness out;
            out.u = derive_witness_fields(context.witness_seed, kU, s.edges);
            out.v.reserve(s.width);
            out.hagg.assign(s.width, std::vector<FieldElement>(s.nodes, FieldElement::zero()));
            out.sum.assign(s.nodes, FieldElement::zero());
            std::vector<FieldElement> indegree(s.nodes, FieldElement::zero());

            for (std::size_t lane = 0; lane < s.width; ++lane)
                out.v.push_back(derive_witness_fields(context.witness_seed, v_label(lane), s.edges));

            for (std::size_t edge_index = 0; edge_index < s.edges; ++edge_index)
            {
                const auto &edge = s.topology[edge_index];
                const auto dst = static_cast<std::size_t>(edge.dst);
                out.sum[dst] += out.u[edge_index];
                indegree[dst] += FieldElement::one();
                for (std::size_t lane = 0; lane < s.width; ++lane)
                    out.hagg[lane][dst] += out.v[lane][edge_index];
            }
            for (std::size_t node = 0; node < s.nodes; ++node)
            {
                if (indegree[node].is_zero())
                {
                    if (!out.sum[node].is_zero())
                        throw std::runtime_error("zero-indegree node has a nonzero denominator target");
                    for (std::size_t lane = 0; lane < s.width; ++lane)
                        if (!out.hagg[lane][node].is_zero())
                            throw std::runtime_error("zero-indegree node has a nonzero aggregate target");
                }
            }
            return out;
        }

        std::vector<std::string> source_commitment_labels(const PublicStatement &s)
        {
            std::vector<std::string> labels{kU};
            for (std::size_t lane = 0; lane < s.width; ++lane)
                labels.push_back(v_label(lane));
            labels.push_back(kSum);
            for (std::size_t lane = 0; lane < s.width; ++lane)
                labels.push_back(hagg_label(lane));
            return labels;
        }

        std::vector<std::string> shared_base_commitment_labels(const PublicStatement &s)
        {
            auto labels = source_commitment_labels(s);
            labels.push_back(kVStar);
            labels.push_back(kHaggStar);
            if (s.variant == Variant::Original)
            {
                labels.push_back(kW);
                labels.push_back(kT);
            }
            return labels;
        }

        std::string psq_prefix(const std::string &channel) { return "sn/psq/" + channel; }
        std::string route_prefix(const std::string &channel) { return "sn/route/" + channel; }
        std::string target_edge_label(const std::string &channel) { return "sn/target_edge/" + channel; }

        void commit_raw_witness(ProofBuildContext &context, const RawWitness &raw)
        {
            const auto &s = context.statement;
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            commit_masked_evaluations(context, kU, pad_to_domain(raw.u, s.edge_domain_size), edge_domain, 8);
            for (std::size_t lane = 0; lane < s.width; ++lane)
                commit_masked_evaluations(context, v_label(lane), pad_to_domain(raw.v[lane], s.edge_domain_size), edge_domain, 8);
            commit_masked_evaluations(context, kSum, pad_to_domain(raw.sum, s.node_domain_size), node_domain, 8);
            for (std::size_t lane = 0; lane < s.width; ++lane)
                commit_masked_evaluations(context, hagg_label(lane), pad_to_domain(raw.hagg[lane], s.node_domain_size), node_domain, 8);
        }

        void require_linear_mask_lineage(const ProofBuildContext &context, const std::string &derived_label,
                                         const std::vector<std::pair<std::string, FieldElement>> &terms)
        {
            const auto derived_it = context.objects.find(derived_label);
            if (derived_it == context.objects.end() || derived_it->second.opening_polynomial == nullptr)
                throw std::runtime_error("Shared Neighborhood derived commitment is missing opening material: " + derived_label);
            std::vector<FieldElement> expected_mask;
            for (const auto &[source_label, scalar] : terms)
            {
                const auto source_it = context.objects.find(source_label);
                if (source_it == context.objects.end() || source_it->second.opening_polynomial == nullptr)
                    throw std::runtime_error("Shared Neighborhood linear source is missing opening material: " + source_label);
                const auto &source_poly = *source_it->second.opening_polynomial;
                const auto &derived_poly = *derived_it->second.opening_polynomial;
                if (source_poly.basis != algebra::PolynomialBasis::Evaluation || derived_poly.basis != source_poly.basis ||
                    source_poly.domain == nullptr || derived_poly.domain == nullptr || source_poly.domain->name != derived_poly.domain->name ||
                    source_poly.domain->size != derived_poly.domain->size)
                    throw std::runtime_error("Shared Neighborhood linear commitment mixed incompatible source polynomial domains");
                if (expected_mask.size() < source_poly.zero_domain_blinder.size())
                    expected_mask.resize(source_poly.zero_domain_blinder.size(), FieldElement::zero());
                for (std::size_t i = 0; i < source_poly.zero_domain_blinder.size(); ++i)
                    expected_mask[i] += scalar * source_poly.zero_domain_blinder[i];
            }
            if (expected_mask.size() != 8 || derived_it->second.opening_polynomial->zero_domain_blinder != expected_mask)
                throw std::runtime_error("Shared Neighborhood derived commitment did not preserve its complete rank-8 mask lineage: " + derived_label);
        }

        using Coefficients = std::vector<FieldElement>;

        Coefficients materialize(const Polynomial &polynomial)
        {
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
                return polynomial.values();
            if (polynomial.domain == nullptr)
                throw std::runtime_error("Shared Neighborhood polynomial has no evaluation domain");
            const auto degree = polynomial.domain->size + polynomial.zero_domain_blinder.size() - 1;
            return algebra::materialize_blinded_coefficients(polynomial, degree);
        }

        Coefficients add_poly(const Coefficients &lhs, const Coefficients &rhs)
        {
            Coefficients out(std::max(lhs.size(), rhs.size()), FieldElement::zero());
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                if (i < lhs.size())
                    out[i] += lhs[i];
                if (i < rhs.size())
                    out[i] += rhs[i];
            }
            return out;
        }

        Coefficients sub_poly(const Coefficients &lhs, const Coefficients &rhs)
        {
            Coefficients out(std::max(lhs.size(), rhs.size()), FieldElement::zero());
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                if (i < lhs.size())
                    out[i] += lhs[i];
                if (i < rhs.size())
                    out[i] -= rhs[i];
            }
            return out;
        }

        Coefficients mul_poly(const Coefficients &lhs, const Coefficients &rhs)
        {
            if (lhs.empty() || rhs.empty())
                return {};
            if (lhs.size() > std::numeric_limits<std::size_t>::max() - rhs.size() + 1)
                throw std::runtime_error("Shared Neighborhood quotient convolution size overflow");
            const auto product_size = lhs.size() + rhs.size() - 1;
            std::size_t ntt_size = 1;
            while (ntt_size < product_size)
            {
                if (ntt_size > std::numeric_limits<std::size_t>::max() / 2)
                    throw std::runtime_error("Shared Neighborhood quotient NTT size overflow");
                ntt_size *= 2;
            }
            Coefficients left(ntt_size, FieldElement::zero()), right(ntt_size, FieldElement::zero());
            std::copy(lhs.begin(), lhs.end(), left.begin());
            std::copy(rhs.begin(), rhs.end(), right.begin());
            algebra::ntt_in_place(FieldElement::mutable_native_array_data(left.data()), ntt_size);
            algebra::ntt_in_place(FieldElement::mutable_native_array_data(right.data()), ntt_size);
            for (std::size_t i = 0; i < ntt_size; ++i)
                left[i] *= right[i];
            algebra::intt_in_place(FieldElement::mutable_native_array_data(left.data()), ntt_size);
            left.resize(product_size);
            return left;
        }

        Coefficients rotate_poly(const Coefficients &source, const FieldElement &omega)
        {
            Coefficients out(source.size(), FieldElement::zero());
            auto power = FieldElement::one();
            for (std::size_t i = 0; i < source.size(); ++i)
            {
                out[i] = source[i] * power;
                power *= omega;
            }
            return out;
        }

        Polynomial selector_polynomial(const std::string &label, const std::vector<FieldElement> &evaluations,
                                       const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            return Polynomial::from_evaluations(label, evaluations, domain);
        }

        std::vector<FieldElement> group_start_selector(const PublicStatement &s)
        {
            std::vector<FieldElement> selector(s.edge_domain_size, FieldElement::zero());
            for (std::size_t edge_index = 0; edge_index < s.edges; ++edge_index)
            {
                if (edge_index == 0 || s.topology[edge_index - 1].graph != s.topology[edge_index].graph ||
                    s.topology[edge_index - 1].dst != s.topology[edge_index].dst)
                    selector[edge_index] = FieldElement::one();
            }
            return selector;
        }

        std::vector<FieldElement> group_end_selector(const PublicStatement &s, const std::vector<FieldElement> &starts)
        {
            std::vector<FieldElement> selector(s.edge_domain_size, FieldElement::zero());
            for (std::size_t edge_index = 0; edge_index < s.edges; ++edge_index)
                if (edge_index + 1 == s.edges || !starts[edge_index + 1].is_zero())
                    selector[edge_index] = FieldElement::one();
            return selector;
        }

        std::vector<FieldElement> valid_edge_selector(const PublicStatement &s)
        {
            std::vector<FieldElement> selector(s.edge_domain_size, FieldElement::zero());
            std::fill(selector.begin(), selector.begin() + static_cast<std::ptrdiff_t>(s.edges), FieldElement::one());
            return selector;
        }

        std::vector<FieldElement> edge_target_values(const PublicStatement &s, const std::vector<FieldElement> &node_targets)
        {
            std::vector<FieldElement> values(s.edges, FieldElement::zero());
            for (std::size_t edge_index = 0; edge_index < s.edges; ++edge_index)
                values[edge_index] = node_targets.at(static_cast<std::size_t>(s.topology[edge_index].dst));
            return values;
        }

        struct Psqlabels
        {
            std::string state;
            std::string quotient_step;
            std::string quotient_terminal;
            std::string target_edge;
        };

        Psqlabels psq_labels(const std::string &channel)
        {
            const auto prefix = psq_prefix(channel);
            return Psqlabels{prefix + "/state", prefix + "/Q_step", prefix + "/Q_group_terminal", target_edge_label(channel)};
        }

        void build_and_commit_psq(ProofBuildContext &context, const std::string &channel, const std::string &weight_label,
                                  const std::vector<FieldElement> &weight_values, const std::vector<FieldElement> &node_targets,
                                  const std::vector<FieldElement> &starts, const std::vector<FieldElement> &ends)
        {
            const auto &s = context.statement;
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto labels = psq_labels(channel);
            const auto full_weight = pad_to_domain(weight_values, s.edge_domain_size);
            const auto state_values = protocol::build_group_prefix_state(full_weight, starts);
            const auto target_edge = edge_target_values(s, node_targets);
            commit_masked_evaluations(context, labels.target_edge, pad_to_domain(target_edge, s.edge_domain_size), edge_domain, 8);
            commit_masked_evaluations(context, labels.state, state_values, edge_domain, 8);

            const auto valid = valid_edge_selector(s);
            const auto valid_poly = materialize(selector_polynomial("sn/selector/valid", valid, edge_domain));
            const auto start_poly = materialize(selector_polynomial("sn/selector/start", starts, edge_domain));
            const auto end_poly = materialize(selector_polynomial("sn/selector/end", ends, edge_domain));
            const auto state_poly = materialize(*context.objects.at(labels.state).opening_polynomial);
            const auto weight_poly = materialize(*context.objects.at(weight_label).opening_polynomial);
            const auto target_poly = materialize(*context.objects.at(labels.target_edge).opening_polynomial);
            const auto omega = edge_domain->omega;
            const auto state_next = rotate_poly(state_poly, omega);
            const auto valid_next = rotate_poly(valid_poly, omega);
            const auto start_next = rotate_poly(start_poly, omega);
            const auto weight_next = rotate_poly(weight_poly, omega);
            const auto one_minus_valid = sub_poly({FieldElement::one()}, valid_next);
            const auto one_minus_start = sub_poly({FieldElement::one()}, start_next);
            const auto grouped_weight = add_poly(mul_poly(start_next, weight_next),
                                                 mul_poly(one_minus_start, add_poly(state_poly, weight_next)));
            const auto recurrence_rhs = add_poly(mul_poly(one_minus_valid, state_poly), mul_poly(valid_next, grouped_weight));
            const auto recurrence_residual = sub_poly(state_next, recurrence_rhs);
            auto q_step = quotient_by_domain_vanishing(Polynomial::from_coefficients(labels.quotient_step, recurrence_residual),
                                                       s.edge_domain_size, labels.quotient_step);

            const auto terminal_residual = mul_poly(end_poly, sub_poly(state_poly, target_poly));
            auto q_terminal = quotient_by_domain_vanishing(Polynomial::from_coefficients(labels.quotient_terminal, terminal_residual),
                                                           s.edge_domain_size, labels.quotient_terminal);
            commit_quotient_with_mask(context, labels.quotient_step, std::move(q_step), s.edge_domain_size);
            commit_quotient_with_mask(context, labels.quotient_terminal, std::move(q_terminal), s.edge_domain_size);
        }

        FieldElement evaluate_public_selector(const PublicStatement &s, const std::vector<FieldElement> &selector, const FieldElement &point)
        {
            const auto domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            return selector_polynomial("sn/selector/eval", selector, domain).evaluate(point);
        }

        FieldElement evaluate_opening(const ProofBuildContext &context, const std::string &label, const FieldElement &point)
        {
            const auto it = context.objects.find(label);
            if (it == context.objects.end() || it->second.opening_polynomial == nullptr)
                throw std::runtime_error("Shared Neighborhood opening source is missing: " + label);
            return it->second.opening_polynomial->evaluate(point);
        }

        void prove_psq_openings(ProofBuildContext &context, const std::string &channel, const std::string &weight_label)
        {
            const auto &s = context.statement;
            const auto labels = psq_labels(channel);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto zeta = context.transcript.challenge_excluding("sn/psq/" + channel + "/zeta",
                                                                     [&](const FieldElement &x)
                                                                     { return x.is_zero() || edge_domain->zero_polynomial_eval(x).is_zero(); });
            const auto step_group = psq_prefix(channel) + "/open/step";
            const auto next_group = psq_prefix(channel) + "/open/step_next";
            add_opening_group(context, step_group, {labels.state, weight_label, labels.quotient_step, quotient_mask_label(labels.quotient_step)},
                              {evaluate_opening(context, labels.state, zeta), evaluate_opening(context, weight_label, zeta),
                               evaluate_opening(context, labels.quotient_step, zeta),
                               evaluate_opening(context, quotient_mask_label(labels.quotient_step), zeta)},
                              zeta);
            const auto next = zeta * edge_domain->omega;
            add_opening_group(context, next_group, {labels.state, weight_label},
                              {evaluate_opening(context, labels.state, next), evaluate_opening(context, weight_label, next)}, next);

            const auto terminal_point = context.transcript.challenge_excluding("sn/psq/" + channel + "/terminal_point",
                                                                                [&](const FieldElement &x)
                                                                                { return x.is_zero() || edge_domain->zero_polynomial_eval(x).is_zero(); });
            const auto terminal_group = psq_prefix(channel) + "/open/group_terminal";
            add_opening_group(context, terminal_group,
                              {labels.state, labels.target_edge, labels.quotient_terminal, quotient_mask_label(labels.quotient_terminal)},
                              {evaluate_opening(context, labels.state, terminal_point), evaluate_opening(context, labels.target_edge, terminal_point),
                               evaluate_opening(context, labels.quotient_terminal, terminal_point),
                               evaluate_opening(context, quotient_mask_label(labels.quotient_terminal), terminal_point)},
                              terminal_point);
        }

        bool verify_psq_openings(ProofVerifyContext &context, const std::string &channel, const std::string &weight_label)
        {
            const auto &s = context.statement;
            const auto labels = psq_labels(channel);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto starts = group_start_selector(s);
            const auto ends = group_end_selector(s, starts);
            const auto valid = valid_edge_selector(s);
            const auto zeta = context.transcript.challenge_excluding("sn/psq/" + channel + "/zeta",
                                                                     [&](const FieldElement &x)
                                                                     { return x.is_zero() || edge_domain->zero_polynomial_eval(x).is_zero(); });

            const auto step_group_label = psq_prefix(channel) + "/open/step";
            const auto next_group_label = psq_prefix(channel) + "/open/step_next";
            const auto &step_group = required_opening_group(context.proof, step_group_label);
            const auto &next_group = required_opening_group(context.proof, next_group_label);
            if (!verify_opening_group(context, step_group_label,
                                      {labels.state, weight_label, labels.quotient_step, quotient_mask_label(labels.quotient_step)}, zeta))
                return false;
            const auto next = zeta * edge_domain->omega;
            if (!verify_opening_group(context, next_group_label, {labels.state, weight_label}, next))
                return false;

            const auto state = required_opening_value(step_group, labels.state);
            const auto weight = required_opening_value(step_group, weight_label);
            const auto quotient = required_opening_value(step_group, labels.quotient_step);
            const auto quotient_mask = required_opening_value(step_group, quotient_mask_label(labels.quotient_step));
            const auto state_next = required_opening_value(next_group, labels.state);
            const auto weight_next = required_opening_value(next_group, weight_label);
            const auto q_valid_next = evaluate_public_selector(s, valid, next);
            const auto q_new_next = evaluate_public_selector(s, starts, next);
            const auto grouped_next = q_new_next * weight_next + (FieldElement::one() - q_new_next) * (state + weight_next);
            const auto expected_next = (FieldElement::one() - q_valid_next) * state + q_valid_next * grouped_next;
            const auto step_vanishing = edge_domain->zero_polynomial_eval(zeta);
            if (step_vanishing * (quotient - step_vanishing * quotient_mask) != state_next - expected_next)
                return false;

            const auto terminal_point = context.transcript.challenge_excluding("sn/psq/" + channel + "/terminal_point",
                                                                                [&](const FieldElement &x)
                                                                                { return x.is_zero() || edge_domain->zero_polynomial_eval(x).is_zero(); });
            const auto terminal_label = psq_prefix(channel) + "/open/group_terminal";
            const auto &terminal_group = required_opening_group(context.proof, terminal_label);
            if (!verify_opening_group(context, terminal_label,
                                      {labels.state, labels.target_edge, labels.quotient_terminal,
                                       quotient_mask_label(labels.quotient_terminal)}, terminal_point))
                return false;
            const auto terminal_state = required_opening_value(terminal_group, labels.state);
            const auto target = required_opening_value(terminal_group, labels.target_edge);
            const auto terminal_quotient = required_opening_value(terminal_group, labels.quotient_terminal);
            const auto terminal_mask = required_opening_value(terminal_group, quotient_mask_label(labels.quotient_terminal));
            const auto q_end = evaluate_public_selector(s, ends, terminal_point);
            const auto terminal_vanishing = edge_domain->zero_polynomial_eval(terminal_point);
            return terminal_vanishing * (terminal_quotient - terminal_vanishing * terminal_mask) ==
                   q_end * (terminal_state - target);
        }

        std::vector<FieldElement> node_multiplicities(const PublicStatement &s)
        {
            std::vector<FieldElement> multiplicity(s.node_domain_size, FieldElement::zero());
            for (const auto &edge : s.topology)
                multiplicity.at(static_cast<std::size_t>(edge.dst)) += FieldElement::one();
            return multiplicity;
        }

        std::vector<FieldElement> make_route_table_values(const PublicStatement &s, const std::vector<FieldElement> &targets,
                                                          const FieldElement &eta)
        {
            if (targets.size() != s.nodes)
                throw std::runtime_error("Shared Neighborhood node target shape mismatch");
            std::vector<FieldElement> table(s.node_domain_size, FieldElement::zero());
            for (std::size_t node = 0; node < s.nodes; ++node)
                table[node] = FieldElement(static_cast<std::uint64_t>(node)) + eta * targets[node];
            return table;
        }

        std::vector<FieldElement> make_route_query_values(const PublicStatement &s, const std::vector<FieldElement> &edge_targets,
                                                          const FieldElement &eta)
        {
            if (edge_targets.size() != s.edges)
                throw std::runtime_error("Shared Neighborhood edge target shape mismatch");
            std::vector<FieldElement> query(s.edge_domain_size, FieldElement::zero());
            for (std::size_t edge = 0; edge < s.edges; ++edge)
                query[edge] = FieldElement(s.topology[edge].dst) + eta * edge_targets[edge];
            return query;
        }

        FieldElement evaluate_public_node_id(const PublicStatement &s, const FieldElement &point)
        {
            const auto domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            std::vector<FieldElement> ids(s.node_domain_size, FieldElement::zero());
            for (std::size_t i = 0; i < s.nodes; ++i)
                ids[i] = FieldElement(static_cast<std::uint64_t>(i));
            return Polynomial::from_evaluations("sn/public/node_id", std::move(ids), domain).evaluate(point);
        }

        FieldElement evaluate_public_edge_dst(const PublicStatement &s, const FieldElement &point)
        {
            const auto domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            std::vector<FieldElement> dst(s.edge_domain_size, FieldElement::zero());
            for (std::size_t i = 0; i < s.edges; ++i)
                dst[i] = FieldElement(s.topology[i].dst);
            return Polynomial::from_evaluations("sn/public/edge_dst", std::move(dst), domain).evaluate(point);
        }

        void absorb_recorded_commitment(ProofVerifyContext &context, const std::string &label)
        {
            const auto &record = required_commitment(context.proof, label);
            context.transcript.absorb_commitment(label, record.point);
            context.objects.emplace(label, Commitment{label, record.point, nullptr});
        }

        bool check_route_tuple(const ProofVerifyContext &context, const std::string &stem, const std::string &node_target_label,
                               const std::string &edge_target_label, const FieldElement &eta, const FieldElement &zeta)
        {
            const auto group = required_opening_group(context.proof, stem + "/zeta");
            const auto table = required_opening_value(group, stem + "/Table");
            const auto query = required_opening_value(group, stem + "/Query");
            const auto node_target = required_opening_value(group, node_target_label);
            const auto edge_target = required_opening_value(group, edge_target_label);
            return table == evaluate_public_node_id(context.statement, zeta) + eta * node_target &&
                   query == evaluate_public_edge_dst(context.statement, zeta) + eta * edge_target;
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

        struct ChannelWitness
        {
            std::string channel;
            std::string weight_label;
            std::vector<FieldElement> weights;
            std::string target_node_label;
            std::vector<FieldElement> targets;
        };

        std::vector<ChannelWitness> make_channel_witnesses(const PublicStatement &s, const RawWitness &raw,
                                                           const std::vector<FieldElement> &folded_edges,
                                                           const std::vector<FieldElement> &folded_targets,
                                                           const std::vector<FieldElement> &shared_weights,
                                                           const std::vector<FieldElement> &shared_targets)
        {
            if (s.variant == Variant::Original)
                return {ChannelWitness{"shared", kW, shared_weights, kT, shared_targets}};
            return {ChannelWitness{"den", kU, raw.u, kSum, raw.sum}, ChannelWitness{"agg", kVStar, folded_edges, kHaggStar, folded_targets}};
        }

        void commit_channel_psq_and_route(ProofBuildContext &context, const ChannelWitness &channel,
                                          const std::vector<FieldElement> &starts, const std::vector<FieldElement> &ends,
                                          const std::vector<FieldElement> &multiplicity)
        {
            const auto &s = context.statement;
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            build_and_commit_psq(context, channel.channel, channel.weight_label, channel.weights, channel.targets, starts, ends);

            const auto eta = context.transcript.challenge_excluding(route_prefix(channel.channel) + "/eta",
                                                                    [](const FieldElement &value) { return value.is_zero(); });
            const auto target_edges = edge_target_values(s, channel.targets);
            const auto table = make_route_table_values(s, channel.targets, eta);
            const auto query = make_route_query_values(s, target_edges, eta);
            const auto table_label = route_prefix(channel.channel) + "/Table";
            const auto query_label = route_prefix(channel.channel) + "/Query";
            // Preserve the target's zero-domain mask under tuple compression. Independent masks would
            // break Table = node_id + eta*target and Query = dst + eta*target_edge off-domain.
            auto table_polynomial = Polynomial::from_evaluations(table_label, table, node_domain);
            const auto &node_source = context.objects.at(channel.target_node_label).opening_polynomial;
            if (node_source == nullptr || node_source->zero_domain_blinder.size() != 8)
                throw std::runtime_error("Shared Neighborhood node target lacks its rank-8 privacy mask");
            const auto &node_mask = node_source->zero_domain_blinder;
            if (!node_mask.empty())
            {
                auto tuple_mask = node_mask;
                for (auto &coefficient : tuple_mask)
                    coefficient *= eta;
                table_polynomial.set_zero_domain_blinder(std::move(tuple_mask));
            }
            commit_polynomial(context, table_label, std::move(table_polynomial));

            auto query_polynomial = Polynomial::from_evaluations(query_label, query, edge_domain);
            const auto &edge_source = context.objects.at(target_edge_label(channel.channel)).opening_polynomial;
            if (edge_source == nullptr || edge_source->zero_domain_blinder.size() != 8)
                throw std::runtime_error("Shared Neighborhood edge target lacks its rank-8 privacy mask");
            const auto &edge_mask = edge_source->zero_domain_blinder;
            if (!edge_mask.empty())
            {
                auto tuple_mask = edge_mask;
                for (auto &coefficient : tuple_mask)
                    coefficient *= eta;
                query_polynomial.set_zero_domain_blinder(std::move(tuple_mask));
            }
            commit_polynomial(context, query_label, std::move(query_polynomial));
            prove_route_argument(context, route_prefix(channel.channel), table, query, multiplicity, s.nodes, s.edges,
                                 node_domain, edge_domain, {channel.target_node_label, target_edge_label(channel.channel)});
            prove_shared_neighborhood_psq_openings(context, channel.channel, channel.weight_label);
        }

        bool verify_channel_psq_and_route(ProofVerifyContext &context, const ChannelWitness &channel,
                                         const std::vector<FieldElement> &multiplicity)
        {
            const auto &s = context.statement;
            const auto node_domain = algebra::RootOfUnityDomain::create(s.node_domain, s.node_domain_size);
            const auto edge_domain = algebra::RootOfUnityDomain::create(s.edge_domain, s.edge_domain_size);
            const auto psq = psq_labels(channel.channel);
            absorb_recorded_commitment(context, psq.target_edge);
            absorb_recorded_commitment(context, psq.state);
            absorb_recorded_commitment(context, psq.quotient_step);
            absorb_recorded_commitment(context, quotient_mask_label(psq.quotient_step));
            absorb_recorded_commitment(context, psq.quotient_terminal);
            absorb_recorded_commitment(context, quotient_mask_label(psq.quotient_terminal));

            const auto eta = context.transcript.challenge_excluding(route_prefix(channel.channel) + "/eta",
                                                                    [](const FieldElement &value) { return value.is_zero(); });
            absorb_recorded_commitment(context, route_prefix(channel.channel) + "/Table");
            absorb_recorded_commitment(context, route_prefix(channel.channel) + "/Query");
            FieldElement zeta = FieldElement::zero();
            if (!verify_route_argument(context, route_prefix(channel.channel), multiplicity, s.nodes, s.edges, node_domain, edge_domain,
                                       {channel.target_node_label, target_edge_label(channel.channel)}, &zeta))
                return false;
            if (!check_route_tuple(context, route_prefix(channel.channel), channel.target_node_label, target_edge_label(channel.channel), eta, zeta))
                return false;
            return verify_shared_neighborhood_psq_openings(context, channel.channel, channel.weight_label);
        }
    }

    void prove_shared_neighborhood_psq_openings(ProofBuildContext &context, const std::string &channel,
                                                const std::string &weight_label)
    {
        prove_psq_openings(context, channel, weight_label);
    }

    bool verify_shared_neighborhood_psq_openings(ProofVerifyContext &context, const std::string &channel,
                                                  const std::string &weight_label)
    {
        return verify_psq_openings(context, channel, weight_label);
    }

    std::vector<std::string> shared_neighborhood_commitment_labels(const PublicStatement &s)
    {
        validate_topology(s);
        auto labels = shared_base_commitment_labels(s);
        for (const auto &channel : channels(s))
        {
            const auto psq = psq_prefix(channel);
            labels.push_back(target_edge_label(channel));
            labels.push_back(psq + "/state");
            labels.push_back(psq + "/Q_step");
            labels.push_back(quotient_mask_label(psq + "/Q_step"));
            labels.push_back(psq + "/Q_group_terminal");
            labels.push_back(quotient_mask_label(psq + "/Q_group_terminal"));
            const auto route = route_labels(route_prefix(channel));
            labels.insert(labels.end(), route.begin(), route.end());
        }
        return labels;
    }

    std::vector<std::string> shared_neighborhood_opening_group_labels(const PublicStatement &s)
    {
        validate_topology(s);
        std::vector<std::string> labels;
        for (const auto &channel : channels(s))
        {
            const auto route = route_opening_group_labels(route_prefix(channel));
            labels.insert(labels.end(), route.begin(), route.end());
            labels.push_back(psq_prefix(channel) + "/open/step");
            labels.push_back(psq_prefix(channel) + "/open/step_next");
            labels.push_back(psq_prefix(channel) + "/open/group_terminal");
        }
        return labels;
    }

    void prove_shared_neighborhood(ProofBuildContext &context)
    {
        const auto &s = context.statement;
        validate_topology(s);
        const auto raw = make_raw_witness(context);
        commit_raw_witness(context, raw);

        // Width folding is Fiat-Shamir bound after all lane commitments. Both node targets and edge values
        // use exactly the same challenge so destination sums remain tied to the folded edge column.
        const auto xi = context.transcript.challenge_excluding("sn/width_fold_xi", [](const FieldElement &x) { return x.is_zero(); });
        FieldElement xi_power = FieldElement::one();
        std::vector<std::pair<std::string, FieldElement>> edge_terms;
        std::vector<std::pair<std::string, FieldElement>> node_terms;
        for (std::size_t lane = 0; lane < s.width; ++lane)
        {
            edge_terms.emplace_back(v_label(lane), xi_power);
            node_terms.emplace_back(hagg_label(lane), xi_power);
            xi_power *= xi;
        }
        derive_linear_commitment(context, kVStar, edge_terms);
        derive_linear_commitment(context, kHaggStar, node_terms);
        require_linear_mask_lineage(context, kVStar, edge_terms);
        require_linear_mask_lineage(context, kHaggStar, node_terms);

        std::vector<FieldElement> folded_edges(s.edges, FieldElement::zero());
        std::vector<FieldElement> folded_targets(s.nodes, FieldElement::zero());
        xi_power = FieldElement::one();
        for (std::size_t lane = 0; lane < s.width; ++lane)
        {
            for (std::size_t edge = 0; edge < s.edges; ++edge)
                folded_edges[edge] += xi_power * raw.v[lane][edge];
            for (std::size_t node = 0; node < s.nodes; ++node)
                folded_targets[node] += xi_power * raw.hagg[lane][node];
            xi_power *= xi;
        }

        std::vector<FieldElement> shared_weights;
        std::vector<FieldElement> shared_targets;

        if (s.variant == Variant::Original)
        {
            const auto lambda = context.transcript.challenge_excluding("sn/shared_lambda", [](const FieldElement &x) { return x.is_zero(); });
            const std::vector<std::pair<std::string, FieldElement>> weight_terms{{kU, FieldElement::one()}, {kVStar, lambda}};
            const std::vector<std::pair<std::string, FieldElement>> target_terms{{kSum, FieldElement::one()}, {kHaggStar, lambda}};
            derive_linear_commitment(context, kW, weight_terms);
            derive_linear_commitment(context, kT, target_terms);
            require_linear_mask_lineage(context, kW, weight_terms);
            require_linear_mask_lineage(context, kT, target_terms);
            shared_weights.resize(s.edges, FieldElement::zero());
            shared_targets.resize(s.nodes, FieldElement::zero());
            for (std::size_t edge = 0; edge < s.edges; ++edge)
                shared_weights[edge] = raw.u[edge] + lambda * folded_edges[edge];
            for (std::size_t node = 0; node < s.nodes; ++node)
                shared_targets[node] = raw.sum[node] + lambda * folded_targets[node];
        }

        const auto starts = group_start_selector(s);
        const auto ends = group_end_selector(s, starts);
        const auto multiplicity = node_multiplicities(s);
        const auto route_channels = make_channel_witnesses(s, raw, folded_edges, folded_targets, shared_weights, shared_targets);
        for (const auto &channel : route_channels)
            commit_channel_psq_and_route(context, channel, starts, ends, multiplicity);
    }

    bool verify_shared_neighborhood(ProofVerifyContext &context)
    {
        const auto &s = context.statement;
        validate_topology(s);
        for (const auto &label : source_commitment_labels(s))
            absorb_recorded_commitment(context, label);

        const auto xi = context.transcript.challenge_excluding("sn/width_fold_xi", [](const FieldElement &x) { return x.is_zero(); });
        FieldElement xi_power = FieldElement::one();
        std::vector<std::pair<std::string, FieldElement>> edge_terms;
        std::vector<std::pair<std::string, FieldElement>> node_terms;
        for (std::size_t lane = 0; lane < s.width; ++lane)
        {
            edge_terms.emplace_back(v_label(lane), xi_power);
            node_terms.emplace_back(hagg_label(lane), xi_power);
            xi_power *= xi;
        }
        if (!verify_linear_commitment_point(context.proof, kVStar, edge_terms) ||
            !verify_linear_commitment_point(context.proof, kHaggStar, node_terms))
            return false;
        absorb_recorded_commitment(context, kVStar);
        absorb_recorded_commitment(context, kHaggStar);

        std::vector<ChannelWitness> route_channels;
        if (s.variant == Variant::Original)
        {
            const auto lambda = context.transcript.challenge_excluding("sn/shared_lambda", [](const FieldElement &x) { return x.is_zero(); });
            if (!verify_linear_commitment_point(context.proof, kW, {{kU, FieldElement::one()}, {kVStar, lambda}}) ||
                !verify_linear_commitment_point(context.proof, kT, {{kSum, FieldElement::one()}, {kHaggStar, lambda}}))
                return false;
            absorb_recorded_commitment(context, kW);
            absorb_recorded_commitment(context, kT);
            route_channels.push_back(ChannelWitness{"shared", kW, {}, kT, {}});
        }
        else
        {
            route_channels.push_back(ChannelWitness{"den", kU, {}, kSum, {}});
            route_channels.push_back(ChannelWitness{"agg", kVStar, {}, kHaggStar, {}});
        }

        const auto multiplicity = node_multiplicities(s);
        for (const auto &channel : route_channels)
            if (!verify_channel_psq_and_route(context, channel, multiplicity))
                return false;
        return true;
    }
}
