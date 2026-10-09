#include "gatzk/mechanism/internal.hpp"

#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace gatzk::mechanism::internal
{
    namespace
    {
        using Clock = std::chrono::steady_clock;
        using Coefficients = std::vector<FieldElement>;

        void require(bool condition, const std::string &message)
        {
            if (!condition)
                throw std::runtime_error(message);
        }

        std::size_t checked_add(std::size_t lhs, std::size_t rhs)
        {
            if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
                throw std::runtime_error("mechanism route polynomial length overflow");
            return lhs + rhs;
        }

        std::size_t next_power_of_two(std::size_t value)
        {
            std::size_t out = 1;
            while (out < value)
            {
                if (out > std::numeric_limits<std::size_t>::max() / 2)
                    throw std::runtime_error("mechanism route convolution size overflow");
                out *= 2;
            }
            return out;
        }

        Coefficients trim(Coefficients values)
        {
            while (!values.empty() && values.back().is_zero())
                values.pop_back();
            return values;
        }

        Coefficients add(const Coefficients &lhs, const Coefficients &rhs)
        {
            Coefficients out(std::max(lhs.size(), rhs.size()), FieldElement::zero());
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] += lhs[i];
            for (std::size_t i = 0; i < rhs.size(); ++i)
                out[i] += rhs[i];
            return trim(std::move(out));
        }

        Coefficients subtract(const Coefficients &lhs, const Coefficients &rhs)
        {
            Coefficients out(std::max(lhs.size(), rhs.size()), FieldElement::zero());
            for (std::size_t i = 0; i < lhs.size(); ++i)
                out[i] += lhs[i];
            for (std::size_t i = 0; i < rhs.size(); ++i)
                out[i] -= rhs[i];
            return trim(std::move(out));
        }

        Coefficients multiply(const Coefficients &lhs, const Coefficients &rhs)
        {
            if (lhs.empty() || rhs.empty())
                return {};
            const auto full_length = checked_add(lhs.size(), rhs.size()) - 1;
            const auto fft_length = next_power_of_two(full_length);
            Coefficients left(fft_length, FieldElement::zero());
            Coefficients right(fft_length, FieldElement::zero());
            std::copy(lhs.begin(), lhs.end(), left.begin());
            std::copy(rhs.begin(), rhs.end(), right.begin());
            algebra::ntt_in_place(FieldElement::mutable_native_array_data(left.data()), fft_length);
            algebra::ntt_in_place(FieldElement::mutable_native_array_data(right.data()), fft_length);
            for (std::size_t i = 0; i < fft_length; ++i)
                left[i] *= right[i];
            algebra::intt_in_place(FieldElement::mutable_native_array_data(left.data()), fft_length);
            left.resize(full_length);
            return trim(std::move(left));
        }

        Coefficients materialize(const Polynomial &polynomial)
        {
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                if (!polynomial.zero_domain_blinder.empty())
                    throw std::runtime_error("coefficient route polynomial has an unsupported separate blinder");
                return trim(polynomial.values());
            }
            if (polynomial.domain == nullptr)
                throw std::runtime_error("route evaluation polynomial has no domain");
            const auto degree = polynomial.zero_domain_blinder.empty()
                                    ? polynomial.domain->size - 1
                                    : polynomial.domain->size + polynomial.zero_domain_blinder.size() - 1;
            return trim(algebra::materialize_blinded_coefficients(polynomial, degree));
        }

        Coefficients selector_coefficients(const std::shared_ptr<algebra::RootOfUnityDomain> &domain, std::size_t active_count,
                                           bool last_row)
        {
            if (domain == nullptr || active_count > domain->size)
                throw std::runtime_error("route selector shape is invalid");
            std::vector<FieldElement> values(domain->size, FieldElement::zero());
            if (last_row)
                values.back() = FieldElement::one();
            else
                std::fill(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(active_count), FieldElement::one());
            return materialize(Polynomial::from_evaluations("mechanism_route_selector", std::move(values), domain));
        }

        Coefficients rotate_coefficients(const Coefficients &values, const FieldElement &omega)
        {
            Coefficients out(values.size(), FieldElement::zero());
            auto power = FieldElement::one();
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = values[i] * power;
                power *= omega;
            }
            return trim(std::move(out));
        }

        Polynomial divide_by_domain_vanishing(const Coefficients &residual, std::size_t domain_size, const std::string &label)
        {
            if (domain_size == 0)
                throw std::runtime_error("route quotient has an empty evaluation domain: " + label);
            Coefficients remainder = residual;
            if (remainder.size() <= domain_size)
            {
                if (std::any_of(remainder.begin(), remainder.end(), [](const auto &value) { return !value.is_zero(); }))
                    throw std::runtime_error("route identity residual is not divisible by its domain vanishing polynomial: " + label);
                return Polynomial::from_coefficients(label, {});
            }
            Coefficients quotient(remainder.size() - domain_size, FieldElement::zero());
            for (std::size_t k = remainder.size(); k-- > domain_size;)
            {
                const auto coefficient = remainder[k];
                quotient[k - domain_size] = coefficient;
                remainder[k] -= coefficient;
                remainder[k - domain_size] += coefficient;
            }
            if (std::any_of(remainder.begin(), remainder.begin() + static_cast<std::ptrdiff_t>(domain_size),
                            [](const auto &value) { return !value.is_zero(); }))
                throw std::runtime_error("route identity residual is not exactly divisible by its domain vanishing polynomial: " + label);
            return Polynomial::from_coefficients(label, trim(std::move(quotient)));
        }

        Polynomial masked_poly(const std::string &label, const std::vector<FieldElement> &values,
                               const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                               const std::vector<FieldElement> &mask)
        {
            auto polynomial = Polynomial::from_evaluations(label, values, domain);
            if (!mask.empty())
                polynomial.set_zero_domain_blinder(mask);
            return polynomial;
        }

        FieldElement polynomial_value(const ProofBuildContext &context, const std::string &label, const FieldElement &point)
        {
            const auto it = context.objects.find(label);
            if (it == context.objects.end() || it->second.opening_polynomial == nullptr)
                throw std::runtime_error("mechanism route references unavailable witness polynomial: " + label);
            return it->second.opening_polynomial->evaluate(point);
        }

        FieldElement selector_value(const std::shared_ptr<algebra::RootOfUnityDomain> &domain, std::size_t active_count,
                                    const FieldElement &point, bool last_row)
        {
            std::vector<FieldElement> values(domain->size, FieldElement::zero());
            if (last_row)
                values.back() = FieldElement::one();
            else
                std::fill(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(active_count), FieldElement::one());
            return Polynomial::from_evaluations("mechanism_route_selector", std::move(values), domain).evaluate(point);
        }

        FieldElement public_column_value(const std::vector<FieldElement> &values,
                                         const std::shared_ptr<algebra::RootOfUnityDomain> &domain, const FieldElement &point)
        {
            if (values.size() != domain->size)
                throw std::runtime_error("mechanism route public column does not cover its physical domain");
            return Polynomial::from_evaluations("mechanism_route_public", values, domain).evaluate(point);
        }

        std::vector<std::string> current_group_labels(const std::string &stem, const std::vector<std::string> &extra)
        {
            std::vector<std::string> labels{stem + "/Table", stem + "/Query", stem + "/J_node", stem + "/J_edge", stem + "/R_node",
                                            stem + "/R_edge", stem + "/Q_inv_node", quotient_mask_label(stem + "/Q_inv_node"),
                                            stem + "/Q_inv_edge", quotient_mask_label(stem + "/Q_inv_edge"), stem + "/Q_acc_node",
                                            quotient_mask_label(stem + "/Q_acc_node"), stem + "/Q_acc_edge",
                                            quotient_mask_label(stem + "/Q_acc_edge")};
            for (const auto &label : extra)
            {
                if (std::find(labels.begin(), labels.end(), label) == labels.end())
                    labels.push_back(label);
            }
            return labels;
        }

        std::vector<FieldElement> build_values(const ProofBuildContext &context, const std::vector<std::string> &labels,
                                              const FieldElement &point)
        {
            std::vector<FieldElement> values;
            values.reserve(labels.size());
            for (const auto &label : labels)
                values.push_back(polynomial_value(context, label, point));
            return values;
        }

    }

    RouteWitness build_route_witness(const std::vector<FieldElement> &table, const std::vector<FieldElement> &query,
                                     const std::vector<FieldElement> &multiplicity, std::size_t real_node_count,
                                     std::size_t real_edge_count,
                                     const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                                     const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
                                     const FieldElement &beta)
    {
        require(node_domain != nullptr && edge_domain != nullptr, "mechanism route has a missing domain");
        require(table.size() == node_domain->size && multiplicity.size() == node_domain->size &&
                    query.size() == edge_domain->size && real_node_count < node_domain->size && real_edge_count < edge_domain->size,
                "mechanism route witness/domain shape mismatch");
        require(std::all_of(table.begin() + static_cast<std::ptrdiff_t>(real_node_count), table.end(),
                            [](const auto &value) { return value.is_zero(); }) &&
                    std::all_of(query.begin() + static_cast<std::ptrdiff_t>(real_edge_count), query.end(),
                                [](const auto &value) { return value.is_zero(); }) &&
                    std::all_of(multiplicity.begin() + static_cast<std::ptrdiff_t>(real_node_count), multiplicity.end(),
                                [](const auto &value) { return value.is_zero(); }),
                "mechanism route padding rows must be zero");

        RouteWitness out;
        out.table_inverse.assign(node_domain->size, FieldElement::zero());
        out.query_inverse.assign(edge_domain->size, FieldElement::zero());
        out.node_accumulator.assign(node_domain->size, FieldElement::zero());
        out.edge_accumulator.assign(edge_domain->size, FieldElement::zero());
        for (std::size_t i = 0; i < real_node_count; ++i)
        {
            const auto denominator = table[i] + beta;
            require(!denominator.is_zero(), "mechanism route beta collided with an active table tuple");
            out.table_inverse[i] = denominator.inv();
            out.node_accumulator[i + 1] = out.node_accumulator[i] + multiplicity[i] * out.table_inverse[i];
        }
        for (std::size_t i = real_node_count + 1; i < node_domain->size; ++i)
            out.node_accumulator[i] = out.node_accumulator[real_node_count];
        for (std::size_t i = 0; i < real_edge_count; ++i)
        {
            const auto denominator = query[i] + beta;
            require(!denominator.is_zero(), "mechanism route beta collided with an active query tuple");
            out.query_inverse[i] = denominator.inv();
            out.edge_accumulator[i + 1] = out.edge_accumulator[i] + out.query_inverse[i];
        }
        for (std::size_t i = real_edge_count + 1; i < edge_domain->size; ++i)
            out.edge_accumulator[i] = out.edge_accumulator[real_edge_count];
        return out;
    }

    RouteQuotients build_route_quotient_polynomials(const Polynomial &table, const Polynomial &query,
                                                     const Polynomial &multiplicity, const RouteWitness &witness,
                                                     std::size_t real_node_count, std::size_t real_edge_count,
                                                     const FieldElement &beta,
                                                     const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                                                     const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain)
    {
        require(node_domain != nullptr && edge_domain != nullptr && real_node_count < node_domain->size && real_edge_count < edge_domain->size,
                "mechanism route quotient has an invalid physical domain or active count");
        require(table.basis == algebra::PolynomialBasis::Evaluation && table.domain != nullptr && table.domain->size == node_domain->size &&
                    query.basis == algebra::PolynomialBasis::Evaluation && query.domain != nullptr && query.domain->size == edge_domain->size &&
                    multiplicity.basis == algebra::PolynomialBasis::Evaluation && multiplicity.domain != nullptr &&
                    multiplicity.domain->size == node_domain->size,
                "mechanism route quotient source polynomials have incompatible domains");
        require(witness.table_inverse.size() == node_domain->size && witness.query_inverse.size() == edge_domain->size &&
                    witness.node_accumulator.size() == node_domain->size && witness.edge_accumulator.size() == edge_domain->size,
                "mechanism route quotient witness has an invalid shape");

        auto table_coefficients = materialize(table);
        auto query_coefficients = materialize(query);
        auto multiplicity_coefficients = materialize(multiplicity);
        auto node_inverse = materialize(masked_poly("route_inverse_node", witness.table_inverse, node_domain,
                                                    witness.node_inverse_blinder));
        auto edge_inverse = materialize(masked_poly("route_inverse_edge", witness.query_inverse, edge_domain,
                                                    witness.edge_inverse_blinder));
        auto node_acc = materialize(masked_poly("route_acc_node", witness.node_accumulator, node_domain,
                                                witness.node_accumulator_blinder));
        auto edge_acc = materialize(masked_poly("route_acc_edge", witness.edge_accumulator, edge_domain,
                                                witness.edge_accumulator_blinder));

        const auto node_selector = selector_coefficients(node_domain, real_node_count, false);
        const auto node_last = selector_coefficients(node_domain, 0, true);
        const auto edge_selector = selector_coefficients(edge_domain, real_edge_count, false);
        const auto edge_last = selector_coefficients(edge_domain, 0, true);

        const auto build_inverse_residual = [&](const Coefficients &values, const Coefficients &inverse,
                                                const Coefficients &selector)
        {
            const auto relation = subtract(multiply(add(values, Coefficients{beta}), inverse), Coefficients{FieldElement::one()});
            return multiply(selector, relation);
        };
        const auto node_inv_residual = build_inverse_residual(table_coefficients, node_inverse, node_selector);
        const auto edge_inv_residual = build_inverse_residual(query_coefficients, edge_inverse, edge_selector);

        auto node_acc_residual = subtract(rotate_coefficients(node_acc, node_domain->omega), node_acc);
        node_acc_residual = subtract(node_acc_residual, multiply(multiply(node_selector, multiplicity_coefficients), node_inverse));
        node_acc_residual = add(node_acc_residual, multiply(node_last, node_acc));
        auto edge_weights = edge_selector;
        auto edge_acc_residual = subtract(rotate_coefficients(edge_acc, edge_domain->omega), edge_acc);
        edge_acc_residual = subtract(edge_acc_residual, multiply(edge_weights, edge_inverse));
        edge_acc_residual = add(edge_acc_residual, multiply(edge_last, edge_acc));

        RouteQuotients out;
        out.node_inverse = divide_by_domain_vanishing(node_inv_residual, node_domain->size, "node inverse");
        out.edge_inverse = divide_by_domain_vanishing(edge_inv_residual, edge_domain->size, "edge inverse");
        out.node_accumulator = divide_by_domain_vanishing(node_acc_residual, node_domain->size, "node accumulator");
        out.edge_accumulator = divide_by_domain_vanishing(edge_acc_residual, edge_domain->size, "edge accumulator");
        return out;
    }

    Polynomial quotient_by_domain_vanishing(const Polynomial &identity_residual, std::size_t domain_size, const std::string &label)
    {
        return divide_by_domain_vanishing(materialize(identity_residual), domain_size, label);
    }

    std::vector<std::string> route_commitment_labels(const std::string &stem)
    {
        return {stem + "/Table", stem + "/Query", stem + "/J_node", stem + "/J_edge", stem + "/R_node", stem + "/R_edge",
                stem + "/R_edge_shifted", stem + "/terminal_virtual", stem + "/Q_inv_node", quotient_mask_label(stem + "/Q_inv_node"),
                stem + "/Q_inv_edge", quotient_mask_label(stem + "/Q_inv_edge"), stem + "/Q_acc_node",
                quotient_mask_label(stem + "/Q_acc_node"), stem + "/Q_acc_edge", quotient_mask_label(stem + "/Q_acc_edge")};
    }

    std::vector<std::string> route_opening_group_labels(const std::string &stem)
    {
        return {stem + "/zeta", stem + "/zeta_next_node", stem + "/zeta_next_edge", stem + "/terminal_shifted",
                stem + "/terminal_edge", stem + "/terminal_fixed"};
    }

    FieldElement prove_route_argument(ProofBuildContext &context, const std::string &stem,
                                      const std::vector<FieldElement> &table_values, const std::vector<FieldElement> &query_values,
                                      const std::vector<FieldElement> &multiplicity_values, std::size_t real_node_count,
                                      std::size_t real_edge_count, const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                                      const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
                                      const std::vector<std::string> &extra_zeta_commitments)
    {
        const auto table_label = stem + "/Table";
        const auto query_label = stem + "/Query";
        require(context.objects.contains(table_label) && context.objects.contains(query_label),
                "mechanism route Table/Query commitments must precede the route argument");
        require(table_values.size() == node_domain->size && query_values.size() == edge_domain->size &&
                    multiplicity_values.size() == node_domain->size,
                "mechanism route prover columns have invalid physical sizes");
        const auto &table_poly = context.objects.at(table_label).opening_polynomial;
        const auto &query_poly = context.objects.at(query_label).opening_polynomial;
        require(table_poly != nullptr && query_poly != nullptr && table_poly->values() == table_values && query_poly->values() == query_values &&
                    table_poly->basis == algebra::PolynomialBasis::Evaluation && query_poly->basis == algebra::PolynomialBasis::Evaluation &&
                    table_poly->domain != nullptr && query_poly->domain != nullptr && table_poly->domain->size == node_domain->size &&
                    query_poly->domain->size == edge_domain->size,
                "mechanism route values are not the committed Table/Query source columns");

        const auto beta = context.transcript.challenge_excluding(stem + "/beta", [](const FieldElement &value) { return value.is_zero(); });
        auto witness = build_route_witness(table_values, query_values, multiplicity_values, real_node_count, real_edge_count,
                                           node_domain, edge_domain, beta);
        // Both terminal points are roots of their authenticated physical-domain
        // vanishing polynomials.  Every zero-domain blinder therefore evaluates
        // to zero there; the semantic prefix accumulators already agree at the
        // endpoint.  Keep independent full-rank masks for privacy and never try
        // to divide by Z_H at a point in H.
        witness.node_inverse_blinder = sample_zero_domain_blinder(context.witness_seed, stem + "/J_node", 8);
        witness.edge_inverse_blinder = sample_zero_domain_blinder(context.witness_seed, stem + "/J_edge", 8);
        witness.node_accumulator_blinder = sample_zero_domain_blinder(context.witness_seed, stem + "/R_node", 8);
        witness.edge_accumulator_blinder = sample_zero_domain_blinder(context.witness_seed, stem + "/R_edge", 8);

        const auto node_inverse_poly = masked_poly(stem + "/J_node", witness.table_inverse, node_domain, witness.node_inverse_blinder);
        const auto edge_inverse_poly = masked_poly(stem + "/J_edge", witness.query_inverse, edge_domain, witness.edge_inverse_blinder);
        const auto node_acc_poly = masked_poly(stem + "/R_node", witness.node_accumulator, node_domain, witness.node_accumulator_blinder);
        const auto edge_acc_poly = masked_poly(stem + "/R_edge", witness.edge_accumulator, edge_domain, witness.edge_accumulator_blinder);
        const auto multiplicity_poly = Polynomial::from_evaluations(stem + "/public_multiplicity", multiplicity_values, node_domain);
        const auto quotient_polys = build_route_quotient_polynomials(
            *context.objects.at(table_label).opening_polynomial, *context.objects.at(query_label).opening_polynomial, multiplicity_poly,
            witness, real_node_count, real_edge_count, beta, node_domain, edge_domain);

        commit_polynomial(context, stem + "/J_node", node_inverse_poly);
        commit_polynomial(context, stem + "/J_edge", edge_inverse_poly);
        commit_polynomial(context, stem + "/R_node", node_acc_poly);
        commit_polynomial(context, stem + "/R_edge", edge_acc_poly);

        const auto node_terminal = node_domain->point_at(real_node_count);
        const auto edge_terminal = edge_domain->point_at(real_edge_count);
        const auto shift_delta = edge_terminal - node_terminal;
        const auto shifted_label = stem + "/R_edge_shifted";
        const auto shift_degree = edge_domain->size + witness.edge_accumulator_blinder.size() - 1;
        auto shifted_polynomial = crypto::KZG::materialize_taylor_shifted_polynomial(
            context.objects.at(stem + "/R_edge"), shifted_label, shift_degree, shift_delta, context.key);
        commit_polynomial(context, shifted_label, std::move(shifted_polynomial));

        const auto virtual_label = stem + "/terminal_virtual";
        const auto virtual_degree = std::max(node_domain->size + witness.node_accumulator_blinder.size() - 1, shift_degree);
        auto virtual_coefficients = algebra::materialize_blinded_coefficients(*context.objects.at(stem + "/R_node").opening_polynomial,
                                                                               virtual_degree);
        const auto &shifted_coefficients = context.objects.at(shifted_label).opening_polynomial->values();
        if (shifted_coefficients.size() > virtual_coefficients.size())
            throw std::runtime_error("mechanism route terminal Taylor shift exceeds the native degree envelope");
        for (std::size_t i = 0; i < shifted_coefficients.size(); ++i)
            virtual_coefficients[i] -= shifted_coefficients[i];
        const auto virtual_start = Clock::now();
        Commitment virtual_commitment{.name = virtual_label,
                                      .point = crypto::g1_add(context.objects.at(stem + "/R_node").point,
                                                              crypto::g1_neg(context.objects.at(shifted_label).point)),
                                      .opening_polynomial = std::make_shared<Polynomial>(
                                          Polynomial::from_coefficients(virtual_label, std::move(virtual_coefficients)))};
        context.commitment_time_ms += std::chrono::duration<double, std::milli>(Clock::now() - virtual_start).count();
        record_commitment(context, virtual_label, std::move(virtual_commitment));

        commit_quotient_with_mask(context, stem + "/Q_inv_node", quotient_polys.node_inverse, node_domain->size);
        commit_quotient_with_mask(context, stem + "/Q_inv_edge", quotient_polys.edge_inverse, edge_domain->size);
        commit_quotient_with_mask(context, stem + "/Q_acc_node", quotient_polys.node_accumulator, node_domain->size);
        commit_quotient_with_mask(context, stem + "/Q_acc_edge", quotient_polys.edge_accumulator, edge_domain->size);

        const auto zeta = context.transcript.challenge_excluding(stem + "/zeta", [&](const FieldElement &point)
                                                                 { return point.is_zero() || node_domain->zero_polynomial_eval(point).is_zero() ||
                                                                          edge_domain->zero_polynomial_eval(point).is_zero(); });
        const auto current_labels = current_group_labels(stem, extra_zeta_commitments);
        add_opening_group(context, stem + "/zeta", current_labels, build_values(context, current_labels, zeta), zeta);
        add_opening_group(context, stem + "/zeta_next_node", {stem + "/R_node"},
                          {polynomial_value(context, stem + "/R_node", zeta * node_domain->omega)}, zeta * node_domain->omega);
        add_opening_group(context, stem + "/zeta_next_edge", {stem + "/R_edge"},
                          {polynomial_value(context, stem + "/R_edge", zeta * edge_domain->omega)}, zeta * edge_domain->omega);

        const auto terminal_y = context.transcript.challenge_excluding(stem + "/terminal_y", [&](const FieldElement &point)
                                                                        { return point.is_zero() ||
                                                                                 edge_domain->zero_polynomial_eval(point + shift_delta).is_zero(); });
        const auto shifted_value = polynomial_value(context, shifted_label, terminal_y);
        add_opening_group(context, stem + "/terminal_shifted", {shifted_label}, {shifted_value}, terminal_y);
        add_opening_group(context, stem + "/terminal_edge", {stem + "/R_edge"}, {shifted_value}, terminal_y + shift_delta);
        add_opening_group(context, stem + "/terminal_fixed", {virtual_label}, {FieldElement::zero()}, node_terminal);
        return zeta;
    }

    bool verify_route_argument(ProofVerifyContext &context, const std::string &stem,
                               const std::vector<FieldElement> &multiplicity_values, std::size_t real_node_count,
                               std::size_t real_edge_count, const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                               const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
                               const std::vector<std::string> &extra_zeta_commitments, FieldElement *zeta_out)
    {
        require(node_domain != nullptr && edge_domain != nullptr && multiplicity_values.size() == node_domain->size &&
                    real_node_count < node_domain->size && real_edge_count < edge_domain->size,
                "mechanism route verifier received an invalid public domain shape");
        const auto beta = context.transcript.challenge_excluding(stem + "/beta", [](const FieldElement &value) { return value.is_zero(); });
        for (const auto &label : {stem + "/J_node", stem + "/J_edge", stem + "/R_node", stem + "/R_edge", stem + "/R_edge_shifted",
                                  stem + "/terminal_virtual", stem + "/Q_inv_node", quotient_mask_label(stem + "/Q_inv_node"),
                                  stem + "/Q_inv_edge", quotient_mask_label(stem + "/Q_inv_edge"), stem + "/Q_acc_node",
                                  quotient_mask_label(stem + "/Q_acc_node"), stem + "/Q_acc_edge",
                                  quotient_mask_label(stem + "/Q_acc_edge")})
            context.transcript.absorb_commitment(label, required_commitment(context.proof, label).point);
        const auto zeta = context.transcript.challenge_excluding(stem + "/zeta", [&](const FieldElement &point)
                                                                 { return point.is_zero() || node_domain->zero_polynomial_eval(point).is_zero() ||
                                                                          edge_domain->zero_polynomial_eval(point).is_zero(); });
        if (zeta_out != nullptr)
            *zeta_out = zeta;
        const auto current_labels = current_group_labels(stem, extra_zeta_commitments);
        if (!verify_opening_group(context, stem + "/zeta", current_labels, zeta))
            return false;
        if (!verify_opening_group(context, stem + "/zeta_next_node", {stem + "/R_node"}, zeta * node_domain->omega) ||
            !verify_opening_group(context, stem + "/zeta_next_edge", {stem + "/R_edge"}, zeta * edge_domain->omega))
            return false;

        const auto current = required_opening_group(context.proof, stem + "/zeta");
        const auto read_current = [&](const std::string &label) { return required_opening_value(current, label); };
        const auto node_next = required_opening_value(required_opening_group(context.proof, stem + "/zeta_next_node"), stem + "/R_node");
        const auto edge_next = required_opening_value(required_opening_group(context.proof, stem + "/zeta_next_edge"), stem + "/R_edge");
        const auto node_selector = selector_value(node_domain, real_node_count, zeta, false);
        const auto node_last = selector_value(node_domain, 0, zeta, true);
        const auto edge_selector = selector_value(edge_domain, real_edge_count, zeta, false);
        const auto edge_last = selector_value(edge_domain, 0, zeta, true);
        const auto node_multiplicity = public_column_value(multiplicity_values, node_domain, zeta);
        const auto node_z = node_domain->zero_polynomial_eval(zeta);
        const auto edge_z = edge_domain->zero_polynomial_eval(zeta);
        const auto inv_node_residual = node_selector * ((read_current(stem + "/Table") + beta) * read_current(stem + "/J_node") - FieldElement::one());
        const auto inv_edge_residual = edge_selector * ((read_current(stem + "/Query") + beta) * read_current(stem + "/J_edge") - FieldElement::one());
        const auto acc_node_residual = node_next - read_current(stem + "/R_node") -
                                       node_selector * node_multiplicity * read_current(stem + "/J_node") +
                                       node_last * read_current(stem + "/R_node");
        const auto acc_edge_residual = edge_next - read_current(stem + "/R_edge") -
                                       edge_selector * read_current(stem + "/J_edge") +
                                       edge_last * read_current(stem + "/R_edge");
        const auto inv_node_quotient = read_current(stem + "/Q_inv_node") - node_z * read_current(quotient_mask_label(stem + "/Q_inv_node"));
        const auto inv_edge_quotient = read_current(stem + "/Q_inv_edge") - edge_z * read_current(quotient_mask_label(stem + "/Q_inv_edge"));
        const auto acc_node_quotient = read_current(stem + "/Q_acc_node") - node_z * read_current(quotient_mask_label(stem + "/Q_acc_node"));
        const auto acc_edge_quotient = read_current(stem + "/Q_acc_edge") - edge_z * read_current(quotient_mask_label(stem + "/Q_acc_edge"));
        if (inv_node_residual != node_z * inv_node_quotient || inv_edge_residual != edge_z * inv_edge_quotient ||
            acc_node_residual != node_z * acc_node_quotient || acc_edge_residual != edge_z * acc_edge_quotient)
            return false;

        const auto node_terminal = node_domain->point_at(real_node_count);
        const auto edge_terminal = edge_domain->point_at(real_edge_count);
        const auto shift_delta = edge_terminal - node_terminal;
        const auto terminal_y = context.transcript.challenge_excluding(stem + "/terminal_y", [&](const FieldElement &point)
                                                                        { return point.is_zero() ||
                                                                                 edge_domain->zero_polynomial_eval(point + shift_delta).is_zero(); });
        if (!verify_opening_group(context, stem + "/terminal_shifted", {stem + "/R_edge_shifted"}, terminal_y) ||
            !verify_opening_group(context, stem + "/terminal_edge", {stem + "/R_edge"}, terminal_y + shift_delta) ||
            !verify_opening_group(context, stem + "/terminal_fixed", {stem + "/terminal_virtual"}, node_terminal))
            return false;
        const auto expected_virtual_point = crypto::g1_add(required_commitment(context.proof, stem + "/R_node").point,
                                                           crypto::g1_neg(required_commitment(context.proof, stem + "/R_edge_shifted").point));
        if (!(required_commitment(context.proof, stem + "/terminal_virtual").point == expected_virtual_point))
            return false;
        const auto shifted_value = required_opening_value(required_opening_group(context.proof, stem + "/terminal_shifted"),
                                                         stem + "/R_edge_shifted");
        const auto edge_value = required_opening_value(required_opening_group(context.proof, stem + "/terminal_edge"), stem + "/R_edge");
        const auto terminal_value = required_opening_value(required_opening_group(context.proof, stem + "/terminal_fixed"),
                                                          stem + "/terminal_virtual");
        return shifted_value == edge_value && terminal_value.is_zero();
    }
}
