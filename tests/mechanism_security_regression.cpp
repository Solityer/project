#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/mechanism/internal.hpp"

namespace
{
    using namespace gatzk;
    using namespace gatzk::mechanism;
    using namespace gatzk::mechanism::internal;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    crypto::KZGKeyPair load_key(const std::string &srs_path)
    {
        return crypto::KZG::load_keys_for_domains(srs_path, "trusted_srs", "",
                                                  {{"N", 4096}, {"edge", 32768}}, 4 * 32768 + 256);
    }

    void run_opening_claim_cancellation_negative(const crypto::KZGKeyPair &key)
    {
        using Field = algebra::FieldElement;
        const std::string group_label = "mechanism/security/cancel";
        const Field point(11);
        const std::vector<std::string> labels{"test/P", "test/Q"};
        auto p = algebra::Polynomial::from_coefficients(labels[0], {Field(2), Field(3), Field(5)});
        auto q = algebra::Polynomial::from_coefficients(labels[1], {Field(7), Field(13)});
        auto p_commitment = crypto::KZG::commit(labels[0], p, key);
        auto q_commitment = crypto::KZG::commit(labels[1], q, key);
        p_commitment.opening_polynomial = std::make_shared<algebra::Polynomial>(p);
        q_commitment.opening_polynomial = std::make_shared<algebra::Polynomial>(q);
        const std::vector<crypto::Commitment> commitments{p_commitment, q_commitment};
        const std::vector<Field> correct_values{p.evaluate(point), q.evaluate(point)};

        PublicStatement statement;
        MechanismProof proof;
        proof.commitments = {{labels[0], p_commitment.point}, {labels[1], q_commitment.point}};
        std::vector<std::uint8_t> no_seed;
        double commitment_time_ms = 0.0;
        crypto::Transcript prover_transcript("mechanism-open-group-regression-v1");
        ProofBuildContext build_context{statement, key, prover_transcript, no_seed, proof, commitment_time_ms,
                                        {{labels[0], p_commitment}, {labels[1], q_commitment}}};
        add_opening_group(build_context, group_label, labels, correct_values, point);

        crypto::Transcript valid_transcript("mechanism-open-group-regression-v1");
        ProofVerifyContext valid_context{statement, key, valid_transcript, proof, {}};
        require(verify_opening_group(valid_context, group_label, labels, point),
                "well-formed mechanism point-group opening was rejected");

        // Reproduce the pre-v3 challenge schedule, which omitted claimed values.
        // Choose deltas cancelling in the old folded claim, then build a valid KZG
        // witness for that cancellation. The v3 verifier absorbs the claims first,
        // changes the fold, and must reject the same witness.
        crypto::Transcript old_schedule("mechanism-open-group-regression-v1");
        old_schedule.absorb_text("mechanism_open_group:label", group_label);
        old_schedule.absorb_scalar("mechanism_open_group:point", point);
        old_schedule.absorb_scalar("mechanism_open_group:member_count", Field(labels.size()));
        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            old_schedule.absorb_text("mechanism_open_group:member_label:" + std::to_string(i), labels[i]);
            old_schedule.absorb_commitment("mechanism_open_group:member_point:" + std::to_string(i), commitments[i].point);
        }
        const auto old_fold = old_schedule.challenge_excluding("mechanism_open_fold:v3:" + group_label,
                                                               [](const Field &value) { return value.is_zero(); });
        const std::vector<Field> cancelled_values{correct_values[0] - old_fold, correct_values[1] + Field::one()};
        const auto forged_witness = crypto::KZG::open_point_group(commitments, point, cancelled_values, old_fold, key);
        require(crypto::KZG::verify_point_group(commitments, point, cancelled_values, old_fold, forged_witness, key.vk),
                "claim-cancellation fixture did not satisfy the vulnerable old fold schedule");

        auto forged = proof;
        forged.opening_groups.front().values = cancelled_values;
        forged.opening_groups.front().witness = forged_witness;
        crypto::Transcript strict_transcript("mechanism-open-group-regression-v1");
        ProofVerifyContext strict_context{statement, key, strict_transcript, forged, {}};
        require(!verify_opening_group(strict_context, group_label, labels, point),
                "mechanism verifier accepted claims cancelling under the pre-v3 fold challenge");

        auto reordered = proof;
        std::swap(reordered.opening_groups.front().commitment_labels[0], reordered.opening_groups.front().commitment_labels[1]);
        crypto::Transcript reordered_transcript("mechanism-open-group-regression-v1");
        ProofVerifyContext reordered_context{statement, key, reordered_transcript, reordered, {}};
        require(!verify_opening_group(reordered_context, group_label, labels, point),
                "mechanism verifier accepted a reordered opening member inventory");
    }

    void run_recommitted_invalid_route_negative(const crypto::KZGKeyPair &key)
    {
        using Field = algebra::FieldElement;
        constexpr std::size_t node_rows = 4096;
        constexpr std::size_t edge_rows = 32768;
        constexpr std::size_t active_nodes = 2;
        constexpr std::size_t active_edges = 2;
        const std::string stem = "ff/route/folded";
        const auto node_domain = algebra::RootOfUnityDomain::create("N", node_rows);
        const auto edge_domain = algebra::RootOfUnityDomain::create("edge", edge_rows);

        PublicStatement statement;
        statement.mechanism = Mechanism::FeatureFolding;
        statement.variant = Variant::Original;
        std::vector<Field> table(node_rows, Field::zero());
        std::vector<Field> multiplicity(node_rows, Field::zero());
        std::vector<Field> query(edge_rows, Field::zero());
        table[0] = Field(10);
        table[1] = Field(11);
        multiplicity[0] = Field::one();
        multiplicity[1] = Field::one();
        // Deliberately use the same endpoint twice. This query multiset has a
        // fresh, internally consistent commitment but differs from Table.
        query[0] = Field(10);
        query[1] = Field(10);

        std::vector<std::uint8_t> witness_seed(32, 0x5a);
        MechanismProof proof;
        double commitment_time_ms = 0.0;
        crypto::Transcript prover_transcript("mechanism-invalid-route-regression-v1");
        ProofBuildContext build_context{statement, key, prover_transcript, witness_seed, proof, commitment_time_ms, {}};
        commit_masked_evaluations(build_context, stem + "/Table", table, node_domain, 8);
        commit_masked_evaluations(build_context, stem + "/Query", query, edge_domain, 8);
        bool honest_builder_refused = false;
        try
        {
            (void)prove_route_argument(build_context, stem, table, query, multiplicity, active_nodes, active_edges,
                                      node_domain, edge_domain);
        }
        catch (const std::runtime_error &error)
        {
            honest_builder_refused = std::string(error.what()).find("terminal_virtual") != std::string::npos;
        }
        require(honest_builder_refused,
                "route prover failed to reject a freshly committed source multiset with a nonzero terminal difference");

        // Complete the otherwise valid partial proof with the actual, nonzero
        // terminal opening. This yields a valid KZG proof for every commitment
        // and claimed opening while violating only the route's required zero
        // terminal identity.
        crypto::Transcript partial_transcript("mechanism-invalid-route-regression-v1");
        partial_transcript.absorb_commitment(stem + "/Table", required_commitment(proof, stem + "/Table").point);
        partial_transcript.absorb_commitment(stem + "/Query", required_commitment(proof, stem + "/Query").point);
        ProofVerifyContext partial_context{statement, key, partial_transcript, proof, {}};
        bool reached_terminal_group = false;
        try
        {
            (void)verify_route_argument(partial_context, stem, multiplicity, active_nodes, active_edges, node_domain,
                                        edge_domain);
        }
        catch (const std::runtime_error &error)
        {
            reached_terminal_group = std::string(error.what()).find(stem + "/terminal_fixed") != std::string::npos;
        }
        require(reached_terminal_group, "partial invalid-route fixture failed before its terminal opening group");
        const auto terminal_point = node_domain->point_at(active_nodes);
        const auto terminal_claim = build_context.objects.at(stem + "/terminal_virtual").opening_polynomial->evaluate(terminal_point);
        require(!terminal_claim.is_zero(), "invalid route fixture unexpectedly has a zero terminal difference");
        ProofBuildContext forged_opening_context{statement, key, partial_transcript, witness_seed, proof, commitment_time_ms,
                                                 build_context.objects};
        add_opening_group(forged_opening_context, stem + "/terminal_fixed", {stem + "/terminal_virtual"},
                          {terminal_claim}, terminal_point);

        crypto::Transcript verifier_transcript("mechanism-invalid-route-regression-v1");
        verifier_transcript.absorb_commitment(stem + "/Table", required_commitment(proof, stem + "/Table").point);
        verifier_transcript.absorb_commitment(stem + "/Query", required_commitment(proof, stem + "/Query").point);
        ProofVerifyContext verify_context{statement, key, verifier_transcript, proof, {}};
        require(!verify_route_argument(verify_context, stem, multiplicity, active_nodes, active_edges, node_domain,
                                       edge_domain),
                "mechanism verifier accepted a freshly committed route with a mismatched source multiset");
    }

    void run_recommitted_invalid_psq_terminal_negative(const crypto::KZGKeyPair &key)
    {
        using Field = algebra::FieldElement;
        PublicStatement statement;
        statement.mechanism = Mechanism::SharedNeighborhood;
        statement.variant = Variant::Control;
        statement.nodes = 1;
        statement.edges = 1;
        statement.width = 1;
        statement.graphs = 1;
        statement.node_ptr = {0, 1};
        statement.edge_ptr = {0, 1};
        statement.topology = {Edge{0, 0, 0}};
        statement.node_domain = "N";
        statement.node_domain_size = 4096;
        statement.edge_domain = "edge";
        statement.edge_domain_size = 32768;

        constexpr const char *channel = "den";
        const std::string weight = "sn/U_edge";
        const std::string target = "sn/target_edge/den";
        const std::string state = "sn/psq/den/state";
        const std::string q_step = "sn/psq/den/Q_step";
        const std::string r_step = quotient_mask_label(q_step);
        const std::string q_terminal = "sn/psq/den/Q_group_terminal";
        const std::string r_terminal = quotient_mask_label(q_terminal);
        MechanismProof proof;
        std::vector<std::uint8_t> seed(32, 0x3c);
        double commitment_time_ms = 0.0;
        crypto::Transcript prover_transcript("mechanism-invalid-psq-terminal-regression-v1");
        ProofBuildContext build_context{statement, key, prover_transcript, seed, proof, commitment_time_ms, {}};

        const auto commit_constant = [&](const std::string &label, Field value)
        {
            (void)commit_polynomial(build_context, label, Polynomial::from_coefficients(label, {value}));
        };
        // Fresh source and quotient commitments make every point-group opening
        // cryptographically valid. The step recurrence is satisfied (all zeros),
        // while the target is freshly recommitted to one, violating only the
        // PSQ terminal equality.
        commit_constant(weight, Field::zero());
        commit_constant(target, Field::one());
        commit_constant(state, Field::zero());
        commit_constant(q_step, Field::zero());
        commit_constant(r_step, Field::zero());
        commit_constant(q_terminal, Field::zero());
        commit_constant(r_terminal, Field::zero());
        prove_shared_neighborhood_psq_openings(build_context, channel, weight);

        const std::vector<std::string> absorbed_labels{weight, target, state, q_step, r_step, q_terminal, r_terminal};
        auto prime_verifier_transcript = [&](crypto::Transcript &transcript)
        {
            for (const auto &label : absorbed_labels)
                transcript.absorb_commitment(label, required_commitment(proof, label).point);
        };

        // Confirm the independent KZG point-group openings all verify first, so
        // the following rejection can only be the PSQ identity check.
        crypto::Transcript openings_transcript("mechanism-invalid-psq-terminal-regression-v1");
        prime_verifier_transcript(openings_transcript);
        ProofVerifyContext openings_context{statement, key, openings_transcript, proof, {}};
        const auto edge_domain = algebra::RootOfUnityDomain::create(statement.edge_domain, statement.edge_domain_size);
        const auto zeta = openings_transcript.challenge_excluding(
            "sn/psq/den/zeta", [&](const Field &x) { return x.is_zero() || edge_domain->zero_polynomial_eval(x).is_zero(); });
        require(verify_opening_group(openings_context, "sn/psq/den/open/step", {state, weight, q_step, r_step}, zeta),
                "fresh invalid-PSQ step opening did not verify");
        require(verify_opening_group(openings_context, "sn/psq/den/open/step_next", {state, weight}, zeta * edge_domain->omega),
                "fresh invalid-PSQ next-step opening did not verify");
        const auto terminal_point = openings_transcript.challenge_excluding(
            "sn/psq/den/terminal_point", [&](const Field &x) { return x.is_zero() || edge_domain->zero_polynomial_eval(x).is_zero(); });
        require(verify_opening_group(openings_context, "sn/psq/den/open/group_terminal", {state, target, q_terminal, r_terminal},
                                     terminal_point),
                "fresh invalid-PSQ terminal opening did not verify");

        crypto::Transcript verifier_transcript("mechanism-invalid-psq-terminal-regression-v1");
        prime_verifier_transcript(verifier_transcript);
        ProofVerifyContext verify_context{statement, key, verifier_transcript, proof, {}};
        require(!verify_shared_neighborhood_psq_openings(verify_context, channel, weight),
                "mechanism verifier accepted a freshly committed PSQ target with an invalid terminal sum");
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
            throw std::runtime_error("usage: gatzk_mechanism_security_regression <trusted-srs-directory>");
        const auto key = load_key(argv[1]);
        run_opening_claim_cancellation_negative(key);
        run_recommitted_invalid_route_negative(key);
        run_recommitted_invalid_psq_terminal_negative(key);
        std::cout << "MECHANISM_SECURITY_REGRESSION_OK claim_cancellation=REJECTED reordered_members=REJECTED "
                     "recommitted_invalid_route=PROVER_AND_VERIFIER_REJECTED recommitted_invalid_psq_terminal=VERIFIER_REJECTED\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "mechanism security regression: " << error.what() << '\n';
        return 1;
    }
}
