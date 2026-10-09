#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/mechanism/runtime.hpp"

namespace gatzk::mechanism::internal
{
    using FieldElement = algebra::FieldElement;
    using Polynomial = algebra::Polynomial;
    using Commitment = crypto::Commitment;

    struct ProofBuildContext
    {
        const PublicStatement &statement;
        const crypto::KZGKeyPair &key;
        crypto::Transcript &transcript;
        const std::vector<std::uint8_t> &witness_seed;
        MechanismProof &proof;
        double &commitment_time_ms;
        std::map<std::string, Commitment> objects;
    };

    struct ProofVerifyContext
    {
        const PublicStatement &statement;
        const crypto::KZGKeyPair &key;
        crypto::Transcript &transcript;
        const MechanismProof &proof;
        std::map<std::string, Commitment> objects;
    };

    struct RouteWitness
    {
        std::vector<FieldElement> table_inverse;
        std::vector<FieldElement> query_inverse;
        std::vector<FieldElement> node_accumulator;
        std::vector<FieldElement> edge_accumulator;
        std::vector<FieldElement> node_inverse_blinder;
        std::vector<FieldElement> edge_inverse_blinder;
        std::vector<FieldElement> node_accumulator_blinder;
        std::vector<FieldElement> edge_accumulator_blinder;
    };

    struct RouteQuotients
    {
        Polynomial node_inverse;
        Polynomial edge_inverse;
        Polynomial node_accumulator;
        Polynomial edge_accumulator;
    };

    std::vector<FieldElement> sample_zero_domain_blinder(const std::vector<std::uint8_t> &seed, const std::string &label,
                                                         std::size_t coefficient_count);

    std::vector<FieldElement> derive_witness_fields(const std::vector<std::uint8_t> &seed, const std::string &label,
                                                    std::size_t value_count);

    Polynomial make_masked_evaluation_polynomial(const std::string &label, const std::vector<FieldElement> &values,
                                                 const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                                 const std::vector<FieldElement> &zero_domain_blinder);

    Commitment commit_polynomial(ProofBuildContext &context, const std::string &label, Polynomial polynomial,
                                 bool absorb_in_transcript = true);

    Commitment commit_masked_evaluations(ProofBuildContext &context, const std::string &label,
                                         const std::vector<FieldElement> &values,
                                         const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                         std::size_t observation_rank, bool absorb_in_transcript = true);

    std::string quotient_mask_label(const std::string &quotient_label);
    void commit_quotient_with_mask(ProofBuildContext &context, const std::string &quotient_label,
                                   Polynomial quotient, std::size_t domain_size);

    Commitment derive_linear_commitment(ProofBuildContext &context, const std::string &label,
                                        const std::vector<std::pair<std::string, FieldElement>> &terms,
                                        bool absorb_in_transcript = true);

    void record_commitment(ProofBuildContext &context, const std::string &label, Commitment commitment,
                           bool absorb_in_transcript = true);

    void add_opening_group(ProofBuildContext &context, const std::string &label,
                           const std::vector<std::string> &commitment_labels,
                           const std::vector<FieldElement> &values, const FieldElement &point);

    RouteWitness build_route_witness(const std::vector<FieldElement> &table, const std::vector<FieldElement> &query,
                                     const std::vector<FieldElement> &multiplicity, std::size_t real_node_count,
                                     std::size_t real_edge_count,
                                     const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                                     const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
                                     const FieldElement &beta);

    RouteQuotients build_route_quotient_polynomials(const Polynomial &table, const Polynomial &query,
                                                     const Polynomial &multiplicity, const RouteWitness &witness,
                                                     std::size_t real_node_count, std::size_t real_edge_count,
                                                     const FieldElement &beta,
                                                     const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                                                     const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain);

    std::vector<std::string> route_commitment_labels(const std::string &stem);
    std::vector<std::string> route_opening_group_labels(const std::string &stem);
    FieldElement prove_route_argument(ProofBuildContext &context, const std::string &stem,
                              const std::vector<FieldElement> &table_values, const std::vector<FieldElement> &query_values,
                              const std::vector<FieldElement> &multiplicity_values, std::size_t real_node_count,
                              std::size_t real_edge_count, const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                              const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
                              const std::vector<std::string> &extra_zeta_commitments = {});
    bool verify_route_argument(ProofVerifyContext &context, const std::string &stem,
                               const std::vector<FieldElement> &multiplicity_values, std::size_t real_node_count,
                               std::size_t real_edge_count, const std::shared_ptr<algebra::RootOfUnityDomain> &node_domain,
                               const std::shared_ptr<algebra::RootOfUnityDomain> &edge_domain,
                               const std::vector<std::string> &extra_zeta_commitments = {}, FieldElement *zeta_out = nullptr);

    Polynomial quotient_by_domain_vanishing(const Polynomial &identity_residual, std::size_t domain_size,
                                            const std::string &label);

    const CommitmentRecord &required_commitment(const MechanismProof &proof, const std::string &label);
    const OpeningGroup &required_opening_group(const MechanismProof &proof, const std::string &label);
    FieldElement required_opening_value(const OpeningGroup &group, const std::string &commitment_label);

    bool verify_opening_group(ProofVerifyContext &context, const std::string &group_label,
                              const std::vector<std::string> &commitment_labels, const FieldElement &point);

    void prove_shared_neighborhood_psq_openings(ProofBuildContext &context, const std::string &channel,
                                                const std::string &weight_label);
    bool verify_shared_neighborhood_psq_openings(ProofVerifyContext &context, const std::string &channel,
                                                  const std::string &weight_label);

    std::string compute_mechanism_schema_id(const PublicStatement &statement);
    std::string compute_security_budget_ledger_sha256(const PublicStatement &statement);
    std::string security_budget_ledger_json(const PublicStatement &statement);

    void prove_shared_neighborhood(ProofBuildContext &context);
    bool verify_shared_neighborhood(ProofVerifyContext &context);
    std::vector<std::string> shared_neighborhood_commitment_labels(const PublicStatement &statement);
    std::vector<std::string> shared_neighborhood_opening_group_labels(const PublicStatement &statement);

    void prove_feature_folding(ProofBuildContext &context);
    bool verify_feature_folding(ProofVerifyContext &context);
    std::vector<std::string> feature_folding_commitment_labels(const PublicStatement &statement);
    std::vector<std::string> feature_folding_opening_group_labels(const PublicStatement &statement);
}
