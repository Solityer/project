#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/curve.hpp"

namespace gatzk::crypto {

struct KZGKeyPair {
    algebra::FieldElement tau = algebra::FieldElement::zero();
    G1Point g1_generator;
    G2Point g2_one;
    G2Point g2_tau;
    std::shared_ptr<PreparedG2> g2_one_prepared;
    std::shared_ptr<PreparedG2> g2_tau_prepared;
};

struct Commitment {
    std::string name;
    G1Point point;
    algebra::FieldElement tau_evaluation = algebra::FieldElement::zero();
};

std::size_t serialized_size(const Commitment& commitment);

class KZG {
  public:
    static KZGKeyPair setup(std::uint64_t seed);

    static Commitment commit_tau_evaluation(
        const std::string& name,
        const algebra::FieldElement& tau_evaluation,
        const KZGKeyPair& key);
    static Commitment commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key);
    static std::vector<Commitment> commit_tau_evaluation_batch(
        const std::vector<std::pair<std::string, algebra::FieldElement>>& named_tau_evaluations,
        const KZGKeyPair& key);
    static std::vector<Commitment> commit_batch(
        const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
        const KZGKeyPair& key);

    static G1Point open_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<algebra::FieldElement>& points,
        const std::vector<std::vector<algebra::FieldElement>>& claimed_values,
        const algebra::FieldElement& folding_challenge,
        const KZGKeyPair& key);

    static bool verify_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<algebra::FieldElement>& points,
        const std::vector<std::vector<algebra::FieldElement>>& claimed_values,
        const algebra::FieldElement& folding_challenge,
        const G1Point& witness,
        const KZGKeyPair& key);

    static G1Point open_external_fold(
        const std::vector<std::pair<Commitment, algebra::FieldElement>>& commitments_and_values,
        const std::vector<algebra::FieldElement>& points,
        const algebra::FieldElement& folding_challenge,
        const KZGKeyPair& key);

    static bool verify_external_fold(
        const std::vector<std::pair<Commitment, algebra::FieldElement>>& commitments_and_values,
        const std::vector<algebra::FieldElement>& points,
        const algebra::FieldElement& folding_challenge,
        const G1Point& witness,
        const KZGKeyPair& key);
};

}  // namespace gatzk::crypto
