#ifndef ZKGPT_NATIVE_PEDERSEN_RELATION_HPP
#define ZKGPT_NATIVE_PEDERSEN_RELATION_HPP

// Baseline-local scalar Pedersen Sigma relations.  This module is deliberately
// independent from the MLE/Hyrax opening code: it only supplies reusable
// hidden-scalar relations which can be composed in a NativeProofSession.

#include "proof_session.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace zkgpt {
namespace native_pedersen {

// The domain is part of the public basis identity.  G and H are derived from
// the same domain-separated namespace and are checked against that derivation
// by every prover and verifier entry point.
struct PedersenBasis {
    std::string domain;
    G1 G;
    G1 H;

    bool valid(std::string *reason = nullptr) const;
};

PedersenBasis derive_basis(const std::string &domain);

// Non-serialized provenance for the native prover fast path.  A token is
// created only after a complete basis validation and is bound to one live
// NativeProofSession plus the exact domain and G/H points.  Artifact readers
// never construct or receive this object, so they always use full checks.
class ProverFastPath {
public:
    bool matches(const NativeProofSession &session,
                 const PedersenBasis &basis) const;
    bool matches_basis(const PedersenBasis &basis) const;

private:
    friend std::shared_ptr<const ProverFastPath> make_prover_fast_path(
        NativeProofSession &, const PedersenBasis &);
    ProverFastPath(NativeProofSession &session, const PedersenBasis &basis);

    NativeProofSession *session_ = nullptr;
    std::string domain_;
    G1 G_;
    G1 H_;
};

std::shared_ptr<const ProverFastPath> make_prover_fast_path(
    NativeProofSession &session, const PedersenBasis &basis);

struct ScalarCommitment {
    std::string domain;
    G1 point;
    std::shared_ptr<const ProverFastPath> prover_fast_path;

    bool valid(const PedersenBasis &basis, std::string *reason = nullptr) const;
};

// Prover-only opening state.  It is intentionally not accepted by any wire
// reader and is never copied into a proof object or NativeProofSession.
struct ScalarOpening {
    ScalarCommitment commitment;
    Fr value;
    Fr blinding;
    std::shared_ptr<const ProverFastPath> prover_fast_path;

    bool valid(const PedersenBasis &basis, std::string *reason = nullptr) const;
};

ScalarCommitment commit(const PedersenBasis &basis, const Fr &value,
                        const Fr &blinding,
                        std::shared_ptr<const ProverFastPath> fast_path = nullptr);
ScalarOpening commit_fresh(
    const PedersenBasis &basis, const Fr &value,
    std::shared_ptr<const ProverFastPath> fast_path = nullptr);

// Public proof objects contain only the first Sigma message, the FS challenge
// and responses.  They do not contain value/blinding witnesses; the session
// APIs below are the artifact boundary used by the independent verifier.
struct OpeningProof {
    G1 commitment_message;
    Fr challenge;
    Fr response_value;
    Fr response_blinding;
};

bool prove_opening(NativeProofSession &session, const PedersenBasis &basis,
                   const std::string &label, const ScalarOpening &opening,
                   OpeningProof *proof = nullptr,
                   std::string *reason = nullptr,
                   std::shared_ptr<const ProverFastPath> fast_path = nullptr);

bool verify_opening(const NativeProofSession &session,
                    const PedersenBasis &basis, const std::string &label,
                    const ScalarCommitment &commitment,
                    std::string *reason = nullptr);

// C_rel = sum_i coeff_i C_i - G*b.  If the committed values satisfy
// sum_i coeff_i x_i = b, then C_rel = H*r_rel.  The prover proves knowledge
// of r_rel with a one-generator Schnorr proof.  The value/blinding openings
// are prover-only and are checked before anything is appended to the wire.
struct LinearRelationProof {
    G1 relation_commitment;
    G1 commitment_message;
    Fr challenge;
    Fr response_blinding;
};

G1 linear_residual_commitment(const PedersenBasis &basis,
                              const std::vector<ScalarCommitment> &operands,
                              const std::vector<Fr> &coefficients,
                              const Fr &public_value,
                              std::shared_ptr<const ProverFastPath> fast_path = nullptr);

bool prove_hidden_linear(
    NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const std::vector<ScalarOpening> &operands,
    const std::vector<Fr> &coefficients, const Fr &public_value,
    LinearRelationProof *proof = nullptr, std::string *reason = nullptr,
    std::shared_ptr<const ProverFastPath> fast_path = nullptr);

bool verify_hidden_linear(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label,
    const std::vector<ScalarCommitment> &operands,
    const std::vector<Fr> &coefficients, const Fr &public_value,
    std::string *reason = nullptr);

// Multiplication Sigma relation.  The links identify already emitted
// representation proofs for A, B and C.  Requiring those identities makes a
// composed proof explicit about which committed operands it consumes.
struct MultiplicationLinks {
    std::string a_opening_label;
    std::string b_opening_label;
    std::string c_opening_label;

    bool valid(std::string *reason = nullptr) const;
};

struct MultiplicationProof {
    G1 first_commitment_message;
    G1 second_commitment_message;
    Fr challenge;
    Fr response_a;
    Fr response_a_blinding;
    Fr response_multiplication_blinding;
};

bool prove_multiplication(
    NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const ScalarOpening &a,
    const ScalarOpening &b, const ScalarOpening &c,
    const MultiplicationLinks &links, MultiplicationProof *proof = nullptr,
    std::string *reason = nullptr,
    std::shared_ptr<const ProverFastPath> fast_path = nullptr);

bool verify_multiplication(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const ScalarCommitment &A,
    const ScalarCommitment &B, const ScalarCommitment &C,
    const MultiplicationLinks &links, std::string *reason = nullptr);

// Product3 is composed from an intermediate commitment M=AB and two native
// multiplication Sigma blocks: (A,B,M), then (M,C,D), where D=ABC.
struct Product3Links {
    MultiplicationLinks first;
    MultiplicationLinks second;

    bool valid(std::string *reason = nullptr) const;
};

struct Product3Proof {
    G1 intermediate_commitment;
    G1 output_commitment;
};

bool prove_product3(
    NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const ScalarOpening &a,
    const ScalarOpening &b, const ScalarOpening &c,
    const ScalarOpening &intermediate, const ScalarOpening &output,
    const Product3Links &links, Product3Proof *proof = nullptr,
    std::string *reason = nullptr,
    std::shared_ptr<const ProverFastPath> fast_path = nullptr);

bool verify_product3(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const ScalarCommitment &A,
    const ScalarCommitment &B, const ScalarCommitment &C,
    const ScalarCommitment &intermediate, const ScalarCommitment &output,
    const Product3Links &links, std::string *reason = nullptr);

// Public affine helpers are commitment-level adapters for compositional
// relations.  The output has y = scalar*x + offset and a fresh blinding.
ScalarOpening public_affine(const PedersenBasis &basis,
                            const ScalarOpening &input, const Fr &scalar,
                            const Fr &offset);

bool prove_public_affine(
    NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const ScalarOpening &input,
    const ScalarOpening &output, const Fr &scalar, const Fr &offset,
    LinearRelationProof *proof = nullptr, std::string *reason = nullptr);

bool verify_public_affine(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const ScalarCommitment &input,
    const ScalarCommitment &output, const Fr &scalar, const Fr &offset,
    std::string *reason = nullptr);

} // namespace native_pedersen
} // namespace zkgpt

#endif // ZKGPT_NATIVE_PEDERSEN_RELATION_HPP
