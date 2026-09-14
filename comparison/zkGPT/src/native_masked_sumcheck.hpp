#ifndef ZKGPT_NATIVE_MASKED_SUMCHECK_HPP
#define ZKGPT_NATIVE_MASKED_SUMCHECK_HPP

#include "proof_session.hpp"
#include "native_pedersen_relation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The native zkGPT transcript uses evaluations at 0,1,...,degree rather
// than coefficients in the monomial basis.  This is the same representation
// used by the original degree-one/degree-two/degree-three wires, but the
// values below are masked before they cross the session boundary.
enum class NativeMaskedTerminalKind : std::uint8_t {
    Scalar = 1,
    Product2 = 2,
    Product3 = 3,
    Claims = 4,
    GkrPredicate = 5,
    // Degree-two relation whose second terminal factor is a verifier-known
    // public MLE evaluation.  Only the masked private terminal is emitted.
    Product2Public = 6,
    // Degree-two GKR predicate with one side fixed by the preceding
    // Sumcheck terminal.  The fixed side is carried only as a masked
    // transcript value; the prover may additionally supply the unmasked
    // value through unmasked_fixed_values for its local consistency check.
    GkrPredicateFixedU = 7,
    GkrPredicateFixedV = 8,
};

struct NativeMaskedTerminalSpec {
    NativeMaskedTerminalKind kind = NativeMaskedTerminalKind::Scalar;

    // GKR predicate coefficients, in the order
    //   bin0*u0*v0 + bin1*u1*v1 + bin2*u1*v0
    //       + uni0*u0 + uni1*u1.
    // The vector must contain exactly five elements for GkrPredicate.
    std::vector<F> gkr_coefficients;
    // Public vector used by Product2Public.  It is supplied independently by
    // the verifier and is never serialized as witness data.
    std::vector<F> public_vector;
    // Filled internally from public_vector and the transcript challenges.
    F public_evaluation = F_ZERO;
    // Masked fixed-side terminal values.  These are public transcript values
    // consumed by the independent verifier, never witness evaluations.
    std::vector<F> fixed_values;
    // Prover-only unmasked fixed-side values.  The native wire never carries
    // this vector; it is used solely to check the unmasked predicate before
    // masking the terminal relation.
    std::vector<F> unmasked_fixed_values;

    // Optional separation between the public/unmasked initial claim and the
    // masked claim consumed by the streaming API.  When absent, the masked
    // claim equals the public claim.  These values are never serialized.
    bool has_masked_initial_claim = false;
    F masked_initial_claim = F_ZERO;

    // Claims relation: final masked claim = c0*t0 + c1*t1 + offset.
    std::vector<F> terminal_linear_coefficients;
    F terminal_linear_offset = F_ZERO;

    // External binding is mandatory for the sound masked path.  Prover
    // callers fill terminal_openings and initial_opening; verifier callers
    // fill the matching commitment vectors.  The ScalarOpening objects are
    // prover-only and are never serialized by this module.
    bool has_external_binding = false;
    // Only the first scope whose claim is explicitly public may use the
    // zero-mask initial mode.  Every composed/private scope must set this to
    // false and provide both sides of its incoming masked/unmasked binding.
    bool public_initial_claim = false;
    zkgpt::native_pedersen::ScalarOpening initial_opening;
    zkgpt::native_pedersen::ScalarCommitment initial_commitment;
    bool has_masked_initial_binding = false;
    zkgpt::native_pedersen::ScalarOpening masked_initial_opening;
    zkgpt::native_pedersen::ScalarCommitment masked_initial_commitment;
    std::vector<zkgpt::native_pedersen::ScalarOpening> terminal_openings;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> terminal_commitments;
    std::vector<zkgpt::native_pedersen::ScalarOpening> fixed_openings;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> fixed_commitments;
    // Prover-only openings for the unmasked terminal operands.  Their
    // commitments are bound to the masked terminal values by the terminal
    // mask relation; no opening value is serialized.
    std::vector<zkgpt::native_pedersen::ScalarOpening> unmasked_terminal_openings;
    std::vector<zkgpt::native_pedersen::ScalarOpening> unmasked_fixed_openings;
    // Verifier-only bindings for externally supplied unmasked fixed operands.
    // The openings remain prover-private and are never serialized.
    std::vector<zkgpt::native_pedersen::ScalarCommitment>
        expected_unmasked_fixed_commitments;
};

struct NativeMaskedRoundBinding {
    std::vector<zkgpt::native_pedersen::ScalarCommitment> p_coefficients;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> h_coefficients;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> q_coefficients;
    zkgpt::native_pedersen::ScalarCommitment p_claim;
    zkgpt::native_pedersen::ScalarCommitment h_claim;
    zkgpt::native_pedersen::ScalarCommitment q_claim;
    zkgpt::native_pedersen::ScalarCommitment h_evaluation;
};

struct NativeMaskedSumcheckResult {
    F final_claim = F_ZERO;
    std::vector<F> terminal_values;
    std::vector<F> challenges;
    zkgpt::native_pedersen::PedersenBasis basis;
    zkgpt::native_pedersen::ScalarCommitment initial_commitment;
    std::vector<NativeMaskedRoundBinding> rounds;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> terminal_commitments;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> fixed_commitments;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> unmasked_terminal_commitments;
    std::vector<zkgpt::native_pedersen::ScalarCommitment> unmasked_fixed_commitments;
    zkgpt::native_pedersen::ScalarCommitment claim_commitment;
    zkgpt::native_pedersen::ScalarCommitment unmasked_claim_commitment;
    zkgpt::native_pedersen::ScalarCommitment claim_mask_commitment;
    // Prover-only opening witnesses retained for compositional callers.
    std::vector<zkgpt::native_pedersen::ScalarOpening> terminal_openings;
    std::vector<zkgpt::native_pedersen::ScalarOpening> fixed_openings;
    std::vector<zkgpt::native_pedersen::ScalarOpening> unmasked_terminal_openings;
    std::vector<zkgpt::native_pedersen::ScalarOpening> unmasked_fixed_openings;
    zkgpt::native_pedersen::ScalarOpening claim_opening;
    zkgpt::native_pedersen::ScalarOpening unmasked_claim_opening;
    zkgpt::native_pedersen::ScalarOpening claim_mask_opening;
    bool has_claim_opening = false;
    bool has_unmasked_claim_opening = false;
    bool has_claim_mask_opening = false;
    // End cursor of this authenticated scope in the session transcript.  It
    // is populated only by the independent verifier and is not proof data.
    std::size_t transcript_end = 0;
    // Intermediate/anchor commitments used by the terminal relation.  They
    // are public proof operands only; no opening values or blinders cross the
    // session boundary.
    std::vector<zkgpt::native_pedersen::ScalarCommitment> operand_commitments;
    bool external_binding_required = true;
    bool external_binding_verified = false;
};

// Streaming prover form for native GKR/MatMul code, whose next unmasked
// polynomial is computed only after the previous Fiat--Shamir challenge.  It
// emits the same wire as native_masked_sumcheck_prove and keeps all masks in
// the prover object only.
class NativeMaskedSumcheckProver {
public:
    NativeMaskedSumcheckProver(
        NativeProofSession &session, const std::string &label,
        std::size_t degree, const F &initial_claim,
        const NativeMaskedTerminalSpec &terminal_spec,
        const F *unmasked_initial_claim = nullptr);
    ~NativeMaskedSumcheckProver();
    NativeMaskedSumcheckProver(const NativeMaskedSumcheckProver &) = delete;
    NativeMaskedSumcheckProver &operator=(const NativeMaskedSumcheckProver &) = delete;

    struct Impl;

    bool append_round(const std::vector<F> &unmasked_poly, F *challenge = nullptr,
                      std::string *reason = nullptr);
    // The GKR predicate coefficients depend on the phase challenges that are
    // generated while rounds are streamed.  The prover therefore starts with
    // a placeholder terminal spec and freezes the relation immediately
    // before finish(); no terminal relation is serialized before that point.
    bool set_terminal_spec(const NativeMaskedTerminalSpec &terminal_spec,
                           std::string *reason = nullptr);
    bool finish(const std::vector<F> &unmasked_terminal,
                NativeMaskedSumcheckResult *result = nullptr,
                std::string *reason = nullptr);
    std::size_t rounds() const;
    const NativeMaskedSumcheckResult &result() const;

private:
    std::unique_ptr<Impl> impl_;
};

// A composable, session-bound masked Sumcheck.  The prover input is private
// and is never serialized: each round_polys entry is the unmasked polynomial
// evaluated at 0..degree, and terminal_values contains its private terminal
// witness values.  The resulting session contains only masked evaluations,
// Pedersen mask commitments, and Schnorr responses.
//
// For every round i the hidden mask h_i satisfies
//   h_0(0)+h_0(1)=0,
//   h_i(0)+h_i(1)=h_{i-1}(r_{i-1})  (i>0).
// Consequently the public initial claim and all masked claim transitions
// remain exact.  The terminal relation is checked over the masked terminal
// values, so no terminal witness evaluation is emitted.
bool native_masked_sumcheck_prove(
    NativeProofSession &session, const std::string &label, std::size_t degree,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const NativeMaskedTerminalSpec &terminal_spec,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

// Independent verifier.  It consumes only the public initial claim, public
// degree/round count/terminal relation, commitments, Schnorr responses and
// masked round messages already present in session.  It never consults the
// prover or any witness array.
bool native_masked_sumcheck_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t degree, std::size_t rounds, const F &initial_claim,
    const NativeMaskedTerminalSpec &terminal_spec,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

// Decode the commitment carried by an opening block at an already checked
// transcript cursor.  This is an artifact-boundary helper for compositional
// verifiers which need to bind a newly produced output opening before calling
// native_pedersen verification; it never exposes the opening value/blinding.
bool native_masked_opening_commitment_at(
    const NativeProofSession &session,
    const zkgpt::native_pedersen::PedersenBasis &basis,
    std::size_t cursor, const std::string &label,
    zkgpt::native_pedersen::ScalarCommitment &commitment,
    std::string *reason = nullptr);

// Convenience entry points used by the native protocol adapters.  They make
// the supported relation families explicit without duplicating transcript
// code: GKR and dedicated MatMul use degree two, while Lasso/range uses the
// degree selected by its relation (1, 2 or 3).
bool native_masked_gkr_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_gkr_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_gkr_fixed_u_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_u,
    const std::vector<F> &unmasked_u,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_gkr_fixed_u_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_u,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_gkr_fixed_v_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_v,
    const std::vector<F> &unmasked_v,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_gkr_fixed_v_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_v,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_matmul_product_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

bool native_masked_matmul_product_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    NativeMaskedSumcheckResult *result = nullptr,
    std::string *reason = nullptr);

#endif
