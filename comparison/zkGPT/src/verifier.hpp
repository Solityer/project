//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_CONVVERIFIER_HPP
#define ZKCNN_CONVVERIFIER_HPP

#include "prover.hpp"
#include "hyrax.hpp"
#include "proof_session.hpp"
#include "native_masked_sumcheck.hpp"
using namespace hyrax;
using std::unique_ptr;
class verifier 
{
public:
    prover *p;
    const layeredCircuit &C;

    verifier(prover *pr, const layeredCircuit &cir);

    // Run the native verifier/prover loop.  When a proof session is supplied,
    // verifier challenges are transcript-bound and all native polynomial
    // messages are appended to that session; the arithmetic protocol itself
    // remains the original GKR/Lasso/Hyrax implementation.
    bool prove(int commit_thread=4);
    bool prove_with_session(NativeProofSession &session, int commit_thread=4);

    // Baseline-local adapter entry point for a composed proof.  It runs only
    // the native GKR/MatMul layer and leaves the session at the exact input
    // claim boundary; callers may then append the native Lasso and terminal
    // Hyrax openings in the same Fiat--Shamir session.  The legacy `prove`
    // entry point remains unchanged for the upstream demo.
    bool prove_gkr_with_session(NativeProofSession &session, int commit_thread=4);
    void range_prove(double range_prover_time_) { range_prover_time = range_prover_time_; };

    timer total_timer, total_slow_timer;
    double verifier_time;
    double prover_time;
    double range_prover_time = 0;
    double matrix_time;
    Commit_return comm;
    
private:
    NativeProofSession *session_ = nullptr;
    F challenge(const std::string &label);
    void absorb_field(const std::string &label, const F &value);
    void absorb_fields(const std::string &label, const std::vector<F> &values);

    vector<vector<F>> r_u, r_v;
    vector<F> final_claim_u0, final_claim_v0;
    bool verifyGKR();
    bool verifyLasso();
    bool openCommit();
    

    vector<F> beta_g;
    void betaInitPhase1(u8 depth, const F &alpha, const F &beta, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1, const F &relu_rou);
    void betaInitPhase2(u8 depth);

    F uni_value[2];
    F bin_value[3];
    void predicatePhase1(u8 layer_id);
    void predicatePhase2(u8 layer_id);

    F getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1);

    F eval_in;
};

struct NativeGkrMatmulShape {
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t cols = 0;
    // The adapter's canonical descriptor identity.  Empty values preserve
    // the upstream synthetic API, but formal GAT contexts populate every
    // field and the native walk compares them to the FCONN layer metadata.
    std::string relation_id;
    std::string family_id;
    std::size_t lhs_offset = 0;
    std::size_t rhs_offset = 0;
    std::size_t output_offset = 0;
};

// Prover-only witness for one native FCONN matrix relation.  The LHS is
// canonical row-major [row,term] with rows*inner entries.  The RHS has
// already crossed the adapter boundary and is native column-major
// [col,term], with a full next-power-of-two padded-inner stride per column.
// They are consumed while producing the transcript and are never part of the
// independent verifier input.
struct NativeGkrMatmulWitness {
    NativeGkrMatmulShape shape;
    std::vector<F> left;
    std::vector<F> right;
};

struct NativeLayeredGkrWitness {
    // One exact value vector per circuit layer, including the public input
    // layer and the public output layer.  Only the prover API accepts this
    // object; the verifier API below receives no witness values.
    std::vector<std::vector<F>> layer_values;
    // Descriptors are ordered by ascending circuit layer, matching the
    // native FCONN registry.  The descending GKR walk consumes them in
    // reverse order and rejects leftovers.
    std::vector<NativeGkrMatmulWitness> matmuls;
};

// Authenticated boundary returned by the masked layered walk.  The point is
// public transcript data; both commitments are Pedersen commitments whose
// openings stay prover-side.  The combine adapter can consume these fields
// without receiving a witness or a raw terminal claim.
struct NativeGkrInputBinding {
    std::vector<F> point;
    zkgpt::native_pedersen::ScalarCommitment claim_commitment;
    zkgpt::native_pedersen::ScalarCommitment unmasked_claim_commitment;
    zkgpt::native_pedersen::ScalarCommitment mask_commitment;
};

// Prover-only result for the combine adapter.  The opening is deliberately
// not part of NativeGkrInputBinding and is never appended to a
// NativeProofSession; it remains an in-memory witness owned by the caller.
struct NativeGkrProverResult {
    NativeGkrInputBinding input_binding;
    zkgpt::native_pedersen::ScalarOpening unmasked_input_claim_opening;
    bool has_unmasked_input_claim_opening = false;
};

// Official masked layered-circuit prover.  It performs the native GKR walk
// directly with streaming masked Sumcheck (ordinary phase1/phase2 and
// dedicated FCONN product Sumcheck); it does not call the clear prover loop
// and does not serialize raw round polynomials or witness terminal values.
bool prove_masked_gkr_transcript(
    NativeProofSession &session, const layeredCircuit &circuit,
    const NativeLayeredGkrWitness &witness,
    const std::vector<F> &public_output, std::string *reason = nullptr,
    NativeGkrInputBinding *input_binding = nullptr,
    NativeGkrProverResult *prover_result = nullptr);

// Independent GKR replay boundary used by the GAT adapter.  The circuit,
// final public output, and public FCONN dimensions are supplied by the
// verifier; all round polynomials, claims and Fiat--Shamir challenges are
// read from NativeProofSession.  This path deliberately does not consult a
// prover object or witness values.
bool verify_gkr_transcript(const NativeProofSession &session,
                           const layeredCircuit &circuit,
                           const std::vector<F> &public_output,
                           const std::vector<NativeGkrMatmulShape> &matmuls,
                           std::string *reason = nullptr);

// Independent verifier for prove_masked_gkr_transcript.  The input surface
// intentionally contains only the layered circuit, public output and public
// FCONN descriptors; it never receives a prover, layer values or matrix
// witness.
bool verify_masked_gkr_transcript(
    const NativeProofSession &session, const layeredCircuit &circuit,
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::string *reason = nullptr,
    NativeGkrInputBinding *input_binding = nullptr);

// Scoped replay boundary for composed artifacts.  `start` points at the
// canonical gkr/public-output message after a preceding Lasso prefix;
// `end` receives the first message after the complete GKR input-claim scope.
// Non-GKR suffixes (combine/openings) remain in the same Fiat--Shamir session
// and are intentionally left for the adapter verifier.
bool verify_masked_gkr_transcript(
    const NativeProofSession &session, const layeredCircuit &circuit,
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::size_t start, std::size_t *end, std::string *reason,
    NativeGkrInputBinding *input_binding = nullptr);


#endif //ZKCNN_CONVVERIFIER_HPP
