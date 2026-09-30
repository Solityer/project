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

    bool prove(int commit_thread=4);
    bool prove_with_session(NativeProofSession &session, int commit_thread=4);

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
    std::string relation_id;
    std::string family_id;
    std::size_t lhs_offset = 0;
    std::size_t rhs_offset = 0;
    std::size_t output_offset = 0;
    std::size_t input_segment_domain = 0;
    std::string lhs_segment;
    std::string rhs_segment;
    std::size_t lhs_segment_offset = 0;
    std::size_t rhs_segment_offset = 0;
    std::vector<std::size_t> input_segment_logical_lengths;
    std::string output_segment;
    std::size_t output_segment_offset = 0;
};

struct NativeGkrMatmulWitness {
    NativeGkrMatmulShape shape;
    std::vector<F> left;
    std::vector<F> right;
};

struct NativeLayeredGkrWitness {
    std::vector<std::vector<F>> layer_values;
    std::vector<NativeGkrMatmulWitness> matmuls;
};

struct NativeGkrInputBinding {
    std::vector<F> point;
    zkgpt::native_pedersen::ScalarCommitment claim_commitment;
    zkgpt::native_pedersen::ScalarCommitment unmasked_claim_commitment;
    zkgpt::native_pedersen::ScalarCommitment mask_commitment;
    zkgpt::native_pedersen::ScalarCommitment unmasked_input_value_commitment;
    zkgpt::native_pedersen::ScalarCommitment masked_input_value_commitment;
    F input_weight_evaluation = F_ZERO;
};

struct NativeGkrProverResult {
    NativeGkrInputBinding input_binding;
    zkgpt::native_pedersen::ScalarOpening unmasked_input_claim_opening;
    bool has_unmasked_input_claim_opening = false;
    zkgpt::native_pedersen::ScalarOpening unmasked_input_value_opening;
    bool has_unmasked_input_value_opening = false;
};

bool prove_masked_gkr_transcript(
    NativeProofSession &session, const layeredCircuit &circuit,
    const NativeLayeredGkrWitness &witness,
    const std::vector<F> &public_output, std::string *reason = nullptr,
    NativeGkrInputBinding *input_binding = nullptr,
    NativeGkrProverResult *prover_result = nullptr);

bool verify_gkr_transcript(const NativeProofSession &session,
                           const layeredCircuit &circuit,
                           const std::vector<F> &public_output,
                           const std::vector<NativeGkrMatmulShape> &matmuls,
                           std::string *reason = nullptr);

bool verify_masked_gkr_transcript(
    const NativeProofSession &session, const layeredCircuit &circuit,
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::string *reason = nullptr,
    NativeGkrInputBinding *input_binding = nullptr);

bool verify_masked_gkr_transcript(
    const NativeProofSession &session, const layeredCircuit &circuit,
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::size_t start, std::size_t *end, std::string *reason,
    NativeGkrInputBinding *input_binding = nullptr);

#endif //ZKCNN_CONVVERIFIER_HPP
