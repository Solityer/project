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

enum class NativeMaskedTerminalKind : std::uint8_t {
    Scalar = 1,
    Product2 = 2,
    Product3 = 3,
    Claims = 4,
    GkrPredicate = 5,

    Product2Public = 6,

    GkrPredicateFixedU = 7,
    GkrPredicateFixedV = 8,
};

struct NativeMaskedTerminalSpec {
    NativeMaskedTerminalKind kind = NativeMaskedTerminalKind::Scalar;

    std::vector<F> gkr_coefficients;

    std::vector<F> public_vector;

    F public_evaluation = F_ZERO;

    std::vector<F> fixed_values;

    std::vector<F> unmasked_fixed_values;

    bool has_masked_initial_claim = false;
    F masked_initial_claim = F_ZERO;

    std::vector<F> terminal_linear_coefficients;
    F terminal_linear_offset = F_ZERO;

    bool defer_terminal_relation = false;

    bool has_external_binding = false;

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

    std::vector<zkgpt::native_pedersen::ScalarOpening> unmasked_terminal_openings;
    std::vector<zkgpt::native_pedersen::ScalarOpening> unmasked_fixed_openings;

    std::vector<zkgpt::native_pedersen::ScalarCommitment> expected_unmasked_fixed_commitments;
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

    std::size_t transcript_end = 0;

    std::vector<zkgpt::native_pedersen::ScalarCommitment> operand_commitments;
    bool external_binding_required = true;
    bool external_binding_verified = false;
};

class NativeMaskedSumcheckProver {
  public:
    NativeMaskedSumcheckProver(NativeProofSession& session, const std::string& label, std::size_t degree, const F& initial_claim,
                               const NativeMaskedTerminalSpec& terminal_spec, const F* unmasked_initial_claim = nullptr,
                               const zkgpt::native_pedersen::PedersenBasis* basis = nullptr,
                               std::shared_ptr<const zkgpt::native_pedersen::ProverFastPath> fast_path = nullptr);
    ~NativeMaskedSumcheckProver();
    NativeMaskedSumcheckProver(const NativeMaskedSumcheckProver&) = delete;
    NativeMaskedSumcheckProver& operator=(const NativeMaskedSumcheckProver&) = delete;

    struct Impl;

    bool append_round(const std::vector<F>& unmasked_poly, F* challenge = nullptr, std::string* reason = nullptr);

    bool set_terminal_spec(const NativeMaskedTerminalSpec& terminal_spec, std::string* reason = nullptr);
    bool finish(const std::vector<F>& unmasked_terminal, NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);
    std::size_t rounds() const;
    const NativeMaskedSumcheckResult& result() const;

  private:
    std::unique_ptr<Impl> impl_;
};

bool native_masked_sumcheck_prove(NativeProofSession& session, const std::string& label, std::size_t degree, const std::vector<std::vector<F>>& round_polys,
                                  const F& initial_claim, const std::vector<F>& terminal_values, const NativeMaskedTerminalSpec& terminal_spec,
                                  NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

bool native_masked_sumcheck_verify(const NativeProofSession& session, const std::string& label, std::size_t degree, std::size_t rounds, const F& initial_claim,
                                   const NativeMaskedTerminalSpec& terminal_spec, NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

bool native_masked_opening_commitment_at(const NativeProofSession& session, const zkgpt::native_pedersen::PedersenBasis& basis, std::size_t cursor,
                                         const std::string& label, zkgpt::native_pedersen::ScalarCommitment& commitment, std::string* reason = nullptr);

bool native_masked_gkr_quadratic_prove(NativeProofSession& session, const std::string& label, const std::vector<std::vector<F>>& round_polys,
                                       const F& initial_claim, const std::vector<F>& terminal_values, const std::vector<F>& gkr_coefficients,
                                       NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

bool native_masked_gkr_quadratic_verify(const NativeProofSession& session, const std::string& label, std::size_t rounds, const F& initial_claim,
                                        const std::vector<F>& gkr_coefficients, NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

bool native_masked_gkr_fixed_u_quadratic_prove(NativeProofSession& session, const std::string& label, const std::vector<std::vector<F>>& round_polys,
                                               const F& initial_claim, const std::vector<F>& terminal_values, const std::vector<F>& gkr_coefficients,
                                               const std::vector<F>& masked_u, const std::vector<F>& unmasked_u, NativeMaskedSumcheckResult* result = nullptr,
                                               std::string* reason = nullptr);

bool native_masked_gkr_fixed_u_quadratic_verify(const NativeProofSession& session, const std::string& label, std::size_t rounds, const F& initial_claim,
                                                const std::vector<F>& gkr_coefficients, const std::vector<F>& masked_u,
                                                NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

bool native_masked_gkr_fixed_v_quadratic_prove(NativeProofSession& session, const std::string& label, const std::vector<std::vector<F>>& round_polys,
                                               const F& initial_claim, const std::vector<F>& terminal_values, const std::vector<F>& gkr_coefficients,
                                               const std::vector<F>& masked_v, const std::vector<F>& unmasked_v, NativeMaskedSumcheckResult* result = nullptr,
                                               std::string* reason = nullptr);

bool native_masked_gkr_fixed_v_quadratic_verify(const NativeProofSession& session, const std::string& label, std::size_t rounds, const F& initial_claim,
                                                const std::vector<F>& gkr_coefficients, const std::vector<F>& masked_v,
                                                NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

bool native_masked_matmul_product_prove(NativeProofSession& session, const std::string& label, const std::vector<std::vector<F>>& round_polys,
                                        const F& initial_claim, const std::vector<F>& terminal_values, NativeMaskedSumcheckResult* result = nullptr,
                                        std::string* reason = nullptr);

bool native_masked_matmul_product_verify(const NativeProofSession& session, const std::string& label, std::size_t rounds, const F& initial_claim,
                                         NativeMaskedSumcheckResult* result = nullptr, std::string* reason = nullptr);

#endif
