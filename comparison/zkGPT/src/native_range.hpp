#ifndef ZKGPT_NATIVE_RANGE_HPP
#define ZKGPT_NATIVE_RANGE_HPP

#include "proof_session.hpp"

#include <string>
#include <vector>

// Session-bound native Sumcheck wires used by the baseline-local Lasso/range
// adapter.  These routines never obtain randomness from the prover or from a
// verifier-side CSPRNG: every challenge is replayable from NativeProofSession.
struct NativeRangeResult {
    F terminal = F_ZERO;
    F terminal_left = F_ZERO;
    F terminal_right = F_ZERO;
    std::vector<F> challenges;
};

bool native_range_prove_deg1(NativeProofSession &session,
                             const std::string &label,
                             const std::vector<F> &values,
                             const F &claim,
                             NativeRangeResult *result = nullptr,
                             std::string *reason = nullptr);

bool native_range_verify_deg1(const NativeProofSession &session,
                              const std::string &label,
                              const F &claim,
                              F *terminal = nullptr,
                              std::string *reason = nullptr);

// Prove sum_i eq[i] * f[i] * g[i] = claim.  The per-round wire contains the
// four evaluations at 0,1,2,3 of the degree-three round polynomial.  The
// independent verifier checks all sum transitions and the terminal product.
bool native_range_prove_deg3(NativeProofSession &session,
                             const std::string &label,
                             const std::vector<F> &eq,
                             const std::vector<F> &f,
                             const std::vector<F> &g,
                             const F &claim,
                             NativeRangeResult *result = nullptr,
                             std::string *reason = nullptr);

bool native_range_verify_deg3(const NativeProofSession &session,
                              const std::string &label,
                              const F &claim,
                              F *terminal = nullptr,
                              std::string *reason = nullptr);

// A degree-two product Sumcheck is the native lookup terminal relation
// sum_y M(r,y) T[y] = ell.  It is kept separate from the degree-three range
// relation because the public table contributes one linear factor.
bool native_range_prove_product(NativeProofSession &session,
                                const std::string &label,
                                const std::vector<F> &left,
                                const std::vector<F> &right,
                                const F &claim,
                                NativeRangeResult *result = nullptr,
                                std::string *reason = nullptr);

bool native_range_verify_product(const NativeProofSession &session,
                                 const std::string &label,
                                 const F &claim,
                                 F *terminal_left = nullptr,
                                 F *terminal_right = nullptr,
                                 std::string *reason = nullptr);

#endif
