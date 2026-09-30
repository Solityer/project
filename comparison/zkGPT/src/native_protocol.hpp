#ifndef ZKGPT_NATIVE_PROTOCOL_HPP
#define ZKGPT_NATIVE_PROTOCOL_HPP

#include "proof_session.hpp"

#include <cstddef>
#include <string>
#include <vector>

bool verify_quadratic_sumcheck_session(const NativeProofSession& session, const std::string& label, const F& initial_claim, std::size_t rounds,
                                       F* final_sum = nullptr, std::vector<F>* terminal_claims = nullptr, std::string* reason = nullptr);

bool verify_product_sumcheck_session(const NativeProofSession& session, const std::string& label, const F& initial_claim, std::size_t rounds,
                                     F* terminal_left = nullptr, F* terminal_right = nullptr, std::string* reason = nullptr);

bool verify_gkr_predicate_alpha_beta_transition(const F& sumcheck_final, const F& predicate_value, const F& claim_u1, const F& claim_v1, const F& alpha,
                                                const F& beta, const F& next_claim, std::string* reason = nullptr);

#endif
