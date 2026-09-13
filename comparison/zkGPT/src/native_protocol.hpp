#ifndef ZKGPT_NATIVE_PROTOCOL_HPP
#define ZKGPT_NATIVE_PROTOCOL_HPP

#include "proof_session.hpp"

#include <cstddef>
#include <string>
#include <vector>

// Independent replay helpers for the native sumcheck wires.  They consume
// only public claims and the typed transcript; no prover object, witness, or
// circuit scratch buffers are consulted.
bool verify_quadratic_sumcheck_session(
    const NativeProofSession &session, const std::string &label,
    const F &initial_claim, std::size_t rounds, F *final_sum = nullptr,
    std::vector<F> *terminal_claims = nullptr, std::string *reason = nullptr);

bool verify_product_sumcheck_session(
    const NativeProofSession &session, const std::string &label,
    const F &initial_claim, std::size_t rounds, F *terminal_left = nullptr,
    F *terminal_right = nullptr, std::string *reason = nullptr);

#endif
