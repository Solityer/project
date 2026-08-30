#pragma once
// 本文件声明 verifier 入口，
// 它会重放 transcript、检查 metadata 契约，并复验各域 opening 与 quotient。

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

// 验证 proof 是否与独立、point-only verifier context 一致。
bool verify(const VerifierContext& context, const Proof& proof, RunMetrics* metrics = nullptr);

// Final run acceptance is separate from the cryptographic verifier result.
// Both independent checks must pass, while callers preserve each status.
bool final_formal_run_accepted(
    bool verifier_accepted,
    bool reference_consistent,
    bool security_budget_passed = true);

// Verifier-side leakage audit over the actual typed proof payload.
std::size_t private_plain_value_count(const VerifierContext& context, const Proof& proof);

}  // namespace gatzk::protocol
