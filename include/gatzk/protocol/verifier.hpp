#pragma once

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    bool verify(const VerifierContext &context, const Proof &proof, RunMetrics *metrics = nullptr);

    bool final_formal_run_accepted(bool verifier_accepted, bool reference_consistent, bool security_budget_passed = true);

    std::size_t private_plain_value_count(const VerifierContext &context, const Proof &proof);

}
