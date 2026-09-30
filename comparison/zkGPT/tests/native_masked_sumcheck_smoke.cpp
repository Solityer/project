#include "native_masked_sumcheck.hpp"

#include <mcl/bn256.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;

static NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "0000000000000000000000000000000000000000000000000000000000000121", "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000122", "0000000000000000000000000000000000000000000000000000000000000123",
        "0000000000000000000000000000000000000000000000000000000000000124", "0000000000000000000000000000000000000000000000000000000000000125",
        "0000000000000000000000000000000000000000000000000000000000000126", "0000000000000000000000000000000000000000000000000000000000000127",
        "0000000000000000000000000000000000000000000000000000000000000128", 4,
        "0000000000000000000000000000000000000000000000000000000000000121"};
}

static bool mutate_initial_mode(std::vector<NativeProofMessage>& messages) {
    for (NativeProofMessage& message : messages) {
        if (message.label != "matmul/masked/initial/mode") continue;
        message.payload = "with-mask-v2";
        return true;
    }
    return false;
}

static bool mutate_terminal_value(std::vector<NativeProofMessage>& messages) {
    for (NativeProofMessage& message : messages) {
        if (message.kind != NativeProofMessageKind::FieldVector || message.label != "matmul/masked/terminal/values" || message.payload.empty()) continue;
        for (std::size_t index = message.payload.size(); index-- > 0;) {
            const char value = message.payload[index];
            if (value < '0' || value > 'f' || (value > '9' && value < 'a')) continue;
            message.payload[index] = value == '0' ? '1' : '0';
            return true;
        }
    }
    return false;
}

static bool mutate_terminal_commitment(std::vector<NativeProofMessage>& messages) {
    std::string replacement;
    for (const NativeProofMessage& message : messages)
        if (message.kind == NativeProofMessageKind::Data && message.label == "matmul/masked/terminal/value/1/meta") replacement = message.payload;
    if (replacement.empty()) return false;
    for (NativeProofMessage& message : messages) {
        if (message.kind != NativeProofMessageKind::Data || message.label != "matmul/masked/terminal/value/0/meta") continue;
        message.payload = replacement;
        return true;
    }
    return false;
}

static bool fixed_v_binding_fixture() {
    std::string reason;
    NativeProofSession session("native-masked-fixed-v-v2", statement_fixture());
    const auto basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest());
    const auto initial_unmasked = zkgpt::native_pedersen::commit_fresh(basis, Fr(0));
    const auto initial_masked = zkgpt::native_pedersen::commit_fresh(basis, Fr(0));
    const auto zero_commitment = zkgpt::native_pedersen::commit(basis, F_ZERO, F_ZERO);
    const zkgpt::native_pedersen::ScalarOpening zero_opening{zero_commitment, F_ZERO, F_ZERO};

    NativeMaskedTerminalSpec prover_spec;
    prover_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedV;

    prover_spec.gkr_coefficients = {F_ZERO, F_ZERO, F_ZERO, F_ONE, F_ZERO};
    prover_spec.fixed_values = {F_ZERO, F_ZERO};
    prover_spec.unmasked_fixed_values = {F_ZERO, F_ZERO};
    prover_spec.has_external_binding = true;
    prover_spec.public_initial_claim = false;
    prover_spec.initial_opening = initial_unmasked;
    prover_spec.initial_commitment = initial_unmasked.commitment;
    prover_spec.has_masked_initial_binding = true;
    prover_spec.masked_initial_opening = initial_masked;
    prover_spec.masked_initial_commitment = initial_masked.commitment;
    prover_spec.unmasked_fixed_openings = {zero_opening, zero_opening};
    prover_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(basis, Fr(0)), zkgpt::native_pedersen::commit_fresh(basis, Fr(9))};

    const F unmasked_initial = F_ZERO;
    NativeMaskedSumcheckProver prover(session, "fixed-v/masked", 2, F_ZERO, prover_spec, &unmasked_initial);
    if (!prover.append_round({F_ZERO, F_ZERO, F_ZERO}, nullptr, &reason)) {
        std::cerr << "fixed-V prover round failed: " << reason << '\n';
        return false;
    }
    NativeMaskedSumcheckResult proved;
    if (!prover.finish({Fr(0), Fr(9)}, &proved, &reason)) {
        std::cerr << "fixed-V prover finish failed: " << reason << '\n';
        return false;
    }

    NativeMaskedTerminalSpec verify_spec = prover_spec;
    verify_spec.initial_opening = {};
    verify_spec.masked_initial_opening = {};
    verify_spec.terminal_openings.clear();
    verify_spec.fixed_openings.clear();
    verify_spec.unmasked_terminal_openings.clear();
    verify_spec.unmasked_fixed_openings.clear();
    verify_spec.terminal_commitments = proved.terminal_commitments;
    verify_spec.fixed_commitments = proved.fixed_commitments;
    verify_spec.expected_unmasked_fixed_commitments = {zero_commitment, zero_commitment};
    NativeMaskedSumcheckResult verified;
    if (!native_masked_sumcheck_verify(session, "fixed-v/masked", 2, 1, F_ZERO, verify_spec, &verified, &reason) || !verified.external_binding_verified) {
        std::cerr << "fixed-V independent verify failed: " << reason << '\n';
        return false;
    }
    if (verified.unmasked_fixed_commitments.size() != 2 || verified.unmasked_fixed_commitments[0].point != zero_commitment.point ||
        verified.unmasked_fixed_commitments[1].point != zero_commitment.point) {
        std::cerr << "fixed-V unmasked fixed binding was not returned\n";
        return false;
    }

    NativeMaskedTerminalSpec tampered = verify_spec;
    tampered.expected_unmasked_fixed_commitments[0] = zkgpt::native_pedersen::commit(basis, F_ZERO, F_ONE);
    if (native_masked_sumcheck_verify(session, "fixed-v/masked", 2, 1, F_ZERO, tampered, nullptr, &reason)) {
        std::cerr << "fixed-V unmasked fixed commitment tamper was accepted\n";
        return false;
    }
    return true;
}

int main() {
    initPairing(mcl::BN254);
    std::string reason;

    NativeProofSession first("native-masked-smoke-v2", statement_fixture());
    const auto first_basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + first.statement_digest());
    const auto first_initial = zkgpt::native_pedersen::commit_fresh(first_basis, Fr(2));
    NativeMaskedTerminalSpec product_spec;
    product_spec.kind = NativeMaskedTerminalKind::Product2;
    product_spec.has_external_binding = true;
    product_spec.public_initial_claim = true;
    product_spec.initial_opening = first_initial;
    product_spec.initial_commitment = first_initial.commitment;
    product_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(first_basis, Fr(1)),
                                               zkgpt::native_pedersen::commit_fresh(first_basis, Fr(1))};
    NativeMaskedSumcheckResult proved;
    if (!native_masked_sumcheck_prove(first, "matmul/masked", 2, {{Fr(1), Fr(1), Fr(1)}}, Fr(2), {Fr(1), Fr(1)}, product_spec, &proved, &reason)) {
        std::cerr << "masked product honest path failed: " << reason << '\n';
        return 1;
    }
    if (proved.terminal_values.size() != 2 || proved.terminal_values[0] == Fr(1) || proved.terminal_values[1] == Fr(1)) {
        std::cerr << "masked terminal exposed the witness values\n";
        return 2;
    }
    NativeMaskedTerminalSpec verify_spec = product_spec;
    verify_spec.initial_opening = {};
    verify_spec.terminal_commitments = proved.terminal_commitments;
    if (!native_masked_sumcheck_verify(first, "matmul/masked", 2, 1, Fr(2), verify_spec, nullptr, &reason)) {
        std::cerr << "masked product independent verify failed: " << reason << '\n';
        return 3;
    }
    if (native_masked_sumcheck_verify(first, "matmul/masked", 2, 1, Fr(3), verify_spec, nullptr, &reason)) {
        std::cerr << "masked public-claim tamper was accepted\n";
        return 4;
    }
    std::vector<NativeProofMessage> terminal_value_tampered = first.messages();
    if (!mutate_terminal_value(terminal_value_tampered)) {
        std::cerr << "masked terminal value fixture was not found\n";
        return 5;
    }
    NativeProofSession terminal_value_session;
    if (NativeProofSession::from_messages(first.domain(), first.statement(), terminal_value_tampered, terminal_value_session, &reason) &&
        native_masked_sumcheck_verify(terminal_value_session, "matmul/masked", 2, 1, Fr(2), verify_spec, nullptr, &reason)) {
        std::cerr << "masked terminal public value tamper was accepted\n";
        return 5;
    }
    std::vector<NativeProofMessage> terminal_commitment_tampered = first.messages();
    if (!mutate_terminal_commitment(terminal_commitment_tampered)) {
        std::cerr << "masked terminal commitment fixture was not found\n";
        return 5;
    }
    NativeProofSession terminal_commitment_session;
    if (NativeProofSession::from_messages(first.domain(), first.statement(), terminal_commitment_tampered, terminal_commitment_session, &reason) &&
        native_masked_sumcheck_verify(terminal_commitment_session, "matmul/masked", 2, 1, Fr(2), verify_spec, nullptr, &reason)) {
        std::cerr << "masked terminal commitment tamper was accepted\n";
        return 5;
    }
    std::vector<NativeProofMessage> mode_tampered_messages = first.messages();
    if (!mutate_initial_mode(mode_tampered_messages)) {
        std::cerr << "masked mode tamper fixture was not found\n";
        return 5;
    }
    NativeProofSession mode_tampered;
    if (NativeProofSession::from_messages(first.domain(), first.statement(), mode_tampered_messages, mode_tampered, &reason) &&
        native_masked_sumcheck_verify(mode_tampered, "matmul/masked", 2, 1, Fr(2), verify_spec, nullptr, &reason)) {
        std::cerr << "masked mode swap was accepted\n";
        return 5;
    }

    NativeProofSession second("native-masked-smoke-v2", statement_fixture());
    const auto second_basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + second.statement_digest());
    const auto second_initial = zkgpt::native_pedersen::commit_fresh(second_basis, Fr(2));
    NativeMaskedTerminalSpec second_spec = product_spec;
    second_spec.initial_opening = second_initial;
    second_spec.initial_commitment = second_initial.commitment;
    second_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(second_basis, Fr(1)),
                                              zkgpt::native_pedersen::commit_fresh(second_basis, Fr(1))};
    NativeMaskedSumcheckResult second_proved;
    if (!native_masked_sumcheck_prove(second, "matmul/masked", 2, {{Fr(1), Fr(1), Fr(1)}}, Fr(2), {Fr(1), Fr(1)}, second_spec, &second_proved, &reason)) {
        std::cerr << "second masked proof failed: " << reason << '\n';
        return 6;
    }
    if (proved.initial_commitment.point.serializeToHexStr() == second_proved.initial_commitment.point.serializeToHexStr()) {
        std::cerr << "per-proof mask commitment was reused\n";
        return 7;
    }
    if (!fixed_v_binding_fixture()) return 8;
    std::cout << "native_masked_sumcheck_smoke: PASS\n";
    return 0;
}
