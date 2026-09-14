#include "native_masked_sumcheck.hpp"

#include <mcl/bn256.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;

static NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "0000000000000000000000000000000000000000000000000000000000000121",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000122",
        "0000000000000000000000000000000000000000000000000000000000000123",
        "0000000000000000000000000000000000000000000000000000000000000124",
        "0000000000000000000000000000000000000000000000000000000000000125",
        "0000000000000000000000000000000000000000000000000000000000000126",
        "0000000000000000000000000000000000000000000000000000000000000127",
        "0000000000000000000000000000000000000000000000000000000000000128",
        4,
        "0000000000000000000000000000000000000000000000000000000000000121"};
}

static bool mutate_initial_mode(std::vector<NativeProofMessage> &messages) {
    for (NativeProofMessage &message : messages) {
        if (message.label != "matmul/masked/initial/mode") continue;
        message.payload = "with-mask-v2";
        return true;
    }
    return false;
}

int main() {
    initPairing(mcl::BN254);
    std::string reason;

    NativeProofSession first("native-masked-smoke-v2", statement_fixture());
    const auto first_basis = zkgpt::native_pedersen::derive_basis(
        "zkgpt/masked-sumcheck/v2/" + first.statement_digest());
    const auto first_initial =
        zkgpt::native_pedersen::commit_fresh(first_basis, Fr(2));
    NativeMaskedTerminalSpec product_spec;
    product_spec.kind = NativeMaskedTerminalKind::Product2;
    product_spec.has_external_binding = true;
    product_spec.public_initial_claim = true;
    product_spec.initial_opening = first_initial;
    product_spec.initial_commitment = first_initial.commitment;
    product_spec.unmasked_terminal_openings = {
        zkgpt::native_pedersen::commit_fresh(first_basis, Fr(1)),
        zkgpt::native_pedersen::commit_fresh(first_basis, Fr(1))};
    NativeMaskedSumcheckResult proved;
    if (!native_masked_sumcheck_prove(
            first, "matmul/masked", 2, {{Fr(1), Fr(1), Fr(1)}}, Fr(2),
            {Fr(1), Fr(1)}, product_spec, &proved, &reason)) {
        std::cerr << "masked product honest path failed: " << reason << '\n';
        return 1;
    }
    if (proved.terminal_values.size() != 2 ||
        proved.terminal_values[0] == Fr(1) ||
        proved.terminal_values[1] == Fr(1)) {
        std::cerr << "masked terminal exposed the witness values\n";
        return 2;
    }
    NativeMaskedTerminalSpec verify_spec = product_spec;
    verify_spec.initial_opening = {};
    verify_spec.terminal_commitments = proved.terminal_commitments;
    if (!native_masked_sumcheck_verify(
            first, "matmul/masked", 2, 1, Fr(2), verify_spec, nullptr,
            &reason)) {
        std::cerr << "masked product independent verify failed: " << reason
                  << '\n';
        return 3;
    }
    if (native_masked_sumcheck_verify(
            first, "matmul/masked", 2, 1, Fr(3), verify_spec, nullptr,
            &reason)) {
        std::cerr << "masked public-claim tamper was accepted\n";
        return 4;
    }
    std::vector<NativeProofMessage> mode_tampered_messages = first.messages();
    if (!mutate_initial_mode(mode_tampered_messages)) {
        std::cerr << "masked mode tamper fixture was not found\n";
        return 5;
    }
    NativeProofSession mode_tampered;
    if (NativeProofSession::from_messages(
            first.domain(), first.statement(), mode_tampered_messages,
            mode_tampered, &reason) &&
        native_masked_sumcheck_verify(mode_tampered, "matmul/masked", 2, 1,
                                      Fr(2), verify_spec, nullptr, &reason)) {
        std::cerr << "masked mode swap was accepted\n";
        return 5;
    }

    NativeProofSession second("native-masked-smoke-v2", statement_fixture());
    const auto second_basis = zkgpt::native_pedersen::derive_basis(
        "zkgpt/masked-sumcheck/v2/" + second.statement_digest());
    const auto second_initial =
        zkgpt::native_pedersen::commit_fresh(second_basis, Fr(2));
    NativeMaskedTerminalSpec second_spec = product_spec;
    second_spec.initial_opening = second_initial;
    second_spec.initial_commitment = second_initial.commitment;
    second_spec.unmasked_terminal_openings = {
        zkgpt::native_pedersen::commit_fresh(second_basis, Fr(1)),
        zkgpt::native_pedersen::commit_fresh(second_basis, Fr(1))};
    NativeMaskedSumcheckResult second_proved;
    if (!native_masked_sumcheck_prove(
            second, "matmul/masked", 2, {{Fr(1), Fr(1), Fr(1)}}, Fr(2),
            {Fr(1), Fr(1)}, second_spec, &second_proved, &reason)) {
        std::cerr << "second masked proof failed: " << reason << '\n';
        return 6;
    }
    if (proved.initial_commitment.point.serializeToHexStr() ==
        second_proved.initial_commitment.point.serializeToHexStr()) {
        std::cerr << "per-proof mask commitment was reused\n";
        return 7;
    }
    std::cout << "native_masked_sumcheck_smoke: PASS\n";
    return 0;
}
