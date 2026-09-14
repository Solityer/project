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

int main() {
    initPairing(mcl::BN254);
    std::string reason;

    NativeMaskedTerminalSpec product_spec;
    product_spec.kind = NativeMaskedTerminalKind::Product2;
    NativeProofSession first("native-masked-smoke-v1", statement_fixture());
    NativeMaskedSumcheckResult proved;
    if (!native_masked_matmul_product_prove(
            first, "matmul/masked", {{Fr(1), Fr(1), Fr(1)}}, Fr(2),
            {Fr(1), Fr(1)}, &proved, &reason) ||
        !native_masked_matmul_product_verify(
            first, "matmul/masked", 1, Fr(2), nullptr, &reason)) {
        std::cerr << "masked product honest path failed: " << reason << '\n';
        return 1;
    }
    if (proved.terminal_values.size() != 2 ||
        proved.terminal_values[0] == Fr(1) ||
        proved.terminal_values[1] == Fr(1)) {
        std::cerr << "masked terminal exposed the witness values\n";
        return 2;
    }
    if (native_masked_matmul_product_verify(
            first, "matmul/masked", 1, Fr(3), nullptr, &reason)) {
        std::cerr << "masked public-claim tamper was accepted\n";
        return 3;
    }

    NativeMaskedTerminalSpec gkr_spec;
    gkr_spec.kind = NativeMaskedTerminalKind::GkrPredicate;
    gkr_spec.gkr_coefficients = {Fr(0), Fr(1), Fr(0), Fr(0), Fr(0)};
    NativeProofSession gkr("native-masked-gkr-smoke-v1", statement_fixture());
    NativeMaskedSumcheckProver gkr_prover(
        gkr, "gkr/phase1/masked", 2, Fr(12), gkr_spec);
    Fr gkr_challenge;
    if (!gkr_prover.append_round({Fr(4), Fr(8), Fr(99)}, &gkr_challenge,
                                 &reason)) {
        std::cerr << "masked GKR round failed: " << reason << '\n';
        return 4;
    }
    const Fr gkr_unmasked_final =
        Fr(4) * ((gkr_challenge - Fr(1)) * (gkr_challenge - Fr(2)) / Fr(2)) +
        Fr(8) * ((gkr_challenge - Fr(0)) * (gkr_challenge - Fr(2)) / Fr(-1)) +
        Fr(99) * ((gkr_challenge - Fr(0)) * (gkr_challenge - Fr(1)) / Fr(2));
    if (!gkr_prover.finish({Fr(2), Fr(3), Fr(5),
                             gkr_unmasked_final / Fr(3)}, nullptr, &reason) ||
        !native_masked_gkr_quadratic_verify(
            gkr, "gkr/phase1/masked", 1, Fr(12),
            gkr_spec.gkr_coefficients, nullptr, &reason)) {
        std::cerr << "masked GKR honest path failed: " << reason << '\n';
        return 4;
    }

    NativeProofSession second("native-masked-smoke-v1", statement_fixture());
    if (!native_masked_matmul_product_prove(
            second, "matmul/masked", {{Fr(1), Fr(1), Fr(1)}}, Fr(2),
            {Fr(1), Fr(1)}, nullptr, &reason)) {
        std::cerr << "second masked proof failed: " << reason << '\n';
        return 5;
    }
    std::string first_commitment;
    std::string second_commitment;
    for (const NativeProofMessage &message : first.messages())
        if (message.label == "matmul/masked/mask/0/commitment")
            first_commitment = message.payload;
    for (const NativeProofMessage &message : second.messages())
        if (message.label == "matmul/masked/mask/0/commitment")
            second_commitment = message.payload;
    if (first_commitment.empty() || second_commitment.empty() ||
        first_commitment == second_commitment) {
        std::cerr << "per-proof mask commitment was reused\n";
        return 6;
    }
    std::cout << "native_masked_sumcheck_smoke: PASS\n";
    return 0;
}
