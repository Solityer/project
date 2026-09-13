#include "native_protocol.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;

static NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "0000000000000000000000000000000000000000000000000000000000000021",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000022",
        "0000000000000000000000000000000000000000000000000000000000000023",
        "0000000000000000000000000000000000000000000000000000000000000024",
        "0000000000000000000000000000000000000000000000000000000000000025",
        "0000000000000000000000000000000000000000000000000000000000000026",
        "0000000000000000000000000000000000000000000000000000000000000027",
        "0000000000000000000000000000000000000000000000000000000000000028",
        4,
        "0000000000000000000000000000000000000000000000000000000000000021"};
}

static void append_product_round(NativeProofSession &session,
                                 const std::string &label, std::size_t round,
                                 const Fr &constant) {
    session.absorb_fields(label + "/poly/" + std::to_string(round),
                          std::vector<F>{Fr(0), Fr(0), constant});
    session.challenge(label + "/r/" + std::to_string(round));
}

int main() {
    initPairing(mcl::BN254);
    std::string reason;

    NativeProofSession product("native-matmul-session-v1", statement_fixture());
    const Fr initial = 4;
    product.absorb_field("matmul/claim", initial);
    append_product_round(product, "matmul", 0, Fr(2));
    append_product_round(product, "matmul", 1, Fr(1));
    product.absorb_fields("matmul/claims", std::vector<F>{Fr(1), Fr(0), Fr(1), Fr(0)});
    Fr left, right;
    if (!verify_product_sumcheck_session(product, "matmul", initial, 2,
                                         &left, &right, &reason) ||
        left != Fr(1) || right != Fr(1)) {
        std::cerr << "matrix Sumcheck replay failed: " << reason << '\n';
        return 1;
    }
    if (verify_product_sumcheck_session(product, "matmul", initial + Fr(1), 2,
                                        nullptr, nullptr, &reason)) {
        std::cerr << "matrix public claim tamper was accepted\n";
        return 2;
    }

    NativeProofSession quadratic("native-gkr-session-v1", statement_fixture());
    append_product_round(quadratic, "gkr/phase1", 0, Fr(2));
    append_product_round(quadratic, "gkr/phase1", 1, Fr(1));
    quadratic.absorb_fields("gkr/phase1/claims", std::vector<F>{Fr(7), Fr(9)});
    Fr final_sum;
    std::vector<F> terminal_claims;
    if (!verify_quadratic_sumcheck_session(quadratic, "gkr/phase1", Fr(4), 2,
                                           &final_sum, &terminal_claims, &reason) ||
        final_sum != Fr(1) || terminal_claims.size() != 2) {
        std::cerr << "GKR Sumcheck replay failed: " << reason << '\n';
        return 3;
    }

    NativeProofSession reordered("native-matmul-session-v1", statement_fixture());
    reordered.absorb_field("matmul/claim", initial);
    reordered.challenge("matmul/r/0");
    reordered.absorb_fields("matmul/poly/0", std::vector<F>{Fr(0), Fr(0), Fr(2)});
    reordered.absorb_fields("matmul/poly/1", std::vector<F>{Fr(0), Fr(0), Fr(1)});
    reordered.challenge("matmul/r/1");
    reordered.absorb_fields("matmul/claims", std::vector<F>{Fr(1), Fr(0), Fr(1), Fr(0)});
    if (verify_product_sumcheck_session(reordered, "matmul", initial, 2,
                                        nullptr, nullptr, &reason)) {
        std::cerr << "matrix message reorder was accepted\n";
        return 4;
    }

    NativeProofSession inserted("native-matmul-session-v1", statement_fixture());
    inserted.absorb_field("matmul/claim", initial);
    append_product_round(inserted, "matmul", 0, Fr(2));
    inserted.absorb_field("matmul/unexpected", Fr(9));
    append_product_round(inserted, "matmul", 1, Fr(1));
    inserted.absorb_fields("matmul/claims", std::vector<F>{Fr(1), Fr(0), Fr(1), Fr(0)});
    if (verify_product_sumcheck_session(inserted, "matmul", initial, 2,
                                         nullptr, nullptr, &reason)) {
        std::cerr << "matrix message insertion was accepted\n";
        return 5;
    }
    if (!verify_gkr_predicate_alpha_beta_transition(
            Fr(17), Fr(17), Fr(2), Fr(3), Fr(5), Fr(7), Fr(31), &reason) ||
        verify_gkr_predicate_alpha_beta_transition(
            Fr(17), Fr(17), Fr(2), Fr(3), Fr(5), Fr(7), Fr(30), &reason)) {
        std::cerr << "GKR predicate/alpha-beta transition check failed\n";
        return 6;
    }
    std::cout << "native_gkr_matmul_session_smoke: PASS\n";
    return 0;
}
