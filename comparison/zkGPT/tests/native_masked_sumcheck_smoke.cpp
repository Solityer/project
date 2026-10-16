#include "native_masked_sumcheck.hpp"

#include <mcl/bn256.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
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

static std::shared_ptr<const NativeMaskedPublicProvider> constant_public_provider(const F& coefficient) {
    auto provider = std::make_shared<NativeMaskedPublicProvider>();
    provider->size = 1;
    provider->value_at = [coefficient](std::uint64_t index) {
        if (index != 0) throw std::out_of_range("constant public coefficient index is outside its domain");
        return coefficient;
    };
    provider->evaluate_mle = [coefficient](const std::vector<F>& point) {
        if (!point.empty()) throw std::invalid_argument("constant public coefficient expects the empty point");
        return coefficient;
    };
    return provider;
}

static bool product3_public_constant_factor_fixture(const F& coefficient, const F& left, const F& right, const std::string& label) {
    std::string reason;
    NativeProofSession session("native-product3-public-constant-v1", statement_fixture());
    const auto basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest());
    const auto initial = zkgpt::native_pedersen::commit_fresh(basis, F_ZERO);

    NativeMaskedTerminalSpec prover_spec;
    prover_spec.kind = NativeMaskedTerminalKind::Product3Public;
    prover_spec.public_provider = constant_public_provider(coefficient);
    prover_spec.has_external_binding = true;
    prover_spec.public_initial_claim = true;
    prover_spec.initial_opening = initial;
    prover_spec.initial_commitment = initial.commitment;
    prover_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(basis, left),
                                              zkgpt::native_pedersen::commit_fresh(basis, right)};

    NativeMaskedSumcheckProver prover(session, label, 3, F_ZERO, prover_spec);
    NativeMaskedSumcheckResult proved;
    if (!prover.finish({left, right}, &proved, &reason)) {
        std::cerr << "Product3Public constant-factor prover failed: " << reason << '\n';
        return false;
    }

    NativeMaskedTerminalSpec verify_spec = prover_spec;
    verify_spec.initial_opening = {};
    verify_spec.unmasked_terminal_openings.clear();
    verify_spec.terminal_commitments = proved.terminal_commitments;
    NativeMaskedSumcheckResult verified;
    if (!native_masked_sumcheck_verify(session, label, 3, 0, F_ZERO, verify_spec, &verified, &reason)) {
        std::cerr << "Product3Public constant-factor verifier failed: " << reason << '\n';
        return false;
    }
    return true;
}

static bool product3_public_degree3_reference_fixture() {
    std::string reason;
    const F beta(7);
    auto provider = std::make_shared<NativeMaskedPublicProvider>();
    provider->size = 2;
    provider->value_at = [beta](std::uint64_t index) {
        if (index > 1) throw std::out_of_range("equality coefficient index is outside its domain");
        return index == 0 ? F_ONE - beta : beta;
    };
    provider->evaluate_mle = [beta](const std::vector<F>& point) {
        if (point.size() != 1) throw std::invalid_argument("one-coordinate equality coefficient has wrong point arity");
        return (F_ONE - beta) * (F_ONE - point[0]) + beta * point[0];
    };

    NativeProofSession session("native-product3-public-degree3-v1", statement_fixture());
    const auto basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest());
    const auto initial = zkgpt::native_pedersen::commit_fresh(basis, F_ZERO);
    NativeMaskedTerminalSpec prover_spec;
    prover_spec.kind = NativeMaskedTerminalKind::Product3Public;
    prover_spec.public_provider = provider;
    prover_spec.has_external_binding = true;
    prover_spec.public_initial_claim = true;
    prover_spec.initial_opening = initial;
    prover_spec.initial_commitment = initial.commitment;
    prover_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(basis, F_ZERO),
                                              zkgpt::native_pedersen::commit_fresh(basis, F_ZERO)};

    // Dense reference for sum_x eq(beta,x) m(x)(m(x)-1), m=[0,1].
    // Its one-variable round polynomial is cubic, while its Boolean-cube sum is zero.
    std::vector<F> round_poly(4);
    for (std::size_t i = 0; i < round_poly.size(); ++i) {
        const F t(static_cast<int>(i));
        const F beta_at_t = (F_ONE - beta) * (F_ONE - t) + beta * t;
        round_poly[i] = beta_at_t * t * (t - F_ONE);
    }
    if (round_poly[0] + round_poly[1] != F_ZERO) {
        std::cerr << "Product3Public dense reference does not sum to its Boolean-cube claim\n";
        return false;
    }

    NativeMaskedSumcheckProver prover(session, "booleanity/weighted/product3", 3, F_ZERO, prover_spec);
    F challenge;
    if (!prover.append_round(round_poly, &challenge, &reason)) {
        std::cerr << "Product3Public degree-three round failed: " << reason << '\n';
        return false;
    }
    const F beta_at_challenge = (F_ONE - beta) * (F_ONE - challenge) + beta * challenge;
    if (beta_at_challenge.isZero()) {
        std::cerr << "Product3Public reference challenge hit the zero public factor\n";
        return false;
    }
    prover_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(basis, challenge),
                                              zkgpt::native_pedersen::commit_fresh(basis, challenge - F_ONE)};
    if (!prover.set_terminal_spec(prover_spec, &reason)) {
        std::cerr << "Product3Public degree-three terminal spec failed: " << reason << '\n';
        return false;
    }
    NativeMaskedSumcheckResult proved;
    if (!prover.finish({challenge, challenge - F_ONE}, &proved, &reason)) {
        std::cerr << "Product3Public degree-three prover failed: " << reason << '\n';
        return false;
    }
    if (proved.unmasked_claim_opening.value != beta_at_challenge * challenge * (challenge - F_ONE)) {
        std::cerr << "Product3Public terminal claim differs from dense reference\n";
        return false;
    }

    NativeMaskedTerminalSpec verify_spec = prover_spec;
    verify_spec.initial_opening = {};
    verify_spec.unmasked_terminal_openings.clear();
    verify_spec.terminal_commitments = proved.terminal_commitments;
    if (!native_masked_sumcheck_verify(session, "booleanity/weighted/product3", 3, 1, F_ZERO, verify_spec, nullptr, &reason)) {
        std::cerr << "Product3Public degree-three verifier failed: " << reason << '\n';
        return false;
    }

    auto tampered_provider = std::make_shared<NativeMaskedPublicProvider>();
    tampered_provider->size = 2;
    tampered_provider->value_at = [](std::uint64_t index) {
        if (index > 1) throw std::out_of_range("tampered coefficient index is outside its domain");
        return F_ZERO;
    };
    tampered_provider->evaluate_mle = [](const std::vector<F>& point) {
        if (point.size() != 1) throw std::invalid_argument("tampered coefficient point arity mismatch");
        return F_ZERO;
    };
    verify_spec.public_provider = tampered_provider;
    if (native_masked_sumcheck_verify(session, "booleanity/weighted/product3", 3, 1, F_ZERO, verify_spec, nullptr, &reason)) {
        std::cerr << "Product3Public coefficient tamper was accepted\n";
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

    NativeProofSession malformed_opening_session(
        "native-masked-unmasked-opening-guard-v1", statement_fixture());
    const auto malformed_opening_basis = zkgpt::native_pedersen::derive_basis(
        "zkgpt/masked-sumcheck/v2/" + malformed_opening_session.statement_digest());
    const auto malformed_initial =
        zkgpt::native_pedersen::commit_fresh(malformed_opening_basis, Fr(2));
    NativeMaskedTerminalSpec malformed_opening_spec = product_spec;
    malformed_opening_spec.initial_opening = malformed_initial;
    malformed_opening_spec.initial_commitment = malformed_initial.commitment;
    malformed_opening_spec.unmasked_terminal_openings = {
        zkgpt::native_pedersen::commit_fresh(malformed_opening_basis, Fr(1)),
        zkgpt::native_pedersen::commit_fresh(malformed_opening_basis, Fr(1))};
    NativeMaskedSumcheckProver malformed_opening_prover(
        malformed_opening_session, "malformed-unmasked-opening/masked", 2,
        Fr(2), malformed_opening_spec);
    if (!malformed_opening_prover.append_round(
            {Fr(1), Fr(1), Fr(1)}, nullptr, &reason)) {
        std::cerr << "malformed-opening guard round setup failed: " << reason << '\n';
        return 8;
    }
    malformed_opening_spec.unmasked_terminal_openings[0] = {};
    if (!malformed_opening_prover.set_terminal_spec(malformed_opening_spec,
                                                    &reason)) {
        std::cerr << "malformed-opening terminal spec setup failed: " << reason << '\n';
        return 8;
    }
    NativeMaskedSumcheckResult malformed_opening_result;
    if (malformed_opening_prover.finish({Fr(1), Fr(1)},
                                        &malformed_opening_result, &reason)) {
        std::cerr << "prover accepted malformed unmasked terminal opening\n";
        return 8;
    }
    if (reason != "masked Sumcheck opening value does not match relation") {
        std::cerr << "malformed opening was rejected before the relation guard: "
                  << reason << '\n';
        return 8;
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
    if (!product3_public_constant_factor_fixture(F_ZERO, Fr(2), Fr(3), "product3-public/zero")) return 9;
    if (!product3_public_constant_factor_fixture(F_ONE, F_ZERO, Fr(3), "product3-public/one")) return 10;
    if (!product3_public_constant_factor_fixture(Fr(7), F_ZERO, Fr(3), "product3-public/general")) return 11;
    if (!product3_public_degree3_reference_fixture()) return 12;
    std::cout << "native_masked_sumcheck_smoke: PASS\n";
    return 0;
}
