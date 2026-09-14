#include "native_pedersen_relation.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;
using namespace zkgpt::native_pedersen;

static NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "00000000000000000000000000000000000000000000000000000000000000a1",
        "mcl-bn254-fr-v1",
        "00000000000000000000000000000000000000000000000000000000000000a2",
        "00000000000000000000000000000000000000000000000000000000000000a3",
        "00000000000000000000000000000000000000000000000000000000000000a4",
        "00000000000000000000000000000000000000000000000000000000000000a5",
        "00000000000000000000000000000000000000000000000000000000000000a6",
        "00000000000000000000000000000000000000000000000000000000000000a7",
        "00000000000000000000000000000000000000000000000000000000000000a8",
        4,
        "00000000000000000000000000000000000000000000000000000000000000a1"};
}

static bool restored_with_messages(const NativeProofSession &source,
                                   std::vector<NativeProofMessage> messages,
                                   NativeProofSession &restored,
                                   std::string *reason) {
    return NativeProofSession::from_messages(source.domain(), source.statement(),
                                              messages, restored, reason);
}

int main() {
    initPairing(BN254);
    const PedersenBasis basis = derive_basis("smoke/pedersen");
    NativeProofSession session("native-pedersen-relation-smoke-v1",
                               statement_fixture());
    std::string reason;

    ScalarOpening a = commit_fresh(basis, Fr(2));
    ScalarOpening b = commit_fresh(basis, Fr(3));
    ScalarOpening c = commit_fresh(basis, Fr(6));
    ScalarOpening third = commit_fresh(basis, Fr(3));
    ScalarOpening intermediate = commit_fresh(basis, Fr(6));
    ScalarOpening product3_output = commit_fresh(basis, Fr(18));
    ScalarOpening affine_input = commit_fresh(basis, Fr(5));
    ScalarOpening affine_output = public_affine(basis, affine_input, Fr(2), Fr(7));

    const std::vector<std::pair<std::string, const ScalarOpening *> > openings{
        {"pedersen/01-A", &a},
        {"pedersen/02-B", &b},
        {"pedersen/03-C", &c},
        {"pedersen/04-M", &intermediate},
        {"pedersen/05-D", &product3_output},
        {"pedersen/06-affine-in", &affine_input},
        {"pedersen/07-affine-out", &affine_output},
        {"pedersen/08-third", &third}};
    for (const auto &entry : openings) {
        if (!prove_opening(session, basis, entry.first, *entry.second, nullptr,
                           &reason) ||
            !verify_opening(session, basis, entry.first,
                            entry.second->commitment, &reason)) {
            std::cerr << "opening failed: " << reason << '\n';
            return 1;
        }
    }

    const std::vector<ScalarOpening> linear_openings{a, b};
    const std::vector<Fr> linear_coefficients{Fr(2), Fr(3)};
    if (!prove_hidden_linear(session, basis, "pedersen/10-linear",
                             linear_openings, linear_coefficients, Fr(13),
                             nullptr, &reason) ||
        !verify_hidden_linear(session, basis, "pedersen/10-linear",
                              {a.commitment, b.commitment},
                              linear_coefficients, Fr(13), &reason)) {
        std::cerr << "linear relation failed: " << reason << '\n';
        return 2;
    }
    if (verify_hidden_linear(session, basis, "pedersen/10-linear",
                             {a.commitment, b.commitment}, {Fr(2), Fr(4)},
                             Fr(13), &reason) ||
        verify_hidden_linear(session, basis, "pedersen/10-linear",
                             {a.commitment, b.commitment},
                             linear_coefficients, Fr(14), &reason)) {
        std::cerr << "linear coefficient/public-value tamper was accepted\n";
        return 3;
    }

    const MultiplicationLinks multiplication_links{
        "pedersen/01-A", "pedersen/02-B", "pedersen/03-C"};
    if (!prove_multiplication(session, basis, "pedersen/20-mul", a, b, c,
                              multiplication_links, nullptr, &reason) ||
        !verify_multiplication(session, basis, "pedersen/20-mul", a.commitment,
                               b.commitment, c.commitment, multiplication_links,
                               &reason)) {
        std::cerr << "multiplication failed: " << reason << '\n';
        return 4;
    }
    ScalarCommitment wrong_c = c.commitment;
    wrong_c.point += basis.G;
    if (verify_multiplication(session, basis, "pedersen/20-mul", a.commitment,
                              b.commitment, wrong_c, multiplication_links,
                              &reason)) {
        std::cerr << "C commitment tamper was accepted\n";
        return 5;
    }
    ScalarCommitment wrong_a = a.commitment;
    wrong_a.point += basis.G;
    if (verify_opening(session, basis, "pedersen/01-A", wrong_a, &reason)) {
        std::cerr << "A commitment tamper was accepted\n";
        return 6;
    }
    ScalarCommitment wrong_b = b.commitment;
    wrong_b.point += basis.G;
    if (verify_multiplication(session, basis, "pedersen/20-mul", a.commitment,
                              wrong_b, c.commitment, multiplication_links,
                              &reason)) {
        std::cerr << "B commitment tamper was accepted\n";
        return 7;
    }
    const PedersenBasis wrong_basis = derive_basis("smoke/pedersen-other");
    if (verify_opening(session, wrong_basis, "pedersen/01-A", a.commitment,
                       &reason)) {
        std::cerr << "generator/basis tamper was accepted\n";
        return 8;
    }
    PedersenBasis tampered_generators = basis;
    tampered_generators.H += basis.G;
    if (verify_opening(session, tampered_generators, "pedersen/01-A",
                       a.commitment, &reason)) {
        std::cerr << "direct generator tamper was accepted\n";
        return 9;
    }

    const Product3Links product3_links{
        {"pedersen/01-A", "pedersen/02-B", "pedersen/04-M"},
        {"pedersen/04-M", "pedersen/08-third", "pedersen/05-D"}};
    if (!prove_product3(session, basis, "pedersen/30-product3", a, b, third,
                        intermediate, product3_output, product3_links, nullptr,
                        &reason) ||
        !verify_product3(session, basis, "pedersen/30-product3", a.commitment,
                         b.commitment, third.commitment, intermediate.commitment,
                         product3_output.commitment, product3_links, &reason)) {
        std::cerr << "product3 failed: " << reason << '\n';
        return 10;
    }

    if (!prove_public_affine(session, basis, "pedersen/40-affine",
                             affine_input, affine_output, Fr(2), Fr(7), nullptr,
                             &reason) ||
        !verify_public_affine(session, basis, "pedersen/40-affine",
                              affine_input.commitment, affine_output.commitment,
                              Fr(2), Fr(7), &reason)) {
        std::cerr << "affine relation failed: " << reason << '\n';
        return 11;
    }
    if (verify_public_affine(session, basis, "pedersen/40-affine",
                             affine_input.commitment, affine_output.commitment,
                             Fr(2), Fr(8), &reason)) {
        std::cerr << "affine public-value tamper was accepted\n";
        return 12;
    }

    const std::string path = "/tmp/zkgpt-native-pedersen-relation-smoke.bin";
    if (!session.write(path, &reason)) {
        std::cerr << "session write failed: " << reason << '\n';
        return 13;
    }
    NativeProofSession restored;
    if (!NativeProofSession::read(path, restored, &reason) ||
        !verify_hidden_linear(restored, basis, "pedersen/10-linear",
                              {a.commitment, b.commitment},
                              linear_coefficients, Fr(13), &reason) ||
        !verify_product3(restored, basis, "pedersen/30-product3", a.commitment,
                         b.commitment, third.commitment, intermediate.commitment,
                         product3_output.commitment, product3_links, &reason)) {
        std::cerr << "serialized independent verification failed: " << reason
                  << '\n';
        std::remove(path.c_str());
        return 14;
    }

    std::vector<NativeProofMessage> response_tamper = restored.messages();
    for (NativeProofMessage &message : response_tamper) {
        if (message.label != "pedersen/20-mul/response") continue;
        if (message.payload.empty()) return 12;
        message.payload.back() = message.payload.back() == '0' ? '1' : '0';
    }
    NativeProofSession response_session;
    if (restored_with_messages(restored, response_tamper, response_session,
                               &reason) &&
        verify_multiplication(response_session, basis, "pedersen/20-mul",
                              a.commitment, b.commitment, c.commitment,
                              multiplication_links, &reason)) {
        std::cerr << "response tamper was accepted\n";
        std::remove(path.c_str());
        return 16;
    }

    std::vector<NativeProofMessage> challenge_tamper = restored.messages();
    bool changed_challenge = false;
    for (NativeProofMessage &message : challenge_tamper) {
        if (message.label != "challenge/pedersen/20-mul/challenge") continue;
        if (message.payload.empty()) return 14;
        message.payload.back() = message.payload.back() == '0' ? '1' : '0';
        changed_challenge = true;
    }
    NativeProofSession challenge_session;
    if (!changed_challenge ||
        restored_with_messages(restored, challenge_tamper, challenge_session,
                               &reason)) {
        std::cerr << "challenge tamper was accepted\n";
        std::remove(path.c_str());
        return 18;
    }

    NativeProofSession first_fresh("native-pedersen-freshness-v1",
                                   statement_fixture());
    NativeProofSession second_fresh("native-pedersen-freshness-v1",
                                    statement_fixture());
    OpeningProof first_opening;
    OpeningProof second_opening;
    if (!prove_opening(first_fresh, basis, "pedersen/fresh", a,
                       &first_opening, &reason) ||
        !prove_opening(second_fresh, basis, "pedersen/fresh", a,
                       &second_opening, &reason) ||
        first_opening.commitment_message == second_opening.commitment_message ||
        first_opening.challenge == second_opening.challenge) {
        std::cerr << "fresh randomness was reused\n";
        std::remove(path.c_str());
        return 19;
    }

    std::remove(path.c_str());
    std::cout << "native_pedersen_relation_smoke: PASS\n";
    return 0;
}
