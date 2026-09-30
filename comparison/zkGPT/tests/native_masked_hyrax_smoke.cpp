#include "hyrax.hpp"
#include "proof_session.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

std::vector<Fr> mle_basis(const std::vector<Fr>& point) {
    std::vector<Fr> basis{Fr(1) - point[0], point[0]};
    for (std::size_t bit = 1; bit < point.size(); ++bit) {
        const std::size_t old_size = std::size_t(1) << bit;
        basis.resize(old_size * 2);
        for (std::size_t index = old_size; index-- > 0;) {
            const Fr previous = basis[index];
            basis[index] = previous * (Fr(1) - point[bit]);
            basis[index + old_size] = previous * point[bit];
        }
    }
    return basis;
}

NativeProofStatement make_statement(const G1& H, const std::vector<G1>& g, const std::vector<Fr>& basis) {
    const std::string zeros = "0000000000000000000000000000000000000000000000000000000000000011";
    return NativeProofStatement{
        zeros, "mcl-bn254-fr-v1", zeros, zeros, zeros, zeros, zeros, zeros, NativeProofStatement::basis_digest_for(H, g, basis, g.size()), g.size(), zeros};
}

bool verify(const NativeProofSession& session, const G1& C, const hyrax::MaskedMleOpening& claim, const std::string& basis_id, const std::vector<G1>& g,
            const std::vector<Fr>& point, std::string& reason) {
    return hyrax::verify_masked_mle_opening_with_session(C, claim.evaluation_mask_commitment, claim.masked_evaluation, basis_id, g, point, session,
                                                         "masked/opening", &reason);
}

}

int main() {
    initPairing(mcl::BN254);
    const std::string basis_id = "00000000000000000000000000000000000000000000000000000000000000ab";
    const std::vector<Fr> point{Fr(2), Fr(3)};
    const std::vector<Fr> values{Fr(5), Fr(7), Fr(11), Fr(13)};
    std::vector<G1> generators(values.size());
    for (std::size_t index = 0; index < generators.size(); ++index) hashAndMapToG1(generators[index], "masked-hyrax-smoke/G/" + std::to_string(index));
    const hyrax::PedersenMleCommitment commitment = hyrax::pedersen_commit_mle(generators, values, basis_id);
    const std::vector<Fr> basis = mle_basis(point);
    const NativeProofStatement statement = make_statement(commitment.blinding_generator, generators, basis);

    NativeProofSession first("masked-hyrax-smoke-v1", statement);
    hyrax::MaskedMleOpening first_claim;
    std::string reason;
    if (!hyrax::open_masked_mle_with_session(commitment, generators, point, values, first, "masked/opening", first_claim, &reason) ||
        !verify(first, commitment.commitment, first_claim, basis_id, generators, point, reason)) {
        std::cerr << "honest masked opening rejected: " << reason << '\n';
        return 1;
    }
    Fr raw_evaluation = 0;
    for (std::size_t index = 0; index < values.size(); ++index) raw_evaluation += basis[index] * values[index];
    if (first_claim.masked_evaluation == raw_evaluation) {
        std::cerr << "evaluation mask was not fresh\n";
        return 2;
    }
    for (const NativeProofMessage& message : first.messages()) {
        if (message.label.find("terminal") != std::string::npos || message.label.find("eval") != std::string::npos) {
            std::cerr << "masked opening leaked a raw witness marker\n";
            return 3;
        }
    }

    NativeProofSession second("masked-hyrax-smoke-v1", statement);
    hyrax::MaskedMleOpening second_claim;
    if (!hyrax::open_masked_mle_with_session(commitment, generators, point, values, second, "masked/opening", second_claim, &reason) ||
        !verify(second, commitment.commitment, second_claim, basis_id, generators, point, reason) || first.digest() == second.digest()) {
        std::cerr << "fresh masked opening check failed: " << reason << '\n';
        return 4;
    }

    hyrax::MaskedMleOpening tampered_claim = first_claim;
    tampered_claim.masked_evaluation += Fr(1);
    if (verify(first, commitment.commitment, tampered_claim, basis_id, generators, point, reason)) {
        std::cerr << "masked claim tamper was accepted\n";
        return 5;
    }
    std::vector<Fr> wrong_point = point;
    wrong_point[0] += Fr(1);
    if (verify(first, commitment.commitment, first_claim, basis_id, generators, wrong_point, reason)) {
        std::cerr << "evaluation point tamper was accepted\n";
        return 6;
    }
    std::cout << "native_masked_hyrax_smoke: PASS\n";
    return 0;
}
