#include "gat_proof.hpp"

#include <iostream>

using namespace mcl::bn;
using namespace zkgpt::gat;

namespace {

std::string zeros() {
    return std::string(64, '0');
}

} // namespace

int main() {
    initPairing(mcl::BN254);
    const std::string basis_label = "masked-core-smoke/basis";
    const ProverCommitment com_w = commit_private_segment_for_prover(
        {3, 5, 7, 11}, basis_label, 4);
    const ProverCommitment com_aux = commit_private_segment_for_prover(
        {13, 17, 19, 23}, basis_label, 4);
    const ProverCommitment com_l = commit_private_segment_for_prover(
        {0, 1, 0, 1}, basis_label, 4);
    const std::string basis_id = com_w.wire.basis_id;
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    std::vector<G1> generators(4);
    for (std::size_t index = 0; index < generators.size(); ++index)
        hashAndMapToG1(generators[index],
                       basis_label + "/coordinate/" + std::to_string(index));

    const std::string context_digest = zeros();
    NativeProofStatement statement{
        context_digest, "bn254-fr-v1", zeros(), zeros(), zeros(), zeros(),
        zeros(), zeros(),
        NativeProofStatement::basis_digest_for(
            H, generators, std::vector<Fr>(4, Fr(0)), 4),
        4, context_digest};

    MaskedCoreWitness witness;
    witness.public_input = {Fr(2), Fr(4)};
    witness.w = {Fr(3), Fr(5), Fr(7), Fr(11)};
    witness.aux = {Fr(13), Fr(17), Fr(19), Fr(23)};
    witness.l_qry = {Fr(0), Fr(1), Fr(0), Fr(1)};
    witness.public_output = {Fr(0), Fr(0)};
    witness.gkr_round_polys = {{Fr(0), Fr(0), Fr(0)}};
    witness.gkr_terminal_values = {Fr(0), Fr(0), Fr(0), Fr(0)};
    witness.gkr_coefficients = {Fr(0), Fr(1), Fr(0), Fr(0), Fr(0)};
    witness.matmul_round_polys = {{Fr(2), Fr(2), Fr(2)}};
    witness.matmul_terminal_values = {Fr(1), Fr(2)};
    witness.com_aux = com_aux;
    witness.com_L = com_l;
    witness.has_dynamic_commitments = true;

    MaskedCoreResult result = prove_masked_core(
        statement, com_w.wire, com_w.blinder, basis_id, H, generators,
        witness, context_digest);
    if (!result.generated) {
        std::cerr << "masked core honest path failed: " << result.diagnostic << '\n';
        return 1;
    }
    std::string reason;
    if (!verify_masked_core(
            result.artifact, com_w.wire, basis_id, H, generators,
            witness.public_input, witness.public_output,
            witness.gkr_coefficients, 1, Fr(4), 1, &reason)) {
        std::cerr << "masked core independent verification failed: " << reason << '\n';
        return 2;
    }
    if (result.artifact.has_private_payload()) {
        std::cerr << "masked core artifact contains a private marker\n";
        return 3;
    }
    GatProofArtifact tampered = result.artifact;
    bool changed = false;
    for (NativeProofMessage &message : tampered.native_messages) {
        if (message.label == "matmul/claim") {
            message.payload[message.payload.size() - 1] =
                message.payload.back() == '4' ? '5' : '4';
            changed = true;
            break;
        }
    }
    if (!changed || verify_masked_core(
            tampered, com_w.wire, basis_id, H, generators,
            witness.public_input, witness.public_output,
            witness.gkr_coefficients, 1, Fr(4), 1, &reason)) {
        std::cerr << "masked core claim tamper was accepted\n";
        return 4;
    }
    MaskedCoreResult second = prove_masked_core(
        statement, com_w.wire, com_w.blinder, basis_id, H, generators,
        witness, context_digest);
    if (!second.generated ||
        second.artifact.transcript_digest == result.artifact.transcript_digest) {
        std::cerr << "masked core freshness check failed\n";
        return 5;
    }
    std::cout << "masked_core_smoke: PASS\n";
    return 0;
}
