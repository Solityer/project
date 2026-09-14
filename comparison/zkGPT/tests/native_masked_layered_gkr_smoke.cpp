#include "verifier.hpp"

#include <mcl/bn256.hpp>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

using namespace mcl::bn;

namespace {

NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "0000000000000000000000000000000000000000000000000000000000000221",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000222",
        "0000000000000000000000000000000000000000000000000000000000000223",
        "0000000000000000000000000000000000000000000000000000000000000224",
        "0000000000000000000000000000000000000000000000000000000000000225",
        "0000000000000000000000000000000000000000000000000000000000000226",
        "0000000000000000000000000000000000000000000000000000000000000227",
        "0000000000000000000000000000000000000000000000000000000000000228",
        4,
        "0000000000000000000000000000000000000000000000000000000000000221"};
}

layeredCircuit make_circuit() {
    layeredCircuit circuit;
    circuit.init(0, 3);

    circuit.circuit[0].ty = layerType::INPUT;
    circuit.circuit[0].size = 2;
    circuit.circuit[0].zero_start_id = 2;
    circuit.circuit[0].bit_length = 1;

    circuit.circuit[1].ty = layerType::FCONN;
    circuit.circuit[1].size = 2;
    circuit.circuit[1].zero_start_id = 2;
    circuit.circuit[1].bit_length = 1;
    circuit.circuit[1].need_phase2 = true;

    circuit.circuit[2].ty = layerType::ADD_BIAS;
    circuit.circuit[2].size = 2;
    circuit.circuit[2].zero_start_id = 2;
    circuit.circuit[2].bit_length = 1;
    circuit.circuit[2].uni_gates.emplace_back(0, 0, 1, 1);
    circuit.circuit[2].uni_gates.emplace_back(1, 1, 1, 1);
    circuit.initSubset();
    return circuit;
}

NativeLayeredGkrWitness make_witness() {
    NativeLayeredGkrWitness witness;
    witness.layer_values = {{Fr(3), Fr(5)}, {Fr(11), Fr(27)},
                            {Fr(11), Fr(27)}};
    NativeGkrMatmulWitness matmul;
    matmul.shape = {1, 2, 2};
    matmul.left = {Fr(3), Fr(5)};
    matmul.right = {Fr(2), Fr(1), Fr(4), Fr(3)};
    witness.matmuls.push_back(matmul);
    return witness;
}

bool mutate_input_binding_payload(std::vector<NativeProofMessage> &messages) {
    for (auto message_it = messages.rbegin(); message_it != messages.rend(); ++message_it) {
        NativeProofMessage &message = *message_it;
        if (message.kind != NativeProofMessageKind::Field ||
            message.label != "gkr/input/bind/response")
            continue;
        if (message.payload.empty()) return false;
        message.payload.back() = message.payload.back() == '0' ? '1' : '0';
        return true;
    }
    return false;
}

bool mutate_mode_to_zero(std::vector<NativeProofMessage> &messages) {
    for (NativeProofMessage &message : messages) {
        if (message.kind != NativeProofMessageKind::Data ||
            message.label.find("/initial/mode") == std::string::npos)
            continue;
        message.payload = "zero-mask-v2";
        return true;
    }
    return false;
}

bool drop_masked_initial_opening(std::vector<NativeProofMessage> &messages) {
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const std::string marker = "/initial/masked/meta";
        if (messages[index].label.size() < marker.size() ||
            messages[index].label.compare(messages[index].label.size() - marker.size(),
                                          marker.size(), marker) != 0)
            continue;
        if (index + 4 > messages.size()) return false;
        messages.erase(messages.begin() + static_cast<std::ptrdiff_t>(index),
                       messages.begin() + static_cast<std::ptrdiff_t>(index + 4));
        for (std::size_t sequence = 0; sequence < messages.size(); ++sequence)
            messages[sequence].sequence = sequence;
        return true;
    }
    return false;
}

bool replace_intermediate_claim(std::vector<NativeProofMessage> &messages) {
    std::string replacement;
    for (const NativeProofMessage &message : messages) {
        if (message.label.find("/initial/opening/meta") != std::string::npos) {
            replacement = message.payload;
            break;
        }
    }
    if (replacement.empty()) return false;
    for (NativeProofMessage &message : messages) {
        if (message.label.find("/next-claim/unmasked/output/meta") ==
            std::string::npos)
            continue;
        message.payload = replacement;
        return true;
    }
    return false;
}

} // namespace

int main() {
    initPairing(BN254);
    const layeredCircuit circuit = make_circuit();
    const NativeLayeredGkrWitness witness = make_witness();
    const std::vector<Fr> output{Fr(11), Fr(27)};
    const std::vector<NativeGkrMatmulShape> descriptors{{1, 2, 2}};
    std::string reason;
    NativeGkrInputBinding produced_binding;
    NativeGkrInputBinding verified_binding;
    NativeGkrProverResult prover_result;

    NativeProofSession session("masked-layered-gkr-smoke-v1", statement_fixture());
    const std::size_t messages_before_prover_result = session.messages().size();
    if (!prove_masked_gkr_transcript(session, circuit, witness, output, &reason,
                                     &produced_binding, &prover_result)) {
        std::cerr << "prove failed: " << reason << '\n';
        return 1;
    }
    assert(session.messages().size() > messages_before_prover_result);
    assert(prover_result.has_unmasked_input_claim_opening);
    const auto basis = zkgpt::native_pedersen::derive_basis(
        "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
    assert(prover_result.unmasked_input_claim_opening.valid(basis, &reason));
    assert(prover_result.unmasked_input_claim_opening.commitment.point ==
           produced_binding.unmasked_claim_commitment.point);
    assert(prover_result.input_binding.unmasked_claim_commitment.point ==
           produced_binding.unmasked_claim_commitment.point);
    if (!verify_masked_gkr_transcript(session, circuit, output, descriptors,
                                      &reason, &verified_binding)) {
        std::cerr << "verify failed: " << reason << '\n';
        return 1;
    }
    assert(produced_binding.point == verified_binding.point);
    assert(produced_binding.claim_commitment.point ==
           verified_binding.claim_commitment.point);
    assert(produced_binding.unmasked_claim_commitment.point ==
           verified_binding.unmasked_claim_commitment.point);
    assert(produced_binding.mask_commitment.point ==
           verified_binding.mask_commitment.point);
    bool saw_private_scope = false;
    for (const NativeProofMessage &message : session.messages()) {
        if (message.label.find("/initial/mode") == std::string::npos) continue;
        saw_private_scope = true;
        assert(message.payload == "with-mask-v2");
    }
    assert(saw_private_scope);

    const std::string wire = "/tmp/zkgpt-native-masked-layered-gkr.smoke";
    assert(session.write(wire, &reason));
    NativeProofSession serialized;
    assert(NativeProofSession::read(wire, serialized, &reason));
    assert(verify_masked_gkr_transcript(serialized, circuit, output,
                                        descriptors, &reason, nullptr));
    std::remove(wire.c_str());

    std::vector<NativeProofMessage> tampered_messages = session.messages();
    assert(mutate_input_binding_payload(tampered_messages));
    NativeProofSession tampered;
    assert(NativeProofSession::from_messages(session.domain(), session.statement(),
                                             tampered_messages, tampered, &reason));
    assert(!verify_masked_gkr_transcript(tampered, circuit, output,
                                         descriptors, &reason));

    auto rejects_tampered_artifact = [&](std::vector<NativeProofMessage> messages) {
        NativeProofSession candidate;
        if (!NativeProofSession::from_messages(session.domain(), session.statement(),
                                               messages, candidate, &reason))
            return true;
        return !verify_masked_gkr_transcript(candidate, circuit, output,
                                             descriptors, &reason);
    };
    std::vector<NativeProofMessage> mode_tampered = session.messages();
    assert(mutate_mode_to_zero(mode_tampered));
    assert(rejects_tampered_artifact(std::move(mode_tampered)));
    std::vector<NativeProofMessage> dropped_mask = session.messages();
    assert(drop_masked_initial_opening(dropped_mask));
    assert(rejects_tampered_artifact(std::move(dropped_mask)));
    std::vector<NativeProofMessage> replaced_claim = session.messages();
    assert(replace_intermediate_claim(replaced_claim));
    assert(rejects_tampered_artifact(std::move(replaced_claim)));

    NativeProofSession wrong_output("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(wrong_output, circuit, witness, output, &reason));
    assert(!verify_masked_gkr_transcript(wrong_output, circuit,
                                         {Fr(11), Fr(28)}, descriptors, &reason));

    NativeProofSession fresh("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(fresh, circuit, witness, output, &reason));
    std::string first_commitment, second_commitment;
    for (const NativeProofMessage &message : session.messages())
        if (message.label == "gkr/layer/2/phase1/round/0/h/0/meta")
            first_commitment = message.payload;
    for (const NativeProofMessage &message : fresh.messages())
        if (message.label == "gkr/layer/2/phase1/round/0/h/0/meta")
            second_commitment = message.payload;
    assert(!first_commitment.empty() && first_commitment != second_commitment);

    std::cout << "native_masked_layered_gkr_smoke: PASS\n";
    return 0;
}
