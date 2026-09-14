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

bool mutate_terminal_label(std::vector<NativeProofMessage> &messages) {
    for (auto message_it = messages.rbegin(); message_it != messages.rend(); ++message_it) {
        NativeProofMessage &message = *message_it;
        if (message.kind != NativeProofMessageKind::FieldVector ||
            message.label.find("/terminal") == std::string::npos)
            continue;
        message.label.back() = 'x';
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

    NativeProofSession session("masked-layered-gkr-smoke-v1", statement_fixture());
    if (!prove_masked_gkr_transcript(session, circuit, witness, output, &reason)) {
        std::cerr << "prove failed: " << reason << '\n';
        return 1;
    }
    assert(verify_masked_gkr_transcript(session, circuit, output, descriptors, &reason));

    const std::string wire = "/tmp/zkgpt-native-masked-layered-gkr.smoke";
    assert(session.write(wire, &reason));
    NativeProofSession serialized;
    assert(NativeProofSession::read(wire, serialized, &reason));
    assert(verify_masked_gkr_transcript(serialized, circuit, output,
                                        descriptors, &reason));
    std::remove(wire.c_str());

    std::vector<NativeProofMessage> tampered_messages = session.messages();
    assert(mutate_terminal_label(tampered_messages));
    NativeProofSession tampered;
    assert(NativeProofSession::from_messages(session.domain(), session.statement(),
                                             tampered_messages, tampered, &reason));
    assert(!verify_masked_gkr_transcript(tampered, circuit, output,
                                         descriptors, &reason));

    NativeProofSession wrong_output("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(wrong_output, circuit, witness, output, &reason));
    assert(!verify_masked_gkr_transcript(wrong_output, circuit,
                                         {Fr(11), Fr(28)}, descriptors, &reason));

    NativeProofSession fresh("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(fresh, circuit, witness, output, &reason));
    std::string first_commitment, second_commitment;
    for (const NativeProofMessage &message : session.messages())
        if (message.label.find("/mask/0/commitment") != std::string::npos)
            first_commitment = message.payload;
    for (const NativeProofMessage &message : fresh.messages())
        if (message.label.find("/mask/0/commitment") != std::string::npos)
            second_commitment = message.payload;
    assert(!first_commitment.empty() && first_commitment != second_commitment);

    std::cout << "native_masked_layered_gkr_smoke: PASS\n";
    return 0;
}
