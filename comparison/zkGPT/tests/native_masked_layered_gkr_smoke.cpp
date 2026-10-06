#include "verifier.hpp"

#include <mcl/bn256.hpp>

#include <cassert>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "0000000000000000000000000000000000000000000000000000000000000221", "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000222", "0000000000000000000000000000000000000000000000000000000000000223",
        "0000000000000000000000000000000000000000000000000000000000000224", "0000000000000000000000000000000000000000000000000000000000000225",
        "0000000000000000000000000000000000000000000000000000000000000226", "0000000000000000000000000000000000000000000000000000000000000227",
        "0000000000000000000000000000000000000000000000000000000000000228", 4,
        "0000000000000000000000000000000000000000000000000000000000000221"};
}

layeredCircuit make_circuit() {
    layeredCircuit circuit;
    circuit.init(0, 3);

    circuit.circuit[0].ty = layerType::INPUT;
    circuit.circuit[0].size = 16;
    circuit.circuit[0].zero_start_id = 16;
    circuit.circuit[0].bit_length = 4;

    circuit.circuit[1].ty = layerType::FCONN;
    circuit.circuit[1].size = 2;
    circuit.circuit[1].zero_start_id = 2;
    circuit.circuit[1].bit_length = 1;
    circuit.circuit[1].need_phase2 = true;
    circuit.circuit[1].native_relation_id = "masked-layered-gkr/fconn0";
    circuit.circuit[1].native_family_id = "masked-layered-gkr/family0";
    circuit.circuit[1].native_rows = 1;
    circuit.circuit[1].native_inner = 2;
    circuit.circuit[1].native_cols = 2;
    circuit.circuit[1].native_lhs_offset = 0;
    circuit.circuit[1].native_rhs_offset = 4;
    circuit.circuit[1].native_output_offset = 8;

    circuit.circuit[2].ty = layerType::ADD_BIAS;
    circuit.circuit[2].size = 2;
    circuit.circuit[2].zero_start_id = 2;
    circuit.circuit[2].bit_length = 1;
    circuit.circuit[2].reserve_gate_arrays(2, 0);
    circuit.circuit[2].reserve_coefficients(1);
    circuit.circuit[2].add_unary(0, 0, 1, 1);
    circuit.circuit[2].add_unary(1, 1, 1, 1);
    circuit.initSubset();
    return circuit;
}

layeredCircuit make_fconn_then_ordinary_circuit() {
    layeredCircuit circuit;
    circuit.init(0, 3);

    circuit.circuit[0].ty = layerType::INPUT;
    circuit.circuit[0].size = 16;
    circuit.circuit[0].zero_start_id = 16;
    circuit.circuit[0].bit_length = 4;

    circuit.circuit[1].ty = layerType::ADD_BIAS;
    circuit.circuit[1].size = 2;
    circuit.circuit[1].zero_start_id = 2;
    circuit.circuit[1].bit_length = 1;
    circuit.circuit[1].reserve_gate_arrays(2, 0);
    circuit.circuit[1].reserve_coefficients(1);
    circuit.circuit[1].add_unary(0, 8, 0, 1);
    circuit.circuit[1].add_unary(1, 9, 0, 1);

    circuit.circuit[2].ty = layerType::FCONN;
    circuit.circuit[2].size = 2;
    circuit.circuit[2].zero_start_id = 2;
    circuit.circuit[2].bit_length = 1;
    circuit.circuit[2].need_phase2 = true;
    circuit.circuit[2].native_relation_id = "masked-layered-gkr/fconn0";
    circuit.circuit[2].native_family_id = "masked-layered-gkr/family0";
    circuit.circuit[2].native_rows = 1;
    circuit.circuit[2].native_inner = 2;
    circuit.circuit[2].native_cols = 2;
    circuit.circuit[2].native_lhs_offset = 0;
    circuit.circuit[2].native_rhs_offset = 4;
    circuit.circuit[2].native_output_offset = 8;
    circuit.initSubset();
    return circuit;
}

layeredCircuit make_ordinary_phase2_circuit() {
    layeredCircuit circuit;
    circuit.init(0, 3);
    circuit.circuit[0].ty = layerType::INPUT;
    circuit.circuit[0].size = 16;
    circuit.circuit[0].zero_start_id = 16;
    circuit.circuit[0].bit_length = 4;

    circuit.circuit[1].ty = layerType::FCONN;
    circuit.circuit[1].size = 2;
    circuit.circuit[1].zero_start_id = 2;
    circuit.circuit[1].bit_length = 1;
    circuit.circuit[1].native_relation_id = "masked-layered-gkr/phase2-fconn";
    circuit.circuit[1].native_family_id = "masked-layered-gkr/phase2-family";
    circuit.circuit[1].native_rows = 1;
    circuit.circuit[1].native_inner = 2;
    circuit.circuit[1].native_cols = 2;
    circuit.circuit[1].native_rhs_offset = 4;
    circuit.circuit[1].native_output_offset = 8;

    circuit.circuit[2].ty = layerType::Sqr;
    circuit.circuit[2].size = 2;
    circuit.circuit[2].zero_start_id = 2;
    circuit.circuit[2].bit_length = 1;
    circuit.circuit[2].need_phase2 = true;
    circuit.circuit[2].reserve_gate_arrays(0, 2);
    circuit.circuit[2].reserve_coefficients(1);
    circuit.circuit[2].add_binary(0, 0, 0, 1, 0);
    circuit.circuit[2].add_binary(1, 1, 1, 1, 0);
    circuit.initSubset();
    return circuit;
}

void test_bitmap_subset_matches_raw_sort_unique() {
    layeredCircuit circuit;
    circuit.init(0, 3);
    circuit.circuit[0].ty = layerType::INPUT;
    circuit.circuit[0].size = 131;
    circuit.circuit[0].bit_length = 8;
    circuit.circuit[1].ty = layerType::INPUT;
    circuit.circuit[1].size = 10;
    circuit.circuit[1].bit_length = 4;
    circuit.circuit[2].ty = layerType::ADD_BIAS;
    circuit.circuit[2].size = 6;
    circuit.circuit[2].reserve_gate_arrays(5, 4);
    circuit.circuit[2].reserve_coefficients(1);
    circuit.circuit[2].add_unary(0, 17, 0, 1);
    circuit.circuit[2].add_unary(1, 17, 0, 1);
    circuit.circuit[2].add_unary(2, 130, 0, 1);
    circuit.circuit[2].add_unary(3, 2, 0, 1);
    circuit.circuit[2].add_unary(4, 5, 1, 1); // Previous-layer source: not part of the layer-0 map.
    circuit.circuit[2].add_binary(0, 4, 129, 1, 0);
    circuit.circuit[2].add_binary(1, 17, 4, 1, 0);
    circuit.circuit[2].add_binary(2, 2, 3, 1, 1); // Both operands come from layer 1.
    circuit.circuit[2].add_binary(3, 8, 17, 1, 2); // U from layer 1, V from layer 0.

    std::vector<u64> expected_u{17, 17, 130, 2, 4, 17};
    std::vector<u64> expected_v{129, 4, 17};
    std::sort(expected_u.begin(), expected_u.end());
    expected_u.erase(std::unique(expected_u.begin(), expected_u.end()), expected_u.end());
    std::sort(expected_v.begin(), expected_v.end());
    expected_v.erase(std::unique(expected_v.begin(), expected_v.end()), expected_v.end());
    const auto old_sort_unique_index = [](const std::vector<u64>& coordinates, u64 coordinate) {
        const auto found = std::lower_bound(coordinates.begin(), coordinates.end(), coordinate);
        assert(found != coordinates.end() && *found == coordinate);
        return static_cast<u64>(found - coordinates.begin());
    };

    circuit.initSubset();
    const layer& mapped = circuit.circuit[2];
    assert(std::vector<u64>(mapped.ori_id_u.begin(), mapped.ori_id_u.end()) == expected_u);
    assert(std::vector<u64>(mapped.ori_id_v.begin(), mapped.ori_id_v.end()) == expected_v);
    assert(mapped.uni_gates[0].u == old_sort_unique_index(expected_u, 17));
    assert(mapped.uni_gates[1].u == old_sort_unique_index(expected_u, 17));
    assert(mapped.uni_gates[2].u == old_sort_unique_index(expected_u, 130));
    assert(mapped.uni_gates[3].u == old_sort_unique_index(expected_u, 2));
    assert(mapped.uni_gates[4].u == 5);
    assert(mapped.bin_gates[0].u == old_sort_unique_index(expected_u, 4));
    assert(mapped.bin_gates[0].v == old_sort_unique_index(expected_v, 129));
    assert(mapped.bin_gates[1].u == old_sort_unique_index(expected_u, 17));
    assert(mapped.bin_gates[1].v == old_sort_unique_index(expected_v, 4));
    assert(mapped.bin_gates[2].u == 2 && mapped.bin_gates[2].v == 3);
    assert(mapped.bin_gates[3].u == 8 && mapped.bin_gates[3].v == old_sort_unique_index(expected_v, 17));
}

NativeLayeredGkrWitness make_witness() {
    NativeLayeredGkrWitness witness;
    zkgpt::NativeFieldVector input(16, Fr(0));
    input[0] = Fr(3);
    input[1] = Fr(5);
    // Canonical W input coordinates are row-major; the bounded matrix helper
    // consumes its dense RHS operand in column-major order below.
    input[4] = Fr(2);
    input[5] = Fr(4);
    input[6] = Fr(1);
    input[7] = Fr(3);
    input[8] = Fr(11);
    input[9] = Fr(27);
    witness.layer_values = {std::move(input), {Fr(11), Fr(27)}, {Fr(11), Fr(27)}};
    NativeGkrMatmulWitness matmul;
    matmul.shape = {1, 2, 2};
    matmul.shape.relation_id = "masked-layered-gkr/fconn0";
    matmul.shape.family_id = "masked-layered-gkr/family0";
    matmul.shape.lhs_offset = 0;
    matmul.shape.rhs_offset = 4;
    matmul.shape.output_offset = 8;
    matmul.shape.input_segment_domain = 4;
    matmul.shape.lhs_segment = "X_pub";
    matmul.shape.rhs_segment = "W";
    matmul.shape.lhs_segment_offset = 0;
    matmul.shape.rhs_segment_offset = 0;
    matmul.shape.input_segment_logical_lengths = {2, 4, 2, 0};
    matmul.shape.output_segment = "Aux";
    matmul.shape.output_segment_offset = 0;
    matmul.left = {Fr(3), Fr(5)};
    matmul.right = {Fr(2), Fr(1), Fr(4), Fr(3)};
    witness.matmuls.push_back(matmul);
    return witness;
}

bool mutate_input_binding_payload(std::vector<NativeProofMessage>& messages) {
    for (auto message_it = messages.rbegin(); message_it != messages.rend(); ++message_it) {
        NativeProofMessage& message = *message_it;
        if (message.kind != NativeProofMessageKind::Field || message.label != "gkr/input/bind/response") continue;
        if (message.payload.empty()) return false;
        message.payload.back() = message.payload.back() == '0' ? '1' : '0';
        return true;
    }
    return false;
}

bool mutate_mode_to_zero(std::vector<NativeProofMessage>& messages) {
    for (NativeProofMessage& message : messages) {
        if (message.kind != NativeProofMessageKind::Data || message.label.find("/initial/mode") == std::string::npos) continue;
        message.payload = "zero-mask-v2";
        return true;
    }
    return false;
}

bool drop_masked_initial_opening(std::vector<NativeProofMessage>& messages) {
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const std::string marker = "/initial/masked/meta";
        if (messages[index].label.size() < marker.size() ||
            messages[index].label.compare(messages[index].label.size() - marker.size(), marker.size(), marker) != 0)
            continue;
        if (index + 4 > messages.size()) return false;
        messages.erase(messages.begin() + static_cast<std::ptrdiff_t>(index), messages.begin() + static_cast<std::ptrdiff_t>(index + 4));
        for (std::size_t sequence = 0; sequence < messages.size(); ++sequence) messages[sequence].sequence = sequence;
        return true;
    }
    return false;
}

bool replace_intermediate_claim(std::vector<NativeProofMessage>& messages) {
    std::string replacement;
    for (const NativeProofMessage& message : messages) {
        if (message.label.find("/initial/opening/meta") != std::string::npos) {
            replacement = message.payload;
            break;
        }
    }
    if (replacement.empty()) return false;
    for (NativeProofMessage& message : messages) {
        if (message.label.find("/next-claim/unmasked/output/meta") == std::string::npos) continue;
        message.payload = replacement;
        return true;
    }
    return false;
}

bool replace_opening_payload(std::vector<NativeProofMessage>& messages, const std::string& target, const std::string& source) {
    std::string replacement;
    for (const NativeProofMessage& message : messages)
        if (message.kind == NativeProofMessageKind::Data && message.label == source) replacement = message.payload;
    if (replacement.empty()) return false;
    for (NativeProofMessage& message : messages) {
        if (message.kind != NativeProofMessageKind::Data || message.label != target) continue;
        message.payload = replacement;
        return true;
    }
    return false;
}

}

int main() {
    test_bitmap_subset_matches_raw_sort_unique();
    initPairing(BN254);
    const layeredCircuit circuit = make_circuit();
    NativeLayeredGkrWitness witness = make_witness();
    const std::vector<Fr> output{Fr(11), Fr(27)};
    const zkgpt::NativeFieldView output_view =
        zkgpt::NativeFieldView::from_vector(output);
    const std::vector<NativeGkrMatmulShape> descriptors{witness.matmuls.front().shape};
    std::string reason;
    NativeGkrInputBinding produced_binding;
    NativeGkrInputBinding verified_binding;
    NativeGkrProverResult prover_result;

    NativeProofSession session("masked-layered-gkr-smoke-v1", statement_fixture());
    const std::size_t messages_before_prover_result = session.messages().size();
    if (!prove_masked_gkr_transcript(session, circuit, witness, output_view, &reason, &produced_binding, &prover_result)) {
        std::cerr << "prove failed: " << reason << '\n';
        return 1;
    }
    assert(session.messages().size() > messages_before_prover_result);
    assert(prover_result.has_unmasked_input_claim_opening);
    const auto basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest());
    assert(prover_result.unmasked_input_claim_opening.valid(basis, &reason));
    assert(prover_result.unmasked_input_claim_opening.commitment.point == produced_binding.unmasked_claim_commitment.point);
    assert(prover_result.input_binding.unmasked_claim_commitment.point == produced_binding.unmasked_claim_commitment.point);
    if (!verify_masked_gkr_transcript(session, circuit, output_view, descriptors, &reason, &verified_binding)) {
        std::cerr << "verify failed: " << reason << '\n';
        return 1;
    }
    assert(produced_binding.point == verified_binding.point);
    assert(produced_binding.claim_commitment.point == verified_binding.claim_commitment.point);
    assert(produced_binding.unmasked_claim_commitment.point == verified_binding.unmasked_claim_commitment.point);
    assert(produced_binding.mask_commitment.point == verified_binding.mask_commitment.point);
    bool saw_private_scope = false;
    for (const NativeProofMessage& message : session.messages()) {
        if (message.label.find("/initial/mode") == std::string::npos) continue;
        saw_private_scope = true;
        assert(message.payload == "with-mask-v2");
    }
    assert(saw_private_scope);

    NativeGkrMatmulShape family_tampered = descriptors.front();
    family_tampered.family_id += ".tampered";
    NativeProofSession family_session("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(family_session, circuit, witness, output_view, &reason));
    assert(!verify_masked_gkr_transcript(family_session, circuit, output_view, {family_tampered}, &reason));
    NativeGkrMatmulShape offset_tampered = descriptors.front();
    offset_tampered.output_offset += 1;
    assert(!verify_masked_gkr_transcript(family_session, circuit, output_view, {offset_tampered}, &reason));

    const std::string wire = "/tmp/zkgpt-native-masked-layered-gkr.smoke";
    assert(session.write(wire, &reason));
    NativeProofSession serialized;
    assert(NativeProofSession::read(wire, serialized, &reason));
    assert(verify_masked_gkr_transcript(serialized, circuit, output_view, descriptors, &reason, nullptr));
    std::remove(wire.c_str());

    std::vector<NativeProofMessage> tampered_messages = session.messages();
    assert(mutate_input_binding_payload(tampered_messages));
    NativeProofSession tampered;
    assert(NativeProofSession::from_messages(session.domain(), session.statement(), tampered_messages, tampered, &reason));
    assert(!verify_masked_gkr_transcript(tampered, circuit, output_view, descriptors, &reason));

    auto rejects_tampered_artifact = [&](std::vector<NativeProofMessage> messages) {
        NativeProofSession candidate;
        if (!NativeProofSession::from_messages(session.domain(), session.statement(), messages, candidate, &reason)) return true;
        return !verify_masked_gkr_transcript(candidate, circuit, output_view, descriptors, &reason);
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
    std::vector<NativeProofMessage> final_claim_tampered = session.messages();
    assert(replace_opening_payload(final_claim_tampered, "gkr/input/claim/meta", "gkr/output/public-claim/meta"));
    assert(rejects_tampered_artifact(std::move(final_claim_tampered)));

    NativeProofSession wrong_output("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(wrong_output, circuit, witness, output_view, &reason));
    assert(!verify_masked_gkr_transcript(wrong_output, circuit, {Fr(11), Fr(28)}, descriptors, &reason));

    // Traverse a high FCONN layer followed by a lower ordinary layer. This
    // exercises the driver-owned equality-point handoff across the FCONN
    // early-continue branch, where the legacy prover point arrays are stale.
    const layeredCircuit fconn_then_ordinary =
        make_fconn_then_ordinary_circuit();
    NativeLayeredGkrWitness fconn_then_ordinary_witness = make_witness();
    const std::vector<NativeGkrMatmulShape> fconn_then_ordinary_descriptors{
        fconn_then_ordinary_witness.matmuls.front().shape};
    assert(validate_native_gkr_point_domains(
        fconn_then_ordinary, fconn_then_ordinary_descriptors, 16, &reason));
    std::vector<NativeGkrMatmulShape> wrong_terminal_domain =
        fconn_then_ordinary_descriptors;
    wrong_terminal_domain.front().cols += 1;
    assert(!validate_native_gkr_point_domains(
        fconn_then_ordinary, wrong_terminal_domain, 16, &reason));
    NativeProofSession fconn_then_ordinary_session(
        "masked-layered-gkr-fconn-to-ordinary-v1", statement_fixture());
    if (!prove_masked_gkr_transcript(
            fconn_then_ordinary_session, fconn_then_ordinary,
            fconn_then_ordinary_witness, output_view, &reason) ||
        !verify_masked_gkr_transcript(
            fconn_then_ordinary_session, fconn_then_ordinary, output_view,
            fconn_then_ordinary_descriptors, &reason)) {
        std::cerr << "FCONN-to-ordinary proof/verification failed: " << reason << '\n';
        return 13;
    }
    const std::string mixed_wire = "/tmp/zkgpt-native-masked-layered-gkr-mixed.smoke";
    assert(fconn_then_ordinary_session.write(mixed_wire, &reason));
    NativeProofSession mixed_fresh;
    assert(NativeProofSession::read(mixed_wire, mixed_fresh, &reason));
    assert(verify_masked_gkr_transcript(
        mixed_fresh, fconn_then_ordinary, output_view,
        fconn_then_ordinary_descriptors, &reason));
    std::remove(mixed_wire.c_str());
    std::vector<NativeProofMessage> fconn_then_ordinary_mutant =
        fconn_then_ordinary_session.messages();
    assert(mutate_input_binding_payload(fconn_then_ordinary_mutant));
    NativeProofSession fconn_then_ordinary_mutant_session;
    assert(NativeProofSession::from_messages(
        fconn_then_ordinary_session.domain(),
        fconn_then_ordinary_session.statement(),
        fconn_then_ordinary_mutant, fconn_then_ordinary_mutant_session,
        &reason));
    assert(!verify_masked_gkr_transcript(
        fconn_then_ordinary_mutant_session, fconn_then_ordinary,
        output_view, fconn_then_ordinary_descriptors, &reason));
    const std::vector<Fr> wrong_fconn_output{Fr(11), Fr(28)};
    const zkgpt::NativeFieldView wrong_fconn_output_view =
        zkgpt::NativeFieldView::from_vector(wrong_fconn_output);
    assert(!verify_masked_gkr_transcript(
        fconn_then_ordinary_session, fconn_then_ordinary,
        wrong_fconn_output_view, fconn_then_ordinary_descriptors, &reason));

    NativeProofSession fresh("masked-layered-gkr-smoke-v1", statement_fixture());
    assert(prove_masked_gkr_transcript(fresh, circuit, witness, output_view, &reason));
    std::string first_commitment, second_commitment;
    for (const NativeProofMessage& message : session.messages())
        if (message.label == "gkr/layer/2/phase1/round/0/h/0/meta") first_commitment = message.payload;
    for (const NativeProofMessage& message : fresh.messages())
        if (message.label == "gkr/layer/2/phase1/round/0/h/0/meta") second_commitment = message.payload;
    assert(!first_commitment.empty() && first_commitment != second_commitment);

    const layeredCircuit ordinary_phase2_circuit = make_ordinary_phase2_circuit();
    NativeLayeredGkrWitness ordinary_witness;
    ordinary_witness.layer_values = {
        zkgpt::NativeFieldVector(16, F_ZERO), {F_ZERO, F_ZERO}, {F_ZERO, F_ZERO}};
    NativeGkrMatmulWitness ordinary_matmul;
    ordinary_matmul.shape = {1, 2, 2};
    ordinary_matmul.shape.relation_id = "masked-layered-gkr/phase2-fconn";
    ordinary_matmul.shape.family_id = "masked-layered-gkr/phase2-family";
    ordinary_matmul.shape.rhs_offset = 4;
    ordinary_matmul.shape.output_offset = 8;
    ordinary_matmul.shape.input_segment_domain = 4;
    ordinary_matmul.shape.lhs_segment = "X_pub";
    ordinary_matmul.shape.rhs_segment = "W";
    ordinary_matmul.shape.input_segment_logical_lengths = {2, 4, 2, 0};
    ordinary_matmul.shape.output_segment = "Aux";
    ordinary_matmul.left = {F_ZERO, F_ZERO};
    ordinary_matmul.right = {F_ZERO, F_ZERO, F_ZERO, F_ZERO};
    const std::vector<NativeGkrMatmulShape> ordinary_descriptors{
        ordinary_matmul.shape};
    ordinary_witness.matmuls.push_back(ordinary_matmul);
    const std::vector<Fr> ordinary_output{F_ZERO, F_ZERO};
    const zkgpt::NativeFieldView ordinary_output_view =
        zkgpt::NativeFieldView::from_vector(ordinary_output);
    NativeProofSession ordinary_phase2("masked-layered-gkr-phase2-smoke-v1", statement_fixture());
    NativeGkrInputBinding ordinary_binding;
    if (!prove_masked_gkr_transcript(ordinary_phase2, ordinary_phase2_circuit, ordinary_witness, ordinary_output_view, &reason, &ordinary_binding, nullptr) ||
        !verify_masked_gkr_transcript(ordinary_phase2, ordinary_phase2_circuit, ordinary_output_view, ordinary_descriptors, &reason, nullptr)) {
        std::cerr << "ordinary phase2 fixture failed: " << reason << '\n';
        return 9;
    }
    bool saw_phase2 = false;
    for (const NativeProofMessage& message : ordinary_phase2.messages())
        if (message.label == "gkr/layer/2/phase2/terminal/unmasked/fixed/0/meta") saw_phase2 = true;
    if (!saw_phase2) {
        std::cerr << "ordinary phase2 fixture did not emit fixed binding\n";
        return 10;
    }
    std::vector<NativeProofMessage> ordinary_tampered = ordinary_phase2.messages();
    if (!replace_opening_payload(ordinary_tampered, "gkr/layer/2/phase2/terminal/unmasked/fixed/0/meta", "gkr/layer/2/phase2/terminal/unmasked/value/0/meta")) {
        std::cerr << "ordinary phase2 fixed binding fixture was not found\n";
        return 11;
    }
    NativeProofSession ordinary_tampered_session;
    if (NativeProofSession::from_messages(ordinary_phase2.domain(), ordinary_phase2.statement(), ordinary_tampered, ordinary_tampered_session, &reason) &&
        verify_masked_gkr_transcript(ordinary_tampered_session, ordinary_phase2_circuit, ordinary_output_view, ordinary_descriptors, &reason, nullptr)) {
        std::cerr << "ordinary phase2 fixed binding tamper was accepted\n";
        return 12;
    }

    std::cout << "native_masked_layered_gkr_smoke: PASS\n";
    return 0;
}
