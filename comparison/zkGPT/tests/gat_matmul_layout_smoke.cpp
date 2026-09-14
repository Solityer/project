#include "gat_proof.hpp"

#include <mcl/bn256.hpp>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zkgpt::gat;
using namespace mcl::bn;

namespace {

NativeProofStatement statement_fixture() {
    return NativeProofStatement{
        "0000000000000000000000000000000000000000000000000000000000000311",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000312",
        "0000000000000000000000000000000000000000000000000000000000000313",
        "0000000000000000000000000000000000000000000000000000000000000314",
        "0000000000000000000000000000000000000000000000000000000000000315",
        "0000000000000000000000000000000000000000000000000000000000000316",
        "0000000000000000000000000000000000000000000000000000000000000317",
        "0000000000000000000000000000000000000000000000000000000000000318",
        4,
        "0000000000000000000000000000000000000000000000000000000000000311"};
}

MatMulDescriptor descriptor(std::size_t rows, std::size_t inner,
                            std::size_t cols, std::size_t rhs_offset) {
    MatMulDescriptor result;
    result.relation_id = "layout-smoke";
    result.family_id = "layout-smoke";
    result.lhs = "L";
    result.rhs = "R";
    result.output = "O";
    result.rows = rows;
    result.inner = inner;
    result.cols = cols;
    result.output_coordinates = rows * cols;
    result.rhs_offset = rhs_offset;
    result.coordinate_rule = "row_major_exact[row,term,col]";
    return result;
}

layeredCircuit make_power_of_two_circuit() {
    layeredCircuit circuit;
    circuit.init(0, 3);
    circuit.circuit[0].ty = layerType::INPUT;
    circuit.circuit[0].size = 4;
    circuit.circuit[0].zero_start_id = 4;
    circuit.circuit[0].bit_length = 2;
    circuit.circuit[1].ty = layerType::FCONN;
    circuit.circuit[1].size = 4;
    circuit.circuit[1].zero_start_id = 4;
    circuit.circuit[1].bit_length = 2;
    circuit.circuit[1].need_phase2 = true;
    circuit.circuit[2].ty = layerType::ADD_BIAS;
    circuit.circuit[2].size = 4;
    circuit.circuit[2].zero_start_id = 4;
    circuit.circuit[2].bit_length = 2;
    for (u32 index = 0; index < 4; ++index)
        circuit.circuit[2].uni_gates.emplace_back(index, index, 1, 1);
    circuit.initSubset();
    return circuit;
}

} // namespace

int main() {
    initPairing(BN254);

    // Canonical RHS is row-major [term,col].  The native MLE RHS is
    // column-major [col,term], so this non-square/asymmetric shape must move
    // the second column across the inner-dimension stride.
    const MatMulDescriptor rectangular = descriptor(2, 3, 2, 1);
    const std::vector<Fr> canonical_input{
        Fr(99), Fr(11), Fr(12), Fr(21), Fr(22), Fr(31), Fr(32), Fr(77)};
    const std::vector<Fr> packed = pack_native_gkr_matmul_rhs(
        rectangular, canonical_input);
    const std::vector<Fr> expected{Fr(11), Fr(21), Fr(31),
                                   Fr(12), Fr(22), Fr(32)};
    assert(packed == expected);

    // A one-element truncation and a one-element offset/stride violation are
    // rejected at the adapter boundary instead of being reinterpreted.
    bool rejected = false;
    try {
        (void)pack_native_gkr_matmul_rhs(
            rectangular, std::vector<Fr>(canonical_input.begin(),
                                         canonical_input.end() - 1));
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    MatMulDescriptor bad_stride = rectangular;
    bad_stride.coordinate_rule = "column_major_reinterpreted";
    rejected = false;
    try {
        (void)pack_native_gkr_matmul_rhs(bad_stride, canonical_input);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    // Walk the native masked FCONN with the adapter's packed RHS.  All
    // dimensions are powers of two here so this smoke isolates the layout
    // transition from the separately owned native padding implementation.
    const MatMulDescriptor native_descriptor = descriptor(2, 4, 2, 8);
    const std::vector<Fr> canonical_native_input{
        Fr(1), Fr(2), Fr(3), Fr(4), Fr(5), Fr(6), Fr(7), Fr(8),
        Fr(1), Fr(2), Fr(3), Fr(4), Fr(5), Fr(6), Fr(7), Fr(8)};
    const std::vector<Fr> native_right = pack_native_gkr_matmul_rhs(
        native_descriptor, canonical_native_input);
    assert((native_right == std::vector<Fr>{Fr(1), Fr(3), Fr(5), Fr(7),
                                            Fr(2), Fr(4), Fr(6), Fr(8)}));

    NativeLayeredGkrWitness witness;
    witness.layer_values = {{Fr(0), Fr(0), Fr(0), Fr(0)},
                            {Fr(50), Fr(60), Fr(162), Fr(188)},
                            {Fr(50), Fr(60), Fr(162), Fr(188)}};
    NativeGkrMatmulWitness matmul;
    matmul.shape = {2, 4, 2};
    matmul.left = {Fr(1), Fr(2), Fr(3), Fr(4),
                   Fr(5), Fr(6), Fr(7), Fr(8)};
    matmul.right = native_right;
    witness.matmuls.push_back(matmul);

    const layeredCircuit circuit = make_power_of_two_circuit();
    const std::vector<Fr> output{Fr(50), Fr(60), Fr(162), Fr(188)};
    std::string reason;
    NativeProofSession session("gat-matmul-layout-smoke-v1", statement_fixture());
    NativeGkrInputBinding binding;
    NativeGkrProverResult prover_result;
    assert(prove_masked_gkr_transcript(session, circuit, witness, output,
                                       &reason, &binding, &prover_result));
    assert(prover_result.has_unmasked_input_claim_opening);
    assert(verify_masked_gkr_transcript(session, circuit, output,
                                        {{2, 4, 2}}, &reason));

    NativeLayeredGkrWitness tampered = witness;
    tampered.matmuls.front().right.at(1) += Fr(1);
    NativeProofSession tampered_session(
        "gat-matmul-layout-smoke-v1", statement_fixture());
    assert(!prove_masked_gkr_transcript(tampered_session, circuit, tampered,
                                        output, &reason));

    std::cout << "gat_matmul_layout_smoke: PASS row-major/column-major pack, "
                 "native transition, bounds/tamper rejection\n";
    return 0;
}
