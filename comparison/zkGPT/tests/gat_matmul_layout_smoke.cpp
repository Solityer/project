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
    circuit.circuit[1].size = 2;
    circuit.circuit[1].zero_start_id = 2;
    circuit.circuit[1].bit_length = 1;
    circuit.circuit[1].need_phase2 = true;
    circuit.circuit[2].ty = layerType::ADD_BIAS;
    circuit.circuit[2].size = 2;
    circuit.circuit[2].zero_start_id = 2;
    circuit.circuit[2].bit_length = 1;
    for (u32 index = 0; index < 2; ++index)
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
    const std::vector<Fr> expected{Fr(11), Fr(21), Fr(31), Fr(0),
                                   Fr(12), Fr(22), Fr(32), Fr(0)};
    assert(packed == expected);

    // A short span and a stride/layout violation are rejected at the adapter
    // boundary instead of being silently reinterpreted.
    bool rejected = false;
    try {
        (void)pack_native_gkr_matmul_rhs(
            rectangular, std::vector<Fr>(canonical_input.begin(),
                                         canonical_input.end() - 2));
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

    // Walk the native masked FCONN with an asymmetric, non-power-of-two
    // inner dimension.  The zero at the end of every RHS column is padding,
    // not the first coordinate of the next column.
    const MatMulDescriptor native_descriptor = descriptor(2, 3, 2, 8);
    const std::vector<Fr> canonical_native_input{
        Fr(1), Fr(2), Fr(3), Fr(4), Fr(5), Fr(6),
        Fr(0), Fr(0), Fr(10), Fr(40), Fr(20), Fr(50), Fr(30), Fr(60),
        Fr(0), Fr(0)};
    const std::vector<Fr> native_right = pack_native_gkr_matmul_rhs(
        native_descriptor, canonical_native_input);
    assert((native_right == std::vector<Fr>{Fr(10), Fr(20), Fr(30), Fr(0),
                                            Fr(40), Fr(50), Fr(60), Fr(0)}));

    NativeLayeredGkrWitness witness;
    witness.layer_values = {canonical_native_input,
                            {Fr(140), Fr(320), Fr(320), Fr(770)},
                            {Fr(140), Fr(320), Fr(320), Fr(770)}};
    NativeGkrMatmulWitness matmul;
    matmul.shape = {2, 3, 2};
    matmul.shape.relation_id = native_descriptor.relation_id;
    matmul.shape.family_id = native_descriptor.family_id;
    matmul.shape.lhs_offset = native_descriptor.lhs_offset;
    matmul.shape.rhs_offset = native_descriptor.rhs_offset;
    matmul.shape.output_offset = native_descriptor.output_offset;
    matmul.left = {Fr(1), Fr(2), Fr(3), Fr(4), Fr(5), Fr(6)};
    matmul.right = native_right;
    witness.matmuls.push_back(matmul);

    layeredCircuit circuit = make_power_of_two_circuit();
    circuit.circuit[0].size = 16;
    circuit.circuit[0].zero_start_id = 16;
    circuit.circuit[0].bit_length = 4;
    circuit.circuit[1].size = 4;
    circuit.circuit[1].zero_start_id = 4;
    circuit.circuit[1].bit_length = 2;
    circuit.circuit[2].size = 4;
    circuit.circuit[2].zero_start_id = 4;
    circuit.circuit[2].bit_length = 2;
    circuit.circuit[2].uni_gates.clear();
    for (u32 index = 0; index < 4; ++index)
        circuit.circuit[2].uni_gates.emplace_back(index, index, 1, 1);
    circuit.circuit[1].native_relation_id = native_descriptor.relation_id;
    circuit.circuit[1].native_family_id = native_descriptor.family_id;
    circuit.circuit[1].native_rows = native_descriptor.rows;
    circuit.circuit[1].native_inner = native_descriptor.inner;
    circuit.circuit[1].native_cols = native_descriptor.cols;
    circuit.circuit[1].native_lhs_offset = native_descriptor.lhs_offset;
    circuit.circuit[1].native_rhs_offset = native_descriptor.rhs_offset;
    circuit.circuit[1].native_output_offset = native_descriptor.output_offset;
    circuit.initSubset();
    const std::vector<Fr> output{Fr(140), Fr(320), Fr(320), Fr(770)};
    std::string reason;
    NativeProofSession session("gat-matmul-layout-smoke-v1", statement_fixture());
    NativeGkrInputBinding binding;
    NativeGkrProverResult prover_result;
    if (!prove_masked_gkr_transcript(session, circuit, witness, output,
                                     &reason, &binding, &prover_result)) {
        std::cerr << "native honest transition rejected: " << reason << '\n';
        return 1;
    }
    assert(prover_result.has_unmasked_input_claim_opening);
    assert(verify_masked_gkr_transcript(session, circuit, output,
                                        {matmul.shape}, &reason));

    NativeGkrMatmulShape relation_tampered = matmul.shape;
    relation_tampered.relation_id += ".tampered";
    assert(!verify_masked_gkr_transcript(session, circuit, output,
                                         {relation_tampered}, &reason));
    NativeGkrMatmulShape family_tampered = matmul.shape;
    family_tampered.family_id += ".tampered";
    assert(!verify_masked_gkr_transcript(session, circuit, output,
                                         {family_tampered}, &reason));
    NativeGkrMatmulShape offset_tampered = matmul.shape;
    offset_tampered.rhs_offset += 1;
    assert(!verify_masked_gkr_transcript(session, circuit, output,
                                         {offset_tampered}, &reason));
    NativeGkrMatmulShape output_tampered = matmul.shape;
    output_tampered.output_offset += 1;
    assert(!verify_masked_gkr_transcript(session, circuit, output,
                                         {output_tampered}, &reason));
    NativeGkrMatmulShape dims_tampered = matmul.shape;
    dims_tampered.inner += 1;
    assert(!verify_masked_gkr_transcript(session, circuit, output,
                                         {dims_tampered}, &reason));

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
