#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "../cpu-backend.cuh"

namespace {

zkdl::FrWire scalar(unsigned value) { return zkdl::fr_from_u32(value); }

bool same_fr(const zkdl::FrWire& a, const zkdl::FrWire& b) {
    return zkdl::fr_compare(a, b) == 0;
}

}  // namespace

int main() {
    using namespace zkdl;

    assert(same_fr(fr_mul(scalar(7U), scalar(9U)), scalar(63U)));
    const FrWire modulus_minus_two = fr_modulus_minus_two();
    assert(modulus_minus_two[0] == 0xffffffffU);
    assert(modulus_minus_two[1] == 0xfffffffeU);
    assert(modulus_minus_two[2] == kScalarModulus[2]);

    CpuFp fp_value{};
    fp_value[0] = 123U;
    assert(cpu_fp_compare(cpu_fp_from_mont(cpu_fp_to_mont(fp_value)), fp_value) == 0);

    const CpuG1Point generator = cpu_g1_generator();
    const G1Jacobian_t native_generator = cpu_g1_to_native(generator);
    assert(cpu_g1_equal(cpu_g1_from_native(native_generator), generator));

    std::vector<CpuG1Point> generators(4U, generator);
    std::vector<FrWire> values{scalar(1U), scalar(2U), scalar(3U), scalar(4U)};
    const std::vector<FrWire> point{scalar(7U), scalar(11U)};
    EvaluationOpeningProof opening = cpu_open(generators, values, point);
    std::string error;
    assert(opening_shape_valid(opening, &error));
    assert(cpu_verify_opening(opening,
                              cpu_g1_from_native(opening.commitment_eval), &error));

    ProofEnvelope envelope;
    envelope.openings.push_back(opening);
    const std::vector<zkdl::byte> encoded = envelope.serialize();
    ProofEnvelope decoded;
    assert(ProofEnvelope::deserialize(encoded, decoded, &error));
    assert(decoded.openings.size() == 1U);
    assert(cpu_verify_opening(decoded.openings[0],
                              cpu_g1_from_native(decoded.openings[0].commitment_eval),
                              &error));
    decoded.openings[0].value.val[0] ^= 1U;
    assert(!cpu_verify_opening(decoded.openings[0],
                               cpu_g1_from_native(opening.commitment_eval), &error));

    const CpuHidingCommitment model_commitment(
        "model-family", 1U, generators, generator, values, scalar(17U), scalar(23U));
    const CpuHidingCommitment dynamic_commitment(
        "dynamic-family", 2U, generators, generator, values, scalar(19U), scalar(29U));
    assert(!same_fr(model_commitment.randomness, dynamic_commitment.randomness));
    assert(cpu_g1_equal(model_commitment.commitment, model_commitment.commitment));
    assert(cpu_g1_equal(dynamic_commitment.commitment, dynamic_commitment.commitment));
    assert(validate_hiding_commitments(
        std::vector<CommitmentRecord>{model_commitment.record(), dynamic_commitment.record()},
        &error));
    CommitmentRecord tampered = dynamic_commitment.record();
    tampered.schnorr_response.val[0] ^= 1U;
    assert(!CpuHidingCommitment::verify_record(tampered, &error));

    SumcheckProof sumcheck;
    sumcheck.label = "cpu-output-binding";
    sumcheck.degree = 1U;
    SumcheckRound round;
    round.evaluations = {fr_value(scalar(2U)), fr_value(scalar(3U))};
    sumcheck.rounds.push_back(round);
    Transcript prover_transcript;
    prover_transcript.absorb_fr(sumcheck.label + ".round", round.evaluations);
    const Fr_t challenge = prover_transcript.challenge(sumcheck.label + ".challenge");
    sumcheck.final_claim = fr_value(fr_add(scalar(2U), fr_wire(challenge)));
    BindingBlock binding;
    binding.label = sumcheck.label;
    binding.operation_kind = kOperationProduct;
    binding.input_claims = {fr_value(scalar(2U)), fr_value(scalar(3U))};
    binding.output_claims = {sumcheck.final_claim};
    binding.sumcheck = sumcheck;
    ProofEnvelope proof;
    proof.bindings.push_back(binding);
    const std::vector<zkdl::byte> proof_bytes = proof.serialize();
    ProofEnvelope parsed;
    assert(ProofEnvelope::deserialize(proof_bytes, parsed, &error));
    Transcript verifier_transcript;
    assert(verify_output_binding(parsed.bindings[0], fr_value(scalar(5U)),
                                 parsed.bindings[0].output_claims[0],
                                 verifier_transcript, &error));
    parsed.bindings[0].output_claims[0] = fr_value(scalar(0U));
    Transcript tampered_transcript;
    assert(!verify_output_binding(parsed.bindings[0], fr_value(scalar(5U)),
                                  fr_value(scalar(0U)), tampered_transcript, &error));

    std::cout << "zkdl CPU core smoke ok" << std::endl;
    return 0;
}
