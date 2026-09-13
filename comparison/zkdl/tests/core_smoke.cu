#include <cassert>
#include <iostream>

#include "../proof.cuh"
#include "../commitment.cuh"

int main() {
    using namespace zkdl;

    const FrWire one = fr_from_u32(1U);
    const FrWire two = fr_from_u32(2U);
    assert(fr_compare(fr_mul(one, two), two) == 0);
    assert(fr_compare(fr_add(two, fr_from_u32(3U)), fr_from_u32(5U)) == 0);

    SumcheckProof sumcheck;
    sumcheck.label = "core-smoke";
    sumcheck.degree = 1;
    SumcheckRound round;
    round.evaluations.push_back(fr_value(fr_from_u32(2U)));
    round.evaluations.push_back(fr_value(fr_from_u32(3U)));
    sumcheck.rounds.push_back(round);

    Transcript challenge_transcript;
    challenge_transcript.absorb_fr(sumcheck.label + ".round", round.evaluations);
    const Fr_t challenge = challenge_transcript.challenge(sumcheck.label + ".challenge");
    sumcheck.final_claim = fr_value(fr_add(fr_from_u32(2U), fr_wire(challenge)));

    Transcript verifier_transcript;
    std::string error;
    assert(verify_sumcheck(sumcheck, fr_value(fr_from_u32(5U)), verifier_transcript,
                           nullptr, &error));

    ProofEnvelope envelope;
    envelope.statement_digest = challenge_transcript.digest();
    envelope.public_output.push_back(0x2aU);
    envelope.sumchecks.push_back(sumcheck);
    const std::vector<zkdl::byte> encoded = envelope.serialize();
    ProofEnvelope decoded;
    assert(ProofEnvelope::deserialize(encoded, decoded, &error));
    assert(decoded.public_output == envelope.public_output);
    assert(decoded.sumchecks.size() == 1U);
    assert(decoded.verify_public_binding(envelope.statement_digest, envelope.public_output, &error));

    EvaluationOpeningProof malformed;
    malformed.generator_size = 1U;
    malformed.inner_point.push_back(fr_value(one));
    malformed.rounds.push_back(G1Jacobian_generator);
    assert(!opening_shape_valid(malformed, &error));

    // Exercise the native commitment/IPA wire path on a four-row tensor.  GPU
    // IPA verification remains an explicit resource blocker in this phase.
    Fr_t values[4] = {
        {1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
        {2U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
        {3U, 0U, 0U, 0U, 0U, 0U, 0U, 0U},
        {4U, 0U, 0U, 0U, 0U, 0U, 0U, 0U}
    };
    FrTensor tensor(4U, values);
    Commitment generators(4U, G1Jacobian_generator);
    G1TensorJacobian commitment = generators.commit(tensor);
    std::vector<Fr_t> opening_point;
    opening_point.push_back(fr_value(fr_from_u32(7U)));
    opening_point.push_back(fr_value(fr_from_u32(11U)));
    opening_point.push_back(fr_value(fr_from_u32(13U)));
    opening_point.push_back(fr_value(fr_from_u32(17U)));
    const zkdl::CpuG1Point ipa_h = zkdl::cpu_g1_mul(
        zkdl::cpu_g1_generator(), zkdl::fr_from_u32(2U));
    const zkdl::CpuG1Point ipa_u = zkdl::cpu_g1_mul(
        zkdl::cpu_g1_generator(), zkdl::fr_from_u32(3U));
    zkdl::CpuIpaOpeningMetadata metadata;
    const std::vector<zkdl::byte> metadata_label{'c', 'o', 'r', 'e', '-', 'i', 'p', 'a'};
    metadata.commitment_index = 0U;
    metadata.descriptor_digest = zkdl::sha256(metadata_label);
    std::vector<zkdl::byte> registry_label = metadata_label;
    registry_label.push_back('r');
    metadata.registry_digest = zkdl::sha256(registry_label);
    std::vector<zkdl::byte> srs_label = metadata_label;
    srs_label.push_back('s');
    metadata.srs_digest = zkdl::sha256(srs_label);
    metadata.outer_point = {
        zkdl::fr_wire(opening_point[2]), zkdl::fr_wire(opening_point[3])};
    uint opening_size = 0;
    EvaluationOpeningProof retained;
    retained = generators.open_with_ipa_proof(
        tensor, commitment, opening_point, zkdl::cpu_g1_to_native(ipa_h),
        zkdl::cpu_g1_to_native(ipa_u), zkdl::FrWire{}, metadata);
    assert(opening_shape_valid(retained, &error));
    (void)opening_size;
    assert(!generators.verify_opening_gpu(retained, commitment({}), &error));

    std::cout << "zkdl core smoke ok" << std::endl;
    return 0;
}
