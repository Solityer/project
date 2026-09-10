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

    // Exercise the native commitment/opening path and the independent CUDA
    // folding verifier on a four-entry tensor.  The second check must fail
    // after a single verifier-facing point is tampered with.
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
    uint opening_size = 0;
    EvaluationOpeningProof retained;
    generators.open(tensor, commitment, opening_point, opening_size, &retained);
    assert(opening_shape_valid(retained, &error));
    const G1Jacobian_t expected_eval = commitment(std::vector<Fr_t>());
    assert(generators.verify_opening_gpu(retained, expected_eval, &error));
    retained.rounds[0].x.val[0] ^= 1U;
    assert(!generators.verify_opening_gpu(retained, expected_eval, &error));

    std::cout << "zkdl core smoke ok" << std::endl;
    return 0;
}
