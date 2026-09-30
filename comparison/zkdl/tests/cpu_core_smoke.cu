#include <cassert>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../cpu-backend.cuh"

namespace {

zkdl::FrWire scalar(unsigned value) { return zkdl::fr_from_u32(value); }

zkdl::CpuG1Point fixture_generator(const char *domain) {
    std::vector<zkdl::byte> seed;
    for (const char *cursor = domain; *cursor != '\0'; ++cursor) seed.push_back(static_cast<zkdl::byte>(*cursor));
    const std::array<zkdl::byte, 32U> digest = zkdl::sha256(seed);
    zkdl::FrWire scalar_value{};
    for (std::size_t limb = 0U; limb < scalar_value.size(); ++limb) {
        const std::size_t offset = 4U * limb;
        scalar_value[limb] = static_cast<std::uint32_t>(digest[offset]) | (static_cast<std::uint32_t>(digest[offset + 1U]) << 8U) |
                             (static_cast<std::uint32_t>(digest[offset + 2U]) << 16U) | (static_cast<std::uint32_t>(digest[offset + 3U]) << 24U);
    }

    scalar_value.back() &= 0x3fU;
    if (zkdl::fr_compare(scalar_value, zkdl::FrWire{}) == 0U) scalar_value[0] = 1U;
    return zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), scalar_value);
}

bool same_fr(const zkdl::FrWire &a, const zkdl::FrWire &b) { return zkdl::fr_compare(a, b) == 0; }

zkdl::CpuIpaOpeningMetadata test_metadata() {
    const std::vector<zkdl::byte> label{'c', 'p', 'u', '-', 'i', 'p', 'a'};
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = 0U;
    metadata.descriptor_digest = zkdl::sha256(label);
    std::vector<zkdl::byte> registry = label;
    registry.push_back('r');
    metadata.registry_digest = zkdl::sha256(registry);
    std::vector<zkdl::byte> srs = label;
    srs.push_back('s');
    metadata.srs_digest = zkdl::sha256(srs);
    return metadata;
}

}

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
    const CpuG1Point ipa_h = fixture_generator("zkdl.cpu.smoke.H");
    const CpuG1Point ipa_u = fixture_generator("zkdl.cpu.smoke.U");
    assert(!ipa_h.is_zero());
    assert(!ipa_u.is_zero());
    assert(!cpu_g1_equal(ipa_h, ipa_u));
    const CpuIpaOpeningMetadata metadata = test_metadata();
    EvaluationOpeningProof opening = cpu_open(generators, values, point, ipa_h, ipa_u, FrWire{}, metadata);
    std::string error;
    assert(opening_shape_valid(opening, &error));
    const bool opening_verified = cpu_verify_opening(opening, cpu_g1_from_native(opening.commitment_eval), generators, ipa_h, ipa_u, metadata, &error);
    if (!opening_verified) std::cerr << "IPA positive verify failed: " << error << std::endl;
    assert(opening_verified);

    ProofEnvelope envelope;
    envelope.openings.push_back(opening);
    const std::vector<zkdl::byte> encoded = envelope.serialize();
    ProofEnvelope decoded;
    assert(ProofEnvelope::deserialize(encoded, decoded, &error));
    assert(decoded.openings.size() == 1U);
    assert(cpu_verify_opening(decoded.openings[0], cpu_g1_from_native(decoded.openings[0].commitment_eval), generators, ipa_h, ipa_u, metadata, &error));
    decoded.openings[0].value.val[0] ^= 1U;
    assert(!cpu_verify_opening(decoded.openings[0], cpu_g1_from_native(opening.commitment_eval), generators, ipa_h, ipa_u, metadata, &error));
    EvaluationOpeningProof tampered_left = opening;
    tampered_left.rounds[0].x.val[0] ^= 1U;
    assert(!cpu_verify_opening(tampered_left, cpu_g1_from_native(opening.commitment_eval), generators, ipa_h, ipa_u, metadata, &error));
    EvaluationOpeningProof tampered_right = opening;
    tampered_right.rounds[1].y.val[0] ^= 1U;
    assert(!cpu_verify_opening(tampered_right, cpu_g1_from_native(opening.commitment_eval), generators, ipa_h, ipa_u, metadata, &error));

    HyraxOpeningProverResult v3_result = cpu_open_hyrax_v3(generators, values, point, ipa_h, ipa_u, FrWire{}, metadata);
    assert(hyrax_opening_v3_shape_valid(v3_result.proof, &error));
    const CpuG1Point v3_xi = cpu_g1_from_native(v3_result.proof.xi);
    const CpuG1Point v3_tau = cpu_g1_from_native(v3_result.proof.tau);
    assert(cpu_verify_hyrax_v3(v3_result.proof, v3_xi, v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));

    std::vector<zkdl::byte> v3_bytes;
    serialize_hyrax_opening_v3(v3_result.proof, v3_bytes);
    HyraxOpeningProofV3 v3_decoded;
    size_t v3_offset = 0U;
    assert(deserialize_hyrax_opening_v3(v3_bytes, v3_offset, v3_decoded));
    assert(v3_offset == v3_bytes.size());
    assert(cpu_verify_hyrax_v3(v3_decoded, v3_xi, v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));

    HyraxOpeningProofV3 v3_tampered_tau = v3_result.proof;
    v3_tampered_tau.tau.x.val[0] ^= 1U;
    assert(!cpu_verify_hyrax_v3(v3_tampered_tau, v3_xi, v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));
    HyraxOpeningProofV3 v3_tampered_round = v3_result.proof;
    v3_tampered_round.rounds[0].x.val[0] ^= 1U;
    assert(!cpu_verify_hyrax_v3(v3_tampered_round, v3_xi, v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));
    HyraxOpeningProofV3 v3_tampered_z1 = v3_result.proof;
    v3_tampered_z1.z1.val[0] ^= 1U;
    assert(!cpu_verify_hyrax_v3(v3_tampered_z1, v3_xi, v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));
    HyraxOpeningProofV3 v3_tampered_metadata = v3_result.proof;
    v3_tampered_metadata.descriptor_digest[0] ^= 1U;
    assert(!cpu_verify_hyrax_v3(v3_tampered_metadata, v3_xi, v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));
    assert(!cpu_verify_hyrax_v3(v3_result.proof, v3_xi, cpu_g1_add(v3_tau, generator), generators, point, ipa_h, ipa_u, metadata, &error));
    assert(!cpu_verify_hyrax_v3(v3_result.proof, cpu_g1_add(v3_xi, generator), v3_tau, generators, point, ipa_h, ipa_u, metadata, &error));

    const FrWire public_evaluation = cpu_ipa_inner_product(values, cpu_ipa_mle_weights(point));
    HyraxOpeningProverResult v3_public = cpu_open_hyrax_v3_public(generators, values, point, ipa_h, ipa_u, FrWire{}, metadata);
    const CpuG1Point public_xi = cpu_g1_from_native(v3_public.proof.xi);
    const CpuG1Point expected_public_tau = cpu_g1_mul(ipa_u, public_evaluation);
    assert(cpu_g1_equal(cpu_g1_from_native(v3_public.proof.tau), expected_public_tau));
    assert(cpu_verify_hyrax_v3_public(v3_public.proof, public_xi, public_evaluation, generators, point, ipa_h, ipa_u, metadata, &error));
    assert(!cpu_verify_hyrax_v3_public(v3_public.proof, public_xi, fr_add(public_evaluation, scalar(1U)), generators, point, ipa_h, ipa_u, metadata, &error));

    const CpuHidingCommitment model_commitment("model-family", 1U, generators, generator, values, scalar(17U), scalar(23U));
    const CpuHidingCommitment dynamic_commitment("dynamic-family", 2U, generators, generator, values, scalar(19U), scalar(29U));
    assert(!same_fr(model_commitment.randomness, dynamic_commitment.randomness));
    assert(cpu_g1_equal(model_commitment.commitment, model_commitment.commitment));
    assert(cpu_g1_equal(dynamic_commitment.commitment, dynamic_commitment.commitment));
    assert(validate_hiding_commitments(std::vector<CommitmentRecord>{model_commitment.record(), dynamic_commitment.record()}, &error));
    CommitmentRecord tampered = dynamic_commitment.record();
    tampered.commitment.x.val[0] ^= 1U;
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
    assert(verify_output_binding(parsed.bindings[0], fr_value(scalar(5U)), parsed.bindings[0].output_claims[0], verifier_transcript, &error));
    parsed.bindings[0].output_claims[0] = fr_value(scalar(0U));
    Transcript tampered_transcript;
    assert(!verify_output_binding(parsed.bindings[0], fr_value(scalar(5U)), fr_value(scalar(0U)), tampered_transcript, &error));

    std::cout << "zkdl CPU core smoke ok" << std::endl;
    return 0;
}
