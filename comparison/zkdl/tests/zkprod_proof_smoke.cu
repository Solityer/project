#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "../zkprod-proof.cuh"
#include "../zkprod.cuh"

namespace {

zkdl::FrWire scalar(std::uint32_t value) { return zkdl::fr_from_u32(value); }

std::vector<zkdl::CpuG1Point> scaled_generators(std::uint32_t first, std::uint32_t second) {
    const zkdl::CpuG1Point generator = zkdl::cpu_g1_generator();
    return {zkdl::cpu_g1_mul(generator, scalar(first)), zkdl::cpu_g1_mul(generator, scalar(second))};
}

std::vector<G1Jacobian_t> native_points(const std::vector<zkdl::CpuG1Point> &points) {
    std::vector<G1Jacobian_t> result;
    result.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) result.push_back(zkdl::cpu_g1_to_native(points[i]));
    return result;
}

}

int main() {
    constexpr std::uint32_t num = 2U;
    constexpr std::uint32_t dim = 2U;
    const std::vector<zkdl::FrWire> a{scalar(1U), scalar(2U), scalar(3U), scalar(4U)};
    const std::vector<zkdl::FrWire> b{scalar(5U), scalar(6U), scalar(7U), scalar(8U)};
    const std::vector<zkdl::FrWire> c{scalar(5U), scalar(12U), scalar(21U), scalar(32U)};
    const std::vector<zkdl::CpuG1Point> ga = scaled_generators(1U, 2U);
    const std::vector<zkdl::CpuG1Point> gb = scaled_generators(3U, 5U);
    const std::vector<zkdl::CpuG1Point> gc = scaled_generators(7U, 11U);

    zkdl_prod::Proof proof = zkdl_prod::prove_host(num, dim, a, b, c, ga, gb, gc);
    const std::vector<G1Jacobian_t> expected_a = zkdl_prod::commitment_rows(ga, a, num, dim);
    const std::vector<G1Jacobian_t> expected_b = zkdl_prod::commitment_rows(gb, b, num, dim);
    const std::vector<G1Jacobian_t> expected_c = zkdl_prod::commitment_rows(gc, c, num, dim);
    const std::vector<zkdl::byte> encoded = proof.serialize();
    std::vector<zkdl::byte> sc_encoded;
    zkdl::serialize_sumcheck(proof.sumcheck, sc_encoded);
    zkdl::SumcheckProof sc_decoded;
    size_t sc_offset = 0;
    const bool sc_ok = zkdl::deserialize_sumcheck(sc_encoded, sc_offset, sc_decoded);
    assert(sc_ok && sc_offset == sc_encoded.size());
    zkdl_prod::Proof decoded;
    std::string error;
    assert(zkdl_prod::Proof::deserialize(encoded, decoded, &error));
    const bool positive_ok = zkdl_prod::verify_host(decoded, num, dim, ga, gb, gc, expected_a, expected_b, expected_c, &error);
    if (!positive_ok) std::cerr << "positive verify error: " << error << std::endl;
    assert(positive_ok);

    zkdl_prod::Proof c_tampered = decoded;
    c_tampered.commitment_c[0].x.val[0] ^= 1U;
    assert(!zkdl_prod::verify_host(c_tampered, num, dim, ga, gb, gc, expected_a, expected_b, expected_c, &error));

    zkdl_prod::Proof transcript_tampered = decoded;
    transcript_tampered.transcript_digest[0] ^= 1U;
    assert(!zkdl_prod::verify_host(transcript_tampered, num, dim, ga, gb, gc, expected_a, expected_b, expected_c, &error));

    zkdl_prod::Proof opening_tampered = decoded;
    opening_tampered.opening_a.rounds[0].x.val[0] ^= 1U;
    assert(!zkdl_prod::verify_host(opening_tampered, num, dim, ga, gb, gc, expected_a, expected_b, expected_c, &error));

    zkdl_prod::Proof output_tampered = decoded;
    output_tampered.opening_c.value.val[0] ^= 1U;
    assert(!zkdl_prod::verify_host(output_tampered, num, dim, ga, gb, gc, expected_a, expected_b, expected_c, &error));

    const std::vector<G1Jacobian_t> native_ga = native_points(ga);
    const std::vector<G1Jacobian_t> native_gb = native_points(gb);
    const std::vector<G1Jacobian_t> native_gc = native_points(gc);
    Commitment gen_a(dim, native_ga.data());
    Commitment gen_b(dim, native_gb.data());
    Commitment gen_c(dim, native_gc.data());
    G1TensorJacobian com_a(num, expected_a.data());
    G1TensorJacobian com_b(num, expected_b.data());
    G1TensorJacobian com_c(num, expected_c.data());
    Timer prover_timer, verifier_timer;
    std::uint32_t commitment_size = 0U;
    Fr_t host_a[4], host_b[4], host_c[4];
    for (size_t i = 0; i < 4U; ++i) {
        host_a[i] = zkdl::fr_value(a[i]);
        host_b[i] = zkdl::fr_value(b[i]);
        host_c[i] = zkdl::fr_value(c[i]);
    }
    FrTensor tensor_a(4U, host_a);
    FrTensor tensor_b(4U, host_b);
    FrTensor tensor_c(4U, host_c);
    zkProd product(tensor_a, tensor_b, tensor_c, num, dim, gen_a, gen_b, gen_c, prover_timer, verifier_timer, commitment_size);
    const zkdl_prod::Proof &native_proof = product.prove(gen_a, gen_b, gen_c, commitment_size);
    assert(zkProd::verify_cpu(native_proof, gen_a, gen_b, gen_c, product.comA, product.comB, product.comC, &error));
    assert(zkProd::verify_gpu(native_proof, gen_a, gen_b, gen_c, product.comA, product.comB, product.comC, &error));

    std::cout << "zkProd proof smoke ok" << std::endl;
    return 0;
}
