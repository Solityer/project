#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "../zkmatmul.cuh"

namespace {

Fr_t scalar(unsigned value) {
    Fr_t out{};
    out.val[0] = value;
    return out;
}

std::vector<Fr_t> tensor_values(const std::vector<unsigned>& values) {
    std::vector<Fr_t> out;
    out.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) out.push_back(scalar(values[i]));
    return out;
}

zkdl::MatMulCpuCommitmentView cpu_view(const Commitment& generators,
                                       const std::vector<G1Jacobian_t>& commitment) {
    zkdl::MatMulCpuCommitmentView out;
    const std::vector<G1Jacobian_t> native_generators = generators.to_host();
    out.generators.reserve(native_generators.size());
    out.commitments.reserve(commitment.size());
    for (size_t i = 0; i < native_generators.size(); ++i)
        out.generators.push_back(zkdl::cpu_g1_from_native(native_generators[i]));
    for (size_t i = 0; i < commitment.size(); ++i)
        out.commitments.push_back(zkdl::cpu_g1_from_native(commitment[i]));
    return out;
}

}  // namespace

int main() {
    const zkdl::MatMulShape shape{2U, 2U, 2U, 2U};
    const std::vector<Fr_t> a_values = tensor_values({1U, 2U, 3U, 4U,
                                                       2U, 1U, 0U, 3U});
    const std::vector<Fr_t> b_values = tensor_values({5U, 6U, 7U, 8U,
                                                       1U, 4U, 2U, 5U});
    const std::vector<Fr_t> c_values = tensor_values({19U, 22U, 43U, 50U,
                                                       4U, 13U, 6U, 15U});
    FrTensor a(shape.a_size(), a_values.data());
    FrTensor b(shape.b_size(), b_values.data());
    FrTensor c(shape.c_size(), c_values.data());
    Commitment gen_a = Commitment::random_generators(shape.n);
    Commitment gen_b = Commitment::random_generators(shape.k);
    Commitment gen_c = Commitment::random_generators(shape.k);

    std::uint32_t proof_size = 0U;
    zkdl::MatMulProof proof = zkdl::matmul_prove(
        a, b, c, shape, gen_a, gen_b, gen_c, &proof_size);
    assert(proof_size == proof.serialize().size());

    const std::vector<zkdl::byte> encoded = proof.serialize();
    zkdl::MatMulProof decoded;
    std::string error;
    assert(zkdl::MatMulProof::deserialize(encoded, decoded, &error));

    G1TensorJacobian com_a(decoded.commitment_a.size(), decoded.commitment_a.data());
    G1TensorJacobian com_b(decoded.commitment_b.size(), decoded.commitment_b.data());
    G1TensorJacobian com_c(decoded.commitment_c.size(), decoded.commitment_c.data());
    assert(zkdl::matmul_verify_gpu(decoded, shape, gen_a, com_a, gen_b, com_b,
                                   &gen_c, &com_c, nullptr, &error));

    const zkdl::MatMulCpuCommitmentView cpu_a = cpu_view(gen_a, decoded.commitment_a);
    const zkdl::MatMulCpuCommitmentView cpu_b = cpu_view(gen_b, decoded.commitment_b);
    const zkdl::MatMulCpuCommitmentView cpu_c = cpu_view(gen_c, decoded.commitment_c);
    assert(zkdl::matmul_verify_cpu(decoded, shape, cpu_a, cpu_b, &cpu_c,
                                   nullptr, &error));

    zkdl::MatMulProof public_output = zkdl::matmul_prove_public(
        a, b, c, shape, gen_a, gen_b, &proof_size);
    assert(!public_output.has_c_opening);
    assert(zkdl::matmul_verify_cpu(public_output, shape, cpu_a, cpu_b,
                                   nullptr, &public_output.output_claim, &error));

    zkdl::MatMulProof bad_c = decoded;
    bad_c.commitment_c[0].x.val[0] ^= 1U;
    assert(!zkdl::matmul_verify_cpu(bad_c, shape, cpu_a, cpu_b, &cpu_c,
                                    nullptr, &error));

    zkdl::MatMulShape bad_shape = shape;
    bad_shape.m = 1U;
    assert(!zkdl::matmul_verify_cpu(decoded, bad_shape, cpu_a, cpu_b, &cpu_c,
                                    nullptr, &error));

    zkdl::MatMulProof bad_transcript = decoded;
    bad_transcript.transcript_digest[0] ^= 1U;
    assert(!zkdl::matmul_verify_cpu(bad_transcript, shape, cpu_a, cpu_b,
                                    &cpu_c, nullptr, &error));

    zkdl::MatMulProof bad_opening = decoded;
    bad_opening.opening_a.value.val[0] ^= 1U;
    assert(!zkdl::matmul_verify_cpu(bad_opening, shape, cpu_a, cpu_b,
                                    &cpu_c, nullptr, &error));

    std::cout << "zkdl zkMatMul proof smoke ok (" << proof_size << " bytes)" << std::endl;
    return 0;
}
