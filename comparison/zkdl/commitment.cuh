#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include "cpu-backend.cuh"

namespace zkdl {

/* Process-local evidence counter for native CUDA cryptographic kernels.  It
 * is incremented only after a successful CUDA synchronization in a proof
 * operation; callers use a before/after sample as a fail-closed provenance
 * check. */
inline std::uint64_t& gpu_crypto_kernel_counter() noexcept {
    static std::uint64_t count = 0U;
    return count;
}

inline std::uint64_t gpu_crypto_kernel_invocations() noexcept {
    return gpu_crypto_kernel_counter();
}

}  // namespace zkdl

class Commitment: public G1TensorJacobian
{   
    public:
    using G1TensorJacobian::G1TensorJacobian;

    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;
    using G1TensorJacobian::operator*=;

    G1TensorJacobian commit(const FrTensor& t) const;

    static Commitment random_generators(uint size);

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u, uint& proof_size_count) const;

    // Retain the complete native Hyrax folding transcript.  The old open()
    // helper only returned the value and counted bytes, which made an
    // independent verifier impossible.  This overload preserves the old API
    // while exposing the exact verifier-facing opening message.
    zkdl::EvaluationOpeningProof open_with_proof(const FrTensor& t,
                                                 const G1TensorJacobian& c,
                                                 const vector<Fr_t>& u) const;
    zkdl::EvaluationOpeningProof open_with_ipa_proof(
        const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u,
        const G1Jacobian_t& blinding_generator,
        const G1Jacobian_t& u_generator,
        const zkdl::FrWire& randomness,
        const zkdl::CpuIpaOpeningMetadata& metadata) const;
    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u,
              uint& proof_size_count, zkdl::EvaluationOpeningProof* retained) const;

    bool verify_opening_gpu(const zkdl::EvaluationOpeningProof& opening,
                            const G1Jacobian_t& expected_commitment_eval,
                            std::string* error = nullptr) const;

    static Fr_t me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);
};

KERNEL void sum_axis_n_optimized(GLOBAL G1Jacobian_t* arr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m) return;
    
    __shared__ G1Jacobian_t shared_data[G1NumThread];
    uint local_id = threadIdx.x;
    
    G1Jacobian_t sum = (local_id < n)
        ? arr[gid * n + local_id]
        : blstrs__g1__G1Affine_ZERO;
    for (uint i = local_id + G1NumThread; i < n; i += G1NumThread) {
        sum = blstrs__g1__G1Affine_add(sum, arr[gid * n + i]);
    }
    shared_data[local_id] = sum;
    
    __syncthreads();
    
    // Reduce shared_data to a single value
    for (uint s = G1NumThread / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] = blstrs__g1__G1Affine_add(shared_data[local_id], shared_data[local_id + s]);
        }
        __syncthreads();
    }
    
    if (local_id == 0) arr_out[gid] = shared_data[0];
}

G1TensorJacobian Commitment::commit(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Incompatible dimensions");

    uint m = t.size / size;
    // FrTensor wire values are canonical residues.  G1TensorJacobian::operator*
    // consumes the same canonical scalar encoding; converting here with
    // Scalar_unmont would apply R^-1 a second time and make openings fail.
    G1TensorJacobian temp = (*this) * t;
    G1TensorJacobian out(m);
    sum_axis_n_optimized<<<(m+G1NumThread-1)/G1NumThread,G1NumThread>>>(temp.gpu_data, out.gpu_data, size, m);
    cudaDeviceSynchronize();
    return out;
}

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // canonical scalar wires throughout
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint new_size) // old size is even for each fold
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    const G1Jacobian_t g0 = generators[gid0];
    const G1Jacobian_t g1 = generators[gid1];
    const Fr_t s0 = scalars[gid0];
    const Fr_t s1 = scalars[gid1];
    const Fr_t delta = blstrs__scalar__Scalar_sub(s1, s0);
    new_scalars[gid] = blstrs__scalar__Scalar_add(s0, Fr_field_mul(u, delta));

    // Keep every point operation in the explicit native Jacobian helper.
    // This is algebraically identical to g1 + (g0-g1)u and avoids the
    // affine-named generic add on the inverse-generator case.
    const G1Jacobian_t generator_delta = G1Jacobian_add_safe(g0, G1Jacobian_minus(g1));
    const G1Jacobian_t scaled_delta = G1Jacobian_mul(generator_delta, u);
    new_generators[gid] = G1Jacobian_add_safe(g1, scaled_delta);

    const G1Jacobian_t term0 = G1Jacobian_mul(g0, s0);
    const G1Jacobian_t term1 = G1Jacobian_mul(g1, s1);
    temp_out[gid] = G1Jacobian_add_safe(term0, term1);
    temp_out0[gid] = G1Jacobian_mul(g1, s0);
    temp_out1[gid] = G1Jacobian_mul(g0, s1);
}

Fr_t Commitment::me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    if (t.size != generators.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end)
    {
        proof.push_back(generators(0));
        return t(0);
    }
    uint new_size = t.size / 2;
    FrTensor new_scalars(new_size);
    Commitment new_generators(new_size);
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size+G1NumThread-1)/G1NumThread,G1NumThread>>>(t.gpu_data, generators.gpu_data, *begin, 
    new_scalars.gpu_data, new_generators.gpu_data, temp.gpu_data, temp0.gpu_data, temp1.gpu_data, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
    return me_open(new_scalars, new_generators, begin + 1, end, proof);
}

Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u, uint& proof_size_count) const
{
    return open(t, com, u, proof_size_count, nullptr);
}

zkdl::EvaluationOpeningProof Commitment::open_with_proof(const FrTensor& t,
                                                          const G1TensorJacobian& com,
                                                          const vector<Fr_t>& u) const
{
    (void)t;
    (void)com;
    (void)u;
    throw std::invalid_argument("authenticated IPA H/U basis is required");
}

zkdl::EvaluationOpeningProof Commitment::open_with_ipa_proof(
    const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u,
    const G1Jacobian_t& blinding_generator,
    const G1Jacobian_t& u_generator,
    const zkdl::FrWire& randomness,
    const zkdl::CpuIpaOpeningMetadata& metadata) const
{
    if (u.empty()) throw std::runtime_error("Opening point must not be empty");
    const std::uint32_t outer_bits = ceilLog2(com.size);
    if (outer_bits > u.size()) throw std::runtime_error("Opening point has too few coordinates");
    const vector<Fr_t> u_out(u.end() - outer_bits, u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - outer_bits);
    if (size != (std::size_t(1) << u_in.size()) ||
        t.size != static_cast<std::uint64_t>(size) * com.size ||
        com.size == 0U) throw std::runtime_error("Incompatible IPA opening dimensions");

    const std::vector<G1Jacobian_t> native_generators = to_host();
    const std::vector<Fr_t> host_values = t.to_host();
    std::vector<zkdl::CpuG1Point> generators;
    generators.reserve(native_generators.size());
    for (const auto& generator : native_generators)
        generators.push_back(zkdl::cpu_g1_from_native(generator));
    std::vector<zkdl::FrWire> inner_values(size);
    std::vector<zkdl::FrWire> inner_point;
    inner_point.reserve(u_in.size());
    for (const auto& coordinate : u_in) inner_point.push_back(zkdl::fr_wire(coordinate));
    std::vector<zkdl::FrWire> outer_point;
    outer_point.reserve(u_out.size());
    for (const auto& coordinate : u_out) outer_point.push_back(zkdl::fr_wire(coordinate));
    if (metadata.outer_point.size() != outer_point.size())
        throw std::invalid_argument("IPA outer metadata does not match tensor view");
    for (std::size_t i = 0U; i < outer_point.size(); ++i)
        if (zkdl::fr_compare(metadata.outer_point[i], outer_point[i]) != 0)
            throw std::invalid_argument("IPA outer metadata differs from tensor view");
    for (uint column = 0U; column < size; ++column) {
        std::vector<zkdl::FrWire> rows(com.size);
        for (uint row = 0U; row < com.size; ++row)
            rows[row] = zkdl::fr_wire(host_values[static_cast<std::size_t>(row) * size + column]);
        for (const auto& coordinate : outer_point) {
            const zkdl::FrWire one_minus = zkdl::fr_sub(
                zkdl::fr_from_u32(1U), coordinate);
            std::vector<zkdl::FrWire> next((rows.size() + 1U) / 2U);
            for (std::size_t row = 0U; row < next.size(); ++row) {
                const std::size_t left = 2U * row;
                next[row] = zkdl::fr_mul(rows[left], one_minus);
                if (left + 1U < rows.size())
                    next[row] = zkdl::fr_add(next[row],
                        zkdl::fr_mul(rows[left + 1U], coordinate));
            }
            rows.swap(next);
        }
        inner_values[column] = rows.front();
    }
    zkdl::EvaluationOpeningProof opening = zkdl::cpu_open(
        generators, inner_values,
        inner_point,
        zkdl::cpu_g1_from_native(blinding_generator),
        zkdl::cpu_g1_from_native(u_generator), randomness, metadata);
    std::vector<zkdl::CpuG1Point> row_commitments;
    const std::vector<G1Jacobian_t> native_commitments = com.to_host();
    row_commitments.reserve(native_commitments.size());
    for (const auto& row : native_commitments)
        row_commitments.push_back(zkdl::cpu_g1_from_native(row));
    if (row_commitments.size() != (std::size_t(1U) << outer_point.size()))
        throw std::runtime_error("IPA outer commitment domain is not a power of two");
    const zkdl::CpuG1Point expected_commitment = zkdl::cpu_g1_sum(
        row_commitments, zkdl::cpu_ipa_mle_weights(outer_point));
    if (!zkdl::cpu_g1_equal(
            zkdl::cpu_g1_from_native(opening.commitment_eval), expected_commitment))
        throw std::runtime_error("IPA opening is not bound to the outer tensor commitment");
    opening.outer_point = u_out;
    return opening;
}

Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com,
                      const vector<Fr_t>& u, uint& proof_size_count,
                      zkdl::EvaluationOpeningProof* retained) const
{
    zkdl::EvaluationOpeningProof opening = open_with_proof(t, com, u);
    const std::vector<zkdl::byte> encoded = [&]() {
        std::vector<zkdl::byte> bytes;
        zkdl::serialize_opening(opening, bytes);
        return bytes;
    }();
    if (encoded.size() > std::numeric_limits<uint>::max() - proof_size_count)
        throw std::overflow_error("IPA opening size exceeds uint");
    proof_size_count += static_cast<uint>(encoded.size());
    if (retained != nullptr) *retained = opening;
    return opening.value;
}

Commitment Commitment::random_generators(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out; 
}

DEVICE bool zkdl_g1_equal(const G1Jacobian_t& a, const G1Jacobian_t& b)
{
    const Fp_t zero = blstrs__fp__Fp_ZERO;
    const bool a_zero = blstrs__fp__Fp_eq(a.z, zero);
    const bool b_zero = blstrs__fp__Fp_eq(b.z, zero);
    if (a_zero || b_zero) return a_zero && b_zero;

    // Native group operations return projective Jacobian points, so comparing
    // (x,y,z) byte-for-byte is not a valid equality test.  Cross-multiply the
    // affine coordinates in the native Montgomery Fp arithmetic instead.
    const Fp_t z1z1 = blstrs__fp__Fp_sqr(a.z);
    const Fp_t z2z2 = blstrs__fp__Fp_sqr(b.z);
    const Fp_t x1 = blstrs__fp__Fp_mul(a.x, z2z2);
    const Fp_t x2 = blstrs__fp__Fp_mul(b.x, z1z1);
    if (!blstrs__fp__Fp_eq(x1, x2)) return false;
    const Fp_t y1 = blstrs__fp__Fp_mul(a.y,
        blstrs__fp__Fp_mul(b.z, z2z2));
    const Fp_t y2 = blstrs__fp__Fp_mul(b.y,
        blstrs__fp__Fp_mul(a.z, z1z1));
    return blstrs__fp__Fp_eq(y1, y2);
}

/* Verify the native Hyrax folding equations with one independent CUDA
 * verifier thread.  Coefficients are prepared in the host field helper and
 * are passed as canonical scalars; the group implementation remains the
 * official BLS12-381 CUDA primitive. */
KERNEL void zkdl_opening_verify_kernel(
    GLOBAL G1Jacobian_t* folds, GLOBAL Fr_t* coefficients,
    G1Jacobian_t opening_initial, G1Jacobian_t expected_initial,
    G1Jacobian_t final_generator,
    Fr_t value, uint fold_count, GLOBAL int* accepted)
{
    if (GET_GLOBAL_ID() != 0) return;
    if (!zkdl_g1_equal(opening_initial, expected_initial)) {
        *accepted = 0;
        return;
    }
    G1Jacobian_t current = opening_initial;
    for (uint i = 0; i < fold_count; ++i) {
        const G1Jacobian_t& p = folds[3U * i];
        const G1Jacobian_t& a = folds[3U * i + 1U];
        const G1Jacobian_t& b = folds[3U * i + 2U];
        G1Jacobian_t next = G1Jacobian_mul(a, coefficients[3U * i]);
        next = G1Jacobian_add_safe(next, G1Jacobian_mul(p, coefficients[3U * i + 1U]));
        next = G1Jacobian_add_safe(next, G1Jacobian_mul(b, coefficients[3U * i + 2U]));
        current = next;
    }
    const G1Jacobian_t expected_final = G1Jacobian_mul(final_generator, value);
    *accepted = zkdl_g1_equal(current, expected_final) ? 1 : 0;
}

/* Version-2 IPA verifier.  The Fiat--Shamir transcript and challenge
 * derivation remain on the host, but the group folding equations and terminal
 * equation are checked by the native CUDA BLS12-381 primitives in one
 * independent cryptographic kernel. */
KERNEL void zkdl_ipa_v2_verify_kernel(
    GLOBAL G1Jacobian_t* current_generators,
    GLOBAL G1Jacobian_t* scratch_generators,
    GLOBAL Fr_t* current_b,
    GLOBAL Fr_t* scratch_b,
    GLOBAL G1Jacobian_t* rounds,
    GLOBAL Fr_t* challenges,
    GLOBAL Fr_t* inverse_challenges,
    G1Jacobian_t commitment_eval,
    G1Jacobian_t expected_commitment_eval,
    G1Jacobian_t u_generator,
    Fr_t value,
    G1Jacobian_t expected_final_generator,
    Fr_t final_a,
    Fr_t final_b,
    uint fold_count,
    uint padded_size,
    GLOBAL int* accepted)
{
    if (GET_GLOBAL_ID() != 0) return;
    if (!zkdl_g1_equal(commitment_eval, expected_commitment_eval)) {
        *accepted = 0;
        return;
    }
    G1Jacobian_t current_p = G1Jacobian_add_safe(
        commitment_eval, G1Jacobian_mul(u_generator, value));
    G1Jacobian_t* generators = current_generators;
    G1Jacobian_t* next_generators = scratch_generators;
    Fr_t* b = current_b;
    Fr_t* next_b = scratch_b;
    uint current_size = padded_size;
    for (uint round = 0U; round < fold_count; ++round) {
        const Fr_t x = challenges[round];
        const Fr_t x_inv = inverse_challenges[round];
        const Fr_t x_sq = blstrs__scalar__Scalar_mul(x, x);
        const Fr_t x_inv_sq = blstrs__scalar__Scalar_mul(x_inv, x_inv);
        current_p = G1Jacobian_add_safe(
            current_p, G1Jacobian_mul(rounds[2U * round], x_sq));
        current_p = G1Jacobian_add_safe(
            current_p, G1Jacobian_mul(rounds[2U * round + 1U], x_inv_sq));
        const uint half = current_size / 2U;
        for (uint i = 0U; i < half; ++i) {
            next_generators[i] = G1Jacobian_add_safe(
                G1Jacobian_mul(generators[i], x_inv),
                G1Jacobian_mul(generators[half + i], x));
            next_b[i] = blstrs__scalar__Scalar_add(
                blstrs__scalar__Scalar_mul(b[i], x_inv),
                blstrs__scalar__Scalar_mul(b[half + i], x));
        }
        G1Jacobian_t* generator_swap = generators;
        generators = next_generators;
        next_generators = generator_swap;
        Fr_t* b_swap = b;
        b = next_b;
        next_b = b_swap;
        current_size = half;
    }
    if (!blstrs__scalar__Scalar_eq(b[0], final_b) ||
        !zkdl_g1_equal(generators[0], expected_final_generator)) {
        *accepted = 0;
        return;
    }
    G1Jacobian_t expected = G1Jacobian_add_safe(
        G1Jacobian_mul(generators[0], final_a),
        G1Jacobian_mul(u_generator,
                       blstrs__scalar__Scalar_mul(final_a, final_b)));
    *accepted = zkdl_g1_equal(current_p, expected) ? 1 : 0;
}

inline bool Commitment::verify_opening_gpu(const zkdl::EvaluationOpeningProof& opening,
                                           const G1Jacobian_t& expected_commitment_eval,
                                           std::string* error) const
{
    try {
        if (!zkdl::opening_shape_valid(opening, error) ||
            opening.generator_size != size || opening.padded_generator_size !=
                zkdl::cpu_ipa_padded_size(opening.generator_size) ||
            opening.rounds.size() !=
                2U * static_cast<std::size_t>(__builtin_ctz(opening.padded_generator_size))) {
            if (error && error->empty()) *error = "GPU IPA opening shape mismatch";
            return false;
        }
        const std::uint32_t padded_size = opening.padded_generator_size;
        const std::uint32_t fold_count =
            static_cast<std::uint32_t>(__builtin_ctz(padded_size));
        std::vector<G1Jacobian_t> host_generators = to_host();
        host_generators.resize(padded_size, G1Jacobian_t{});
        host_generators[opening.generator_size] = opening.blinding_generator;
        std::vector<Fr_t> host_b(padded_size, blstrs__scalar__Scalar_ZERO);
        const std::vector<zkdl::FrWire> weights = zkdl::cpu_ipa_mle_weights(
            [&]() {
                std::vector<zkdl::FrWire> point;
                point.reserve(opening.inner_point.size());
                for (const auto& coordinate : opening.inner_point)
                    point.push_back(zkdl::fr_wire(coordinate));
                return point;
            }());
        for (std::size_t i = 0U; i < weights.size(); ++i)
            host_b[i] = zkdl::fr_value(weights[i]);
        std::vector<Fr_t> host_challenges, host_inverse;
        host_challenges.reserve(fold_count);
        host_inverse.reserve(fold_count);
        zkdl::Transcript transcript("zkdl-ipa-opening-v2");
        zkdl::cpu_ipa_absorb_header(transcript, opening);
        for (std::uint32_t round = 0U; round < fold_count; ++round) {
            std::vector<zkdl::byte> round_bytes;
            zkdl::append_g1(round_bytes, opening.rounds[2U * round]);
            zkdl::append_g1(round_bytes, opening.rounds[2U * round + 1U]);
            transcript.absorb("zkdl.ipa.round", round_bytes);
            const zkdl::FrWire challenge = zkdl::cpu_ipa_nonzero_challenge(
                transcript, "zkdl.ipa.challenge");
            host_challenges.push_back(zkdl::fr_value(challenge));
            host_inverse.push_back(zkdl::fr_value(zkdl::cpu_fr_inv(challenge)));
        }
        G1TensorJacobian device_generators(padded_size, host_generators.data());
        G1TensorJacobian device_scratch(padded_size);
        FrTensor device_b(padded_size, host_b.data());
        FrTensor device_b_scratch(padded_size);
        G1TensorJacobian device_rounds(
            static_cast<uint>(opening.rounds.size()), opening.rounds.data());
        FrTensor device_challenges(fold_count, host_challenges.data());
        FrTensor device_inverse(fold_count, host_inverse.data());
        int* accepted = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&accepted), sizeof(int)) != cudaSuccess)
            throw std::runtime_error("GPU IPA verifier allocation failed");
        const int initial = 0;
        cudaMemcpy(accepted, &initial, sizeof(int), cudaMemcpyHostToDevice);
        zkdl_ipa_v2_verify_kernel<<<1, 1>>>(
            device_generators.device_data(), device_scratch.device_data(),
            device_b.device_data(), device_b_scratch.device_data(),
            device_rounds.device_data(), device_challenges.device_data(),
            device_inverse.device_data(), opening.commitment_eval,
            expected_commitment_eval, opening.u_generator, opening.value,
            opening.final_generator, opening.final_a, opening.final_b,
            fold_count, padded_size, accepted);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess)
            throw std::runtime_error(std::string("GPU IPA verifier launch failed: ") +
                                     cudaGetErrorString(launch_status));
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess)
            throw std::runtime_error(std::string("GPU IPA verifier synchronization failed: ") +
                                     cudaGetErrorString(sync_status));
        zkdl::gpu_crypto_kernel_counter()++;
        int accepted_host = 0;
        cudaMemcpy(&accepted_host, accepted, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(accepted);
        if (accepted_host != 1) {
            if (error) *error = "GPU IPA terminal/folding equation rejected";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

#endif
