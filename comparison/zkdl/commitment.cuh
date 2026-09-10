#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"

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
    if (u.size() < 1) throw std::runtime_error("Opening point must not be empty");
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));

    auto g_temp = com(u_out);
    if (size != (1 << u_in.size())) throw std::runtime_error("Incompatible dimensions");
    vector<G1Jacobian_t> proof;
    auto val = me_open(t.partial_me(u_out, 1 << u_in.size()), *this, u_in.begin(), u_in.end(), proof);
    zkdl::EvaluationOpeningProof opening;
    opening.generator_size = size;
    opening.inner_point = u_in;
    opening.outer_point = u_out;
    opening.value = val;
    opening.commitment_eval = g_temp;
    opening.rounds = proof;
    return opening;
}

Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com,
                      const vector<Fr_t>& u, uint& proof_size_count,
                      zkdl::EvaluationOpeningProof* retained) const
{
    zkdl::EvaluationOpeningProof opening = open_with_proof(t, com, u);
    // G1 points are serialized as three 381-bit Jacobian coordinates.  Count
    // the actual verifier-facing message instead of the historical constant
    // 36-byte approximation.
    proof_size_count += static_cast<uint>(opening.rounds.size() * 3U * blstrs__fp__Fp_LIMBS * sizeof(uint));
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

inline bool Commitment::verify_opening_gpu(const zkdl::EvaluationOpeningProof& opening,
                                           const G1Jacobian_t& expected_commitment_eval,
                                           std::string* error) const
{
    if (!zkdl::opening_shape_valid(opening, error)) return false;

    const uint fold_count = static_cast<uint>(opening.inner_point.size());
    std::vector<Fr_t> coefficients(3U * fold_count);
    const zkdl::FrWire one = zkdl::fr_from_u32(1U);
    for (uint i = 0; i < fold_count; ++i) {
        const zkdl::FrWire u = zkdl::fr_wire(opening.inner_point[i]);
        const zkdl::FrWire one_minus = zkdl::fr_sub(one, u);
        const zkdl::FrWire c_a = zkdl::fr_mul(one_minus, one_minus);
        const zkdl::FrWire c_p = zkdl::fr_mul(u, one_minus);
        const zkdl::FrWire c_b = zkdl::fr_mul(u, u);
        coefficients[3U * i] = zkdl::fr_value(c_a);
        coefficients[3U * i + 1U] = zkdl::fr_value(c_p);
        coefficients[3U * i + 2U] = zkdl::fr_value(c_b);
    }

    G1Jacobian_t* d_folds = nullptr;
    Fr_t* d_coefficients = nullptr;
    int* d_accepted = nullptr;
    int accepted = 0;
    const size_t fold_bytes = opening.rounds.size() * sizeof(G1Jacobian_t);
    const size_t coefficient_bytes = coefficients.size() * sizeof(Fr_t);
    cudaError_t status = cudaSuccess;
    if ((status = cudaMalloc(reinterpret_cast<void**>(&d_folds), fold_bytes)) != cudaSuccess ||
        (coefficient_bytes != 0 && (status = cudaMalloc(reinterpret_cast<void**>(&d_coefficients), coefficient_bytes)) != cudaSuccess) ||
        (status = cudaMalloc(reinterpret_cast<void**>(&d_accepted), sizeof(int))) != cudaSuccess) {
        if (d_folds) cudaFree(d_folds); if (d_coefficients) cudaFree(d_coefficients); if (d_accepted) cudaFree(d_accepted);
        if (error) *error = cudaGetErrorString(status);
        return false;
    }
    cudaMemcpy(d_folds, opening.rounds.data(), fold_bytes, cudaMemcpyHostToDevice);
    if (coefficient_bytes != 0) cudaMemcpy(d_coefficients, coefficients.data(), coefficient_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_accepted, 0, sizeof(int));
    zkdl_opening_verify_kernel<<<1, 1>>>(d_folds, d_coefficients,
                                         opening.commitment_eval,
                                         expected_commitment_eval,
                                         opening.rounds.back(), opening.value,
                                         fold_count, d_accepted);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaMemcpy(&accepted, d_accepted, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_folds); cudaFree(d_coefficients); cudaFree(d_accepted);
    if (status != cudaSuccess) {
        if (error) *error = cudaGetErrorString(status);
        return false;
    }
    if (accepted != 1) {
        if (error) *error = "Hyrax opening equation failed";
        return false;
    }
    return true;
}

#endif
