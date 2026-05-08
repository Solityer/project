#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include <algorithm>
#include <cstring>
#include <sstream>

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
    Fr_t open_with_proof(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u, CommitmentOpeningProof& proof) const;
    static bool verify_opening(const Commitment& generators, const G1TensorJacobian& c, const CommitmentOpeningProof& proof);
    static vector<G1Jacobian_t> host_points(const G1TensorJacobian& tensor);
    static vector<Fr_t> host_scalars(const FrTensor& tensor);

    static Fr_t me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);
    static Fr_t me_open_fixed(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);
    static FrTensor partial_me_checked(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size, uint round);
};

KERNEL void fr_mont_one_minus_kernel(Fr_t u, GLOBAL Fr_t* out)
{
    *out = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE, u);
}

KERNEL void sum_axis_n_optimized(GLOBAL G1Jacobian_t* arr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m) {
    const uint gid = GET_GROUP_ID();
    if (gid >= m) return;
    
    __shared__ G1Jacobian_t shared_data[G1NumThread];
    uint local_id = threadIdx.x;
    
    G1Jacobian_t sum = (local_id < n) ? arr[gid * n + local_id] : blstrs__g1__G1Affine_ZERO;
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
    auto t_unmont = t;
    t_unmont.unmont();

    uint m = t.size / size;
    G1TensorJacobian temp = (*this) * t_unmont;
    G1TensorJacobian out(m);
    sum_axis_n_optimized<<<m,G1NumThread>>>(temp.gpu_data, out.gpu_data, size, m);
    cudaDeviceSynchronize();
    return out;
}

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint new_size) // always assume that old size is even, as will be implemented
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    Fr_t u_unmont = blstrs__scalar__Scalar_unmont(u);
    new_scalars[gid] = blstrs__scalar__Scalar_add(scalars[gid0], blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0])));
    new_generators[gid] = blstrs__g1__G1Affine_add(generators[gid1], G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u_unmont));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalars[gid0]), G1Jacobian_mul(generators[gid1], scalars[gid1]));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalars[gid0]);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalars[gid1]);
}

Fr_t Commitment::me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    return me_open_fixed(t, generators, begin, end, proof);
}

Fr_t me_open_fixed_host_recursive(const vector<Fr_t>& scalars, const Commitment& generators,
    vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    if (scalars.size() != generators.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end)
    {
        proof.push_back(generators(0));
        return scalars.empty() ? Fr_host_zero() : scalars.front();
    }
    if (scalars.size() % 2U != 0U) throw std::runtime_error("Inner opening requires power-of-two generator size");

    const uint new_size = static_cast<uint>(scalars.size() / 2U);
    vector<Fr_t> left_scalars(new_size, Fr_host_zero());
    vector<Fr_t> right_scalars(new_size, Fr_host_zero());
    vector<Fr_t> folded_scalars(new_size, Fr_host_zero());
    for (uint i = 0; i < new_size; ++i)
    {
        left_scalars[i] = scalars[2U * i];
        right_scalars[i] = scalars[2U * i + 1U];
        folded_scalars[i] = Fr_host_mle_step(left_scalars[i], right_scalars[i], *begin);
    }

    FrTensor scalar_tensor(static_cast<uint>(scalars.size()), scalars.data());
    FrTensor left_tensor(new_size, left_scalars.data());
    FrTensor right_tensor(new_size, right_scalars.data());

    const auto generator_points = Commitment::host_points(generators);
    vector<G1Jacobian_t> left_points(new_size), right_points(new_size);
    for (uint i = 0; i < new_size; ++i)
    {
        left_points[i] = generator_points[2U * i];
        right_points[i] = generator_points[2U * i + 1U];
    }

    Commitment left_generators(new_size, left_points.data());
    Commitment right_generators(new_size, right_points.data());
    const auto current_commitment = generators.commit(scalar_tensor);
    const auto cross0_commitment = right_generators.commit(left_tensor);
    const auto cross1_commitment = left_generators.commit(right_tensor);
    proof.push_back(current_commitment(0));
    proof.push_back(cross0_commitment(0));
    proof.push_back(cross1_commitment(0));

    const Fr_t u_raw = Fr_host_unmont(*begin);
    const Fr_t one_minus_u = Fr_host_sub(Fr_host_one_mont(), *begin);
    const Fr_t one_minus_u_raw = Fr_host_unmont(one_minus_u);
    vector<G1Jacobian_t> folded_points(new_size);
    for (uint i = 0; i < new_size; ++i)
    {
        folded_points[i] = G1Jacobian_host_add(
            G1Jacobian_host_mul(left_points[i], u_raw),
            G1Jacobian_host_mul(right_points[i], one_minus_u_raw));
    }
    Commitment new_generators(new_size, folded_points.data());
    return me_open_fixed_host_recursive(folded_scalars, new_generators, begin + 1, end, proof);
}

KERNEL void me_open_fixed_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u,
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1,
    uint new_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    Fr_t u_unmont = blstrs__scalar__Scalar_unmont(u);
    Fr_t scalar0_unmont = blstrs__scalar__Scalar_unmont(scalars[gid0]);
    Fr_t scalar1_unmont = blstrs__scalar__Scalar_unmont(scalars[gid1]);
    new_scalars[gid] = blstrs__scalar__Scalar_add(scalars[gid0], blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0])));
    new_generators[gid] = blstrs__g1__G1Affine_add(generators[gid1], G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u_unmont));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalar0_unmont), G1Jacobian_mul(generators[gid1], scalar1_unmont));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalar0_unmont);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalar1_unmont);
}

Fr_t Commitment::me_open_fixed(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    if (t.size != generators.size) throw std::runtime_error("Incompatible dimensions");
    return me_open_fixed_host_recursive(t.to_host(), generators, begin, end, proof);
}

Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u, uint& proof_size_count) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));

    (void)com;
    if (size != (1 << u_in.size())) throw std::runtime_error("Incompatible dimensions");
    vector<G1Jacobian_t> proof;
    auto val = me_open(t.partial_me(u_out, 1 << u_in.size()), *this, u_in.begin(), u_in.end(), proof);
    proof_size_count += proof.size() * 36;
    return val;
}

FrTensor Commitment::partial_me_checked(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size, uint round)
{
    if (begin >= end) return t;
    if (window_size == 0)
    {
        throw std::runtime_error("partial_me_checked invalid window_size=0");
    }
    const auto current = t.to_host();
    const uint out_size = Fr_partial_me_out_size(t.size, window_size);
    vector<Fr_t> folded(out_size, Fr_host_zero());
    for (uint gid = 0; gid < out_size; ++gid)
    {
        const uint window_id = gid / window_size;
        const uint idx_in_window = gid % window_size;
        const uint gid0 = 2U * window_id * window_size + idx_in_window;
        const uint gid1 = (2U * window_id + 1U) * window_size + idx_in_window;
        const Fr_t low = (gid0 < t.size) ? current[gid0] : Fr_host_zero();
        const Fr_t high = (gid1 < t.size) ? current[gid1] : Fr_host_zero();
        folded[gid] = Fr_host_mle_step(low, high, *begin);
    }
    FrTensor t_new(out_size, folded.data());
    return partial_me_checked(t_new, begin + 1, end, window_size, round + 1);
}

Fr_t Commitment::open_with_proof(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u, CommitmentOpeningProof& proof) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));

    if (size != (1 << u_in.size())) throw std::runtime_error("Incompatible dimensions");
    proof.point = u;
    proof.transcript.clear();
    auto reduced = partial_me_checked(t, u_out.begin(), u_out.end(), 1 << u_in.size(), 0);
    proof.value = me_open_fixed(reduced, *this, u_in.begin(), u_in.end(), proof.transcript);
    return proof.value;
}

DEVICE bool Fp_is_zero_zkdl(Fp_t x)
{
    return blstrs__fp__Fp_eq(x, blstrs__fp__Fp_ZERO);
}

DEVICE bool G1Jacobian_is_zero_zkdl(G1Jacobian_t p)
{
    return Fp_is_zero_zkdl(p.z);
}

DEVICE bool G1Jacobian_eq_zkdl(G1Jacobian_t a, G1Jacobian_t b)
{
    const bool a_zero = G1Jacobian_is_zero_zkdl(a);
    const bool b_zero = G1Jacobian_is_zero_zkdl(b);
    if (a_zero || b_zero) return a_zero && b_zero;

    const Fp_t z1z1 = blstrs__fp__Fp_sqr(a.z);
    const Fp_t z2z2 = blstrs__fp__Fp_sqr(b.z);
    const Fp_t lhs_x = blstrs__fp__Fp_mul(a.x, z2z2);
    const Fp_t rhs_x = blstrs__fp__Fp_mul(b.x, z1z1);
    if (!blstrs__fp__Fp_eq(lhs_x, rhs_x)) return false;

    const Fp_t z1z1z1 = blstrs__fp__Fp_mul(z1z1, a.z);
    const Fp_t z2z2z2 = blstrs__fp__Fp_mul(z2z2, b.z);
    const Fp_t lhs_y = blstrs__fp__Fp_mul(a.y, z2z2z2);
    const Fp_t rhs_y = blstrs__fp__Fp_mul(b.y, z1z1z1);
    return blstrs__fp__Fp_eq(lhs_y, rhs_y);
}

KERNEL void g1_equal_kernel(G1Jacobian_t a, G1Jacobian_t b, GLOBAL bool* out)
{
    *out = G1Jacobian_eq_zkdl(a, b);
}

KERNEL void g1_open_expected_next_kernel(G1Jacobian_t p0, G1Jacobian_t p1, G1Jacobian_t p2, Fr_t u, GLOBAL G1Jacobian_t* out)
{
    Fr_t one = blstrs__scalar__Scalar_ONE;
    Fr_t one_minus_u = blstrs__scalar__Scalar_sub(one, u);
    Fr_t c0 = blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_mul(one_minus_u, one_minus_u));
    Fr_t c1 = blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_mul(u, one_minus_u));
    Fr_t c2 = blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_mul(u, u));
    G1Jacobian_t acc = G1Jacobian_mul(p1, c0);
    acc = blstrs__g1__G1Affine_add(acc, G1Jacobian_mul(p0, c1));
    acc = blstrs__g1__G1Affine_add(acc, G1Jacobian_mul(p2, c2));
    *out = acc;
}

KERNEL void g1_final_commitment_kernel(G1Jacobian_t generator, Fr_t value, GLOBAL G1Jacobian_t* out)
{
    *out = G1Jacobian_mul(generator, blstrs__scalar__Scalar_unmont(value));
}

bool g1_equal_host(G1Jacobian_t a, G1Jacobian_t b)
{
    return G1Jacobian_host_eq(a, b);
}

G1Jacobian_t g1_open_expected_next_host(G1Jacobian_t p0, G1Jacobian_t p1, G1Jacobian_t p2, Fr_t u)
{
    const Fr_t one_minus_u = Fr_host_sub(Fr_host_one_mont(), u);
    const Fr_t c0 = Fr_host_unmont(Fr_host_mul(one_minus_u, one_minus_u));
    const Fr_t c1 = Fr_host_unmont(Fr_host_mul(u, one_minus_u));
    const Fr_t c2 = Fr_host_unmont(Fr_host_mul(u, u));
    G1Jacobian_t acc = G1Jacobian_host_mul(p1, c0);
    acc = G1Jacobian_host_add(acc, G1Jacobian_host_mul(p0, c1));
    acc = G1Jacobian_host_add(acc, G1Jacobian_host_mul(p2, c2));
    return acc;
}

G1Jacobian_t g1_final_commitment_host(G1Jacobian_t generator, Fr_t value)
{
    return G1Jacobian_host_mul(generator, Fr_host_unmont(value));
}

bool Commitment::verify_opening(const Commitment& generators, const G1TensorJacobian& com, const CommitmentOpeningProof& proof)
{
    const uint outer_log = ceilLog2(com.size);
    if (proof.point.size() < outer_log) return false;
    const uint inner_log = static_cast<uint>(proof.point.size()) - outer_log;
    if (generators.size != (1U << inner_log)) return false;
    if (proof.transcript.size() != static_cast<std::size_t>(3 * inner_log + 1)) return false;

    vector<Fr_t> u_out(proof.point.end() - outer_log, proof.point.end());
    G1Jacobian_t public_commitment = outer_log == 0
        ? com(0)
        : G1Jacobian_host_mle(host_points(com), u_out.begin(), u_out.end());
    if (inner_log == 0)
    {
        if (!g1_equal_host(generators(0), proof.transcript.back())) return false;
        const auto expected = g1_final_commitment_host(proof.transcript.back(), proof.value);
        return g1_equal_host(public_commitment, expected);
    }
    if (!g1_equal_host(public_commitment, proof.transcript.front())) return false;

    for (uint round = 0; round < inner_log; ++round)
    {
        const auto current = proof.transcript.at(3 * round);
        const auto cross0 = proof.transcript.at(3 * round + 1);
        const auto cross1 = proof.transcript.at(3 * round + 2);
        const auto expected_next = g1_open_expected_next_host(current, cross0, cross1, proof.point.at(round));
        const G1Jacobian_t actual_next = (round + 1 < inner_log)
            ? proof.transcript.at(3 * (round + 1))
            : g1_final_commitment_host(proof.transcript.back(), proof.value);
        if (!g1_equal_host(expected_next, actual_next)) return false;
    }

    return true;
}

vector<G1Jacobian_t> Commitment::host_points(const G1TensorJacobian& tensor)
{
    vector<G1Jacobian_t> out(tensor.size);
    if (!out.empty())
    {
        cudaMemcpy(out.data(), tensor.gpu_data, sizeof(G1Jacobian_t) * out.size(), cudaMemcpyDeviceToHost);
    }
    return out;
}

vector<Fr_t> Commitment::host_scalars(const FrTensor& tensor)
{
    return tensor.to_host();
}

Commitment Commitment::random_generators(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out; 
}

#endif
