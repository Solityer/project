#ifndef ZK_PROD_H
#define ZK_PROD_H
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "g1-tensor.cuh"
#include "proof.cuh"
#include "formal_relations.cuh"
#include "timer.hpp"
#include "commitment.cuh"

class zkProd {
public:
    FrTensor A, B;
    uint num, dim;
    G1TensorJacobian comA, comB;
    Timer &p_timer, &v_timer;
    zkProd(const FrTensor& A, const FrTensor& B, uint num, uint dim, Commitment& genA, Commitment& genB, Timer& p_timer, Timer& v_timer, uint& commit_size_count): 
        A(A), B(B), num(num), dim(dim), comA(genA.commit(A)), comB(genB.commit(B)), p_timer(p_timer), v_timer(v_timer) {
        if (A.size != num * dim || B.size != num * dim) throw std::runtime_error("size mismatch");
        commit_size_count += (comA.size + comB.size) * 36;
        // cout << "Commitment size: " << comA.size + comB.size << endl;
    }

    // static std::pair<FrTensor, FrTensor> reduce(const FrTensor& A, const FrTensor& B, uint num, uint m, uint n, uint k);
    // static std::pair<FrTensor, FrTensor> phase1(const FrTensor& A_reduced, const FrTensor& B_reduced, uint num, uint n, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);
    void prove(const Commitment& genA, const Commitment& genB, uint& proof_size_count);
};

inline HadamardSumcheckProof zkprod_formal_prove(
    const FrTensor& A,
    const FrTensor& B,
    const FrTensor& C,
    const Commitment& generators,
    const G1TensorJacobian& comA,
    const G1TensorJacobian& comB,
    const G1TensorJacobian& comC,
    const std::string& relation_label)
{
    return formal_hadamard_relation_prove(A, B, C, generators, comA, comB, comC, relation_label);
}

inline bool zkprod_formal_verify(
    const HadamardSumcheckProof& proof,
    const Commitment& generators,
    const G1TensorJacobian& comA,
    const G1TensorJacobian& comB,
    const G1TensorJacobian& comC,
    const std::string& relation_label,
    std::string* failure = nullptr)
{
    return formal_hadamard_relation_verify(proof, generators, comA, comB, comC, relation_label, failure);
}

void zkProd::prove(const Commitment& genA, const Commitment& genB, uint& proof_size_count)
{
    p_timer.start();
    auto u = random_vec(ceilLog2(num * dim)), v = random_vec(ceilLog2(num * dim));
    vector<Fr_t> proof = hadamard_product_sumcheck(A, B, u, v);
    // auto u_num = random_vec(ceilLog2(num));
    // auto v_num = random_vec(ceilLog2(num));
    // auto u_m = random_vec(ceilLog2(m));
    // auto u_n = random_vec(ceilLog2(n));
    // auto u_k = random_vec(ceilLog2(k));
    
    // auto A_reduced = Fr_partial_me(A, u_m.begin(), u_m.end(), n); // num * n
    // auto B_reduced = Fr_partial_me(B, u_k.begin(), u_k.end(), 1); // num * n
    
    // auto phase1_out = zkMatMul::phase1(A_reduced, B_reduced, num, n, u_num.begin(), u_num.end(), v_num.begin(), v_num.end(), proof);
    // auto& a = phase1_out.first;
    // auto& b = phase1_out.second;
    // auto phase_2_proof = inner_product_sumcheck(a, b, u_n);
    // proof.insert(proof.end(), phase_2_proof.begin(), phase_2_proof.end());
    // proof_size_count += proof.size() * 8;
    // // cout << "zkMatMul sumcheck proof size: " << proof.size() << endl;

    v_timer.start();
    genA.open(A, comA, v, proof_size_count);
    genB.open(B, comB, v, proof_size_count);
    v_timer.stop();
    p_timer.stop();
}

#endif
