#ifndef ZK_PROD_H
#define ZK_PROD_H

#include <cuda_runtime.h>

#include "bls12-381.cuh"
#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"
#include "timer.hpp"
#include "commitment.cuh"
#include "zkprod-proof.cuh"

/*
 * Native zkDL product adapter.  The proof object is produced by the small
 * host-side orchestration in zkprod-proof.cuh, but every commitment, terminal
 * opening and Sumcheck wire object is the official zkDL representation.  A
 * caller may continue to use the historical two-input constructor; that
 * overload derives C only for source compatibility.  New proof callers must
 * pass C explicitly so the public output is actually bound by the relation.
 */
class zkProd {
public:
    FrTensor A, B, C;
    uint num, dim;
    G1TensorJacobian comA, comB, comC;
    Timer &p_timer, &v_timer;
    zkdl_prod::Proof last_proof;

    zkProd(const FrTensor& A_, const FrTensor& B_, const FrTensor& C_,
           uint num_, uint dim_, Commitment& genA, Commitment& genB,
           Commitment& genC, Timer& p_timer_, Timer& v_timer_,
           uint& commit_size_count)
        : A(A_), B(B_), C(C_), num(num_), dim(dim_),
          comA(genA.commit(A_)), comB(genB.commit(B_)), comC(genC.commit(C_)),
          p_timer(p_timer_), v_timer(v_timer_), last_proof() {
        validate_shape();
        commit_size_count += (comA.size + comB.size + comC.size) * 36U;
    }

    // Compatibility overload for the original demo API.  It deliberately
    // uses an explicitly derived C and the supplied A generators for C; the
    // overload is not used by the GAT adapter, which always passes C/genC.
    zkProd(const FrTensor& A_, const FrTensor& B_, uint num_, uint dim_,
           Commitment& genA, Commitment& genB, Timer& p_timer_,
           Timer& v_timer_, uint& commit_size_count)
        : A(A_), B(B_), C(A_ * B_), num(num_), dim(dim_),
          comA(genA.commit(A_)), comB(genB.commit(B_)), comC(genA.commit(C)),
          p_timer(p_timer_), v_timer(v_timer_), last_proof() {
        validate_shape();
        commit_size_count += (comA.size + comB.size + comC.size) * 36U;
    }

    const zkdl_prod::Proof& prove(const Commitment& genA,
                                  const Commitment& genB,
                                  const Commitment& genC,
                                  uint& proof_size_count) {
        validate_shape();
        p_timer.start();
        const std::vector<Fr_t> host_a = A.to_host();
        const std::vector<Fr_t> host_b = B.to_host();
        const std::vector<Fr_t> host_c = C.to_host();
        const std::vector<G1Jacobian_t> actual_com_a = comA.to_host();
        const std::vector<G1Jacobian_t> actual_com_b = comB.to_host();
        const std::vector<G1Jacobian_t> actual_com_c = comC.to_host();
        last_proof = zkdl_prod::prove_host(
            num, dim, to_wires(host_a), to_wires(host_b), to_wires(host_c),
            to_cpu_generators(genA), to_cpu_generators(genB),
            to_cpu_generators(genC));
        if (!zkdl_prod::commitments_match(last_proof.commitment_a,
                                          actual_com_a, nullptr) ||
            !zkdl_prod::commitments_match(last_proof.commitment_b,
                                          actual_com_b, nullptr) ||
            !zkdl_prod::commitments_match(last_proof.commitment_c,
                                          actual_com_c, nullptr)) {
            p_timer.stop();
            throw std::runtime_error("CPU/native zkProd commitment mismatch");
        }
        const std::vector<zkdl::byte> encoded = last_proof.serialize();
        proof_size_count += static_cast<uint>(encoded.size());
        p_timer.stop();
        return last_proof;
    }

    // Preserve the historical call shape.  The product commitment is opened
    // with the A generator family created by the compatibility constructor.
    const zkdl_prod::Proof& prove(const Commitment& genA,
                                  const Commitment& genB,
                                  uint& proof_size_count) {
        return prove(genA, genB, genA, proof_size_count);
    }

    static bool verify_cpu(
        const zkdl_prod::Proof& proof, const Commitment& genA,
        const Commitment& genB, const Commitment& genC,
        const G1TensorJacobian& comA, const G1TensorJacobian& comB,
        const G1TensorJacobian& comC, std::string* error = nullptr) {
        return zkdl_prod::verify_host(
            proof, proof.num, proof.dim, to_cpu_generators(genA),
            to_cpu_generators(genB), to_cpu_generators(genC), comA.to_host(),
            comB.to_host(), comC.to_host(), error);
    }

    static bool verify_serialized(
        const std::vector<zkdl::byte>& encoded, const Commitment& genA,
        const Commitment& genB, const Commitment& genC,
        const G1TensorJacobian& comA, const G1TensorJacobian& comB,
        const G1TensorJacobian& comC, std::string* error = nullptr) {
        zkdl_prod::Proof proof;
        if (!zkdl_prod::Proof::deserialize(encoded, proof, error)) return false;
        return verify_cpu(proof, genA, genB, genC, comA, comB, comC, error);
    }

    // GPU dispatch keeps exactly the same wire object and relation checks as
    // verify_cpu, then asks the official CUDA Hyrax opening implementation to
    // replay all three terminal openings.  No witness is passed to either
    // verifier path.
    static bool verify_gpu(
        const zkdl_prod::Proof& proof, const Commitment& genA,
        const Commitment& genB, const Commitment& genC,
        const G1TensorJacobian& comA, const G1TensorJacobian& comB,
        const G1TensorJacobian& comC, std::string* error = nullptr) {
        if (!verify_cpu(proof, genA, genB, genC, comA, comB, comC, error))
            return false;
        if (!genA.verify_opening_gpu(proof.opening_a,
                                     comA(proof.opening_a.outer_point), error) ||
            !genB.verify_opening_gpu(proof.opening_b,
                                     comB(proof.opening_b.outer_point), error) ||
            !genC.verify_opening_gpu(proof.opening_c,
                                     comC(proof.opening_c.outer_point), error))
            return false;
        return true;
    }

private:
    static std::vector<zkdl::FrWire> to_wires(
        const std::vector<Fr_t>& values) {
        std::vector<zkdl::FrWire> result;
        result.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i)
            result.push_back(zkdl::fr_wire(values[i]));
        return result;
    }

    static std::vector<zkdl::CpuG1Point> to_cpu_generators(
        const Commitment& generators) {
        const std::vector<G1Jacobian_t> native = generators.to_host();
        std::vector<zkdl::CpuG1Point> result;
        result.reserve(native.size());
        for (size_t i = 0; i < native.size(); ++i)
            result.push_back(zkdl::cpu_g1_from_native(native[i]));
        return result;
    }

    void validate_shape() const {
        if (num == 0U || dim == 0U || A.size != num * dim ||
            B.size != num * dim || C.size != num * dim ||
            (num & (num - 1U)) != 0U || (dim & (dim - 1U)) != 0U)
            throw std::runtime_error("zkProd tensor shape must be powers of two");
        if (comA.size != num || comB.size != num || comC.size != num)
            throw std::runtime_error("zkProd commitment shape mismatch");
    }
};

#endif
