#ifndef ZKDL_ZKMATMUL_PROOF_CUH
#define ZKDL_ZKMATMUL_PROOF_CUH

/*
 * A small verifier-facing adapter for the restored upstream zkMatMul
 * primitive.  The arithmetic producer below deliberately calls the native
 * CUDA kernels from zkmatmul.cuh; this file only keeps the transcript and
 * opening messages that the old demo discarded.
 */

#include "cpu-backend.cuh"

// This header is also consumed directly by the adapter runner, before the
// legacy zkmatmul.cuh implementation is deliberately suppressed in that
// translation unit.  Keep only the original Commitment type and phase-1
// kernel declarations available in that case; the definitions remain owned
// by the native core's zkmatmul.cuh include.  When zkmatmul.cuh is included
// first its normal guards make this fallback inert.
#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH
#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

class Commitment: public G1TensorJacobian {
public:
    using G1TensorJacobian::G1TensorJacobian;
    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;
    using G1TensorJacobian::operator*=;

    G1TensorJacobian commit(const FrTensor& t) const;
    static Commitment random_generators(uint size);
    Fr_t open(const FrTensor& t, const G1TensorJacobian& c,
              const vector<Fr_t>& u, uint& proof_size_count) const;
    zkdl::EvaluationOpeningProof open_with_proof(
        const FrTensor& t, const G1TensorJacobian& c,
        const vector<Fr_t>& u) const;
    Fr_t open(const FrTensor& t, const G1TensorJacobian& c,
              const vector<Fr_t>& u, uint& proof_size_count,
              zkdl::EvaluationOpeningProof* retained) const;
    bool verify_opening_gpu(const zkdl::EvaluationOpeningProof& opening,
                            const G1Jacobian_t& expected_commitment_eval,
                            std::string* error = nullptr) const;
    static Fr_t me_open(const FrTensor& t, const Commitment& generators,
                        vector<Fr_t>::const_iterator begin,
                        vector<Fr_t>::const_iterator end,
                        vector<G1Jacobian_t>& proof);
};
#endif

#ifndef ZK_MATMUL_H
KERNEL void zkMatMul_phase1_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b,
                                 GLOBAL Fr_t *out0, GLOBAL Fr_t *out1,
                                 GLOBAL Fr_t *out2, uint in_num,
                                 uint out_num, uint n);
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zkdl {

struct MatMulShape {
    std::uint32_t num = 0;
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;

    bool operator==(const MatMulShape& other) const {
        return num == other.num && m == other.m && n == other.n && k == other.k;
    }
    bool operator!=(const MatMulShape& other) const { return !(*this == other); }

    static bool power_of_two(std::uint32_t value) {
        return value != 0U && (value & (value - 1U)) == 0U;
    }

    static std::uint32_t log2_ceil(std::uint32_t value) {
        if (value == 0U) return 0U;
        std::uint32_t result = 0U;
        std::uint32_t domain = 1U;
        while (domain < value) {
            if (domain > (std::numeric_limits<std::uint32_t>::max() >> 1U))
                throw std::invalid_argument("zkMatMul dimension is too large");
            domain <<= 1U;
            ++result;
        }
        return result;
    }

    std::uint64_t a_size() const {
        return static_cast<std::uint64_t>(num) * m * n;
    }
    std::uint64_t b_size() const {
        return static_cast<std::uint64_t>(num) * n * k;
    }
    std::uint64_t c_size() const {
        return static_cast<std::uint64_t>(num) * m * k;
    }

    bool valid(std::string* error = nullptr) const {
        if (num == 0U || m == 0U || n == 0U || k == 0U) {
            if (error) *error = "zkMatMul shape contains zero dimension";
            return false;
        }
        if (!power_of_two(n) || !power_of_two(k)) {
            if (error) *error = "zkMatMul opening dimensions n and k must be powers of two";
            return false;
        }
        if (a_size() > std::numeric_limits<std::uint32_t>::max() ||
            b_size() > std::numeric_limits<std::uint32_t>::max() ||
            c_size() > std::numeric_limits<std::uint32_t>::max()) {
            if (error) *error = "zkMatMul tensor shape exceeds native uint size";
            return false;
        }
        try {
            if (log2_ceil(num) >= 31U || log2_ceil(m) >= 31U ||
                log2_ceil(n) >= 31U || log2_ceil(k) >= 31U) {
                if (error) *error = "zkMatMul MLE dimension is too large";
                return false;
            }
        } catch (const std::exception& exception) {
            if (error) *error = exception.what();
            return false;
        }
        return true;
    }
};

struct MatMulCpuCommitmentView {
    std::vector<CpuG1Point> generators;
    std::vector<CpuG1Point> commitments;
};

struct MatMulNativeCommitmentView {
    std::vector<G1Jacobian_t> generators;
    std::vector<G1Jacobian_t> commitments;
};

inline bool matmul_fr_equal(const Fr_t& left, const Fr_t& right) {
    return fr_compare(fr_wire(left), fr_wire(right)) == 0;
}

inline void matmul_append_shape(std::vector<byte>& out, const MatMulShape& shape) {
    append_u32(out, shape.num);
    append_u32(out, shape.m);
    append_u32(out, shape.n);
    append_u32(out, shape.k);
}

inline bool matmul_read_shape(const std::vector<byte>& in, size_t& offset,
                              MatMulShape& shape) {
    return read_u32(in, offset, shape.num) && read_u32(in, offset, shape.m) &&
           read_u32(in, offset, shape.n) && read_u32(in, offset, shape.k);
}

inline void matmul_append_points(std::vector<byte>& out,
                                 const std::vector<G1Jacobian_t>& points) {
    append_u64(out, static_cast<std::uint64_t>(points.size()));
    for (size_t i = 0; i < points.size(); ++i) append_g1(out, points[i]);
}

inline bool matmul_read_points(const std::vector<byte>& in, size_t& offset,
                               std::vector<G1Jacobian_t>& points,
                               std::uint64_t max_points = (1U << 20)) {
    std::uint64_t count = 0;
    if (!read_u64(in, offset, count) || count > max_points) return false;
    points.resize(static_cast<size_t>(count));
    for (size_t i = 0; i < points.size(); ++i)
        if (!read_g1(in, offset, points[i])) return false;
    return true;
}

inline bool matmul_same_points(const std::vector<G1Jacobian_t>& left,
                               const std::vector<G1Jacobian_t>& right) {
    if (left.size() != right.size()) return false;
    std::vector<byte> left_bytes, right_bytes;
    matmul_append_points(left_bytes, left);
    matmul_append_points(right_bytes, right);
    return left_bytes == right_bytes;
}

inline void matmul_absorb_points(Transcript& transcript, const std::string& label,
                                 const std::vector<G1Jacobian_t>& points) {
    std::vector<byte> message;
    matmul_append_points(message, points);
    transcript.absorb(label, message);
}

inline void matmul_absorb_statement(
    Transcript& transcript, const MatMulShape& shape, std::uint32_t c_mode,
    const MatMulNativeCommitmentView& a, const MatMulNativeCommitmentView& b,
    const MatMulNativeCommitmentView* c) {
    std::vector<byte> message;
    matmul_append_shape(message, shape);
    append_u32(message, c_mode);
    matmul_append_points(message, a.generators);
    matmul_append_points(message, a.commitments);
    matmul_append_points(message, b.generators);
    matmul_append_points(message, b.commitments);
    if (c != nullptr) {
        matmul_append_points(message, c->generators);
        matmul_append_points(message, c->commitments);
    }
    transcript.absorb("zkMatMul.statement", message);
}

inline void matmul_append_fr_vector(std::vector<byte>& out,
                                    const std::vector<Fr_t>& values) {
    append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (size_t i = 0; i < values.size(); ++i) append_fr(out, values[i]);
}

inline bool matmul_read_fr_vector(const std::vector<byte>& in, size_t& offset,
                                  std::vector<Fr_t>& values,
                                  std::uint64_t max_values = (1U << 20)) {
    std::uint64_t count = 0;
    if (!read_u64(in, offset, count) || count > max_values) return false;
    values.resize(static_cast<size_t>(count));
    for (size_t i = 0; i < values.size(); ++i)
        if (!read_fr(in, offset, values[i])) return false;
    return true;
}

inline bool matmul_same_vector(const std::vector<Fr_t>& left,
                               const std::vector<Fr_t>& right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i)
        if (!matmul_fr_equal(left[i], right[i])) return false;
    return true;
}

inline Fr_t matmul_interpolate(const std::vector<Fr_t>& values, const Fr_t& point) {
    return fr_value(interpolate_at(values, fr_wire(point)));
}

inline std::vector<Fr_t> matmul_concat(const std::vector<Fr_t>& first,
                                       const std::vector<Fr_t>& second,
                                       const std::vector<Fr_t>& third) {
    std::vector<Fr_t> result;
    result.reserve(first.size() + second.size() + third.size());
    result.insert(result.end(), first.begin(), first.end());
    result.insert(result.end(), second.begin(), second.end());
    result.insert(result.end(), third.begin(), third.end());
    return result;
}

struct MatMulProof {
    static const std::uint32_t kVersion = 1U;
    static const std::uint32_t kCCommitment = 1U;
    static const std::uint32_t kCPublicClaim = 2U;

    std::uint32_t version = kVersion;
    std::string protocol_id = "zkdl-zkmatmul-proof-v1";
    MatMulShape shape;
    std::uint32_t c_binding_mode = kCCommitment;
    std::array<byte, 32> statement_digest{};
    std::array<byte, 32> transcript_digest{};
    Fr_t output_claim{};
    std::vector<Fr_t> target_num;
    std::vector<Fr_t> target_m;
    std::vector<Fr_t> target_k;
    std::vector<Fr_t> phase1_challenges;
    std::vector<Fr_t> phase2_challenges;
    // These are verifier-facing commitment messages, not witness tensors.
    // The verifier compares them with the independently supplied commitment
    // objects before replaying the transcript.
    std::vector<G1Jacobian_t> commitment_a;
    std::vector<G1Jacobian_t> commitment_b;
    std::vector<G1Jacobian_t> commitment_c;
    SumcheckProof phase1;
    SumcheckProof phase2;
    EvaluationOpeningProof opening_a;
    EvaluationOpeningProof opening_b;
    EvaluationOpeningProof opening_c;
    bool has_c_opening = false;

    std::vector<byte> serialize() const {
        std::vector<byte> out;
        out.reserve(1024);
        out.push_back('Z'); out.push_back('M'); out.push_back('M'); out.push_back('1');
        append_u32(out, version);
        append_string(out, protocol_id);
        matmul_append_shape(out, shape);
        append_u32(out, c_binding_mode);
        out.insert(out.end(), statement_digest.begin(), statement_digest.end());
        out.insert(out.end(), transcript_digest.begin(), transcript_digest.end());
        append_fr(out, output_claim);
        matmul_append_fr_vector(out, target_num);
        matmul_append_fr_vector(out, target_m);
        matmul_append_fr_vector(out, target_k);
        matmul_append_fr_vector(out, phase1_challenges);
        matmul_append_fr_vector(out, phase2_challenges);
        matmul_append_points(out, commitment_a);
        matmul_append_points(out, commitment_b);
        matmul_append_points(out, commitment_c);

        std::vector<byte> encoded;
        serialize_sumcheck(phase1, encoded);
        append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
        encoded.clear();
        serialize_sumcheck(phase2, encoded);
        append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());

        serialize_opening(opening_a, out);
        serialize_opening(opening_b, out);
        out.push_back(has_c_opening ? 1U : 0U);
        if (has_c_opening) serialize_opening(opening_c, out);
        return out;
    }

    static bool deserialize(const std::vector<byte>& in, MatMulProof& proof,
                            std::string* error = nullptr) {
        const std::uint64_t max_bytes = (1U << 28);
        if (in.size() > max_bytes) {
            if (error) *error = "zkMatMul proof is too large";
            return false;
        }
        size_t offset = 0;
        MatMulProof parsed;
        std::uint32_t parsed_version = 0;
        std::uint64_t block_size = 0;
        if (in.size() < 4 || in[0] != 'Z' || in[1] != 'M' || in[2] != 'M' || in[3] != '1') {
            if (error) *error = "invalid zkMatMul proof magic";
            return false;
        }
        offset = 4;
        if (!read_u32(in, offset, parsed_version) || parsed_version != kVersion ||
            !read_string(in, offset, parsed.protocol_id) ||
            !matmul_read_shape(in, offset, parsed.shape) ||
            !read_u32(in, offset, parsed.c_binding_mode)) {
            if (error) *error = "invalid zkMatMul proof header";
            return false;
        }
        parsed.version = parsed_version;
        if (in.size() - offset < parsed.statement_digest.size() + parsed.transcript_digest.size())
            return false;
        std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.statement_digest.size()),
                  parsed.statement_digest.begin());
        offset += parsed.statement_digest.size();
        std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.transcript_digest.size()),
                  parsed.transcript_digest.begin());
        offset += parsed.transcript_digest.size();
        if (!read_fr(in, offset, parsed.output_claim) ||
            !matmul_read_fr_vector(in, offset, parsed.target_num) ||
            !matmul_read_fr_vector(in, offset, parsed.target_m) ||
            !matmul_read_fr_vector(in, offset, parsed.target_k) ||
            !matmul_read_fr_vector(in, offset, parsed.phase1_challenges) ||
            !matmul_read_fr_vector(in, offset, parsed.phase2_challenges)) {
            if (error) *error = "invalid zkMatMul proof field";
            return false;
        }
        if (!matmul_read_points(in, offset, parsed.commitment_a) ||
            !matmul_read_points(in, offset, parsed.commitment_b) ||
            !matmul_read_points(in, offset, parsed.commitment_c)) {
            if (error) *error = "invalid zkMatMul commitment field";
            return false;
        }
        if (!read_u64(in, offset, block_size) || block_size > in.size() - offset) return false;
        std::vector<byte> block(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                in.begin() + static_cast<std::ptrdiff_t>(offset + block_size));
        offset += static_cast<size_t>(block_size);
        size_t block_offset = 0;
        if (!deserialize_sumcheck(block, block_offset, parsed.phase1) ||
            block_offset != block.size()) return false;
        if (!read_u64(in, offset, block_size) || block_size > in.size() - offset) return false;
        block.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                     in.begin() + static_cast<std::ptrdiff_t>(offset + block_size));
        offset += static_cast<size_t>(block_size);
        block_offset = 0;
        if (!deserialize_sumcheck(block, block_offset, parsed.phase2) ||
            block_offset != block.size()) return false;
        if (!deserialize_opening(in, offset, parsed.opening_a) ||
            !deserialize_opening(in, offset, parsed.opening_b) || offset >= in.size()) return false;
        parsed.has_c_opening = in[offset++] != 0U;
        if (parsed.has_c_opening && !deserialize_opening(in, offset, parsed.opening_c)) return false;
        if (offset != in.size()) {
            if (error) *error = "trailing zkMatMul proof bytes";
            return false;
        }
        proof = parsed;
        return true;
    }
};

inline bool matmul_check_opening_metadata(const EvaluationOpeningProof& opening,
                                          std::uint32_t commitment_index,
                                          std::uint32_t generator_size,
                                          const std::vector<Fr_t>& expected_point,
                                          std::string* error) {
    if (opening.commitment_index != commitment_index ||
        opening.generator_size != generator_size ||
        !opening_shape_valid(opening, error)) {
        if (error && error->empty()) *error = "zkMatMul opening metadata mismatch";
        return false;
    }
    std::vector<Fr_t> actual = matmul_concat(opening.inner_point, opening.outer_point, {});
    if (!matmul_same_vector(actual, expected_point)) {
        if (error) *error = "zkMatMul opening point mismatch";
        return false;
    }
    return true;
}

inline bool matmul_read_expected_point(const EvaluationOpeningProof& opening,
                                       const std::vector<Fr_t>& expected,
                                       std::string* error) {
    std::vector<Fr_t> actual = matmul_concat(opening.inner_point, opening.outer_point, {});
    if (!matmul_same_vector(actual, expected)) {
        if (error) *error = "zkMatMul opening point mismatch";
        return false;
    }
    return true;
}

inline bool matmul_validate_proof_shape(const MatMulProof& proof,
                                        const MatMulShape& shape,
                                        std::string* error = nullptr) {
    if (!shape.valid(error) || proof.version != MatMulProof::kVersion ||
        proof.protocol_id != "zkdl-zkmatmul-proof-v1" || proof.shape != shape ||
        (proof.c_binding_mode != MatMulProof::kCCommitment &&
         proof.c_binding_mode != MatMulProof::kCPublicClaim)) {
        if (error && error->empty()) *error = "zkMatMul proof metadata mismatch";
        return false;
    }
    if (proof.target_num.size() != MatMulShape::log2_ceil(shape.num) ||
        proof.target_m.size() != MatMulShape::log2_ceil(shape.m) ||
        proof.target_k.size() != MatMulShape::log2_ceil(shape.k) ||
        proof.phase1_challenges.size() != proof.target_num.size() ||
        proof.phase2_challenges.size() != MatMulShape::log2_ceil(shape.n)) {
        if (error) *error = "zkMatMul challenge vector shape mismatch";
        return false;
    }
    if (proof.phase1.label != "zkMatMul.phase1" || proof.phase1.degree != 2U ||
        proof.phase1.rounds.size() != proof.target_num.size() ||
        proof.phase1.terminal_claims.size() != 1U ||
        proof.phase2.label != "zkMatMul.phase2" || proof.phase2.degree != 2U ||
        proof.phase2.rounds.size() != proof.phase2_challenges.size() ||
        proof.phase2.terminal_claims.size() != 2U) {
        if (error) *error = "zkMatMul Sumcheck shape mismatch";
        return false;
    }
    if (proof.c_binding_mode == MatMulProof::kCCommitment && !proof.has_c_opening) {
        if (error) *error = "zkMatMul commitment mode is missing C opening";
        return false;
    }
    if (proof.c_binding_mode == MatMulProof::kCPublicClaim && proof.has_c_opening) {
        if (error) *error = "zkMatMul public-claim mode carries an unexpected C opening";
        return false;
    }
    return true;
}

inline bool matmul_replay_core(
    const MatMulProof& proof, const MatMulShape& shape,
    const MatMulNativeCommitmentView& a, const MatMulNativeCommitmentView& b,
    const MatMulNativeCommitmentView* c, const Fr_t* expected_public_claim,
    std::string* error = nullptr) {
    if (!matmul_validate_proof_shape(proof, shape, error)) return false;
    if (proof.c_binding_mode == MatMulProof::kCCommitment && c == nullptr) {
        if (error) *error = "zkMatMul verifier requires C commitment";
        return false;
    }
    if (proof.c_binding_mode == MatMulProof::kCPublicClaim && expected_public_claim == nullptr) {
        if (error) *error = "zkMatMul verifier requires public C claim";
        return false;
    }
    if (!matmul_same_points(proof.commitment_a, a.commitments) ||
        !matmul_same_points(proof.commitment_b, b.commitments) ||
        (proof.c_binding_mode == MatMulProof::kCCommitment &&
         (c == nullptr || !matmul_same_points(proof.commitment_c, c->commitments))) ||
        (proof.c_binding_mode == MatMulProof::kCPublicClaim && !proof.commitment_c.empty())) {
        if (error) *error = "zkMatMul commitment message mismatch";
        return false;
    }

    Transcript transcript("zkdl-zkmatmul-fs-v1");
    matmul_absorb_statement(transcript, shape, proof.c_binding_mode, a, b, c);
    if (transcript.digest() != proof.statement_digest) {
        if (error) *error = "zkMatMul statement digest mismatch";
        return false;
    }

    std::vector<Fr_t> target_num, target_m, target_k;
    target_num.reserve(proof.target_num.size());
    target_m.reserve(proof.target_m.size());
    target_k.reserve(proof.target_k.size());
    for (size_t i = 0; i < proof.target_num.size(); ++i)
        target_num.push_back(transcript.challenge("zkMatMul.target.num." + std::to_string(i)));
    for (size_t i = 0; i < proof.target_m.size(); ++i)
        target_m.push_back(transcript.challenge("zkMatMul.target.m." + std::to_string(i)));
    for (size_t i = 0; i < proof.target_k.size(); ++i)
        target_k.push_back(transcript.challenge("zkMatMul.target.k." + std::to_string(i)));
    if (!matmul_same_vector(target_num, proof.target_num) ||
        !matmul_same_vector(target_m, proof.target_m) ||
        !matmul_same_vector(target_k, proof.target_k)) {
        if (error) *error = "zkMatMul FS target mismatch";
        return false;
    }
    if (expected_public_claim != nullptr &&
        !matmul_fr_equal(*expected_public_claim, proof.output_claim)) {
        if (error) *error = "zkMatMul public C claim mismatch";
        return false;
    }
    transcript.absorb_fr("zkMatMul.output.claim", {proof.output_claim});

    Fr_t claim = proof.output_claim;
    for (size_t round = 0; round < proof.phase1.rounds.size(); ++round) {
        const std::vector<Fr_t>& evaluations = proof.phase1.rounds[round].evaluations;
        if (evaluations.size() != 3U ||
            !matmul_fr_equal(matmul_interpolate(evaluations, target_num[round]), claim)) {
            if (error) *error = "zkMatMul phase1 target equation failed";
            return false;
        }
        transcript.absorb_fr("zkMatMul.phase1.round", evaluations);
        const Fr_t challenge = transcript.challenge("zkMatMul.phase1.challenge");
        if (!matmul_fr_equal(challenge, proof.phase1_challenges[round])) {
            if (error) *error = "zkMatMul phase1 FS challenge mismatch";
            return false;
        }
        claim = matmul_interpolate(evaluations, challenge);
    }
    if (!matmul_fr_equal(claim, proof.phase1.final_claim) ||
        !matmul_fr_equal(proof.phase1.terminal_claims[0], proof.phase1.final_claim)) {
        if (error) *error = "zkMatMul phase1 terminal claim mismatch";
        return false;
    }

    std::vector<Fr_t> replayed_phase2;
    if (!verify_sumcheck(proof.phase2, proof.phase1.final_claim, transcript,
                         &replayed_phase2, error)) return false;
    if (!matmul_same_vector(replayed_phase2, proof.phase2_challenges)) {
        if (error) *error = "zkMatMul phase2 FS challenge mismatch";
        return false;
    }
    const Fr_t terminal_product = fr_value(fr_mul(fr_wire(proof.phase2.terminal_claims[0]),
                                                   fr_wire(proof.phase2.terminal_claims[1])));
    if (!matmul_fr_equal(terminal_product, proof.phase2.final_claim)) {
        if (error) *error = "zkMatMul phase2 output claim mismatch";
        return false;
    }

    const std::vector<Fr_t> expected_a =
        matmul_concat(proof.phase2_challenges, target_m, proof.phase1_challenges);
    const std::vector<Fr_t> expected_b =
        matmul_concat(target_k, proof.phase2_challenges, proof.phase1_challenges);
    const std::vector<Fr_t> expected_c =
        matmul_concat(target_k, target_m, target_num);
    if (!matmul_check_opening_metadata(proof.opening_a, 0U, shape.n, expected_a, error) ||
        !matmul_check_opening_metadata(proof.opening_b, 1U, shape.k, expected_b, error) ||
        !matmul_fr_equal(proof.opening_a.value, proof.phase2.terminal_claims[0]) ||
        !matmul_fr_equal(proof.opening_b.value, proof.phase2.terminal_claims[1]) ||
        (proof.has_c_opening &&
         !matmul_fr_equal(proof.opening_c.value, proof.output_claim))) {
        if (error && error->empty()) *error = "zkMatMul terminal opening claim mismatch";
        return false;
    }
    if (proof.has_c_opening &&
        !matmul_check_opening_metadata(proof.opening_c, 2U, shape.k, expected_c, error)) return false;

    std::vector<byte> opening_bytes;
    serialize_opening(proof.opening_a, opening_bytes);
    transcript.absorb("zkMatMul.opening.A", opening_bytes);
    opening_bytes.clear();
    serialize_opening(proof.opening_b, opening_bytes);
    transcript.absorb("zkMatMul.opening.B", opening_bytes);
    if (proof.has_c_opening) {
        opening_bytes.clear();
        serialize_opening(proof.opening_c, opening_bytes);
        transcript.absorb("zkMatMul.opening.C", opening_bytes);
    }
    if (transcript.digest() != proof.transcript_digest) {
        if (error) *error = "zkMatMul transcript digest mismatch";
        return false;
    }
    return true;
}

inline CpuG1Point matmul_cpu_mle_eval(std::vector<CpuG1Point> points,
                                      const std::vector<Fr_t>& point) {
    if (points.empty() || points.size() < (size_t(1) << point.size()) / 2U ||
        points.size() > (size_t(1) << point.size()))
        throw std::invalid_argument("CPU commitment MLE shape mismatch");
    for (size_t round = 0; round < point.size(); ++round) {
        const size_t next_size = (points.size() + 1U) / 2U;
        std::vector<CpuG1Point> next(next_size);
        const FrWire u = fr_wire(point[round]);
        const FrWire one_minus = fr_sub(fr_from_u32(1U), u);
        for (size_t i = 0; i < next_size; ++i) {
            const size_t i0 = 2U * i;
            const size_t i1 = i0 + 1U;
            if (i1 >= points.size()) {
                next[i] = cpu_g1_mul(points[i0], one_minus);
            } else {
                next[i] = cpu_g1_add(cpu_g1_mul(points[i0], one_minus),
                                     cpu_g1_mul(points[i1], u));
            }
        }
        points.swap(next);
    }
    return points[0];
}

inline MatMulNativeCommitmentView matmul_native_view(const Commitment& generators,
                                                     const G1TensorJacobian& commitment) {
    MatMulNativeCommitmentView view;
    view.generators = generators.to_host();
    view.commitments = commitment.to_host();
    return view;
}

inline MatMulNativeCommitmentView matmul_native_view(const MatMulCpuCommitmentView& view) {
    MatMulNativeCommitmentView native;
    native.generators.reserve(view.generators.size());
    native.commitments.reserve(view.commitments.size());
    for (size_t i = 0; i < view.generators.size(); ++i)
        native.generators.push_back(cpu_g1_to_native(view.generators[i]));
    for (size_t i = 0; i < view.commitments.size(); ++i)
        native.commitments.push_back(cpu_g1_to_native(view.commitments[i]));
    return native;
}

inline bool matmul_verify_gpu_opening(const Commitment& generators,
                                      const G1TensorJacobian& commitment,
                                      const EvaluationOpeningProof& opening,
                                      std::string* error) {
    try {
        const G1Jacobian_t expected = commitment(opening.outer_point);
        return generators.verify_opening_gpu(opening, expected, error);
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool matmul_verify_cpu_opening(const MatMulCpuCommitmentView& view,
                                      const EvaluationOpeningProof& opening,
                                      std::string* error) {
    try {
        const CpuG1Point expected = matmul_cpu_mle_eval(view.commitments, opening.outer_point);
        if (!cpu_g1_equal(cpu_g1_from_native(opening.commitment_eval), expected)) {
            if (error) *error = "CPU zkMatMul commitment evaluation mismatch";
            return false;
        }
        return cpu_verify_opening(opening, expected, error);
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool matmul_verify_cpu(const MatMulProof& proof, const MatMulShape& shape,
                              const MatMulCpuCommitmentView& a,
                              const MatMulCpuCommitmentView& b,
                              const MatMulCpuCommitmentView* c,
                              const Fr_t* expected_public_claim,
                              std::string* error = nullptr) {
    const MatMulNativeCommitmentView native_a = matmul_native_view(a);
    const MatMulNativeCommitmentView native_b = matmul_native_view(b);
    MatMulNativeCommitmentView native_c;
    const MatMulNativeCommitmentView* native_c_ptr = nullptr;
    if (c != nullptr) {
        native_c = matmul_native_view(*c);
        native_c_ptr = &native_c;
    }
    if (!matmul_replay_core(proof, shape, native_a, native_b, native_c_ptr,
                            expected_public_claim, error)) return false;
    if (!matmul_verify_cpu_opening(a, proof.opening_a, error) ||
        !matmul_verify_cpu_opening(b, proof.opening_b, error)) return false;
    if (proof.has_c_opening && (c == nullptr ||
                                !matmul_verify_cpu_opening(*c, proof.opening_c, error))) return false;
    return true;
}

inline bool matmul_verify_gpu(const MatMulProof& proof, const MatMulShape& shape,
                              const Commitment& gen_a, const G1TensorJacobian& com_a,
                              const Commitment& gen_b, const G1TensorJacobian& com_b,
                              const Commitment* gen_c, const G1TensorJacobian* com_c,
                              const Fr_t* expected_public_claim,
                              std::string* error = nullptr) {
    MatMulNativeCommitmentView native_a = matmul_native_view(gen_a, com_a);
    MatMulNativeCommitmentView native_b = matmul_native_view(gen_b, com_b);
    MatMulNativeCommitmentView native_c;
    const MatMulNativeCommitmentView* native_c_ptr = nullptr;
    if (gen_c != nullptr && com_c != nullptr) {
        native_c = matmul_native_view(*gen_c, *com_c);
        native_c_ptr = &native_c;
    }
    if (!matmul_replay_core(proof, shape, native_a, native_b, native_c_ptr,
                            expected_public_claim, error)) return false;
    if (!matmul_verify_gpu_opening(gen_a, com_a, proof.opening_a, error) ||
        !matmul_verify_gpu_opening(gen_b, com_b, proof.opening_b, error)) return false;
    if (proof.has_c_opening && (gen_c == nullptr || com_c == nullptr ||
                                !matmul_verify_gpu_opening(*gen_c, *com_c,
                                                            proof.opening_c, error))) return false;
    return true;
}

inline void matmul_cuda_sync(const char* operation) {
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(launch_status));
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(sync_status));
}

inline MatMulProof matmul_prove_internal(
    const FrTensor& a_tensor, const FrTensor& b_tensor, const FrTensor& c_tensor,
    const MatMulShape& shape, const Commitment& gen_a, const Commitment& gen_b,
    const Commitment* gen_c, std::uint32_t* proof_size_count = nullptr) {
    std::string error;
    if (!shape.valid(&error)) throw std::invalid_argument(error);
    if (a_tensor.size != shape.a_size() || b_tensor.size != shape.b_size() ||
        c_tensor.size != shape.c_size()) throw std::invalid_argument("zkMatMul tensor size mismatch");
    if (gen_a.size != shape.n || gen_b.size != shape.k ||
        (gen_c != nullptr && gen_c->size != shape.k))
        throw std::invalid_argument("zkMatMul generator shape mismatch");
    if (gen_c == nullptr && c_tensor.size == 0U)
        throw std::invalid_argument("zkMatMul requires output C witness");

    const G1TensorJacobian com_a = gen_a.commit(a_tensor);
    const G1TensorJacobian com_b = gen_b.commit(b_tensor);
    std::unique_ptr<G1TensorJacobian> com_c;
    if (gen_c != nullptr) com_c.reset(new G1TensorJacobian(gen_c->commit(c_tensor)));

    MatMulProof proof;
    proof.shape = shape;
    proof.c_binding_mode = gen_c == nullptr ? MatMulProof::kCPublicClaim : MatMulProof::kCCommitment;
    MatMulNativeCommitmentView native_a = matmul_native_view(gen_a, com_a);
    MatMulNativeCommitmentView native_b = matmul_native_view(gen_b, com_b);
    MatMulNativeCommitmentView native_c;
    const MatMulNativeCommitmentView* native_c_ptr = nullptr;
    if (gen_c != nullptr) {
        native_c = matmul_native_view(*gen_c, *com_c);
        native_c_ptr = &native_c;
    }
    proof.commitment_a = native_a.commitments;
    proof.commitment_b = native_b.commitments;
    if (native_c_ptr != nullptr) proof.commitment_c = native_c.commitments;

    Transcript transcript("zkdl-zkmatmul-fs-v1");
    matmul_absorb_statement(transcript, shape, proof.c_binding_mode,
                            native_a, native_b, native_c_ptr);
    proof.statement_digest = transcript.digest();
    for (std::uint32_t i = 0; i < MatMulShape::log2_ceil(shape.num); ++i)
        proof.target_num.push_back(transcript.challenge("zkMatMul.target.num." + std::to_string(i)));
    for (std::uint32_t i = 0; i < MatMulShape::log2_ceil(shape.m); ++i)
        proof.target_m.push_back(transcript.challenge("zkMatMul.target.m." + std::to_string(i)));
    for (std::uint32_t i = 0; i < MatMulShape::log2_ceil(shape.k); ++i)
        proof.target_k.push_back(transcript.challenge("zkMatMul.target.k." + std::to_string(i)));

    proof.output_claim = c_tensor(matmul_concat(proof.target_k, proof.target_m, proof.target_num));
    transcript.absorb_fr("zkMatMul.output.claim", {proof.output_claim});

    FrTensor a_reduced = a_tensor.partial_me(proof.target_m, shape.n);
    FrTensor b_reduced = b_tensor.partial_me(proof.target_k, 1U);
    std::unique_ptr<FrTensor> current_a(new FrTensor(a_reduced));
    std::unique_ptr<FrTensor> current_b(new FrTensor(b_reduced));
    std::uint32_t current_num = shape.num;
    proof.phase1.label = "zkMatMul.phase1";
    proof.phase1.degree = 2U;
    for (size_t round = 0; round < proof.target_num.size(); ++round) {
        const std::uint32_t out_num = (current_num + 1U) / 2U;
        const std::uint32_t out_size = out_num * shape.n;
        FrTensor out0(out_size), out1(out_size), out2(out_size);
        zkMatMul_phase1_step<<<(out_size + FrNumThread - 1U) / FrNumThread, FrNumThread>>>(
            current_a->device_data(), current_b->device_data(), out0.device_data(),
            out1.device_data(), out2.device_data(), current_num, out_num, shape.n);
        matmul_cuda_sync("zkMatMul phase1 polynomial");
        const std::vector<Fr_t> tail(proof.target_num.begin() + static_cast<std::ptrdiff_t>(round + 1U),
                                     proof.target_num.end());
        SumcheckRound message;
        message.evaluations.push_back(tail.empty() ? out0.sum() : out0.partial_me(tail, shape.n).sum());
        message.evaluations.push_back(tail.empty() ? out1.sum() : out1.partial_me(tail, shape.n).sum());
        message.evaluations.push_back(tail.empty() ? out2.sum() : out2.partial_me(tail, shape.n).sum());
        proof.phase1.rounds.push_back(message);
        transcript.absorb_fr("zkMatMul.phase1.round", message.evaluations);
        const Fr_t challenge = transcript.challenge("zkMatMul.phase1.challenge");
        proof.phase1_challenges.push_back(challenge);
        FrTensor next_a(out_size), next_b(out_size);
        Fr_partial_me_step<<<(out_size + FrNumThread - 1U) / FrNumThread, FrNumThread>>>(
            current_a->device_data(), next_a.device_data(), challenge,
            current_num * shape.n, out_size, shape.n);
        matmul_cuda_sync("zkMatMul phase1 A fold");
        Fr_partial_me_step<<<(out_size + FrNumThread - 1U) / FrNumThread, FrNumThread>>>(
            current_b->device_data(), next_b.device_data(), challenge,
            current_num * shape.n, out_size, shape.n);
        matmul_cuda_sync("zkMatMul phase1 B fold");
        current_a.reset(new FrTensor(next_a));
        current_b.reset(new FrTensor(next_b));
        current_num = out_num;
    }
    proof.phase1.final_claim = ((*current_a) * (*current_b)).sum();
    proof.phase1.terminal_claims.push_back(proof.phase1.final_claim);

    proof.phase2.label = "zkMatMul.phase2";
    proof.phase2.degree = 2U;
    for (std::uint32_t round = 0; round < MatMulShape::log2_ceil(shape.n); ++round) {
        const std::uint32_t in_size = current_a->size;
        const std::uint32_t out_size = (in_size + 1U) / 2U;
        FrTensor out0(out_size), out1(out_size), out2(out_size);
        Fr_ip_sc_step<<<(out_size + FrNumThread - 1U) / FrNumThread, FrNumThread>>>(
            current_a->device_data(), current_b->device_data(), out0.device_data(),
            out1.device_data(), out2.device_data(), in_size, out_size);
        matmul_cuda_sync("zkMatMul phase2 polynomial");
        SumcheckRound message;
        message.evaluations = {out0.sum(), out1.sum(), out2.sum()};
        proof.phase2.rounds.push_back(message);
        transcript.absorb_fr("zkMatMul.phase2.round", message.evaluations);
        const Fr_t challenge = transcript.challenge("zkMatMul.phase2.challenge");
        proof.phase2_challenges.push_back(challenge);
        FrTensor next_a(out_size), next_b(out_size);
        Fr_me_step<<<(out_size + FrNumThread - 1U) / FrNumThread, FrNumThread>>>(
            current_a->device_data(), next_a.device_data(), challenge, in_size, out_size);
        matmul_cuda_sync("zkMatMul phase2 A fold");
        Fr_me_step<<<(out_size + FrNumThread - 1U) / FrNumThread, FrNumThread>>>(
            current_b->device_data(), next_b.device_data(), challenge, in_size, out_size);
        matmul_cuda_sync("zkMatMul phase2 B fold");
        current_a.reset(new FrTensor(next_a));
        current_b.reset(new FrTensor(next_b));
    }
    const Fr_t terminal_a = (*current_a)(0U);
    const Fr_t terminal_b = (*current_b)(0U);
    proof.phase2.terminal_claims = {terminal_a, terminal_b};
    proof.phase2.final_claim = fr_value(fr_mul(fr_wire(terminal_a), fr_wire(terminal_b)));

    const std::vector<Fr_t> point_a =
        matmul_concat(proof.phase2_challenges, proof.target_m, proof.phase1_challenges);
    const std::vector<Fr_t> point_b =
        matmul_concat(proof.target_k, proof.phase2_challenges, proof.phase1_challenges);
    const std::vector<Fr_t> point_c =
        matmul_concat(proof.target_k, proof.target_m, proof.target_num);
    std::uint32_t ignored_size = 0U;
    gen_a.open(a_tensor, com_a, point_a, ignored_size, &proof.opening_a);
    gen_b.open(b_tensor, com_b, point_b, ignored_size, &proof.opening_b);
    proof.opening_a.commitment_index = 0U;
    proof.opening_b.commitment_index = 1U;
    if (!matmul_fr_equal(proof.opening_a.value, terminal_a) ||
        !matmul_fr_equal(proof.opening_b.value, terminal_b))
        throw std::runtime_error("zkMatMul terminal A/B opening mismatch");
    if (gen_c != nullptr) {
        gen_c->open(c_tensor, *com_c, point_c, ignored_size, &proof.opening_c);
        proof.opening_c.commitment_index = 2U;
        proof.has_c_opening = true;
        if (!matmul_fr_equal(proof.opening_c.value, proof.output_claim))
            throw std::runtime_error("zkMatMul terminal C opening mismatch");
    }

    std::vector<byte> opening_bytes;
    serialize_opening(proof.opening_a, opening_bytes);
    transcript.absorb("zkMatMul.opening.A", opening_bytes);
    opening_bytes.clear();
    serialize_opening(proof.opening_b, opening_bytes);
    transcript.absorb("zkMatMul.opening.B", opening_bytes);
    if (proof.has_c_opening) {
        opening_bytes.clear();
        serialize_opening(proof.opening_c, opening_bytes);
        transcript.absorb("zkMatMul.opening.C", opening_bytes);
    }
    proof.transcript_digest = transcript.digest();
    if (proof_size_count != nullptr) {
        const std::vector<byte> encoded = proof.serialize();
        if (encoded.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("zkMatMul proof size exceeds uint32");
        *proof_size_count += static_cast<std::uint32_t>(encoded.size());
    }
    return proof;
}

inline MatMulProof matmul_prove(const FrTensor& a, const FrTensor& b, const FrTensor& c,
                                const MatMulShape& shape, const Commitment& gen_a,
                                const Commitment& gen_b, const Commitment& gen_c,
                                std::uint32_t* proof_size_count = nullptr) {
    return matmul_prove_internal(a, b, c, shape, gen_a, gen_b, &gen_c, proof_size_count);
}

inline MatMulProof matmul_prove_public(const FrTensor& a, const FrTensor& b, const FrTensor& c,
                                       const MatMulShape& shape, const Commitment& gen_a,
                                       const Commitment& gen_b,
                                       std::uint32_t* proof_size_count = nullptr) {
    return matmul_prove_internal(a, b, c, shape, gen_a, gen_b, nullptr, proof_size_count);
}

}  // namespace zkdl

using zkMatMulProof = zkdl::MatMulProof;
using zkMatMulShape = zkdl::MatMulShape;

#endif
