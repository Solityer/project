#ifndef ZKDL_ZKPROD_PROOF_CUH
#define ZKDL_ZKPROD_PROOF_CUH

/*
 * Verifier-facing wire object for the native zkProd relation.  The object is
 * intentionally a thin protocol envelope around the official FrTensor,
 * Sumcheck and Hyrax opening primitives; it does not contain A, B or C.
 *
 * The three commitment vectors are public statement material.  A proof binds
 * the public C commitment to the same pointwise product relation as A and B,
 * while the opening messages contain only terminal evaluations and the
 * native Hyrax folding points.
 */

#include "cpu-backend.cuh"

#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace zkdl_prod {

using byte = zkdl::byte;

struct Proof {
    static const std::uint32_t kVersion = 1U;

    std::uint32_t version;
    std::uint32_t num;
    std::uint32_t dim;
    std::uint32_t log_domain;
    std::vector<Fr_t> evaluation_point;
    std::vector<G1Jacobian_t> commitment_a;
    std::vector<G1Jacobian_t> commitment_b;
    std::vector<G1Jacobian_t> commitment_c;
    zkdl::SumcheckProof sumcheck;
    zkdl::EvaluationOpeningProof opening_a;
    zkdl::EvaluationOpeningProof opening_b;
    zkdl::EvaluationOpeningProof opening_c;
    std::array<byte, 32> transcript_digest;

    Proof()
        : version(kVersion), num(0U), dim(0U), log_domain(0U),
          evaluation_point(), commitment_a(), commitment_b(), commitment_c(),
          sumcheck(), opening_a(), opening_b(), opening_c(), transcript_digest{{}} {}

    static void append_points(std::vector<byte>& out,
                              const std::vector<G1Jacobian_t>& points) {
        zkdl::append_u64(out, static_cast<std::uint64_t>(points.size()));
        for (size_t i = 0; i < points.size(); ++i) zkdl::append_g1(out, points[i]);
    }

    static bool read_points(const std::vector<byte>& in, size_t& offset,
                            std::vector<G1Jacobian_t>& points,
                            size_t max_points = (1U << 20)) {
        std::uint64_t count = 0;
        if (!zkdl::read_u64(in, offset, count) || count > max_points) return false;
        points.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < points.size(); ++i)
            if (!zkdl::read_g1(in, offset, points[i])) return false;
        return true;
    }

    static void append_opening(std::vector<byte>& out,
                               const zkdl::EvaluationOpeningProof& opening) {
        std::vector<byte> encoded;
        zkdl::serialize_opening(opening, encoded);
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }

    static bool read_opening(const std::vector<byte>& in, size_t& offset,
                             zkdl::EvaluationOpeningProof& opening) {
        std::uint64_t bytes = 0;
        if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset)
            return false;
        std::vector<byte> encoded(
            in.begin() + static_cast<std::ptrdiff_t>(offset),
            in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<size_t>(bytes);
        size_t inner = 0;
        return zkdl::deserialize_opening(encoded, inner, opening) &&
               inner == encoded.size();
    }

    static void append_sumcheck(std::vector<byte>& out,
                                const zkdl::SumcheckProof& sumcheck) {
        std::vector<byte> encoded;
        zkdl::serialize_sumcheck(sumcheck, encoded);
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }

    static bool read_sumcheck(const std::vector<byte>& in, size_t& offset,
                              zkdl::SumcheckProof& sumcheck) {
        std::uint64_t bytes = 0;
        if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset)
            return false;
        std::vector<byte> encoded(
            in.begin() + static_cast<std::ptrdiff_t>(offset),
            in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<size_t>(bytes);
        size_t inner = 0;
        return zkdl::deserialize_sumcheck(encoded, inner, sumcheck) &&
               inner == encoded.size();
    }

    std::vector<byte> serialize() const {
        std::vector<byte> out;
        out.reserve(1024);
        out.push_back('Z'); out.push_back('P'); out.push_back('R'); out.push_back('D');
        zkdl::append_u32(out, version);
        zkdl::append_u32(out, num);
        zkdl::append_u32(out, dim);
        zkdl::append_u32(out, log_domain);
        zkdl::append_u64(out, static_cast<std::uint64_t>(evaluation_point.size()));
        for (size_t i = 0; i < evaluation_point.size(); ++i)
            zkdl::append_fr(out, evaluation_point[i]);
        append_points(out, commitment_a);
        append_points(out, commitment_b);
        append_points(out, commitment_c);
        append_sumcheck(out, sumcheck);
        append_opening(out, opening_a);
        append_opening(out, opening_b);
        append_opening(out, opening_c);
        out.insert(out.end(), transcript_digest.begin(), transcript_digest.end());
        return out;
    }

    static bool deserialize(const std::vector<byte>& in, Proof& proof,
                            std::string* error = nullptr) {
        const size_t max_items = (1U << 20);
        size_t offset = 0;
        Proof parsed;
        std::uint64_t count = 0;
        if (in.size() < 4U || in[0] != 'Z' || in[1] != 'P' ||
            in[2] != 'R' || in[3] != 'D') {
            if (error) *error = "invalid zkProd proof magic";
            return false;
        }
        offset = 4U;
        if (!zkdl::read_u32(in, offset, parsed.version) ||
            parsed.version != kVersion ||
            !zkdl::read_u32(in, offset, parsed.num) ||
            !zkdl::read_u32(in, offset, parsed.dim) ||
            !zkdl::read_u32(in, offset, parsed.log_domain)) {
            if (error) *error = "invalid zkProd proof header";
            return false;
        }
        if (!zkdl::read_u64(in, offset, count) || count > max_items) return false;
        parsed.evaluation_point.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < parsed.evaluation_point.size(); ++i)
            if (!zkdl::read_fr(in, offset, parsed.evaluation_point[i])) return false;
        if (!read_points(in, offset, parsed.commitment_a, max_items) ||
            !read_points(in, offset, parsed.commitment_b, max_items) ||
            !read_points(in, offset, parsed.commitment_c, max_items) ||
            !read_sumcheck(in, offset, parsed.sumcheck) ||
            !read_opening(in, offset, parsed.opening_a) ||
            !read_opening(in, offset, parsed.opening_b) ||
            !read_opening(in, offset, parsed.opening_c)) {
            if (error) *error = "truncated zkProd proof body";
            return false;
        }
        if (in.size() - offset < parsed.transcript_digest.size()) {
            if (error) *error = "truncated zkProd transcript digest";
            return false;
        }
        std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(
                      offset + parsed.transcript_digest.size()),
                  parsed.transcript_digest.begin());
        offset += parsed.transcript_digest.size();
        if (offset != in.size()) {
            if (error) *error = "trailing zkProd proof bytes";
            return false;
        }
        proof = parsed;
        return true;
    }

    bool shape_valid(std::string* error = nullptr) const {
        if (version != kVersion || num == 0U || dim == 0U ||
            log_domain >= 32U || evaluation_point.size() != log_domain) {
            if (error) *error = "invalid zkProd proof shape";
            return false;
        }
        const std::uint64_t domain = std::uint64_t(1) << log_domain;
        if (domain != std::uint64_t(num) * dim ||
            commitment_a.size() != num || commitment_b.size() != num ||
            commitment_c.size() != num) {
            if (error) *error = "zkProd proof dimensions do not match commitments";
            return false;
        }
        if (sumcheck.degree != 3U || sumcheck.rounds.size() != log_domain ||
            sumcheck.terminal_claims.size() != 3U) {
            if (error) *error = "zkProd proof has incomplete Hadamard Sumcheck";
            return false;
        }
        if (opening_a.commitment_index != 0U ||
            opening_b.commitment_index != 1U ||
            opening_c.commitment_index != 2U ||
            !zkdl::opening_shape_valid(opening_a, error) ||
            !zkdl::opening_shape_valid(opening_b, error) ||
            !zkdl::opening_shape_valid(opening_c, error)) return false;
        return true;
    }
};

inline bool same_fr(const zkdl::FrWire& left, const zkdl::FrWire& right) {
    return zkdl::fr_compare(left, right) == 0;
}

inline std::uint32_t log2_exact(std::uint32_t value) {
    if (value == 0U || (value & (value - 1U)) != 0U) return 0xffffffffU;
    std::uint32_t result = 0U;
    while (value > 1U) { value >>= 1U; ++result; }
    return result;
}

inline zkdl::FrWire mle_eval(const std::vector<zkdl::FrWire>& values,
                             const std::vector<zkdl::FrWire>& point) {
    if (point.size() >= 32U || values.size() != (size_t(1) << point.size()))
        throw std::invalid_argument("zkProd MLE shape mismatch");
    std::vector<zkdl::FrWire> current = values;
    const zkdl::FrWire zero{};
    (void)zero;
    for (size_t round = 0; round < point.size(); ++round) {
        const zkdl::FrWire& challenge = point[round];
        std::vector<zkdl::FrWire> next(current.size() / 2U);
        for (size_t i = 0; i < next.size(); ++i) {
            next[i] = zkdl::fr_add(
                current[2U * i],
                zkdl::fr_mul(challenge,
                             zkdl::fr_sub(current[2U * i + 1U], current[2U * i])));
        }
        current.swap(next);
    }
    return current[0];
}

inline std::vector<zkdl::FrWire> to_wires(const std::vector<Fr_t>& values) {
    std::vector<zkdl::FrWire> result;
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        result.push_back(zkdl::fr_wire(values[i]));
    return result;
}

inline zkdl::FrWire eq_weight(const std::vector<zkdl::FrWire>& point,
                              std::size_t index) {
    zkdl::FrWire result = zkdl::fr_from_u32(1U);
    const zkdl::FrWire one = zkdl::fr_from_u32(1U);
    for (size_t bit = 0; bit < point.size(); ++bit) {
        const bool set = ((index >> bit) & 1U) != 0U;
        result = zkdl::fr_mul(result, set ? point[bit] :
                                      zkdl::fr_sub(one, point[bit]));
    }
    return result;
}

inline std::vector<zkdl::FrWire> fold_outer(
    const std::vector<zkdl::FrWire>& values, std::uint32_t num,
    std::uint32_t dim, const std::vector<zkdl::FrWire>& outer_point) {
    if (num == 0U || dim == 0U || values.size() != size_t(num) * dim ||
        outer_point.size() != log2_exact(num))
        throw std::invalid_argument("zkProd outer MLE shape mismatch");
    std::vector<zkdl::FrWire> current = values;
    std::uint32_t current_num = num;
    for (size_t round = 0; round < outer_point.size(); ++round) {
        std::vector<zkdl::FrWire> next(size_t(current_num / 2U) * dim);
        for (std::uint32_t row = 0; row < current_num / 2U; ++row) {
            for (std::uint32_t column = 0; column < dim; ++column) {
                const size_t left = size_t(2U * row) * dim + column;
                const size_t right = left + dim;
                next[size_t(row) * dim + column] = zkdl::fr_add(
                    current[left],
                    zkdl::fr_mul(outer_point[round],
                                 zkdl::fr_sub(current[right], current[left])));
            }
        }
        current.swap(next);
        current_num /= 2U;
    }
    return current;
}

inline std::vector<zkdl::CpuG1Point> flatten_generators(
    const std::vector<zkdl::CpuG1Point>& generators, std::uint32_t num) {
    std::vector<zkdl::CpuG1Point> result;
    result.reserve(generators.size() * num);
    for (std::uint32_t row = 0; row < num; ++row)
        result.insert(result.end(), generators.begin(), generators.end());
    return result;
}

inline std::vector<G1Jacobian_t> commitment_rows(
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<zkdl::FrWire>& values, std::uint32_t num,
    std::uint32_t dim) {
    if (values.size() != size_t(num) * dim || generators.size() != dim)
        throw std::invalid_argument("zkProd commitment shape mismatch");
    std::vector<G1Jacobian_t> result;
    result.reserve(num);
    for (std::uint32_t row = 0; row < num; ++row) {
        std::vector<zkdl::FrWire> row_values(
            values.begin() + static_cast<std::ptrdiff_t>(size_t(row) * dim),
            values.begin() + static_cast<std::ptrdiff_t>(size_t(row + 1U) * dim));
        result.push_back(zkdl::cpu_g1_to_native(
            zkdl::cpu_g1_sum(generators, row_values)));
    }
    return result;
}

inline zkdl::CpuG1Point commitment_eval(
    const std::vector<G1Jacobian_t>& commitments,
    const std::vector<zkdl::FrWire>& outer_point) {
    if (commitments.size() != (size_t(1) << outer_point.size()))
        throw std::invalid_argument("zkProd commitment evaluation shape mismatch");
    std::vector<zkdl::CpuG1Point> points;
    std::vector<zkdl::FrWire> weights;
    points.reserve(commitments.size());
    weights.reserve(commitments.size());
    for (size_t row = 0; row < commitments.size(); ++row) {
        points.push_back(zkdl::cpu_g1_from_native(commitments[row]));
        weights.push_back(eq_weight(outer_point, row));
    }
    return zkdl::cpu_g1_sum(points, weights);
}

inline zkdl::CpuG1Point mle_g1_eval(
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<zkdl::FrWire>& point) {
    if (generators.size() != (size_t(1) << point.size()))
        throw std::invalid_argument("zkProd generator MLE shape mismatch");
    std::vector<zkdl::FrWire> weights;
    weights.reserve(generators.size());
    for (size_t index = 0; index < generators.size(); ++index)
        weights.push_back(eq_weight(point, index));
    return zkdl::cpu_g1_sum(generators, weights);
}

/* The native Hyrax fold stores the generator side in the orientation
 * G' = (1-u)G_1 + uG_0 while the scalar side uses the usual
 * s' = (1-u)s_0 + us_1.  This is intentional: it is what makes the
 * P/A/B quadratic opening equation telescope.  Consequently the final
 * generator is the reverse MLE of the generator vector, not the ordinary
 * MLE used for scalar tensors. */
inline zkdl::CpuG1Point hyrax_generator_eval(
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<zkdl::FrWire>& point) {
    if (generators.size() != (size_t(1) << point.size()))
        throw std::invalid_argument("zkProd generator MLE shape mismatch");
    std::vector<zkdl::FrWire> weights;
    weights.reserve(generators.size());
    const zkdl::FrWire one = zkdl::fr_from_u32(1U);
    for (size_t index = 0; index < generators.size(); ++index) {
        zkdl::FrWire weight = one;
        for (size_t bit = 0; bit < point.size(); ++bit) {
            const bool set = ((index >> bit) & 1U) != 0U;
            weight = zkdl::fr_mul(weight, set ? zkdl::fr_sub(one, point[bit])
                                               : point[bit]);
        }
        weights.push_back(weight);
    }
    return zkdl::cpu_g1_sum(generators, weights);
}

inline zkdl::EvaluationOpeningProof open_partitioned(
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<zkdl::FrWire>& values, std::uint32_t num,
    std::uint32_t dim, const std::vector<zkdl::FrWire>& point,
    std::uint32_t commitment_index) {
    const std::uint32_t inner_bits = log2_exact(dim);
    const std::uint32_t outer_bits = log2_exact(num);
    if (inner_bits == 0xffffffffU || outer_bits == 0xffffffffU ||
        point.size() != inner_bits + outer_bits)
        throw std::invalid_argument("zkProd opening point shape mismatch");
    const std::vector<zkdl::FrWire> inner(point.begin(),
                                           point.begin() + inner_bits);
    const std::vector<zkdl::FrWire> outer(point.begin() + inner_bits,
                                           point.end());
    const std::vector<zkdl::FrWire> partial = fold_outer(values, num, dim, outer);
    zkdl::EvaluationOpeningProof opening = zkdl::cpu_open(generators, partial, inner);
    opening.commitment_index = commitment_index;
    opening.outer_point.clear();
    for (size_t i = 0; i < outer.size(); ++i)
        opening.outer_point.push_back(zkdl::fr_value(outer[i]));
    return opening;
}

inline void absorb_statement(
    zkdl::Transcript& transcript, std::uint32_t num, std::uint32_t dim,
    const std::vector<G1Jacobian_t>& commitment_a,
    const std::vector<G1Jacobian_t>& commitment_b,
    const std::vector<G1Jacobian_t>& commitment_c) {
    std::vector<byte> statement;
    zkdl::append_u32(statement, num);
    zkdl::append_u32(statement, dim);
    Proof::append_points(statement, commitment_a);
    Proof::append_points(statement, commitment_b);
    Proof::append_points(statement, commitment_c);
    transcript.absorb("zkprod.statement.v1", statement);
}

inline void absorb_opening(zkdl::Transcript& transcript, const char* label,
                           const zkdl::EvaluationOpeningProof& opening) {
    std::vector<byte> encoded;
    zkdl::serialize_opening(opening, encoded);
    transcript.absorb(label, encoded);
}

inline Proof prove_host(
    std::uint32_t num, std::uint32_t dim,
    const std::vector<zkdl::FrWire>& values_a,
    const std::vector<zkdl::FrWire>& values_b,
    const std::vector<zkdl::FrWire>& values_c,
    const std::vector<zkdl::CpuG1Point>& generators_a,
    const std::vector<zkdl::CpuG1Point>& generators_b,
    const std::vector<zkdl::CpuG1Point>& generators_c) {
    const std::uint32_t inner_bits = log2_exact(dim);
    const std::uint32_t outer_bits = log2_exact(num);
    if (inner_bits == 0xffffffffU || outer_bits == 0xffffffffU ||
        num == 0U || dim == 0U ||
        values_a.size() != size_t(num) * dim ||
        values_b.size() != size_t(num) * dim ||
        values_c.size() != size_t(num) * dim ||
        generators_a.size() != dim || generators_b.size() != dim ||
        generators_c.size() != dim)
        throw std::invalid_argument("zkProd host proof shape mismatch");

    Proof proof;
    proof.num = num;
    proof.dim = dim;
    proof.log_domain = inner_bits + outer_bits;
    proof.commitment_a = commitment_rows(generators_a, values_a, num, dim);
    proof.commitment_b = commitment_rows(generators_b, values_b, num, dim);
    proof.commitment_c = commitment_rows(generators_c, values_c, num, dim);

    zkdl::Transcript transcript("zkdl-zkprod-proof-v1");
    absorb_statement(transcript, num, dim, proof.commitment_a,
                     proof.commitment_b, proof.commitment_c);
    proof.evaluation_point.reserve(proof.log_domain);
    for (std::uint32_t i = 0; i < proof.log_domain; ++i)
        proof.evaluation_point.push_back(
            transcript.challenge("zkprod.eval." + std::to_string(i)));

    const std::vector<zkdl::FrWire> evaluation_point_wires =
        to_wires(proof.evaluation_point);
    const zkdl::FrWire initial_claim =
        mle_eval(values_c, evaluation_point_wires);
    std::vector<zkdl::FrWire> current_a = values_a;
    std::vector<zkdl::FrWire> current_b = values_b;
    zkdl::FrWire weight = zkdl::fr_from_u32(1U);
    zkdl::SumcheckProof& sumcheck = proof.sumcheck;
    sumcheck.label = "zkprod.hadamard.v1";
    sumcheck.degree = 3U;
    sumcheck.rounds.reserve(proof.log_domain);
    zkdl::FrWire claim = initial_claim;
    const zkdl::FrWire one = zkdl::fr_from_u32(1U);
    const zkdl::FrWire two = zkdl::fr_from_u32(2U);
    const zkdl::FrWire three = zkdl::fr_from_u32(3U);
    for (std::uint32_t round = 0; round < proof.log_domain; ++round) {
        const zkdl::FrWire z = zkdl::fr_wire(proof.evaluation_point[round]);
        const zkdl::FrWire e0 = zkdl::fr_sub(one, z);
        const zkdl::FrWire slope = zkdl::fr_sub(z, e0);
        const zkdl::FrWire e2 = zkdl::fr_add(e0, zkdl::fr_mul(two, slope));
        const zkdl::FrWire e3 = zkdl::fr_add(e0, zkdl::fr_mul(three, slope));
        const std::vector<zkdl::FrWire> tail(
            evaluation_point_wires.begin() + round + 1,
            evaluation_point_wires.end());
        std::vector<zkdl::FrWire> p0(current_a.size() / 2U);
        std::vector<zkdl::FrWire> p1(current_a.size() / 2U);
        std::vector<zkdl::FrWire> p2(current_a.size() / 2U);
        for (size_t i = 0; i < p0.size(); ++i) {
            const zkdl::FrWire& a0 = current_a[2U * i];
            const zkdl::FrWire& a1 = current_a[2U * i + 1U];
            const zkdl::FrWire& b0 = current_b[2U * i];
            const zkdl::FrWire& b1 = current_b[2U * i + 1U];
            const zkdl::FrWire a2 = zkdl::fr_sub(zkdl::fr_add(a1, a1), a0);
            const zkdl::FrWire b2 = zkdl::fr_sub(zkdl::fr_add(b1, b1), b0);
            p0[i] = zkdl::fr_mul(a0, b0);
            p1[i] = zkdl::fr_mul(a1, b1);
            p2[i] = zkdl::fr_mul(a2, b2);
        }
        zkdl::SumcheckRound message;
        message.evaluations.push_back(zkdl::fr_value(
            zkdl::fr_mul(weight, zkdl::fr_mul(e0, mle_eval(p0, tail)))));
        message.evaluations.push_back(zkdl::fr_value(
            zkdl::fr_mul(weight, zkdl::fr_mul(z, mle_eval(p1, tail)))));
        message.evaluations.push_back(zkdl::fr_value(
            zkdl::fr_mul(weight, zkdl::fr_mul(e2, mle_eval(p2, tail)))));
        std::vector<zkdl::FrWire> p3(p0.size());
        for (size_t i = 0; i < p3.size(); ++i) {
            const zkdl::FrWire& a0 = current_a[2U * i];
            const zkdl::FrWire& a1 = current_a[2U * i + 1U];
            const zkdl::FrWire& b0 = current_b[2U * i];
            const zkdl::FrWire& b1 = current_b[2U * i + 1U];
            const zkdl::FrWire a3 = zkdl::fr_sub(zkdl::fr_mul(three, a1),
                                                  zkdl::fr_mul(two, a0));
            const zkdl::FrWire b3 = zkdl::fr_sub(zkdl::fr_mul(three, b1),
                                                  zkdl::fr_mul(two, b0));
            p3[i] = zkdl::fr_mul(a3, b3);
        }
        message.evaluations.push_back(zkdl::fr_value(
            zkdl::fr_mul(weight, zkdl::fr_mul(e3, mle_eval(p3, tail)))));
        sumcheck.rounds.push_back(message);
        transcript.absorb_fr(sumcheck.label + ".round", message.evaluations);
        const zkdl::FrWire challenge = zkdl::fr_wire(
            transcript.challenge(sumcheck.label + ".challenge"));
        claim = zkdl::interpolate_at(message.evaluations, challenge);
        const zkdl::FrWire affine = zkdl::fr_add(
            e0, zkdl::fr_mul(challenge, zkdl::fr_sub(z, e0)));
        weight = zkdl::fr_mul(weight, affine);
        std::vector<zkdl::FrWire> next_a(current_a.size() / 2U);
        std::vector<zkdl::FrWire> next_b(current_b.size() / 2U);
        for (size_t i = 0; i < next_a.size(); ++i) {
            next_a[i] = zkdl::fr_add(
                current_a[2U * i],
                zkdl::fr_mul(challenge,
                             zkdl::fr_sub(current_a[2U * i + 1U],
                                          current_a[2U * i])));
            next_b[i] = zkdl::fr_add(
                current_b[2U * i],
                zkdl::fr_mul(challenge,
                             zkdl::fr_sub(current_b[2U * i + 1U],
                                          current_b[2U * i])));
        }
        current_a.swap(next_a);
        current_b.swap(next_b);
        (void)claim;
    }
    const zkdl::FrWire terminal_a = current_a[0];
    const zkdl::FrWire terminal_b = current_b[0];
    sumcheck.terminal_claims = {zkdl::fr_value(terminal_a),
                                zkdl::fr_value(terminal_b),
                                zkdl::fr_value(initial_claim)};
    sumcheck.final_claim = zkdl::fr_value(
        zkdl::fr_mul(weight, zkdl::fr_mul(terminal_a, terminal_b)));

    std::vector<zkdl::FrWire> sumcheck_point;
    zkdl::Transcript replay("zkdl-zkprod-proof-v1");
    absorb_statement(replay, num, dim, proof.commitment_a,
                     proof.commitment_b, proof.commitment_c);
    for (std::uint32_t i = 0; i < proof.log_domain; ++i)
        (void)replay.challenge("zkprod.eval." + std::to_string(i));
    for (size_t i = 0; i < sumcheck.rounds.size(); ++i) {
        replay.absorb_fr(sumcheck.label + ".round", sumcheck.rounds[i].evaluations);
        sumcheck_point.push_back(zkdl::fr_wire(
            replay.challenge(sumcheck.label + ".challenge")));
    }
    proof.opening_a = open_partitioned(generators_a, values_a, num, dim,
                                       sumcheck_point, 0U);
    proof.opening_b = open_partitioned(generators_b, values_b, num, dim,
                                       sumcheck_point, 1U);
    proof.opening_c = open_partitioned(generators_c, values_c, num, dim,
                                       evaluation_point_wires, 2U);
    absorb_opening(transcript, "zkprod.opening.a", proof.opening_a);
    absorb_opening(transcript, "zkprod.opening.b", proof.opening_b);
    absorb_opening(transcript, "zkprod.opening.c", proof.opening_c);
    proof.transcript_digest = transcript.digest();
    std::string error;
    if (!proof.shape_valid(&error)) throw std::runtime_error(error);
    return proof;
}

inline bool opening_point_matches(
    const zkdl::EvaluationOpeningProof& opening,
    const std::vector<zkdl::FrWire>& point, std::uint32_t num,
    std::uint32_t dim, std::string* error) {
    const std::uint32_t inner_bits = log2_exact(dim);
    const std::uint32_t outer_bits = log2_exact(num);
    if (inner_bits == 0xffffffffU || outer_bits == 0xffffffffU ||
        opening.generator_size != dim ||
        opening.inner_point.size() != inner_bits ||
        opening.outer_point.size() != outer_bits ||
        point.size() != inner_bits + outer_bits) {
        if (error) *error = "zkProd terminal opening shape mismatch";
        return false;
    }
    for (std::uint32_t i = 0; i < inner_bits; ++i)
        if (!same_fr(zkdl::fr_wire(opening.inner_point[i]), point[i])) {
            if (error) *error = "zkProd terminal opening inner point mismatch";
            return false;
        }
    for (std::uint32_t i = 0; i < outer_bits; ++i)
        if (!same_fr(zkdl::fr_wire(opening.outer_point[i]),
                     point[inner_bits + i])) {
            if (error) *error = "zkProd terminal opening outer point mismatch";
            return false;
        }
    return true;
}

inline bool verify_one_opening(
    const zkdl::EvaluationOpeningProof& opening,
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<G1Jacobian_t>& commitments,
    const std::vector<zkdl::FrWire>& point, std::uint32_t num,
    std::uint32_t dim, std::string* error) {
    if (!opening_point_matches(opening, point, num, dim, error)) return false;
    const zkdl::CpuG1Point expected = commitment_eval(
        commitments,
        std::vector<zkdl::FrWire>(point.begin() + log2_exact(dim), point.end()));
    const zkdl::CpuG1Point expected_final =
        hyrax_generator_eval(generators,
                             std::vector<zkdl::FrWire>(point.begin(),
                                                       point.begin() + log2_exact(dim)));
    if (!zkdl::cpu_g1_equal(
            zkdl::cpu_g1_from_native(opening.rounds.back()), expected_final)) {
        if (error) *error = "zkProd terminal generator is not bound to VK";
        return false;
    }
    if (!zkdl::cpu_verify_opening(opening, expected, error)) return false;
    return true;
}

inline bool commitments_match(
    const std::vector<G1Jacobian_t>& embedded,
    const std::vector<G1Jacobian_t>& expected, std::string* error) {
    if (embedded.size() != expected.size()) {
        if (error) *error = "zkProd commitment count mismatch";
        return false;
    }
    for (size_t i = 0; i < embedded.size(); ++i)
        if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(embedded[i]),
                                zkdl::cpu_g1_from_native(expected[i]))) {
            if (error) *error = "zkProd public commitment mismatch";
            return false;
        }
    return true;
}

inline bool verify_host(
    const Proof& proof, std::uint32_t num, std::uint32_t dim,
    const std::vector<zkdl::CpuG1Point>& generators_a,
    const std::vector<zkdl::CpuG1Point>& generators_b,
    const std::vector<zkdl::CpuG1Point>& generators_c,
    const std::vector<G1Jacobian_t>& expected_commitment_a,
    const std::vector<G1Jacobian_t>& expected_commitment_b,
    const std::vector<G1Jacobian_t>& expected_commitment_c,
    std::string* error = nullptr) {
    if (!proof.shape_valid(error) || proof.num != num || proof.dim != dim)
        return false;
    if (generators_a.size() != dim || generators_b.size() != dim ||
        generators_c.size() != dim ||
        expected_commitment_a.size() != num ||
        expected_commitment_b.size() != num ||
        expected_commitment_c.size() != num) {
        if (error) *error = "zkProd verifier public shape mismatch";
        return false;
    }
    if (!commitments_match(proof.commitment_a, expected_commitment_a, error) ||
        !commitments_match(proof.commitment_b, expected_commitment_b, error) ||
        !commitments_match(proof.commitment_c, expected_commitment_c, error))
        return false;

    zkdl::Transcript transcript("zkdl-zkprod-proof-v1");
    absorb_statement(transcript, num, dim, expected_commitment_a,
                     expected_commitment_b, expected_commitment_c);
    std::vector<zkdl::FrWire> evaluation_point;
    for (std::uint32_t i = 0; i < proof.log_domain; ++i)
        evaluation_point.push_back(zkdl::fr_wire(
            transcript.challenge("zkprod.eval." + std::to_string(i))));
    for (size_t i = 0; i < evaluation_point.size(); ++i)
        if (!same_fr(evaluation_point[i], zkdl::fr_wire(proof.evaluation_point[i]))) {
            if (error) *error = "zkProd FS evaluation point mismatch";
            return false;
        }

    if (!verify_one_opening(proof.opening_c, generators_c,
                            expected_commitment_c, evaluation_point, num, dim,
                            error)) return false;
    std::vector<Fr_t> challenges_native;
    if (!zkdl::verify_sumcheck(proof.sumcheck, proof.opening_c.value,
                               transcript, &challenges_native, error)) return false;
    std::vector<zkdl::FrWire> challenges;
    challenges.reserve(challenges_native.size());
    for (size_t i = 0; i < challenges_native.size(); ++i)
        challenges.push_back(zkdl::fr_wire(challenges_native[i]));
    if (challenges.size() != proof.log_domain) {
        if (error) *error = "zkProd Sumcheck challenge count mismatch";
        return false;
    }
    if (!verify_one_opening(proof.opening_a, generators_a,
                            expected_commitment_a, challenges, num, dim,
                            error) ||
        !verify_one_opening(proof.opening_b, generators_b,
                            expected_commitment_b, challenges, num, dim,
                            error)) return false;

    if (!same_fr(zkdl::fr_wire(proof.sumcheck.terminal_claims[0]),
                 zkdl::fr_wire(proof.opening_a.value)) ||
        !same_fr(zkdl::fr_wire(proof.sumcheck.terminal_claims[1]),
                 zkdl::fr_wire(proof.opening_b.value)) ||
        !same_fr(zkdl::fr_wire(proof.sumcheck.terminal_claims[2]),
                 zkdl::fr_wire(proof.opening_c.value))) {
        if (error) *error = "zkProd terminal claims are not opening-bound";
        return false;
    }
    zkdl::FrWire weight = zkdl::fr_from_u32(1U);
    const zkdl::FrWire one = zkdl::fr_from_u32(1U);
    for (size_t i = 0; i < evaluation_point.size(); ++i) {
        const zkdl::FrWire z = evaluation_point[i];
        const zkdl::FrWire e0 = zkdl::fr_sub(one, z);
        const zkdl::FrWire slope = zkdl::fr_sub(z, e0);
        weight = zkdl::fr_mul(weight,
                              zkdl::fr_add(e0,
                                           zkdl::fr_mul(challenges[i], slope)));
    }
    const zkdl::FrWire expected_final = zkdl::fr_mul(
        weight, zkdl::fr_mul(zkdl::fr_wire(proof.opening_a.value),
                             zkdl::fr_wire(proof.opening_b.value)));
    if (!same_fr(expected_final, zkdl::fr_wire(proof.sumcheck.final_claim))) {
        if (error) *error = "zkProd Hadamard terminal equation failed";
        return false;
    }
    absorb_opening(transcript, "zkprod.opening.a", proof.opening_a);
    absorb_opening(transcript, "zkprod.opening.b", proof.opening_b);
    absorb_opening(transcript, "zkprod.opening.c", proof.opening_c);
    if (transcript.digest() != proof.transcript_digest) {
        if (error) *error = "zkProd transcript digest mismatch";
        return false;
    }
    return true;
}

}  // namespace zkdl_prod

#endif
