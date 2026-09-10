#ifndef ZKDL_CPU_BACKEND_CUH
#define ZKDL_CPU_BACKEND_CUH

/*
 * Portable host implementation of the same BLS12-381 scalar/ G1 Jacobian
 * equations used by the native CUDA tensors.  This header intentionally has
 * no CUDA calls in its execution path: a CPU runner can use the same
 * ProofEnvelope, Transcript and EvaluationOpeningProof wire objects while
 * evaluating commitment/opening equations on the host.
 */

#include "proof.cuh"

namespace zkdl {

using CpuFp = std::array<std::uint32_t, blstrs__fp__Fp_LIMBS>;

static const CpuFp kFpModulus = {{
    4294945451U, 3120496639U, 2975072255U, 514588670U,
    4138792484U, 1731252896U, 4085584575U, 1685539716U,
    1129032919U, 1260103606U, 964683418U, 436277738U
}};

inline int cpu_fp_compare(const CpuFp& a, const CpuFp& b) {
    for (int i = blstrs__fp__Fp_LIMBS - 1; i >= 0; --i) {
        if (a[static_cast<size_t>(i)] < b[static_cast<size_t>(i)]) return -1;
        if (a[static_cast<size_t>(i)] > b[static_cast<size_t>(i)]) return 1;
    }
    return 0;
}

inline CpuFp cpu_fp_sub_raw(const CpuFp& a, const CpuFp& b) {
    CpuFp out{};
    std::uint64_t borrow = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t ai = a[i];
        const std::uint64_t bi = static_cast<std::uint64_t>(b[i]) + borrow;
        out[i] = static_cast<std::uint32_t>(ai - bi);
        borrow = ai < bi ? 1U : 0U;
    }
    return out;
}

inline CpuFp cpu_fp_add(const CpuFp& a, const CpuFp& b) {
    CpuFp out{};
    std::uint64_t carry = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32;
    }
    if (carry != 0 || cpu_fp_compare(out, kFpModulus) >= 0) out = cpu_fp_sub_raw(out, kFpModulus);
    return out;
}

inline CpuFp cpu_fp_sub(const CpuFp& a, const CpuFp& b) {
    if (cpu_fp_compare(a, b) >= 0) return cpu_fp_sub_raw(a, b);
    return cpu_fp_sub_raw(kFpModulus, cpu_fp_sub_raw(b, a));
}

inline CpuFp cpu_fp_reduce(const std::array<std::uint32_t, 2 * blstrs__fp__Fp_LIMBS>& product) {
    std::array<std::uint32_t, blstrs__fp__Fp_LIMBS + 1> remainder{};
    std::array<std::uint32_t, blstrs__fp__Fp_LIMBS + 1> modulus{};
    for (size_t i = 0; i < blstrs__fp__Fp_LIMBS; ++i) modulus[i] = kFpModulus[i];
    for (int bit = static_cast<int>(product.size() * 32) - 1; bit >= 0; --bit) {
        std::uint64_t carry = (product[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U;
        for (size_t i = 0; i < remainder.size(); ++i) {
            const std::uint64_t shifted = static_cast<std::uint64_t>(remainder[i]) * 2U + carry;
            remainder[i] = static_cast<std::uint32_t>(shifted);
            carry = shifted >> 32;
        }
        int comparison = remainder.back() == 0 ? 0 : 1;
        if (comparison == 0) {
            for (int i = blstrs__fp__Fp_LIMBS - 1; i >= 0; --i) {
                if (remainder[static_cast<size_t>(i)] != modulus[static_cast<size_t>(i)]) {
                    comparison = remainder[static_cast<size_t>(i)] > modulus[static_cast<size_t>(i)] ? 1 : -1;
                    break;
                }
            }
        }
        if (comparison >= 0) {
            std::uint64_t borrow = 0;
            for (size_t i = 0; i < remainder.size(); ++i) {
                const std::uint64_t subtrahend = static_cast<std::uint64_t>(modulus[i]) + borrow;
                const std::uint64_t value = remainder[i];
                remainder[i] = static_cast<std::uint32_t>(value - subtrahend);
                borrow = value < subtrahend ? 1U : 0U;
            }
        }
    }
    CpuFp out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = remainder[i];
    return out;
}

inline CpuFp cpu_fp_mul(const CpuFp& a, const CpuFp& b) {
    std::array<std::uint32_t, 2 * blstrs__fp__Fp_LIMBS> product{};
    for (size_t i = 0; i < a.size(); ++i) {
        std::uint64_t carry = 0;
        for (size_t j = 0; j < b.size(); ++j) {
            const size_t at = i + j;
            const unsigned __int128 cur =
                static_cast<unsigned __int128>(a[i]) * b[j] + product[at] + carry;
            product[at] = static_cast<std::uint32_t>(cur);
            carry = static_cast<std::uint64_t>(cur >> 32);
        }
        size_t at = i + b.size();
        while (carry != 0 && at < product.size()) {
            const std::uint64_t cur = static_cast<std::uint64_t>(product[at]) + carry;
            product[at] = static_cast<std::uint32_t>(cur);
            carry = cur >> 32;
            ++at;
        }
    }
    return cpu_fp_reduce(product);
}

inline CpuFp cpu_fp_pow(CpuFp base, const CpuFp& exponent) {
    CpuFp result{};
    result[0] = 1U;
    for (int bit = blstrs__fp__Fp_LIMBS * 32 - 1; bit >= 0; --bit) {
        result = cpu_fp_mul(result, result);
        if ((exponent[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U)
            result = cpu_fp_mul(result, base);
    }
    return result;
}

inline CpuFp cpu_fp_modulus_minus_two() {
    CpuFp out = kFpModulus;
    std::uint64_t borrow = 2U;
    for (size_t i = 0; i < out.size() && borrow != 0; ++i) {
        const std::uint64_t value = out[i];
        out[i] = static_cast<std::uint32_t>(value - borrow);
        borrow = value < borrow ? 1U : 0U;
    }
    return out;
}

inline CpuFp cpu_fp_inv(const CpuFp& value) {
    CpuFp zero{};
    if (cpu_fp_compare(value, zero) == 0) throw std::invalid_argument("inverse of zero");
    return cpu_fp_pow(value, cpu_fp_modulus_minus_two());
}

inline CpuFp cpu_fp_from_native(const Fp_t& value) {
    CpuFp out{};
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; ++i) out[i] = value.val[i];
    return out;
}

inline Fp_t cpu_fp_to_native(const CpuFp& value) {
    Fp_t out{};
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; ++i) out.val[i] = value[i];
    return out;
}

inline CpuFp cpu_fp_one() { CpuFp out{}; out[0] = 1U; return out; }

inline CpuFp cpu_fp_mont_factor() {
    CpuFp out = cpu_fp_one();
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS * 32U; ++i) out = cpu_fp_add(out, out);
    return out;
}

inline CpuFp cpu_fp_from_mont(const CpuFp& value) {
    // Native Fp coordinates are Montgomery residues (Fp_ONE is R mod p).
    // The host implementation stores ordinary residues, so multiply by
    // R^{-1} rather than treating a Montgomery coordinate as canonical.
    static const CpuFp r_inverse = cpu_fp_inv(cpu_fp_from_native(Fp_t{ {
        196605U, 1980301312U, 3289120770U, 3958636555U,
        1405573306U, 1598593111U, 1884444485U, 2010011731U,
        2723605613U, 1543969431U, 4202751123U, 368467651U
    } }));
    return cpu_fp_mul(value, r_inverse);
}

inline CpuFp cpu_fp_to_mont(const CpuFp& value) {
    return cpu_fp_mul(value, cpu_fp_mont_factor());
}

struct CpuG1Point {
    CpuFp x;
    CpuFp y;
    CpuFp z;

    CpuG1Point() : x{}, y{}, z{} {}
    static CpuG1Point zero() { return CpuG1Point(); }
    bool is_zero() const { CpuFp zero{}; return cpu_fp_compare(z, zero) == 0; }
};

inline CpuG1Point cpu_g1_generator() {
    CpuG1Point out;
    out.x = cpu_fp_from_mont(cpu_fp_from_native(G1_generator_x_mont));
    out.y = cpu_fp_from_mont(cpu_fp_from_native(G1_generator_y_mont));
    out.z = cpu_fp_one();
    return out;
}

inline CpuG1Point cpu_g1_from_native(const G1Jacobian_t& point) {
    CpuG1Point out;
    out.x = cpu_fp_from_mont(cpu_fp_from_native(point.x));
    out.y = cpu_fp_from_mont(cpu_fp_from_native(point.y));
    out.z = cpu_fp_from_mont(cpu_fp_from_native(point.z));
    return out;
}

inline G1Jacobian_t cpu_g1_to_native(const CpuG1Point& point) {
    G1Jacobian_t out{};
    out.x = cpu_fp_to_native(cpu_fp_to_mont(point.x));
    out.y = cpu_fp_to_native(cpu_fp_to_mont(point.y));
    out.z = cpu_fp_to_native(cpu_fp_to_mont(point.z));
    return out;
}

inline CpuG1Point cpu_g1_double(const CpuG1Point& input) {
    if (input.is_zero()) return input;
    const CpuFp a = cpu_fp_mul(input.x, input.x);
    const CpuFp b = cpu_fp_mul(input.y, input.y);
    CpuFp c = cpu_fp_mul(b, b);
    CpuFp d = cpu_fp_add(input.x, b);
    d = cpu_fp_mul(d, d); d = cpu_fp_sub(cpu_fp_sub(d, a), c); d = cpu_fp_add(d, d);
    const CpuFp e = cpu_fp_add(cpu_fp_add(a, a), a);
    const CpuFp f = cpu_fp_mul(e, e);
    CpuG1Point out;
    out.z = cpu_fp_add(input.y, input.z); out.z = cpu_fp_sub(cpu_fp_sub(cpu_fp_mul(out.z, out.z), cpu_fp_mul(input.y, input.y)), cpu_fp_mul(input.z, input.z));
    out.x = cpu_fp_sub(cpu_fp_sub(f, d), d);
    c = cpu_fp_add(cpu_fp_add(c, c), cpu_fp_add(c, c)); c = cpu_fp_add(c, c);
    out.y = cpu_fp_sub(cpu_fp_mul(cpu_fp_sub(d, out.x), e), c);
    return out;
}

inline CpuG1Point cpu_g1_add(const CpuG1Point& left, const CpuG1Point& right) {
    if (left.is_zero()) return right;
    if (right.is_zero()) return left;
    const CpuFp z1z1 = cpu_fp_mul(left.z, left.z);
    const CpuFp z2z2 = cpu_fp_mul(right.z, right.z);
    const CpuFp u1 = cpu_fp_mul(left.x, z2z2);
    const CpuFp u2 = cpu_fp_mul(right.x, z1z1);
    CpuFp s1 = cpu_fp_mul(cpu_fp_mul(left.y, right.z), z2z2);
    const CpuFp s2 = cpu_fp_mul(cpu_fp_mul(right.y, left.z), z1z1);
    if (cpu_fp_compare(u1, u2) == 0 && cpu_fp_compare(s1, s2) == 0) return cpu_g1_double(left);
    const CpuFp h = cpu_fp_sub(u2, u1);
    CpuFp i = cpu_fp_add(h, h); i = cpu_fp_mul(i, i);
    const CpuFp j = cpu_fp_mul(h, i);
    CpuFp r = cpu_fp_sub(s2, s1); r = cpu_fp_add(r, r);
    const CpuFp v = cpu_fp_mul(u1, i);
    CpuG1Point out;
    out.x = cpu_fp_sub(cpu_fp_sub(cpu_fp_mul(r, r), j), cpu_fp_add(v, v));
    out.y = cpu_fp_mul(cpu_fp_sub(v, out.x), r);
    s1 = cpu_fp_add(cpu_fp_mul(s1, j), cpu_fp_mul(s1, j));
    out.y = cpu_fp_sub(out.y, s1);
    out.z = cpu_fp_add(left.z, right.z); out.z = cpu_fp_mul(out.z, out.z);
    out.z = cpu_fp_sub(cpu_fp_sub(out.z, z1z1), z2z2); out.z = cpu_fp_mul(out.z, h);
    return out;
}

inline CpuG1Point cpu_g1_neg(const CpuG1Point& input) {
    if (input.is_zero()) return input;
    CpuG1Point out = input;
    CpuFp zero{};
    out.y = cpu_fp_sub(zero, out.y);
    return out;
}

inline CpuG1Point cpu_g1_mul(const CpuG1Point& input, const FrWire& scalar) {
    CpuG1Point result = CpuG1Point::zero();
    CpuG1Point base = input;
    int highest = blstrs__scalar__Scalar_LIMBS * 32 - 1;
    while (highest >= 0 && ((scalar[static_cast<size_t>(highest / 32)] >>
                             (highest % 32)) & 1U) == 0U) --highest;
    for (int bit = 0; bit <= highest; ++bit) {
        if ((scalar[bit / 32U] >> (bit % 32U)) & 1U) result = cpu_g1_add(result, base);
        base = cpu_g1_double(base);
    }
    return result;
}

inline bool cpu_g1_equal(const CpuG1Point& left, const CpuG1Point& right) {
    if (left.is_zero() || right.is_zero()) return left.is_zero() && right.is_zero();
    const CpuFp z1z1 = cpu_fp_mul(left.z, left.z);
    const CpuFp z2z2 = cpu_fp_mul(right.z, right.z);
    const CpuFp x1 = cpu_fp_mul(left.x, z2z2);
    const CpuFp x2 = cpu_fp_mul(right.x, z1z1);
    if (cpu_fp_compare(x1, x2) != 0) return false;
    const CpuFp y1 = cpu_fp_mul(left.y, cpu_fp_mul(right.z, z2z2));
    const CpuFp y2 = cpu_fp_mul(right.y, cpu_fp_mul(left.z, z1z1));
    return cpu_fp_compare(y1, y2) == 0;
}

inline CpuG1Point cpu_g1_sum(const std::vector<CpuG1Point>& points,
                             const std::vector<FrWire>& scalars) {
    if (points.size() != scalars.size()) throw std::invalid_argument("G1 MSM shape mismatch");
    CpuG1Point result = CpuG1Point::zero();
    for (size_t i = 0; i < points.size(); ++i) result = cpu_g1_add(result, cpu_g1_mul(points[i], scalars[i]));
    return result;
}

inline CpuG1Point cpu_opening_commitment_eval(const std::vector<CpuG1Point>& generators,
                                              const std::vector<FrWire>& values) {
    return cpu_g1_sum(generators, values);
}

inline EvaluationOpeningProof cpu_open(const std::vector<CpuG1Point>& generators,
                                       const std::vector<FrWire>& values,
                                       const std::vector<FrWire>& point) {
    if (generators.empty() || generators.size() != values.size() ||
        generators.size() != (size_t(1) << point.size()))
        throw std::invalid_argument("CPU opening shape mismatch");
    EvaluationOpeningProof opening;
    opening.generator_size = static_cast<std::uint32_t>(generators.size());
    opening.inner_point.reserve(point.size());
    for (size_t i = 0; i < point.size(); ++i) opening.inner_point.push_back(fr_value(point[i]));
    opening.commitment_eval = cpu_g1_to_native(cpu_opening_commitment_eval(generators, values));

    std::vector<CpuG1Point> current_generators = generators;
    std::vector<FrWire> current_values = values;
    for (size_t round = 0; round < point.size(); ++round) {
        const FrWire u = point[round];
        const size_t next_size = current_values.size() / 2U;
        std::vector<CpuG1Point> next_generators(next_size);
        std::vector<FrWire> next_values(next_size);
        CpuG1Point round_p = CpuG1Point::zero();
        CpuG1Point round_a = CpuG1Point::zero();
        CpuG1Point round_b = CpuG1Point::zero();
        for (size_t i = 0; i < next_size; ++i) {
            const size_t i0 = 2U * i, i1 = i0 + 1U;
            const FrWire delta = fr_sub(current_values[i1], current_values[i0]);
            next_values[i] = fr_add(current_values[i0], fr_mul(u, delta));
            next_generators[i] = cpu_g1_add(current_generators[i1],
                                            cpu_g1_mul(cpu_g1_add(current_generators[i0], cpu_g1_neg(current_generators[i1])), u));
            round_p = cpu_g1_add(round_p, cpu_g1_add(
                cpu_g1_mul(current_generators[i0], current_values[i0]),
                cpu_g1_mul(current_generators[i1], current_values[i1])));
            round_a = cpu_g1_add(round_a,
                cpu_g1_mul(current_generators[i1], current_values[i0]));
            round_b = cpu_g1_add(round_b,
                cpu_g1_mul(current_generators[i0], current_values[i1]));
        }
        opening.rounds.push_back(cpu_g1_to_native(round_p));
        opening.rounds.push_back(cpu_g1_to_native(round_a));
        opening.rounds.push_back(cpu_g1_to_native(round_b));
        current_generators.swap(next_generators);
        current_values.swap(next_values);
    }
    opening.value = fr_value(current_values[0]);
    opening.rounds.push_back(cpu_g1_to_native(current_generators[0]));
    return opening;
}

inline bool cpu_verify_opening(const EvaluationOpeningProof& opening,
                               const CpuG1Point& expected_commitment_eval,
                               std::string* error = nullptr) {
    if (!opening_shape_valid(opening, error)) return false;
    if (!cpu_g1_equal(cpu_g1_from_native(opening.commitment_eval), expected_commitment_eval)) {
        if (error) *error = "CPU opening commitment evaluation mismatch";
        return false;
    }
    CpuG1Point current = expected_commitment_eval;
    for (size_t i = 0; i < opening.inner_point.size(); ++i) {
        const CpuG1Point p = cpu_g1_from_native(opening.rounds[3U * i]);
        const CpuG1Point a = cpu_g1_from_native(opening.rounds[3U * i + 1U]);
        const CpuG1Point b = cpu_g1_from_native(opening.rounds[3U * i + 2U]);
        const FrWire u = fr_wire(opening.inner_point[i]);
        const FrWire one = fr_from_u32(1U);
        const FrWire one_minus = fr_sub(one, u);
        const FrWire c_a = fr_mul(one_minus, one_minus);
        const FrWire c_p = fr_mul(u, one_minus);
        const FrWire c_b = fr_mul(u, u);
        current = cpu_g1_add(cpu_g1_add(cpu_g1_mul(a, c_a), cpu_g1_mul(p, c_p)), cpu_g1_mul(b, c_b));
    }
    const CpuG1Point expected = cpu_g1_mul(cpu_g1_from_native(opening.rounds.back()), fr_wire(opening.value));
    if (!cpu_g1_equal(current, expected)) {
        if (error) *error = "CPU Hyrax opening equation failed";
        return false;
    }
    return true;
}

/* Explicit hiding commitment metadata.  The secret randomness is retained
 * only by the prover object; the serialized CommitmentRecord contains the
 * value commitment, blinding generator, full commitment and a Schnorr proof
 * of knowledge of the blinder. */
class CpuHidingCommitment {
public:
    std::string label;
    std::uint32_t visibility;
    std::vector<CpuG1Point> generators;
    CpuG1Point blinding_generator;
    std::vector<FrWire> values;
    FrWire randomness;
    FrWire schnorr_nonce;
    CpuG1Point value_commitment;
    CpuG1Point commitment;
    CpuG1Point schnorr_R;
    FrWire schnorr_response;

    CpuHidingCommitment(const std::string& label_, std::uint32_t visibility_,
                       const std::vector<CpuG1Point>& generators_,
                       const CpuG1Point& blinding_generator_,
                       const std::vector<FrWire>& values_, const FrWire& randomness_,
                       const FrWire& schnorr_nonce_)
        : label(label_), visibility(visibility_), generators(generators_),
          blinding_generator(blinding_generator_), values(values_), randomness(randomness_),
          schnorr_nonce(schnorr_nonce_), value_commitment(cpu_opening_commitment_eval(generators_, values_)),
          commitment(), schnorr_R(cpu_g1_mul(blinding_generator_, schnorr_nonce_)), schnorr_response{} {
        commitment = cpu_g1_add(value_commitment, cpu_g1_mul(blinding_generator, randomness));
        const FrWire challenge = hiding_challenge(label, visibility, value_commitment,
                                                  blinding_generator, commitment, schnorr_R);
        schnorr_response = fr_add(schnorr_nonce, fr_mul(challenge, randomness));
    }

    static FrWire hiding_challenge(const std::string& label_, std::uint32_t visibility_,
                                   const CpuG1Point& value_commitment_,
                                   const CpuG1Point& blinding_generator_,
                                   const CpuG1Point& commitment_, const CpuG1Point& schnorr_R_) {
        Transcript transcript("zkdl-hiding-commitment-v1");
        std::vector<byte> message;
        append_u32(message, visibility_);
        append_g1(message, cpu_g1_to_native(value_commitment_));
        append_g1(message, cpu_g1_to_native(blinding_generator_));
        append_g1(message, cpu_g1_to_native(commitment_));
        append_g1(message, cpu_g1_to_native(schnorr_R_));
        transcript.absorb(label_, message);
        return fr_wire(transcript.challenge(label_ + ".schnorr"));
    }

    CommitmentRecord record() const {
        CommitmentRecord out;
        out.label = label;
        out.visibility = visibility;
        out.points.reserve(generators.size());
        for (size_t i = 0; i < generators.size(); ++i) out.points.push_back(cpu_g1_to_native(generators[i]));
        out.value_commitment = cpu_g1_to_native(value_commitment);
        out.blinding_generator = cpu_g1_to_native(blinding_generator);
        out.commitment = cpu_g1_to_native(commitment);
        out.schnorr_R = cpu_g1_to_native(schnorr_R);
        out.schnorr_response = fr_value(schnorr_response);
        return out;
    }

    static bool verify_record(const CommitmentRecord& record, std::string* error = nullptr) {
        if (record.label.empty() || (record.visibility != 1U && record.visibility != 2U)) {
            if (error) *error = "invalid hiding commitment metadata";
            return false;
        }
        const CpuG1Point h = cpu_g1_from_native(record.blinding_generator);
        if (h.is_zero()) { if (error) *error = "zero hiding generator"; return false; }
        const FrWire challenge = hiding_challenge(record.label, record.visibility,
                                                  cpu_g1_from_native(record.value_commitment), h,
                                                  cpu_g1_from_native(record.commitment),
                                                  cpu_g1_from_native(record.schnorr_R));
        const CpuG1Point lhs = cpu_g1_mul(h, fr_wire(record.schnorr_response));
        const CpuG1Point delta = cpu_g1_add(cpu_g1_from_native(record.commitment),
                                            cpu_g1_neg(cpu_g1_from_native(record.value_commitment)));
        const CpuG1Point rhs = cpu_g1_add(cpu_g1_from_native(record.schnorr_R), cpu_g1_mul(delta, challenge));
        if (!cpu_g1_equal(lhs, rhs)) {
            if (error) *error = "hiding commitment Schnorr equation failed";
            return false;
        }
        return true;
    }
};

inline bool validate_hiding_commitments(const std::vector<CommitmentRecord>& records,
                                        std::string* error = nullptr) {
    std::vector<std::string> labels;
    bool have_model = false, have_dynamic = false;
    for (size_t i = 0; i < records.size(); ++i) {
        if (std::find(labels.begin(), labels.end(), records[i].label) != labels.end()) {
            if (error) *error = "duplicate commitment label";
            return false;
        }
        labels.push_back(records[i].label);
        if (!CpuHidingCommitment::verify_record(records[i], error)) return false;
        have_model = have_model || records[i].visibility == 1U;
        have_dynamic = have_dynamic || records[i].visibility == 2U;
    }
    (void)have_model; (void)have_dynamic;
    return true;
}

} // namespace zkdl

#endif
