#ifndef PROOF_CUH
#define PROOF_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"

#include <vector>
#include <random>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>

/*
 * The original zkDL example kept Sumcheck values in an ephemeral vector and
 * discarded both the round messages and the opening messages.  The GAT
 * adaptation needs an actual verifier-facing object, so the small wire
 * protocol below deliberately lives next to the native Sumcheck routines.
 * It is only a serialization/verifier layer around the existing native
 * primitives; it is not a second proving system.
 */
namespace zkdl {

using byte = std::uint8_t;
using FrWire = std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS>;

static const FrWire kScalarModulus = {{
    1U, 4294967295U, 4294859774U, 1404937218U,
    161601541U, 859428872U, 698187080U, 1944954707U
}};

inline FrWire fr_wire(const Fr_t& value) {
    FrWire out;
    for (uint i = 0; i < blstrs__scalar__Scalar_LIMBS; ++i) out[i] = value.val[i];
    return out;
}

inline Fr_t fr_value(const FrWire& value) {
    Fr_t out{};
    for (uint i = 0; i < blstrs__scalar__Scalar_LIMBS; ++i) out.val[i] = value[i];
    return out;
}

inline int fr_compare(const FrWire& a, const FrWire& b) {
    for (int i = blstrs__scalar__Scalar_LIMBS - 1; i >= 0; --i) {
        if (a[static_cast<size_t>(i)] < b[static_cast<size_t>(i)]) return -1;
        if (a[static_cast<size_t>(i)] > b[static_cast<size_t>(i)]) return 1;
    }
    return 0;
}

inline bool fr_canonical(const FrWire& a) { return fr_compare(a, kScalarModulus) < 0; }

inline FrWire fr_sub_raw(const FrWire& a, const FrWire& b) {
    FrWire out{};
    std::uint64_t borrow = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t ai = a[i];
        const std::uint64_t bi = static_cast<std::uint64_t>(b[i]) + borrow;
        out[i] = static_cast<std::uint32_t>(ai - bi);
        borrow = ai < bi ? 1U : 0U;
    }
    return out;
}

inline FrWire fr_add(const FrWire& a, const FrWire& b) {
    FrWire out{};
    std::uint64_t carry = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32;
    }
    if (carry != 0 || fr_compare(out, kScalarModulus) >= 0) out = fr_sub_raw(out, kScalarModulus);
    return out;
}

inline FrWire fr_sub(const FrWire& a, const FrWire& b) {
    if (fr_compare(a, b) >= 0) return fr_sub_raw(a, b);
    FrWire delta = fr_sub_raw(b, a);
    return fr_sub_raw(kScalarModulus, delta);
}

inline FrWire fr_neg(const FrWire& a) {
    FrWire zero{};
    return fr_canonical(a) && fr_compare(a, zero) != 0 ? fr_sub_raw(kScalarModulus, a) : zero;
}

inline FrWire fr_reduce(const std::array<std::uint32_t, 2 * blstrs__scalar__Scalar_LIMBS>& product) {
    // Keep one extra limb while shifting.  A 256-bit remainder can overflow
    // its low limbs during the bit-by-bit reduction even though the reduced
    // value is below p; subtracting p from the truncated low limbs loses the
    // 2^256 carry and produces an invalid field element.
    std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS + 1> remainder{};
    std::array<std::uint32_t, blstrs__scalar__Scalar_LIMBS + 1> modulus{};
    for (size_t i = 0; i < blstrs__scalar__Scalar_LIMBS; ++i) modulus[i] = kScalarModulus[i];
    for (int bit = static_cast<int>(product.size() * 32) - 1; bit >= 0; --bit) {
        std::uint64_t carry = (product[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U;
        for (size_t i = 0; i < remainder.size(); ++i) {
            const std::uint64_t shifted = static_cast<std::uint64_t>(remainder[i]) * 2U + carry;
            remainder[i] = static_cast<std::uint32_t>(shifted);
            carry = shifted >> 32;
        }
        int comparison = remainder.back() == 0 ? 0 : 1;
        if (comparison == 0) {
            for (int i = blstrs__scalar__Scalar_LIMBS - 1; i >= 0; --i) {
                if (remainder[static_cast<size_t>(i)] != modulus[static_cast<size_t>(i)]) {
                    comparison = remainder[static_cast<size_t>(i)] > modulus[static_cast<size_t>(i)] ? 1 : -1;
                    break;
                }
            }
        }
        const bool ge_modulus = comparison >= 0;
        if (ge_modulus) {
            std::uint64_t borrow = 0;
            for (size_t i = 0; i < remainder.size(); ++i) {
                const std::uint64_t subtrahend = static_cast<std::uint64_t>(modulus[i]) + borrow;
                const std::uint64_t value = remainder[i];
                remainder[i] = static_cast<std::uint32_t>(value - subtrahend);
                borrow = value < subtrahend ? 1U : 0U;
            }
        }
    }
    FrWire out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = remainder[i];
    return out;
}

inline FrWire fr_mul(const FrWire& a, const FrWire& b) {
    std::array<std::uint32_t, 2 * blstrs__scalar__Scalar_LIMBS> product{};
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
    return fr_reduce(product);
}

inline FrWire fr_pow(FrWire base, std::uint32_t exponent_bit_count,
                     const FrWire& exponent) {
    FrWire result{};
    result[0] = 1U;
    for (int bit = static_cast<int>(exponent_bit_count) - 1; bit >= 0; --bit) {
        result = fr_mul(result, result);
        if ((exponent[static_cast<size_t>(bit / 32)] >> (bit % 32)) & 1U)
            result = fr_mul(result, base);
    }
    return result;
}

inline FrWire fr_from_u32(std::uint32_t value) {
    FrWire out{};
    out[0] = value;
    return out;
}

inline FrWire fr_modulus_minus_two() {
    FrWire out = kScalarModulus;
    std::uint64_t borrow = 2U;
    for (size_t i = 0; i < out.size() && borrow != 0; ++i) {
        const std::uint64_t value = out[i];
        out[i] = static_cast<std::uint32_t>(value - borrow);
        borrow = value < borrow ? 1U : 0U;
    }
    return out;
}

inline void append_u32(std::vector<byte>& out, std::uint32_t value) {
    out.push_back(static_cast<byte>(value));
    out.push_back(static_cast<byte>(value >> 8));
    out.push_back(static_cast<byte>(value >> 16));
    out.push_back(static_cast<byte>(value >> 24));
}

inline void append_u64(std::vector<byte>& out, std::uint64_t value) {
    append_u32(out, static_cast<std::uint32_t>(value));
    append_u32(out, static_cast<std::uint32_t>(value >> 32));
}

inline bool read_u32(const std::vector<byte>& in, size_t& offset, std::uint32_t& value) {
    if (offset > in.size() || in.size() - offset < 4) return false;
    value = static_cast<std::uint32_t>(in[offset]) |
            (static_cast<std::uint32_t>(in[offset + 1]) << 8) |
            (static_cast<std::uint32_t>(in[offset + 2]) << 16) |
            (static_cast<std::uint32_t>(in[offset + 3]) << 24);
    offset += 4;
    return true;
}

inline bool read_u64(const std::vector<byte>& in, size_t& offset, std::uint64_t& value) {
    std::uint32_t lo = 0, hi = 0;
    if (!read_u32(in, offset, lo) || !read_u32(in, offset, hi)) return false;
    value = static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
    return true;
}

inline void append_fr(std::vector<byte>& out, const Fr_t& value) {
    const FrWire wire = fr_wire(value);
    for (size_t i = 0; i < wire.size(); ++i) append_u32(out, wire[i]);
}

inline bool read_fr(const std::vector<byte>& in, size_t& offset, Fr_t& value,
                    bool require_canonical = true) {
    FrWire wire{};
    for (size_t i = 0; i < wire.size(); ++i) {
        if (!read_u32(in, offset, wire[i])) return false;
    }
    if (require_canonical && !fr_canonical(wire)) return false;
    value = fr_value(wire);
    return true;
}

/* A compact SHA-256 implementation keeps Fiat--Shamir independent of an
 * ambient OpenSSL installation and gives every build the same transcript. */
class Sha256 {
    std::array<std::uint32_t, 8> state_;
    std::array<byte, 64> block_;
    std::uint64_t bit_count_;
    size_t block_size_;

    static std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
        return (x >> n) | (x << (32U - n));
    }
    static std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (~x & z);
    }
    static std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    static std::uint32_t big0(std::uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static std::uint32_t big1(std::uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static std::uint32_t small0(std::uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static std::uint32_t small1(std::uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform(const byte* data) {
        static const std::uint32_t k[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
        };
        std::array<std::uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(data[4 * i]) << 24) |
                   (static_cast<std::uint32_t>(data[4 * i + 1]) << 16) |
                   (static_cast<std::uint32_t>(data[4 * i + 2]) << 8) |
                   static_cast<std::uint32_t>(data[4 * i + 3]);
        }
        for (size_t i = 16; i < 64; ++i)
            w[i] = small1(w[i - 2]) + w[i - 7] + small0(w[i - 15]) + w[i - 16];
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + big1(e) + ch(e, f, g) + k[i] + w[i];
            const std::uint32_t t2 = big0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

public:
    Sha256()
        : state_{{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                  0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U}},
          block_{}, bit_count_(0), block_size_(0) {}

    void update(const byte* data, size_t len) {
        bit_count_ += static_cast<std::uint64_t>(len) * 8U;
        while (len != 0) {
            const size_t take = std::min(len, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take; data += take; len -= take;
            if (block_size_ == block_.size()) { transform(block_.data()); block_size_ = 0; }
        }
    }
    std::array<byte, 32> final() {
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            while (block_size_ < 64) block_[block_size_++] = 0;
            transform(block_.data()); block_size_ = 0;
        }
        while (block_size_ < 56) block_[block_size_++] = 0;
        for (int i = 7; i >= 0; --i) block_[block_size_++] = static_cast<byte>(bit_count_ >> (8 * i));
        transform(block_.data());
        std::array<byte, 32> out{};
        for (size_t i = 0; i < state_.size(); ++i) {
            out[4 * i] = static_cast<byte>(state_[i] >> 24);
            out[4 * i + 1] = static_cast<byte>(state_[i] >> 16);
            out[4 * i + 2] = static_cast<byte>(state_[i] >> 8);
            out[4 * i + 3] = static_cast<byte>(state_[i]);
        }
        return out;
    }
};

inline std::array<byte, 32> sha256(const std::vector<byte>& bytes) {
    Sha256 hash;
    if (!bytes.empty()) hash.update(bytes.data(), bytes.size());
    return hash.final();
}

class Transcript {
    std::vector<byte> state_;
    std::uint64_t challenge_counter_;

    void absorb_tag(const char* tag, size_t len) {
        append_u32(state_, static_cast<std::uint32_t>(len));
        state_.insert(state_.end(), tag, tag + len);
    }

public:
    explicit Transcript(const std::string& domain = "zkdl-gat-transcript-v1")
        : state_(), challenge_counter_(0) {
        absorb_tag(domain.data(), domain.size());
    }

    void absorb(const std::string& label, const std::vector<byte>& message) {
        absorb_tag(label.data(), label.size());
        append_u64(state_, static_cast<std::uint64_t>(message.size()));
        state_.insert(state_.end(), message.begin(), message.end());
    }

    void absorb_fr(const std::string& label, const std::vector<Fr_t>& values) {
        std::vector<byte> message;
        append_u64(message, static_cast<std::uint64_t>(values.size()));
        for (size_t i = 0; i < values.size(); ++i) append_fr(message, values[i]);
        absorb(label, message);
    }

    Fr_t challenge(const std::string& label) {
        std::vector<byte> input = state_;
        absorb_tag(label.data(), label.size());
        append_u64(input, challenge_counter_++);
        input.insert(input.end(), label.begin(), label.end());
        const std::array<byte, 32> digest = sha256(input);
        FrWire wire{};
        for (size_t i = 0; i < wire.size(); ++i)
            wire[i] = static_cast<std::uint32_t>(digest[4 * i]) |
                      (static_cast<std::uint32_t>(digest[4 * i + 1]) << 8) |
                      (static_cast<std::uint32_t>(digest[4 * i + 2]) << 16) |
                      (static_cast<std::uint32_t>(digest[4 * i + 3]) << 24);
        while (!fr_canonical(wire)) wire = fr_sub_raw(wire, kScalarModulus);
        const Fr_t result = fr_value(wire);
        append_fr(state_, result);
        return result;
    }

    std::array<byte, 32> digest() const { return sha256(state_); }
    const std::vector<byte>& bytes() const { return state_; }
};

struct SumcheckRound {
    std::vector<Fr_t> evaluations; // g(0), g(1), ..., g(degree)
};

struct SumcheckProof {
    std::string label;
    std::uint32_t degree;
    std::vector<SumcheckRound> rounds;
    std::vector<Fr_t> terminal_claims;
    Fr_t final_claim;

    SumcheckProof() : label(), degree(0), rounds(), terminal_claims(), final_claim{} {}
};

inline void append_string(std::vector<byte>& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

inline bool read_string(const std::vector<byte>& in, size_t& offset, std::string& value,
                        size_t max_size = (1U << 20)) {
    std::uint64_t length = 0;
    if (!read_u64(in, offset, length) || length > max_size ||
        length > static_cast<std::uint64_t>(in.size() - offset)) return false;
    value.assign(reinterpret_cast<const char*>(in.data() + offset), static_cast<size_t>(length));
    offset += static_cast<size_t>(length);
    return true;
}

inline void serialize_sumcheck(const SumcheckProof& proof, std::vector<byte>& out) {
    append_string(out, proof.label);
    append_u32(out, proof.degree);
    append_u64(out, static_cast<std::uint64_t>(proof.rounds.size()));
    for (size_t i = 0; i < proof.rounds.size(); ++i) {
        append_u64(out, static_cast<std::uint64_t>(proof.rounds[i].evaluations.size()));
        for (size_t j = 0; j < proof.rounds[i].evaluations.size(); ++j)
            append_fr(out, proof.rounds[i].evaluations[j]);
    }
    append_u64(out, static_cast<std::uint64_t>(proof.terminal_claims.size()));
    for (size_t i = 0; i < proof.terminal_claims.size(); ++i) append_fr(out, proof.terminal_claims[i]);
    append_fr(out, proof.final_claim);
}

inline bool deserialize_sumcheck(const std::vector<byte>& in, size_t& offset,
                                 SumcheckProof& proof,
                                 size_t max_rounds = (1U << 20)) {
    std::uint32_t degree = 0;
    std::uint64_t rounds = 0;
    SumcheckProof parsed;
    if (!read_string(in, offset, parsed.label) || !read_u32(in, offset, degree) ||
        !read_u64(in, offset, rounds) || degree == 0 || degree > 16 || rounds > max_rounds)
        return false;
    parsed.degree = degree;
    parsed.rounds.reserve(static_cast<size_t>(rounds));
    for (std::uint64_t i = 0; i < rounds; ++i) {
        std::uint64_t values = 0;
        SumcheckRound round;
        if (!read_u64(in, offset, values) || values != static_cast<std::uint64_t>(degree + 1)) return false;
        round.evaluations.resize(static_cast<size_t>(values));
        for (size_t j = 0; j < round.evaluations.size(); ++j)
            if (!read_fr(in, offset, round.evaluations[j])) return false;
        parsed.rounds.push_back(round);
    }
    std::uint64_t terminals = 0;
    if (!read_u64(in, offset, terminals) || terminals > 64U) return false;
    parsed.terminal_claims.resize(static_cast<size_t>(terminals));
    for (size_t i = 0; i < parsed.terminal_claims.size(); ++i)
        if (!read_fr(in, offset, parsed.terminal_claims[i])) return false;
    if (!read_fr(in, offset, parsed.final_claim)) return false;
    proof = parsed;
    return true;
}

inline void append_g1(std::vector<byte>& out, const G1Jacobian_t& point) {
    for (uint coordinate = 0; coordinate < 3; ++coordinate)
        for (uint limb = 0; limb < blstrs__fp__Fp_LIMBS; ++limb) {
            const uint value = coordinate == 0 ? point.x.val[limb] :
                               (coordinate == 1 ? point.y.val[limb] : point.z.val[limb]);
            append_u32(out, value);
        }
}

inline bool read_g1(const std::vector<byte>& in, size_t& offset, G1Jacobian_t& point) {
    for (uint coordinate = 0; coordinate < 3; ++coordinate)
        for (uint limb = 0; limb < blstrs__fp__Fp_LIMBS; ++limb) {
            uint value = 0;
            if (!read_u32(in, offset, value)) return false;
            if (coordinate == 0) point.x.val[limb] = value;
            else if (coordinate == 1) point.y.val[limb] = value;
            else point.z.val[limb] = value;
        }
    return true;
}

/* One opening record is the complete verifier-facing message produced by
 * Commitment::open_with_proof.  The final generator and three points per
 * folding round are retained; no prover-only witness is serialized. */
struct EvaluationOpeningProof {
    std::uint32_t commitment_index;
    std::uint32_t generator_size;
    std::vector<Fr_t> inner_point;
    std::vector<Fr_t> outer_point;
    Fr_t value;
    G1Jacobian_t commitment_eval;
    std::vector<G1Jacobian_t> rounds; // P,A,B for every fold, then final generator

    EvaluationOpeningProof()
        : commitment_index(std::numeric_limits<std::uint32_t>::max()), generator_size(0),
          inner_point(), outer_point(), value{}, commitment_eval{}, rounds() {}
};

inline void serialize_opening(const EvaluationOpeningProof& opening, std::vector<byte>& out) {
    append_u32(out, opening.commitment_index);
    append_u32(out, opening.generator_size);
    append_u64(out, static_cast<std::uint64_t>(opening.inner_point.size()));
    for (size_t i = 0; i < opening.inner_point.size(); ++i) append_fr(out, opening.inner_point[i]);
    append_u64(out, static_cast<std::uint64_t>(opening.outer_point.size()));
    for (size_t i = 0; i < opening.outer_point.size(); ++i) append_fr(out, opening.outer_point[i]);
    append_fr(out, opening.value);
    append_g1(out, opening.commitment_eval);
    append_u64(out, static_cast<std::uint64_t>(opening.rounds.size()));
    for (size_t i = 0; i < opening.rounds.size(); ++i) append_g1(out, opening.rounds[i]);
}

inline bool deserialize_opening(const std::vector<byte>& in, size_t& offset,
                                EvaluationOpeningProof& opening,
                                size_t max_point = 4096, size_t max_round_points = (1U << 20)) {
    EvaluationOpeningProof parsed;
    std::uint64_t points = 0;
    if (!read_u32(in, offset, parsed.commitment_index) ||
        !read_u32(in, offset, parsed.generator_size) ||
        !read_u64(in, offset, points) || points > max_point) return false;
    parsed.inner_point.resize(static_cast<size_t>(points));
    for (size_t i = 0; i < parsed.inner_point.size(); ++i)
        if (!read_fr(in, offset, parsed.inner_point[i])) return false;
    if (!read_u64(in, offset, points) || points > max_point) return false;
    parsed.outer_point.resize(static_cast<size_t>(points));
    for (size_t i = 0; i < parsed.outer_point.size(); ++i)
        if (!read_fr(in, offset, parsed.outer_point[i])) return false;
    if (!read_fr(in, offset, parsed.value) || !read_g1(in, offset, parsed.commitment_eval)) return false;
    if (!read_u64(in, offset, points) || points == 0 || points > max_round_points) return false;
    parsed.rounds.resize(static_cast<size_t>(points));
    for (size_t i = 0; i < parsed.rounds.size(); ++i)
        if (!read_g1(in, offset, parsed.rounds[i])) return false;
    opening = parsed;
    return true;
}

inline bool opening_shape_valid(const EvaluationOpeningProof& opening, std::string* error = nullptr) {
    if (opening.rounds.empty() || (opening.rounds.size() - 1U) % 3U != 0U) {
        if (error) *error = "opening fold message has invalid arity";
        return false;
    }
    if (opening.inner_point.size() != (opening.rounds.size() - 1U) / 3U) {
        if (error) *error = "opening point/fold round mismatch";
        return false;
    }
    if (opening.generator_size == 0 || opening.inner_point.size() >= 32 ||
        opening.generator_size < (1U << opening.inner_point.size())) {
        if (error) *error = "opening generator shape is not compatible with point";
        return false;
    }
    for (size_t i = 0; i < opening.inner_point.size(); ++i)
        if (!fr_canonical(fr_wire(opening.inner_point[i]))) {
            if (error) *error = "opening point is not canonical";
            return false;
        }
    for (size_t i = 0; i < opening.outer_point.size(); ++i)
        if (!fr_canonical(fr_wire(opening.outer_point[i]))) {
            if (error) *error = "opening outer point is not canonical";
            return false;
        }
    return true;
}

struct CommitmentRecord {
    std::string label;
    // 1 = registered model, 2 = per-inference dynamic tensor.
    std::uint32_t visibility;
    std::vector<G1Jacobian_t> points;
    // A hiding record separates the value commitment used by the native
    // opening proof from the externally advertised Pedersen commitment.
    // `commitment = value_commitment + blinding_generator * r`; r is never
    // serialized.  The Schnorr fields below prove knowledge of r.
    G1Jacobian_t value_commitment;
    G1Jacobian_t blinding_generator;
    G1Jacobian_t commitment;
    G1Jacobian_t schnorr_R;
    Fr_t schnorr_response;

    CommitmentRecord()
        : label(), visibility(0), points(), value_commitment{}, blinding_generator{},
          commitment{}, schnorr_R{}, schnorr_response{} {}
};

/* A binding block is deliberately generic: GAT adapter code supplies the
 * operation label and terminal claims while native zkMatMul/zkProd fill the
 * same Sumcheck wire format. */
struct BindingBlock {
    std::string label;
    std::uint32_t operation_kind;
    std::vector<Fr_t> input_claims;
    std::vector<Fr_t> output_claims;
    SumcheckProof sumcheck;

    BindingBlock() : label(), operation_kind(0), input_claims(), output_claims(), sumcheck() {}
};

enum : std::uint32_t {
    kOperationMatMul = 1U,
    kOperationProduct = 2U,
    kOperationOutputAverage = 3U
};

inline bool verify_sumcheck(const SumcheckProof& proof, const Fr_t& initial_claim,
                            Transcript& transcript, std::vector<Fr_t>* challenges,
                            std::string* error);

/* The operation-specific polynomial is still supplied by the native
 * prover.  This common check makes the output claim explicit and prevents a
 * caller from treating a Sumcheck terminal value as an unbound intermediate.
 * The caller supplies the independently reconstructed initial claim. */
inline bool verify_output_binding(const BindingBlock& block, const Fr_t& initial_claim,
                                  const Fr_t& expected_output, Transcript& transcript,
                                  std::string* error = nullptr) {
    if (block.output_claims.size() != 1U) {
        if (error) *error = "operation does not carry exactly one output claim";
        return false;
    }
    if (!verify_sumcheck(block.sumcheck, initial_claim, transcript, nullptr, error)) return false;
    if (fr_compare(fr_wire(block.sumcheck.final_claim), fr_wire(block.output_claims[0])) != 0 ||
        fr_compare(fr_wire(block.output_claims[0]), fr_wire(expected_output)) != 0) {
        if (error) *error = "operation output claim is not bound to expected output";
        return false;
    }
    return true;
}

struct ProofEnvelope {
    static const std::uint32_t kVersion = 1;
    std::uint32_t version;
    std::string protocol_id;
    std::array<byte, 32> statement_digest;
    std::vector<byte> public_output;
    std::vector<CommitmentRecord> commitments;
    std::vector<BindingBlock> bindings;
    std::vector<EvaluationOpeningProof> openings;
    std::vector<SumcheckProof> sumchecks;

    ProofEnvelope()
        : version(kVersion), protocol_id("zkdl-gat-proof-v1"), statement_digest{{}},
          public_output(), commitments(), bindings(), openings(), sumchecks() {}

    std::vector<byte> serialize() const {
        std::vector<byte> out;
        out.reserve(256);
        out.push_back('Z'); out.push_back('K'); out.push_back('D'); out.push_back('L');
        append_u32(out, version);
        append_string(out, protocol_id);
        out.insert(out.end(), statement_digest.begin(), statement_digest.end());
        append_u64(out, static_cast<std::uint64_t>(public_output.size()));
        out.insert(out.end(), public_output.begin(), public_output.end());
        append_u64(out, static_cast<std::uint64_t>(commitments.size()));
        for (size_t i = 0; i < commitments.size(); ++i) {
            append_string(out, commitments[i].label);
            append_u32(out, commitments[i].visibility);
            append_u64(out, static_cast<std::uint64_t>(commitments[i].points.size()));
            for (size_t j = 0; j < commitments[i].points.size(); ++j) append_g1(out, commitments[i].points[j]);
            append_g1(out, commitments[i].value_commitment);
            append_g1(out, commitments[i].blinding_generator);
            append_g1(out, commitments[i].commitment);
            append_g1(out, commitments[i].schnorr_R);
            append_fr(out, commitments[i].schnorr_response);
        }
        append_u64(out, static_cast<std::uint64_t>(bindings.size()));
        for (size_t i = 0; i < bindings.size(); ++i) {
            append_string(out, bindings[i].label);
            append_u32(out, bindings[i].operation_kind);
            append_u64(out, static_cast<std::uint64_t>(bindings[i].input_claims.size()));
            for (size_t j = 0; j < bindings[i].input_claims.size(); ++j) append_fr(out, bindings[i].input_claims[j]);
            append_u64(out, static_cast<std::uint64_t>(bindings[i].output_claims.size()));
            for (size_t j = 0; j < bindings[i].output_claims.size(); ++j) append_fr(out, bindings[i].output_claims[j]);
            std::vector<byte> encoded;
            serialize_sumcheck(bindings[i].sumcheck, encoded);
            append_u64(out, static_cast<std::uint64_t>(encoded.size()));
            out.insert(out.end(), encoded.begin(), encoded.end());
        }
        append_u64(out, static_cast<std::uint64_t>(openings.size()));
        for (size_t i = 0; i < openings.size(); ++i) serialize_opening(openings[i], out);
        append_u64(out, static_cast<std::uint64_t>(sumchecks.size()));
        for (size_t i = 0; i < sumchecks.size(); ++i) {
            std::vector<byte> encoded;
            serialize_sumcheck(sumchecks[i], encoded);
            append_u64(out, static_cast<std::uint64_t>(encoded.size()));
            out.insert(out.end(), encoded.begin(), encoded.end());
        }
        return out;
    }

    static bool deserialize(const std::vector<byte>& in, ProofEnvelope& envelope,
                            std::string* error = nullptr) {
        const size_t max_items = (1U << 20);
        size_t offset = 0;
        ProofEnvelope parsed;
        std::uint32_t version = 0;
        std::uint64_t count = 0, bytes = 0;
        if (in.size() < 4 || in[0] != 'Z' || in[1] != 'K' || in[2] != 'D' || in[3] != 'L') {
            if (error) *error = "invalid zkDL proof magic"; return false;
        }
        offset = 4;
        if (!read_u32(in, offset, version) || version != kVersion ||
            !read_string(in, offset, parsed.protocol_id) || in.size() - offset < parsed.statement_digest.size()) {
            if (error) *error = "unsupported zkDL proof version"; return false;
        }
        parsed.version = version;
        std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
                  in.begin() + static_cast<std::ptrdiff_t>(offset + parsed.statement_digest.size()),
                  parsed.statement_digest.begin());
        offset += parsed.statement_digest.size();
        if (!read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
        parsed.public_output.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                    in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<size_t>(bytes);
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.commitments.reserve(static_cast<size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            CommitmentRecord record;
            if (!read_string(in, offset, record.label) || !read_u32(in, offset, record.visibility) ||
                !read_u64(in, offset, bytes) || bytes > max_items) return false;
            record.points.resize(static_cast<size_t>(bytes));
            for (size_t j = 0; j < record.points.size(); ++j) if (!read_g1(in, offset, record.points[j])) return false;
            if (!read_g1(in, offset, record.value_commitment) ||
                !read_g1(in, offset, record.blinding_generator) ||
                !read_g1(in, offset, record.commitment) ||
                !read_g1(in, offset, record.schnorr_R) ||
                !read_fr(in, offset, record.schnorr_response)) return false;
            parsed.commitments.push_back(record);
        }
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.bindings.reserve(static_cast<size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            BindingBlock binding;
            if (!read_string(in, offset, binding.label) || !read_u32(in, offset, binding.operation_kind) ||
                !read_u64(in, offset, bytes) || bytes > max_items) return false;
            binding.input_claims.resize(static_cast<size_t>(bytes));
            for (size_t j = 0; j < binding.input_claims.size(); ++j) if (!read_fr(in, offset, binding.input_claims[j])) return false;
            if (!read_u64(in, offset, bytes) || bytes > max_items) return false;
            binding.output_claims.resize(static_cast<size_t>(bytes));
            for (size_t j = 0; j < binding.output_claims.size(); ++j) if (!read_fr(in, offset, binding.output_claims[j])) return false;
            if (!read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
            std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                      in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
            offset += static_cast<size_t>(bytes);
            size_t inner = 0;
            if (!deserialize_sumcheck(encoded, inner, binding.sumcheck) || inner != encoded.size()) return false;
            parsed.bindings.push_back(binding);
        }
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.openings.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < parsed.openings.size(); ++i)
            if (!deserialize_opening(in, offset, parsed.openings[i])) return false;
        if (!read_u64(in, offset, count) || count > max_items) return false;
        parsed.sumchecks.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < parsed.sumchecks.size(); ++i) {
            if (!read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
            std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                      in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
            offset += static_cast<size_t>(bytes);
            size_t inner = 0;
            if (!deserialize_sumcheck(encoded, inner, parsed.sumchecks[i]) || inner != encoded.size()) return false;
        }
        if (offset != in.size()) { if (error) *error = "trailing proof bytes"; return false; }
        envelope = parsed;
        return true;
    }

    bool verify_public_binding(const std::array<byte, 32>& expected_statement,
                               const std::vector<byte>& expected_output,
                               std::string* error = nullptr) const {
        if (statement_digest != expected_statement) {
            if (error) *error = "proof statement digest mismatch";
            return false;
        }
        if (public_output != expected_output) {
            if (error) *error = "proof output is not the verifier supplied output";
            return false;
        }
        return true;
    }
};

inline FrWire interpolate_at(const std::vector<Fr_t>& values, const FrWire& x) {
    if (values.empty()) throw std::invalid_argument("sumcheck round has no evaluations");
    FrWire result{};
    for (size_t i = 0; i < values.size(); ++i) {
        FrWire term = fr_wire(values[i]);
        const FrWire xi = fr_from_u32(static_cast<std::uint32_t>(i));
        for (size_t j = 0; j < values.size(); ++j) if (i != j) {
            const FrWire xj = fr_from_u32(static_cast<std::uint32_t>(j));
            term = fr_mul(term, fr_sub(x, xj));
            term = fr_mul(term, fr_pow(fr_sub(xi, xj), 255, fr_modulus_minus_two()));
        }
        result = fr_add(result, term);
    }
    return result;
}

inline bool verify_sumcheck(const SumcheckProof& proof, const Fr_t& initial_claim,
                            Transcript& transcript, std::vector<Fr_t>* challenges = nullptr,
                            std::string* error = nullptr) {
    if (proof.degree == 0 || proof.degree > 16) {
        if (error) *error = "invalid sumcheck degree";
        return false;
    }
    FrWire claim = fr_wire(initial_claim);
    if (!fr_canonical(claim)) { if (error) *error = "non-canonical initial claim"; return false; }
    if (challenges) challenges->clear();
    for (size_t round = 0; round < proof.rounds.size(); ++round) {
        const SumcheckRound& message = proof.rounds[round];
        if (message.evaluations.size() != static_cast<size_t>(proof.degree + 1)) {
            if (error) *error = "wrong sumcheck round arity";
            return false;
        }
        FrWire sum01 = fr_add(fr_wire(message.evaluations[0]), fr_wire(message.evaluations[1]));
        if (fr_compare(sum01, claim) != 0) {
            if (error) *error = "sumcheck boolean endpoint equation failed";
            return false;
        }
        transcript.absorb_fr(proof.label + ".round", message.evaluations);
        const Fr_t challenge = transcript.challenge(proof.label + ".challenge");
        if (challenges) challenges->push_back(challenge);
        claim = interpolate_at(message.evaluations, fr_wire(challenge));
    }
    if (fr_compare(claim, fr_wire(proof.final_claim)) != 0) {
        if (error) *error = "sumcheck terminal claim mismatch";
        return false;
    }
    return true;
}

} // namespace zkdl


vector<Fr_t> random_vec(uint len)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    vector<Fr_t> out(len);
    for (uint i = 0; i < len; ++ i) out[i] = {dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt) % 1944954707};
    return out;
}

uint ceilLog2(uint num) {
    if (num == 0) return 0;
    
    // Decrease num to handle the case where num is already a power of 2
    num--;

    uint result = 0;
    
    // Keep shifting the number to the right until it becomes zero. 
    // Each shift means the number is halved, which corresponds to 
    // a division by 2 in logarithmic terms.
    while (num > 0) {
        num >>= 1;
        result++;
    }

    return result;
}

template<typename T>
std::vector<T> concatenate(const std::vector<std::vector<T>>& vecs) {
    // First, compute the total size for the result vector.
    size_t totalSize = 0;
    for (const auto& v : vecs) {
        totalSize += v.size();
    }

    // Allocate space for the result vector.
    std::vector<T> result;
    result.reserve(totalSize);

    // Append each vector's contents to the result vector.
    for (const auto& v : vecs) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}


KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    proof.push_back(out0.sum());
    proof.push_back(out1.sum());
    proof.push_back(out2.sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_ip_sc(a_new, b_new, begin + 1, end, proof);
}

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u)
{
    vector<Fr_t> proof;
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_ip_sc(a, b, u.begin(), u.end(), proof);
    return proof;
}

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 5");
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_hp_sc(a_new, b_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions 1");
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 2");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions 3");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions 4");

    Fr_hp_sc(a, b, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    Fr_t a0 = (2 * gid < in_size) ? a[2 * gid] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (2 * gid + 1 < in_size) ? a[2 * gid + 1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(a0, a0), a0);
    Fr_t diff = blstrs__scalar__Scalar_sub(a1, a0);
    out1[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_double(a0), diff), diff);
    out2[gid] = blstrs__scalar__Scalar_sqr(diff);
}

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_bin_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_bin_sc(a_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions");
    uint log_size = u.size();
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_bin_sc(a, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

#endif
