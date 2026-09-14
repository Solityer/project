#include "native_masked_sumcheck.hpp"

#include <mcl/bn256.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace mcl::bn;

namespace {

const char kMaskMagic[] = "zkgpt-masked-sumcheck-v1";

void set_reason(std::string *reason, const std::string &message) {
    if (reason != nullptr) *reason = message;
}

bool has_prefix(const std::string &value, const std::string &prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool valid_scope_label(const std::string &label) {
    if (label.empty() || label.size() > 512 ||
        has_prefix(label, "challenge/") || label.front() == '/' ||
        label.back() == '/' || label.find("//") != std::string::npos ||
        label.find('|') != std::string::npos)
        return false;
    for (unsigned char c : label)
        if (std::iscntrl(c) || std::isspace(c)) return false;
    return true;
}

std::string field_hex(const F &value) {
    const std::string encoded = value.getStr(16);
    if (encoded.empty() || encoded.size() > 128)
        throw std::runtime_error("masked Sumcheck field encoding is invalid");
    return encoded;
}

bool decode_field(const std::string &encoded, F &value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
        return value.getStr(16) == encoded;
    } catch (...) {
        return false;
    }
}

bool decode_fields(const NativeProofMessage *message, std::size_t expected,
                   std::vector<F> &values, std::string *reason) {
    values.clear();
    if (message == nullptr || message->kind != NativeProofMessageKind::FieldVector)
        return false;
    const std::string &payload = message->payload;
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t &value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(
                         static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count != expected) {
        set_reason(reason, "masked Sumcheck field-vector arity mismatch");
        return false;
    }
    values.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > 128 ||
            size > payload.size() - offset) {
            set_reason(reason, "malformed masked Sumcheck field vector");
            return false;
        }
        F value;
        if (!decode_field(payload.substr(offset, static_cast<std::size_t>(size)), value)) {
            set_reason(reason, "non-canonical masked Sumcheck field vector");
            return false;
        }
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    if (offset != payload.size()) {
        set_reason(reason, "trailing bytes in masked Sumcheck field vector");
        return false;
    }
    return true;
}

bool decode_point(const std::string &encoded, G1 &point) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        point.deserializeHexStr(encoded);
        return point.serializeToHexStr() == encoded && point.isValid() &&
               !point.isZero() && point.isValidOrder();
    } catch (...) {
        return false;
    }
}

bool valid_point(const G1 &point) {
    return point.isValid() && !point.isZero() && point.isValidOrder();
}

bool parse_decimal(const std::string &encoded, std::size_t &value) {
    if (encoded.empty() || (encoded.size() > 1 && encoded[0] == '0')) return false;
    std::uint64_t parsed = 0;
    for (unsigned char c : encoded) {
        if (c < '0' || c > '9') return false;
        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
        if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            return false;
        parsed = parsed * 10 + digit;
    }
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

std::vector<std::string> split_payload(const std::string &payload) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = payload.find('|', begin);
        parts.push_back(payload.substr(begin,
                                      end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parts;
}

struct ParsedMaskCommitment {
    std::size_t degree = 0;
    std::size_t round = 0;
    G1 commitment;
    G1 proof_commitment;
};

bool parse_mask_commitment(const NativeProofMessage *message,
                           std::size_t degree, std::size_t round,
                           ParsedMaskCommitment &parsed,
                           std::string *reason) {
    if (message == nullptr || message->kind != NativeProofMessageKind::Data) {
        set_reason(reason, "masked Sumcheck mask commitment has wrong kind");
        return false;
    }
    const std::vector<std::string> parts = split_payload(message->payload);
    if (parts.size() != 5 || parts[0] != kMaskMagic ||
        !parse_decimal(parts[1], parsed.degree) ||
        !parse_decimal(parts[2], parsed.round) ||
        parsed.degree != degree || parsed.round != round ||
        !decode_point(parts[3], parsed.commitment) ||
        !decode_point(parts[4], parsed.proof_commitment)) {
        set_reason(reason, "malformed masked Sumcheck mask commitment");
        return false;
    }
    return true;
}

G1 generator_for(const NativeProofSession &session, const std::string &label,
                 std::size_t round, std::size_t index) {
    G1 generator;
    hashAndMapToG1(generator,
                   "zkgpt/masked-sumcheck/v1/" + session.statement_digest() +
                       "/" + label + "/" + std::to_string(round) + "/" +
                       std::to_string(index));
    return generator;
}

G1 blinding_generator_for(const NativeProofSession &session,
                          const std::string &label, std::size_t round) {
    G1 generator;
    hashAndMapToG1(generator,
                   "zkgpt/masked-sumcheck/v1/" + session.statement_digest() +
                       "/" + label + "/" + std::to_string(round) + "/blind");
    return generator;
}

G1 commit_vector(const NativeProofSession &session, const std::string &label,
                 std::size_t round, const std::vector<F> &values,
                 const F &blind) {
    G1 commitment;
    commitment.clear();
    for (std::size_t i = 0; i < values.size(); ++i)
        commitment += generator_for(session, label, round, i) * values[i];
    commitment += blinding_generator_for(session, label, round) * blind;
    return commitment;
}

F random_field() {
    F value;
    value.setByCSPRNG();
    return value;
}

F random_nonzero_field() {
    F value;
    do {
        value = random_field();
    } while (value.isZero());
    return value;
}

F interpolate_at(const std::vector<F> &values, const F &point) {
    if (values.empty() || values.size() > 4)
        throw std::invalid_argument("masked Sumcheck degree is outside 1..3");
    F value = F_ZERO;
    for (std::size_t i = 0; i < values.size(); ++i) {
        F basis = F_ONE;
        const F coordinate_i = F(static_cast<int>(i));
        for (std::size_t j = 0; j < values.size(); ++j) {
            if (i == j) continue;
            const F coordinate_j = F(static_cast<int>(j));
            basis *= (point - coordinate_j) / (coordinate_i - coordinate_j);
        }
        value += values[i] * basis;
    }
    return value;
}

F mle_eval(const std::vector<F> &values, const std::vector<F> &point) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0 ||
        values.size() != (std::size_t(1) << point.size()))
        throw std::invalid_argument("masked Sumcheck public MLE dimensions are invalid");
    std::vector<F> current = values;
    for (const F &challenge : point) {
        const std::size_t half = current.size() / 2;
        std::vector<F> next(half);
        for (std::size_t index = 0; index < half; ++index)
            next[index] = current[2 * index] +
                          challenge * (current[2 * index + 1] - current[2 * index]);
        current.swap(next);
    }
    return current.front();
}

bool terminal_arity(const NativeMaskedTerminalSpec &spec, std::size_t &arity,
                    std::string *reason) {
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar: arity = 1; return true;
    case NativeMaskedTerminalKind::Product2: arity = 2; return true;
    case NativeMaskedTerminalKind::Product3: arity = 3; return true;
    case NativeMaskedTerminalKind::Claims: arity = 2; return true;
    case NativeMaskedTerminalKind::GkrPredicate:
        if (spec.gkr_coefficients.size() != 5) {
            set_reason(reason, "masked GKR terminal predicate needs five coefficients");
            return false;
        }
        arity = 4;
        return true;
    case NativeMaskedTerminalKind::Product2Public:
        if (spec.public_vector.empty() ||
            (spec.public_vector.size() & (spec.public_vector.size() - 1)) != 0) {
            set_reason(reason, "masked public product relation needs a power-of-two vector");
            return false;
        }
        arity = 1;
        return true;
    }
    set_reason(reason, "unknown masked Sumcheck terminal relation");
    return false;
}

F gkr_predicate(const std::vector<F> &coefficients,
                const std::vector<F> &values) {
    if (coefficients.size() != 5 || values.size() != 4)
        throw std::invalid_argument("masked GKR predicate dimensions are invalid");
    const F &bin0 = coefficients[0];
    const F &bin1 = coefficients[1];
    const F &bin2 = coefficients[2];
    const F &uni0 = coefficients[3];
    const F &uni1 = coefficients[4];
    const F &u0 = values[0];
    const F &u1 = values[1];
    const F &v0 = values[2];
    const F &v1 = values[3];
    return bin0 * u0 * v0 + bin1 * u1 * v1 + bin2 * u1 * v0 +
           uni0 * u0 + uni1 * u1;
}

// Produce a fresh terminal vector on the public GKR predicate surface.  The
// original terminal values are intentionally not used: the round masks have
// already randomized the final claim, and this step samples a hiding point
// in the same public quadratic relation.
bool sample_gkr_terminal(const std::vector<F> &coefficients, const F &target,
                         std::vector<F> &values) {
    const F &bin0 = coefficients[0];
    const F &bin1 = coefficients[1];
    const F &bin2 = coefficients[2];
    const F &uni0 = coefficients[3];
    const F &uni1 = coefficients[4];
    for (unsigned attempt = 0; attempt < 96; ++attempt) {
        F u0 = random_field();
        F u1 = random_field();
        F v0 = random_field();
        F v1 = random_field();

        F denominator = bin1 * u1;
        if (!denominator.isZero()) {
            const F base = bin0 * u0 * v0 + bin2 * u1 * v0 +
                           uni0 * u0 + uni1 * u1;
            v1 = (target - base) / denominator;
            values = {u0, u1, v0, v1};
            return gkr_predicate(coefficients, values) == target;
        }

        denominator = bin0 * v0 + uni0;
        if (!denominator.isZero()) {
            const F base = bin1 * u1 * v1 + bin2 * u1 * v0 + uni1 * u1;
            u0 = (target - base) / denominator;
            values = {u0, u1, v0, v1};
            return gkr_predicate(coefficients, values) == target;
        }

        denominator = bin0 * u0 + bin2 * u1;
        if (!denominator.isZero()) {
            const F base = bin1 * u1 * v1 + uni0 * u0 + uni1 * u1;
            v0 = (target - base) / denominator;
            values = {u0, u1, v0, v1};
            return gkr_predicate(coefficients, values) == target;
        }
    }
    // A constant-zero predicate still has a full random terminal surface.
    if (bin0.isZero() && bin1.isZero() && bin2.isZero() &&
        uni0.isZero() && uni1.isZero() && target.isZero()) {
        values = {random_field(), random_field(), random_field(), random_field()};
        return true;
    }
    return false;
}

std::string mask_commitment_label(const std::string &label, std::size_t round) {
    return label + "/mask/" + std::to_string(round) + "/commitment";
}

std::string mask_knowledge_label(const std::string &label, std::size_t round) {
    return label + "/mask/" + std::to_string(round) + "/knowledge";
}

std::string mask_response_label(const std::string &label, std::size_t round) {
    return label + "/mask/" + std::to_string(round) + "/response";
}

struct ScopeMessage {
    const NativeProofMessage *message = nullptr;
    std::size_t index = 0;
};

bool collect_scope(const NativeProofSession &session, const std::string &label,
                   std::vector<ScopeMessage> &messages, std::string *reason) {
    const std::string prefix = label + "/";
    const std::string challenge_prefix = "challenge/" + prefix;
    for (std::size_t i = 0; i < session.messages().size(); ++i) {
        const NativeProofMessage &message = session.messages()[i];
        if (has_prefix(message.label, prefix) ||
            has_prefix(message.label, challenge_prefix))
            messages.push_back(ScopeMessage{&message, i});
    }
    if (messages.empty()) {
        set_reason(reason, "masked Sumcheck scope is absent");
        return false;
    }
    return true;
}

bool expect_scope_message(const std::vector<ScopeMessage> &messages,
                          std::size_t &cursor, NativeProofMessageKind kind,
                          const std::string &label, const NativeProofMessage *&out,
                          std::string *reason) {
    if (cursor >= messages.size() || messages[cursor].message->kind != kind ||
        messages[cursor].message->label != label) {
        set_reason(reason, "masked Sumcheck cursor expected " + label);
        return false;
    }
    out = messages[cursor++].message;
    return true;
}

bool validate_terminal_spec(const NativeMaskedTerminalSpec &spec,
                            std::size_t degree, std::size_t &arity,
                            std::string *reason) {
    if (degree < 1 || degree > 3) {
        set_reason(reason, "masked Sumcheck degree must be in 1..3");
        return false;
    }
    if (!terminal_arity(spec, arity, reason)) return false;
    if (spec.kind == NativeMaskedTerminalKind::Product2 && degree != 2) {
        set_reason(reason, "masked Product2 relation requires degree two");
        return false;
    }
    if (spec.kind == NativeMaskedTerminalKind::Product3 && degree != 3) {
        set_reason(reason, "masked Product3 relation requires degree three");
        return false;
    }
    if (spec.kind == NativeMaskedTerminalKind::GkrPredicate && degree != 2) {
        set_reason(reason, "masked GKR predicate requires degree two");
        return false;
    }
    if (spec.kind == NativeMaskedTerminalKind::Product2Public && degree != 2) {
        set_reason(reason, "masked public product relation requires degree two");
        return false;
    }
    return true;
}

bool emit_mask_commitment_and_proof(
    NativeProofSession &session, const std::string &label, std::size_t degree,
    std::size_t round, const std::vector<F> &mask, const F &blind,
    std::string *reason) {
    try {
        const G1 commitment = commit_vector(session, label, round, mask, blind);
        std::vector<F> nonce(mask.size());
        for (F &value : nonce) value = random_field();
        const F nonce_blind = random_field();
        const G1 proof_commitment =
            commit_vector(session, label, round, nonce, nonce_blind);
        const std::string commitment_payload =
            std::string(kMaskMagic) + "|" + std::to_string(degree) + "|" +
            std::to_string(round) + "|" + commitment.serializeToHexStr() +
            "|" + proof_commitment.serializeToHexStr();
        session.absorb(mask_commitment_label(label, round), commitment_payload);
        const F challenge = session.challenge(mask_knowledge_label(label, round));
        std::vector<F> responses(mask.size() + 1);
        for (std::size_t i = 0; i < mask.size(); ++i)
            responses[i] = nonce[i] + challenge * mask[i];
        responses.back() = nonce_blind + challenge * blind;
        session.absorb_fields(mask_response_label(label, round), responses);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_mask_commitment_and_proof(
    const NativeProofSession &session, const std::string &label,
    std::size_t degree, std::size_t round, const NativeProofMessage *commitment_message,
    const NativeProofMessage *knowledge_message,
    const NativeProofMessage *response_message, std::set<std::string> &seen,
    std::string *reason) {
    ParsedMaskCommitment parsed;
    if (!parse_mask_commitment(commitment_message, degree, round, parsed, reason))
        return false;
    const std::string commitment_hex = parsed.commitment.serializeToHexStr();
    if (!seen.insert(commitment_hex).second) {
        set_reason(reason, "masked Sumcheck reused a mask commitment");
        return false;
    }
    if (knowledge_message == nullptr || knowledge_message->kind != NativeProofMessageKind::Challenge ||
        knowledge_message->label != "challenge/" + mask_knowledge_label(label, round)) {
        set_reason(reason, "masked Sumcheck mask knowledge challenge is malformed");
        return false;
    }
    F challenge;
    if (!decode_field(knowledge_message->payload, challenge)) {
        set_reason(reason, "masked Sumcheck mask knowledge challenge is not canonical");
        return false;
    }
    std::vector<F> responses;
    if (!decode_fields(response_message, degree + 2, responses, reason)) {
        set_reason(reason, "masked Sumcheck mask knowledge response is malformed");
        return false;
    }
    G1 lhs = commit_vector(session, label, round,
                           std::vector<F>(responses.begin(), responses.end() - 1),
                           responses.back());
    G1 rhs = parsed.proof_commitment + parsed.commitment * challenge;
    if (lhs != rhs) {
        set_reason(reason, "masked Sumcheck mask knowledge proof failed");
        return false;
    }
    return true;
}

bool verify_terminal(const NativeMaskedTerminalSpec &spec, const F &running,
                     const std::vector<F> &terminal, std::string *reason) {
    std::size_t arity = 0;
    if (!terminal_arity(spec, arity, reason) || terminal.size() != arity) return false;
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar:
        if (terminal[0] != running) {
            set_reason(reason, "masked scalar terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Product2:
        if (terminal[0] * terminal[1] != running) {
            set_reason(reason, "masked product-two terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Product3:
        if (terminal[0] * terminal[1] * terminal[2] != running) {
            set_reason(reason, "masked product-three terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Claims:
        return true;
    case NativeMaskedTerminalKind::GkrPredicate:
        if (gkr_predicate(spec.gkr_coefficients, terminal) != running) {
            set_reason(reason, "masked GKR terminal predicate relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Product2Public:
        // The public factor is evaluated by the verifier at the same
        // transcript-derived point and is not a witness terminal.
        if (terminal.size() != 1 || spec.public_vector.empty()) {
            set_reason(reason, "masked public product terminal is malformed");
            return false;
        }
        if (terminal[0] * spec.public_evaluation != running) {
            set_reason(reason, "masked public product terminal relation failed");
            return false;
        }
        return true;
    }
    set_reason(reason, "unknown masked terminal relation");
    return false;
}

bool sample_masked_terminal(const NativeMaskedTerminalSpec &spec, const F &running,
                            std::vector<F> &terminal, std::string *reason) {
    std::size_t arity = 0;
    if (!terminal_arity(spec, arity, reason)) return false;
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar:
        terminal = {running};
        return true;
    case NativeMaskedTerminalKind::Product2: {
        const F left = random_nonzero_field();
        terminal = {left, running / left};
        return true;
    }
    case NativeMaskedTerminalKind::Product3: {
        const F left = random_nonzero_field();
        const F middle = random_nonzero_field();
        terminal = {left, middle, running / (left * middle)};
        return true;
    }
    case NativeMaskedTerminalKind::Claims:
        terminal = {random_field(), random_field()};
        return true;
    case NativeMaskedTerminalKind::Product2Public:
        if (spec.public_evaluation.isZero()) {
            if (!running.isZero()) {
                set_reason(reason, "masked public product has nonzero claim at zero factor");
                return false;
            }
            terminal = {random_field()};
        } else {
            terminal = {running / spec.public_evaluation};
        }
        return true;
    case NativeMaskedTerminalKind::GkrPredicate:
        if (!sample_gkr_terminal(spec.gkr_coefficients, running, terminal)) {
            set_reason(reason, "masked GKR terminal relation has no sampled point");
            return false;
        }
        return true;
    }
    set_reason(reason, "unknown masked terminal relation");
    return false;
}

} // namespace

bool native_masked_sumcheck_prove(
    NativeProofSession &session, const std::string &label, std::size_t degree,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const NativeMaskedTerminalSpec &terminal_spec,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (!valid_scope_label(label))
            throw std::invalid_argument("masked Sumcheck scope label is invalid");
        if (round_polys.empty() || round_polys.size() > (1ULL << 20))
            throw std::invalid_argument("masked Sumcheck round count is invalid");
        NativeMaskedTerminalSpec effective_spec = terminal_spec;
        std::size_t arity = 0;
        if (!validate_terminal_spec(effective_spec, degree, arity, reason)) return false;
        if (terminal_values.size() != arity)
            throw std::invalid_argument("masked Sumcheck terminal witness arity is invalid");
        for (const std::vector<F> &poly : round_polys)
            if (poly.size() != degree + 1)
                throw std::invalid_argument("masked Sumcheck round polynomial arity is invalid");

        F running = initial_claim;
        F unmasked_running = initial_claim;
        F mask_target = F_ZERO;
        NativeMaskedSumcheckResult local;
        local.challenges.reserve(round_polys.size());
        std::set<std::string> local_commitments;
        for (std::size_t round = 0; round < round_polys.size(); ++round) {
            std::vector<F> mask(degree + 1);
            mask[0] = random_field();
            mask[1] = mask_target - mask[0];
            for (std::size_t i = 2; i < mask.size(); ++i)
                mask[i] = random_field();
            const F blind = random_field();

            const G1 commitment = commit_vector(session, label, round, mask, blind);
            const std::string commitment_hex = commitment.serializeToHexStr();
            if (!local_commitments.insert(commitment_hex).second)
                throw std::runtime_error("masked Sumcheck mask commitment collision");
            if (!emit_mask_commitment_and_proof(session, label, degree, round,
                                                mask, blind, reason))
                return false;

            std::vector<F> masked = round_polys[round];
            for (std::size_t i = 0; i < masked.size(); ++i) masked[i] += mask[i];
            if (unmasked_running != round_polys[round][0] + round_polys[round][1])
                throw std::runtime_error("unmasked Sumcheck claim transition failed");
            if (running != masked[0] + masked[1])
                throw std::runtime_error("masked Sumcheck claim transition failed");
            session.absorb_fields(label + "/poly/" + std::to_string(round), masked);
            const F challenge = session.challenge(label + "/r/" + std::to_string(round));
            local.challenges.push_back(challenge);
            unmasked_running = interpolate_at(round_polys[round], challenge);
            running = interpolate_at(masked, challenge);
            mask_target = interpolate_at(mask, challenge);
        }

        if (effective_spec.kind == NativeMaskedTerminalKind::Product2Public)
            effective_spec.public_evaluation = mle_eval(
                effective_spec.public_vector, local.challenges);
        if (!verify_terminal(effective_spec, unmasked_running, terminal_values, reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "unmasked terminal relation failed");
        if (!sample_masked_terminal(effective_spec, running, local.terminal_values, reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "masked terminal sampling failed");
        if (!verify_terminal(effective_spec, running, local.terminal_values, reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "masked terminal relation failed");
        session.absorb_fields(label + "/terminal", local.terminal_values);
        local.final_claim = running;
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_masked_sumcheck_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t degree, std::size_t rounds, const F &initial_claim,
    const NativeMaskedTerminalSpec &terminal_spec,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    try {
        if (!valid_scope_label(label) || rounds == 0 || rounds > (1ULL << 20)) {
            set_reason(reason, "masked Sumcheck verifier dimensions are invalid");
            return false;
        }
        NativeMaskedTerminalSpec effective_spec = terminal_spec;
        std::size_t arity = 0;
        if (!validate_terminal_spec(effective_spec, degree, arity, reason)) return false;
        if (!session.valid(reason) || !session.replay(reason)) return false;

        std::vector<ScopeMessage> messages;
        if (!collect_scope(session, label, messages, reason)) return false;
        const std::size_t expected_messages = rounds * 5 + 1;
        if (messages.size() != expected_messages) {
            set_reason(reason, "masked Sumcheck scope message count mismatch");
            return false;
        }

        std::size_t cursor = 0;
        F running = initial_claim;
        NativeMaskedSumcheckResult local;
        local.challenges.reserve(rounds);
        std::set<std::string> commitments;
        for (std::size_t round = 0; round < rounds; ++round) {
            const NativeProofMessage *commitment = nullptr;
            const NativeProofMessage *knowledge = nullptr;
            const NativeProofMessage *response = nullptr;
            const NativeProofMessage *poly = nullptr;
            const NativeProofMessage *challenge = nullptr;
            if (!expect_scope_message(messages, cursor, NativeProofMessageKind::Data,
                                      mask_commitment_label(label, round), commitment,
                                      reason) ||
                !expect_scope_message(messages, cursor, NativeProofMessageKind::Challenge,
                                      "challenge/" + mask_knowledge_label(label, round),
                                      knowledge, reason) ||
                !expect_scope_message(messages, cursor, NativeProofMessageKind::FieldVector,
                                      mask_response_label(label, round), response, reason) ||
                !expect_scope_message(messages, cursor, NativeProofMessageKind::FieldVector,
                                      label + "/poly/" + std::to_string(round), poly,
                                      reason) ||
                !expect_scope_message(messages, cursor, NativeProofMessageKind::Challenge,
                                      "challenge/" + label + "/r/" + std::to_string(round),
                                      challenge, reason))
                return false;

            if (!verify_mask_commitment_and_proof(
                    session, label, degree, round, commitment, knowledge, response,
                    commitments, reason))
                return false;
            std::vector<F> masked;
            if (!decode_fields(poly, degree + 1, masked, reason)) {
                set_reason(reason, "malformed masked Sumcheck polynomial");
                return false;
            }
            if (running != masked[0] + masked[1]) {
                set_reason(reason, "masked Sumcheck claim transition failed");
                return false;
            }
            if (challenge->kind != NativeProofMessageKind::Challenge) {
                set_reason(reason, "masked Sumcheck round challenge has wrong kind");
                return false;
            }
            F point;
            if (!decode_field(challenge->payload, point)) {
                set_reason(reason, "malformed masked Sumcheck round challenge");
                return false;
            }
            running = interpolate_at(masked, point);
            local.challenges.push_back(point);
        }

        const NativeProofMessage *terminal = nullptr;
        if (!expect_scope_message(messages, cursor, NativeProofMessageKind::FieldVector,
                                  label + "/terminal", terminal, reason) ||
            cursor != messages.size()) {
            set_reason(reason, "masked Sumcheck terminal cursor is not at scope end");
            return false;
        }
        if (!decode_fields(terminal, arity, local.terminal_values, reason)) {
            set_reason(reason, "malformed masked Sumcheck terminal values");
            return false;
        }
        if (effective_spec.kind == NativeMaskedTerminalKind::Product2Public)
            effective_spec.public_evaluation = mle_eval(
                effective_spec.public_vector, local.challenges);
        if (!verify_terminal(effective_spec, running, local.terminal_values, reason))
            return false;
        local.final_claim = running;
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool native_masked_gkr_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::GkrPredicate;
    spec.gkr_coefficients = gkr_coefficients;
    return native_masked_sumcheck_prove(session, label, 2, round_polys,
                                        initial_claim, terminal_values, spec,
                                        result, reason);
}

bool native_masked_gkr_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::GkrPredicate;
    spec.gkr_coefficients = gkr_coefficients;
    return native_masked_sumcheck_verify(session, label, 2, rounds,
                                         initial_claim, spec, result, reason);
}

bool native_masked_matmul_product_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::Product2;
    return native_masked_sumcheck_prove(session, label, 2, round_polys,
                                        initial_claim, terminal_values, spec,
                                        result, reason);
}

bool native_masked_matmul_product_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::Product2;
    return native_masked_sumcheck_verify(session, label, 2, rounds,
                                         initial_claim, spec, result, reason);
}

NativeMaskedSumcheckProver::NativeMaskedSumcheckProver(
    NativeProofSession &session, const std::string &label, std::size_t degree,
    const F &initial_claim, const NativeMaskedTerminalSpec &terminal_spec,
    const F *unmasked_initial_claim)
    : session_(&session), label_(label), degree_(degree),
      terminal_spec_(terminal_spec), checkpoint_(session.checkpoint()),
      masked_running_(initial_claim),
      unmasked_running_(unmasked_initial_claim == nullptr ? initial_claim
                                                            : *unmasked_initial_claim),
      mask_target_(unmasked_initial_claim == nullptr
                       ? F_ZERO
                       : initial_claim - *unmasked_initial_claim) {
    if (!valid_scope_label(label_) || !session_->valid())
        throw std::invalid_argument("masked streaming Sumcheck identity is invalid");
    std::size_t arity = 0;
    std::string reason;
    if (!validate_terminal_spec(terminal_spec_, degree_, arity, &reason))
        throw std::invalid_argument(reason);
}

bool NativeMaskedSumcheckProver::append_round(
    const std::vector<F> &unmasked_poly, F *challenge, std::string *reason) {
    if (finished_) {
        set_reason(reason, "masked streaming Sumcheck is already finished");
        return false;
    }
    try {
        if (unmasked_poly.size() != degree_ + 1)
            throw std::invalid_argument("masked streaming round polynomial arity is invalid");
        if (rounds_ >= (1ULL << 20))
            throw std::invalid_argument("masked streaming Sumcheck has too many rounds");

        std::vector<F> mask(degree_ + 1);
        mask[0] = random_field();
        mask[1] = mask_target_ - mask[0];
        for (std::size_t i = 2; i < mask.size(); ++i)
            mask[i] = random_field();
        const F blind = random_field();
        const G1 commitment = commit_vector(*session_, label_, rounds_, mask, blind);
        if (!commitments_.insert(commitment.serializeToHexStr()).second)
            throw std::runtime_error("masked streaming Sumcheck mask commitment collision");
        if (!emit_mask_commitment_and_proof(*session_, label_, degree_, rounds_,
                                            mask, blind, reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "masked Sumcheck mask emission failed");

        std::vector<F> masked = unmasked_poly;
        for (std::size_t i = 0; i < masked.size(); ++i) masked[i] += mask[i];
        if (unmasked_running_ != unmasked_poly[0] + unmasked_poly[1])
            throw std::runtime_error("unmasked Sumcheck claim transition failed");
        if (masked_running_ != masked[0] + masked[1])
            throw std::runtime_error("masked Sumcheck claim transition failed");
        session_->absorb_fields(label_ + "/poly/" + std::to_string(rounds_), masked);
        const F point = session_->challenge(label_ + "/r/" + std::to_string(rounds_));
        result_.challenges.push_back(point);
        unmasked_running_ = interpolate_at(unmasked_poly, point);
        masked_running_ = interpolate_at(masked, point);
        mask_target_ = interpolate_at(mask, point);
        ++rounds_;
        if (challenge != nullptr) *challenge = point;
        return true;
    } catch (const std::exception &error) {
        session_->rollback(checkpoint_);
        finished_ = true;
        set_reason(reason, error.what());
        return false;
    }
}

bool NativeMaskedSumcheckProver::finish(
    const std::vector<F> &unmasked_terminal, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    if (finished_) {
        set_reason(reason, "masked streaming Sumcheck is already finished");
        return false;
    }
    try {
        if (rounds_ == 0)
            throw std::invalid_argument("masked streaming Sumcheck has no rounds");
        std::size_t arity = 0;
        if (!terminal_arity(terminal_spec_, arity, reason))
            throw std::invalid_argument(reason != nullptr ? *reason :
                                        "masked terminal relation is invalid");
        if (unmasked_terminal.size() != arity)
            throw std::invalid_argument("masked streaming terminal arity is invalid");
        NativeMaskedTerminalSpec effective_spec = terminal_spec_;
        if (effective_spec.kind == NativeMaskedTerminalKind::Product2Public)
            effective_spec.public_evaluation = mle_eval(
                effective_spec.public_vector, result_.challenges);
        if (!verify_terminal(effective_spec, unmasked_running_, unmasked_terminal,
                             reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "unmasked terminal relation failed");
        if (!sample_masked_terminal(effective_spec, masked_running_,
                                    result_.terminal_values, reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "masked terminal sampling failed");
        if (!verify_terminal(effective_spec, masked_running_,
                             result_.terminal_values, reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "masked terminal relation failed");
        session_->absorb_fields(label_ + "/terminal", result_.terminal_values);
        result_.final_claim = masked_running_;
        finished_ = true;
        if (result != nullptr) *result = result_;
        return true;
    } catch (const std::exception &error) {
        session_->rollback(checkpoint_);
        finished_ = true;
        set_reason(reason, error.what());
        return false;
    }
}
