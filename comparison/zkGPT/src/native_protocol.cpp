#include "native_protocol.hpp"

#include <cstdint>
#include <sstream>

namespace {

void set_reason(std::string* reason, const std::string& message) {
    if (reason != nullptr) *reason = message;
}

const NativeProofMessage* find_message(const NativeProofSession& session, NativeProofMessageKind kind, const std::string& label, std::string* reason) {
    const NativeProofMessage* found = nullptr;
    const std::vector<std::size_t>* positions = session.message_positions(label);
    if (positions == nullptr) {
        set_reason(reason, "missing native protocol message: " + label);
        return nullptr;
    }
    for (std::size_t position : *positions) {
        if (position >= session.messages().size()) {
            set_reason(reason, "native protocol message index is out of range: " + label);
            return nullptr;
        }
        const NativeProofMessage& message = session.messages()[position];
        if (message.kind != kind || message.label != label) continue;
        if (found != nullptr) {
            set_reason(reason, "duplicate native protocol message: " + label);
            return nullptr;
        }
        found = &message;
    }
    if (found == nullptr) set_reason(reason, "missing native protocol message: " + label);
    return found;
}

bool message_index(const NativeProofSession& session, NativeProofMessageKind kind, const std::string& label, std::size_t& index, std::string* reason) {
    bool found = false;
    const std::vector<std::size_t>* positions = session.message_positions(label);
    if (positions == nullptr) {
        set_reason(reason, "missing native protocol message: " + label);
        return false;
    }
    for (std::size_t i : *positions) {
        if (i >= session.messages().size()) {
            set_reason(reason, "native protocol message index is out of range: " + label);
            return false;
        }
        const NativeProofMessage& message = session.messages()[i];
        if (message.kind != kind || message.label != label) continue;
        if (found) {
            set_reason(reason, "duplicate native protocol message: " + label);
            return false;
        }
        found = true;
        index = i;
    }
    if (!found) {
        set_reason(reason, "missing native protocol message: " + label);
        return false;
    }
    return true;
}

bool require_after(std::size_t previous, std::size_t current, const std::string& what, std::string* reason) {
    if (current > previous) return true;
    set_reason(reason, "native protocol message order failed: " + what);
    return false;
}

bool decode_field(const std::string& encoded, F& value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
        return value.getStr(16) == encoded;
    } catch (...) { return false; }
}

bool decode_fields(const std::string& payload, std::vector<F>& values) {
    values.clear();
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count > (1ULL << 20)) return false;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > 128 || size > payload.size() - offset) return false;
        F value;
        if (!decode_field(payload.substr(offset, static_cast<std::size_t>(size)), value)) return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    return offset == payload.size();
}

bool check_scope(const NativeProofSession& session, const std::string& label, const std::vector<std::string>& allowed, std::string* reason) {
    for (const NativeProofMessage& message : session.messages()) {
        bool okay = false;
        for (const std::string& expected : allowed)
            if (message.label == expected) okay = true;

        const std::string scope_prefix = label + "/";
        const std::string challenge_prefix = "challenge/" + scope_prefix;
        const bool in_scope =
            message.label.compare(0, scope_prefix.size(), scope_prefix) == 0 || message.label.compare(0, challenge_prefix.size(), challenge_prefix) == 0;
        if (!okay && in_scope) {
            set_reason(reason, "unexpected native protocol message: " + message.label);
            return false;
        }
    }
    return true;
}

bool decode_claim_vector(const NativeProofMessage* message, std::size_t expected, std::vector<F>& values, std::string* reason) {
    if (message == nullptr || !decode_fields(message->payload, values) || values.size() != expected) {
        set_reason(reason, "malformed native protocol claim vector");
        return false;
    }
    return true;
}

}

bool verify_quadratic_sumcheck_session(const NativeProofSession& session, const std::string& label, const F& initial_claim, std::size_t rounds, F* final_sum,
                                       std::vector<F>* terminal_claims, std::string* reason) {
    try {
        if (label.empty() || rounds == 0 || !session.replay(reason)) return false;
        std::vector<std::string> allowed;
        allowed.reserve(rounds * 2 + 1);
        for (std::size_t round = 0; round < rounds; ++round) {
            allowed.push_back(label + "/poly/" + std::to_string(round));
            allowed.push_back("challenge/" + label + "/r/" + std::to_string(round));
        }
        allowed.push_back(label + "/claims");
        if (!check_scope(session, label, allowed, reason)) return false;

        F claim = initial_claim;
        std::size_t previous_index = 0;
        bool have_previous = false;
        for (std::size_t round = 0; round < rounds; ++round) {
            std::size_t poly_index = 0, challenge_index = 0;
            if (!message_index(session, NativeProofMessageKind::FieldVector, label + "/poly/" + std::to_string(round), poly_index, reason) ||
                !message_index(session, NativeProofMessageKind::Challenge, "challenge/" + label + "/r/" + std::to_string(round), challenge_index, reason))
                return false;
            if ((have_previous && !require_after(previous_index, poly_index, "polynomial", reason)) ||
                !require_after(poly_index, challenge_index, "challenge", reason))
                return false;
            previous_index = challenge_index;
            have_previous = true;
            const NativeProofMessage* poly = find_message(session, NativeProofMessageKind::FieldVector, label + "/poly/" + std::to_string(round), reason);
            std::vector<F> coefficients;
            if (!decode_claim_vector(poly, 3, coefficients, reason)) return false;
            const F at_zero = coefficients[2];
            const F at_one = coefficients[0] + coefficients[1] + coefficients[2];
            if (claim != at_zero + at_one) {
                set_reason(reason, "quadratic Sumcheck claim transition failed");
                return false;
            }
            const NativeProofMessage* challenge =
                find_message(session, NativeProofMessageKind::Challenge, "challenge/" + label + "/r/" + std::to_string(round), reason);
            if (challenge == nullptr) return false;
            F r;
            if (!decode_field(challenge->payload, r)) {
                set_reason(reason, "malformed quadratic Sumcheck challenge");
                return false;
            }
            claim = coefficients[0] * r * r + coefficients[1] * r + coefficients[2];
        }

        const NativeProofMessage* terminal = find_message(session, NativeProofMessageKind::FieldVector, label + "/claims", reason);
        std::size_t terminal_index = 0;
        if (!message_index(session, NativeProofMessageKind::FieldVector, label + "/claims", terminal_index, reason) ||
            (have_previous && !require_after(previous_index, terminal_index, "terminal claim", reason)))
            return false;
        std::vector<F> claims;
        if (!decode_claim_vector(terminal, 2, claims, reason)) return false;
        if (final_sum != nullptr) *final_sum = claim;
        if (terminal_claims != nullptr) *terminal_claims = claims;
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_product_sumcheck_session(const NativeProofSession& session, const std::string& label, const F& initial_claim, std::size_t rounds, F* terminal_left,
                                     F* terminal_right, std::string* reason) {
    try {
        if (label.empty() || rounds == 0 || !session.replay(reason)) return false;
        std::vector<std::string> allowed;
        allowed.reserve(rounds * 2 + 3);
        allowed.push_back(label + "/claim");
        for (std::size_t round = 0; round < rounds; ++round) {
            allowed.push_back(label + "/poly/" + std::to_string(round));
            allowed.push_back("challenge/" + label + "/r/" + std::to_string(round));
        }
        allowed.push_back(label + "/claims");
        if (!check_scope(session, label, allowed, reason)) return false;

        const NativeProofMessage* claim_message = find_message(session, NativeProofMessageKind::Field, label + "/claim", reason);
        if (claim_message == nullptr) return false;
        F encoded_claim;
        if (!decode_field(claim_message->payload, encoded_claim) || encoded_claim != initial_claim) {
            set_reason(reason, "matrix Sumcheck public claim mismatch");
            return false;
        }

        F claim = initial_claim;
        std::size_t claim_index = 0;
        if (!message_index(session, NativeProofMessageKind::Field, label + "/claim", claim_index, reason)) return false;
        std::size_t previous_index = claim_index;
        for (std::size_t round = 0; round < rounds; ++round) {
            std::size_t poly_index = 0, challenge_index = 0;
            if (!message_index(session, NativeProofMessageKind::FieldVector, label + "/poly/" + std::to_string(round), poly_index, reason) ||
                !message_index(session, NativeProofMessageKind::Challenge, "challenge/" + label + "/r/" + std::to_string(round), challenge_index, reason) ||
                !require_after(previous_index, poly_index, "polynomial", reason) || !require_after(poly_index, challenge_index, "challenge", reason))
                return false;
            previous_index = challenge_index;
            const NativeProofMessage* poly = find_message(session, NativeProofMessageKind::FieldVector, label + "/poly/" + std::to_string(round), reason);
            std::vector<F> coefficients;
            if (!decode_claim_vector(poly, 3, coefficients, reason)) return false;
            const F at_zero = coefficients[2];
            const F at_one = coefficients[0] + coefficients[1] + coefficients[2];
            if (claim != at_zero + at_one) {
                set_reason(reason, "matrix Sumcheck claim transition failed");
                return false;
            }
            const NativeProofMessage* challenge =
                find_message(session, NativeProofMessageKind::Challenge, "challenge/" + label + "/r/" + std::to_string(round), reason);
            if (challenge == nullptr) return false;
            F r;
            if (!decode_field(challenge->payload, r)) {
                set_reason(reason, "malformed matrix Sumcheck challenge");
                return false;
            }
            claim = coefficients[0] * r * r + coefficients[1] * r + coefficients[2];
        }

        const NativeProofMessage* terminal = find_message(session, NativeProofMessageKind::FieldVector, label + "/claims", reason);
        std::size_t terminal_index = 0;
        if (!message_index(session, NativeProofMessageKind::FieldVector, label + "/claims", terminal_index, reason) ||
            !require_after(previous_index, terminal_index, "terminal claim", reason))
            return false;
        std::vector<F> claims;
        if (!decode_claim_vector(terminal, 4, claims, reason)) return false;
        if (!claims[1].isZero() || !claims[3].isZero() || claim != claims[0] * claims[2]) {
            set_reason(reason, "matrix Sumcheck terminal claim relation failed");
            return false;
        }
        if (terminal_left != nullptr) *terminal_left = claims[0];
        if (terminal_right != nullptr) *terminal_right = claims[2];
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_gkr_predicate_alpha_beta_transition(const F& sumcheck_final, const F& predicate_value, const F& claim_u1, const F& claim_v1, const F& alpha,
                                                const F& beta, const F& next_claim, std::string* reason) {
    if (sumcheck_final != predicate_value) {
        set_reason(reason, "GKR predicate transition failed");
        return false;
    }
    if (next_claim != alpha * claim_u1 + beta * claim_v1) {
        set_reason(reason, "GKR alpha/beta transition failed");
        return false;
    }
    return true;
}
