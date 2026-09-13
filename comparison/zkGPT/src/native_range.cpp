#include "native_range.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace {

void set_reason(std::string *reason, const std::string &message) {
    if (reason != nullptr) *reason = message;
}

bool power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t log2_power(std::size_t value) {
    std::size_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

F interpolate2(const F &p0, const F &p1, const F &p2, const F &u) {
    const F half = F(1) / F(2);
    return p0 * (u - F(1)) * (u - F(2)) * half -
           p1 * u * (u - F(2)) +
           p2 * u * (u - F(1)) * half;
}

F interpolate3(const F &p0, const F &p1, const F &p2, const F &p3,
                const F &u) {
    const F sixth = F(1) / F(6);
    const F half = F(1) / F(2);
    return -p0 * (u - F(1)) * (u - F(2)) * (u - F(3)) * sixth +
           p1 * u * (u - F(2)) * (u - F(3)) * half -
           p2 * u * (u - F(1)) * (u - F(3)) * half +
           p3 * u * (u - F(1)) * (u - F(2)) * sixth;
}

bool has_prefix(const std::string &value, const std::string &prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool protocol_label(const std::string &label, const std::string &message) {
    return has_prefix(message, label + "/") ||
           has_prefix(message, "challenge/" + label + "/");
}

const NativeProofMessage *consume(
    const NativeProofSession &session, std::size_t &position,
    NativeProofMessageKind kind, const std::string &label,
    std::string *reason) {
    if (position >= session.messages().size() ||
        session.messages()[position].kind != kind ||
        session.messages()[position].label != label) {
        set_reason(reason, "native range grammar mismatch at " + label);
        return nullptr;
    }
    return &session.messages()[position++];
}

bool decode_vector(const NativeProofMessage *message, std::size_t expected,
                   std::vector<F> &values, std::string *reason) {
    if (message == nullptr || message->kind != NativeProofMessageKind::FieldVector)
        return false;
    // NativeProofSession's canonical decoder is intentionally private.  A
    // temporary one-message session is unnecessary here: use the public
    // replay boundary and decode the exact length-prefixed wire locally.
    const std::string &payload = message->payload;
    if (payload.size() < sizeof(std::uint64_t)) return false;
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
    if (!read_u64(count) || count != expected) return false;
    values.clear();
    values.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > 128 ||
            size > payload.size() - offset)
            return false;
        F value;
        const std::string encoded = payload.substr(offset,
                                                   static_cast<std::size_t>(size));
        try {
            value.setStr(encoded, 16);
        } catch (...) {
            return false;
        }
        if (value.getStr(16) != encoded) return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    if (offset != payload.size()) return false;
    return true;
}

bool decode_field(const NativeProofMessage *message, F &value) {
    if (message == nullptr ||
        (message->kind != NativeProofMessageKind::Field &&
         message->kind != NativeProofMessageKind::Challenge) ||
        message->payload.empty() || message->payload.size() > 128)
        return false;
    try {
        value.setStr(message->payload, 16);
    } catch (...) {
        return false;
    }
    return value.getStr(16) == message->payload;
}

bool verify_scope(const NativeProofSession &session, const std::string &label,
                 std::string *reason) {
    for (const NativeProofMessage &message : session.messages()) {
        if (!protocol_label(label, message.label)) continue;
        // The exact consumer below checks every protocol message.  This
        // early guard rejects aliases and unrelated challenge namespaces.
        if (message.label == label + "/terminal" ||
            message.label == label + "/poly/0" ||
            has_prefix(message.label, label + "/poly/") ||
            has_prefix(message.label, label + "/r/") ||
            has_prefix(message.label, "challenge/" + label + "/r/") ||
            has_prefix(message.label, label + "/terminal"))
            continue;
        set_reason(reason, "unexpected native range message: " + message.label);
        return false;
    }
    return true;
}

std::size_t find_first_poly(const NativeProofSession &session,
                            const std::string &label, std::string *reason) {
    const std::string expected = label + "/poly/0";
    std::size_t found = session.messages().size();
    for (std::size_t i = 0; i < session.messages().size(); ++i) {
        if (session.messages()[i].label != expected) continue;
        if (found != session.messages().size()) {
            set_reason(reason, "duplicate native range polynomial");
            return session.messages().size();
        }
        found = i;
    }
    if (found == session.messages().size())
        set_reason(reason, "missing native range polynomial");
    return found;
}

bool prove_input(const std::vector<F> &values, const F &claim,
                 std::string *reason) {
    if (!power_of_two(values.size())) {
        set_reason(reason, "native range vector is not a power of two");
        return false;
    }
    F sum = 0;
    for (const F &value : values) sum += value;
    if (sum != claim) {
        set_reason(reason, "native range initial claim mismatch");
        return false;
    }
    return true;
}

} // namespace

bool native_range_prove_deg1(NativeProofSession &session,
                             const std::string &label,
                             const std::vector<F> &values,
                             const F &claim,
                             NativeRangeResult *result,
                             std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (label.empty() || !prove_input(values, claim, reason)) return false;
        std::vector<F> current = values;
        F running = claim;
        NativeRangeResult local;
        const std::size_t rounds = log2_power(current.size());
        for (std::size_t round = 0; round < rounds; ++round) {
            const std::size_t half = current.size() / 2;
            F p0 = 0, p1 = 0;
            for (std::size_t i = 0; i < half; ++i) {
                p0 += current[2 * i];
                p1 += current[2 * i + 1];
            }
            if (running != p0 + p1)
                throw std::runtime_error("native range degree-one transition failed");
            session.absorb_fields(label + "/poly/" + std::to_string(round), {p0, p1});
            const F challenge = session.challenge(label + "/r/" + std::to_string(round));
            local.challenges.push_back(challenge);
            running = p0 + challenge * (p1 - p0);
            std::vector<F> next(half);
            for (std::size_t i = 0; i < half; ++i)
                next[i] = current[2 * i] +
                          challenge * (current[2 * i + 1] - current[2 * i]);
            current.swap(next);
        }
        if (current.size() != 1 || current[0] != running)
            throw std::runtime_error("native range degree-one terminal mismatch");
        session.absorb_field(label + "/terminal", running);
        local.terminal = running;
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_range_verify_deg1(const NativeProofSession &session,
                              const std::string &label,
                              const F &claim,
                              F *terminal,
                              std::string *reason) {
    try {
        if (label.empty() || !session.replay(reason) || !verify_scope(session, label, reason))
            return false;
        const std::size_t start = find_first_poly(session, label, reason);
        if (start == session.messages().size()) return false;
        std::size_t position = start;
        F running = claim;
        std::size_t rounds = 0;
        while (position < session.messages().size() &&
               session.messages()[position].label ==
                   label + "/poly/" + std::to_string(rounds)) {
            std::vector<F> values;
            if (!decode_vector(&session.messages()[position], 2, values, reason)) {
                set_reason(reason, "malformed native degree-one polynomial");
                return false;
            }
            if (running != values[0] + values[1]) {
                set_reason(reason, "native range degree-one claim transition failed");
                return false;
            }
            ++position;
            const NativeProofMessage *challenge = consume(
                session, position, NativeProofMessageKind::Challenge,
                "challenge/" + label + "/r/" + std::to_string(rounds), reason);
            if (challenge == nullptr) return false;
            F value;
            if (!decode_field(challenge, value)) {
                set_reason(reason, "malformed native degree-one challenge");
                return false;
            }
            running = values[0] + value * (values[1] - values[0]);
            ++rounds;
        }
        if (rounds == 0) {
            set_reason(reason, "native degree-one range has no rounds");
            return false;
        }
        const NativeProofMessage *terminal_message = consume(
            session, position, NativeProofMessageKind::Field,
            label + "/terminal", reason);
        F encoded_terminal;
        if (terminal_message == nullptr || !decode_field(terminal_message, encoded_terminal) ||
            encoded_terminal != running) {
            set_reason(reason, "native degree-one terminal mismatch");
            return false;
        }
        for (std::size_t i = position; i < session.messages().size(); ++i) {
            if (protocol_label(label, session.messages()[i].label)) {
                set_reason(reason, "native degree-one range has trailing protocol message");
                return false;
            }
        }
        if (terminal != nullptr) *terminal = running;
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool native_range_prove_deg3(NativeProofSession &session,
                             const std::string &label,
                             const std::vector<F> &eq,
                             const std::vector<F> &f,
                             const std::vector<F> &g,
                             const F &claim,
                             NativeRangeResult *result,
                             std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (label.empty() || !power_of_two(eq.size()) || eq.size() != f.size() ||
            f.size() != g.size())
            throw std::invalid_argument("native degree-three range dimensions are invalid");
        std::vector<F> a = eq, b = f, c = g;
        F running = 0;
        for (std::size_t i = 0; i < a.size(); ++i) running += a[i] * b[i] * c[i];
        if (running != claim)
            throw std::invalid_argument("native degree-three range initial claim mismatch");
        NativeRangeResult local;
        const std::size_t rounds = log2_power(a.size());
        for (std::size_t round = 0; round < rounds; ++round) {
            const std::size_t half = a.size() / 2;
            F p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            for (std::size_t i = 0; i < half; ++i) {
                const F a0 = a[2 * i], a1 = a[2 * i + 1];
                const F b0 = b[2 * i], b1 = b[2 * i + 1];
                const F c0 = c[2 * i], c1 = c[2 * i + 1];
                p0 += a0 * b0 * c0;
                p1 += a1 * b1 * c1;
                p2 += (a0 + F(2) * (a1 - a0)) *
                      (b0 + F(2) * (b1 - b0)) *
                      (c0 + F(2) * (c1 - c0));
                p3 += (a0 + F(3) * (a1 - a0)) *
                      (b0 + F(3) * (b1 - b0)) *
                      (c0 + F(3) * (c1 - c0));
            }
            if (running != p0 + p1)
                throw std::runtime_error("native degree-three range transition failed");
            session.absorb_fields(label + "/poly/" + std::to_string(round),
                                  {p0, p1, p2, p3});
            const F challenge = session.challenge(label + "/r/" + std::to_string(round));
            local.challenges.push_back(challenge);
            running = interpolate3(p0, p1, p2, p3, challenge);
            std::vector<F> next_a(half), next_b(half), next_c(half);
            for (std::size_t i = 0; i < half; ++i) {
                next_a[i] = a[2 * i] + challenge * (a[2 * i + 1] - a[2 * i]);
                next_b[i] = b[2 * i] + challenge * (b[2 * i + 1] - b[2 * i]);
                next_c[i] = c[2 * i] + challenge * (c[2 * i + 1] - c[2 * i]);
            }
            a.swap(next_a);
            b.swap(next_b);
            c.swap(next_c);
        }
        if (a.size() != 1 || running != a[0] * b[0] * c[0])
            throw std::runtime_error("native degree-three range terminal mismatch");
        session.absorb_fields(label + "/terminal", {a[0], b[0], c[0]});
        local.terminal = running;
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_range_verify_deg3(const NativeProofSession &session,
                              const std::string &label,
                              const F &claim,
                              F *terminal,
                              std::string *reason) {
    try {
        if (label.empty() || !session.replay(reason) || !verify_scope(session, label, reason))
            return false;
        const std::size_t start = find_first_poly(session, label, reason);
        if (start == session.messages().size()) return false;
        std::size_t position = start;
        F running = claim;
        std::size_t rounds = 0;
        while (position < session.messages().size() &&
               session.messages()[position].label ==
                   label + "/poly/" + std::to_string(rounds)) {
            std::vector<F> values;
            if (!decode_vector(&session.messages()[position], 4, values, reason)) {
                set_reason(reason, "malformed native degree-three polynomial");
                return false;
            }
            if (running != values[0] + values[1]) {
                set_reason(reason, "native degree-three claim transition failed");
                return false;
            }
            ++position;
            const NativeProofMessage *challenge = consume(
                session, position, NativeProofMessageKind::Challenge,
                "challenge/" + label + "/r/" + std::to_string(rounds), reason);
            if (challenge == nullptr) return false;
            F value;
            if (!decode_field(challenge, value)) {
                set_reason(reason, "malformed native degree-three challenge");
                return false;
            }
            running = interpolate3(values[0], values[1], values[2], values[3], value);
            ++rounds;
        }
        if (rounds == 0) {
            set_reason(reason, "native degree-three range has no rounds");
            return false;
        }
        const NativeProofMessage *terminal_message = consume(
            session, position, NativeProofMessageKind::FieldVector,
            label + "/terminal", reason);
        std::vector<F> terminal_values;
        if (!decode_vector(terminal_message, 3, terminal_values, reason) ||
            running != terminal_values[0] * terminal_values[1] * terminal_values[2]) {
            set_reason(reason, "native degree-three terminal mismatch");
            return false;
        }
        for (std::size_t i = position; i < session.messages().size(); ++i) {
            if (protocol_label(label, session.messages()[i].label)) {
                set_reason(reason, "native degree-three range has trailing protocol message");
                return false;
            }
        }
        if (terminal != nullptr) *terminal = running;
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool native_range_prove_product(NativeProofSession &session,
                                const std::string &label,
                                const std::vector<F> &left,
                                const std::vector<F> &right,
                                const F &claim,
                                NativeRangeResult *result,
                                std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (label.empty() || !power_of_two(left.size()) || left.size() != right.size())
            throw std::invalid_argument("native product dimensions are invalid");
        std::vector<F> a = left, b = right;
        F running = 0;
        for (std::size_t i = 0; i < a.size(); ++i) running += a[i] * b[i];
        if (running != claim)
            throw std::invalid_argument("native product initial claim mismatch");
        NativeRangeResult local;
        const std::size_t rounds = log2_power(a.size());
        for (std::size_t round = 0; round < rounds; ++round) {
            const std::size_t half = a.size() / 2;
            F p0 = 0, p1 = 0, p2 = 0;
            for (std::size_t i = 0; i < half; ++i) {
                const F a0 = a[2 * i], a1 = a[2 * i + 1];
                const F b0 = b[2 * i], b1 = b[2 * i + 1];
                p0 += a0 * b0;
                p1 += a1 * b1;
                p2 += (a0 + F(2) * (a1 - a0)) *
                      (b0 + F(2) * (b1 - b0));
            }
            if (running != p0 + p1)
                throw std::runtime_error("native product transition failed");
            session.absorb_fields(label + "/poly/" + std::to_string(round),
                                  {p0, p1, p2});
            const F challenge = session.challenge(label + "/r/" + std::to_string(round));
            local.challenges.push_back(challenge);
            running = interpolate2(p0, p1, p2, challenge);
            std::vector<F> next_a(half), next_b(half);
            for (std::size_t i = 0; i < half; ++i) {
                next_a[i] = a[2 * i] + challenge * (a[2 * i + 1] - a[2 * i]);
                next_b[i] = b[2 * i] + challenge * (b[2 * i + 1] - b[2 * i]);
            }
            a.swap(next_a);
            b.swap(next_b);
        }
        if (a.size() != 1 || running != a[0] * b[0])
            throw std::runtime_error("native product terminal mismatch");
        session.absorb_fields(label + "/terminal", {a[0], b[0]});
        local.terminal = running;
        local.terminal_left = a[0];
        local.terminal_right = b[0];
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_range_verify_product(const NativeProofSession &session,
                                 const std::string &label,
                                 const F &claim,
                                 F *terminal_left,
                                 F *terminal_right,
                                 std::string *reason) {
    try {
        if (label.empty() || !session.replay(reason) || !verify_scope(session, label, reason))
            return false;
        const std::size_t start = find_first_poly(session, label, reason);
        if (start == session.messages().size()) return false;
        std::size_t position = start;
        F running = claim;
        std::size_t rounds = 0;
        while (position < session.messages().size() &&
               session.messages()[position].label ==
                   label + "/poly/" + std::to_string(rounds)) {
            std::vector<F> values;
            if (!decode_vector(&session.messages()[position], 3, values, reason)) {
                set_reason(reason, "malformed native product polynomial");
                return false;
            }
            if (running != values[0] + values[1]) {
                set_reason(reason, "native product claim transition failed");
                return false;
            }
            ++position;
            const NativeProofMessage *challenge = consume(
                session, position, NativeProofMessageKind::Challenge,
                "challenge/" + label + "/r/" + std::to_string(rounds), reason);
            if (challenge == nullptr) return false;
            F value;
            if (!decode_field(challenge, value)) {
                set_reason(reason, "malformed native product challenge: " +
                                      challenge->payload);
                return false;
            }
            running = interpolate2(values[0], values[1], values[2], value);
            ++rounds;
        }
        if (rounds == 0) {
            set_reason(reason, "native product has no rounds");
            return false;
        }
        const NativeProofMessage *terminal_message = consume(
            session, position, NativeProofMessageKind::FieldVector,
            label + "/terminal", reason);
        std::vector<F> terminal_values;
        if (!decode_vector(terminal_message, 2, terminal_values, reason) ||
            running != terminal_values[0] * terminal_values[1]) {
            set_reason(reason, "native product terminal mismatch");
            return false;
        }
        for (std::size_t i = position; i < session.messages().size(); ++i) {
            if (protocol_label(label, session.messages()[i].label)) {
                set_reason(reason, "native product has trailing protocol message");
                return false;
            }
        }
        if (terminal_left != nullptr) *terminal_left = terminal_values[0];
        if (terminal_right != nullptr) *terminal_right = terminal_values[1];
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}
