#include "proof_session.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

const char kMagic[] = "ZKGPTNS1";
const std::size_t kMaxString = 1ULL << 26;
const std::size_t kMaxMessages = 1ULL << 20;

void set_reason(std::string *reason, const std::string &value) {
    if (reason != nullptr) *reason = value;
}

void append_u32(std::string &out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_u8(std::string &out, std::uint8_t value) {
    out.push_back(static_cast<char>(value));
}

void append_u64(std::string &out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_string(std::string &out, const std::string &value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

bool read_exact(std::istream &in, char *buffer, std::size_t size) {
    return static_cast<bool>(in.read(buffer, static_cast<std::streamsize>(size)));
}

bool read_u32(std::istream &in, std::uint32_t &value) {
    char bytes[4];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return true;
}

bool read_u64(std::istream &in, std::uint64_t &value) {
    char bytes[8];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return true;
}

bool read_string(std::istream &in, std::string &value) {
    std::uint64_t size = 0;
    if (!read_u64(in, size) || size > kMaxString) return false;
    value.assign(static_cast<std::size_t>(size), '\0');
    return size == 0 || read_exact(in, &value[0], static_cast<std::size_t>(size));
}

void append_statement_encoding(std::string &out,
                               const NativeProofStatement &statement) {
    append_string(out, "zkgpt-native-statement-v1");
    append_string(out, statement.verifier_context_digest);
    append_string(out, statement.field_id);
    append_string(out, statement.plan_digest);
    append_string(out, statement.resources_digest);
    append_string(out, statement.tables_digest);
    append_string(out, statement.com_w_digest);
    append_string(out, statement.aux_digest);
    append_string(out, statement.l_digest);
    append_string(out, statement.basis_digest);
    append_u64(out, statement.l_dimension);
}

void append_transcript_encoding(std::string &out,
                                const std::string &domain,
                                const NativeProofStatement &statement,
                                const std::vector<NativeProofMessage> &messages) {
    append_string(out, "zkgpt-native-fs-v1");
    append_string(out, domain);
    append_statement_encoding(out, statement);
    append_u64(out, static_cast<std::uint64_t>(messages.size()));
    for (const NativeProofMessage &message : messages) {
        append_u8(out, static_cast<std::uint8_t>(message.kind));
        append_string(out, message.label);
        append_string(out, message.payload);
    }
}

bool is_challenge_label(const std::string &label) {
    return label.compare(0, sizeof("challenge/") - 1, "challenge/") == 0;
}

bool is_lower_hex_digest(const std::string &value) {
    if (value.size() != 64) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

bool is_field_id(const std::string &value) {
    if (value.empty() || value.size() > 64) return false;
    for (char c : value) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_' || c == '.')) return false;
    }
    return true;
}

} // namespace

std::string NativeProofStatement::root() const {
    std::string encoded;
    append_statement_encoding(encoded, *this);
    F root;
    root.setHashOf(encoded.data(), encoded.size());
    return root.getStr(16);
}

bool NativeProofStatement::valid(std::string *reason) const {
    const auto fail = [&](const std::string &message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (!is_lower_hex_digest(verifier_context_digest) ||
        !is_lower_hex_digest(plan_digest) ||
        !is_lower_hex_digest(resources_digest) ||
        !is_lower_hex_digest(tables_digest) ||
        !is_lower_hex_digest(com_w_digest) ||
        !is_lower_hex_digest(aux_digest) ||
        !is_lower_hex_digest(l_digest) ||
        !is_lower_hex_digest(basis_digest))
        return fail("native proof statement has a non-canonical digest");
    if (!is_field_id(field_id)) return fail("native proof statement field id is invalid");
    if (l_dimension == 0 || l_dimension > (1ULL << 30))
        return fail("native proof statement L dimension is invalid");
    return true;
}

NativeProofSession::NativeProofSession(const std::string &domain,
                                       const NativeProofStatement &statement)
    : domain_(domain), statement_(statement) {
    if (domain_.empty() || !statement_.valid())
        throw std::invalid_argument("native proof session identity is incomplete");
}

std::string NativeProofSession::field_encoding(const F &value) {
    // mcl's base-16 field encoding is canonical and round-trips through
    // Fr::setStr.  It avoids locale-dependent decimal formatting.
    return value.getStr(16);
}

bool NativeProofSession::decode_field(const std::string &payload, F &value) {
    if (payload.empty() || payload.size() > 128) return false;
    try {
        value.setStr(payload, 16);
        // Reject alternate spellings (leading zeroes, upper-case digits, or
        // non-canonical zero) so a serialized challenge has one wire form.
        return field_encoding(value) == payload;
    } catch (...) {
        return false;
    }
}

bool NativeProofSession::decode_fields(const std::string &payload,
                                       std::vector<F> &values) {
    values.clear();
    if (payload.size() < sizeof(std::uint64_t)) return false;
    std::size_t offset = 0;
    auto read_u64_bytes = [&](std::uint64_t &value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64_bytes(count) || count > kMaxMessages) return false;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t size = 0;
        if (!read_u64_bytes(size) || size == 0 || size > 128 ||
            size > payload.size() - offset)
            return false;
        F value;
        if (!decode_field(payload.substr(offset, static_cast<std::size_t>(size)), value))
            return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    return offset == payload.size();
}

void NativeProofSession::absorb(const std::string &label,
                                const std::string &payload) {
    if (label.empty() || label.size() > kMaxString || payload.size() > kMaxString)
        throw std::invalid_argument("native proof message exceeds wire limits");
    if (is_challenge_label(label))
        throw std::invalid_argument("native proof challenge label is reserved");
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::Data, label, payload});
}

void NativeProofSession::absorb_field(const std::string &label, const F &value) {
    if (label.empty() || label.size() > kMaxString)
        throw std::invalid_argument("native proof field message exceeds wire limits");
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::Field, label,
                                           field_encoding(value)});
}

void NativeProofSession::absorb_fields(const std::string &label,
                                       const std::vector<F> &values) {
    if (label.empty() || label.size() > kMaxString || values.size() > kMaxMessages)
        throw std::invalid_argument("native proof field vector exceeds wire limits");
    std::string payload;
    append_u64(payload, static_cast<std::uint64_t>(values.size()));
    for (const F &value : values) append_string(payload, field_encoding(value));
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::FieldVector, label,
                                           std::move(payload)});
}

F NativeProofSession::derive(const std::string &label) const {
    if (label.empty()) throw std::invalid_argument("native proof challenge label is empty");
    std::string encoded;
    append_transcript_encoding(encoded, domain_, statement_, messages_);
    append_string(encoded, label);
    F result;
    result.setHashOf(encoded.data(), encoded.size());
    return result;
}

F NativeProofSession::challenge(const std::string &label) {
    if (is_challenge_label(label))
        throw std::invalid_argument("native proof challenge label already has reserved prefix");
    const F result = derive(label);
    const std::string challenge_label = "challenge/" + label;
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::Challenge,
                                           challenge_label, field_encoding(result)});
    return result;
}

bool NativeProofSession::rollback(std::size_t checkpoint, std::string *reason) {
    if (checkpoint > messages_.size()) {
        set_reason(reason, "native proof rollback checkpoint is out of range");
        return false;
    }
    messages_.resize(checkpoint);
    return true;
}

bool NativeProofSession::replay(std::string *reason) const {
    try {
        if (!valid(reason)) return false;
        NativeProofSession prefix(domain_, statement_);
        for (const NativeProofMessage &message : messages_) {
            if (message.kind == NativeProofMessageKind::Challenge) {
                F expected;
                if (!decode_field(message.payload, expected)) {
                    set_reason(reason, "native proof challenge payload is not a field element");
                    return false;
                }
                const std::string label = message.label.substr(sizeof("challenge/") - 1);
                const F derived = prefix.derive(label);
                if (derived != expected) {
                    set_reason(reason, "native proof Fiat-Shamir challenge mismatch");
                    return false;
                }
                prefix.messages_.push_back(message);
            } else {
                prefix.messages_.push_back(message);
            }
        }
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

std::string NativeProofSession::digest() const {
    std::string encoded;
    append_transcript_encoding(encoded, domain_, statement_, messages_);
    F digest_field;
    digest_field.setHashOf(encoded.data(), encoded.size());
    return field_encoding(digest_field);
}

bool NativeProofSession::valid(std::string *reason) const {
    if (domain_.empty() || !statement_.valid(reason)) {
        set_reason(reason, "native proof session identity is incomplete");
        return false;
    }
    if (messages_.size() > kMaxMessages) {
        set_reason(reason, "native proof session has too many messages");
        return false;
    }
    std::set<std::string> challenge_labels;
    for (const NativeProofMessage &message : messages_) {
        if (message.label.empty() || message.label.size() > kMaxString ||
            message.payload.size() > kMaxString) {
            set_reason(reason, "native proof session message exceeds wire limits");
            return false;
        }
        switch (message.kind) {
        case NativeProofMessageKind::Data:
            if (is_challenge_label(message.label)) {
                set_reason(reason, "data message uses a reserved challenge label");
                return false;
            }
            break;
        case NativeProofMessageKind::Field: {
            if (is_challenge_label(message.label)) {
                set_reason(reason, "field message uses a reserved challenge label");
                return false;
            }
            F value;
            if (!decode_field(message.payload, value)) {
                set_reason(reason, "field message is not canonical");
                return false;
            }
            break;
        }
        case NativeProofMessageKind::FieldVector: {
            if (is_challenge_label(message.label)) {
                set_reason(reason, "field vector uses a reserved challenge label");
                return false;
            }
            std::vector<F> values;
            if (!decode_fields(message.payload, values)) {
                set_reason(reason, "field vector message is not canonical");
                return false;
            }
            break;
        }
        case NativeProofMessageKind::Challenge:
            if (!is_challenge_label(message.label) || message.label == "challenge/") {
                set_reason(reason, "challenge message label is invalid");
                return false;
            }
            if (!challenge_labels.insert(message.label).second) {
                set_reason(reason, "duplicate native proof challenge label");
                return false;
            }
            F challenge_value;
            if (!decode_field(message.payload, challenge_value)) {
                set_reason(reason, "challenge message is not canonical");
                return false;
            }
            break;
        default:
            set_reason(reason, "unknown native proof message type");
            return false;
        }
    }
    return true;
}

bool NativeProofSession::write(const std::string &path, std::string *reason) const {
    if (!valid(reason) || !replay(reason)) return false;
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        set_reason(reason, "cannot open native proof session wire");
        return false;
    }
    out.write(kMagic, sizeof(kMagic) - 1);
    std::string encoded;
    append_u32(encoded, kVersion);
    append_string(encoded, domain_);
    append_string(encoded, statement_.verifier_context_digest);
    append_string(encoded, statement_.field_id);
    append_string(encoded, statement_.plan_digest);
    append_string(encoded, statement_.resources_digest);
    append_string(encoded, statement_.tables_digest);
    append_string(encoded, statement_.com_w_digest);
    append_string(encoded, statement_.aux_digest);
    append_string(encoded, statement_.l_digest);
    append_string(encoded, statement_.basis_digest);
    append_u64(encoded, statement_.l_dimension);
    append_string(encoded, statement_.root());
    append_u64(encoded, static_cast<std::uint64_t>(messages_.size()));
    for (const NativeProofMessage &message : messages_) {
        append_u8(encoded, static_cast<std::uint8_t>(message.kind));
        append_string(encoded, message.label);
        append_string(encoded, message.payload);
    }
    append_string(encoded, digest());
    out.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    if (!out.good()) {
        set_reason(reason, "native proof session wire write failed");
        return false;
    }
    return true;
}

bool NativeProofSession::read(const std::string &path, NativeProofSession &out,
                              std::string *reason) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        set_reason(reason, "cannot open native proof session wire");
        return false;
    }
    char magic[sizeof(kMagic) - 1];
    if (!read_exact(in, magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        set_reason(reason, "invalid native proof session wire magic");
        return false;
    }
    std::uint32_t version = 0;
    if (!read_u32(in, version) || version != kVersion) {
        set_reason(reason, "unsupported native proof session wire version");
        return false;
    }
    NativeProofSession parsed;
    if (!read_string(in, parsed.domain_) ||
        !read_string(in, parsed.statement_.verifier_context_digest) ||
        !read_string(in, parsed.statement_.field_id) ||
        !read_string(in, parsed.statement_.plan_digest) ||
        !read_string(in, parsed.statement_.resources_digest) ||
        !read_string(in, parsed.statement_.tables_digest) ||
        !read_string(in, parsed.statement_.com_w_digest) ||
        !read_string(in, parsed.statement_.aux_digest) ||
        !read_string(in, parsed.statement_.l_digest) ||
        !read_string(in, parsed.statement_.basis_digest) ||
        !read_u64(in, parsed.statement_.l_dimension)) {
        set_reason(reason, "truncated native proof session identity");
        return false;
    }
    std::string statement_root;
    if (!read_string(in, statement_root) ||
        !parsed.statement_.valid(reason) || statement_root != parsed.statement_.root()) {
        set_reason(reason, "native proof statement root mismatch");
        return false;
    }
    std::uint64_t count = 0;
    if (!read_u64(in, count) || count > kMaxMessages) {
        set_reason(reason, "invalid native proof session message count");
        return false;
    }
    parsed.messages_.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        NativeProofMessage message;
        char kind = 0;
        if (!in.read(&kind, 1) || !read_string(in, message.label) ||
            !read_string(in, message.payload)) {
            set_reason(reason, "truncated native proof session message");
            return false;
        }
        message.kind = static_cast<NativeProofMessageKind>(
            static_cast<std::uint8_t>(kind));
        parsed.messages_.push_back(std::move(message));
    }
    std::string wire_digest;
    if (!read_string(in, wire_digest)) {
        set_reason(reason, "truncated native proof session digest");
        return false;
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) {
        set_reason(reason, "native proof session wire has trailing bytes");
        return false;
    }
    if (!parsed.valid(reason) || parsed.digest() != wire_digest || !parsed.replay(reason)) {
        if (reason != nullptr && reason->empty()) *reason = "native proof session authentication failed";
        return false;
    }
    out = std::move(parsed);
    return true;
}
