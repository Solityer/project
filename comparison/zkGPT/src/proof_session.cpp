#include "proof_session.hpp"

#include <cybozu/sha2.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

const char kMagic[] = "ZKGPTNS5";
const std::size_t kMaxString = 1ULL << 26;
const std::size_t kMaxMessages = 1ULL << 20;
const std::uint64_t kMaxWireBytes = 1ULL << 30;

bool decode_challenge_round(const std::string& label, std::string& stream, std::uint64_t& round, bool& is_round) {
    const std::size_t slash = label.rfind('/');
    is_round = false;
    if (slash == std::string::npos) return true;
    const std::string suffix = label.substr(slash + 1);
    const std::size_t parent_slash = label.rfind('/', slash - 1);
    const std::string axis = label.substr(parent_slash == std::string::npos ? 0 : parent_slash + 1, slash - (parent_slash == std::string::npos ? 0 : parent_slash + 1));
    is_round = axis == "r" || axis == "r_u" || axis == "r_v" || axis == "c" || axis == "row-point" || axis == "sig_u" || axis == "sig_v" || axis == "r_minus" || axis == "weight";
    const bool decimal = !suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos;
    if (!decimal) return !is_round;
    if (!is_round) return true;
    if (suffix.size() > 1 && suffix.front() == '0') return false;
    round = 0;
    for (char digit : suffix) {
        const std::uint64_t next = static_cast<std::uint64_t>(digit - '0');
        if (round > (std::numeric_limits<std::uint64_t>::max() - next) / 10) return false;
        round = round * 10 + next;
    }
    stream = label.substr(0, slash);
    return true;
}

bool challenge_round_follows(const std::unordered_map<std::string, std::uint64_t>& rounds, const std::string& stream, std::uint64_t round) {
    const auto previous = rounds.find(stream);
    return previous == rounds.end() ? round == 0 : previous->second != std::numeric_limits<std::uint64_t>::max() && round == previous->second + 1;
}

void set_reason(std::string* reason, const std::string& value) {
    if (reason != nullptr) *reason = value;
}

void append_u32(std::string& out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_u8(std::string& out, std::uint8_t value) { out.push_back(static_cast<char>(value)); }

void append_u64(std::string& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_string(std::string& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

bool read_exact(std::istream& in, char* buffer, std::size_t size) { return static_cast<bool>(in.read(buffer, static_cast<std::streamsize>(size))); }

bool read_u32(std::istream& in, std::uint32_t& value) {
    char bytes[4];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return true;
}

bool read_u64(std::istream& in, std::uint64_t& value) {
    char bytes[8];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return true;
}

bool read_string(std::istream& in, std::string& value) {
    std::uint64_t size = 0;
    if (!read_u64(in, size) || size > kMaxString) return false;
    value.assign(static_cast<std::size_t>(size), '\0');
    return size == 0 || read_exact(in, &value[0], static_cast<std::size_t>(size));
}

bool valid_message_kind(std::uint8_t kind) {
    return kind >= static_cast<std::uint8_t>(NativeProofMessageKind::Data) && kind <= static_cast<std::uint8_t>(NativeProofMessageKind::Challenge);
}

bool wire_size_within_limit(std::ifstream& in) {
    const std::istream::pos_type origin = in.tellg();
    if (origin == std::istream::pos_type(-1)) return false;
    in.seekg(0, std::ios::end);
    const std::istream::pos_type end = in.tellg();
    in.seekg(origin);
    if (end == std::istream::pos_type(-1) || end < origin) return false;
    return static_cast<std::uint64_t>(end) <= kMaxWireBytes;
}

void append_statement_encoding(std::string& out, const NativeProofStatement& statement) {
    append_string(out, "zkgpt-native-statement-v4");
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
    append_string(out, statement.statement_root_digest);
    append_string(out, statement.relation_schema);
    append_string(out, statement.relation_digest);
    append_u64(out, static_cast<std::uint64_t>(statement.relation_order.size()));
    for (const std::string& relation : statement.relation_order) append_string(out, relation);
}

void append_statement_fields(std::string& out, const NativeProofStatement& statement) {
    append_string(out, "zkgpt-native-statement-fields-v4");
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
    append_string(out, statement.relation_schema);
    append_string(out, statement.relation_digest);
    append_u64(out, static_cast<std::uint64_t>(statement.relation_order.size()));
    for (const std::string& relation : statement.relation_order) append_string(out, relation);
}

void append_message_encoding(std::string& out, const NativeProofMessage& message) {
    append_u8(out, static_cast<std::uint8_t>(message.kind));
    append_u64(out, message.sequence);
    append_string(out, message.label);
    append_string(out, message.payload);
}

void append_transcript_prefix(std::string& out, const std::string& domain, const NativeProofStatement& statement, std::size_t message_count) {
    append_string(out, "zkgpt-native-fs-v4");
    append_string(out, domain);
    append_statement_encoding(out, statement);
    append_u64(out, static_cast<std::uint64_t>(message_count));
}

std::string initial_message_body_chain_digest() {
    std::string encoded;
    append_string(encoded, "zkgpt-native-fs-v5/body-init");
    cybozu::Sha256 hash;
    hash.update(encoded.data(), encoded.size());
    std::string digest(32, '\0');
    hash.digest(&digest[0], digest.size(), "", 0);
    return digest;
}

std::string advance_message_body_chain_digest(const std::string& previous_digest, const NativeProofMessage& message) {
    std::string message_prefix;
    append_u8(message_prefix, static_cast<std::uint8_t>(message.kind));
    append_u64(message_prefix, message.sequence);
    append_string(message_prefix, message.label);
    append_u64(message_prefix, static_cast<std::uint64_t>(message.payload.size()));
    const std::uint64_t encoded_size = static_cast<std::uint64_t>(message_prefix.size()) +
                                       static_cast<std::uint64_t>(message.payload.size());

    std::string step_prefix;
    append_string(step_prefix, "zkgpt-native-fs-v5/body-step");
    append_u64(step_prefix, message.sequence);
    append_u64(step_prefix, encoded_size);

    cybozu::Sha256 hash;
    hash.update(step_prefix.data(), step_prefix.size());
    hash.update(previous_digest.data(), previous_digest.size());
    hash.update(message_prefix.data(), message_prefix.size());
    hash.update(message.payload.data(), message.payload.size());
    std::string digest(32, '\0');
    hash.digest(&digest[0], digest.size(), "", 0);
    return digest;
}

F derive_challenge_v5(const std::string& domain, const NativeProofStatement& statement,
                      std::size_t message_count, const std::string& message_body_chain_digest,
                      const std::string& challenge_label) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-fs-v5/challenge");
    append_string(encoded, domain);
    append_statement_encoding(encoded, statement);
    append_u64(encoded, static_cast<std::uint64_t>(message_count));
    append_string(encoded, challenge_label);
    append_string(encoded, message_body_chain_digest);
    F result;
    result.setHashOf(encoded.data(), encoded.size());
    return result;
}

void append_transcript_encoding(std::string& out, const std::string& domain, const NativeProofStatement& statement,
                                const std::vector<NativeProofMessage>& messages) {
    append_transcript_prefix(out, domain, statement, messages.size());
    for (const NativeProofMessage& message : messages) append_message_encoding(out, message);
}

bool is_challenge_label(const std::string& label) { return label.compare(0, sizeof("challenge/") - 1, "challenge/") == 0; }

bool is_lower_hex_digest(const std::string& value) {
    if (value.size() != 64) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

bool is_field_id(const std::string& value) {
    if (value.empty() || value.size() > 64) return false;
    for (char c : value) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) return false;
    }
    return true;
}

}

std::string NativeProofStatement::root() const { return statement_root_digest; }

std::string NativeProofStatement::structured_root() const {
    std::string encoded;
    append_statement_fields(encoded, *this);
    F root;
    root.setHashOf(encoded.data(), encoded.size());
    std::string value = root.getStr(16);
    if (value.empty()) value = "0";
    if (value.size() > 64) throw std::runtime_error("native statement root overflow");
    return std::string(64 - value.size(), '0') + value;
}

std::string NativeProofStatement::basis_digest_for(const G1& G, const std::vector<G1>& generators, const std::vector<F>& basis, std::uint64_t l_dimension) {
    if (generators.empty() || generators.size() != basis.size() || (generators.size() & (generators.size() - 1)) != 0 || l_dimension == 0)
        throw std::invalid_argument("invalid native Hyrax basis setup");
    std::string encoded;
    append_string(encoded, "zkgpt-native-basis-v1");
    append_u64(encoded, l_dimension);
    append_string(encoded, G.serializeToHexStr());
    append_u64(encoded, generators.size());

    (void)basis;
    for (std::size_t i = 0; i < generators.size(); ++i) { append_string(encoded, generators[i].serializeToHexStr()); }
    F digest;
    digest.setHashOf(encoded.data(), encoded.size());
    std::string value = digest.getStr(16);
    if (value.empty()) value = "0";
    if (value.size() > 64) throw std::runtime_error("native basis digest overflow");
    return std::string(64 - value.size(), '0') + value;
}

std::string NativeProofStatement::basis_descriptor_digest_for(const G1& H, const std::string& canonical_basis_label,
                                                               const std::string& canonical_basis_id,
                                                               const std::string& generator_derivation_id,
                                                               std::uint64_t full_domain) {
    if (canonical_basis_label.empty() || canonical_basis_id.empty() || generator_derivation_id.empty() || full_domain == 0 ||
        (full_domain & (full_domain - 1)) != 0)
        throw std::invalid_argument("invalid native canonical generator descriptor");
    std::string encoded;
    append_string(encoded, "zkgpt-native-basis-descriptor-v2");
    append_string(encoded, canonical_basis_label);
    append_string(encoded, canonical_basis_id);
    append_string(encoded, generator_derivation_id);
    append_u64(encoded, full_domain);
    append_string(encoded, H.serializeToHexStr());
    F digest;
    digest.setHashOf(encoded.data(), encoded.size());
    std::string value = digest.getStr(16);
    if (value.empty()) value = "0";
    if (value.size() > 64) throw std::runtime_error("native basis descriptor digest overflow");
    return std::string(64 - value.size(), '0') + value;
}

bool NativeProofStatement::valid(std::string* reason) const {
    const auto fail = [&](const std::string& message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (!is_lower_hex_digest(verifier_context_digest) || !is_lower_hex_digest(plan_digest) || !is_lower_hex_digest(resources_digest) ||
        !is_lower_hex_digest(tables_digest) || !is_lower_hex_digest(com_w_digest) || !is_lower_hex_digest(aux_digest) || !is_lower_hex_digest(l_digest) ||
        !is_lower_hex_digest(basis_digest) || !is_lower_hex_digest(statement_root_digest))
        return fail("native proof statement has a non-canonical digest");
    if (statement_root_digest != verifier_context_digest) return fail("native proof statement root does not match verifier context digest");
    if (!is_field_id(field_id)) return fail("native proof statement field id is invalid");
    if (l_dimension == 0 || (l_dimension & (l_dimension - 1)) != 0 ||
        l_dimension > std::numeric_limits<std::uint64_t>::max() / 4)
        return fail("native proof statement L dimension is invalid");
    if (relation_schema != "zkgpt-gat-relation-v1" && relation_schema != "zkgpt-gat-sparse-gkr-v1")
        return fail("native proof statement relation schema is unsupported");
    if (!is_lower_hex_digest(relation_digest)) return fail("native proof statement relation digest is not canonical");
    std::set<std::string> relation_ids;
    for (const std::string& relation : relation_order) {
        if (relation.empty() || relation.size() > kMaxString || !relation_ids.insert(relation).second)
            return fail("native proof statement relation order is not canonical");
    }
    return true;
}

NativeProofSession::NativeProofSession()
    : message_body_chain_digest_(initial_message_body_chain_digest()), replayed_body_chain_digest_(initial_message_body_chain_digest()) {}

NativeProofSession::NativeProofSession(const std::string& domain, const NativeProofStatement& statement)
    : domain_(domain), statement_(statement), message_body_chain_digest_(initial_message_body_chain_digest()),
      replayed_body_chain_digest_(initial_message_body_chain_digest()) {
    if (domain_.empty() || !statement_.valid()) throw std::invalid_argument("native proof session identity is incomplete");
}

std::string NativeProofSession::field_encoding(const F& value) { return value.getStr(16); }

bool NativeProofSession::decode_field(const std::string& payload, F& value) {
    if (payload.empty() || payload.size() > 128) return false;
    try {
        value.setStr(payload, 16);

        return field_encoding(value) == payload;
    } catch (...) { return false; }
}

bool NativeProofSession::decode_fields(const std::string& payload, std::vector<F>& values) {
    values.clear();
    if (payload.size() < sizeof(std::uint64_t)) return false;
    std::size_t offset = 0;
    auto read_u64_bytes = [&](std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64_bytes(count) || count > kMaxMessages) return false;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t size = 0;
        if (!read_u64_bytes(size) || size == 0 || size > 128 || size > payload.size() - offset) return false;
        F value;
        if (!decode_field(payload.substr(offset, static_cast<std::size_t>(size)), value)) return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    return offset == payload.size();
}

void NativeProofSession::absorb(const std::string& label, const std::string& payload) {
    if (label.empty() || label.size() > kMaxString || payload.size() > kMaxString) throw std::invalid_argument("native proof message exceeds wire limits");
    if (is_challenge_label(label)) throw std::invalid_argument("native proof challenge label is reserved");
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::Data, label, payload, static_cast<std::uint64_t>(messages_.size())});
    invalidate_validation_cache();
    label_positions_[messages_.back().label].push_back(messages_.size() - 1);
    append_message_body_chain(messages_.back());
}

void NativeProofSession::absorb_field(const std::string& label, const F& value) {
    if (label.empty() || label.size() > kMaxString || is_challenge_label(label)) throw std::invalid_argument("native proof field message exceeds wire limits");
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::Field, label, field_encoding(value), static_cast<std::uint64_t>(messages_.size())});
    invalidate_validation_cache();
    label_positions_[messages_.back().label].push_back(messages_.size() - 1);
    append_message_body_chain(messages_.back());
}

void NativeProofSession::absorb_fields(const std::string& label, const std::vector<F>& values) {
    if (label.empty() || label.size() > kMaxString || is_challenge_label(label) || values.size() > kMaxMessages)
        throw std::invalid_argument("native proof field vector exceeds wire limits");
    std::string payload;
    append_u64(payload, static_cast<std::uint64_t>(values.size()));
    for (const F& value : values) append_string(payload, field_encoding(value));
    messages_.push_back(NativeProofMessage{NativeProofMessageKind::FieldVector, label, std::move(payload), static_cast<std::uint64_t>(messages_.size())});
    invalidate_validation_cache();
    label_positions_[messages_.back().label].push_back(messages_.size() - 1);
    append_message_body_chain(messages_.back());
}

F NativeProofSession::derive(const std::string& label) const {
    if (label.empty()) throw std::invalid_argument("native proof challenge label is empty");
    return derive_challenge_v5(domain_, statement_, messages_.size(), message_body_chain_digest_, label);
}

F NativeProofSession::challenge(const std::string& label) {
    if (is_challenge_label(label)) throw std::invalid_argument("native proof challenge label already has reserved prefix");
    std::string round_stream;
    std::uint64_t round = 0;
    bool is_round = false;
    const std::string challenge_label = "challenge/" + label;
    if (!decode_challenge_round(challenge_label, round_stream, round, is_round)) throw std::invalid_argument("native proof challenge round index is not canonical");
    if (is_round && !challenge_round_follows(challenge_round_last_, round_stream, round))
        throw std::invalid_argument("native proof challenge round schedule is not sequential at " + challenge_label);
    const F result = derive(label);
    messages_.push_back(
        NativeProofMessage{NativeProofMessageKind::Challenge, challenge_label, field_encoding(result), static_cast<std::uint64_t>(messages_.size())});
    if (is_round) challenge_round_last_[round_stream] = round;
    invalidate_validation_cache();
    label_positions_[messages_.back().label].push_back(messages_.size() - 1);
    append_message_body_chain(messages_.back());
    return result;
}

bool NativeProofSession::rollback(std::size_t checkpoint, std::string* reason) {
    if (checkpoint > messages_.size()) {
        set_reason(reason, "native proof rollback checkpoint is out of range");
        return false;
    }
    messages_.resize(checkpoint);
    rebuild_message_body_chain();
    return true;
}

void NativeProofSession::append_message_body_chain(const NativeProofMessage& message) {
    message_body_chain_digest_ = advance_message_body_chain_digest(message_body_chain_digest_, message);
}

void NativeProofSession::rebuild_message_body_chain() {
    invalidate_validation_cache();
    validated_prefix_size_ = 0;
    validated_challenge_labels_.clear();
    validated_challenge_round_last_.clear();
    challenge_round_last_.clear();
    replayed_prefix_size_ = 0;
    message_body_chain_digest_ = initial_message_body_chain_digest();
    replayed_body_chain_digest_ = initial_message_body_chain_digest();
    label_positions_.clear();
    label_positions_.reserve(messages_.size());
    for (std::size_t index = 0; index < messages_.size(); ++index) {
        const NativeProofMessage& message = messages_[index];
        label_positions_[message.label].push_back(index);
        if (message.kind == NativeProofMessageKind::Challenge) {
            std::string stream;
            std::uint64_t round = 0;
            bool is_round = false;
            if (decode_challenge_round(message.label, stream, round, is_round) && is_round) challenge_round_last_[stream] = round;
        }
        append_message_body_chain(message);
    }
}

void NativeProofSession::invalidate_validation_cache() {
    ++content_version_;
    valid_cache_ = false;
    replay_cache_ = false;
    valid_cache_reason_.clear();
    replay_cache_reason_.clear();
    verified_opening_cache_.clear();
}

bool NativeProofSession::replay(std::string* reason) const {
    if (replay_cache_ && replay_cache_version_ == content_version_) {
        if (!replay_cache_result_) set_reason(reason, replay_cache_reason_);
        return replay_cache_result_;
    }
    std::string local_reason;
    const bool result = replay_uncached(&local_reason);
    replay_cache_ = true;
    replay_cache_version_ = content_version_;
    replay_cache_result_ = result;
    replay_cache_reason_ = result ? std::string() : local_reason;
    if (!result) set_reason(reason, replay_cache_reason_);
    return result;
}

bool NativeProofSession::replay_uncached(std::string* reason) const {
    try {
        if (!valid(reason)) return false;
        for (std::size_t index = replayed_prefix_size_; index < messages_.size(); ++index) {
            const NativeProofMessage& message = messages_[index];
            if (message.kind == NativeProofMessageKind::Challenge) {
                F expected;
                if (!decode_field(message.payload, expected)) { set_reason(reason, "native proof challenge payload is not a field element"); return false; }
                const std::string challenge_label = message.label.substr(sizeof("challenge/") - 1);
                const F derived = derive_challenge_v5(
                    domain_, statement_, index, replayed_body_chain_digest_, challenge_label);
                if (derived != expected) { set_reason(reason, "native proof Fiat-Shamir challenge mismatch"); return false; }
            }
            replayed_body_chain_digest_ = advance_message_body_chain_digest(replayed_body_chain_digest_, message);
            replayed_prefix_size_ = index + 1;
        }
        return true;
    } catch (const std::exception& error) {
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

bool NativeProofSession::valid(std::string* reason) const {
    if (valid_cache_ && valid_cache_version_ == content_version_) {
        if (!valid_cache_result_) set_reason(reason, valid_cache_reason_);
        return valid_cache_result_;
    }
    std::string local_reason;
    const bool result = valid_uncached(&local_reason);
    valid_cache_ = true;
    valid_cache_version_ = content_version_;
    valid_cache_result_ = result;
    valid_cache_reason_ = result ? std::string() : local_reason;
    if (!result) set_reason(reason, valid_cache_reason_);
    return result;
}

bool NativeProofSession::valid_uncached(std::string* reason) const {
    if (domain_.empty() || domain_.size() > kMaxString) {
        set_reason(reason, "native proof session domain is invalid");
        return false;
    }
    if (!statement_.valid(reason)) return false;
    if (messages_.size() > kMaxMessages) {
        set_reason(reason, "native proof session has too many messages");
        return false;
    }
    std::unordered_set<std::string> new_challenge_labels;
    std::unordered_map<std::string, std::uint64_t> challenge_rounds = validated_challenge_round_last_;
    for (std::size_t index = validated_prefix_size_; index < messages_.size(); ++index) {
        const NativeProofMessage& message = messages_[index];
        if (message.label.empty() || message.label.size() > kMaxString || message.payload.size() > kMaxString) {
            set_reason(reason, "native proof session message exceeds wire limits");
            return false;
        }
        if (message.sequence != static_cast<std::uint64_t>(index)) {
            set_reason(reason, "native proof session message sequence is not canonical");
            return false;
        }
        if (!valid_message_kind(static_cast<std::uint8_t>(message.kind))) {
            set_reason(reason, "unknown native proof message type");
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
        case NativeProofMessageKind::Challenge: {
            std::string round_stream;
            std::uint64_t round = 0;
            bool is_round = false;
            if (!is_challenge_label(message.label) || message.label == "challenge/" || !decode_challenge_round(message.label, round_stream, round, is_round)) {
                set_reason(reason, "challenge message label is invalid");
                return false;
            }
            if (validated_challenge_labels_.count(message.label) != 0 || !new_challenge_labels.insert(message.label).second) {
                set_reason(reason, "duplicate native proof challenge label");
                return false;
            }
            if (is_round && !challenge_round_follows(challenge_rounds, round_stream, round)) {
                set_reason(reason, "native proof challenge round schedule is not sequential at " + message.label);
                return false;
            }
            if (is_round) challenge_rounds[round_stream] = round;
            F challenge_value;
            if (!decode_field(message.payload, challenge_value)) {
                set_reason(reason, "challenge message is not canonical");
                return false;
            }
            break;
        }
        default:

            set_reason(reason, "unknown native proof message type");
            return false;
        }
    }
    validated_challenge_labels_.insert(new_challenge_labels.begin(), new_challenge_labels.end());
    validated_challenge_round_last_.swap(challenge_rounds);
    validated_prefix_size_ = messages_.size();
    return true;
}

bool NativeProofSession::write(const std::string& path, std::string* reason) const {
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
    append_string(encoded, statement_.relation_schema);
    append_string(encoded, statement_.relation_digest);
    append_u64(encoded, static_cast<std::uint64_t>(statement_.relation_order.size()));
    for (const std::string& relation : statement_.relation_order) append_string(encoded, relation);
    append_u64(encoded, static_cast<std::uint64_t>(messages_.size()));
    for (const NativeProofMessage& message : messages_) {
        append_u8(encoded, static_cast<std::uint8_t>(message.kind));
        append_u64(encoded, message.sequence);
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

bool NativeProofSession::read(const std::string& path, NativeProofSession& out, std::string* reason) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        set_reason(reason, "cannot open native proof session wire");
        return false;
    }
    if (!wire_size_within_limit(in)) {
        set_reason(reason, "native proof session wire exceeds size limit");
        return false;
    }
    in.seekg(0, std::ios::beg);
    char magic[sizeof(kMagic) - 1];
    if (!read_exact(in, magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        set_reason(reason, "invalid native proof session wire magic");
        return false;
    }
    std::uint32_t version = 0;
    if (!read_u32(in, version) || version != kVersion) {
        set_reason(reason, "unsupported native proof session wire version");
        return false;
    }
    NativeProofSession parsed;
    if (!read_string(in, parsed.domain_) || !read_string(in, parsed.statement_.verifier_context_digest) || !read_string(in, parsed.statement_.field_id) ||
        !read_string(in, parsed.statement_.plan_digest) || !read_string(in, parsed.statement_.resources_digest) ||
        !read_string(in, parsed.statement_.tables_digest) || !read_string(in, parsed.statement_.com_w_digest) ||
        !read_string(in, parsed.statement_.aux_digest) || !read_string(in, parsed.statement_.l_digest) || !read_string(in, parsed.statement_.basis_digest) ||
        !read_u64(in, parsed.statement_.l_dimension)) {
        set_reason(reason, "truncated native proof session identity");
        return false;
    }
    std::string statement_root;
    if (!read_string(in, statement_root)) {
        set_reason(reason, "native proof statement root mismatch");
        return false;
    }
    parsed.statement_.statement_root_digest = statement_root;
    std::uint64_t relation_count = 0;
    if (!read_string(in, parsed.statement_.relation_schema) || !read_string(in, parsed.statement_.relation_digest) || !read_u64(in, relation_count) ||
        relation_count > kMaxMessages) {
        set_reason(reason, "truncated native proof relation identity");
        return false;
    }
    parsed.statement_.relation_order.reserve(static_cast<std::size_t>(relation_count));
    for (std::uint64_t index = 0; index < relation_count; ++index) {
        std::string relation;
        if (!read_string(in, relation)) {
            set_reason(reason, "truncated native proof relation order");
            return false;
        }
        parsed.statement_.relation_order.push_back(std::move(relation));
    }
    if (!parsed.statement_.valid(reason) || statement_root != parsed.statement_.root()) {
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
        if (!in.read(&kind, 1) || !read_u64(in, message.sequence) || !read_string(in, message.label) || !read_string(in, message.payload)) {
            set_reason(reason, "truncated native proof session message");
            return false;
        }
        message.kind = static_cast<NativeProofMessageKind>(static_cast<std::uint8_t>(kind));
        parsed.messages_.push_back(std::move(message));
    }
    parsed.rebuild_message_body_chain();
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

bool NativeProofSession::from_messages(const std::string& domain, const NativeProofStatement& statement, const std::vector<NativeProofMessage>& messages,
                                       NativeProofSession& out, std::string* reason) {
    try {
        NativeProofSession parsed(domain, statement);
        if (messages.size() > kMaxMessages) {
            set_reason(reason, "native proof session message count exceeds wire limit");
            return false;
        }
        parsed.messages_ = messages;
        parsed.rebuild_message_body_chain();
        if (!parsed.valid(reason) || !parsed.replay(reason)) return false;
        out = std::move(parsed);
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}
