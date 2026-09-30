#include "native_pedersen_relation.hpp"

#include <mcl/bn256.hpp>

#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace zkgpt {
namespace native_pedersen {
namespace {

const char kMagic[] = "zkgpt-native-pedersen-relation-v1";
const char kBasisPrefix[] = "zkgpt/native-pedersen-relation/v1/";
const std::size_t kMaxDomain = 256;
const std::size_t kMaxLabel = 512;
const std::size_t kMaxPoint = 4096;
const std::size_t kMaxOperands = 1U << 20;

constexpr std::size_t kGeneratorCacheCapacity = 64;

std::once_flag field_once;

void ensure_field() {
    std::call_once(field_once, []() { mcl::bn::initPairing(mcl::BN254); });
}

void set_reason(std::string* reason, const std::string& message) {
    if (reason != nullptr) *reason = message;
}

std::string reason_or(const std::string* reason, const char* fallback) { return reason != nullptr && !reason->empty() ? *reason : fallback; }

bool valid_scope(const std::string& value, std::size_t maximum, bool allow_slash) {
    if (value.empty() || value.size() > maximum || value.front() == '/' || value.back() == '/' || value.find("//") != std::string::npos ||
        value.find('|') != std::string::npos)
        return false;
    if (!allow_slash && value.find('/') != std::string::npos) return false;
    for (unsigned char c : value) {
        if (std::iscntrl(c) || std::isspace(c)) return false;
    }
    return value.compare(0, sizeof("challenge/") - 1, "challenge/") != 0;
}

bool valid_domain(const std::string& domain) { return valid_scope(domain, kMaxDomain, true); }

bool valid_label(const std::string& label) { return valid_scope(label, kMaxLabel, true); }

void append_u64(std::string& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_string(std::string& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

struct Reader {
    const std::string& payload;
    std::size_t offset = 0;

    bool read_u64(std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    }

    bool read_string(std::string& value, std::size_t maximum, bool allow_empty = false) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size > maximum || size > payload.size() - offset || (!allow_empty && size == 0)) return false;
        value.assign(payload.data() + offset, static_cast<std::size_t>(size));
        offset += static_cast<std::size_t>(size);
        return true;
    }

    bool done() const { return offset == payload.size(); }
};

std::string point_encoding(const G1& point) { return point.serializeToHexStr(); }

bool decode_point(const std::string& encoded, G1& point, bool require_nonzero, std::string* reason = nullptr) {
    if (encoded.empty() || encoded.size() > kMaxPoint) {
        set_reason(reason, "Pedersen point encoding length is invalid");
        return false;
    }
    try {
        point.deserializeHexStr(encoded);
        if (point.serializeToHexStr() != encoded || !point.isValid() || !point.isValidOrder() || (require_nonzero && point.isZero())) {
            set_reason(reason, "Pedersen point encoding is not canonical");
            return false;
        }
    } catch (...) {
        set_reason(reason, "Pedersen point decoding failed");
        return false;
    }
    return true;
}

bool point_equal(const G1& left, const G1& right) { return point_encoding(left) == point_encoding(right); }

void add_scaled(G1& result, const G1& point, const Fr& scalar) {
    if (scalar.isZero()) return;
    if (scalar == Fr(1)) {
        result += point;
    } else if (scalar == -Fr(1)) {
        result -= point;
    } else {
        result += point * scalar;
    }
}

bool valid_generator(const G1& generator) { return generator.isValid() && generator.isValidOrder() && !generator.isZero(); }

G1 expected_generator(const std::string& domain, const char* name) {
    const std::string label = name == nullptr ? std::string() : name;
    const std::string cache_key = std::string(kBasisPrefix) + domain + "/" + label;
    const bool cacheable = name != nullptr && (label == "G" || label == "H") && valid_domain(domain);
    struct CacheEntry {
        std::string key;
        G1 point;
    };
    static std::mutex cache_mutex;
    static std::array<CacheEntry, kGeneratorCacheCapacity> cache;
    static std::size_t cache_size = 0;
    static std::size_t cache_next = 0;

    if (cacheable) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (std::size_t index = 0; index < cache_size; ++index)
            if (cache[index].key == cache_key) return cache[index].point;
    }

    G1 result;
    hashAndMapToG1(result, cache_key);
    if (cacheable) {
        std::lock_guard<std::mutex> lock(cache_mutex);

        for (std::size_t index = 0; index < cache_size; ++index)
            if (cache[index].key == cache_key) return cache[index].point;
        const std::size_t slot = cache_size < kGeneratorCacheCapacity ? cache_size++ : cache_next++ % kGeneratorCacheCapacity;
        cache[slot].key = cache_key;
        cache[slot].point = result;
    }
    return result;
}

Fr random_field() {
    Fr result;
    result.setByCSPRNG();
    return result;
}

Fr random_nonzero_field() {
    Fr result;
    do { result = random_field(); } while (result.isZero());
    return result;
}

std::string field_encoding(const Fr& value) {
    const std::string encoded = value.getStr(16);
    if (encoded.empty() || encoded.size() > 128) throw std::runtime_error("Pedersen field encoding is invalid");
    return encoded;
}

bool decode_field(const std::string& encoded, Fr& value, std::string* reason = nullptr) {
    if (encoded.empty() || encoded.size() > 128) {
        set_reason(reason, "Pedersen field encoding length is invalid");
        return false;
    }
    try {
        value.setStr(encoded, 16);
        if (field_encoding(value) != encoded) {
            set_reason(reason, "Pedersen field encoding is not canonical");
            return false;
        }
    } catch (...) {
        set_reason(reason, "Pedersen field decoding failed");
        return false;
    }
    return true;
}

bool decode_field_vector(const NativeProofMessage& message, std::size_t expected, std::vector<Fr>& values, std::string* reason) {
    if (message.kind != NativeProofMessageKind::FieldVector) {
        set_reason(reason, "Pedersen response has wrong message kind");
        return false;
    }
    Reader reader{message.payload};
    std::uint64_t count = 0;
    if (!reader.read_u64(count) || count != expected || count > kMaxOperands) {
        set_reason(reason, "Pedersen field-vector length is invalid");
        return false;
    }
    values.clear();
    values.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::string encoded;
        Fr value;
        if (!reader.read_string(encoded, 128) || !decode_field(encoded, value, reason)) {
            set_reason(reason, "Pedersen field-vector element is invalid");
            return false;
        }
        values.push_back(value);
    }
    if (!reader.done()) {
        set_reason(reason, "Pedersen field-vector has trailing bytes");
        return false;
    }
    return true;
}

bool parse_field_message(const NativeProofMessage& message, const std::string& label, Fr& value, std::string* reason) {
    if (message.kind != NativeProofMessageKind::Field || message.label != label) {
        set_reason(reason, "Pedersen scalar response message is invalid");
        return false;
    }
    return decode_field(message.payload, value, reason);
}

void append_header(std::string& payload, const std::string& kind, const PedersenBasis& basis) {
    append_string(payload, kMagic);
    append_string(payload, kind);
    append_string(payload, basis.domain);
    append_string(payload, point_encoding(basis.G));
    append_string(payload, point_encoding(basis.H));
}

struct ParsedHeader {
    std::string kind;
    std::string domain;
    G1 G;
    G1 H;
};

bool parse_header(Reader& reader, ParsedHeader& header, std::string* reason) {
    std::string magic;
    std::string g_encoded;
    std::string h_encoded;
    if (!reader.read_string(magic, sizeof(kMagic) - 1) || magic != kMagic || !reader.read_string(header.kind, 128) ||
        !reader.read_string(header.domain, kMaxDomain) || !reader.read_string(g_encoded, kMaxPoint) || !reader.read_string(h_encoded, kMaxPoint) ||
        !valid_domain(header.domain) || !decode_point(g_encoded, header.G, true, reason) || !decode_point(h_encoded, header.H, true, reason)) {
        set_reason(reason, "Pedersen relation header is malformed");
        return false;
    }
    return true;
}

bool header_matches(const ParsedHeader& header, const PedersenBasis& basis, const std::string& kind, std::string* reason) {
    if (header.kind != kind || header.domain != basis.domain || !point_equal(header.G, basis.G) || !point_equal(header.H, basis.H)) {
        set_reason(reason, "Pedersen basis or relation kind does not match");
        return false;
    }
    return true;
}

bool parse_point_message(const NativeProofMessage& message, const std::string& label, G1& point, bool require_nonzero, std::string* reason) {
    if (message.kind != NativeProofMessageKind::Data || message.label != label || !decode_point(message.payload, point, require_nonzero, reason)) {
        set_reason(reason, "Pedersen point message is invalid");
        return false;
    }
    return true;
}

bool label_free(const NativeProofSession& session, const std::vector<std::string>& labels, std::string* reason) {
    std::set<std::string> expected;
    for (const std::string& label : labels) {
        if (label.empty() || !expected.insert(label).second) {
            set_reason(reason, "Pedersen transcript labels are not unique");
            return false;
        }
    }
    for (const std::string& label : expected) {
        if (session.has_label(label)) {
            set_reason(reason, "Pedersen transcript label is already used");
            return false;
        }
    }
    return true;
}

bool session_ok(const NativeProofSession& session, std::string* reason) {
    if (!session.replay(reason)) return false;
    return true;
}

struct BlockRange {
    std::size_t start = 0;
    std::size_t end = 0;
};

bool find_unique_message(const NativeProofSession& session, const std::string& label, std::size_t& index, std::string* reason) {
    const std::vector<std::size_t>* positions = session.message_positions(label);
    if (positions == nullptr || positions->empty()) {
        set_reason(reason, "Pedersen transcript message is missing");
        return false;
    }
    if (positions->size() != 1) {
        set_reason(reason, "Pedersen transcript contains duplicate labels");
        return false;
    }
    index = positions->front();
    if (index >= session.messages().size()) {
        set_reason(reason, "Pedersen transcript message index is out of range");
        return false;
    }
    return true;
}

bool derive_challenge(const NativeProofSession& session, std::size_t challenge_index, const std::string& challenge_label, Fr& challenge, std::string* reason) {
    const std::vector<NativeProofMessage>& messages = session.messages();
    if (challenge_index >= messages.size() || messages[challenge_index].kind != NativeProofMessageKind::Challenge ||
        messages[challenge_index].label != "challenge/" + challenge_label) {
        set_reason(reason, "Pedersen challenge message is invalid");
        return false;
    }

    if (!session.replay(reason)) return false;
    return decode_field(messages[challenge_index].payload, challenge, reason);
}

std::string opening_validation_cache_key(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label,
                                         const ScalarCommitment& commitment) {
    std::string key;
    append_u64(key, session.content_version());
    append_string(key, label);
    append_string(key, basis.domain);
    append_string(key, point_encoding(basis.G));
    append_string(key, point_encoding(basis.H));
    append_string(key, commitment.domain);
    append_string(key, point_encoding(commitment.point));
    return key;
}

bool compare_string(const std::string& expected, const std::string& actual, const char* what, std::string* reason) {
    if (expected != actual) {
        set_reason(reason, std::string("Pedersen ") + what + " does not match");
        return false;
    }
    return true;
}

bool compare_field(const Fr& expected, const Fr& actual, const char* what, std::string* reason) {
    if (expected != actual) {
        set_reason(reason, std::string("Pedersen ") + what + " does not match");
        return false;
    }
    return true;
}

bool require_prior_opening(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& commitment,
                           std::size_t before, BlockRange* range, std::string* reason) {
    if (!verify_opening(session, basis, label, commitment, reason)) return false;
    const std::vector<NativeProofMessage>& messages = session.messages();
    std::size_t start = 0;
    if (!find_unique_message(session, label + "/meta", start, reason)) return false;
    const std::size_t end = start + 4;
    if (end > before || end > messages.size()) {
        set_reason(reason, "Pedersen operand opening must precede relation");
        return false;
    }
    if (range != nullptr) *range = BlockRange{start, end};
    return true;
}

bool parse_opening_block(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& commitment,
                         BlockRange& range, OpeningProof* proof, std::string* reason) {
    if (!valid_label(label)) {
        set_reason(reason, "Pedersen opening label is invalid");
        return false;
    }
    if (!basis.valid(reason) || !commitment.valid(basis, reason) || !session_ok(session, reason)) return false;
    std::size_t start = 0;
    if (!find_unique_message(session, label + "/meta", start, reason)) return false;
    const std::vector<NativeProofMessage>& messages = session.messages();
    if (start + 4 > messages.size()) {
        set_reason(reason, "Pedersen opening transcript is truncated");
        return false;
    }
    const NativeProofMessage& meta = messages[start];
    if (meta.kind != NativeProofMessageKind::Data || messages[start + 1].label != label + "/T" ||
        messages[start + 2].kind != NativeProofMessageKind::Challenge || messages[start + 3].label != label + "/response") {
        set_reason(reason, "Pedersen opening message order is invalid");
        return false;
    }
    Reader reader{meta.payload};
    ParsedHeader header;
    std::string commitment_encoded;
    if (!parse_header(reader, header, reason) || !header_matches(header, basis, "opening", reason) || !reader.read_string(commitment_encoded, kMaxPoint) ||
        !reader.done() || !compare_string(point_encoding(commitment.point), commitment_encoded, "opening commitment", reason))
        return false;
    G1 T;
    if (!parse_point_message(messages[start + 1], label + "/T", T, true, reason)) return false;
    Fr challenge;
    if (!derive_challenge(session, start + 2, label + "/challenge", challenge, reason)) return false;
    std::vector<Fr> response;
    if (!decode_field_vector(messages[start + 3], 2, response, reason)) return false;
    if (basis.G * response[0] + basis.H * response[1] != T + commitment.point * challenge) {
        set_reason(reason, "Pedersen opening Schnorr equation failed");
        return false;
    }
    if (proof != nullptr) {
        proof->commitment_message = T;
        proof->challenge = challenge;
        proof->response_value = response[0];
        proof->response_blinding = response[1];
    }
    range = BlockRange{start, start + 4};
    return true;
}

bool parse_linear_meta(const NativeProofMessage& message, const PedersenBasis& basis, const std::vector<ScalarCommitment>& operands, G1& relation_commitment,
                       std::string* reason) {
    if (message.kind != NativeProofMessageKind::Data) {
        set_reason(reason, "Pedersen linear metadata has wrong message kind");
        return false;
    }
    Reader reader{message.payload};
    ParsedHeader header;
    if (!parse_header(reader, header, reason) || !header_matches(header, basis, "linear", reason)) return false;
    std::uint64_t count = 0;
    if (!reader.read_u64(count) || count != operands.size() || count == 0 || count > kMaxOperands) {
        set_reason(reason, "Pedersen linear operand length does not match");
        return false;
    }
    for (std::size_t i = 0; i < operands.size(); ++i) {
        std::string encoded;
        if (!reader.read_string(encoded, kMaxPoint) || !compare_string(point_encoding(operands[i].point), encoded, "linear operand commitment", reason))
            return false;
    }
    std::string relation_encoded;
    if (!reader.read_string(relation_encoded, kMaxPoint) || !reader.done() || !decode_point(relation_encoded, relation_commitment, false, reason)) return false;
    return true;
}

bool parse_multiplication_block(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& A,
                                const ScalarCommitment& B, const ScalarCommitment& C, const MultiplicationLinks& links, BlockRange& range,
                                MultiplicationProof* proof, std::string* reason) {
    if (!valid_label(label) || !links.valid(reason) || !basis.valid(reason) || !A.valid(basis, reason) || !B.valid(basis, reason) || !C.valid(basis, reason) ||
        !session_ok(session, reason))
        return false;
    const std::vector<NativeProofMessage>& messages = session.messages();
    std::size_t start = 0;
    if (!find_unique_message(session, label + "/meta", start, reason)) return false;
    if (start + 5 > messages.size()) {
        set_reason(reason, "Pedersen multiplication transcript is truncated");
        return false;
    }
    if (messages[start].kind != NativeProofMessageKind::Data || messages[start + 1].label != label + "/T1" || messages[start + 2].label != label + "/T2" ||
        messages[start + 3].kind != NativeProofMessageKind::Challenge || messages[start + 4].label != label + "/response") {
        set_reason(reason, "Pedersen multiplication message order is invalid");
        return false;
    }
    Reader reader{messages[start].payload};
    ParsedHeader header;
    if (!parse_header(reader, header, reason) || !header_matches(header, basis, "multiplication", reason)) return false;
    std::string encoded;
    if (!reader.read_string(encoded, kMaxPoint) || !compare_string(point_encoding(A.point), encoded, "A commitment", reason) ||
        !reader.read_string(encoded, kMaxPoint) || !compare_string(point_encoding(B.point), encoded, "B commitment", reason) ||
        !reader.read_string(encoded, kMaxPoint) || !compare_string(point_encoding(C.point), encoded, "C commitment", reason))
        return false;
    const std::vector<std::string> expected_links{links.a_opening_label, links.b_opening_label, links.c_opening_label};
    for (const std::string& expected : expected_links) {
        if (!reader.read_string(encoded, kMaxLabel) || !compare_string(expected, encoded, "operand representation identity", reason)) return false;
    }
    if (!reader.done()) {
        set_reason(reason, "Pedersen multiplication metadata has trailing bytes");
        return false;
    }
    G1 T1;
    G1 T2;
    if (!parse_point_message(messages[start + 1], label + "/T1", T1, true, reason) ||
        !parse_point_message(messages[start + 2], label + "/T2", T2, true, reason))
        return false;
    Fr challenge;
    if (!derive_challenge(session, start + 3, label + "/challenge", challenge, reason)) return false;
    std::vector<Fr> response;
    if (!decode_field_vector(messages[start + 4], 3, response, reason)) return false;

    for (const std::pair<std::string, ScalarCommitment>& operand :
         {std::make_pair(links.a_opening_label, A), std::make_pair(links.b_opening_label, B), std::make_pair(links.c_opening_label, C)}) {
        BlockRange opening_range;
        if (!require_prior_opening(session, basis, operand.first, operand.second, start, &opening_range, reason)) return false;
    }
    if (basis.G * response[0] + basis.H * response[1] != T1 + A.point * challenge ||
        B.point * response[0] + basis.H * response[2] != T2 + C.point * challenge) {
        set_reason(reason, "Pedersen multiplication Sigma equation failed");
        return false;
    }
    if (proof != nullptr) {
        proof->first_commitment_message = T1;
        proof->second_commitment_message = T2;
        proof->challenge = challenge;
        proof->response_a = response[0];
        proof->response_a_blinding = response[1];
        proof->response_multiplication_blinding = response[2];
    }
    range = BlockRange{start, start + 5};
    return true;
}

bool parse_product3_meta(const NativeProofMessage& message, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& A,
                         const ScalarCommitment& B, const ScalarCommitment& C, const ScalarCommitment& intermediate, const ScalarCommitment& output,
                         const Product3Links& links, std::string* reason) {
    if (message.kind != NativeProofMessageKind::Data) {
        set_reason(reason, "Pedersen product3 metadata has wrong message kind");
        return false;
    }
    Reader reader{message.payload};
    ParsedHeader header;
    if (!parse_header(reader, header, reason) || !header_matches(header, basis, "product3", reason)) return false;
    const std::vector<const ScalarCommitment*> commitments{&A, &B, &C, &intermediate, &output};
    for (const ScalarCommitment* commitment : commitments) {
        std::string encoded;
        if (!reader.read_string(encoded, kMaxPoint) || !compare_string(point_encoding(commitment->point), encoded, "product3 commitment", reason)) return false;
    }
    const std::vector<std::string> sublabels{label + "/mul1", label + "/mul2"};
    for (const std::string& expected : sublabels) {
        std::string encoded;
        if (!reader.read_string(encoded, kMaxLabel) || !compare_string(expected, encoded, "product3 subrelation identity", reason)) return false;
    }
    const std::vector<std::string> link_labels{links.first.a_opening_label,  links.first.b_opening_label,  links.first.c_opening_label,
                                               links.second.a_opening_label, links.second.b_opening_label, links.second.c_opening_label};
    for (const std::string& expected : link_labels) {
        std::string encoded;
        if (!reader.read_string(encoded, kMaxLabel) || !compare_string(expected, encoded, "product3 representation identity", reason)) return false;
    }
    if (!reader.done()) {
        set_reason(reason, "Pedersen product3 metadata has trailing bytes");
        return false;
    }
    return true;
}

bool locate_product3_children(const NativeProofSession& session, const std::string& label, BlockRange& meta_range, BlockRange& first_range,
                              BlockRange& second_range, std::string* reason) {
    std::size_t meta_start = 0;
    if (!find_unique_message(session, label + "/meta", meta_start, reason)) return false;
    if (!find_unique_message(session, label + "/mul1/meta", first_range.start, reason) ||
        !find_unique_message(session, label + "/mul2/meta", second_range.start, reason))
        return false;
    first_range.end = first_range.start + 5;
    second_range.end = second_range.start + 5;
    meta_range = BlockRange{meta_start, meta_start + 1};
    const std::vector<NativeProofMessage>& messages = session.messages();
    if (first_range.end > messages.size() || second_range.end > messages.size() || first_range.start != meta_range.end ||
        second_range.start != first_range.end) {
        set_reason(reason, "Pedersen product3 child relation order is invalid");
        return false;
    }
    return true;
}

}

ProverFastPath::ProverFastPath(NativeProofSession& session, const PedersenBasis& basis) : session_(&session), domain_(basis.domain), G_(basis.G), H_(basis.H) {}

bool ProverFastPath::matches_basis(const PedersenBasis& basis) const { return domain_ == basis.domain && point_equal(G_, basis.G) && point_equal(H_, basis.H); }

bool ProverFastPath::matches(const NativeProofSession& session, const PedersenBasis& basis) const { return session_ == &session && matches_basis(basis); }

std::shared_ptr<const ProverFastPath> make_prover_fast_path(NativeProofSession& session, const PedersenBasis& basis) {
    std::string reason;
    if (!basis.valid(&reason)) throw std::invalid_argument(reason_or(&reason, "invalid Pedersen basis for prover fast path"));
    return std::shared_ptr<const ProverFastPath>(new ProverFastPath(session, basis));
}

namespace {

using FastPath = std::shared_ptr<const ProverFastPath>;

bool fast_commitment_matches(const FastPath& fast_path, const PedersenBasis& basis, const ScalarCommitment& commitment) {
    return fast_path != nullptr && fast_path->matches_basis(basis) && commitment.prover_fast_path == fast_path && commitment.domain == basis.domain;
}

bool fast_opening_matches(const FastPath& fast_path, const NativeProofSession& session, const PedersenBasis& basis, const ScalarOpening& opening) {
    return fast_path != nullptr && fast_path->matches(session, basis) && opening.prover_fast_path == fast_path &&
           fast_commitment_matches(fast_path, basis, opening.commitment);
}

FastPath common_opening_fast_path(const NativeProofSession& session, const PedersenBasis& basis, const std::vector<ScalarOpening>& openings) {
    if (openings.empty()) return nullptr;
    const FastPath candidate = openings.front().prover_fast_path;
    if (candidate == nullptr || !candidate->matches(session, basis)) return nullptr;
    for (const ScalarOpening& opening : openings)
        if (!fast_opening_matches(candidate, session, basis, opening)) return nullptr;
    return candidate;
}

bool require_prior_opening_fast(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& commitment,
                                std::size_t before, BlockRange* range, std::string* reason, const FastPath& fast_path) {
    if (!fast_commitment_matches(fast_path, basis, commitment) || !commitment.point.isValid() || !commitment.point.isValidOrder() ||
        commitment.point.isZero() || !valid_label(label)) {
        set_reason(reason, "Pedersen fast-path opening provenance is invalid");
        return false;
    }
    std::size_t start = 0;
    if (!find_unique_message(session, label + "/meta", start, reason)) return false;
    const std::vector<NativeProofMessage>& messages = session.messages();
    const std::size_t end = start + 4;
    if (end > before || end > messages.size()) {
        set_reason(reason, "Pedersen operand opening must precede relation");
        return false;
    }
    const NativeProofMessage& meta = messages[start];
    if (meta.kind != NativeProofMessageKind::Data || meta.label != label + "/meta" || messages[start + 1].kind != NativeProofMessageKind::Data ||
        messages[start + 1].label != label + "/T" || messages[start + 2].kind != NativeProofMessageKind::Challenge ||
        messages[start + 2].label != "challenge/" + label + "/challenge" || messages[start + 3].kind != NativeProofMessageKind::FieldVector ||
        messages[start + 3].label != label + "/response") {
        set_reason(reason, "Pedersen opening message order is invalid");
        return false;
    }
    Reader reader{meta.payload};
    ParsedHeader header;
    std::string encoded_commitment;
    if (!parse_header(reader, header, reason) || !header_matches(header, basis, "opening", reason) || !reader.read_string(encoded_commitment, kMaxPoint) ||
        !reader.done() || !compare_string(point_encoding(commitment.point), encoded_commitment, "opening commitment", reason))
        return false;
    if (range != nullptr) *range = BlockRange{start, end};
    return true;
}

}

bool PedersenBasis::valid(std::string* reason) const {
    ensure_field();
    if (!valid_domain(domain)) {
        set_reason(reason, "Pedersen basis domain is invalid");
        return false;
    }
    if (!valid_generator(G) || !valid_generator(H) || point_equal(G, H)) {
        set_reason(reason, "Pedersen basis generators are invalid");
        return false;
    }
    if (!point_equal(G, expected_generator(domain, "G")) || !point_equal(H, expected_generator(domain, "H"))) {
        set_reason(reason, "Pedersen basis generator derivation mismatch");
        return false;
    }
    return true;
}

PedersenBasis derive_basis(const std::string& domain) {
    ensure_field();
    if (!valid_domain(domain)) throw std::invalid_argument("invalid Pedersen basis domain");
    PedersenBasis basis;
    basis.domain = domain;
    basis.G = expected_generator(domain, "G");
    basis.H = expected_generator(domain, "H");
    if (!basis.valid()) throw std::runtime_error("derived Pedersen basis is invalid");
    return basis;
}

bool ScalarCommitment::valid(const PedersenBasis& basis, std::string* reason) const {
    ensure_field();
    if (!basis.valid(reason) || domain != basis.domain) {
        set_reason(reason, "Pedersen commitment basis domain mismatch");
        return false;
    }
    if (!point.isValid() || !point.isValidOrder()) {
        set_reason(reason, "Pedersen commitment point is invalid");
        return false;
    }
    return true;
}

bool ScalarOpening::valid(const PedersenBasis& basis, std::string* reason) const {
    if (!commitment.valid(basis, reason)) return false;
    const G1 expected = basis.G * value + basis.H * blinding;
    if (!point_equal(expected, commitment.point)) {
        set_reason(reason, "Pedersen opening does not match commitment");
        return false;
    }
    return true;
}

ScalarCommitment commit(const PedersenBasis& basis, const Fr& value, const Fr& blinding, FastPath fast_path) {
    ensure_field();
    if (fast_path != nullptr) {
        if (!fast_path->matches_basis(basis)) throw std::invalid_argument("Pedersen prover fast-path basis mismatch");
    } else if (!basis.valid()) {
        throw std::invalid_argument("invalid Pedersen basis");
    }
    ScalarCommitment result;
    result.domain = basis.domain;
    result.point.clear();
    add_scaled(result.point, basis.G, value);
    add_scaled(result.point, basis.H, blinding);
    result.prover_fast_path = fast_path;
    return result;
}

ScalarOpening commit_fresh(const PedersenBasis& basis, const Fr& value, FastPath fast_path) {
    ensure_field();
    Fr blinding = random_nonzero_field();
    ScalarOpening result{commit(basis, value, blinding, fast_path), value, blinding, fast_path};
    return result;
}

bool prove_opening(NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarOpening& opening, OpeningProof* proof,
                   std::string* reason, FastPath fast_path) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        ensure_field();
        const bool explicit_fast_path = fast_path != nullptr;
        if (fast_path == nullptr) fast_path = opening.prover_fast_path;
        const bool use_fast_path = fast_opening_matches(fast_path, session, basis, opening);
        if (explicit_fast_path && !use_fast_path) throw std::invalid_argument("Pedersen prover fast-path provenance mismatch");
        if (!valid_label(label) || ((!use_fast_path) && (!basis.valid(reason) || !opening.valid(basis, reason))) ||
            !label_free(session, {label + "/meta", label + "/T", "challenge/" + label + "/challenge", label + "/response"}, reason))
            throw std::invalid_argument(reason != nullptr ? *reason : "invalid opening");
        Fr nonce_value = random_nonzero_field();
        Fr nonce_blinding = random_nonzero_field();
        const G1 T = basis.G * nonce_value + basis.H * nonce_blinding;
        std::string meta;
        append_header(meta, "opening", basis);
        append_string(meta, point_encoding(opening.commitment.point));
        session.absorb(label + "/meta", meta);
        session.absorb(label + "/T", point_encoding(T));
        const Fr challenge = session.challenge(label + "/challenge");
        const Fr response_value = nonce_value + challenge * opening.value;
        const Fr response_blinding = nonce_blinding + challenge * opening.blinding;
        session.absorb_fields(label + "/response", {response_value, response_blinding});
        if (proof != nullptr) {
            proof->commitment_message = T;
            proof->challenge = challenge;
            proof->response_value = response_value;
            proof->response_blinding = response_blinding;
        }
        return true;
    } catch (const std::exception& error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_opening(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& commitment,
                    std::string* reason) {
    if (!valid_label(label) || !basis.valid(reason) || !commitment.valid(basis, reason)) return false;
    const std::string cache_key = opening_validation_cache_key(session, basis, label, commitment);
    if (session.has_cached_verified_opening(cache_key)) return true;
    BlockRange range;
    if (!parse_opening_block(session, basis, label, commitment, range, nullptr, reason)) return false;
    session.cache_verified_opening(cache_key);
    return true;
}

G1 linear_residual_commitment(const PedersenBasis& basis, const std::vector<ScalarCommitment>& operands, const std::vector<Fr>& coefficients,
                              const Fr& public_value, FastPath fast_path) {
    ensure_field();
    std::string reason;
    const bool use_fast_path = fast_path != nullptr;
    if ((use_fast_path && !fast_path->matches_basis(basis)) || ((!use_fast_path) && !basis.valid(&reason)) || operands.empty() ||
        operands.size() != coefficients.size() || operands.size() > kMaxOperands)
        throw std::invalid_argument(reason.empty() ? "invalid linear relation" : reason);
    G1 result;
    result.clear();
    add_scaled(result, basis.G, -public_value);
    for (std::size_t i = 0; i < operands.size(); ++i) {
        if (use_fast_path && !fast_commitment_matches(fast_path, basis, operands[i]))
            throw std::invalid_argument("Pedersen linear fast-path provenance mismatch");
        if (!use_fast_path && !operands[i].valid(basis, &reason)) throw std::invalid_argument(reason);
        add_scaled(result, operands[i].point, coefficients[i]);
    }
    return result;
}

bool prove_hidden_linear(NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const std::vector<ScalarOpening>& operands,
                         const std::vector<Fr>& coefficients, const Fr& public_value, LinearRelationProof* proof, std::string* reason, FastPath fast_path) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        ensure_field();
        const bool explicit_fast_path = fast_path != nullptr;
        if (fast_path == nullptr) fast_path = common_opening_fast_path(session, basis, operands);
        const bool use_fast_path = fast_path != nullptr;
        if (explicit_fast_path && (!fast_path->matches(session, basis) || common_opening_fast_path(session, basis, operands) != fast_path))
            throw std::invalid_argument("Pedersen linear fast-path provenance mismatch");
        if (!valid_label(label) || operands.empty() || operands.size() != coefficients.size() || operands.size() > kMaxOperands ||
            ((!use_fast_path) && !basis.valid(reason)))
            throw std::invalid_argument("invalid Pedersen linear relation shape");
        G1 operand_residual;
        Fr value_residual = -public_value;
        Fr blinding_residual = Fr(0);
        std::vector<ScalarCommitment> public_operands;
        public_operands.reserve(operands.size());
        for (std::size_t i = 0; i < operands.size(); ++i) {
            if (!use_fast_path && !operands[i].valid(basis, reason)) throw std::invalid_argument(reason_or(reason, "invalid opening witness"));
            public_operands.push_back(operands[i].commitment);
            value_residual += coefficients[i] * operands[i].value;
            blinding_residual += coefficients[i] * operands[i].blinding;
        }
        if (!value_residual.isZero()) throw std::invalid_argument("Pedersen linear relation witness is false");
        operand_residual = linear_residual_commitment(basis, public_operands, coefficients, public_value, fast_path);
        if (operand_residual != basis.H * blinding_residual) throw std::invalid_argument("Pedersen linear residual blinding mismatch");
        if (!label_free(session,
                        {label + "/meta", label + "/coefficients", label + "/public", label + "/T", "challenge/" + label + "/challenge", label + "/response"},
                        reason))
            throw std::invalid_argument(reason_or(reason, "invalid linear relation labels"));
        std::string meta;
        append_header(meta, "linear", basis);
        append_u64(meta, static_cast<std::uint64_t>(public_operands.size()));
        for (const ScalarCommitment& operand : public_operands) append_string(meta, point_encoding(operand.point));
        append_string(meta, point_encoding(operand_residual));
        session.absorb(label + "/meta", meta);
        session.absorb_fields(label + "/coefficients", coefficients);
        session.absorb_field(label + "/public", public_value);
        const Fr nonce = random_nonzero_field();
        const G1 T = basis.H * nonce;
        session.absorb(label + "/T", point_encoding(T));
        const Fr challenge = session.challenge(label + "/challenge");
        const Fr response = nonce + challenge * blinding_residual;
        session.absorb_field(label + "/response", response);
        if (proof != nullptr) {
            proof->relation_commitment = operand_residual;
            proof->commitment_message = T;
            proof->challenge = challenge;
            proof->response_blinding = response;
        }
        return true;
    } catch (const std::exception& error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_hidden_linear(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label,
                          const std::vector<ScalarCommitment>& operands, const std::vector<Fr>& coefficients, const Fr& public_value, std::string* reason) {
    try {
        ensure_field();
        if (!valid_label(label) || !basis.valid(reason) || operands.empty() || operands.size() != coefficients.size() || operands.size() > kMaxOperands)
            return false;
        for (const ScalarCommitment& operand : operands)
            if (!operand.valid(basis, reason)) return false;
        if (!session_ok(session, reason)) return false;
        std::size_t start = 0;
        if (!find_unique_message(session, label + "/meta", start, reason)) return false;
        const std::vector<NativeProofMessage>& messages = session.messages();
        if (start + 6 > messages.size() || messages[start + 1].label != label + "/coefficients" || messages[start + 2].label != label + "/public" ||
            messages[start + 3].label != label + "/T" || messages[start + 4].kind != NativeProofMessageKind::Challenge ||
            messages[start + 5].label != label + "/response") {
            set_reason(reason, "Pedersen linear message order is invalid");
            return false;
        }
        G1 relation_commitment;
        if (!parse_linear_meta(messages[start], basis, operands, relation_commitment, reason)) return false;
        std::vector<Fr> encoded_coefficients;
        if (!decode_field_vector(messages[start + 1], coefficients.size(), encoded_coefficients, reason)) return false;
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            if (!compare_field(coefficients[i], encoded_coefficients[i], "linear coefficient", reason)) return false;
        Fr encoded_public;
        if (!parse_field_message(messages[start + 2], label + "/public", encoded_public, reason) ||
            !compare_field(public_value, encoded_public, "linear public value", reason))
            return false;
        const G1 expected_residual = linear_residual_commitment(basis, operands, coefficients, public_value);
        if (!point_equal(expected_residual, relation_commitment)) {
            set_reason(reason, "Pedersen linear residual commitment mismatch");
            return false;
        }
        G1 T;
        if (!parse_point_message(messages[start + 3], label + "/T", T, true, reason)) return false;
        Fr challenge;
        if (!derive_challenge(session, start + 4, label + "/challenge", challenge, reason)) return false;
        Fr response;
        if (!parse_field_message(messages[start + 5], label + "/response", response, reason)) return false;
        if (basis.H * response != T + relation_commitment * challenge) {
            set_reason(reason, "Pedersen hidden-linear Schnorr equation failed");
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool MultiplicationLinks::valid(std::string* reason) const {
    if (!valid_label(a_opening_label) || !valid_label(b_opening_label) || !valid_label(c_opening_label) || a_opening_label == b_opening_label ||
        a_opening_label == c_opening_label || b_opening_label == c_opening_label) {
        set_reason(reason, "Pedersen multiplication representation links are invalid");
        return false;
    }
    return true;
}

bool prove_multiplication(NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarOpening& a, const ScalarOpening& b,
                          const ScalarOpening& c, const MultiplicationLinks& links, MultiplicationProof* proof, std::string* reason, FastPath fast_path) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        ensure_field();
        const bool explicit_fast_path = fast_path != nullptr;
        if (fast_path == nullptr) fast_path = common_opening_fast_path(session, basis, {a, b, c});
        const bool use_fast_path = fast_path != nullptr;
        if (explicit_fast_path && (!fast_path->matches(session, basis) || common_opening_fast_path(session, basis, {a, b, c}) != fast_path))
            throw std::invalid_argument("Pedersen multiplication fast-path provenance mismatch");
        if (!valid_label(label) || !links.valid(reason) ||
            ((!use_fast_path) && (!basis.valid(reason) || !a.valid(basis, reason) || !b.valid(basis, reason) || !c.valid(basis, reason))) ||
            c.value != a.value * b.value)
            throw std::invalid_argument("invalid Pedersen multiplication witness");
        if (!label_free(session, {label + "/meta", label + "/T1", label + "/T2", "challenge/" + label + "/challenge", label + "/response"}, reason))
            throw std::invalid_argument(reason_or(reason, "invalid multiplication labels"));
        BlockRange ignored;
        const auto require = [&](const std::string& opening_label, const ScalarCommitment& opening_commitment) {
            return fast_path != nullptr ?
                       require_prior_opening_fast(session, basis, opening_label, opening_commitment, checkpoint, &ignored, reason, fast_path) :
                       require_prior_opening(session, basis, opening_label, opening_commitment, checkpoint, &ignored, reason);
        };
        if (!require(links.a_opening_label, a.commitment) || !require(links.b_opening_label, b.commitment) || !require(links.c_opening_label, c.commitment))
            throw std::invalid_argument(reason_or(reason, "invalid multiplication operand opening"));
        Fr d = random_nonzero_field();
        Fr s1 = random_nonzero_field();
        Fr s2 = random_nonzero_field();
        const G1 T1 = basis.G * d + basis.H * s1;
        const G1 T2 = b.commitment.point * d + basis.H * s2;
        std::string meta;
        append_header(meta, "multiplication", basis);
        append_string(meta, point_encoding(a.commitment.point));
        append_string(meta, point_encoding(b.commitment.point));
        append_string(meta, point_encoding(c.commitment.point));
        append_string(meta, links.a_opening_label);
        append_string(meta, links.b_opening_label);
        append_string(meta, links.c_opening_label);
        session.absorb(label + "/meta", meta);
        session.absorb(label + "/T1", point_encoding(T1));
        session.absorb(label + "/T2", point_encoding(T2));
        const Fr challenge = session.challenge(label + "/challenge");
        const Fr response_a = d + challenge * a.value;
        const Fr response_a_blinding = s1 + challenge * a.blinding;
        const Fr response_multiplication_blinding = s2 + challenge * (c.blinding - a.value * b.blinding);
        session.absorb_fields(label + "/response", {response_a, response_a_blinding, response_multiplication_blinding});
        if (proof != nullptr) {
            proof->first_commitment_message = T1;
            proof->second_commitment_message = T2;
            proof->challenge = challenge;
            proof->response_a = response_a;
            proof->response_a_blinding = response_a_blinding;
            proof->response_multiplication_blinding = response_multiplication_blinding;
        }
        return true;
    } catch (const std::exception& error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_multiplication(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& A,
                           const ScalarCommitment& B, const ScalarCommitment& C, const MultiplicationLinks& links, std::string* reason) {
    BlockRange range;
    return parse_multiplication_block(session, basis, label, A, B, C, links, range, nullptr, reason);
}

bool Product3Links::valid(std::string* reason) const {
    if (!first.valid(reason) || !second.valid(reason) || first.c_opening_label != second.a_opening_label || first.a_opening_label == second.b_opening_label ||
        first.a_opening_label == second.c_opening_label || first.b_opening_label == second.b_opening_label || first.b_opening_label == second.c_opening_label) {
        set_reason(reason, "Pedersen product3 intermediate identity is invalid");
        return false;
    }
    return true;
}

bool prove_product3(NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarOpening& a, const ScalarOpening& b,
                    const ScalarOpening& c, const ScalarOpening& intermediate, const ScalarOpening& output, const Product3Links& links, Product3Proof* proof,
                    std::string* reason, FastPath fast_path) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        ensure_field();
        const bool explicit_fast_path = fast_path != nullptr;
        if (fast_path == nullptr) fast_path = common_opening_fast_path(session, basis, {a, b, c, intermediate, output});
        const bool use_fast_path = fast_path != nullptr;
        if (explicit_fast_path &&
            (!fast_path->matches(session, basis) || common_opening_fast_path(session, basis, {a, b, c, intermediate, output}) != fast_path))
            throw std::invalid_argument("Pedersen product3 fast-path provenance mismatch");
        if (!valid_label(label) || !links.valid(reason) ||
            ((!use_fast_path) && (!basis.valid(reason) || !a.valid(basis, reason) || !b.valid(basis, reason) || !c.valid(basis, reason) ||
                                  !intermediate.valid(basis, reason) || !output.valid(basis, reason))) ||
            intermediate.value != a.value * b.value || output.value != intermediate.value * c.value)
            throw std::invalid_argument("invalid Pedersen product3 witness");
        if (links.first.c_opening_label != links.second.a_opening_label) throw std::invalid_argument("product3 intermediate opening labels differ");
        if (!label_free(session,
                        {label + "/meta", label + "/mul1/meta", label + "/mul1/T1", label + "/mul1/T2", "challenge/" + label + "/mul1/challenge",
                         label + "/mul1/response", label + "/mul2/meta", label + "/mul2/T1", label + "/mul2/T2", "challenge/" + label + "/mul2/challenge",
                         label + "/mul2/response"},
                        reason))
            throw std::invalid_argument(reason_or(reason, "invalid product3 labels"));
        const std::vector<std::pair<std::string, const ScalarOpening*>> openings{{links.first.a_opening_label, &a},
                                                                                 {links.first.b_opening_label, &b},
                                                                                 {links.first.c_opening_label, &intermediate},
                                                                                 {links.second.b_opening_label, &c},
                                                                                 {links.second.c_opening_label, &output}};
        for (const auto& entry : openings) {
            BlockRange ignored;
            const bool prior_ok =
                fast_path != nullptr ?
                    require_prior_opening_fast(session, basis, entry.first, entry.second->commitment, checkpoint, &ignored, reason, fast_path) :
                    require_prior_opening(session, basis, entry.first, entry.second->commitment, checkpoint, &ignored, reason);
            if (!prior_ok) throw std::invalid_argument(reason_or(reason, "invalid product3 operand opening"));
        }
        std::string meta;
        append_header(meta, "product3", basis);
        append_string(meta, point_encoding(a.commitment.point));
        append_string(meta, point_encoding(b.commitment.point));
        append_string(meta, point_encoding(c.commitment.point));
        append_string(meta, point_encoding(intermediate.commitment.point));
        append_string(meta, point_encoding(output.commitment.point));
        append_string(meta, label + "/mul1");
        append_string(meta, label + "/mul2");
        for (const std::string& link : {links.first.a_opening_label, links.first.b_opening_label, links.first.c_opening_label, links.second.a_opening_label,
                                        links.second.b_opening_label, links.second.c_opening_label})
            append_string(meta, link);
        session.absorb(label + "/meta", meta);
        MultiplicationProof first_proof;
        if (!prove_multiplication(session, basis, label + "/mul1", a, b, intermediate, links.first, &first_proof, reason, fast_path))
            throw std::invalid_argument(reason_or(reason, "invalid product3 first multiplication"));
        MultiplicationProof second_proof;
        if (!prove_multiplication(session, basis, label + "/mul2", intermediate, c, output, links.second, &second_proof, reason, fast_path))
            throw std::invalid_argument(reason_or(reason, "invalid product3 second multiplication"));
        if (proof != nullptr) {
            proof->intermediate_commitment = intermediate.commitment.point;
            proof->output_commitment = output.commitment.point;
        }
        return true;
    } catch (const std::exception& error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool verify_product3(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& A,
                     const ScalarCommitment& B, const ScalarCommitment& C, const ScalarCommitment& intermediate, const ScalarCommitment& output,
                     const Product3Links& links, std::string* reason) {
    try {
        ensure_field();
        if (!valid_label(label) || !basis.valid(reason) || !links.valid(reason) || !A.valid(basis, reason) || !B.valid(basis, reason) ||
            !C.valid(basis, reason) || !intermediate.valid(basis, reason) || !output.valid(basis, reason) || !session_ok(session, reason))
            return false;
        std::size_t meta_start = 0;
        if (!find_unique_message(session, label + "/meta", meta_start, reason)) return false;
        const std::vector<NativeProofMessage>& messages = session.messages();
        if (meta_start >= messages.size() || !parse_product3_meta(messages[meta_start], basis, label, A, B, C, intermediate, output, links, reason))
            return false;
        BlockRange meta_range;
        BlockRange first_range;
        BlockRange second_range;
        if (!locate_product3_children(session, label, meta_range, first_range, second_range, reason)) return false;
        if (!verify_multiplication(session, basis, label + "/mul1", A, B, intermediate, links.first, reason) ||
            !verify_multiplication(session, basis, label + "/mul2", intermediate, C, output, links.second, reason))
            return false;
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

ScalarOpening public_affine(const PedersenBasis& basis, const ScalarOpening& input, const Fr& scalar, const Fr& offset) {
    ensure_field();
    std::string reason;
    if (!input.valid(basis, &reason)) throw std::invalid_argument(reason);
    const Fr value = scalar * input.value + offset;
    return commit_fresh(basis, value);
}

bool prove_public_affine(NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarOpening& input,
                         const ScalarOpening& output, const Fr& scalar, const Fr& offset, LinearRelationProof* proof, std::string* reason) {
    return prove_hidden_linear(session, basis, label, {output, input}, {Fr(1), -scalar}, offset, proof, reason);
}

bool verify_public_affine(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& input,
                          const ScalarCommitment& output, const Fr& scalar, const Fr& offset, std::string* reason) {
    return verify_hidden_linear(session, basis, label, {output, input}, {Fr(1), -scalar}, offset, reason);
}

}
}
