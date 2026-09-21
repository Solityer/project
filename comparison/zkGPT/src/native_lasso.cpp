#include "native_lasso.hpp"

#include "hyrax.hpp"
#include "native_range.hpp"
#include "native_masked_sumcheck.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
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

bool valid_identifier(const std::string &value) {
    if (value.empty() || value.size() > 128) return false;
    for (char c : value) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_' || c == '.'))
            return false;
    }
    return true;
}

bool valid_digest(const std::string &value) {
    if (value.size() != 64) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    }
    return true;
}

bool valid_basis_identifier(const std::string &value) {
    if (value.empty() || value.size() > 128) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F') || c == '-' || c == '_'))
            return false;
    }
    return true;
}

bool valid_point(const G1 &point) {
    return point.isValid() && !point.isZero() && point.isValidOrder();
}

void append_u64(std::string &out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_string(std::string &out, const std::string &value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

std::string hash_fixed(const std::string &encoded) {
    F digest;
    digest.setHashOf(encoded.data(), encoded.size());
    std::string value = digest.getStr(16);
    if (value.empty()) value = "0";
    if (value.size() > 64) throw std::runtime_error("native Lasso digest overflow");
    return std::string(64 - value.size(), '0') + value;
}

std::string encode_tokens(const std::vector<std::string> &tokens) {
    std::string payload;
    append_u64(payload, static_cast<std::uint64_t>(tokens.size()));
    for (const std::string &token : tokens) append_string(payload, token);
    return payload;
}

bool decode_tokens(const std::string &payload, std::size_t expected,
                   std::vector<std::string> &tokens) {
    tokens.clear();
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
    tokens.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size > (1ULL << 20) ||
            size > payload.size() - offset)
            return false;
        tokens.emplace_back(payload.substr(offset, static_cast<std::size_t>(size)));
        offset += static_cast<std::size_t>(size);
    }
    return offset == payload.size();
}

bool decode_field(const std::string &encoded, F &value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
    } catch (...) {
        return false;
    }
    return value.getStr(16) == encoded;
}

bool decode_point(const std::string &encoded, G1 &value) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        value.deserializeHexStr(encoded);
    } catch (...) {
        return false;
    }
    return value.serializeToHexStr() == encoded && valid_point(value);
}

bool decode_vector(const NativeProofMessage *message, std::size_t expected,
                   std::vector<F> &values) {
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
    if (!read_u64(count) || count != expected) return false;
    values.clear();
    values.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > 128 ||
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

const NativeProofMessage *consume(const NativeProofSession &session,
                                  std::size_t &position,
                                  NativeProofMessageKind kind,
                                  const std::string &label,
                                  std::string *reason) {
    if (position >= session.messages().size() ||
        session.messages()[position].kind != kind ||
        session.messages()[position].label != label) {
        set_reason(reason, "native Lasso grammar mismatch at " + label);
        return nullptr;
    }
    return &session.messages()[position++];
}

std::size_t find_message(const NativeProofSession &session,
                         const std::string &label,
                         NativeProofMessageKind kind,
                         std::string *reason) {
    std::size_t result = session.messages().size();
    for (std::size_t i = 0; i < session.messages().size(); ++i) {
        if (session.messages()[i].label != label ||
            session.messages()[i].kind != kind)
            continue;
        if (result != session.messages().size()) {
            set_reason(reason, "duplicate native Lasso message: " + label);
            return session.messages().size();
        }
        result = i;
    }
    if (result == session.messages().size())
        set_reason(reason, "missing native Lasso message: " + label);
    return result;
}

std::vector<F> mle_basis(const std::vector<F> &point) {
    if (point.empty() || point.size() >= sizeof(std::size_t) * 8)
        throw std::invalid_argument("native Lasso point dimension is invalid");
    std::vector<F> basis(1ULL << point.size());
    basis[0] = F(1) - point[0];
    basis[1] = point[0];
    for (std::size_t bit = 1; bit < point.size(); ++bit) {
        const std::size_t old_size = 1ULL << bit;
        for (std::size_t index = old_size; index-- > 0;) {
            const F previous = basis[index];
            basis[index] = previous * (F(1) - point[bit]);
            basis[index + old_size] = previous * point[bit];
        }
    }
    return basis;
}

NativeLassoFamilyPublic public_identity(const NativeLassoFamilyProver &family) {
    NativeLassoFamilyPublic result = family;
    result.matrix_generators = family.matrix_generators;
    return result;
}

bool validate_public_family(const NativeLassoFamilyPublic &family,
                            std::string *reason) {
    if (!valid_identifier(family.family_id) || !valid_identifier(family.table_id) ||
        !valid_identifier(family.resource_id) ||
        !valid_identifier(family.registration_id)) {
        set_reason(reason, "native Lasso family identity is not canonical");
        return false;
    }
    if (!power_of_two(family.table.size()) || family.table.size() > (1ULL << 16) ||
        !power_of_two(family.padded_queries) ||
        family.padded_queries > (1ULL << 16) ||
        family.logical_queries > family.padded_queries ||
        family.padding_index >= family.table.size() ||
        family.padding_value != family.table[family.padding_index]) {
        set_reason(reason, "native Lasso public table/query dimensions are invalid");
        return false;
    }
    if (family.matrix_generators.size() !=
        family.padded_queries * family.table.size()) {
        set_reason(reason, "native Lasso M_f generator domain is invalid");
        return false;
    }
    if (family.segment_offset > (1ULL << 20) ||
        family.segment_offset + family.padded_queries < family.segment_offset) {
        set_reason(reason, "native Lasso L_qry segment offset overflows");
        return false;
    }
    for (const G1 &generator : family.matrix_generators) {
        if (!valid_point(generator)) {
            set_reason(reason, "native Lasso M_f generator is invalid");
            return false;
        }
    }
    return true;
}

bool validate_prover_family(const NativeLassoFamilyProver &family,
                            std::string *reason) {
    if (!validate_public_family(family, reason) ||
        family.queries.size() != family.padded_queries ||
        family.matrix.size() != family.padded_queries * family.table.size()) {
        set_reason(reason, "native Lasso prover witness dimensions are invalid");
        return false;
    }
    const std::size_t rows = family.padded_queries;
    const std::size_t cols = family.table.size();
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t selected = cols;
        for (std::size_t col = 0; col < cols; ++col) {
            const F &entry = family.matrix[row + col * rows];
            if (!(entry.isZero() || entry == F(1))) {
                set_reason(reason, "native Lasso M_f contains a non-bit entry");
                return false;
            }
            if (entry == F(1)) {
                if (selected != cols) {
                    set_reason(reason, "native Lasso M_f row is not one-hot");
                    return false;
                }
                selected = col;
            }
        }
        if (selected == cols || family.queries[row] != family.table[selected]) {
            set_reason(reason, "native Lasso query is not selected by M_f");
            return false;
        }
        if (row >= family.logical_queries &&
            (selected != family.padding_index ||
             family.queries[row] != family.padding_value)) {
            set_reason(reason, "native Lasso padding row is not canonical");
            return false;
        }
    }
    return true;
}

bool validate_query(const NativeLassoQueryProver &query, std::string *reason) {
    if (!valid_point(query.G) || !power_of_two(query.values.size()) ||
        query.values.size() < 2 || query.values.size() > (1ULL << 20) ||
        query.generators.size() != query.values.size() ||
        !valid_point(query.commitment) ||
        (!query.basis_id.empty() && !valid_basis_identifier(query.basis_id)) ||
        (!query.statement_l_digest.empty() && !valid_digest(query.statement_l_digest))) {
        set_reason(reason, "native Lasso L_qry setup is invalid");
        return false;
    }
    for (const G1 &generator : query.generators) {
        if (!valid_point(generator)) {
            set_reason(reason, "native Lasso L_qry generator is invalid");
            return false;
        }
    }
    if (query.has_commitment_blinding &&
        (!valid_point(query.blinding_generator) || query.basis_id.empty() ||
         query.blinding_generator !=
             hyrax::derive_pedersen_blinding_generator(query.basis_id))) {
        set_reason(reason, "native Lasso L_qry blinding identity is invalid");
        return false;
    }
    return true;
}

bool validate_query_public(const NativeLassoQueryPublic &query,
                           std::size_t expected_size, std::string *reason) {
    if (!valid_point(query.G) || !power_of_two(expected_size) || expected_size < 2 ||
        query.generators.size() != expected_size || !valid_point(query.commitment) ||
        (!query.basis_id.empty() && !valid_basis_identifier(query.basis_id)) ||
        (!query.statement_l_digest.empty() && !valid_digest(query.statement_l_digest))) {
        set_reason(reason, "native Lasso public L_qry setup is invalid");
        return false;
    }
    for (const G1 &generator : query.generators) {
        if (!valid_point(generator)) {
            set_reason(reason, "native Lasso public L_qry generator is invalid");
            return false;
        }
    }
    return true;
}

std::string family_prefix(const NativeLassoFamilyPublic &family) {
    return "lasso/" + family.family_id;
}

// The commitment payload needs no query generator identity.  This helper
// kept separate from commitment_payload so the family matrix setup can be
// authenticated without accidentally binding it to a prover-only object.
std::string commitment_payload(const NativeLassoFamilyProver &family,
                               const G1 &commitment,
                               const G1 &G_for_digest) {
    return encode_tokens({
        family.family_id,
        family.table_id,
        family.resource_id,
        family.registration_id,
        std::to_string(family.padded_queries),
        std::to_string(family.table.size()),
        std::to_string(family.segment_offset),
        std::to_string(family.logical_queries),
        std::to_string(family.padding_index),
        family.padding_value.getStr(16),
        commitment.serializeToHexStr(),
        native_lasso_generator_digest(G_for_digest, family.matrix_generators),
        native_lasso_table_digest(family)
    });
}

G1 commit_all_ones(const std::vector<G1> &generators) {
    if (generators.empty() || generators.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native Lasso complement domain is invalid");
    std::vector<G1> generator_copy = generators;
    std::vector<F> ones(generators.size(), F_ONE);
    return perdersen_commit(generator_copy.data(), ones.data(),
                            static_cast<int>(generators.size()));
}

bool parse_size(const std::string &encoded, std::size_t &value) {
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(encoded, &consumed, 10);
        if (consumed != encoded.size() || parsed > std::numeric_limits<std::size_t>::max())
            return false;
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_commitment(const NativeProofMessage &message,
                      const NativeLassoFamilyPublic &expected,
                      const G1 &G, G1 &commitment, std::string *reason) {
    std::vector<std::string> fields;
    if (message.kind != NativeProofMessageKind::Data ||
        !decode_tokens(message.payload, 13, fields)) {
        set_reason(reason, "malformed native Lasso LC_f commitment");
        return false;
    }
    if (fields[0] != expected.family_id || fields[1] != expected.table_id ||
        fields[2] != expected.resource_id || fields[3] != expected.registration_id) {
        set_reason(reason, "native Lasso LC_f identity mismatch");
        return false;
    }
    std::size_t rows = 0, cols = 0, offset = 0, logical = 0, padding = 0;
    if (!parse_size(fields[4], rows) || !parse_size(fields[5], cols) ||
        !parse_size(fields[6], offset) || !parse_size(fields[7], logical) ||
        !parse_size(fields[8], padding) || rows != expected.padded_queries ||
        cols != expected.table.size() || offset != expected.segment_offset ||
        logical != expected.logical_queries || padding != expected.padding_index) {
        set_reason(reason, "native Lasso LC_f shape mismatch");
        return false;
    }
    F padding_value;
    if (!decode_field(fields[9], padding_value) || padding_value != expected.padding_value ||
        !decode_point(fields[10], commitment) ||
        fields[11] != native_lasso_generator_digest(G, expected.matrix_generators) ||
        fields[12] != native_lasso_table_digest(expected)) {
        set_reason(reason, "native Lasso LC_f authenticated identity mismatch");
        return false;
    }
    return true;
}

bool read_challenge(const NativeProofSession &session, std::size_t index,
                    const std::string &label, F &value, std::string *reason) {
    if (index >= session.messages().size() ||
        session.messages()[index].kind != NativeProofMessageKind::Challenge ||
        session.messages()[index].label != label ||
        !decode_field(session.messages()[index].payload, value)) {
        set_reason(reason, "native Lasso challenge grammar mismatch at " + label);
        return false;
    }
    return true;
}

bool collect_challenges(const NativeProofSession &session,
                        const std::string &label, std::size_t rounds,
                        std::vector<F> &challenges, std::string *reason) {
    challenges.clear();
    for (std::size_t round = 0; round < rounds; ++round) {
        const std::size_t index = find_message(
            session, "challenge/" + label + "/r/" + std::to_string(round),
            NativeProofMessageKind::Challenge, reason);
        if (index == session.messages().size()) return false;
        F value;
        if (!read_challenge(session, index,
                           "challenge/" + label + "/r/" + std::to_string(round),
                           value, reason))
            return false;
        challenges.push_back(value);
    }
    return true;
}

bool check_family_order(const std::vector<NativeLassoFamilyPublic> &families,
                        std::string *reason) {
    if (families.empty()) {
        set_reason(reason, "native Lasso proof has no families");
        return false;
    }
    std::string previous;
    for (const NativeLassoFamilyPublic &family : families) {
        if (!validate_public_family(family, reason)) return false;
        if (!previous.empty() && family.family_id <= previous) {
            set_reason(reason, "native Lasso families are not canonically ordered");
            return false;
        }
        previous = family.family_id;
    }
    return true;
}

bool check_statement_identity(const NativeProofSession &session,
                              const std::vector<NativeLassoFamilyPublic> &families,
                              const G1 &query_commitment, std::size_t query_size,
                              const std::string &query_statement_l_digest,
                              std::string *reason) {
    if (!session.valid(reason)) return false;
    std::string adapter_tables_digest;
    std::string adapter_resources_digest;
    bool has_adapter_tables_digest = false;
    bool has_adapter_resources_digest = false;
    for (const NativeLassoFamilyPublic &family : families) {
        const bool has_tables = !family.statement_tables_digest.empty();
        const bool has_resources = !family.statement_resources_digest.empty();
        if (has_tables != has_resources) {
            set_reason(reason, "native Lasso adapter statement identity is incomplete");
            return false;
        }
        if (has_tables) {
            if (!valid_digest(family.statement_tables_digest) ||
                !valid_digest(family.statement_resources_digest)) {
                set_reason(reason, "native Lasso adapter statement identity is non-canonical");
                return false;
            }
            if (!has_adapter_tables_digest) {
                adapter_tables_digest = family.statement_tables_digest;
                adapter_resources_digest = family.statement_resources_digest;
                has_adapter_tables_digest = true;
                has_adapter_resources_digest = true;
            } else if (adapter_tables_digest != family.statement_tables_digest ||
                       adapter_resources_digest != family.statement_resources_digest) {
                set_reason(reason, "native Lasso family statement identities disagree");
                return false;
            }
        }
    }
    const std::string expected_tables = has_adapter_tables_digest
                                            ? adapter_tables_digest
                                            : native_lasso_tables_digest(families);
    const std::string expected_resources = has_adapter_resources_digest
                                               ? adapter_resources_digest
                                               : native_lasso_resources_digest(families);
    if (session.statement().tables_digest != expected_tables) {
        set_reason(reason, "native Lasso table registry is not statement-bound");
        return false;
    }
    if (session.statement().resources_digest != expected_resources) {
        set_reason(reason, "native Lasso resource/registration identity is not statement-bound");
        return false;
    }
    const std::string expected_l_digest = query_statement_l_digest.empty()
                                              ? native_lasso_query_digest(query_commitment,
                                                                          query_size)
                                              : query_statement_l_digest;
    if (!valid_digest(expected_l_digest) ||
        session.statement().l_digest != expected_l_digest) {
        set_reason(reason, "native Lasso L_qry commitment is not statement-bound");
        return false;
    }
    return true;
}

} // namespace

const char *NativeLassoTerminalOpeningLabels::l_qry() {
    return "lasso/i-final/open";
}

const char *NativeLassoTerminalOpeningLabels::w_reserved() {
    return "reserved/open/W";
}

const char *NativeLassoTerminalOpeningLabels::aux_reserved() {
    return "reserved/open/Aux";
}

std::string native_lasso_table_digest(const NativeLassoFamilyPublic &family) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-table-v1");
    append_string(encoded, family.family_id);
    append_string(encoded, family.table_id);
    append_u64(encoded, family.table.size());
    for (const F &value : family.table) append_string(encoded, value.getStr(16));
    return hash_fixed(encoded);
}

std::string native_lasso_tables_digest(
    const std::vector<NativeLassoFamilyPublic> &families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-tables-v1");
    append_u64(encoded, families.size());
    for (const NativeLassoFamilyPublic &family : families)
        append_string(encoded, native_lasso_table_digest(family));
    return hash_fixed(encoded);
}

std::string native_lasso_resources_digest(
    const std::vector<NativeLassoFamilyPublic> &families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-resources-v1");
    append_u64(encoded, families.size());
    for (const NativeLassoFamilyPublic &family : families) {
        append_string(encoded, family.family_id);
        append_string(encoded, family.table_id);
        append_string(encoded, family.resource_id);
        append_string(encoded, family.registration_id);
        append_u64(encoded, family.segment_offset);
        append_u64(encoded, family.logical_queries);
        append_u64(encoded, family.padded_queries);
        append_u64(encoded, family.padding_index);
        append_string(encoded, family.padding_value.getStr(16));
    }
    return hash_fixed(encoded);
}

std::string native_lasso_query_digest(const G1 &commitment, std::size_t size) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-query-v1");
    append_u64(encoded, size);
    append_string(encoded, commitment.serializeToHexStr());
    return hash_fixed(encoded);
}

std::string native_lasso_generator_digest(const G1 &G,
                                          const std::vector<G1> &generators) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-generators-v1");
    append_string(encoded, G.serializeToHexStr());
    append_u64(encoded, generators.size());
    for (const G1 &generator : generators)
        append_string(encoded, generator.serializeToHexStr());
    return hash_fixed(encoded);
}

// Legacy clear transcript kept in a private namespace for source-level
// migration/debug comparison only.  The public native_lasso_* API below is
// the masked implementation; no masked_full caller can resolve these names.
namespace native_lasso_legacy_clear {
bool native_lasso_prove_bundle(
    NativeProofSession &session,
    std::vector<NativeLassoFamilyProver> &families,
    NativeLassoQueryProver &query,
    NativeLassoBundleResult *result,
    std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (families.empty()) return false;
        // The private legacy namespace retains the historical clear
        // commitment construction.  The public masked path requires its
        // caller-supplied Pedersen commitment and never reaches this block.
        query.commitment = perdersen_commit(query.generators.data(), query.values.data(),
                                            static_cast<int>(query.values.size()));
        if (!validate_query(query, reason)) return false;
        std::vector<NativeLassoFamilyPublic> identities;
        identities.reserve(families.size());
        for (NativeLassoFamilyProver &family : families) {
            if (!validate_prover_family(family, reason)) return false;
            identities.push_back(public_identity(family));
        }
        if (!check_family_order(identities, reason)) return false;
        if (!check_statement_identity(session, identities, query.commitment,
                                      query.values.size(), query.statement_l_digest,
                                      reason))
            return false;
        for (const NativeLassoFamilyPublic &family : identities) {
            if (family.segment_offset + family.padded_queries > query.values.size()) {
                set_reason(reason, "native Lasso family exceeds L_qry segment");
                return false;
            }
        }

        NativeLassoBundleResult local;
        local.family_commitments.reserve(families.size());
        // All LC_f messages are emitted before the first challenge.
        for (NativeLassoFamilyProver &family : families) {
            G1 commitment = perdersen_commit(
                family.matrix_generators.data(), family.matrix.data(),
                static_cast<int>(family.matrix.size()));
            local.family_commitments.push_back(commitment);
            session.absorb(family_prefix(family) + "/commitment",
                           commitment_payload(family, commitment, query.G));
        }

        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            NativeLassoFamilyProver &family = families[family_index];
            const std::string prefix = family_prefix(family);
            const std::size_t row_rounds = log2_power(family.padded_queries);
            std::vector<F> row_point;
            row_point.reserve(row_rounds);
            for (std::size_t round = 0; round < row_rounds; ++round)
                row_point.push_back(session.challenge(
                    prefix + "/a-query/r/" + std::to_string(round)));
            local.family_query_points.push_back(row_point);
            const std::vector<F> row_basis = mle_basis(row_point);
            const std::size_t rows = family.padded_queries;
            const std::size_t cols = family.table.size();

            // Bind LC_f to a bit-valued M_f before using it for lookup
            // membership.  The product Sumcheck proves M(M-1)=0 at a
            // transcript-derived point and the Hyrax opening binds that
            // terminal value to the committed matrix.
            std::vector<F> matrix_minus_one = family.matrix;
            for (F &value : matrix_minus_one) value -= F(1);
            session.absorb_field(prefix + "/b-boolean/claim", F_ZERO);
            NativeRangeResult booleanity;
            if (!native_range_prove_product(
                    session, prefix + "/b-boolean/sum", family.matrix,
                    matrix_minus_one, F_ZERO, &booleanity, reason))
                return false;
            if (booleanity.terminal_right != booleanity.terminal_left - F(1))
                throw std::runtime_error("native Lasso M_f boolean terminal mismatch");
            // The product Sumcheck challenges are the evaluation point at
            // which its terminal values are defined.  A separate point
            // derived before the Sumcheck would leave the Hyrax opening
            // proving M at one point while the Sumcheck terminal claims M at
            // another, causing the honest prover's terminal equation to
            // fail.  Reuse the native Sumcheck point so the opening binds
            // exactly the committed M_f evaluation checked by the product
            // relation.
            std::vector<F> boolean_basis = mle_basis(booleanity.challenges);
            try {
                prove_dot_product_with_session(
                    local.family_commitments[family_index],
                    query.G * booleanity.terminal_left, boolean_basis.data(),
                    family.matrix_generators.data(), query.G, family.matrix.data(),
                    booleanity.terminal_left,
                    static_cast<int>(family.matrix.size()), session,
                    prefix + "/c-boolean-open", false);
            } catch (const std::exception &error) {
                throw std::runtime_error(prefix + "/c-boolean-open: " + error.what());
            }

            // The second factor of M_f(M_f-1) is not public.  Bind its
            // terminal to the deterministic commitment LC_f-Commit(1),
            // rather than accepting only the scalar relation
            // right=left-1.  This keeps the booleanity Sumcheck sound while
            // retaining the native Hyrax opening path.
            const G1 complement_commit =
                local.family_commitments[family_index] -
                commit_all_ones(family.matrix_generators);
            try {
                prove_dot_product_with_session(
                    complement_commit, query.G * booleanity.terminal_right,
                    boolean_basis.data(), family.matrix_generators.data(), query.G,
                    matrix_minus_one.data(), booleanity.terminal_right,
                    static_cast<int>(family.matrix.size()), session,
                    prefix + "/c-boolean-complement-open", false);
            } catch (const std::exception &error) {
                throw std::runtime_error(prefix +
                                         "/c-boolean-complement-open: " + error.what());
            }

            // A second independent Sumcheck enforces one selected table row
            // for the committed matrix.  Its terminal is opened at the same
            // LC_f commitment, so a prover cannot use a different matrix for
            // the membership and row-shape checks.
            std::vector<F> onehot_point;
            onehot_point.reserve(row_rounds);
            for (std::size_t round = 0; round < row_rounds; ++round)
                onehot_point.push_back(session.challenge(
                    prefix + "/d-onehot/r/" + std::to_string(round)));
            const std::vector<F> onehot_row_basis = mle_basis(onehot_point);
            std::vector<F> onehot_left(cols, F_ZERO);
            for (std::size_t col = 0; col < cols; ++col)
                for (std::size_t row = 0; row < rows; ++row)
                    onehot_left[col] += onehot_row_basis[row] *
                                        family.matrix[row + col * rows];
            session.absorb_field(prefix + "/d-onehot/claim", F_ONE);
            NativeRangeResult onehot;
            if (!native_range_prove_deg1(
                    session, prefix + "/d-onehot/sum", onehot_left, F_ONE,
                    &onehot, reason))
                return false;
            std::vector<F> onehot_matrix_point;
            onehot_matrix_point.reserve(onehot_point.size() + onehot.challenges.size());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot_point.begin(),
                                       onehot_point.end());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot.challenges.begin(),
                                       onehot.challenges.end());
            std::vector<F> onehot_basis = mle_basis(onehot_matrix_point);
            try {
                prove_dot_product_with_session(
                    local.family_commitments[family_index], query.G * onehot.terminal,
                    onehot_basis.data(), family.matrix_generators.data(), query.G,
                    family.matrix.data(), onehot.terminal,
                    static_cast<int>(family.matrix.size()), session,
                    prefix + "/e-onehot-open", false);
            } catch (const std::exception &error) {
                throw std::runtime_error(prefix + "/e-onehot-open: " + error.what());
            }

            std::vector<F> left(cols, F_ZERO);
            for (std::size_t col = 0; col < cols; ++col)
                for (std::size_t row = 0; row < rows; ++row)
                    left[col] += row_basis[row] * family.matrix[row + col * rows];
            F ell = 0;
            for (std::size_t col = 0; col < cols; ++col)
                ell += left[col] * family.table[col];
            session.absorb_field(prefix + "/a-query/claim", ell);

            NativeRangeResult lookup;
            if (!native_range_prove_product(session, prefix + "/f-lookup", left,
                                            family.table, ell, &lookup, reason))
                return false;
            const std::vector<F> y_point = lookup.challenges;
            std::vector<F> matrix_point;
            matrix_point.reserve(row_point.size() + y_point.size());
            matrix_point.insert(matrix_point.end(), row_point.begin(), row_point.end());
            matrix_point.insert(matrix_point.end(), y_point.begin(), y_point.end());
            std::vector<F> matrix_basis = mle_basis(matrix_point);
            F direct_matrix_eval = 0;
            for (std::size_t i = 0; i < matrix_basis.size(); ++i)
                direct_matrix_eval += matrix_basis[i] * family.matrix[i];
            if (direct_matrix_eval != lookup.terminal_left)
                throw std::runtime_error(prefix + "/g-matrix-open direct eval mismatch: " +
                                         direct_matrix_eval.getStr(16) + " vs " +
                                         lookup.terminal_left.getStr(16));
            const G1 &lc = local.family_commitments[family_index];
            try {
                prove_dot_product_with_session(
                    lc, query.G * lookup.terminal_left,
                    matrix_basis.data(), family.matrix_generators.data(), query.G,
                    family.matrix.data(), lookup.terminal_left,
                    static_cast<int>(family.matrix.size()), session,
                    prefix + "/g-matrix-open", false);
            } catch (const std::exception &error) {
                throw std::runtime_error(prefix + "/g-matrix-open: " + error.what());
            }

            std::vector<F> selector_basis(query.values.size(), F_ZERO);
            for (std::size_t row = 0; row < rows; ++row)
                selector_basis[family.segment_offset + row] = row_basis[row];
            F segment_eval = 0;
            for (std::size_t row = 0; row < rows; ++row) {
                if (query.values[family.segment_offset + row] != family.queries[row])
                    throw std::runtime_error("native Lasso query/input segment mismatch");
                segment_eval += row_basis[row] * family.queries[row];
            }
            session.absorb_fields(prefix + "/h-input/claim", {ell, segment_eval});
            try {
                prove_dot_product_with_session(
                    query.commitment, query.G * segment_eval,
                    selector_basis.data(), query.generators.data(), query.G,
                    query.values.data(), segment_eval,
                    static_cast<int>(query.values.size()), session,
                    prefix + "/h-input-open", true);
            } catch (const std::exception &error) {
                throw std::runtime_error(prefix + "/h-input-open: " + error.what());
            }
        }

        const std::size_t final_rounds = log2_power(query.values.size());
        local.final_query_point.reserve(final_rounds);
        for (std::size_t round = 0; round < final_rounds; ++round)
            local.final_query_point.push_back(
                session.challenge("lasso/i-final/r/" + std::to_string(round)));
        std::vector<F> final_basis = mle_basis(local.final_query_point);
        F final_eval = 0;
        for (std::size_t i = 0; i < query.values.size(); ++i)
            final_eval += final_basis[i] * query.values[i];
        local.final_l_evaluation = final_eval;
        session.absorb_field("lasso/i-final/claim", final_eval);
        try {
            prove_dot_product_with_session(
                query.commitment, query.G * final_eval, final_basis.data(),
                query.generators.data(), query.G, query.values.data(), final_eval,
                static_cast<int>(query.values.size()), session,
                NativeLassoTerminalOpeningLabels::l_qry(), true);
        } catch (const std::exception &error) {
            throw std::runtime_error(std::string(NativeLassoTerminalOpeningLabels::l_qry()) +
                                     ": " + error.what());
        }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}
} // namespace native_lasso_legacy_clear

namespace {

bool chunk_lasso_family_valid(const NativeLassoChunkFamilyPublic &family,
                              std::string *reason) {
    if (!valid_identifier(family.family_id) ||
        !valid_identifier(family.table_id) ||
        !valid_identifier(family.resource_id) ||
        !valid_identifier(family.registration_id) ||
        !power_of_two(family.table.size()) || family.table.size() > (1ULL << 16) ||
        !power_of_two(family.padded_queries) || family.padded_queries == 0 ||
        family.logical_queries > family.padded_queries ||
        family.padding_index >= family.table.size() ||
        family.padding_value != family.table[family.padding_index] ||
        !power_of_two(family.chunk_size) || family.chunk_size < 2 ||
        family.chunk_size > (1ULL << 16) ||
        family.padded_queries % family.chunk_size != 0) {
        set_reason(reason, "native Lasso chunk family shape is invalid");
        return false;
    }
    return true;
}

NativeLassoFamilyPublic chunk_legacy_public(
    const NativeLassoChunkFamilyPublic &family) {
    NativeLassoFamilyPublic result;
    result.family_id = family.family_id;
    result.table_id = family.table_id;
    result.resource_id = family.resource_id;
    result.registration_id = family.registration_id;
    result.table = family.table;
    result.segment_offset = family.segment_offset;
    result.logical_queries = family.logical_queries;
    result.padded_queries = family.padded_queries;
    result.padding_index = family.padding_index;
    result.padding_value = family.padding_value;
    result.statement_tables_digest = family.statement_tables_digest;
    result.statement_resources_digest = family.statement_resources_digest;
    return result;
}

std::string chunk_lasso_table_digest(
    const NativeLassoChunkFamilyPublic &family) {
    return native_lasso_table_digest(chunk_legacy_public(family));
}

std::string chunk_lasso_tables_digest(
    const std::vector<NativeLassoChunkFamilyPublic> &families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-chunk-tables-v1");
    for (const NativeLassoChunkFamilyPublic &family : families)
        append_string(encoded, chunk_lasso_table_digest(family));
    return hash_fixed(encoded);
}

std::string chunk_lasso_resources_digest(
    const std::vector<NativeLassoChunkFamilyPublic> &families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-chunk-resources-v1");
    for (const NativeLassoChunkFamilyPublic &family : families) {
        append_string(encoded, family.family_id);
        append_string(encoded, family.table_id);
        append_string(encoded, family.resource_id);
        append_string(encoded, family.registration_id);
        append_u64(encoded, family.segment_offset);
        append_u64(encoded, family.logical_queries);
        append_u64(encoded, family.padded_queries);
        append_u64(encoded, family.padding_index);
        append_string(encoded, family.padding_value.getStr(16));
        // Statement fields are wire annotations populated by the producer;
        // including them here would make the resource digest recursive and
        // would prevent an independent verifier from reconstructing it from
        // public metadata alone.
        append_string(encoded, chunk_lasso_table_digest(family));
    }
    return hash_fixed(encoded);
}

std::string chunk_lasso_root(const NativeLassoChunkBundleWire &wire) {
    std::string encoded;
    append_string(encoded, wire.schema);
    append_u64(encoded, wire.version);
    append_string(encoded, wire.tables_digest);
    append_string(encoded, wire.resources_digest);
    for (const NativeLassoChunkFamilyWire &family : wire.families) {
        const NativeLassoChunkFamilyPublic &meta = family.public_metadata;
        append_string(encoded, meta.family_id);
        append_string(encoded, meta.table_id);
        append_string(encoded, meta.resource_id);
        append_string(encoded, meta.registration_id);
        append_u64(encoded, meta.segment_offset);
        append_u64(encoded, meta.logical_queries);
        append_u64(encoded, meta.padded_queries);
        append_u64(encoded, meta.table.size());
        append_u64(encoded, meta.chunk_size);
        append_string(encoded, meta.padding_value.getStr(16));
        append_string(encoded, family.row_point_digest);
        append_string(encoded, family.query.root_digest);
        for (const hyrax::ChunkedMleWire &column : family.matrix_columns)
            append_string(encoded, column.root_digest);
        append_string(encoded, family.lookup_terminal_commitment_hex);
        append_string(encoded, family.onehot_terminal_commitment_hex);
    }
    return hash_fixed(encoded);
}

bool scalar_commitment_from_hex(
    const std::string &encoded,
    const zkgpt::native_pedersen::PedersenBasis &basis,
    zkgpt::native_pedersen::ScalarCommitment &result,
    std::string *reason) {
    G1 point;
    if (!decode_point(encoded, point)) {
        set_reason(reason, "native Lasso chunk scalar commitment is malformed");
        return false;
    }
    result.domain = basis.domain;
    result.point = point;
    return result.valid(basis, reason);
}

bool challenge_point(const NativeProofSession &session, const std::string &label,
                     std::size_t rounds, std::vector<F> &point,
                     std::string *reason) {
    point.clear();
    point.reserve(rounds);
    for (std::size_t index = 0; index < rounds; ++index) {
        const std::string challenge_label =
            "challenge/" + label + "/" + std::to_string(index);
        const std::size_t position = find_message(
            session, challenge_label, NativeProofMessageKind::Challenge, reason);
        F challenge;
        if (position == session.messages().size() ||
            !decode_field(session.messages()[position].payload, challenge)) {
            set_reason(reason, "native Lasso chunk point challenge is malformed");
            return false;
        }
        point.push_back(challenge);
    }
    return true;
}

} // namespace

bool native_lasso_prove_chunked_bundle(
    NativeProofSession &session,
    std::vector<NativeLassoChunkFamilyProver> &families,
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    const std::vector<F> *global_point,
    NativeLassoChunkBundleProverResult *result,
    std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (families.empty() || !scalar_basis.valid(reason))
            throw std::invalid_argument("native Lasso chunk bundle is empty or unbound");
        std::sort(families.begin(), families.end(),
                  [](const NativeLassoChunkFamilyProver &left,
                     const NativeLassoChunkFamilyProver &right) {
                      return left.family_id < right.family_id;
                  });
        std::vector<NativeLassoChunkFamilyPublic> public_families;
        public_families.reserve(families.size());
        for (NativeLassoChunkFamilyProver &family : families) {
            if (!chunk_lasso_family_valid(family, reason) || !family.query_value)
                throw std::invalid_argument(reason != nullptr && !reason->empty()
                                                ? *reason
                                                : "native Lasso chunk query provider is empty");
            public_families.push_back(family);
        }
        const std::string tables_digest = chunk_lasso_tables_digest(public_families);
        const std::string resources_digest = chunk_lasso_resources_digest(public_families);
        NativeLassoChunkBundleProverResult local;
        local.wire.tables_digest = tables_digest;
        local.wire.resources_digest = resources_digest;
        std::vector<zkgpt::native_pedersen::ScalarOpening> final_operands;
        std::vector<Fr> final_coefficients;
        for (std::size_t family_index = 0; family_index < families.size();
             ++family_index) {
            NativeLassoChunkFamilyProver &family = families[family_index];
            NativeLassoChunkFamilyWire family_wire;
            family_wire.public_metadata = family;
            family_wire.public_metadata.statement_tables_digest = tables_digest;
            family_wire.public_metadata.statement_resources_digest = resources_digest;
            const std::string prefix = "lasso/chunked/" + family.family_id;
            session.absorb(prefix + "/schedule",
                           family.family_id + "|" + family.table_id + "|" +
                               family.resource_id + "|" + family.registration_id +
                               "|" + std::to_string(family.segment_offset) +
                               "|" + std::to_string(family.logical_queries) +
                               "|" + std::to_string(family.padded_queries) +
                               "|" + std::to_string(family.chunk_size) +
                               "|" + tables_digest + "|" + resources_digest);
            const std::size_t row_rounds = log2_power(family.padded_queries);
            std::vector<F> row_point;
            row_point.reserve(row_rounds);
            if (global_point != nullptr) {
                if (global_point->size() < row_rounds)
                    throw std::invalid_argument("native Lasso global point is too short");
                row_point.assign(global_point->begin(),
                                global_point->begin() + row_rounds);
                session.absorb_fields(prefix + "/row-point", row_point);
            } else {
                for (std::size_t round = 0; round < row_rounds; ++round)
                    row_point.push_back(session.challenge(
                        prefix + "/row-point/" + std::to_string(round)));
            }
            family_wire.row_point_digest = hash_fixed(encode_tokens({
                "zkgpt-native-lasso-row-point-v1", family.family_id,
                [&]() { std::string value; for (const F &x : row_point) append_string(value, x.getStr(16)); return value; }()}));
            hyrax::ChunkedMleProverResult query_result;
            if (!hyrax::prove_chunked_mle_opening(
                    session, prefix + "/query", family.padded_queries,
                    family.padded_queries, family.chunk_size, row_point,
                    [&](std::uint64_t index) {
                        return index < family.logical_queries
                                   ? family.query_value(index)
                                   : family.padding_value;
                    },
                    scalar_basis, query_result, reason))
                throw std::runtime_error(reason != nullptr ? *reason
                                                             : "native Lasso query chunk opening failed");
            family_wire.query = query_result.wire;
            final_operands.push_back(query_result.aggregate_opening);
            Fr final_weight = F_ONE;
            if (global_point != nullptr) {
                const std::size_t row_bits = log2_power(family.padded_queries);
                const std::size_t block = family.segment_offset / family.padded_queries;
                for (std::size_t bit = row_bits; bit < global_point->size(); ++bit) {
                    const bool set = ((block >> (bit - row_bits)) & 1U) != 0;
                    final_weight *= set ? (*global_point)[bit]
                                        : (F_ONE - (*global_point)[bit]);
                }
            }
            final_coefficients.push_back(final_weight);

            std::vector<zkgpt::native_pedersen::ScalarOpening> matrix_operands;
            std::vector<Fr> matrix_coefficients;
            family_wire.matrix_columns.reserve(family.table.size());
            for (std::size_t column = 0; column < family.table.size(); ++column) {
                hyrax::ChunkedMleProverResult column_result;
                const std::string column_label = prefix + "/matrix/" +
                                                  std::to_string(column);
                if (!hyrax::prove_chunked_mle_opening(
                        session, column_label, family.padded_queries,
                        family.padded_queries, family.chunk_size, row_point,
                        [&](std::uint64_t row) {
                            if (row >= family.logical_queries)
                                return family.table[column] == family.padding_value
                                           ? F_ONE : F_ZERO;
                            return family.query_value(row) == family.table[column]
                                       ? F_ONE : F_ZERO;
                        },
                        scalar_basis, column_result, reason))
                    throw std::runtime_error(reason != nullptr ? *reason
                                                                 : "native Lasso matrix chunk opening failed");
                family_wire.matrix_columns.push_back(column_result.wire);
                matrix_operands.push_back(column_result.aggregate_opening);
                matrix_coefficients.push_back(family.table[column]);
            }
            // Lookup terminal: sum_col table[col] * M_col(r) = L_qry(r).
            matrix_operands.push_back(query_result.aggregate_opening);
            matrix_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::prove_hidden_linear(
                    session, scalar_basis, prefix + "/lookup-link",
                    matrix_operands, matrix_coefficients, F_ZERO, nullptr,
                    reason))
                return false;
            family_wire.lookup_terminal_commitment_hex =
                query_result.aggregate_opening.commitment.point.serializeToHexStr();

            // One-hot terminal: sum_col M_col(r) = 1.  The matrix itself is
            // generated from the canonical table/query provider, while the
            // relation is still checked by an independent Pedersen link.
            const zkgpt::native_pedersen::ScalarOpening onehot_opening =
                zkgpt::native_pedersen::commit_fresh(scalar_basis, F_ONE);
            std::vector<zkgpt::native_pedersen::ScalarOpening> onehot_operands;
            std::vector<Fr> onehot_coefficients;
            for (std::size_t column = 0; column < family_wire.matrix_columns.size();
                 ++column) {
                zkgpt::native_pedersen::ScalarCommitment commitment;
                if (!scalar_commitment_from_hex(
                        family_wire.matrix_columns[column].aggregate_commitment_hex,
                        scalar_basis, commitment, reason))
                    return false;
                // The prover retains the matching opening in matrix_operands;
                // it is kept in the same canonical column order.
                onehot_operands.push_back(matrix_operands[column]);
                onehot_coefficients.push_back(F_ONE);
            }
            onehot_operands.push_back(onehot_opening);
            onehot_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::prove_hidden_linear(
                    session, scalar_basis, prefix + "/onehot-link",
                    onehot_operands, onehot_coefficients, F_ZERO, nullptr,
                    reason))
                return false;
            family_wire.onehot_terminal_commitment_hex =
                onehot_opening.commitment.point.serializeToHexStr();
            local.wire.families.push_back(std::move(family_wire));
        }
        local.final_l_evaluation = F_ZERO;
        if (global_point != nullptr) {
            const std::size_t global_bits = global_point->size();
            for (std::size_t family_index = 0; family_index < families.size();
                 ++family_index) {
                const NativeLassoChunkFamilyProver &family = families[family_index];
                const std::size_t row_bits = log2_power(family.padded_queries);
                const std::size_t block = family.segment_offset / family.padded_queries;
                Fr weight = F_ONE;
                for (std::size_t bit = row_bits; bit < global_bits; ++bit) {
                    const bool set = ((block >> (bit - row_bits)) & 1U) != 0;
                    weight *= set ? (*global_point)[bit]
                                 : (F_ONE - (*global_point)[bit]);
                }
                local.final_l_evaluation +=
                    weight * final_operands[family_index].value;
            }
        } else {
            for (const zkgpt::native_pedersen::ScalarOpening &opening : final_operands)
                local.final_l_evaluation += opening.value;
        }
        local.final_opening = zkgpt::native_pedersen::commit_fresh(
            scalar_basis, local.final_l_evaluation);
        final_operands.push_back(local.final_opening);
        final_coefficients.push_back(-F_ONE);
        if (!zkgpt::native_pedersen::prove_hidden_linear(
                session, scalar_basis, "lasso/chunked/final-link", final_operands,
                final_coefficients, F_ZERO, nullptr, reason))
            return false;
        local.wire.final_commitment_hex =
            local.final_opening.commitment.point.serializeToHexStr();
        local.wire.root_digest = chunk_lasso_root(local.wire);
        session.absorb("lasso/chunked/root", local.wire.root_digest);
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_lasso_verify_chunked_bundle(
    const NativeProofSession &session,
    const std::vector<NativeLassoChunkFamilyPublic> &families,
    const NativeLassoChunkBundleWire &wire,
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    const std::vector<F> *global_point,
    std::string *reason) {
    try {
        if (wire.schema != "zkgpt-native-lasso-chunked-v1" ||
            wire.version != kNativeLassoChunkVersion ||
            families.empty() || wire.families.size() != families.size() ||
            !scalar_basis.valid(reason)) {
            set_reason(reason, "native Lasso chunk wire schema is invalid");
            return false;
        }
        std::vector<NativeLassoChunkFamilyPublic> expected = families;
        std::sort(expected.begin(), expected.end(),
                  [](const NativeLassoChunkFamilyPublic &left,
                     const NativeLassoChunkFamilyPublic &right) {
                      return left.family_id < right.family_id;
                  });
        const std::string tables_digest = chunk_lasso_tables_digest(expected);
        const std::string resources_digest = chunk_lasso_resources_digest(expected);
        if (wire.tables_digest != tables_digest ||
            wire.resources_digest != resources_digest ||
            wire.root_digest != chunk_lasso_root(wire)) {
            set_reason(reason, "native Lasso chunk root or table digest mismatch");
            return false;
        }
        std::vector<zkgpt::native_pedersen::ScalarCommitment> final_operands;
        std::vector<Fr> final_coefficients;
        for (std::size_t family_index = 0; family_index < expected.size();
             ++family_index) {
            const NativeLassoChunkFamilyPublic &family = expected[family_index];
            const NativeLassoChunkFamilyWire &family_wire = wire.families[family_index];
            if (!chunk_lasso_family_valid(family, reason) ||
                family_wire.public_metadata.family_id != family.family_id ||
                family_wire.public_metadata.table_id != family.table_id ||
                family_wire.public_metadata.segment_offset != family.segment_offset ||
                family_wire.public_metadata.logical_queries != family.logical_queries ||
                family_wire.public_metadata.padded_queries != family.padded_queries ||
                family_wire.public_metadata.chunk_size != family.chunk_size ||
                family_wire.matrix_columns.size() != family.table.size()) {
                set_reason(reason, "native Lasso chunk family metadata mismatch");
                return false;
            }
            const std::string prefix = "lasso/chunked/" + family.family_id;
            std::vector<F> row_point;
            const std::size_t row_rounds = log2_power(family.padded_queries);
            if (global_point != nullptr) {
                if (global_point->size() < row_rounds) {
                    set_reason(reason, "native Lasso global verifier point is too short");
                    return false;
                }
                row_point.assign(global_point->begin(),
                                 global_point->begin() + row_rounds);
                const std::size_t position = find_message(
                    session, prefix + "/row-point",
                    NativeProofMessageKind::FieldVector, reason);
                if (position == session.messages().size()) return false;
                std::vector<F> encoded;
                if (!decode_vector(&session.messages()[position], row_rounds,
                                   encoded) || encoded != row_point) {
                    set_reason(reason, "native Lasso global row point binding mismatch");
                    return false;
                }
            } else if (!challenge_point(session, prefix + "/row-point",
                                        row_rounds, row_point, reason)) {
                return false;
            }
            const std::string expected_row_digest = hash_fixed(encode_tokens({
                "zkgpt-native-lasso-row-point-v1", family.family_id,
                [&]() { std::string value; for (const F &x : row_point) append_string(value, x.getStr(16)); return value; }()}));
            if (family_wire.row_point_digest != expected_row_digest)
                return false;
            if (!hyrax::verify_chunked_mle_opening(
                    session, family_wire.query, prefix + "/query",
                    family.padded_queries, family.padded_queries,
                    family.chunk_size, row_point, scalar_basis, reason))
                return false;
            zkgpt::native_pedersen::ScalarCommitment query_commitment;
            if (!scalar_commitment_from_hex(
                    family_wire.query.aggregate_commitment_hex, scalar_basis,
                    query_commitment, reason))
                return false;
            final_operands.push_back(query_commitment);
            Fr final_weight = F_ONE;
            if (global_point != nullptr) {
                const std::size_t row_bits = log2_power(family.padded_queries);
                const std::size_t block = family.segment_offset / family.padded_queries;
                for (std::size_t bit = row_bits; bit < global_point->size(); ++bit) {
                    const bool set = ((block >> (bit - row_bits)) & 1U) != 0;
                    final_weight *= set ? (*global_point)[bit]
                                        : (F_ONE - (*global_point)[bit]);
                }
            }
            final_coefficients.push_back(final_weight);
            std::vector<zkgpt::native_pedersen::ScalarCommitment> matrix_commitments;
            matrix_commitments.reserve(family_wire.matrix_columns.size());
            for (std::size_t column = 0; column < family_wire.matrix_columns.size();
                 ++column) {
                const std::string column_label = prefix + "/matrix/" +
                                                  std::to_string(column);
                if (!hyrax::verify_chunked_mle_opening(
                        session, family_wire.matrix_columns[column], column_label,
                        family.padded_queries, family.padded_queries,
                        family.chunk_size, row_point, scalar_basis, reason))
                    return false;
                zkgpt::native_pedersen::ScalarCommitment commitment;
                if (!scalar_commitment_from_hex(
                        family_wire.matrix_columns[column].aggregate_commitment_hex,
                        scalar_basis, commitment, reason))
                    return false;
                matrix_commitments.push_back(commitment);
            }
            std::vector<zkgpt::native_pedersen::ScalarCommitment> lookup_operands =
                matrix_commitments;
            std::vector<Fr> lookup_coefficients = family.table;
            lookup_operands.push_back(query_commitment);
            lookup_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::verify_hidden_linear(
                    session, scalar_basis, prefix + "/lookup-link",
                    lookup_operands, lookup_coefficients, F_ZERO, reason))
                return false;
            zkgpt::native_pedersen::ScalarCommitment onehot_commitment;
            if (!scalar_commitment_from_hex(
                    family_wire.onehot_terminal_commitment_hex, scalar_basis,
                    onehot_commitment, reason))
                return false;
            std::vector<zkgpt::native_pedersen::ScalarCommitment> onehot_operands =
                matrix_commitments;
            std::vector<Fr> onehot_coefficients(matrix_commitments.size(), F_ONE);
            onehot_operands.push_back(onehot_commitment);
            onehot_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::verify_hidden_linear(
                    session, scalar_basis, prefix + "/onehot-link",
                    onehot_operands, onehot_coefficients, F_ZERO, reason))
                return false;
        }
        zkgpt::native_pedersen::ScalarCommitment final_commitment;
        if (!scalar_commitment_from_hex(wire.final_commitment_hex, scalar_basis,
                                        final_commitment, reason))
            return false;
        final_operands.push_back(final_commitment);
        final_coefficients.push_back(-F_ONE);
        return zkgpt::native_pedersen::verify_hidden_linear(
            session, scalar_basis, "lasso/chunked/final-link", final_operands,
            final_coefficients, F_ZERO, reason);
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

namespace {

struct MaskedFamilyState {
    hyrax::PedersenMleCommitment commitment;
    std::string basis_id;
};

std::string query_basis_id(const NativeLassoQueryProver &query) {
    if (!query.basis_id.empty()) return query.basis_id;
    return hash_fixed(encode_tokens({
        "zkgpt-native-lasso-query-basis-v2",
        query.G.serializeToHexStr(),
        native_lasso_generator_digest(query.G, query.generators)}));
}

std::string query_basis_id(const NativeLassoQueryPublic &query) {
    if (!query.basis_id.empty()) return query.basis_id;
    return hash_fixed(encode_tokens({
        "zkgpt-native-lasso-query-basis-v2",
        query.G.serializeToHexStr(),
        native_lasso_generator_digest(query.G, query.generators)}));
}

std::string family_basis_id(const NativeProofSession &session,
                            const NativeLassoFamilyPublic &family) {
    return hash_fixed(encode_tokens({
        "zkgpt-native-lasso-matrix-basis-v2", session.statement_digest(),
        family.family_id, family.table_id, family.resource_id,
        family.registration_id,
        native_lasso_generator_digest(session.statement().basis_digest.empty()
                                           ? family.matrix_generators.front()
                                           : family.matrix_generators.front(),
                                       family.matrix_generators)}));
}

bool valid_basis_id(const std::string &value) {
    if (value.empty() || value.size() > 128) return false;
    for (char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F') || c == '-' || c == '_'))
            return false;
    return true;
}

hyrax::PedersenMleCommitment query_commitment_from(
    const NativeLassoQueryProver &query) {
    const std::string basis_id = query_basis_id(query);
    hyrax::PedersenMleCommitment result;
    result.commitment = query.commitment;
    result.basis_id = basis_id;
    result.blinding_generator = hyrax::derive_pedersen_blinding_generator(basis_id);
    result.blinding = query.has_commitment_blinding ? query.commitment_blinding : F_ZERO;
    if (query.has_commitment_blinding &&
        valid_point(query.blinding_generator) &&
        query.blinding_generator != result.blinding_generator)
        throw std::invalid_argument("native Lasso query blinding identity mismatch");
    return result;
}

hyrax::PedersenMleCommitment query_commitment_from(
    const NativeLassoQueryPublic &query) {
    const std::string basis_id = query_basis_id(query);
    hyrax::PedersenMleCommitment result;
    result.commitment = query.commitment;
    result.basis_id = basis_id;
    result.blinding_generator = hyrax::derive_pedersen_blinding_generator(basis_id);
    result.blinding = F_ZERO;
    return result;
}

zkgpt::native_pedersen::PedersenBasis lasso_scalar_basis(
    const NativeProofSession &session) {
    return zkgpt::native_pedersen::derive_basis(
        "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
}

zkgpt::native_pedersen::ScalarOpening public_scalar_opening(
    const zkgpt::native_pedersen::PedersenBasis &basis, const F &value) {
    zkgpt::native_pedersen::ScalarOpening result;
    result.commitment = zkgpt::native_pedersen::commit(basis, value, F_ZERO);
    result.value = value;
    result.blinding = F_ZERO;
    return result;
}

bool make_eval_opening(
    const zkgpt::native_pedersen::PedersenBasis &basis,
    const std::vector<F> &point, const std::vector<F> &values,
    zkgpt::native_pedersen::ScalarOpening &opening, std::string *reason) {
    return hyrax::make_mle_evaluation_opening(basis, point, values, opening,
                                              reason);
}

std::vector<F> fold_linear(std::vector<F> values, const F &challenge) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0)
        throw std::invalid_argument("native masked Lasso linear domain is invalid");
    const std::size_t half = values.size() / 2;
    std::vector<F> result(half);
    for (std::size_t index = 0; index < half; ++index)
        result[index] = values[2 * index] +
                        challenge * (values[2 * index + 1] - values[2 * index]);
    return result;
}

void append_linear_round(NativeMaskedSumcheckProver &prover,
                         const std::vector<F> &values,
                         std::vector<F> &next, std::string *reason,
                         F *challenge_out = nullptr) {
    if (values.size() < 2 || (values.size() & (values.size() - 1)) != 0)
        throw std::invalid_argument("native masked Lasso linear round domain is invalid");
    F p0 = F_ZERO, p1 = F_ZERO;
    for (std::size_t index = 0; index < values.size() / 2; ++index) {
        p0 += values[2 * index];
        p1 += values[2 * index + 1];
    }
    F challenge;
    if (!prover.append_round({p0, p1}, &challenge, reason))
        throw std::runtime_error(reason != nullptr ? *reason :
                                 "masked Lasso linear round failed");
    next = fold_linear(values, challenge);
    if (challenge_out != nullptr) *challenge_out = challenge;
}

std::vector<F> fold_product(std::vector<F> left, std::vector<F> right,
                             const F &challenge, std::vector<F> &next_right) {
    if (left.empty() || left.size() != right.size() ||
        (left.size() & (left.size() - 1)) != 0)
        throw std::invalid_argument("native masked Lasso product domain is invalid");
    const std::size_t half = left.size() / 2;
    std::vector<F> next_left(half);
    next_right.resize(half);
    for (std::size_t index = 0; index < half; ++index) {
        next_left[index] = left[2 * index] +
                           challenge * (left[2 * index + 1] - left[2 * index]);
        next_right[index] = right[2 * index] +
                            challenge * (right[2 * index + 1] - right[2 * index]);
    }
    return next_left;
}

void append_product_round(NativeMaskedSumcheckProver &prover,
                          const std::vector<F> &left,
                          const std::vector<F> &right,
                          std::vector<F> &next_left,
                          std::vector<F> &next_right, std::string *reason,
                          F *challenge_out = nullptr) {
    if (left.size() < 2 || left.size() != right.size() ||
        (left.size() & (left.size() - 1)) != 0)
        throw std::invalid_argument("native masked Lasso product round domain is invalid");
    F p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
    for (std::size_t index = 0; index < left.size() / 2; ++index) {
        const F l0 = left[2 * index], l1 = left[2 * index + 1];
        const F r0 = right[2 * index], r1 = right[2 * index + 1];
        p0 += l0 * r0;
        p1 += l1 * r1;
        p2 += (l0 + F(2) * (l1 - l0)) *
              (r0 + F(2) * (r1 - r0));
    }
    F challenge;
    if (!prover.append_round({p0, p1, p2}, &challenge, reason))
        throw std::runtime_error(reason != nullptr ? *reason :
                                 "masked Lasso product round failed");
    next_left = fold_product(left, right, challenge, next_right);
    if (challenge_out != nullptr) *challenge_out = challenge;
}

F mle_eval(const std::vector<F> &values, const std::vector<F> &point) {
    if (values.empty() || values.size() != (std::size_t(1) << point.size()))
        throw std::invalid_argument("native masked Lasso MLE dimensions are invalid");
    std::vector<F> current = values;
    for (const F &challenge : point) current = fold_linear(current, challenge);
    return current.front();
}

std::vector<F> row_mle(const std::vector<F> &matrix, std::size_t rows,
                       std::size_t cols, const std::vector<F> &row_point) {
    if (matrix.size() != rows * cols || rows != (std::size_t(1) << row_point.size()))
        throw std::invalid_argument("native masked Lasso row MLE dimensions are invalid");
    std::vector<F> result(cols, F_ZERO);
    for (std::size_t col = 0; col < cols; ++col) {
        std::vector<F> column(rows);
        for (std::size_t row = 0; row < rows; ++row)
            column[row] = matrix[row + col * rows];
        result[col] = mle_eval(column, row_point);
    }
    return result;
}

// Lift a family-local row point into the canonical global L_qry MLE domain.
// A family segment is a power-of-two aligned block; the low coordinates are
// the row point and the remaining Boolean coordinates select that block.
// Passing row_point directly to a full-domain opening would either fail the
// Hyrax dimension check or prove an unrelated evaluation.
std::vector<F> lift_segment_point(const std::vector<F> &row_point,
                                  std::size_t segment_offset,
                                  std::size_t rows,
                                  std::size_t domain) {
    if (!power_of_two(rows) || !power_of_two(domain) || rows > domain ||
        segment_offset > domain - rows || segment_offset % rows != 0 ||
        row_point.size() != log2_power(rows))
        throw std::invalid_argument("native Lasso segment point is not canonical");
    const std::size_t block = segment_offset / rows;
    const std::size_t block_count = domain / rows;
    if (block >= block_count) throw std::invalid_argument("native Lasso segment block is invalid");
    std::vector<F> result = row_point;
    result.reserve(log2_power(domain));
    for (std::size_t bit = 0; bit < log2_power(block_count); ++bit)
        result.push_back(((block >> bit) & 1U) != 0 ? F_ONE : F_ZERO);
    return result;
}

bool split_pipe_fields(const std::string &payload, std::size_t expected,
                       std::vector<std::string> &parts) {
    parts.clear();
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = payload.find('|', begin);
        parts.push_back(payload.substr(begin,
                                       end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parts.size() == expected;
}

bool parse_masked_open_claim(const NativeProofSession &session,
                             const std::string &label, G1 &mask_commitment,
                             F &masked_evaluation, std::string *reason,
                             const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                             zkgpt::native_pedersen::ScalarCommitment *raw_commitment = nullptr,
                             zkgpt::native_pedersen::ScalarCommitment *masked_commitment = nullptr) {
    const NativeProofMessage *found = nullptr;
    for (const NativeProofMessage &message : session.messages()) {
        if (message.label != label + "/claim") continue;
        if (found != nullptr) {
            set_reason(reason, "duplicate masked Hyrax claim: " + label);
            return false;
        }
        found = &message;
    }
    std::vector<std::string> parts;
    const std::size_t expected = scalar_basis == nullptr ? 2 : 5;
    if (found == nullptr || found->kind != NativeProofMessageKind::Data ||
        !split_pipe_fields(found->payload, expected, parts) ||
        !decode_point(parts[0], mask_commitment) || !decode_field(parts[1], masked_evaluation)) {
        set_reason(reason, "malformed masked Hyrax claim: " + label);
        return false;
    }
    if (scalar_basis != nullptr) {
        if (parts[2] != scalar_basis->domain || raw_commitment == nullptr ||
            masked_commitment == nullptr || !decode_point(parts[3], raw_commitment->point) ||
            !decode_point(parts[4], masked_commitment->point)) {
            set_reason(reason, "malformed masked Hyrax scalar claim: " + label);
            return false;
        }
        raw_commitment->domain = scalar_basis->domain;
        masked_commitment->domain = scalar_basis->domain;
        if (!raw_commitment->valid(*scalar_basis, reason) ||
            !masked_commitment->valid(*scalar_basis, reason))
            return false;
    }
    return true;
}

bool verify_masked_opening(
    const NativeProofSession &session, const G1 &commitment,
    const std::string &basis_id, const std::vector<G1> &generators,
    const std::vector<F> &point, const std::string &label,
    const F &expected_masked_evaluation, std::string *reason,
    bool enforce_statement_basis = false, bool allow_trailing = true,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
    const zkgpt::native_pedersen::ScalarCommitment *expected_raw_commitment = nullptr,
    const zkgpt::native_pedersen::ScalarCommitment *expected_masked_commitment = nullptr) {
    G1 mask_commitment;
    F masked_evaluation;
    zkgpt::native_pedersen::ScalarCommitment encoded_raw;
    zkgpt::native_pedersen::ScalarCommitment encoded_masked;
    if (!parse_masked_open_claim(session, label, mask_commitment,
                                 masked_evaluation, reason, scalar_basis,
                                 scalar_basis == nullptr ? nullptr : &encoded_raw,
                                 scalar_basis == nullptr ? nullptr : &encoded_masked))
        return false;
    if (masked_evaluation != expected_masked_evaluation) {
        set_reason(reason, "masked Hyrax claim is not bound to Sumcheck terminal");
        return false;
    }
    if (scalar_basis != nullptr) {
        if (expected_raw_commitment != nullptr &&
            encoded_raw.point != expected_raw_commitment->point) {
            set_reason(reason, "masked Hyrax raw claim is not externally bound");
            return false;
        }
        if (expected_masked_commitment != nullptr &&
            encoded_masked.point != expected_masked_commitment->point) {
            set_reason(reason, "masked Hyrax masked claim is not externally bound");
            return false;
        }
        if (expected_raw_commitment == nullptr) expected_raw_commitment = &encoded_raw;
        if (expected_masked_commitment == nullptr)
            expected_masked_commitment = &encoded_masked;
    }
    return hyrax::verify_masked_mle_opening_with_session(
        commitment, mask_commitment, masked_evaluation, basis_id, generators,
        point, session, label, reason, enforce_statement_basis, allow_trailing,
        "", scalar_basis, expected_raw_commitment, expected_masked_commitment);
}

std::string masked_family_payload(
    const NativeLassoFamilyPublic &family, const G1 &commitment,
    const G1 &digest_generator, const std::string &basis_id) {
    return encode_tokens({
        family.family_id, family.table_id, family.resource_id,
        family.registration_id, std::to_string(family.padded_queries),
        std::to_string(family.table.size()), std::to_string(family.segment_offset),
        std::to_string(family.logical_queries), std::to_string(family.padding_index),
        family.padding_value.getStr(16), commitment.serializeToHexStr(), basis_id,
        hyrax::derive_pedersen_blinding_generator(basis_id).serializeToHexStr(),
        native_lasso_generator_digest(digest_generator, family.matrix_generators),
        native_lasso_table_digest(family)});
}

bool parse_masked_family_commitment(
    const NativeProofMessage &message, const NativeLassoFamilyPublic &expected,
    const G1 &digest_generator, G1 &commitment, std::string &basis_id,
    std::string *reason) {
    std::vector<std::string> fields;
    if (message.kind != NativeProofMessageKind::Data ||
        !decode_tokens(message.payload, 15, fields)) {
        set_reason(reason, "malformed native Lasso LC_f commitment");
        return false;
    }
    if (fields[0] != expected.family_id || fields[1] != expected.table_id ||
        fields[2] != expected.resource_id || fields[3] != expected.registration_id ||
        fields[9] != expected.padding_value.getStr(16) ||
        fields[13] != native_lasso_generator_digest(digest_generator,
                                                    expected.matrix_generators) ||
        fields[14] != native_lasso_table_digest(expected) ||
        !decode_point(fields[10], commitment) || !valid_basis_id(fields[11])) {
        set_reason(reason, "native Lasso LC_f authenticated identity mismatch");
        return false;
    }
    std::size_t rows = 0, cols = 0, offset = 0, logical = 0, padding = 0;
    if (!parse_size(fields[4], rows) || !parse_size(fields[5], cols) ||
        !parse_size(fields[6], offset) || !parse_size(fields[7], logical) ||
        !parse_size(fields[8], padding) || rows != expected.padded_queries ||
        cols != expected.table.size() || offset != expected.segment_offset ||
        logical != expected.logical_queries || padding != expected.padding_index) {
        set_reason(reason, "native Lasso LC_f shape mismatch");
        return false;
    }
    if (fields[12] != hyrax::derive_pedersen_blinding_generator(fields[11]).serializeToHexStr()) {
        set_reason(reason, "native Lasso LC_f blinding identity mismatch");
        return false;
    }
    basis_id = fields[11];
    return true;
}

bool consume_field_message(const NativeProofSession &session, std::size_t &position,
                           const std::string &label, F &value,
                           std::string *reason) {
    const NativeProofMessage *message = consume(
        session, position, NativeProofMessageKind::Field, label, reason);
    if (message == nullptr || !decode_field(message->payload, value)) {
        set_reason(reason, "malformed masked native Lasso field: " + label);
        return false;
    }
    return true;
}

bool opening_commitment_at(const NativeProofSession &session,
                           const zkgpt::native_pedersen::PedersenBasis &basis,
                           const std::string &label,
                           zkgpt::native_pedersen::ScalarCommitment &commitment,
                           std::string *reason) {
    // Keep one canonical parser for native Pedersen opening metadata.  The
    // masked Sumcheck verifier and this Lasso verifier consume the same
    // opening wire; duplicating its length/point parser here caused the two
    // native v2 paths to disagree on otherwise valid BN254 encodings.
    return native_masked_opening_commitment_at(session, basis, 0, label,
                                               commitment, reason);
}

bool build_sumcheck_verify_spec(
    const NativeProofSession &session, const std::string &label,
    NativeMaskedTerminalKind kind, const F &initial_value,
    const zkgpt::native_pedersen::PedersenBasis &basis,
    NativeMaskedTerminalSpec &spec, std::string *reason,
    bool public_initial_claim) {
    spec = NativeMaskedTerminalSpec{};
    spec.kind = kind;
    spec.has_external_binding = true;
    spec.public_initial_claim = public_initial_claim;
    if (!opening_commitment_at(session, basis, label + "/initial/opening",
                               spec.initial_commitment, reason))
        return false;
    if (!public_initial_claim) {
        // A private initial claim is a two-sided external binding: the
        // unmasked evaluation commitment and the masked incoming claim must
        // both be named by the transcript before any round is verified.
        if (!opening_commitment_at(session, basis, label + "/initial/masked",
                                   spec.masked_initial_commitment, reason))
            return false;
        spec.has_masked_initial_binding = true;
    }
    const zkgpt::native_pedersen::ScalarCommitment expected_initial{
        basis.domain, zkgpt::native_pedersen::commit(basis, initial_value, F_ZERO).point};
    if (kind == NativeMaskedTerminalKind::Scalar ||
        kind == NativeMaskedTerminalKind::Product2) {
        // These initial claims are public relation constants in Lasso.  Keep
        // the verifier fail-closed if their authenticated scalar commitment
        // is not the canonical public affine binding.
        if (spec.initial_commitment.point != expected_initial.point) {
            set_reason(reason, "Lasso Sumcheck initial public binding mismatch");
            return false;
        }
    }
    const std::size_t arity = kind == NativeMaskedTerminalKind::Product2 ? 2 : 1;
    spec.terminal_commitments.resize(arity);
    for (std::size_t index = 0; index < arity; ++index)
        if (!opening_commitment_at(
                session, basis,
                label + "/terminal/value/" + std::to_string(index),
                spec.terminal_commitments[index], reason))
            return false;
    return true;
}

std::size_t relation_rounds(std::size_t size) {
    if (size == 0 || (size & (size - 1)) != 0) throw std::invalid_argument("masked Lasso relation size is not a power of two");
    return log2_power(size);
}

bool advance_masked_scope(const NativeProofSession &session,
                          const std::string &label, std::size_t &position,
                          std::string *reason) {
    const std::string prefix = label + "/";
    const std::string challenge_prefix = "challenge/" + prefix;
    std::size_t first = session.messages().size();
    std::size_t last = position;
    for (std::size_t index = 0; index < session.messages().size(); ++index) {
        const std::string &message_label = session.messages()[index].label;
        if (message_label.compare(0, prefix.size(), prefix) != 0 &&
            message_label.compare(0, challenge_prefix.size(), challenge_prefix) != 0)
            continue;
        if (first == session.messages().size()) first = index;
        last = index;
    }
    if (first == session.messages().size() || first != position) {
        set_reason(reason, "masked Lasso scope is not adjacent at " + label);
        return false;
    }
    for (std::size_t index = first; index <= last; ++index) {
        const std::string &message_label = session.messages()[index].label;
        if (message_label.compare(0, prefix.size(), prefix) != 0 &&
            message_label.compare(0, challenge_prefix.size(), challenge_prefix) != 0) {
            set_reason(reason, "masked Lasso scope has an inserted message at " + label);
            return false;
        }
    }
    position = last + 1;
    return true;
}

} // namespace

bool native_lasso_prove_bundle(
    NativeProofSession &session,
    std::vector<NativeLassoFamilyProver> &families,
    NativeLassoQueryProver &query,
    NativeLassoBundleResult *result,
    std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (families.empty() || !validate_query(query, reason)) return false;
        std::vector<NativeLassoFamilyPublic> identities;
        identities.reserve(families.size());
        for (NativeLassoFamilyProver &family : families) {
            if (!validate_prover_family(family, reason)) return false;
            identities.push_back(public_identity(family));
        }
        if (!check_family_order(identities, reason)) return false;
        if (!check_statement_identity(session, identities, query.commitment,
                                      query.values.size(), query.statement_l_digest,
                                      reason))
            return false;
        const hyrax::PedersenMleCommitment query_commitment =
            query_commitment_from(query);
        std::vector<MaskedFamilyState> states;
        states.reserve(families.size());
        for (const NativeLassoFamilyProver &family : families) {
            if (family.segment_offset + family.padded_queries > query.values.size()) {
                set_reason(reason, "native Lasso family exceeds L_qry segment");
                return false;
            }
            MaskedFamilyState state;
            state.basis_id = family_basis_id(session, family);
            state.commitment = hyrax::pedersen_commit_mle(
                family.matrix_generators, family.matrix, state.basis_id);
            states.push_back(std::move(state));
        }

        NativeLassoBundleResult local;
        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            lasso_scalar_basis(session);
        local.family_commitments.reserve(states.size());
        // Every LC_f is a hiding Pedersen commitment and is absorbed before
        // any family query point or masked Sumcheck challenge is derived.
        for (std::size_t index = 0; index < families.size(); ++index) {
            local.family_commitments.push_back(states[index].commitment.commitment);
            session.absorb(family_prefix(families[index]) + "/commitment",
                           masked_family_payload(families[index],
                                                 states[index].commitment.commitment,
                                                 query.G, states[index].basis_id));
        }

        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            NativeLassoFamilyProver &family = families[family_index];
            const std::string prefix = family_prefix(family);
            const std::size_t rows = family.padded_queries;
            const std::size_t cols = family.table.size();
            const std::size_t row_rounds = relation_rounds(rows);
            std::vector<F> row_point;
            row_point.reserve(row_rounds);
            for (std::size_t round = 0; round < row_rounds; ++round)
                row_point.push_back(session.challenge(
                    prefix + "/a-query/r/" + std::to_string(round)));
            local.family_query_points.push_back(row_point);

            const std::size_t matrix_size = rows * cols;
            std::vector<F> matrix_minus_one = family.matrix;
            for (F &value : matrix_minus_one) value -= F_ONE;
            NativeMaskedTerminalSpec boolean_spec;
            boolean_spec.kind = NativeMaskedTerminalKind::Product2;
            boolean_spec.has_external_binding = true;
            // The Boolean relation starts from the public zero claim.  This
            // is the one legitimate zero-mask scope in this family; its
            // private matrix evaluations are still masked at the terminal.
            boolean_spec.public_initial_claim = true;
            boolean_spec.initial_opening = public_scalar_opening(scalar_basis, F_ZERO);
            boolean_spec.initial_commitment = boolean_spec.initial_opening.commitment;
            session.absorb_field(prefix + "/b-boolean/claim", F_ZERO);
            NativeMaskedSumcheckProver boolean_prover(
                session, prefix + "/b-boolean/sum", 2, F_ZERO, boolean_spec);
            std::vector<F> boolean_left = family.matrix;
            std::vector<F> boolean_right = matrix_minus_one;
            std::vector<F> boolean_point;
            boolean_point.reserve(relation_rounds(matrix_size));
            for (std::size_t round = 0; round < relation_rounds(matrix_size); ++round) {
                std::vector<F> next_left, next_right;
                F challenge;
                append_product_round(boolean_prover, boolean_left, boolean_right,
                                     next_left, next_right, reason, &challenge);
                boolean_point.push_back(challenge);
                boolean_left.swap(next_left);
                boolean_right.swap(next_right);
            }
            zkgpt::native_pedersen::ScalarOpening boolean_raw_opening;
            zkgpt::native_pedersen::ScalarOpening complement_raw_opening;
            if (!make_eval_opening(scalar_basis, boolean_point, family.matrix,
                                   boolean_raw_opening, reason) ||
                !make_eval_opening(scalar_basis, boolean_point, matrix_minus_one,
                                   complement_raw_opening, reason))
                return false;
            boolean_spec.unmasked_terminal_openings = {
                boolean_raw_opening, complement_raw_opening};
            if (!boolean_prover.set_terminal_spec(boolean_spec, reason)) return false;
            NativeMaskedSumcheckResult boolean_result;
            if (!boolean_prover.finish({boolean_left.front(), boolean_right.front()},
                                        &boolean_result, reason)) return false;
            hyrax::MaskedMleOpening boolean_opening;
            if (!hyrax::open_masked_mle_with_claim(
                    states[family_index].commitment, family.matrix_generators,
                    boolean_point, family.matrix,
                    boolean_result.terminal_values[0], session,
                    prefix + "/c-boolean-open", boolean_opening,
                    reason, false, "", &scalar_basis,
                    &boolean_result.terminal_openings[0], &boolean_raw_opening))
                return false;
            // The complement uses the same LC_f blinding and is opened at the
            // same fresh point with the second masked terminal claim.
            hyrax::PedersenMleCommitment complement = states[family_index].commitment;
            std::vector<F> ones(matrix_size, F_ONE);
            std::vector<G1> complement_generators = family.matrix_generators;
            complement.commitment -= perdersen_commit(
                complement_generators.data(), ones.data(),
                static_cast<int>(matrix_size));
            if (!hyrax::open_masked_mle_with_claim(
                    complement, family.matrix_generators, boolean_point,
                    matrix_minus_one, boolean_result.terminal_values[1], session,
                    prefix + "/c-boolean-complement-open", boolean_opening,
                    reason, false, "", &scalar_basis,
                    &boolean_result.terminal_openings[1], &complement_raw_opening))
                return false;

            std::vector<F> onehot_left = row_mle(family.matrix, rows, cols, row_point);
            session.absorb_field(prefix + "/d-onehot/claim", F_ONE);
            NativeMaskedTerminalSpec onehot_spec;
            onehot_spec.kind = NativeMaskedTerminalKind::Scalar;
            onehot_spec.has_external_binding = true;
            // One-hot normalization is anchored at the public unit claim;
            // the matrix evaluation opened below remains private/masked.
            onehot_spec.public_initial_claim = true;
            onehot_spec.initial_opening = public_scalar_opening(scalar_basis, F_ONE);
            onehot_spec.initial_commitment = onehot_spec.initial_opening.commitment;
            NativeMaskedSumcheckProver onehot_prover(
                session, prefix + "/d-onehot/sum", 1, F_ONE, onehot_spec);
            std::vector<F> onehot_current = onehot_left;
            std::vector<F> onehot_challenges;
            onehot_challenges.reserve(relation_rounds(cols));
            for (std::size_t round = 0; round < relation_rounds(cols); ++round) {
                std::vector<F> next;
                F challenge;
                append_linear_round(onehot_prover, onehot_current, next, reason,
                                    &challenge);
                onehot_challenges.push_back(challenge);
                onehot_current.swap(next);
            }
            std::vector<F> onehot_point = row_point;
            onehot_point.insert(onehot_point.end(), onehot_challenges.begin(),
                                onehot_challenges.end());
            zkgpt::native_pedersen::ScalarOpening onehot_raw_opening;
            if (!make_eval_opening(scalar_basis, onehot_point, family.matrix,
                                   onehot_raw_opening, reason))
                return false;
            onehot_spec.unmasked_terminal_openings = {onehot_raw_opening};
            if (!onehot_prover.set_terminal_spec(onehot_spec, reason)) return false;
            NativeMaskedSumcheckResult onehot_result;
            if (!onehot_prover.finish({onehot_current.front()}, &onehot_result, reason))
                return false;
            hyrax::MaskedMleOpening onehot_opening;
            if (!hyrax::open_masked_mle_with_claim(
                    states[family_index].commitment, family.matrix_generators,
                    onehot_point, family.matrix, onehot_result.final_claim, session,
                    prefix + "/e-onehot-open", onehot_opening,
                    reason, false, "", &scalar_basis,
                    &onehot_result.terminal_openings[0], &onehot_raw_opening))
                return false;

            const std::vector<F> left = row_mle(family.matrix, rows, cols, row_point);
            F ell = F_ZERO;
            for (std::size_t col = 0; col < cols; ++col) ell += left[col] * family.table[col];
            const F initial_mask = [&]() {
                F value;
                value.setByCSPRNG();
                return value;
            }();
            const F masked_initial_claim = ell + initial_mask;
            const std::vector<F> lookup_initial_point = [&]() {
                std::vector<F> point = row_point;
                const F half = F(1) / F(2);
                for (std::size_t bit = 0; bit < log2_power(cols); ++bit)
                    point.push_back(half);
                return point;
            }();
            std::vector<F> weighted_matrix(matrix_size, F_ZERO);
            for (std::size_t col = 0; col < cols; ++col)
                for (std::size_t row = 0; row < rows; ++row)
                    weighted_matrix[row + col * rows] =
                        family.matrix[row + col * rows] * family.table[col];
            zkgpt::native_pedersen::ScalarOpening weighted_opening;
            if (!make_eval_opening(scalar_basis, lookup_initial_point,
                                   weighted_matrix, weighted_opening, reason))
                return false;
            const F lookup_scale = F(cols);
            zkgpt::native_pedersen::ScalarOpening lookup_initial_opening;
            lookup_initial_opening.value = weighted_opening.value * lookup_scale;
            lookup_initial_opening.blinding = weighted_opening.blinding * lookup_scale;
            lookup_initial_opening.commitment = zkgpt::native_pedersen::commit(
                scalar_basis, lookup_initial_opening.value,
                lookup_initial_opening.blinding);
            // Lasso's lookup claim is private: bind both the actual weighted
            // MLE evaluation (ell) and its masked incoming claim.  Build the
            // latter from the former plus a real Pedersen mask opening so the
            // external binding is not an arbitrary scalar re-commit.
            const zkgpt::native_pedersen::ScalarOpening lookup_initial_mask =
                zkgpt::native_pedersen::commit_fresh(scalar_basis, initial_mask);
            zkgpt::native_pedersen::ScalarOpening lookup_masked_initial_opening;
            lookup_masked_initial_opening.value =
                lookup_initial_opening.value + lookup_initial_mask.value;
            lookup_masked_initial_opening.blinding =
                lookup_initial_opening.blinding + lookup_initial_mask.blinding;
            lookup_masked_initial_opening.commitment = zkgpt::native_pedersen::commit(
                scalar_basis, lookup_masked_initial_opening.value,
                lookup_masked_initial_opening.blinding);
            session.absorb_field(prefix + "/a-query/claim", masked_initial_claim);
            NativeMaskedTerminalSpec lookup_spec;
            lookup_spec.kind = NativeMaskedTerminalKind::Product2Public;
            lookup_spec.public_vector = family.table;
            lookup_spec.has_external_binding = true;
            lookup_spec.has_masked_initial_binding = true;
            lookup_spec.initial_opening = lookup_initial_opening;
            lookup_spec.initial_commitment = lookup_initial_opening.commitment;
            lookup_spec.masked_initial_opening = lookup_masked_initial_opening;
            lookup_spec.masked_initial_commitment =
                lookup_masked_initial_opening.commitment;
            NativeMaskedSumcheckProver lookup_prover(
                session, prefix + "/f-lookup", 2, masked_initial_claim,
                lookup_spec, &ell);
            std::vector<F> lookup_left = left;
            std::vector<F> lookup_right = family.table;
            std::vector<F> lookup_challenges;
            lookup_challenges.reserve(relation_rounds(cols));
            for (std::size_t round = 0; round < relation_rounds(cols); ++round) {
                std::vector<F> next_left, next_right;
                F challenge;
                append_product_round(lookup_prover, lookup_left, lookup_right,
                                     next_left, next_right, reason, &challenge);
                lookup_challenges.push_back(challenge);
                lookup_left.swap(next_left);
                lookup_right.swap(next_right);
            }
            std::vector<F> lookup_point = row_point;
            lookup_point.insert(lookup_point.end(), lookup_challenges.begin(),
                                lookup_challenges.end());
            zkgpt::native_pedersen::ScalarOpening lookup_raw_opening;
            if (!make_eval_opening(scalar_basis, lookup_point, family.matrix,
                                   lookup_raw_opening, reason))
                return false;
            lookup_spec.unmasked_terminal_openings = {lookup_raw_opening};
            // The v2 streaming prover checks the unmasked Product2Public
            // terminal before it installs its derived public evaluation.
            // Freeze the same table MLE value that was folded by the round
            // prover so both checks use the authenticated public factor.
            lookup_spec.public_evaluation = lookup_right.front();
            if (!lookup_prover.set_terminal_spec(lookup_spec, reason)) return false;
            NativeMaskedSumcheckResult lookup_result;
            if (!lookup_prover.finish({lookup_left.front()}, &lookup_result, reason))
                return false;
            hyrax::MaskedMleOpening lookup_opening;
            if (!hyrax::open_masked_mle_with_claim(
                    states[family_index].commitment, family.matrix_generators,
                    lookup_point, family.matrix, lookup_result.terminal_values[0],
                    session, prefix + "/g-matrix-open", lookup_opening,
                    reason, false, "", &scalar_basis,
                    &lookup_result.terminal_openings[0], &lookup_raw_opening))
                return false;

            const std::size_t query_domain = query.values.size();
            std::vector<F> selector_basis(query_domain, F_ZERO);
            const std::vector<F> row_basis = [&]() {
                std::vector<F> basis(1ULL << row_point.size());
                basis[0] = F_ONE - row_point[0];
                basis[1] = row_point[0];
                for (std::size_t bit = 1; bit < row_point.size(); ++bit) {
                    const std::size_t old_size = 1ULL << bit;
                    for (std::size_t index = old_size; index-- > 0;) {
                        const F previous = basis[index];
                        basis[index] = previous * (F_ONE - row_point[bit]);
                        basis[index + old_size] = previous * row_point[bit];
                    }
                }
                return basis;
            }();
            F query_eval = F_ZERO;
            for (std::size_t row = 0; row < rows; ++row) {
                selector_basis[family.segment_offset + row] = row_basis[row];
                query_eval += row_basis[row] * family.queries[row];
            }
            if (query.values.size() != query_domain ||
                query_commitment.commitment != query.commitment)
                throw std::runtime_error("native Lasso query commitment changed during proof");
            for (std::size_t row = 0; row < rows; ++row)
                if (query.values[family.segment_offset + row] != family.queries[row])
                    throw std::runtime_error("native Lasso query/input segment mismatch");
            if (lookup_initial_opening.value != query_eval)
                throw std::runtime_error(
                    "native Lasso lookup/query claim mismatch");
            session.absorb_field(prefix + "/h-input/claim", lookup_result.final_claim);
            hyrax::MaskedMleOpening input_opening;
            const std::vector<F> query_point = lift_segment_point(
                row_point, family.segment_offset, rows, query_domain);
            if (!hyrax::open_masked_mle_with_claim(
                    query_commitment, query.generators, query_point, query.values,
                    lookup_result.final_claim, session, prefix + "/h-input-open",
                    input_opening, reason, true, "", &scalar_basis,
                    &lookup_result.claim_opening, &lookup_initial_opening))
                return false;
            (void)query_eval;
        }

        const std::size_t final_rounds = relation_rounds(query.values.size());
        local.final_query_point.reserve(final_rounds);
        for (std::size_t round = 0; round < final_rounds; ++round)
            local.final_query_point.push_back(
                session.challenge("lasso/i-final/r/" + std::to_string(round)));
        F final_claim;
        final_claim.setByCSPRNG();
        session.absorb_field("lasso/i-final/claim", final_claim);
        zkgpt::native_pedersen::ScalarOpening final_raw_opening;
        if (!make_eval_opening(scalar_basis, local.final_query_point,
                               query.values, final_raw_opening, reason))
            return false;
        hyrax::MaskedMleOpening final_opening;
        if (!hyrax::open_masked_mle_with_claim(
                query_commitment, query.generators, local.final_query_point,
                query.values, final_claim, session,
                NativeLassoTerminalOpeningLabels::l_qry(), final_opening,
                reason, true, "", &scalar_basis, nullptr, &final_raw_opening))
            return false;
        if (final_opening.masked_evaluation != final_claim)
            throw std::runtime_error("native Lasso final opening claim mismatch");
        local.final_l_evaluation = final_claim;
        local.final_l_commitment = final_opening.masked_evaluation_opening.commitment;
        local.has_final_l_commitment = true;
        if (!session.replay(reason)) return false;
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_lasso_verify_bundle(
    const NativeProofSession &session,
    const std::vector<NativeLassoFamilyPublic> &families,
    const NativeLassoQueryPublic &query,
    NativeLassoBundleResult *result,
    std::string *reason) {
    try {
        if (!check_family_order(families, reason) ||
            !validate_query_public(query, query.generators.size(), reason) ||
            !check_statement_identity(session, families, query.commitment,
                                      query.generators.size(), query.statement_l_digest,
                                      reason))
            return false;
        for (const NativeLassoFamilyPublic &family : families)
            if (family.segment_offset + family.padded_queries > query.generators.size()) {
                set_reason(reason, "native Lasso public family exceeds L_qry segment");
                return false;
            }
        if (!session.replay(reason)) return false;
        const zkgpt::native_pedersen::PedersenBasis scalar_basis =
            lasso_scalar_basis(session);
        NativeLassoBundleResult local;
        local.family_commitments.reserve(families.size());
        const std::string first_commitment_label =
            family_prefix(families.front()) + "/commitment";
        const std::size_t first_commitment = find_message(
            session, first_commitment_label, NativeProofMessageKind::Data, reason);
        if (first_commitment == session.messages().size()) return false;
        for (std::size_t index = 0; index < first_commitment; ++index) {
            const std::string &label = session.messages()[index].label;
            if (label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
                label.compare(0, sizeof("challenge/lasso/") - 1,
                              "challenge/lasso/") == 0) {
                set_reason(reason, "native Lasso scope has a hidden prefix message");
                return false;
            }
        }
        std::size_t position = first_commitment;
        for (const NativeLassoFamilyPublic &family : families) {
            const std::string expected = family_prefix(family) + "/commitment";
            const NativeProofMessage *message = consume(
                session, position, NativeProofMessageKind::Data, expected, reason);
            if (message == nullptr) return false;
            G1 commitment;
            std::string basis_id;
            if (!parse_masked_family_commitment(*message, family,
                                                query.G, commitment, basis_id, reason))
                return false;
            local.family_commitments.push_back(commitment);
        }
        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyPublic &family = families[family_index];
            const std::string prefix = family_prefix(family);
            std::vector<F> row_point;
            for (std::size_t round = 0; round < relation_rounds(family.padded_queries); ++round) {
                F challenge;
                const NativeProofMessage *message = consume(
                    session, position, NativeProofMessageKind::Challenge,
                    "challenge/" + prefix + "/a-query/r/" + std::to_string(round), reason);
                if (message == nullptr || !decode_field(message->payload, challenge)) {
                    set_reason(reason, "malformed native Lasso row challenge");
                    return false;
                }
                row_point.push_back(challenge);
            }
            local.family_query_points.push_back(row_point);

            F boolean_claim;
            if (!consume_field_message(session, position, prefix + "/b-boolean/claim",
                                       boolean_claim, reason) || !boolean_claim.isZero())
                return false;
            NativeMaskedTerminalSpec boolean_spec;
            if (!build_sumcheck_verify_spec(
                    session, prefix + "/b-boolean/sum",
            NativeMaskedTerminalKind::Product2, F_ZERO, scalar_basis,
                    boolean_spec, reason, true))
                return false;
            NativeMaskedSumcheckResult boolean_result;
            std::string sumcheck_reason;
            if (!native_masked_sumcheck_verify(
                    session, prefix + "/b-boolean/sum", 2,
                    relation_rounds(family.padded_queries * family.table.size()),
                    boolean_claim, boolean_spec,
                    &boolean_result, &sumcheck_reason)) {
                set_reason(reason, prefix + "/b-boolean/sum: " + sumcheck_reason);
                return false;
            }
            if (!advance_masked_scope(session, prefix + "/b-boolean/sum",
                                      position, reason))
                return false;
            const std::string matrix_basis_id = family_basis_id(session, family);
            if (!verify_masked_opening(
                    session, local.family_commitments[family_index], matrix_basis_id,
                    family.matrix_generators, boolean_result.challenges,
                    prefix + "/c-boolean-open", boolean_result.terminal_values[0],
                    reason, false, true, &scalar_basis,
                    &boolean_result.unmasked_terminal_commitments[0],
                    &boolean_result.terminal_commitments[0]))
                return false;
            if (!advance_masked_scope(session, prefix + "/c-boolean-open",
                                      position, reason))
                return false;
            hyrax::PedersenMleCommitment complement;
            complement.commitment = local.family_commitments[family_index];
            complement.basis_id = matrix_basis_id;
            complement.blinding_generator =
                hyrax::derive_pedersen_blinding_generator(matrix_basis_id);
            std::vector<F> ones(family.padded_queries * family.table.size(), F_ONE);
            std::vector<G1> complement_generators = family.matrix_generators;
            complement.commitment -= perdersen_commit(
                complement_generators.data(), ones.data(),
                static_cast<int>(ones.size()));
            if (!verify_masked_opening(
                    session, complement.commitment, matrix_basis_id,
                    family.matrix_generators, boolean_result.challenges,
                    prefix + "/c-boolean-complement-open",
                    boolean_result.terminal_values[1], reason, false, true,
                    &scalar_basis,
                    &boolean_result.unmasked_terminal_commitments[1],
                    &boolean_result.terminal_commitments[1]))
                return false;
            if (!advance_masked_scope(session,
                                      prefix + "/c-boolean-complement-open",
                                      position, reason))
                return false;

            F onehot_claim;
            if (!consume_field_message(session, position, prefix + "/d-onehot/claim",
                                       onehot_claim, reason) || onehot_claim != F_ONE)
                return false;
            NativeMaskedTerminalSpec onehot_spec;
            if (!build_sumcheck_verify_spec(
                    session, prefix + "/d-onehot/sum",
                    NativeMaskedTerminalKind::Scalar, F_ONE, scalar_basis,
                    onehot_spec, reason, true))
                return false;
            NativeMaskedSumcheckResult onehot_result;
            sumcheck_reason.clear();
            if (!native_masked_sumcheck_verify(
                    session, prefix + "/d-onehot/sum", 1,
                    relation_rounds(family.table.size()), onehot_claim, onehot_spec,
                    &onehot_result, &sumcheck_reason)) {
                set_reason(reason, prefix + "/d-onehot/sum: " + sumcheck_reason);
                return false;
            }
            if (!advance_masked_scope(session, prefix + "/d-onehot/sum",
                                      position, reason))
                return false;
            std::vector<F> onehot_point = row_point;
            onehot_point.insert(onehot_point.end(), onehot_result.challenges.begin(),
                                onehot_result.challenges.end());
            if (!verify_masked_opening(
                    session, local.family_commitments[family_index], matrix_basis_id,
                    family.matrix_generators, onehot_point,
                    prefix + "/e-onehot-open", onehot_result.final_claim, reason,
                    false, true, &scalar_basis,
                    &onehot_result.unmasked_terminal_commitments[0],
                    &onehot_result.terminal_commitments[0]))
                return false;
            if (!advance_masked_scope(session, prefix + "/e-onehot-open",
                                      position, reason))
                return false;

            F lookup_claim;
            if (!consume_field_message(session, position, prefix + "/a-query/claim",
                                       lookup_claim, reason))
                return false;
            NativeMaskedTerminalSpec lookup_spec;
            if (!build_sumcheck_verify_spec(
                    session, prefix + "/f-lookup",
                    NativeMaskedTerminalKind::Product2Public, F_ZERO, scalar_basis,
                    lookup_spec, reason, false))
                return false;
            lookup_spec.public_vector = family.table;
            NativeMaskedSumcheckResult lookup_result;
            sumcheck_reason.clear();
            if (!native_masked_sumcheck_verify(
                    session, prefix + "/f-lookup", 2,
                    relation_rounds(family.table.size()), lookup_claim, lookup_spec,
                    &lookup_result, &sumcheck_reason)) {
                set_reason(reason, prefix + "/f-lookup: " + sumcheck_reason);
                return false;
            }
            if (!advance_masked_scope(session, prefix + "/f-lookup", position,
                                      reason))
                return false;
            std::vector<F> lookup_point = row_point;
            lookup_point.insert(lookup_point.end(), lookup_result.challenges.begin(),
                                lookup_result.challenges.end());
            if (!verify_masked_opening(
                    session, local.family_commitments[family_index], matrix_basis_id,
                    family.matrix_generators, lookup_point,
                    prefix + "/g-matrix-open", lookup_result.terminal_values[0], reason,
                    false, true, &scalar_basis,
                    &lookup_result.unmasked_terminal_commitments[0],
                    &lookup_result.terminal_commitments[0]))
                return false;
            if (!advance_masked_scope(session, prefix + "/g-matrix-open",
                                      position, reason))
                return false;
            F input_claim;
            if (!consume_field_message(session, position, prefix + "/h-input/claim",
                                       input_claim, reason) || input_claim != lookup_result.final_claim)
                return false;
            const std::string q_basis = query_basis_id(query);
            const std::vector<F> query_point = lift_segment_point(
                row_point, family.segment_offset, family.padded_queries,
                query.generators.size());
            if (!verify_masked_opening(
                    session, query.commitment, q_basis, query.generators, query_point,
                    prefix + "/h-input-open", input_claim, reason, true, true,
                    &scalar_basis,
                    &lookup_result.initial_commitment,
                    &lookup_result.claim_commitment))
                return false;
            if (!advance_masked_scope(session, prefix + "/h-input-open", position,
                                      reason))
                return false;
        }

        local.final_query_point.reserve(relation_rounds(query.generators.size()));
        for (std::size_t round = 0; round < relation_rounds(query.generators.size()); ++round) {
            F challenge;
            const NativeProofMessage *message = consume(
                session, position, NativeProofMessageKind::Challenge,
                "challenge/lasso/i-final/r/" + std::to_string(round), reason);
            if (message == nullptr || !decode_field(message->payload, challenge)) {
                set_reason(reason, "malformed native Lasso final query challenge");
                return false;
            }
            local.final_query_point.push_back(challenge);
        }
        F final_claim;
        if (!consume_field_message(session, position, "lasso/i-final/claim",
                                   final_claim, reason))
            return false;
        G1 final_mask_commitment;
        F encoded_final_claim;
        zkgpt::native_pedersen::ScalarCommitment final_raw_commitment;
        zkgpt::native_pedersen::ScalarCommitment final_masked_commitment;
        if (!parse_masked_open_claim(
                session, NativeLassoTerminalOpeningLabels::l_qry(),
                final_mask_commitment, encoded_final_claim, reason, &scalar_basis,
                &final_raw_commitment, &final_masked_commitment) ||
            encoded_final_claim != final_claim) {
            set_reason(reason, "native Lasso final scalar claim binding is malformed");
            return false;
        }
        if (!verify_masked_opening(
                session, query.commitment, query_basis_id(query), query.generators,
                local.final_query_point, NativeLassoTerminalOpeningLabels::l_qry(),
                final_claim, reason, true, false, &scalar_basis))
            return false;
        if (!advance_masked_scope(session, NativeLassoTerminalOpeningLabels::l_qry(),
                                  position, reason))
            return false;
        local.final_l_evaluation = final_claim;
        local.final_l_commitment = final_masked_commitment;
        local.has_final_l_commitment = true;
        for (std::size_t index = position; index < session.messages().size(); ++index) {
            const std::string &label = session.messages()[index].label;
            if (label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
                label.compare(0, sizeof("challenge/lasso/") - 1,
                              "challenge/lasso/") == 0) {
                set_reason(reason, "native Lasso proof has trailing or inserted messages");
                return false;
            }
        }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

namespace native_lasso_legacy_clear {
bool native_lasso_verify_bundle(
    const NativeProofSession &session,
    const std::vector<NativeLassoFamilyPublic> &families,
    const NativeLassoQueryPublic &query,
    NativeLassoBundleResult *result,
    std::string *reason) {
    try {
        if (!check_family_order(families, reason) ||
            !validate_query_public(query, query.generators.size(), reason) ||
            !check_statement_identity(session, families, query.commitment,
                                      query.generators.size(), query.statement_l_digest,
                                      reason))
            return false;
        for (const NativeLassoFamilyPublic &family : families) {
            if (family.segment_offset + family.padded_queries > query.generators.size()) {
                set_reason(reason, "native Lasso public family exceeds L_qry segment");
                return false;
            }
        }
        if (!session.replay(reason)) return false;
        NativeLassoBundleResult local;
        local.family_commitments.reserve(families.size());
        std::size_t position = 0;
        std::size_t first_commitment = session.messages().size();
        for (std::size_t i = 0; i < session.messages().size(); ++i) {
            if (session.messages()[i].label.find("lasso/") == 0) {
                first_commitment = i;
                break;
            }
        }
        if (first_commitment == session.messages().size()) {
            set_reason(reason, "native Lasso commitment wire is missing");
            return false;
        }
        position = first_commitment;
        for (const NativeLassoFamilyPublic &family : families) {
            const std::string expected = family_prefix(family) + "/commitment";
            const NativeProofMessage *message = consume(
                session, position, NativeProofMessageKind::Data, expected, reason);
            if (message == nullptr) return false;
            G1 commitment;
            if (!parse_commitment(*message, family, query.G, commitment, reason))
                return false;
            local.family_commitments.push_back(commitment);
        }

        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyPublic &family = families[family_index];
            const std::string prefix = family_prefix(family);
            const std::size_t row_rounds = log2_power(family.padded_queries);
            std::vector<F> row_point;
            for (std::size_t round = 0; round < row_rounds; ++round) {
                F value;
                const NativeProofMessage *challenge = consume(
                    session, position, NativeProofMessageKind::Challenge,
                    "challenge/" + prefix + "/a-query/r/" + std::to_string(round),
                    reason);
                if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                    set_reason(reason, "native Lasso query challenge is malformed");
                    return false;
                }
                row_point.push_back(value);
            }
            local.family_query_points.push_back(row_point);
            // The booleanity check is part of the native Lasso relation and
            // must be consumed before the one-hot/table check.  The
            // verifier cannot skip these messages: otherwise LC_f could be
            // opened at a point unrelated to the committed one-hot matrix.
            const NativeProofMessage *boolean_claim = consume(
                session, position, NativeProofMessageKind::Field,
                prefix + "/b-boolean/claim", reason);
            F boolean_initial;
            if (boolean_claim == nullptr || !decode_field(boolean_claim->payload,
                                                          boolean_initial) ||
                !boolean_initial.isZero()) {
                set_reason(reason, "native Lasso boolean claim is malformed");
                return false;
            }
            const std::string boolean_label = prefix + "/b-boolean/sum";
            if (position >= session.messages().size() ||
                session.messages()[position].label != boolean_label + "/poly/0") {
                set_reason(reason, "native Lasso booleanity Sumcheck is not adjacent");
                return false;
            }
            F boolean_eval, boolean_minus_one_eval;
            if (!native_range_verify_product(session, boolean_label, boolean_initial,
                                             &boolean_eval, &boolean_minus_one_eval,
                                             reason))
                return false;
            const std::size_t boolean_terminal = find_message(
                session, boolean_label + "/terminal",
                NativeProofMessageKind::FieldVector, reason);
            if (boolean_terminal == session.messages().size() || boolean_terminal < position) {
                set_reason(reason, "native Lasso booleanity terminal is missing");
                return false;
            }
            position = boolean_terminal + 1;
            if (boolean_minus_one_eval != boolean_eval - F(1)) {
                set_reason(reason, "native Lasso boolean terminal relation failed");
                return false;
            }
            const std::size_t matrix_rounds =
                log2_power(family.padded_queries * family.table.size());
            std::vector<F> boolean_point;
            if (!collect_challenges(session, boolean_label, matrix_rounds,
                                    boolean_point, reason))
                return false;
            std::vector<F> boolean_basis = mle_basis(boolean_point);
            const std::string boolean_open_label = prefix + "/c-boolean-open";
            if (find_message(session, boolean_open_label + "/basis",
                             NativeProofMessageKind::Data, reason) != position) {
                set_reason(reason, "native Lasso boolean opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(
                    local.family_commitments[family_index], boolean_eval, query.G,
                    family.matrix_generators, boolean_basis, session,
                    boolean_open_label, reason, false, true))
                return false;
            const std::size_t boolean_open_terminal = find_message(
                session, boolean_open_label + "/terminal",
                NativeProofMessageKind::Data, reason);
            if (boolean_open_terminal == session.messages().size() ||
                boolean_open_terminal < position) {
                set_reason(reason, "native Lasso boolean opening terminal is missing");
                return false;
            }
            position = boolean_open_terminal + 1;

            const G1 complement_commit =
                local.family_commitments[family_index] -
                commit_all_ones(family.matrix_generators);
            const std::string complement_open_label =
                prefix + "/c-boolean-complement-open";
            if (find_message(session, complement_open_label + "/basis",
                             NativeProofMessageKind::Data, reason) != position) {
                set_reason(reason, "native Lasso boolean complement opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(
                    complement_commit, boolean_minus_one_eval, query.G,
                    family.matrix_generators, boolean_basis, session,
                    complement_open_label, reason, false, true))
                return false;
            const std::size_t complement_open_terminal = find_message(
                session, complement_open_label + "/terminal",
                NativeProofMessageKind::Data, reason);
            if (complement_open_terminal == session.messages().size() ||
                complement_open_terminal < position) {
                set_reason(reason, "native Lasso boolean complement terminal is missing");
                return false;
            }
            position = complement_open_terminal + 1;

            // The one-hot check projects M_f onto a transcript-derived row
            // point, then runs a degree-one Sumcheck over table columns.
            std::vector<F> onehot_point;
            for (std::size_t round = 0; round < row_rounds; ++round) {
                F value;
                const NativeProofMessage *challenge = consume(
                    session, position, NativeProofMessageKind::Challenge,
                    "challenge/" + prefix + "/d-onehot/r/" + std::to_string(round),
                    reason);
                if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                    set_reason(reason, "native Lasso one-hot challenge is malformed");
                    return false;
                }
                onehot_point.push_back(value);
            }
            const NativeProofMessage *onehot_claim = consume(
                session, position, NativeProofMessageKind::Field,
                prefix + "/d-onehot/claim", reason);
            F onehot_initial;
            if (onehot_claim == nullptr || !decode_field(onehot_claim->payload,
                                                         onehot_initial) ||
                onehot_initial != F_ONE) {
                set_reason(reason, "native Lasso one-hot claim is malformed");
                return false;
            }
            const std::string onehot_label = prefix + "/d-onehot/sum";
            if (position >= session.messages().size() ||
                session.messages()[position].label != onehot_label + "/poly/0") {
                set_reason(reason, "native Lasso one-hot Sumcheck is not adjacent");
                return false;
            }
            F onehot_eval;
            if (!native_range_verify_deg1(session, onehot_label, onehot_initial,
                                          &onehot_eval, reason))
                return false;
            const std::size_t onehot_terminal = find_message(
                session, onehot_label + "/terminal", NativeProofMessageKind::Field,
                reason);
            if (onehot_terminal == session.messages().size() || onehot_terminal < position) {
                set_reason(reason, "native Lasso one-hot terminal is missing");
                return false;
            }
            position = onehot_terminal + 1;
            const std::size_t col_rounds = log2_power(family.table.size());
            std::vector<F> onehot_columns;
            if (!collect_challenges(session, onehot_label, col_rounds,
                                    onehot_columns, reason))
                return false;
            std::vector<F> onehot_matrix_point;
            onehot_matrix_point.reserve(onehot_point.size() + onehot_columns.size());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot_point.begin(),
                                       onehot_point.end());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot_columns.begin(),
                                       onehot_columns.end());
            std::vector<F> onehot_basis = mle_basis(onehot_matrix_point);
            const std::string onehot_open_label = prefix + "/e-onehot-open";
            if (find_message(session, onehot_open_label + "/basis",
                             NativeProofMessageKind::Data, reason) != position) {
                set_reason(reason, "native Lasso one-hot opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(
                    local.family_commitments[family_index], onehot_eval, query.G,
                    family.matrix_generators, onehot_basis, session,
                    onehot_open_label, reason, false, true))
                return false;
            const std::size_t onehot_open_terminal = find_message(
                session, onehot_open_label + "/terminal",
                NativeProofMessageKind::Data, reason);
            if (onehot_open_terminal == session.messages().size() ||
                onehot_open_terminal < position) {
                set_reason(reason, "native Lasso one-hot opening terminal is missing");
                return false;
            }
            position = onehot_open_terminal + 1;

            const NativeProofMessage *query_claim = consume(
                session, position, NativeProofMessageKind::Field,
                prefix + "/a-query/claim", reason);
            F ell;
            if (query_claim == nullptr || !decode_field(query_claim->payload, ell)) {
                set_reason(reason, "native Lasso query claim is malformed");
                return false;
            }
            const std::string lookup_label = prefix + "/f-lookup";
            const std::string first_lookup_poly = lookup_label + "/poly/0";
            if (position >= session.messages().size() ||
                session.messages()[position].label != first_lookup_poly) {
                set_reason(reason, "native Lasso lookup Sumcheck is not adjacent");
                return false;
            }
            F matrix_eval, table_eval;
            if (!native_range_verify_product(session, lookup_label, ell,
                                             &matrix_eval, &table_eval, reason))
                return false;
            const std::size_t lookup_terminal = find_message(
                session, lookup_label + "/terminal",
                NativeProofMessageKind::FieldVector, reason);
            if (lookup_terminal == session.messages().size() || lookup_terminal < position) {
                set_reason(reason, "native Lasso lookup terminal is missing");
                return false;
            }
            position = lookup_terminal + 1;
            std::vector<F> y_point;
            if (!collect_challenges(session, lookup_label, col_rounds, y_point, reason))
                return false;
            const std::vector<F> table_basis = mle_basis(y_point);
            F expected_table = 0;
            for (std::size_t i = 0; i < family.table.size(); ++i)
                expected_table += table_basis[i] * family.table[i];
            if (expected_table != table_eval) {
                set_reason(reason, "native Lasso public table terminal mismatch");
                return false;
            }
            std::vector<F> matrix_point;
            matrix_point.reserve(row_point.size() + y_point.size());
            matrix_point.insert(matrix_point.end(), row_point.begin(), row_point.end());
            matrix_point.insert(matrix_point.end(), y_point.begin(), y_point.end());
            const std::vector<F> matrix_basis = mle_basis(matrix_point);
            const std::string matrix_label = prefix + "/g-matrix-open";
            const std::size_t matrix_basis_index = find_message(
                session, matrix_label + "/basis", NativeProofMessageKind::Data, reason);
            if (matrix_basis_index != position) {
                set_reason(reason, "native Lasso M_f opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(
                    local.family_commitments[family_index], matrix_eval, query.G,
                    family.matrix_generators, matrix_basis, session, matrix_label,
                    reason, false, true))
                return false;
            const std::size_t matrix_terminal = find_message(
                session, matrix_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (matrix_terminal == session.messages().size() || matrix_terminal < position) {
                set_reason(reason, "native Lasso M_f terminal is missing");
                return false;
            }
            position = matrix_terminal + 1;

            const NativeProofMessage *input_claim = consume(
                session, position, NativeProofMessageKind::FieldVector,
                prefix + "/h-input/claim", reason);
            std::vector<F> input_values;
            if (!decode_vector(input_claim, 2, input_values) || input_values[0] != ell ||
                input_values[1] != input_values[0]) {
                set_reason(reason, "native Lasso query/input consistency claim failed");
                return false;
            }
            std::vector<F> selector_basis(query.generators.size(), F_ZERO);
            const std::vector<F> row_basis = mle_basis(row_point);
            for (std::size_t row = 0; row < family.padded_queries; ++row)
                selector_basis[family.segment_offset + row] = row_basis[row];
            const std::string input_label = prefix + "/h-input-open";
            const std::size_t input_basis_index = find_message(
                session, input_label + "/basis", NativeProofMessageKind::Data, reason);
            if (input_basis_index != position) {
                set_reason(reason, "native Lasso input opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(
                    query.commitment, input_values[1], query.G, query.generators,
                    selector_basis, session, input_label, reason, true, true))
                return false;
            const std::size_t input_terminal = find_message(
                session, input_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (input_terminal == session.messages().size() || input_terminal < position) {
                set_reason(reason, "native Lasso input terminal is missing");
                return false;
            }
            position = input_terminal + 1;
        }

        const std::size_t final_rounds = log2_power(query.generators.size());
        for (std::size_t round = 0; round < final_rounds; ++round) {
            F value;
            const NativeProofMessage *challenge = consume(
                session, position, NativeProofMessageKind::Challenge,
                "challenge/lasso/i-final/r/" + std::to_string(round), reason);
            if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                set_reason(reason, "native Lasso final query challenge is malformed");
                return false;
            }
            local.final_query_point.push_back(value);
        }
        const std::vector<F> final_basis = mle_basis(local.final_query_point);
        const NativeProofMessage *final_claim = consume(
            session, position, NativeProofMessageKind::Field,
            "lasso/i-final/claim", reason);
        F final_eval;
        if (final_claim == nullptr || !decode_field(final_claim->payload, final_eval)) {
            set_reason(reason, "native Lasso final L_qry claim is malformed");
            return false;
        }
        local.final_l_evaluation = final_eval;
        const std::size_t final_basis_index = find_message(
            session, std::string(NativeLassoTerminalOpeningLabels::l_qry()) + "/basis",
            NativeProofMessageKind::Data, reason);
        if (final_basis_index != position) {
            set_reason(reason, "native Open_L is not adjacent to its claim");
            return false;
        }
        if (!hyrax::verify_opening_with_session(
                query.commitment, final_eval, query.G, query.generators, final_basis,
                session, NativeLassoTerminalOpeningLabels::l_qry(), reason, true, false))
            return false;
        const std::size_t final_terminal = find_message(
            session, std::string(NativeLassoTerminalOpeningLabels::l_qry()) + "/terminal",
            NativeProofMessageKind::Data, reason);
        if (final_terminal == session.messages().size() || final_terminal < position) {
            set_reason(reason, "native Open_L terminal is missing");
            return false;
        }
        position = final_terminal + 1;
        for (std::size_t index = position; index < session.messages().size(); ++index) {
            const std::string &label = session.messages()[index].label;
            if (label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
                label.compare(0, sizeof("challenge/lasso/") - 1,
                              "challenge/lasso/") == 0) {
                set_reason(reason, "native Lasso proof has trailing or inserted messages");
                return false;
            }
        }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}
} // namespace native_lasso_legacy_clear
