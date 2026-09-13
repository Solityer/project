#include "native_lasso.hpp"

#include "hyrax.hpp"

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
        query.generators.size() != query.values.size()) {
        set_reason(reason, "native Lasso L_qry setup is invalid");
        return false;
    }
    for (const G1 &generator : query.generators) {
        if (!valid_point(generator)) {
            set_reason(reason, "native Lasso L_qry generator is invalid");
            return false;
        }
    }
    return true;
}

bool validate_query_public(const NativeLassoQueryPublic &query,
                           std::size_t expected_size, std::string *reason) {
    if (!valid_point(query.G) || !power_of_two(expected_size) || expected_size < 2 ||
        query.generators.size() != expected_size || !valid_point(query.commitment)) {
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
                              std::string *reason) {
    if (!session.valid(reason)) return false;
    if (session.statement().tables_digest != native_lasso_tables_digest(families)) {
        set_reason(reason, "native Lasso table registry is not statement-bound");
        return false;
    }
    if (session.statement().resources_digest != native_lasso_resources_digest(families)) {
        set_reason(reason, "native Lasso resource/registration identity is not statement-bound");
        return false;
    }
    if (session.statement().l_digest != native_lasso_query_digest(query_commitment, query_size)) {
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
        query.commitment = perdersen_commit(query.generators.data(), query.values.data(),
                                            static_cast<int>(query.values.size()));
        if (!check_statement_identity(session, identities, query.commitment,
                                      query.values.size(), reason))
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
            const std::size_t matrix_rounds = log2_power(family.matrix.size());
            std::vector<F> boolean_point;
            boolean_point.reserve(matrix_rounds);
            for (std::size_t round = 0; round < matrix_rounds; ++round)
                boolean_point.push_back(session.challenge(
                    prefix + "/b-boolean/r/" + std::to_string(round)));
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
            std::vector<F> boolean_basis = mle_basis(boolean_point);
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
                                      query.generators.size(), reason))
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
            const NativeProofMessage *query_claim = consume(
                session, position, NativeProofMessageKind::Field,
                prefix + "/a-query/claim", reason);
            F ell;
            if (query_claim == nullptr || !decode_field(query_claim->payload, ell)) {
                set_reason(reason, "native Lasso query claim is malformed");
                return false;
            }
            const std::string lookup_label = prefix + "/b-lookup";
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
            const std::size_t col_rounds = log2_power(family.table.size());
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
            const std::string matrix_label = prefix + "/c-matrix-open";
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
                prefix + "/d-input/claim", reason);
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
            const std::string input_label = prefix + "/d-input-open";
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
                "challenge/lasso/e-final/r/" + std::to_string(round), reason);
            if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                set_reason(reason, "native Lasso final query challenge is malformed");
                return false;
            }
            local.final_query_point.push_back(value);
        }
        const std::vector<F> final_basis = mle_basis(local.final_query_point);
        const NativeProofMessage *final_claim = consume(
            session, position, NativeProofMessageKind::Field,
            "lasso/e-final/claim", reason);
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
        if (position != session.messages().size()) {
            set_reason(reason, "native Lasso proof has trailing or inserted messages");
            return false;
        }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}
