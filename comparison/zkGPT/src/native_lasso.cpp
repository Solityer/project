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

void set_reason(std::string* reason, const std::string& message) {
    if (reason != nullptr) *reason = message;
}

bool power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::size_t log2_power(std::size_t value) {
    std::size_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

bool valid_identifier(const std::string& value) {
    if (value.empty() || value.size() > 128) return false;
    for (char c : value) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) return false;
    }
    return true;
}

bool valid_digest(const std::string& value) {
    if (value.size() != 64) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

bool valid_basis_identifier(const std::string& value) {
    if (value.empty() || value.size() > 128) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == '-' || c == '_')) return false;
    }
    return true;
}

bool valid_point(const G1& point) { return point.isValid() && !point.isZero() && point.isValidOrder(); }

bool checked_product(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) return false;
    result = left * right;
    return result != 0 && result <= std::numeric_limits<std::size_t>::max();
}

bool valid_canonical_generator_id(const std::string& value) {
    if (value.empty() || value.size() > 128) return false;
    for (unsigned char c : value)
        if (c < 0x21 || c > 0x7e) return false;
    return true;
}

bool tensor_query_shape_valid(const hyrax::HyraxTensorMleView& view, const hyrax::HyraxTensorMleCommitment& commitment,
                              const hyrax::HyraxScalarProvider* values, std::string* reason) {
    const bool full_is_power_of_two = view.full_domain != 0 && (view.full_domain & (view.full_domain - 1)) == 0;
    std::uint64_t expected_active = 1;
    while (expected_active < view.logical_length && expected_active <= (std::numeric_limits<std::uint64_t>::max() >> 1)) expected_active <<= 1;
    std::uint64_t matrix_domain = 0;
    const std::size_t active_bits = view.active_domain == 0 || view.active_domain > std::numeric_limits<std::size_t>::max()
                                        ? 0
                                        : log2_power(static_cast<std::size_t>(view.active_domain));
    const std::uint64_t expected_columns = std::uint64_t(1) << (active_bits / 2);
    const std::uint64_t expected_rows = view.active_domain / expected_columns;
    if (!full_is_power_of_two || view.full_domain > std::numeric_limits<std::size_t>::max() || view.logical_length > view.full_domain ||
        expected_active != view.active_domain || view.active_domain > view.full_domain ||
        !checked_product(view.rows, view.columns, matrix_domain) || matrix_domain != view.active_domain ||
        view.rows != expected_rows || view.columns != expected_columns ||
        !power_of_two(static_cast<std::size_t>(view.rows)) || !power_of_two(static_cast<std::size_t>(view.columns)) ||
        view.schema != hyrax::kHyraxTensorMleSchema || view.method != hyrax::kHyraxTensorMleMethod || view.version != hyrax::kHyraxTensorMleVersion ||
        view.role.empty() || !valid_basis_identifier(view.basis_id) || !view.column_generators.at ||
        view.column_generators.size != view.columns || view.column_generators.scheme != hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1 ||
        !valid_canonical_generator_id(view.column_generators.canonical_id) || commitment.schema != view.schema || commitment.method != view.method ||
        commitment.version != view.version || commitment.role != view.role || commitment.basis_id != view.basis_id ||
        commitment.column_generator_scheme != view.column_generators.scheme || commitment.column_generator_id != view.column_generators.canonical_id ||
        commitment.full_domain != view.full_domain || commitment.logical_length != view.logical_length || commitment.active_domain != view.active_domain ||
        commitment.rows != view.rows || commitment.columns != view.columns || commitment.row_commitments.size() != view.rows ||
        !valid_digest(commitment.root_digest) || (values != nullptr && (!values->at || values->size != view.logical_length))) {
        set_reason(reason, "native Lasso typed L_qry tensor descriptor is invalid");
        return false;
    }
    for (const G1& row_commitment : commitment.row_commitments) {
        if (!valid_point(row_commitment)) {
            set_reason(reason, "native Lasso typed L_qry row commitment is invalid");
            return false;
        }
    }
    return true;
}

void append_u64(std::string& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_string(std::string& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

std::string hash_fixed(const std::string& encoded) {
    F digest;
    digest.setHashOf(encoded.data(), encoded.size());
    std::string value = digest.getStr(16);
    if (value.empty()) value = "0";
    if (value.size() > 64) throw std::runtime_error("native Lasso digest overflow");
    return std::string(64 - value.size(), '0') + value;
}

std::string encode_tokens(const std::vector<std::string>& tokens) {
    std::string payload;
    append_u64(payload, static_cast<std::uint64_t>(tokens.size()));
    for (const std::string& token : tokens) append_string(payload, token);
    return payload;
}

bool decode_tokens(const std::string& payload, std::size_t expected, std::vector<std::string>& tokens) {
    tokens.clear();
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count != expected) return false;
    tokens.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size > (1ULL << 20) || size > payload.size() - offset) return false;
        tokens.emplace_back(payload.substr(offset, static_cast<std::size_t>(size)));
        offset += static_cast<std::size_t>(size);
    }
    return offset == payload.size();
}

bool decode_field(const std::string& encoded, F& value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
    } catch (...) { return false; }
    return value.getStr(16) == encoded;
}

bool decode_point(const std::string& encoded, G1& value) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        value.deserializeHexStr(encoded);
    } catch (...) { return false; }
    return value.serializeToHexStr() == encoded && valid_point(value);
}

bool decode_vector(const NativeProofMessage* message, std::size_t expected, std::vector<F>& values) {
    if (message == nullptr || message->kind != NativeProofMessageKind::FieldVector) return false;
    const std::string& payload = message->payload;
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count != expected) return false;
    values.clear();
    values.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > 128 || size > payload.size() - offset) return false;
        F value;
        if (!decode_field(payload.substr(offset, static_cast<std::size_t>(size)), value)) return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    return offset == payload.size();
}

const NativeProofMessage* consume(const NativeProofSession& session, std::size_t& position, NativeProofMessageKind kind, const std::string& label,
                                  std::string* reason) {
    if (position >= session.messages().size() || session.messages()[position].kind != kind || session.messages()[position].label != label) {
        set_reason(reason, "native Lasso grammar mismatch at " + label);
        return nullptr;
    }
    return &session.messages()[position++];
}

std::size_t find_message(const NativeProofSession& session, const std::string& label, NativeProofMessageKind kind, std::string* reason) {
    const std::vector<std::size_t>* positions = session.message_positions(label);
    if (positions == nullptr || positions->empty()) {
        set_reason(reason, "missing native Lasso message: " + label);
        return session.messages().size();
    }
    std::size_t result = session.messages().size();
    for (std::size_t position : *positions) {
        if (position >= session.messages().size()) {
            set_reason(reason, "native Lasso message index is out of range: " + label);
            return session.messages().size();
        }
        if (session.messages()[position].kind != kind) continue;
        if (result != session.messages().size()) {
            set_reason(reason, "duplicate native Lasso message: " + label);
            return session.messages().size();
        }
        result = position;
    }
    if (result == session.messages().size()) set_reason(reason, "missing native Lasso message: " + label);
    return result;
}

std::vector<F> mle_basis(const std::vector<F>& point) {
    if (point.size() >= sizeof(std::size_t) * 8) throw std::invalid_argument("native Lasso point dimension is invalid");
    if (point.empty()) return {F_ONE};
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

NativeLassoFamilyPublic public_identity(const NativeLassoFamilyProver& family) {
    return static_cast<const NativeLassoFamilyPublic&>(family);
}

bool validate_public_family(const NativeLassoFamilyPublic& family, std::string* reason) {
    if (!valid_identifier(family.family_id) || !valid_identifier(family.table_id) || !valid_identifier(family.resource_id) ||
        !valid_identifier(family.registration_id)) {
        set_reason(reason, "native Lasso family identity is not canonical");
        return false;
    }
    std::uint64_t matrix_size = 0;
    if (!power_of_two(family.table.size()) || !power_of_two(family.padded_queries) || family.logical_queries > family.padded_queries ||
        family.padding_index >= family.table.size() || family.padding_value != family.table[family.padding_index] ||
        !checked_product(family.padded_queries, family.table.size(), matrix_size)) {
        set_reason(reason, "native Lasso public table/query dimensions are invalid");
        return false;
    }
    const NativeLassoFamilyPublic::MatrixDescriptor expected = native_lasso_balanced_matrix_layout(family.padded_queries, family.table.size());
    if (expected.vector_size != matrix_size || family.matrix_layout.version != expected.version ||
        family.matrix_layout.padded_queries != expected.padded_queries || family.matrix_layout.table_domain != expected.table_domain ||
        family.matrix_layout.vector_size != expected.vector_size ||
        family.matrix_layout.rows != expected.rows || family.matrix_layout.columns != expected.columns || family.matrix_layout.row_bits != expected.row_bits ||
        family.matrix_layout.column_bits != expected.column_bits || family.matrix_layout.column_generators.size() != expected.columns ||
        family.matrix_layout.flatten_order != expected.flatten_order || family.matrix_layout.generator_scheme != expected.generator_scheme ||
        !valid_canonical_generator_id(family.matrix_layout.trusted_generator_id)) {
        set_reason(reason, "native Lasso balanced M_f layout is not canonical");
        return false;
    }
    if (family.segment_offset > std::numeric_limits<std::size_t>::max() ||
        family.padded_queries > std::numeric_limits<std::uint64_t>::max() - family.segment_offset ||
        family.segment_offset + family.padded_queries > std::numeric_limits<std::size_t>::max()) {
        set_reason(reason, "native Lasso L_qry segment offset overflows");
        return false;
    }
    for (std::uint64_t index = 0; index < expected.columns; ++index) {
        const G1& generator = family.matrix_layout.column_generators[static_cast<std::size_t>(index)];
        G1 canonical;
        hashAndMapToG1(canonical, family.matrix_layout.trusted_generator_id + "/" + std::to_string(index));
        if (!valid_point(generator) || generator != canonical) {
            set_reason(reason, "native Lasso M_f generator is invalid");
            return false;
        }
    }
    return true;
}

bool validate_prover_family(const NativeLassoFamilyProver& family, std::string* reason) {
    if (!validate_public_family(family, reason) || (!family.selected_table_row_at && family.selected_table_rows.size() != family.padded_queries) ||
        (family.selected_table_row_at && !family.selected_table_rows.empty())) {
        set_reason(reason, "native Lasso prover witness dimensions are invalid");
        return false;
    }
    for (std::size_t row = 0; row < family.padded_queries; ++row) {
        const std::uint64_t selected = family.selected_table_row_at ? family.selected_table_row_at(row) : family.selected_table_rows[row];
        if (selected >= family.table.size() || (!family.queries.empty() && family.queries.size() != family.padded_queries) ||
            (!family.queries.empty() && family.queries[row] != family.table[selected])) {
            set_reason(reason, "native Lasso selected table row is invalid");
            return false;
        }
        if (row >= family.logical_queries && selected != family.padding_index) {
            set_reason(reason, "native Lasso padding row is not canonical");
            return false;
        }
    }
    return true;
}

F selected_query_value(const NativeLassoFamilyProver& family, std::uint64_t row) {
    if (row >= family.padded_queries) throw std::out_of_range("native Lasso selected query row is out of range");
    const std::uint64_t selected = family.selected_table_row_at ? family.selected_table_row_at(row) : family.selected_table_rows[static_cast<std::size_t>(row)];
    return family.table[static_cast<std::size_t>(selected)];
}

bool validate_query(const NativeLassoQueryProver& query, std::string* reason) {
    if (query.tensor_view.full_domain != 0) {
        if (!tensor_query_shape_valid(query.tensor_view, query.tensor_commitment, &query.value_provider, reason) ||
            (!query.statement_l_digest.empty() && !valid_digest(query.statement_l_digest)))
            return false;
        return true;
    }
    const std::uint64_t value_domain = query.value_provider.at ? query.value_provider.size : query.values.size();
    const std::uint64_t generator_domain = query.generator_provider.at ? query.generator_provider.size : query.generators.size();
    if (!valid_point(query.G) || value_domain > std::numeric_limits<std::size_t>::max() || !power_of_two(static_cast<std::size_t>(value_domain)) ||
        log2_power(static_cast<std::size_t>(value_domain)) >= 63 ||
        value_domain < 2 || generator_domain != value_domain ||
        !valid_point(query.commitment) ||
        (!query.basis_id.empty() && !valid_basis_identifier(query.basis_id)) ||
        (!query.statement_l_digest.empty() && !valid_digest(query.statement_l_digest)) ||
        (query.value_provider.at && !query.generator_provider.at)) {
        set_reason(reason, "native Lasso L_qry setup is invalid");
        return false;
    }
    if (query.generator_provider.at) {
        if (query.generator_provider.size != value_domain ||
            query.generator_provider.scheme != hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1 ||
            !valid_canonical_generator_id(query.generator_provider.canonical_id)) {
            set_reason(reason, "native Lasso L_qry generator provider is invalid");
            return false;
        }
    } else {
        if (query.generators.size() != value_domain || query.values.size() != value_domain) {
            set_reason(reason, "native Lasso L_qry dense fixture is incomplete");
            return false;
        }
        for (const G1& generator : query.generators) {
            if (!valid_point(generator)) {
                set_reason(reason, "native Lasso L_qry generator is invalid");
                return false;
            }
        }
    }
    if (query.has_commitment_blinding && (!valid_point(query.blinding_generator) || query.basis_id.empty() ||
                                          query.blinding_generator != hyrax::derive_pedersen_blinding_generator(query.basis_id))) {
        set_reason(reason, "native Lasso L_qry blinding identity is invalid");
        return false;
    }
    return true;
}

bool validate_query_public(const NativeLassoQueryPublic& query, std::size_t expected_size, std::string* reason) {
    if (query.tensor_view.full_domain != 0) {
        if (query.tensor_view.full_domain != expected_size || !tensor_query_shape_valid(query.tensor_view, query.tensor_commitment, nullptr, reason) ||
            (!query.statement_l_digest.empty() && !valid_digest(query.statement_l_digest)))
            return false;
        return true;
    }
    const std::uint64_t generator_domain = query.generator_provider.at ? query.generator_provider.size : query.generators.size();
    if (!valid_point(query.G) || !power_of_two(expected_size) || expected_size < 2 || log2_power(expected_size) >= 63 || generator_domain != expected_size ||
        !valid_point(query.commitment) || (!query.basis_id.empty() && !valid_basis_identifier(query.basis_id)) ||
        (!query.statement_l_digest.empty() && !valid_digest(query.statement_l_digest))) {
        set_reason(reason, "native Lasso public L_qry setup is invalid");
        return false;
    }
    if (query.generator_provider.at) {
        if (query.generator_provider.scheme != hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1 ||
            !valid_canonical_generator_id(query.generator_provider.canonical_id)) {
            set_reason(reason, "native Lasso public L_qry generator provider is invalid");
            return false;
        }
    } else {
        if (query.generators.size() != expected_size) {
            set_reason(reason, "native Lasso public L_qry dense fixture is incomplete");
            return false;
        }
        for (const G1& generator : query.generators) {
            if (!valid_point(generator)) {
                set_reason(reason, "native Lasso public L_qry generator is invalid");
                return false;
            }
        }
    }
    return true;
}

std::string family_prefix(const NativeLassoFamilyPublic& family) { return "lasso/" + family.family_id; }

bool is_lasso_protocol_message(const std::string& label) {
    return label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 ||
           label.compare(0, sizeof("challenge/lasso/") - 1, "challenge/lasso/") == 0 ||
           label.compare(0, sizeof("tensor/") - 1, "tensor/") == 0 ||
           label.compare(0, sizeof("challenge/tensor/") - 1, "challenge/tensor/") == 0;
}

std::string commitment_payload(const NativeLassoFamilyProver& family, const G1& commitment, const G1& G_for_digest) {
    return encode_tokens({family.family_id, family.table_id, family.resource_id, family.registration_id, std::to_string(family.padded_queries),
                          std::to_string(family.table.size()), std::to_string(family.segment_offset), std::to_string(family.logical_queries),
                          std::to_string(family.padding_index), family.padding_value.getStr(16), commitment.serializeToHexStr(),
                          native_lasso_generator_digest(G_for_digest, family.legacy_clear_matrix_generators), native_lasso_table_digest(family)});
}

G1 commit_all_ones(const std::vector<G1>& generators) {
    if (generators.empty() || generators.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native Lasso complement domain is invalid");
    std::vector<G1> generator_copy = generators;
    std::vector<F> ones(generators.size(), F_ONE);
    return perdersen_commit(generator_copy.data(), ones.data(), static_cast<int>(generators.size()));
}

bool parse_size(const std::string& encoded, std::size_t& value) {
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(encoded, &consumed, 10);
        if (consumed != encoded.size() || parsed > std::numeric_limits<std::size_t>::max()) return false;
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) { return false; }
}

bool parse_commitment(const NativeProofMessage& message, const NativeLassoFamilyPublic& expected, const G1& G, G1& commitment, std::string* reason) {
    std::vector<std::string> fields;
    if (message.kind != NativeProofMessageKind::Data || !decode_tokens(message.payload, 13, fields)) {
        set_reason(reason, "malformed native Lasso LC_f commitment");
        return false;
    }
    if (fields[0] != expected.family_id || fields[1] != expected.table_id || fields[2] != expected.resource_id || fields[3] != expected.registration_id) {
        set_reason(reason, "native Lasso LC_f identity mismatch");
        return false;
    }
    std::size_t rows = 0, cols = 0, offset = 0, logical = 0, padding = 0;
    if (!parse_size(fields[4], rows) || !parse_size(fields[5], cols) || !parse_size(fields[6], offset) || !parse_size(fields[7], logical) ||
        !parse_size(fields[8], padding) || rows != expected.padded_queries || cols != expected.table.size() || offset != expected.segment_offset ||
        logical != expected.logical_queries || padding != expected.padding_index) {
        set_reason(reason, "native Lasso LC_f shape mismatch");
        return false;
    }
    F padding_value;
    if (!decode_field(fields[9], padding_value) || padding_value != expected.padding_value || !decode_point(fields[10], commitment) ||
        fields[11] != native_lasso_generator_digest(G, expected.legacy_clear_matrix_generators) || fields[12] != native_lasso_table_digest(expected)) {
        set_reason(reason, "native Lasso LC_f authenticated identity mismatch");
        return false;
    }
    return true;
}

bool read_challenge(const NativeProofSession& session, std::size_t index, const std::string& label, F& value, std::string* reason) {
    if (index >= session.messages().size() || session.messages()[index].kind != NativeProofMessageKind::Challenge || session.messages()[index].label != label ||
        !decode_field(session.messages()[index].payload, value)) {
        set_reason(reason, "native Lasso challenge grammar mismatch at " + label);
        return false;
    }
    return true;
}

bool collect_challenges(const NativeProofSession& session, const std::string& label, std::size_t rounds, std::vector<F>& challenges, std::string* reason) {
    challenges.clear();
    for (std::size_t round = 0; round < rounds; ++round) {
        const std::size_t index = find_message(session, "challenge/" + label + "/r/" + std::to_string(round), NativeProofMessageKind::Challenge, reason);
        if (index == session.messages().size()) return false;
        F value;
        if (!read_challenge(session, index, "challenge/" + label + "/r/" + std::to_string(round), value, reason)) return false;
        challenges.push_back(value);
    }
    return true;
}

bool check_family_order(const std::vector<NativeLassoFamilyPublic>& families, std::string* reason) {
    if (families.empty()) {
        set_reason(reason, "native Lasso proof has no families");
        return false;
    }
    std::string previous;
    for (const NativeLassoFamilyPublic& family : families) {
        if (!validate_public_family(family, reason)) return false;
        if (!previous.empty() && family.family_id <= previous) {
            set_reason(reason, "native Lasso families are not canonically ordered");
            return false;
        }
        previous = family.family_id;
    }
    return true;
}

bool check_statement_identity_with_l_digest(const NativeProofSession& session, const std::vector<NativeLassoFamilyPublic>& families,
                                            const std::string& expected_l_digest, const std::string& query_statement_l_digest,
                                            std::string* reason) {
    if (!session.valid(reason)) return false;
    std::string adapter_tables_digest;
    std::string adapter_resources_digest;
    bool has_adapter_tables_digest = false;
    bool has_adapter_resources_digest = false;
    for (const NativeLassoFamilyPublic& family : families) {
        const bool has_tables = !family.statement_tables_digest.empty();
        const bool has_resources = !family.statement_resources_digest.empty();
        if (has_tables != has_resources) {
            set_reason(reason, "native Lasso adapter statement identity is incomplete");
            return false;
        }
        if (has_tables) {
            if (!valid_digest(family.statement_tables_digest) || !valid_digest(family.statement_resources_digest)) {
                set_reason(reason, "native Lasso adapter statement identity is non-canonical");
                return false;
            }
            if (!has_adapter_tables_digest) {
                adapter_tables_digest = family.statement_tables_digest;
                adapter_resources_digest = family.statement_resources_digest;
                has_adapter_tables_digest = true;
                has_adapter_resources_digest = true;
            } else if (adapter_tables_digest != family.statement_tables_digest || adapter_resources_digest != family.statement_resources_digest) {
                set_reason(reason, "native Lasso family statement identities disagree");
                return false;
            }
        }
    }
    const std::string expected_tables = has_adapter_tables_digest ? adapter_tables_digest : native_lasso_tables_digest(families);
    const std::string expected_resources = has_adapter_resources_digest ? adapter_resources_digest : native_lasso_resources_digest(families);
    if (session.statement().tables_digest != expected_tables) {
        set_reason(reason, "native Lasso table registry is not statement-bound");
        return false;
    }
    if (session.statement().resources_digest != expected_resources) {
        set_reason(reason, "native Lasso resource/registration identity is not statement-bound");
        return false;
    }
    if (!valid_digest(expected_l_digest) || (!query_statement_l_digest.empty() && query_statement_l_digest != expected_l_digest) ||
        session.statement().l_digest != expected_l_digest) {
        set_reason(reason, "native Lasso L_qry commitment is not statement-bound");
        return false;
    }
    return true;
}

bool check_statement_identity(const NativeProofSession& session, const std::vector<NativeLassoFamilyPublic>& families, const G1& query_commitment,
                              std::size_t query_size, const std::string& query_statement_l_digest, std::string* reason) {
    return check_statement_identity_with_l_digest(session, families, native_lasso_query_digest(query_commitment, query_size), query_statement_l_digest, reason);
}

}

const char* NativeLassoTerminalOpeningLabels::l_qry() { return "lasso/i-final/open"; }

const char* NativeLassoTerminalOpeningLabels::w_reserved() { return "reserved/open/W"; }

const char* NativeLassoTerminalOpeningLabels::aux_reserved() { return "reserved/open/Aux"; }

NativeLassoFamilyPublic::MatrixDescriptor native_lasso_balanced_matrix_layout(std::uint64_t padded_queries, std::uint64_t table_domain) {
    std::uint64_t vector_size = 0;
    if (padded_queries > std::numeric_limits<std::size_t>::max() || table_domain > std::numeric_limits<std::size_t>::max() ||
        !power_of_two(static_cast<std::size_t>(padded_queries)) || !power_of_two(static_cast<std::size_t>(table_domain)) ||
        !checked_product(padded_queries, table_domain, vector_size))
        throw std::invalid_argument("native Lasso balanced M_f domain is invalid");
    const std::uint32_t total_bits = static_cast<std::uint32_t>(log2_power(static_cast<std::size_t>(padded_queries)) +
                                                               log2_power(static_cast<std::size_t>(table_domain)));
    if (total_bits == 0 || total_bits >= 64) throw std::invalid_argument("native Lasso balanced M_f bit length is invalid");
    NativeLassoFamilyPublic::MatrixDescriptor result;
    result.padded_queries = padded_queries;
    result.table_domain = table_domain;
    result.vector_size = vector_size;
    result.row_bits = total_bits / 2;
    result.column_bits = total_bits - result.row_bits;
    result.rows = std::uint64_t(1) << result.row_bits;
    result.columns = std::uint64_t(1) << result.column_bits;
    return result;
}

NativeLassoSparseView native_lasso_sparse_view(const NativeLassoFamilyProver& family) {
    const std::uint64_t queries = family.padded_queries;
    const std::uint64_t table_domain = family.table.size();
    std::uint64_t full_domain = 0;
    if ((!family.selected_table_row_at && family.selected_table_rows.size() != queries) ||
        (family.selected_table_row_at && !family.selected_table_rows.empty()) || !checked_product(queries, table_domain, full_domain))
        throw std::invalid_argument("native Lasso sparse M_f witness dimensions are invalid");
    NativeLassoSparseView view;
    view.full_domain = full_domain;
    view.nonzero_count = queries;
    const NativeLassoFamilyProver* witness = &family;
    view.for_each_nonzero = [witness, queries, table_domain](const NativeLassoSparseEntryCallback& callback) {
        if (!callback) throw std::invalid_argument("native Lasso sparse M_f callback is empty");
        for (std::uint64_t query = 0; query < queries; ++query) {
            const std::uint64_t selected = witness->selected_table_row_at ? witness->selected_table_row_at(query) : witness->selected_table_rows[static_cast<std::size_t>(query)];
            if (selected >= table_domain || selected > (std::numeric_limits<std::uint64_t>::max() - query) / queries)
                throw std::overflow_error("native Lasso sparse M_f support index overflows");
            callback(selected * queries + query, F_ONE);
        }
    };
    view.value_at = [witness, queries, table_domain, full_domain](std::uint64_t index) {
        if (index >= full_domain) throw std::out_of_range("native Lasso sparse M_f index is out of range");
        const std::uint64_t query = index % queries;
        const std::uint64_t selected = index / queries;
        const std::uint64_t actual = witness->selected_table_row_at ? witness->selected_table_row_at(query) : witness->selected_table_rows[static_cast<std::size_t>(query)];
        return selected < table_domain && actual == selected ? F_ONE : F_ZERO;
    };
    view.evaluate_mle = [witness, queries, table_domain, full_domain](const std::vector<F>& point) {
        if (point.empty() || point.size() >= 64 || (std::uint64_t(1) << point.size()) != full_domain)
            throw std::invalid_argument("native Lasso sparse M_f evaluation point is invalid");
        F result = F_ZERO;
        const std::size_t query_bits = log2_power(static_cast<std::size_t>(queries));
        const std::vector<F> query_point(point.begin(), point.begin() + query_bits);
        const std::vector<F> table_point(point.begin() + query_bits, point.end());
        zkgpt::native_for_each_equality_weight(query_point, static_cast<std::size_t>(queries), [&](std::size_t query, const F& query_weight) {
            const std::uint64_t selected = witness->selected_table_row_at ? witness->selected_table_row_at(query) : witness->selected_table_rows[query];
            if (selected >= table_domain) throw std::out_of_range("native Lasso sparse M_f selected column is out of range");
            F table_weight = F_ONE;
            for (std::size_t bit = 0; bit < table_point.size(); ++bit)
                table_weight *= ((selected >> bit) & 1U) != 0 ? table_point[bit] : F_ONE - table_point[bit];
            result += query_weight * table_weight;
        });
        return result;
    };
    return view;
}

std::string native_lasso_matrix_layout_digest(const NativeLassoFamilyPublic::MatrixDescriptor& layout) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-balanced-rowtensor-v2");
    append_u64(encoded, layout.version);
    append_u64(encoded, layout.padded_queries);
    append_u64(encoded, layout.table_domain);
    append_u64(encoded, layout.vector_size);
    append_u64(encoded, layout.rows);
    append_u64(encoded, layout.columns);
    append_u64(encoded, layout.row_bits);
    append_u64(encoded, layout.column_bits);
    append_string(encoded, layout.flatten_order);
    append_string(encoded, layout.generator_scheme);
    append_string(encoded, layout.trusted_generator_id);
    append_u64(encoded, layout.column_generators.size());
    for (const G1& generator : layout.column_generators) append_string(encoded, generator.serializeToHexStr());
    return hash_fixed(encoded);
}

std::string native_lasso_table_digest(const NativeLassoFamilyPublic& family) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-table-v1");
    append_string(encoded, family.family_id);
    append_string(encoded, family.table_id);
    append_u64(encoded, family.table.size());
    for (const F& value : family.table) append_string(encoded, value.getStr(16));
    return hash_fixed(encoded);
}

std::string native_lasso_tables_digest(const std::vector<NativeLassoFamilyPublic>& families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-tables-v1");
    append_u64(encoded, families.size());
    for (const NativeLassoFamilyPublic& family : families) append_string(encoded, native_lasso_table_digest(family));
    return hash_fixed(encoded);
}

std::string native_lasso_resources_digest(const std::vector<NativeLassoFamilyPublic>& families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-resources-v2-balanced-rowtensor-chunked-4096");
    std::vector<const NativeLassoFamilyPublic*> ordered;
    ordered.reserve(families.size());
    for (const NativeLassoFamilyPublic& family : families) ordered.push_back(&family);
    std::sort(ordered.begin(), ordered.end(), [](const NativeLassoFamilyPublic* left, const NativeLassoFamilyPublic* right) {
        return left->family_id < right->family_id;
    });
    append_u64(encoded, ordered.size());
    for (const NativeLassoFamilyPublic* item : ordered) {
        const NativeLassoFamilyPublic& family = *item;
        append_string(encoded, family.family_id);
        append_string(encoded, family.table_id);
        append_string(encoded, family.resource_id);
        append_string(encoded, family.registration_id);
        append_u64(encoded, family.segment_offset);
        append_u64(encoded, family.logical_queries);
        append_u64(encoded, family.padded_queries);
        append_u64(encoded, family.padding_index);
        append_string(encoded, family.padding_value.getStr(16));
        append_string(encoded, native_lasso_matrix_layout_digest(family.matrix_layout));
    }
    return hash_fixed(encoded);
}

std::string native_lasso_query_digest(const G1& commitment, std::size_t size) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-query-v1");
    append_u64(encoded, size);
    append_string(encoded, commitment.serializeToHexStr());
    return hash_fixed(encoded);
}

std::string native_lasso_tensor_query_digest(const hyrax::HyraxTensorMleView& view, const hyrax::HyraxTensorMleCommitment& commitment) {
    std::vector<std::string> fields = {"zkgpt-native-lasso-query-balanced-tensor-v2",
                                       view.schema,
                                       view.method,
                                       std::to_string(view.version),
                                       view.role,
                                       view.basis_id,
                                       std::to_string(view.full_domain),
                                       std::to_string(view.logical_length),
                                       std::to_string(view.active_domain),
                                       std::to_string(view.rows),
                                       std::to_string(view.columns),
                                       std::to_string(static_cast<std::uint32_t>(view.column_generators.scheme)),
                                       view.column_generators.canonical_id,
                                       std::to_string(view.column_generators.size),
                                       commitment.schema,
                                       commitment.method,
                                       std::to_string(commitment.version),
                                       commitment.role,
                                       commitment.basis_id,
                                       std::to_string(static_cast<std::uint32_t>(commitment.column_generator_scheme)),
                                       commitment.column_generator_id,
                                       std::to_string(commitment.full_domain),
                                       std::to_string(commitment.logical_length),
                                       std::to_string(commitment.active_domain),
                                       std::to_string(commitment.rows),
                                       std::to_string(commitment.columns),
                                       commitment.root_digest};
    fields.reserve(fields.size() + commitment.row_commitments.size());
    for (const G1& row_commitment : commitment.row_commitments) fields.push_back(row_commitment.serializeToHexStr());
    return hash_fixed(encode_tokens(fields));
}

std::string native_lasso_generator_digest(const G1& G, const std::vector<G1>& generators) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-generators-v1");
    append_string(encoded, G.serializeToHexStr());
    append_u64(encoded, generators.size());
    for (const G1& generator : generators) append_string(encoded, generator.serializeToHexStr());
    return hash_fixed(encoded);
}

namespace native_lasso_legacy_clear {
bool native_lasso_prove_bundle(NativeProofSession& session, std::vector<NativeLassoFamilyProver>& families, NativeLassoQueryProver& query,
                               NativeLassoBundleResult* result, std::string* reason, const NativeLassoTensorTailProveHook& tensor_tail_stage) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (families.empty()) return false;

        query.commitment = perdersen_commit(query.generators.data(), query.values.data(), static_cast<int>(query.values.size()));
        if (!validate_query(query, reason)) return false;
        std::vector<NativeLassoFamilyPublic> identities;
        identities.reserve(families.size());
        for (NativeLassoFamilyProver& family : families) {
            if (!validate_prover_family(family, reason)) return false;
            identities.push_back(public_identity(family));
        }
        if (!check_family_order(identities, reason)) return false;
        if (!check_statement_identity(session, identities, query.commitment, query.values.size(), query.statement_l_digest, reason)) return false;
        for (const NativeLassoFamilyPublic& family : identities) {
            if (family.segment_offset + family.padded_queries > query.values.size()) {
                set_reason(reason, "native Lasso family exceeds L_qry segment");
                return false;
            }
        }

        NativeLassoBundleResult local;
        local.family_commitments.reserve(families.size());

        for (NativeLassoFamilyProver& family : families) {
            G1 commitment = perdersen_commit(family.legacy_clear_matrix_generators.data(), family.legacy_clear_matrix.data(), static_cast<int>(family.legacy_clear_matrix.size()));
            local.family_commitments.push_back(commitment);
            session.absorb(family_prefix(family) + "/commitment", commitment_payload(family, commitment, query.G));
        }

        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            NativeLassoFamilyProver& family = families[family_index];
            const std::string prefix = family_prefix(family);
            const std::size_t row_rounds = log2_power(family.padded_queries);
            std::vector<F> row_point;
            row_point.reserve(row_rounds);
            for (std::size_t round = 0; round < row_rounds; ++round) row_point.push_back(session.challenge(prefix + "/a-query/r/" + std::to_string(round)));
            local.family_query_points.push_back(row_point);
            const std::vector<F> row_basis = mle_basis(row_point);
            const std::size_t rows = family.padded_queries;
            const std::size_t cols = family.table.size();

            std::vector<F> matrix_minus_one = family.legacy_clear_matrix;
            for (F& value : matrix_minus_one) value -= F(1);
            session.absorb_field(prefix + "/b-boolean/claim", F_ZERO);
            NativeRangeResult booleanity;
            if (!native_range_prove_product(session, prefix + "/b-boolean/sum", family.legacy_clear_matrix, matrix_minus_one, F_ZERO, &booleanity, reason)) return false;
            if (booleanity.terminal_right != booleanity.terminal_left - F(1)) throw std::runtime_error("native Lasso M_f boolean terminal mismatch");

            std::vector<F> boolean_basis = mle_basis(booleanity.challenges);
            try {
                prove_dot_product_with_session(local.family_commitments[family_index], query.G * booleanity.terminal_left, boolean_basis.data(),
                                               family.legacy_clear_matrix_generators.data(), query.G, family.legacy_clear_matrix.data(), booleanity.terminal_left,
                                               static_cast<int>(family.legacy_clear_matrix.size()), session, prefix + "/c-boolean-open", false);
            } catch (const std::exception& error) { throw std::runtime_error(prefix + "/c-boolean-open: " + error.what()); }

            const G1 complement_commit = local.family_commitments[family_index] - commit_all_ones(family.legacy_clear_matrix_generators);
            try {
                prove_dot_product_with_session(complement_commit, query.G * booleanity.terminal_right, boolean_basis.data(), family.legacy_clear_matrix_generators.data(),
                                               query.G, matrix_minus_one.data(), booleanity.terminal_right, static_cast<int>(family.legacy_clear_matrix.size()), session,
                                               prefix + "/c-boolean-complement-open", false);
            } catch (const std::exception& error) { throw std::runtime_error(prefix + "/c-boolean-complement-open: " + error.what()); }

            std::vector<F> onehot_point;
            onehot_point.reserve(row_rounds);
            for (std::size_t round = 0; round < row_rounds; ++round) onehot_point.push_back(session.challenge(prefix + "/d-onehot/r/" + std::to_string(round)));
            const std::vector<F> onehot_row_basis = mle_basis(onehot_point);
            std::vector<F> onehot_left(cols, F_ZERO);
            for (std::size_t col = 0; col < cols; ++col)
                for (std::size_t row = 0; row < rows; ++row) onehot_left[col] += onehot_row_basis[row] * family.legacy_clear_matrix[row + col * rows];
            session.absorb_field(prefix + "/d-onehot/claim", F_ONE);
            NativeRangeResult onehot;
            if (!native_range_prove_deg1(session, prefix + "/d-onehot/sum", onehot_left, F_ONE, &onehot, reason)) return false;
            std::vector<F> onehot_matrix_point;
            onehot_matrix_point.reserve(onehot_point.size() + onehot.challenges.size());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot_point.begin(), onehot_point.end());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot.challenges.begin(), onehot.challenges.end());
            std::vector<F> onehot_basis = mle_basis(onehot_matrix_point);
            try {
                prove_dot_product_with_session(local.family_commitments[family_index], query.G * onehot.terminal, onehot_basis.data(),
                                               family.legacy_clear_matrix_generators.data(), query.G, family.legacy_clear_matrix.data(), onehot.terminal,
                                               static_cast<int>(family.legacy_clear_matrix.size()), session, prefix + "/e-onehot-open", false);
            } catch (const std::exception& error) { throw std::runtime_error(prefix + "/e-onehot-open: " + error.what()); }

            std::vector<F> left(cols, F_ZERO);
            for (std::size_t col = 0; col < cols; ++col)
                for (std::size_t row = 0; row < rows; ++row) left[col] += row_basis[row] * family.legacy_clear_matrix[row + col * rows];
            F ell = 0;
            for (std::size_t col = 0; col < cols; ++col) ell += left[col] * family.table[col];
            session.absorb_field(prefix + "/a-query/claim", ell);

            NativeRangeResult lookup;
            if (!native_range_prove_product(session, prefix + "/f-lookup", left, family.table, ell, &lookup, reason)) return false;
            const std::vector<F> y_point = lookup.challenges;
            std::vector<F> matrix_point;
            matrix_point.reserve(row_point.size() + y_point.size());
            matrix_point.insert(matrix_point.end(), row_point.begin(), row_point.end());
            matrix_point.insert(matrix_point.end(), y_point.begin(), y_point.end());
            std::vector<F> matrix_basis = mle_basis(matrix_point);
            F direct_matrix_eval = 0;
            for (std::size_t i = 0; i < matrix_basis.size(); ++i) direct_matrix_eval += matrix_basis[i] * family.legacy_clear_matrix[i];
            if (direct_matrix_eval != lookup.terminal_left)
                throw std::runtime_error(prefix + "/g-matrix-open direct eval mismatch: " + direct_matrix_eval.getStr(16) + " vs " +
                                         lookup.terminal_left.getStr(16));
            const G1& lc = local.family_commitments[family_index];
            try {
                prove_dot_product_with_session(lc, query.G * lookup.terminal_left, matrix_basis.data(), family.legacy_clear_matrix_generators.data(), query.G,
                                               family.legacy_clear_matrix.data(), lookup.terminal_left, static_cast<int>(family.legacy_clear_matrix.size()), session,
                                               prefix + "/g-matrix-open", false);
            } catch (const std::exception& error) { throw std::runtime_error(prefix + "/g-matrix-open: " + error.what()); }

            std::vector<F> selector_basis(query.values.size(), F_ZERO);
            for (std::size_t row = 0; row < rows; ++row) selector_basis[family.segment_offset + row] = row_basis[row];
            F segment_eval = 0;
            for (std::size_t row = 0; row < rows; ++row) {
                if (query.values[family.segment_offset + row] != family.queries[row]) throw std::runtime_error("native Lasso query/input segment mismatch");
                segment_eval += row_basis[row] * family.queries[row];
            }
            session.absorb_fields(prefix + "/h-input/claim", {ell, segment_eval});
            try {
                prove_dot_product_with_session(query.commitment, query.G * segment_eval, selector_basis.data(), query.generators.data(), query.G,
                                               query.values.data(), segment_eval, static_cast<int>(query.values.size()), session, prefix + "/h-input-open",
                                               true);
            } catch (const std::exception& error) { throw std::runtime_error(prefix + "/h-input-open: " + error.what()); }
        }

        const std::size_t final_rounds = log2_power(query.values.size());
        local.final_query_point.reserve(final_rounds);
        for (std::size_t round = 0; round < final_rounds; ++round)
            local.final_query_point.push_back(session.challenge("lasso/i-final/r/" + std::to_string(round)));
        std::vector<F> final_basis = mle_basis(local.final_query_point);
        F final_eval = 0;
        for (std::size_t i = 0; i < query.values.size(); ++i) final_eval += final_basis[i] * query.values[i];
        local.final_l_evaluation = final_eval;
        session.absorb_field("lasso/i-final/claim", final_eval);
        try {
            prove_dot_product_with_session(query.commitment, query.G * final_eval, final_basis.data(), query.generators.data(), query.G, query.values.data(),
                                           final_eval, static_cast<int>(query.values.size()), session, NativeLassoTerminalOpeningLabels::l_qry(), true);
        } catch (const std::exception& error) { throw std::runtime_error(std::string(NativeLassoTerminalOpeningLabels::l_qry()) + ": " + error.what()); }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception& error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}
}

namespace {

bool chunk_lasso_family_valid(const NativeLassoChunkFamilyPublic& family, std::string* reason) {
    if (!valid_identifier(family.family_id) || !valid_identifier(family.table_id) || !valid_identifier(family.resource_id) ||
        !valid_identifier(family.registration_id) || !power_of_two(family.table.size()) || family.table.size() > (1ULL << 16) ||
        !power_of_two(family.padded_queries) || family.padded_queries == 0 || family.logical_queries > family.padded_queries ||
        family.padding_index >= family.table.size() || family.padding_value != family.table[family.padding_index] || !power_of_two(family.chunk_size) ||
        family.chunk_size < 2 || family.chunk_size > (1ULL << 16) || family.padded_queries % family.chunk_size != 0) {
        set_reason(reason, "native Lasso chunk family shape is invalid");
        return false;
    }
    return true;
}

NativeLassoFamilyPublic chunk_legacy_public(const NativeLassoChunkFamilyPublic& family) {
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

std::string chunk_lasso_table_digest(const NativeLassoChunkFamilyPublic& family) { return native_lasso_table_digest(chunk_legacy_public(family)); }

std::string chunk_lasso_tables_digest(const std::vector<NativeLassoChunkFamilyPublic>& families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-chunk-tables-v1");
    for (const NativeLassoChunkFamilyPublic& family : families) append_string(encoded, chunk_lasso_table_digest(family));
    return hash_fixed(encoded);
}

std::string chunk_lasso_resources_digest(const std::vector<NativeLassoChunkFamilyPublic>& families) {
    std::string encoded;
    append_string(encoded, "zkgpt-native-lasso-chunk-resources-v1");
    for (const NativeLassoChunkFamilyPublic& family : families) {
        append_string(encoded, family.family_id);
        append_string(encoded, family.table_id);
        append_string(encoded, family.resource_id);
        append_string(encoded, family.registration_id);
        append_u64(encoded, family.segment_offset);
        append_u64(encoded, family.logical_queries);
        append_u64(encoded, family.padded_queries);
        append_u64(encoded, family.padding_index);
        append_string(encoded, family.padding_value.getStr(16));

        append_string(encoded, chunk_lasso_table_digest(family));
    }
    return hash_fixed(encoded);
}

std::string chunk_lasso_root(const NativeLassoChunkBundleWire& wire) {
    std::string encoded;
    append_string(encoded, wire.schema);
    append_u64(encoded, wire.version);
    append_string(encoded, wire.tables_digest);
    append_string(encoded, wire.resources_digest);
    for (const NativeLassoChunkFamilyWire& family : wire.families) {
        const NativeLassoChunkFamilyPublic& meta = family.public_metadata;
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
        for (const hyrax::ChunkedMleWire& column : family.matrix_columns) append_string(encoded, column.root_digest);
        append_string(encoded, family.lookup_terminal_commitment_hex);
        append_string(encoded, family.onehot_terminal_commitment_hex);
    }
    return hash_fixed(encoded);
}

bool scalar_commitment_from_hex(const std::string& encoded, const zkgpt::native_pedersen::PedersenBasis& basis,
                                zkgpt::native_pedersen::ScalarCommitment& result, std::string* reason) {
    G1 point;
    if (!decode_point(encoded, point)) {
        set_reason(reason, "native Lasso chunk scalar commitment is malformed");
        return false;
    }
    result.domain = basis.domain;
    result.point = point;
    return result.valid(basis, reason);
}

bool challenge_point(const NativeProofSession& session, const std::string& label, std::size_t rounds, std::vector<F>& point, std::string* reason) {
    point.clear();
    point.reserve(rounds);
    for (std::size_t index = 0; index < rounds; ++index) {
        const std::string challenge_label = "challenge/" + label + "/" + std::to_string(index);
        const std::size_t position = find_message(session, challenge_label, NativeProofMessageKind::Challenge, reason);
        F challenge;
        if (position == session.messages().size() || !decode_field(session.messages()[position].payload, challenge)) {
            set_reason(reason, "native Lasso chunk point challenge is malformed");
            return false;
        }
        point.push_back(challenge);
    }
    return true;
}

}

bool native_lasso_prove_chunked_bundle(NativeProofSession& session, std::vector<NativeLassoChunkFamilyProver>& families,
                                       const zkgpt::native_pedersen::PedersenBasis& scalar_basis, const std::vector<F>* global_point,
                                       NativeLassoChunkBundleProverResult* result, std::string* reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (families.empty() || !scalar_basis.valid(reason)) throw std::invalid_argument("native Lasso chunk bundle is empty or unbound");
        std::sort(families.begin(), families.end(),
                  [](const NativeLassoChunkFamilyProver& left, const NativeLassoChunkFamilyProver& right) { return left.family_id < right.family_id; });
        std::vector<NativeLassoChunkFamilyPublic> public_families;
        public_families.reserve(families.size());
        for (NativeLassoChunkFamilyProver& family : families) {
            if (!chunk_lasso_family_valid(family, reason) || !family.query_value)
                throw std::invalid_argument(reason != nullptr && !reason->empty() ? *reason : "native Lasso chunk query provider is empty");
            public_families.push_back(family);
        }
        const std::string tables_digest = chunk_lasso_tables_digest(public_families);
        const std::string resources_digest = chunk_lasso_resources_digest(public_families);
        NativeLassoChunkBundleProverResult local;
        local.wire.tables_digest = tables_digest;
        local.wire.resources_digest = resources_digest;
        std::vector<zkgpt::native_pedersen::ScalarOpening> final_operands;
        std::vector<Fr> final_coefficients;
        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            NativeLassoChunkFamilyProver& family = families[family_index];
            NativeLassoChunkFamilyWire family_wire;
            family_wire.public_metadata = family;
            family_wire.public_metadata.statement_tables_digest = tables_digest;
            family_wire.public_metadata.statement_resources_digest = resources_digest;
            const std::string prefix = "lasso/chunked/" + family.family_id;
            session.absorb(prefix + "/schedule", family.family_id + "|" + family.table_id + "|" + family.resource_id + "|" + family.registration_id + "|" +
                                                     std::to_string(family.segment_offset) + "|" + std::to_string(family.logical_queries) + "|" +
                                                     std::to_string(family.padded_queries) + "|" + std::to_string(family.chunk_size) + "|" + tables_digest +
                                                     "|" + resources_digest);
            const std::size_t row_rounds = log2_power(family.padded_queries);
            std::vector<F> row_point;
            row_point.reserve(row_rounds);
            if (global_point != nullptr) {
                if (global_point->size() < row_rounds) throw std::invalid_argument("native Lasso global point is too short");
                row_point.assign(global_point->begin(), global_point->begin() + row_rounds);
                session.absorb_fields(prefix + "/row-point", row_point);
            } else {
                for (std::size_t round = 0; round < row_rounds; ++round) row_point.push_back(session.challenge(prefix + "/row-point/" + std::to_string(round)));
            }
            family_wire.row_point_digest = hash_fixed(encode_tokens({"zkgpt-native-lasso-row-point-v1", family.family_id, [&]() {
                                                                         std::string value;
                                                                         for (const F& x : row_point) append_string(value, x.getStr(16));
                                                                         return value;
                                                                     }()}));
            hyrax::ChunkedMleProverResult query_result;
            if (!hyrax::prove_chunked_mle_opening(
                    session, prefix + "/query", family.padded_queries, family.padded_queries, family.chunk_size, row_point,
                    [&](std::uint64_t index) { return index < family.logical_queries ? family.query_value(index) : family.padding_value; }, scalar_basis,
                    query_result, reason))
                throw std::runtime_error(reason != nullptr ? *reason : "native Lasso query chunk opening failed");
            family_wire.query = query_result.wire;
            final_operands.push_back(query_result.aggregate_opening);
            Fr final_weight = F_ONE;
            if (global_point != nullptr) {
                const std::size_t row_bits = log2_power(family.padded_queries);
                const std::size_t block = family.segment_offset / family.padded_queries;
                for (std::size_t bit = row_bits; bit < global_point->size(); ++bit) {
                    const bool set = ((block >> (bit - row_bits)) & 1U) != 0;
                    final_weight *= set ? (*global_point)[bit] : (F_ONE - (*global_point)[bit]);
                }
            }
            final_coefficients.push_back(final_weight);

            std::vector<zkgpt::native_pedersen::ScalarOpening> matrix_operands;
            std::vector<Fr> matrix_coefficients;
            family_wire.matrix_columns.reserve(family.table.size());
            for (std::size_t column = 0; column < family.table.size(); ++column) {
                hyrax::ChunkedMleProverResult column_result;
                const std::string column_label = prefix + "/matrix/" + std::to_string(column);
                if (!hyrax::prove_chunked_mle_opening(
                        session, column_label, family.padded_queries, family.padded_queries, family.chunk_size, row_point,
                        [&](std::uint64_t row) {
                            if (row >= family.logical_queries) return family.table[column] == family.padding_value ? F_ONE : F_ZERO;
                            return family.query_value(row) == family.table[column] ? F_ONE : F_ZERO;
                        },
                        scalar_basis, column_result, reason))
                    throw std::runtime_error(reason != nullptr ? *reason : "native Lasso matrix chunk opening failed");
                family_wire.matrix_columns.push_back(column_result.wire);
                matrix_operands.push_back(column_result.aggregate_opening);
                matrix_coefficients.push_back(family.table[column]);
            }

            matrix_operands.push_back(query_result.aggregate_opening);
            matrix_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::prove_hidden_linear(session, scalar_basis, prefix + "/lookup-link", matrix_operands, matrix_coefficients, F_ZERO,
                                                             nullptr, reason))
                return false;
            family_wire.lookup_terminal_commitment_hex = query_result.aggregate_opening.commitment.point.serializeToHexStr();

            const zkgpt::native_pedersen::ScalarOpening onehot_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, F_ONE);
            std::vector<zkgpt::native_pedersen::ScalarOpening> onehot_operands;
            std::vector<Fr> onehot_coefficients;
            for (std::size_t column = 0; column < family_wire.matrix_columns.size(); ++column) {
                zkgpt::native_pedersen::ScalarCommitment commitment;
                if (!scalar_commitment_from_hex(family_wire.matrix_columns[column].aggregate_commitment_hex, scalar_basis, commitment, reason)) return false;

                onehot_operands.push_back(matrix_operands[column]);
                onehot_coefficients.push_back(F_ONE);
            }
            onehot_operands.push_back(onehot_opening);
            onehot_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::prove_hidden_linear(session, scalar_basis, prefix + "/onehot-link", onehot_operands, onehot_coefficients, F_ZERO,
                                                             nullptr, reason))
                return false;
            family_wire.onehot_terminal_commitment_hex = onehot_opening.commitment.point.serializeToHexStr();
            local.wire.families.push_back(std::move(family_wire));
        }
        local.final_l_evaluation = F_ZERO;
        if (global_point != nullptr) {
            const std::size_t global_bits = global_point->size();
            for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
                const NativeLassoChunkFamilyProver& family = families[family_index];
                const std::size_t row_bits = log2_power(family.padded_queries);
                const std::size_t block = family.segment_offset / family.padded_queries;
                Fr weight = F_ONE;
                for (std::size_t bit = row_bits; bit < global_bits; ++bit) {
                    const bool set = ((block >> (bit - row_bits)) & 1U) != 0;
                    weight *= set ? (*global_point)[bit] : (F_ONE - (*global_point)[bit]);
                }
                local.final_l_evaluation += weight * final_operands[family_index].value;
            }
        } else {
            for (const zkgpt::native_pedersen::ScalarOpening& opening : final_operands) local.final_l_evaluation += opening.value;
        }
        local.final_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, local.final_l_evaluation);
        final_operands.push_back(local.final_opening);
        final_coefficients.push_back(-F_ONE);
        if (!zkgpt::native_pedersen::prove_hidden_linear(session, scalar_basis, "lasso/chunked/final-link", final_operands, final_coefficients, F_ZERO, nullptr,
                                                         reason))
            return false;
        local.wire.final_commitment_hex = local.final_opening.commitment.point.serializeToHexStr();
        local.wire.root_digest = chunk_lasso_root(local.wire);
        session.absorb("lasso/chunked/root", local.wire.root_digest);
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception& error) {
        session.rollback(checkpoint);
        set_reason(reason, error.what());
        return false;
    }
}

bool native_lasso_verify_chunked_bundle(const NativeProofSession& session, const std::vector<NativeLassoChunkFamilyPublic>& families,
                                        const NativeLassoChunkBundleWire& wire, const zkgpt::native_pedersen::PedersenBasis& scalar_basis,
                                        const std::vector<F>* global_point, std::string* reason) {
    try {
        if (wire.schema != "zkgpt-native-lasso-chunked-v1" || wire.version != kNativeLassoChunkVersion || families.empty() ||
            wire.families.size() != families.size() || !scalar_basis.valid(reason)) {
            set_reason(reason, "native Lasso chunk wire schema is invalid");
            return false;
        }
        std::vector<NativeLassoChunkFamilyPublic> expected = families;
        std::sort(expected.begin(), expected.end(),
                  [](const NativeLassoChunkFamilyPublic& left, const NativeLassoChunkFamilyPublic& right) { return left.family_id < right.family_id; });
        const std::string tables_digest = chunk_lasso_tables_digest(expected);
        const std::string resources_digest = chunk_lasso_resources_digest(expected);
        if (wire.tables_digest != tables_digest || wire.resources_digest != resources_digest || wire.root_digest != chunk_lasso_root(wire)) {
            set_reason(reason, "native Lasso chunk root or table digest mismatch");
            return false;
        }
        std::vector<zkgpt::native_pedersen::ScalarCommitment> final_operands;
        std::vector<Fr> final_coefficients;
        for (std::size_t family_index = 0; family_index < expected.size(); ++family_index) {
            const NativeLassoChunkFamilyPublic& family = expected[family_index];
            const NativeLassoChunkFamilyWire& family_wire = wire.families[family_index];
            if (!chunk_lasso_family_valid(family, reason) || family_wire.public_metadata.family_id != family.family_id ||
                family_wire.public_metadata.table_id != family.table_id || family_wire.public_metadata.segment_offset != family.segment_offset ||
                family_wire.public_metadata.logical_queries != family.logical_queries || family_wire.public_metadata.padded_queries != family.padded_queries ||
                family_wire.public_metadata.chunk_size != family.chunk_size || family_wire.matrix_columns.size() != family.table.size()) {
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
                row_point.assign(global_point->begin(), global_point->begin() + row_rounds);
                const std::size_t position = find_message(session, prefix + "/row-point", NativeProofMessageKind::FieldVector, reason);
                if (position == session.messages().size()) return false;
                std::vector<F> encoded;
                if (!decode_vector(&session.messages()[position], row_rounds, encoded) || encoded != row_point) {
                    set_reason(reason, "native Lasso global row point binding mismatch");
                    return false;
                }
            } else if (!challenge_point(session, prefix + "/row-point", row_rounds, row_point, reason)) {
                return false;
            }
            const std::string expected_row_digest = hash_fixed(encode_tokens({"zkgpt-native-lasso-row-point-v1", family.family_id, [&]() {
                                                                                  std::string value;
                                                                                  for (const F& x : row_point) append_string(value, x.getStr(16));
                                                                                  return value;
                                                                              }()}));
            if (family_wire.row_point_digest != expected_row_digest) return false;
            if (!hyrax::verify_chunked_mle_opening(session, family_wire.query, prefix + "/query", family.padded_queries, family.padded_queries,
                                                   family.chunk_size, row_point, scalar_basis, reason))
                return false;
            zkgpt::native_pedersen::ScalarCommitment query_commitment;
            if (!scalar_commitment_from_hex(family_wire.query.aggregate_commitment_hex, scalar_basis, query_commitment, reason)) return false;
            final_operands.push_back(query_commitment);
            Fr final_weight = F_ONE;
            if (global_point != nullptr) {
                const std::size_t row_bits = log2_power(family.padded_queries);
                const std::size_t block = family.segment_offset / family.padded_queries;
                for (std::size_t bit = row_bits; bit < global_point->size(); ++bit) {
                    const bool set = ((block >> (bit - row_bits)) & 1U) != 0;
                    final_weight *= set ? (*global_point)[bit] : (F_ONE - (*global_point)[bit]);
                }
            }
            final_coefficients.push_back(final_weight);
            std::vector<zkgpt::native_pedersen::ScalarCommitment> matrix_commitments;
            matrix_commitments.reserve(family_wire.matrix_columns.size());
            for (std::size_t column = 0; column < family_wire.matrix_columns.size(); ++column) {
                const std::string column_label = prefix + "/matrix/" + std::to_string(column);
                if (!hyrax::verify_chunked_mle_opening(session, family_wire.matrix_columns[column], column_label, family.padded_queries, family.padded_queries,
                                                       family.chunk_size, row_point, scalar_basis, reason))
                    return false;
                zkgpt::native_pedersen::ScalarCommitment commitment;
                if (!scalar_commitment_from_hex(family_wire.matrix_columns[column].aggregate_commitment_hex, scalar_basis, commitment, reason)) return false;
                matrix_commitments.push_back(commitment);
            }
            std::vector<zkgpt::native_pedersen::ScalarCommitment> lookup_operands = matrix_commitments;
            std::vector<Fr> lookup_coefficients = family.table;
            lookup_operands.push_back(query_commitment);
            lookup_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::verify_hidden_linear(session, scalar_basis, prefix + "/lookup-link", lookup_operands, lookup_coefficients, F_ZERO,
                                                              reason))
                return false;
            zkgpt::native_pedersen::ScalarCommitment onehot_commitment;
            if (!scalar_commitment_from_hex(family_wire.onehot_terminal_commitment_hex, scalar_basis, onehot_commitment, reason)) return false;
            std::vector<zkgpt::native_pedersen::ScalarCommitment> onehot_operands = matrix_commitments;
            std::vector<Fr> onehot_coefficients(matrix_commitments.size(), F_ONE);
            onehot_operands.push_back(onehot_commitment);
            onehot_coefficients.push_back(-F_ONE);
            if (!zkgpt::native_pedersen::verify_hidden_linear(session, scalar_basis, prefix + "/onehot-link", onehot_operands, onehot_coefficients, F_ZERO,
                                                              reason))
                return false;
        }
        zkgpt::native_pedersen::ScalarCommitment final_commitment;
        if (!scalar_commitment_from_hex(wire.final_commitment_hex, scalar_basis, final_commitment, reason)) return false;
        final_operands.push_back(final_commitment);
        final_coefficients.push_back(-F_ONE);
        return zkgpt::native_pedersen::verify_hidden_linear(session, scalar_basis, "lasso/chunked/final-link", final_operands, final_coefficients, F_ZERO,
                                                            reason);
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

namespace {

struct MaskedFamilyState {
    std::string basis_id;
    std::vector<G1> row_commitments;
    std::vector<F> row_blindings;
};

class SessionCheckpointGuard {
  public:
    explicit SessionCheckpointGuard(NativeProofSession& session) : session_(session), checkpoint_(session.checkpoint()) {}
    SessionCheckpointGuard(const SessionCheckpointGuard&) = delete;
    SessionCheckpointGuard& operator=(const SessionCheckpointGuard&) = delete;
    ~SessionCheckpointGuard() {
        if (!committed_) session_.rollback(checkpoint_);
    }
    void commit() { committed_ = true; }

  private:
    NativeProofSession& session_;
    std::size_t checkpoint_;
    bool committed_ = false;
};

F mle_eval(const std::vector<F>& values, const std::vector<F>& point);
std::size_t relation_rounds(std::size_t size);

std::string query_basis_id(const NativeLassoQueryProver& query) {
    if (query.tensor_view.full_domain != 0) return query.tensor_view.basis_id;
    if (!query.basis_id.empty()) return query.basis_id;
    const std::string generator_id = query.generator_provider.at ? query.generator_provider.canonical_id :
                                                                   native_lasso_generator_digest(query.G, query.generators);
    const std::uint64_t domain = query.generator_provider.at ? query.generator_provider.size : query.generators.size();
    return hash_fixed(encode_tokens({"zkgpt-native-lasso-query-basis-v2", query.G.serializeToHexStr(), generator_id, std::to_string(domain)}));
}

std::string query_basis_id(const NativeLassoQueryPublic& query) {
    if (query.tensor_view.full_domain != 0) return query.tensor_view.basis_id;
    if (!query.basis_id.empty()) return query.basis_id;
    const std::string generator_id = query.generator_provider.at ? query.generator_provider.canonical_id :
                                                                   native_lasso_generator_digest(query.G, query.generators);
    const std::uint64_t domain = query.generator_provider.at ? query.generator_provider.size : query.generators.size();
    return hash_fixed(encode_tokens({"zkgpt-native-lasso-query-basis-v2", query.G.serializeToHexStr(), generator_id, std::to_string(domain)}));
}

std::string family_basis_id(const NativeProofSession& session, const NativeLassoFamilyPublic& family) {
    return hash_fixed(encode_tokens(
        {"zkgpt-native-lasso-matrix-basis-v2", session.statement_digest(), family.family_id, family.table_id, family.resource_id, family.registration_id,
         native_lasso_matrix_layout_digest(family.matrix_layout)}));
}

bool valid_basis_id(const std::string& value) {
    if (value.empty() || value.size() > 128) return false;
    for (char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == '-' || c == '_')) return false;
    return true;
}

hyrax::PedersenMleCommitment query_commitment_from(const NativeLassoQueryProver& query) {
    const std::string basis_id = query_basis_id(query);
    hyrax::PedersenMleCommitment result;
    result.commitment = query.commitment;
    result.basis_id = basis_id;
    result.blinding_generator = hyrax::derive_pedersen_blinding_generator(basis_id);
    result.blinding = query.has_commitment_blinding ? query.commitment_blinding : F_ZERO;
    if (query.has_commitment_blinding && valid_point(query.blinding_generator) && query.blinding_generator != result.blinding_generator)
        throw std::invalid_argument("native Lasso query blinding identity mismatch");
    return result;
}

hyrax::PedersenMleCommitment query_commitment_from(const NativeLassoQueryPublic& query) {
    const std::string basis_id = query_basis_id(query);
    hyrax::PedersenMleCommitment result;
    result.commitment = query.commitment;
    result.basis_id = basis_id;
    result.blinding_generator = hyrax::derive_pedersen_blinding_generator(basis_id);
    result.blinding = F_ZERO;
    return result;
}

std::uint64_t query_domain(const NativeLassoQueryProver& query) {
    if (query.tensor_view.full_domain != 0) return query.tensor_view.full_domain;
    return query.value_provider.at ? query.value_provider.size : query.values.size();
}

std::uint64_t query_domain(const NativeLassoQueryPublic& query) {
    if (query.tensor_view.full_domain != 0) return query.tensor_view.full_domain;
    return query.generator_provider.at ? query.generator_provider.size : query.generators.size();
}

hyrax::HyraxGeneratorProvider query_generators(const NativeLassoQueryProver& query) {
    if (query.generator_provider.at) return query.generator_provider;
    hyrax::HyraxGeneratorProvider provider;
    provider.size = query.generators.size();
    provider.canonical_id = native_lasso_generator_digest(query.G, query.generators);
    provider.scheme = hyrax::HyraxGeneratorProviderScheme::OpaqueProvider;
    provider.at = [&query](std::uint64_t index) {
        if (index >= query.generators.size()) throw std::out_of_range("native Lasso L_qry generator index is out of range");
        return query.generators[static_cast<std::size_t>(index)];
    };
    return provider;
}

hyrax::HyraxGeneratorProvider query_generators(const NativeLassoQueryPublic& query) {
    if (query.generator_provider.at) return query.generator_provider;
    hyrax::HyraxGeneratorProvider provider;
    provider.size = query.generators.size();
    provider.canonical_id = native_lasso_generator_digest(query.G, query.generators);
    provider.scheme = hyrax::HyraxGeneratorProviderScheme::OpaqueProvider;
    provider.at = [&query](std::uint64_t index) {
        if (index >= query.generators.size()) throw std::out_of_range("native Lasso public L_qry generator index is out of range");
        return query.generators[static_cast<std::size_t>(index)];
    };
    return provider;
}

hyrax::HyraxScalarProvider query_values(const NativeLassoQueryProver& query) {
    if (query.value_provider.at) return query.value_provider;
    hyrax::HyraxScalarProvider provider;
    provider.size = query.values.size();
    provider.at = [&query](std::uint64_t index) {
        if (index >= query.values.size()) throw std::out_of_range("native Lasso L_qry value index is out of range");
        return query.values[static_cast<std::size_t>(index)];
    };
    return provider;
}

hyrax::HyraxMleBasisView query_view(const NativeLassoQueryProver& query, const std::vector<F>& point, const std::string& statement_basis_digest) {
    hyrax::HyraxMleBasisView view;
    view.full_domain = query_domain(query);
    view.logical_length = view.full_domain;
    view.active_domain = view.full_domain;
    view.role = "native-lasso-L_qry";
    view.basis_id = query_basis_id(query);
    view.point = point;
    view.generators = query_generators(query);
    if (!query.generator_provider.at) view.generators.canonical_id = statement_basis_digest;
    return view;
}

hyrax::HyraxMleBasisView query_view(const NativeLassoQueryPublic& query, const std::vector<F>& point, const std::string& statement_basis_digest) {
    hyrax::HyraxMleBasisView view;
    view.full_domain = query_domain(query);
    view.logical_length = view.full_domain;
    view.active_domain = view.full_domain;
    view.role = "native-lasso-L_qry";
    view.basis_id = query_basis_id(query);
    view.point = point;
    view.generators = query_generators(query);
    if (!query.generator_provider.at) view.generators.canonical_id = statement_basis_digest;
    return view;
}

struct SparseValue {
    std::uint64_t index = 0;
    F value = F_ZERO;
    F remaining_nonzero_product = F_ONE;
    std::size_t remaining_zero_factors = 0;
};

using SparseMatrixValues = zkgpt::NativeVector<SparseValue>;

F equality_boolean_factor(const F& beta, bool bit) { return bit ? beta : F_ONE - beta; }

void remove_equality_factor(const SparseValue& value, const F& factor, const F& factor_inverse, F& suffix_nonzero_product,
                            std::size_t& suffix_zero_factors) {
    suffix_nonzero_product = value.remaining_nonzero_product;
    suffix_zero_factors = value.remaining_zero_factors;
    if (factor.isZero()) {
        if (suffix_zero_factors == 0) throw std::logic_error("native Lasso equality-factor state is inconsistent");
        --suffix_zero_factors;
    } else {
        suffix_nonzero_product *= factor_inverse;
    }
}

SparseMatrixValues sparse_matrix_values(const NativeLassoFamilyProver& family, const std::vector<F>& beta) {
    const NativeLassoSparseView sparse = native_lasso_sparse_view(family);
    if (beta.size() != log2_power(static_cast<std::size_t>(sparse.full_domain)))
        throw std::invalid_argument("native Lasso booleanity weight dimension mismatch");
    SparseMatrixValues values;
    values.reserve(static_cast<std::size_t>(sparse.nonzero_count));
    sparse.for_each_nonzero([&](std::uint64_t index, const F& value) {
        SparseValue entry;
        entry.index = index;
        entry.value = value;
        for (std::size_t bit = 0; bit < beta.size(); ++bit) {
            const F factor = equality_boolean_factor(beta[bit], ((index >> bit) & 1U) != 0);
            if (factor.isZero())
                ++entry.remaining_zero_factors;
            else
                entry.remaining_nonzero_product *= factor;
        }
        values.push_back(entry);
    });
    std::sort(values.begin(), values.end(), [](const SparseValue& left, const SparseValue& right) { return left.index < right.index; });
    for (std::size_t index = 1; index < values.size(); ++index)
        if (values[index - 1].index == values[index].index) throw std::invalid_argument("native Lasso sparse M_f support is not unique");
    return values;
}

// round_trace is local smoke instrumentation only and is never absorbed into the proof transcript.
void append_weighted_boolean_round(NativeMaskedSumcheckProver& prover, SparseMatrixValues& values, const std::vector<F>& beta,
                                  std::size_t round, F& prefix_weight, std::string* reason, F* challenge_out = nullptr,
                                  std::vector<F>* round_trace = nullptr) {
    if (round >= beta.size()) throw std::invalid_argument("native Lasso booleanity round is out of range");
    std::vector<F> polynomial(4, F_ZERO);
    const F zero_factor = F_ONE - beta[round];
    const F one_factor = beta[round];
    const F zero_factor_inverse = zero_factor.isZero() ? F_ZERO : F_ONE / zero_factor;
    const F one_factor_inverse = one_factor.isZero() ? F_ZERO : F_ONE / one_factor;
    std::size_t cursor = 0;
    while (cursor < values.size()) {
        const std::uint64_t pair = values[cursor].index >> 1;
        const SparseValue* even = nullptr;
        const SparseValue* odd = nullptr;
        while (cursor < values.size() && (values[cursor].index >> 1) == pair) {
            if ((values[cursor].index & 1U) == 0)
                even = &values[cursor];
            else
                odd = &values[cursor];
            ++cursor;
        }
        const F zero = even == nullptr ? F_ZERO : even->value;
        const F one = odd == nullptr ? F_ZERO : odd->value;
        const SparseValue& suffix_source = even == nullptr ? *odd : *even;
        const bool source_bit = even == nullptr;
        F suffix_nonzero_product;
        std::size_t suffix_zero_factors = 0;
        remove_equality_factor(suffix_source, source_bit ? one_factor : zero_factor, source_bit ? one_factor_inverse : zero_factor_inverse,
                               suffix_nonzero_product, suffix_zero_factors);
        const F suffix_weight = suffix_zero_factors == 0 ? suffix_nonzero_product : F_ZERO;
        for (std::size_t evaluation = 0; evaluation < polynomial.size(); ++evaluation) {
            const F x(static_cast<int>(evaluation));
            const F matrix_value = zero + x * (one - zero);
            const F current_weight = F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * x;
            polynomial[evaluation] += prefix_weight * suffix_weight * current_weight * matrix_value * (matrix_value - F_ONE);
        }
        if (even != nullptr && odd != nullptr) {
            F other_nonzero_product;
            std::size_t other_zero_factors = 0;
            remove_equality_factor(*odd, one_factor, one_factor_inverse, other_nonzero_product, other_zero_factors);
            const F other_suffix_weight = other_zero_factors == 0 ? other_nonzero_product : F_ZERO;
            if (other_suffix_weight != suffix_weight) throw std::logic_error("native Lasso sparse equality tails disagree");
        }
    }
    if (round_trace != nullptr) *round_trace = polynomial;
    F challenge;
    if (!prover.append_round(polynomial, &challenge, reason)) throw std::runtime_error(reason != nullptr ? *reason : "weighted Lasso booleanity round failed");
    cursor = 0;
    std::size_t write = 0;
    while (cursor < values.size()) {
        const std::uint64_t pair_index = values[cursor].index >> 1;
        const SparseValue* even = nullptr;
        const SparseValue* odd = nullptr;
        while (cursor < values.size() && (values[cursor].index >> 1) == pair_index) {
            if ((values[cursor].index & 1U) == 0) even = &values[cursor];
            else odd = &values[cursor];
            ++cursor;
        }
        const F zero = even == nullptr ? F_ZERO : even->value;
        const F one = odd == nullptr ? F_ZERO : odd->value;
        const F folded = zero + challenge * (one - zero);
        if (folded.isZero()) continue;
        const SparseValue& suffix_source = even == nullptr ? *odd : *even;
        const bool source_bit = even == nullptr;
        F suffix_nonzero_product;
        std::size_t suffix_zero_factors = 0;
        remove_equality_factor(suffix_source, source_bit ? one_factor : zero_factor, source_bit ? one_factor_inverse : zero_factor_inverse,
                               suffix_nonzero_product, suffix_zero_factors);
        if (even != nullptr && odd != nullptr) {
            F other_nonzero_product;
            std::size_t other_zero_factors = 0;
            remove_equality_factor(*odd, one_factor, one_factor_inverse, other_nonzero_product, other_zero_factors);
            if (other_nonzero_product != suffix_nonzero_product || other_zero_factors != suffix_zero_factors)
                throw std::logic_error("native Lasso sparse equality tails disagree after fold");
        }
        SparseValue entry{pair_index, folded, suffix_nonzero_product, suffix_zero_factors};
        values[write++] = entry;
    }
    prefix_weight *= F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * challenge;
    values.resize(write);
    if (challenge_out != nullptr) *challenge_out = challenge;
}

struct WeightedBooleanityFold {
    F matrix_value = F_ZERO;
    std::vector<F> point;
};

std::vector<F> weighted_booleanity_t2_round_polynomial(const zkgpt::NativeFieldVector& column_one, const std::vector<F>& beta, std::size_t round, const F& prefix_weight) {
    if (beta.empty() || round >= beta.size()) throw std::invalid_argument("native Lasso T2 Booleanity round is out of range");
    const std::size_t query_bits = beta.size() - 1;
    std::vector<F> polynomial(4, F_ZERO);
    if (round < query_bits) {
        const std::size_t pair_count = column_one.size() / 2;
        if (column_one.size() != (std::size_t(1) << (query_bits - round))) throw std::invalid_argument("native Lasso T2 query fold size is invalid");
        const std::vector<F> suffix_point(beta.begin() + round + 1, beta.begin() + query_bits);
        zkgpt::native_for_each_equality_weight(suffix_point, pair_count, [&](std::size_t pair, const F& suffix_weight) {
            const F zero = column_one[pair * 2], one = column_one[pair * 2 + 1];
            for (std::size_t evaluation = 0; evaluation < polynomial.size(); ++evaluation) {
                const F x(static_cast<int>(evaluation));
                const F matrix_value = zero + x * (one - zero);
                const F current_weight = F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * x;
                polynomial[evaluation] += prefix_weight * suffix_weight * current_weight * matrix_value * (matrix_value - F_ONE);
            }
        });
        return polynomial;
    }
    if (column_one.size() != 1) throw std::invalid_argument("native Lasso T2 column fold is not scalar");
    const F b = column_one[0], m0 = F_ONE - b, m1 = b;
    for (std::size_t evaluation = 0; evaluation < polynomial.size(); ++evaluation) {
        const F x(static_cast<int>(evaluation));
        const F matrix_value = m0 + x * (m1 - m0);
        const F column_weight = F_ONE - beta[query_bits] + (F(2) * beta[query_bits] - F_ONE) * x;
        polynomial[evaluation] = prefix_weight * column_weight * matrix_value * (matrix_value - F_ONE);
    }
    return polynomial;
}

WeightedBooleanityFold append_weighted_booleanity_t2(NativeMaskedSumcheckProver& prover, const NativeLassoFamilyProver& family,
                                                      const std::vector<F>& beta, std::string* reason) {
    const std::uint64_t queries = family.padded_queries;
    const std::size_t query_bits = relation_rounds(static_cast<std::size_t>(queries));
    if (family.table.size() != 2 || beta.size() != query_bits + 1) throw std::invalid_argument("native Lasso T2 Booleanity dimensions are invalid");
    zkgpt::NativeFieldVector column_one;
    column_one.resize(static_cast<std::size_t>(queries));
    for (std::uint64_t query = 0; query < queries; ++query) {
        const std::uint64_t selected = family.selected_table_row_at ? family.selected_table_row_at(query) : family.selected_table_rows[static_cast<std::size_t>(query)];
        if (selected >= 2) throw std::invalid_argument("native Lasso T2 selected column is invalid");
        column_one[static_cast<std::size_t>(query)] = selected == 1 ? F_ONE : F_ZERO;
    }
    WeightedBooleanityFold result;
    result.point.reserve(beta.size());
    F prefix_weight = F_ONE;
    for (std::size_t round = 0; round < query_bits; ++round) {
        const std::size_t pair_count = column_one.size() / 2;
        const std::vector<F> polynomial = weighted_booleanity_t2_round_polynomial(column_one, beta, round, prefix_weight);
        F challenge;
        if (!prover.append_round(polynomial, &challenge, reason)) throw std::runtime_error(reason != nullptr ? *reason : "native Lasso T2 query Booleanity round failed");
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const F zero = column_one[pair * 2], one = column_one[pair * 2 + 1];
            column_one[pair] = zero + challenge * (one - zero);
        }
        column_one.resize(pair_count);
        prefix_weight *= F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * challenge;
        result.point.push_back(challenge);
    }
    const F b = column_one[0];
    const F m0 = F_ONE - b, m1 = b;
    const std::vector<F> polynomial = weighted_booleanity_t2_round_polynomial(column_one, beta, query_bits, prefix_weight);
    F column_challenge;
    if (!prover.append_round(polynomial, &column_challenge, reason)) throw std::runtime_error(reason != nullptr ? *reason : "native Lasso T2 column Booleanity round failed");
    result.point.push_back(column_challenge);
    result.matrix_value = m0 + column_challenge * (m1 - m0);
    return result;
}

std::shared_ptr<const NativeMaskedPublicProvider> equality_weight_provider(const std::vector<F>& beta) {
    if (beta.empty() || beta.size() >= 63) throw std::invalid_argument("native Lasso equality weight point is invalid");
    const std::uint64_t domain = std::uint64_t(1) << beta.size();
    auto provider = std::make_shared<NativeMaskedPublicProvider>();
    provider->size = domain;
    const auto shared_beta = std::make_shared<const std::vector<F>>(beta);
    provider->value_at = [shared_beta, domain](std::uint64_t index) {
        if (index >= domain) throw std::out_of_range("native Lasso equality weight index is out of range");
        F value = F_ONE;
        for (std::size_t bit = 0; bit < shared_beta->size(); ++bit)
            value *= equality_boolean_factor((*shared_beta)[bit], ((index >> bit) & 1U) != 0);
        return value;
    };
    provider->evaluate_mle = [shared_beta](const std::vector<F>& point) {
        if (point.size() != shared_beta->size()) throw std::invalid_argument("native Lasso equality weight terminal dimension mismatch");
        F value = F_ONE;
        for (std::size_t bit = 0; bit < point.size(); ++bit)
            value *= F_ONE - (*shared_beta)[bit] + (F(2) * (*shared_beta)[bit] - F_ONE) * point[bit];
        return value;
    };
    return provider;
}

hyrax::HyraxGeneratorProvider matrix_generators(const NativeLassoFamilyPublic& family) {
    hyrax::HyraxGeneratorProvider provider;
    provider.size = family.matrix_layout.columns;
    provider.canonical_id = family.matrix_layout.trusted_generator_id;
    provider.scheme = hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    provider.at = [&family](std::uint64_t index) {
        if (index >= family.matrix_layout.column_generators.size()) throw std::out_of_range("native Lasso column generator index is out of range");
        return family.matrix_layout.column_generators[static_cast<std::size_t>(index)];
    };
    return provider;
}

void commit_family_rows(const NativeLassoFamilyProver& family, MaskedFamilyState& state) {
    const std::uint64_t rows = family.matrix_layout.rows;
    const NativeLassoSparseView sparse = native_lasso_sparse_view(family);
    const G1 H = hyrax::derive_pedersen_blinding_generator(state.basis_id);
    state.row_commitments.reserve(static_cast<std::size_t>(rows));
    state.row_blindings.reserve(static_cast<std::size_t>(rows));
    for (std::uint64_t row = 0; row < rows; ++row) {
        F rho;
        rho.setByCSPRNG();
        state.row_commitments.push_back(H * rho);
        state.row_blindings.push_back(rho);
    }
    sparse.for_each_nonzero([&](std::uint64_t index, const F& value) {
        if (value != F_ONE) throw std::invalid_argument("native Lasso M_f is not one-hot");
        const std::uint64_t row = index >> family.matrix_layout.column_bits;
        const std::uint64_t column = index & (family.matrix_layout.columns - 1);
        if (row >= rows) throw std::out_of_range("native Lasso M_f row is out of range");
        state.row_commitments[static_cast<std::size_t>(row)] += family.matrix_layout.column_generators[static_cast<std::size_t>(column)];
    });
}

struct MatrixRowFold {
    hyrax::PedersenMleCommitment commitment;
    hyrax::HyraxMleBasisView view;
    hyrax::HyraxScalarProvider values;
    std::shared_ptr<std::vector<F>> folded_values;
    F raw_evaluation = F_ZERO;
};

MatrixRowFold fold_family_rows(const NativeLassoFamilyProver& family, const MaskedFamilyState& state, const std::vector<F>& point,
                               bool complement = false) {
    const std::uint64_t rows = family.matrix_layout.rows;
    const std::uint64_t columns = family.matrix_layout.columns;
    const std::uint32_t column_bits = family.matrix_layout.column_bits;
    if (point.size() != static_cast<std::size_t>(family.matrix_layout.row_bits + column_bits) || state.row_blindings.size() != rows)
        throw std::invalid_argument("native Lasso row fold dimensions are invalid");
    const std::vector<F> column_point(point.begin(), point.begin() + column_bits);
    const std::vector<F> row_point(point.begin() + column_bits, point.end());
    const std::vector<F> row_weights = mle_basis(row_point);
    if (row_weights.size() != rows) throw std::invalid_argument("native Lasso row fold challenge count is invalid");
    auto folded_values = std::make_shared<std::vector<F>>(static_cast<std::size_t>(columns), F_ZERO);
    F row_weight_sum = F_ZERO;
    F folded_blinding = F_ZERO;
    for (std::uint64_t row = 0; row < rows; ++row) {
        const F& weight = row_weights[static_cast<std::size_t>(row)];
        row_weight_sum += weight;
        folded_blinding += weight * state.row_blindings[static_cast<std::size_t>(row)];
    }
    const NativeLassoSparseView sparse = native_lasso_sparse_view(family);
    sparse.for_each_nonzero([&](std::uint64_t index, const F& value) {
        const std::uint64_t row = index >> column_bits;
        const std::uint64_t column = index & (columns - 1);
        (*folded_values)[static_cast<std::size_t>(column)] += row_weights[static_cast<std::size_t>(row)] * value;
    });
    if (complement) {
        for (F& value : *folded_values) value -= row_weight_sum;
    }
    MatrixRowFold result;
    result.folded_values = folded_values;
    result.values.size = columns;
    result.values.at = [folded_values](std::uint64_t index) {
        if (index >= folded_values->size()) throw std::out_of_range("native Lasso row-folded M_f index is out of range");
        return (*folded_values)[static_cast<std::size_t>(index)];
    };
    result.commitment.basis_id = state.basis_id;
    result.commitment.blinding_generator = hyrax::derive_pedersen_blinding_generator(state.basis_id);
    result.commitment.blinding = folded_blinding;
    result.commitment.commitment = result.commitment.blinding_generator * folded_blinding;
    for (std::uint64_t column = 0; column < columns; ++column)
        result.commitment.commitment += family.matrix_layout.column_generators[static_cast<std::size_t>(column)] * (*folded_values)[static_cast<std::size_t>(column)];
    result.view.full_domain = columns;
    result.view.logical_length = columns;
    result.view.active_domain = columns;
    result.view.role = complement ? "native-lasso-M_f-minus-one-rowfold-v2" : "native-lasso-M_f-rowfold-v2";
    result.view.basis_id = state.basis_id;
    result.view.point = column_point;
    result.view.generators = matrix_generators(family);
    result.raw_evaluation = mle_eval(*folded_values, column_point);
    return result;
}

zkgpt::native_pedersen::ScalarOpening scalar_evaluation_opening(const zkgpt::native_pedersen::PedersenBasis& basis, const F& value) {
    return zkgpt::native_pedersen::commit_fresh(basis, value);
}

zkgpt::native_pedersen::PedersenBasis lasso_scalar_basis(const NativeProofSession& session) {
    return zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + session.statement_digest());
}

zkgpt::native_pedersen::ScalarOpening public_scalar_opening(const zkgpt::native_pedersen::PedersenBasis& basis, const F& value) {
    zkgpt::native_pedersen::ScalarOpening result;
    result.commitment = zkgpt::native_pedersen::commit(basis, value, F_ZERO);
    result.value = value;
    result.blinding = F_ZERO;
    return result;
}

bool make_eval_opening(const zkgpt::native_pedersen::PedersenBasis& basis, const std::vector<F>& point, const std::vector<F>& values,
                       zkgpt::native_pedersen::ScalarOpening& opening, std::string* reason) {
    return hyrax::make_mle_evaluation_opening(basis, point, values, opening, reason);
}

std::vector<F> fold_linear(std::vector<F> values, const F& challenge) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) throw std::invalid_argument("native masked Lasso linear domain is invalid");
    const std::size_t half = values.size() / 2;
    std::vector<F> result(half);
    for (std::size_t index = 0; index < half; ++index) result[index] = values[2 * index] + challenge * (values[2 * index + 1] - values[2 * index]);
    return result;
}

void append_linear_round(NativeMaskedSumcheckProver& prover, const std::vector<F>& values, std::vector<F>& next, std::string* reason,
                         F* challenge_out = nullptr) {
    if (values.size() < 2 || (values.size() & (values.size() - 1)) != 0) throw std::invalid_argument("native masked Lasso linear round domain is invalid");
    F p0 = F_ZERO, p1 = F_ZERO;
    for (std::size_t index = 0; index < values.size() / 2; ++index) {
        p0 += values[2 * index];
        p1 += values[2 * index + 1];
    }
    F challenge;
    if (!prover.append_round({p0, p1}, &challenge, reason)) throw std::runtime_error(reason != nullptr ? *reason : "masked Lasso linear round failed");
    next = fold_linear(values, challenge);
    if (challenge_out != nullptr) *challenge_out = challenge;
}

std::vector<F> fold_product(std::vector<F> left, std::vector<F> right, const F& challenge, std::vector<F>& next_right) {
    if (left.empty() || left.size() != right.size() || (left.size() & (left.size() - 1)) != 0)
        throw std::invalid_argument("native masked Lasso product domain is invalid");
    const std::size_t half = left.size() / 2;
    std::vector<F> next_left(half);
    next_right.resize(half);
    for (std::size_t index = 0; index < half; ++index) {
        next_left[index] = left[2 * index] + challenge * (left[2 * index + 1] - left[2 * index]);
        next_right[index] = right[2 * index] + challenge * (right[2 * index + 1] - right[2 * index]);
    }
    return next_left;
}

void append_product_round(NativeMaskedSumcheckProver& prover, const std::vector<F>& left, const std::vector<F>& right, std::vector<F>& next_left,
                          std::vector<F>& next_right, std::string* reason, F* challenge_out = nullptr) {
    if (left.size() < 2 || left.size() != right.size() || (left.size() & (left.size() - 1)) != 0)
        throw std::invalid_argument("native masked Lasso product round domain is invalid");
    F p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
    for (std::size_t index = 0; index < left.size() / 2; ++index) {
        const F l0 = left[2 * index], l1 = left[2 * index + 1];
        const F r0 = right[2 * index], r1 = right[2 * index + 1];
        p0 += l0 * r0;
        p1 += l1 * r1;
        p2 += (l0 + F(2) * (l1 - l0)) * (r0 + F(2) * (r1 - r0));
    }
    F challenge;
    if (!prover.append_round({p0, p1, p2}, &challenge, reason)) throw std::runtime_error(reason != nullptr ? *reason : "masked Lasso product round failed");
    next_left = fold_product(left, right, challenge, next_right);
    if (challenge_out != nullptr) *challenge_out = challenge;
}

F mle_eval(const std::vector<F>& values, const std::vector<F>& point) {
    if (values.empty() || values.size() != (std::size_t(1) << point.size())) throw std::invalid_argument("native masked Lasso MLE dimensions are invalid");
    std::vector<F> current = values;
    for (const F& challenge : point) current = fold_linear(current, challenge);
    return current.front();
}

std::vector<F> row_mle(const std::vector<F>& matrix, std::size_t rows, std::size_t cols, const std::vector<F>& row_point) {
    if (matrix.size() != rows * cols || rows != (std::size_t(1) << row_point.size()))
        throw std::invalid_argument("native masked Lasso row MLE dimensions are invalid");
    std::vector<F> result(cols, F_ZERO);
    for (std::size_t col = 0; col < cols; ++col) {
        std::vector<F> column(rows);
        for (std::size_t row = 0; row < rows; ++row) column[row] = matrix[row + col * rows];
        result[col] = mle_eval(column, row_point);
    }
    return result;
}

std::vector<F> selected_table_row_mle(const NativeLassoFamilyProver& family, const std::vector<F>& query_point) {
    if (query_point.size() != log2_power(family.padded_queries))
        throw std::invalid_argument("native Lasso selected-row point dimension is invalid");
    std::vector<F> result(family.table.size(), F_ZERO);
    zkgpt::native_for_each_equality_weight(query_point, static_cast<std::size_t>(family.padded_queries), [&](std::size_t row, const F& weight) {
        const std::uint64_t selected = family.selected_table_row_at ? family.selected_table_row_at(row) : family.selected_table_rows[row];
        result[static_cast<std::size_t>(selected)] += weight;
    });
    return result;
}

std::vector<F> lift_segment_point(const std::vector<F>& row_point, std::size_t segment_offset, std::size_t rows, std::size_t domain) {
    if (!power_of_two(rows) || !power_of_two(domain) || rows > domain || segment_offset > domain - rows || segment_offset % rows != 0 ||
        row_point.size() != log2_power(rows))
        throw std::invalid_argument("native Lasso segment point is not canonical");
    const std::size_t block = segment_offset / rows;
    const std::size_t block_count = domain / rows;
    if (block >= block_count) throw std::invalid_argument("native Lasso segment block is invalid");
    std::vector<F> result = row_point;
    result.reserve(log2_power(domain));
    for (std::size_t bit = 0; bit < log2_power(block_count); ++bit) result.push_back(((block >> bit) & 1U) != 0 ? F_ONE : F_ZERO);
    return result;
}

bool split_pipe_fields(const std::string& payload, std::size_t expected, std::vector<std::string>& parts) {
    parts.clear();
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = payload.find('|', begin);
        parts.push_back(payload.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parts.size() == expected;
}

bool parse_masked_open_claim(const NativeProofSession& session, const std::string& label, G1& mask_commitment, F& masked_evaluation, std::string* reason,
                             const zkgpt::native_pedersen::PedersenBasis* scalar_basis = nullptr,
                             zkgpt::native_pedersen::ScalarCommitment* raw_commitment = nullptr,
                             zkgpt::native_pedersen::ScalarCommitment* masked_commitment = nullptr) {
    const std::string claim_label = label + "/claim";
    const std::vector<std::size_t>* positions = session.message_positions(claim_label);
    if (positions == nullptr || positions->empty()) {
        set_reason(reason, "malformed masked Hyrax claim: " + label);
        return false;
    }
    if (positions->size() != 1) {
        set_reason(reason, "duplicate masked Hyrax claim: " + label);
        return false;
    }
    const std::size_t position = positions->front();
    if (position >= session.messages().size()) {
        set_reason(reason, "masked Hyrax claim index is out of range: " + label);
        return false;
    }
    const NativeProofMessage* found = &session.messages()[position];
    std::vector<std::string> parts;
    const std::size_t expected = scalar_basis == nullptr ? 2 : 5;
    if (found == nullptr || found->kind != NativeProofMessageKind::Data || !split_pipe_fields(found->payload, expected, parts) ||
        !decode_point(parts[0], mask_commitment) || !decode_field(parts[1], masked_evaluation)) {
        set_reason(reason, "malformed masked Hyrax claim: " + label);
        return false;
    }
    if (scalar_basis != nullptr) {
        if (parts[2] != scalar_basis->domain || raw_commitment == nullptr || masked_commitment == nullptr || !decode_point(parts[3], raw_commitment->point) ||
            !decode_point(parts[4], masked_commitment->point)) {
            set_reason(reason, "malformed masked Hyrax scalar claim: " + label);
            return false;
        }
        raw_commitment->domain = scalar_basis->domain;
        masked_commitment->domain = scalar_basis->domain;
        if (!raw_commitment->valid(*scalar_basis, reason) || !masked_commitment->valid(*scalar_basis, reason)) return false;
    }
    return true;
}

bool verify_masked_opening(const NativeProofSession& session, const G1& commitment, const std::string& basis_id, const std::vector<G1>& generators,
                           const std::vector<F>& point, const std::string& label, const F& expected_masked_evaluation, std::string* reason,
                           bool enforce_statement_basis = false, bool allow_trailing = true,
                           const zkgpt::native_pedersen::PedersenBasis* scalar_basis = nullptr,
                           const zkgpt::native_pedersen::ScalarCommitment* expected_raw_commitment = nullptr,
                           const zkgpt::native_pedersen::ScalarCommitment* expected_masked_commitment = nullptr) {
    G1 mask_commitment;
    F masked_evaluation;
    zkgpt::native_pedersen::ScalarCommitment encoded_raw;
    zkgpt::native_pedersen::ScalarCommitment encoded_masked;
    if (!parse_masked_open_claim(session, label, mask_commitment, masked_evaluation, reason, scalar_basis, scalar_basis == nullptr ? nullptr : &encoded_raw,
                                 scalar_basis == nullptr ? nullptr : &encoded_masked))
        return false;
    if (masked_evaluation != expected_masked_evaluation) {
        set_reason(reason, "masked Hyrax claim is not bound to Sumcheck terminal");
        return false;
    }
    if (scalar_basis != nullptr) {
        if (expected_raw_commitment != nullptr && encoded_raw.point != expected_raw_commitment->point) {
            set_reason(reason, "masked Hyrax raw claim is not externally bound");
            return false;
        }
        if (expected_masked_commitment != nullptr && encoded_masked.point != expected_masked_commitment->point) {
            set_reason(reason, "masked Hyrax masked claim is not externally bound");
            return false;
        }
        if (expected_raw_commitment == nullptr) expected_raw_commitment = &encoded_raw;
        if (expected_masked_commitment == nullptr) expected_masked_commitment = &encoded_masked;
    }
    return hyrax::verify_masked_mle_opening_with_session(commitment, mask_commitment, masked_evaluation, basis_id, generators, point, session, label, reason,
                                                         enforce_statement_basis, allow_trailing, "", scalar_basis, expected_raw_commitment,
                                                         expected_masked_commitment);
}

G1 fold_public_row_commitments(const NativeLassoFamilyPublic& family, const std::vector<G1>& row_commitments, const std::vector<F>& point,
                               bool complement) {
    const std::uint64_t rows = family.matrix_layout.rows;
    const std::uint64_t columns = family.matrix_layout.columns;
    const std::size_t column_bits = family.matrix_layout.column_bits;
    if (row_commitments.size() != rows || point.size() != static_cast<std::size_t>(family.matrix_layout.row_bits + family.matrix_layout.column_bits))
        throw std::invalid_argument("native Lasso public row fold dimensions are invalid");
    const std::vector<F> row_point(point.begin() + column_bits, point.end());
    const std::vector<F> weights = mle_basis(row_point);
    if (weights.size() != rows) throw std::invalid_argument("native Lasso public row fold challenge count is invalid");
    G1 commitment = row_commitments.front() * weights.front();
    F weight_sum = F_ZERO;
    for (std::uint64_t row = 0; row < rows; ++row) {
        weight_sum += weights[static_cast<std::size_t>(row)];
        if (row != 0) commitment += row_commitments[static_cast<std::size_t>(row)] * weights[static_cast<std::size_t>(row)];
    }
    if (complement) {
        G1 generator_sum = family.matrix_layout.column_generators.front();
        for (std::size_t index = 1; index < family.matrix_layout.column_generators.size(); ++index)
            generator_sum += family.matrix_layout.column_generators[index];
        commitment -= generator_sum * weight_sum;
    }
    return commitment;
}

hyrax::HyraxMleBasisView matrix_view(const NativeLassoFamilyPublic& family, const std::string& basis_id, const std::vector<F>& point,
                                    bool complement) {
    const std::size_t column_bits = family.matrix_layout.column_bits;
    if (point.size() != static_cast<std::size_t>(family.matrix_layout.row_bits + column_bits))
        throw std::invalid_argument("native Lasso matrix opening point dimension is invalid");
    hyrax::HyraxMleBasisView view;
    view.full_domain = family.matrix_layout.columns;
    view.logical_length = view.full_domain;
    view.active_domain = view.full_domain;
    view.role = complement ? "native-lasso-M_f-minus-one-rowfold-v2" : "native-lasso-M_f-rowfold-v2";
    view.basis_id = basis_id;
    view.point.assign(point.begin(), point.begin() + column_bits);
    view.generators = matrix_generators(family);
    return view;
}

bool verify_masked_matrix_opening(const NativeProofSession& session, const NativeLassoFamilyPublic& family, const std::vector<G1>& row_commitments,
                                  const std::string& basis_id, const std::vector<F>& point, const std::string& label,
                                  const F& expected_masked_evaluation, std::string* reason, bool complement = false,
                                  const zkgpt::native_pedersen::PedersenBasis* scalar_basis = nullptr,
                                  const zkgpt::native_pedersen::ScalarCommitment* expected_raw_commitment = nullptr,
                                  const zkgpt::native_pedersen::ScalarCommitment* expected_masked_commitment = nullptr) {
    G1 mask_commitment;
    F masked_evaluation;
    zkgpt::native_pedersen::ScalarCommitment encoded_raw;
    zkgpt::native_pedersen::ScalarCommitment encoded_masked;
    if (!parse_masked_open_claim(session, label, mask_commitment, masked_evaluation, reason, scalar_basis,
                                 scalar_basis == nullptr ? nullptr : &encoded_raw, scalar_basis == nullptr ? nullptr : &encoded_masked))
        return false;
    if (masked_evaluation != expected_masked_evaluation) {
        set_reason(reason, "masked Hyrax row-fold claim is not bound to Sumcheck terminal");
        return false;
    }
    if (scalar_basis != nullptr) {
        if (expected_raw_commitment != nullptr && encoded_raw.point != expected_raw_commitment->point) {
            set_reason(reason, "masked Hyrax row-fold raw claim is not externally bound");
            return false;
        }
        if (expected_masked_commitment != nullptr && encoded_masked.point != expected_masked_commitment->point) {
            set_reason(reason, "masked Hyrax row-fold masked claim is not externally bound");
            return false;
        }
        if (expected_raw_commitment == nullptr) expected_raw_commitment = &encoded_raw;
        if (expected_masked_commitment == nullptr) expected_masked_commitment = &encoded_masked;
    }
    const G1 row_folded_commitment = fold_public_row_commitments(family, row_commitments, point, complement);
    const hyrax::HyraxMleBasisView view = matrix_view(family, basis_id, point, complement);
    return hyrax::verify_masked_mle_opening_with_session(row_folded_commitment, mask_commitment, masked_evaluation, view, session, label, reason, false,
                                                         true, "", scalar_basis, expected_raw_commitment, expected_masked_commitment);
}

bool verify_masked_provider_opening(const NativeProofSession& session, const G1& commitment, const hyrax::HyraxMleBasisView& view,
                                    const std::string& label, const F& expected_masked_evaluation, std::string* reason,
                                    bool enforce_statement_basis = false, bool allow_trailing = true,
                                    const zkgpt::native_pedersen::PedersenBasis* scalar_basis = nullptr,
                                    const zkgpt::native_pedersen::ScalarCommitment* expected_raw_commitment = nullptr,
                                    const zkgpt::native_pedersen::ScalarCommitment* expected_masked_commitment = nullptr) {
    G1 mask_commitment;
    F masked_evaluation;
    zkgpt::native_pedersen::ScalarCommitment encoded_raw;
    zkgpt::native_pedersen::ScalarCommitment encoded_masked;
    if (!parse_masked_open_claim(session, label, mask_commitment, masked_evaluation, reason, scalar_basis,
                                 scalar_basis == nullptr ? nullptr : &encoded_raw, scalar_basis == nullptr ? nullptr : &encoded_masked))
        return false;
    if (masked_evaluation != expected_masked_evaluation) {
        set_reason(reason, "masked Hyrax provider claim is not bound to Sumcheck terminal");
        return false;
    }
    if (scalar_basis != nullptr) {
        if (expected_raw_commitment != nullptr && encoded_raw.point != expected_raw_commitment->point) {
            set_reason(reason, "masked Hyrax provider raw claim is not externally bound");
            return false;
        }
        if (expected_masked_commitment != nullptr && encoded_masked.point != expected_masked_commitment->point) {
            set_reason(reason, "masked Hyrax provider masked claim is not externally bound");
            return false;
        }
        if (expected_raw_commitment == nullptr) expected_raw_commitment = &encoded_raw;
        if (expected_masked_commitment == nullptr) expected_masked_commitment = &encoded_masked;
    }
    return hyrax::verify_masked_mle_opening_with_session(commitment, mask_commitment, masked_evaluation, view, session, label, reason,
                                                         enforce_statement_basis, allow_trailing, "", scalar_basis, expected_raw_commitment,
                                                         expected_masked_commitment);
}

bool verify_masked_tensor_opening(const NativeProofSession& session, const hyrax::HyraxTensorMleView& view,
                                  const hyrax::HyraxTensorMleCommitment& commitment, const std::vector<F>& full_point,
                                  const std::string& label, const F& expected_masked_evaluation, std::string* reason,
                                  bool allow_trailing = true,
                                  const zkgpt::native_pedersen::PedersenBasis* scalar_basis = nullptr,
                                  const zkgpt::native_pedersen::ScalarCommitment* expected_raw_commitment = nullptr,
                                  const zkgpt::native_pedersen::ScalarCommitment* expected_masked_commitment = nullptr) {
    G1 mask_commitment;
    F masked_evaluation;
    zkgpt::native_pedersen::ScalarCommitment encoded_raw;
    zkgpt::native_pedersen::ScalarCommitment encoded_masked;
    const std::string column_label = label + "/column";
    if (!parse_masked_open_claim(session, column_label, mask_commitment, masked_evaluation, reason, scalar_basis,
                                 scalar_basis == nullptr ? nullptr : &encoded_raw, scalar_basis == nullptr ? nullptr : &encoded_masked))
        return false;
    if (masked_evaluation != expected_masked_evaluation) {
        set_reason(reason, "masked Hyrax tensor claim is not bound to Sumcheck terminal");
        return false;
    }
    if (scalar_basis != nullptr) {
        if (expected_raw_commitment != nullptr && encoded_raw.point != expected_raw_commitment->point) {
            set_reason(reason, "masked Hyrax tensor raw claim is not externally bound");
            return false;
        }
        if (expected_masked_commitment != nullptr && encoded_masked.point != expected_masked_commitment->point) {
            set_reason(reason, "masked Hyrax tensor masked claim is not externally bound");
            return false;
        }
        if (expected_raw_commitment == nullptr) expected_raw_commitment = &encoded_raw;
        if (expected_masked_commitment == nullptr) expected_masked_commitment = &encoded_masked;
    }
    return hyrax::verify_hyrax_tensor_mle_opening_with_session(view, commitment, full_point, mask_commitment, masked_evaluation, session, label,
                                                              reason, allow_trailing, "", scalar_basis, expected_raw_commitment,
                                                              expected_masked_commitment);
}

constexpr std::size_t kNativeLassoRowCommitmentChunkEntries = 1ULL << 12;

std::size_t row_commitment_chunk_count(std::uint64_t rows) {
    if (rows == 0 || rows > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("native Lasso row commitment domain is invalid");
    const std::size_t count = static_cast<std::size_t>(rows);
    return count / kNativeLassoRowCommitmentChunkEntries + (count % kNativeLassoRowCommitmentChunkEntries != 0 ? 1 : 0);
}

std::size_t row_commitment_last_chunk_count(std::uint64_t rows) {
    if (rows == 0 || rows > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("native Lasso row commitment domain is invalid");
    const std::size_t count = static_cast<std::size_t>(rows);
    const std::size_t last = count % kNativeLassoRowCommitmentChunkEntries;
    return last == 0 ? kNativeLassoRowCommitmentChunkEntries : last;
}

std::string masked_family_payload(const NativeLassoFamilyPublic& family, const std::vector<G1>& row_commitments, const std::string& basis_id) {
    if (row_commitments.size() != family.matrix_layout.rows) throw std::invalid_argument("native Lasso row commitment count mismatch");
    std::vector<std::string> fields = {"zkgpt-native-lasso-rowtensor-v2-chunked",
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
                                       basis_id,
                                       hyrax::derive_pedersen_blinding_generator(basis_id).serializeToHexStr(),
                                       std::to_string(family.matrix_layout.version),
                                       std::to_string(family.matrix_layout.vector_size),
                                       std::to_string(family.matrix_layout.rows),
                                       std::to_string(family.matrix_layout.columns),
                                       std::to_string(family.matrix_layout.row_bits),
                                       std::to_string(family.matrix_layout.column_bits),
                                       family.matrix_layout.flatten_order,
                                       family.matrix_layout.generator_scheme,
                                       family.matrix_layout.trusted_generator_id,
                                       native_lasso_matrix_layout_digest(family.matrix_layout),
                                       native_lasso_table_digest(family),
                                       std::to_string(kNativeLassoRowCommitmentChunkEntries),
                                       std::to_string(row_commitment_chunk_count(family.matrix_layout.rows)),
                                       std::to_string(row_commitment_last_chunk_count(family.matrix_layout.rows))};
    return encode_tokens(fields);
}

std::string masked_family_row_chunk_payload(const std::vector<G1>& row_commitments, std::size_t begin, std::size_t count) {
    if (begin > row_commitments.size() || count > row_commitments.size() - begin || count == 0 || count > kNativeLassoRowCommitmentChunkEntries)
        throw std::invalid_argument("native Lasso row commitment chunk is invalid");
    std::vector<std::string> fields;
    fields.reserve(count);
    for (std::size_t offset = 0; offset < count; ++offset) fields.push_back(row_commitments[begin + offset].serializeToHexStr());
    return encode_tokens(fields);
}

bool parse_masked_family_commitment(const NativeProofMessage& message, const NativeLassoFamilyPublic& expected, const std::string& expected_basis_id,
                                    std::vector<G1>& row_commitments, std::string* reason) {
    std::vector<std::string> fields;
    if (message.kind != NativeProofMessageKind::Data || !decode_tokens(message.payload, 27, fields)) {
        set_reason(reason, "malformed native Lasso row-tensor commitment");
        return false;
    }
    if (fields[0] != "zkgpt-native-lasso-rowtensor-v2-chunked" || fields[1] != expected.family_id || fields[2] != expected.table_id || fields[3] != expected.resource_id ||
        fields[4] != expected.registration_id || fields[10] != expected.padding_value.getStr(16) || fields[11] != expected_basis_id ||
        fields[12] != hyrax::derive_pedersen_blinding_generator(expected_basis_id).serializeToHexStr() ||
        fields[19] != expected.matrix_layout.flatten_order || fields[20] != expected.matrix_layout.generator_scheme ||
        fields[21] != expected.matrix_layout.trusted_generator_id || fields[22] != native_lasso_matrix_layout_digest(expected.matrix_layout) ||
        fields[23] != native_lasso_table_digest(expected) || fields[24] != std::to_string(kNativeLassoRowCommitmentChunkEntries) ||
        fields[25] != std::to_string(row_commitment_chunk_count(expected.matrix_layout.rows)) ||
        fields[26] != std::to_string(row_commitment_last_chunk_count(expected.matrix_layout.rows)) || !valid_basis_id(fields[11])) {
        set_reason(reason, "native Lasso row-tensor authenticated identity mismatch");
        return false;
    }
    std::size_t padded_queries = 0, table_domain = 0, offset = 0, logical = 0, padding = 0, version = 0, vector_size = 0;
    std::size_t rows = 0, columns = 0, row_bits = 0, column_bits = 0;
    if (!parse_size(fields[5], padded_queries) || !parse_size(fields[6], table_domain) || !parse_size(fields[7], offset) || !parse_size(fields[8], logical) ||
        !parse_size(fields[9], padding) || !parse_size(fields[13], version) || !parse_size(fields[14], vector_size) || !parse_size(fields[15], rows) ||
        !parse_size(fields[16], columns) || !parse_size(fields[17], row_bits) || !parse_size(fields[18], column_bits) ||
        padded_queries != expected.padded_queries || table_domain != expected.table.size() || offset != expected.segment_offset || logical != expected.logical_queries ||
        padding != expected.padding_index || version != expected.matrix_layout.version || vector_size != expected.matrix_layout.vector_size ||
        rows != expected.matrix_layout.rows || columns != expected.matrix_layout.columns || row_bits != expected.matrix_layout.row_bits ||
        column_bits != expected.matrix_layout.column_bits) {
        set_reason(reason, "native Lasso row-tensor shape mismatch");
        return false;
    }
    row_commitments.clear();
    row_commitments.reserve(expected.matrix_layout.rows);
    return true;
}

bool parse_masked_family_row_chunk(const NativeProofMessage& message, std::size_t expected_count, std::vector<G1>& row_commitments,
                                   std::string* reason) {
    std::vector<std::string> fields;
    if (message.kind != NativeProofMessageKind::Data || expected_count == 0 || expected_count > kNativeLassoRowCommitmentChunkEntries ||
        !decode_tokens(message.payload, expected_count, fields)) {
        set_reason(reason, "malformed native Lasso row commitment chunk");
        return false;
    }
    for (const std::string& encoded : fields) {
        G1 commitment;
        if (!decode_point(encoded, commitment)) {
            set_reason(reason, "native Lasso row commitment is malformed");
            return false;
        }
        row_commitments.push_back(commitment);
    }
    return true;
}

bool consume_field_message(const NativeProofSession& session, std::size_t& position, const std::string& label, F& value, std::string* reason) {
    const NativeProofMessage* message = consume(session, position, NativeProofMessageKind::Field, label, reason);
    if (message == nullptr || !decode_field(message->payload, value)) {
        set_reason(reason, "malformed masked native Lasso field: " + label);
        return false;
    }
    return true;
}

bool opening_commitment_at(const NativeProofSession& session, const zkgpt::native_pedersen::PedersenBasis& basis, const std::string& label,
                           zkgpt::native_pedersen::ScalarCommitment& commitment, std::string* reason) {
    return native_masked_opening_commitment_at(session, basis, 0, label, commitment, reason);
}

bool build_sumcheck_verify_spec(const NativeProofSession& session, const std::string& label, NativeMaskedTerminalKind kind, const F& initial_value,
                                const zkgpt::native_pedersen::PedersenBasis& basis, NativeMaskedTerminalSpec& spec, std::string* reason,
                                bool public_initial_claim) {
    spec = NativeMaskedTerminalSpec{};
    spec.kind = kind;
    spec.has_external_binding = true;
    spec.public_initial_claim = public_initial_claim;
    if (!opening_commitment_at(session, basis, label + "/initial/opening", spec.initial_commitment, reason)) return false;
    if (!public_initial_claim) {
        if (!opening_commitment_at(session, basis, label + "/initial/masked", spec.masked_initial_commitment, reason)) return false;
        spec.has_masked_initial_binding = true;
    }
    const zkgpt::native_pedersen::ScalarCommitment expected_initial{basis.domain, zkgpt::native_pedersen::commit(basis, initial_value, F_ZERO).point};
    if (kind == NativeMaskedTerminalKind::Scalar || kind == NativeMaskedTerminalKind::Product2 || kind == NativeMaskedTerminalKind::Product3Public) {
        if (spec.initial_commitment.point != expected_initial.point) {
            set_reason(reason, "Lasso Sumcheck initial public binding mismatch");
            return false;
        }
    }
    const std::size_t arity = kind == NativeMaskedTerminalKind::Product2 ? 2 : (kind == NativeMaskedTerminalKind::Product3Public ? 2 : 1);
    spec.terminal_commitments.resize(arity);
    for (std::size_t index = 0; index < arity; ++index)
        if (!opening_commitment_at(session, basis, label + "/terminal/value/" + std::to_string(index), spec.terminal_commitments[index], reason)) return false;
    return true;
}

std::size_t relation_rounds(std::size_t size) {
    if (size == 0 || (size & (size - 1)) != 0) throw std::invalid_argument("masked Lasso relation size is not a power of two");
    return log2_power(size);
}

bool advance_masked_scope(const NativeProofSession& session, const std::string& label, std::size_t& position, std::string* reason) {
    const std::string prefix = label + "/";
    const std::string challenge_prefix = "challenge/" + prefix;
    std::size_t first = session.messages().size();
    std::size_t last = position;
    for (std::size_t index = 0; index < session.messages().size(); ++index) {
        const std::string& message_label = session.messages()[index].label;
        if (message_label.compare(0, prefix.size(), prefix) != 0 && message_label.compare(0, challenge_prefix.size(), challenge_prefix) != 0) continue;
        if (first == session.messages().size()) first = index;
        last = index;
    }
    if (first == session.messages().size() || first != position) {
        set_reason(reason, "masked Lasso scope is not adjacent at " + label);
        return false;
    }
    for (std::size_t index = first; index <= last; ++index) {
        const std::string& message_label = session.messages()[index].label;
        if (message_label.compare(0, prefix.size(), prefix) != 0 && message_label.compare(0, challenge_prefix.size(), challenge_prefix) != 0) {
            set_reason(reason, "masked Lasso scope has an inserted message at " + label);
            return false;
        }
    }
    position = last + 1;
    return true;
}

}

std::size_t native_lasso_sparse_entry_bytes_for_smoke() { return sizeof(SparseValue); }

NativeLassoCapacityProfile native_lasso_capacity_profile(std::uint64_t padded_queries, std::uint64_t table_domain) {
    const NativeLassoFamilyPublic::MatrixDescriptor layout = native_lasso_balanced_matrix_layout(padded_queries, table_domain);
    const auto bytes = [](std::uint64_t count, std::size_t element_size) {
        if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
            if (element_size > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) throw std::length_error("native Lasso capacity profile element size overflows u64");
        }
        if (element_size != 0 && count > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(element_size))
            throw std::length_error("native Lasso capacity profile byte count overflows u64");
        const std::uint64_t result = count * static_cast<std::uint64_t>(element_size);
        if (result > std::numeric_limits<std::size_t>::max()) throw std::length_error("native Lasso capacity profile exceeds native storage width");
        return result;
    };
    const auto add = [](std::uint64_t left, std::uint64_t right) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) throw std::length_error("native Lasso capacity profile sum overflows u64");
        const std::uint64_t result = left + right;
        if (result > std::numeric_limits<std::size_t>::max()) throw std::length_error("native Lasso capacity profile sum exceeds native storage width");
        return result;
    };
    NativeLassoCapacityProfile profile;
    profile.padded_queries = padded_queries;
    profile.table_domain = table_domain;
    profile.full_domain = layout.vector_size;
    profile.rows = layout.rows;
    profile.columns = layout.columns;
    profile.t2_booleanity = table_domain == 2;
    profile.private_witness_entries = padded_queries;
    profile.private_witness_element_bytes = profile.t2_booleanity ? sizeof(F) : sizeof(SparseValue);
    profile.private_witness_bytes = bytes(profile.private_witness_entries, static_cast<std::size_t>(profile.private_witness_element_bytes));
    profile.row_blinding_bytes = bytes(profile.rows, sizeof(F));
    profile.row_commitment_bytes = bytes(profile.rows, sizeof(G1));
    profile.column_generator_bytes = bytes(profile.columns, sizeof(G1));
    profile.row_fold_weights_bytes = bytes(profile.rows, sizeof(F));
    profile.row_fold_values_bytes = bytes(profile.columns, sizeof(F));
    profile.row_ledger_retained_bytes = add(profile.row_blinding_bytes, profile.column_generator_bytes);
    profile.commit_peak_bytes = add(profile.row_ledger_retained_bytes, profile.row_commitment_bytes);
    profile.booleanity_peak_bytes = add(profile.row_ledger_retained_bytes, profile.private_witness_bytes);
    profile.opening_peak_bytes = add(profile.row_ledger_retained_bytes, add(profile.row_fold_weights_bytes, profile.row_fold_values_bytes));
    profile.family_peak_bytes = std::max(profile.commit_peak_bytes, std::max(profile.booleanity_peak_bytes, profile.opening_peak_bytes));
    return profile;
}

bool native_lasso_sparse_booleanity_dense_reference_smoke() {
    try {
        const std::vector<std::uint64_t> selected = {0, 1, 2, 3};
        const std::vector<F> beta = {F_ZERO, F_ONE, F(3), F_ONE};
        NativeLassoFamilyProver family;
        family.table = {F(5), F(7), F(11), F(13)};
        family.padded_queries = selected.size();
        family.selected_table_row_at = [selected](std::uint64_t row) { return selected.at(static_cast<std::size_t>(row)); };
        const NativeLassoSparseView sparse = native_lasso_sparse_view(family);
        SparseMatrixValues sparse_values = sparse_matrix_values(family, beta);
        std::vector<F> dense(static_cast<std::size_t>(sparse.full_domain), F_ZERO);
        for (std::size_t query = 0; query < selected.size(); ++query) dense[static_cast<std::size_t>(selected[query] * selected.size() + query)] = F_ONE;
        bool saw_zero_factor = false, saw_different_column_pair = false;
        for (const SparseValue& value : sparse_values) saw_zero_factor = saw_zero_factor || value.remaining_zero_factors != 0;
        F initial_claim = F_ZERO;
        for (std::uint64_t index = 0; index < sparse.full_domain; ++index) {
            F weight = F_ONE;
            for (std::size_t bit = 0; bit < beta.size(); ++bit) weight *= ((index >> bit) & 1U) != 0 ? beta[bit] : F_ONE - beta[bit];
            initial_claim += weight * dense[static_cast<std::size_t>(index)] * (dense[static_cast<std::size_t>(index)] - F_ONE);
        }
        if (!saw_zero_factor || !initial_claim.isZero() || sparse_values.size() != selected.size()) return false;
        const std::string zero_digest(64, '0');
        const NativeProofStatement statement{zero_digest, "mcl-bn254-fr-v1", zero_digest, zero_digest, zero_digest, zero_digest, zero_digest, zero_digest, zero_digest, 4, zero_digest};
        NativeProofSession session("native-lasso-sparse-booleanity-dense-v1", statement);
        session.absorb_fields("lasso/smoke/booleanity/weights", beta);
        const zkgpt::native_pedersen::PedersenBasis scalar_basis = lasso_scalar_basis(session);
        NativeMaskedTerminalSpec terminal_spec;
        terminal_spec.kind = NativeMaskedTerminalKind::Product3Public;
        terminal_spec.public_provider = equality_weight_provider(beta);
        terminal_spec.has_external_binding = true;
        terminal_spec.public_initial_claim = true;
        terminal_spec.initial_opening = public_scalar_opening(scalar_basis, initial_claim);
        terminal_spec.initial_commitment = terminal_spec.initial_opening.commitment;
        NativeMaskedSumcheckProver sumcheck(session, "lasso/smoke/booleanity/sum", 3, initial_claim, terminal_spec);
        const auto dense_claim = [&beta](const std::vector<F>& values, std::size_t first_bit, const F& prefix_weight) {
            F claim = F_ZERO;
            for (std::size_t index = 0; index < values.size(); ++index) {
                F weight = prefix_weight;
                for (std::size_t bit = first_bit; bit < beta.size(); ++bit)
                    weight *= ((index >> (bit - first_bit)) & 1U) != 0 ? beta[bit] : F_ONE - beta[bit];
                claim += weight * values[index] * (values[index] - F_ONE);
            }
            return claim;
        };
        const auto interpolate = [](const std::vector<F>& evaluations, const F& point) {
            F result = F_ZERO;
            for (std::size_t i = 0; i < evaluations.size(); ++i) {
                F basis = F_ONE;
                for (std::size_t j = 0; j < evaluations.size(); ++j)
                    if (i != j) basis *= (point - F(static_cast<int>(j))) / F(static_cast<int>(i) - static_cast<int>(j));
                result += evaluations[i] * basis;
            }
            return result;
        };
        F prefix_weight = F_ONE, running_claim = initial_claim;
        for (std::size_t round = 0; round < beta.size(); ++round) {
            if (running_claim != dense_claim(dense, round, prefix_weight)) return false;
            const F prefix_before = prefix_weight;
            std::vector<F> actual_polynomial;
            F challenge;
            const std::size_t previous_support = sparse_values.size();
            if (round == 2) for (std::size_t index = 1; index < sparse_values.size(); ++index) saw_different_column_pair = saw_different_column_pair || sparse_values[index - 1].index / 2 == sparse_values[index].index / 2;
            append_weighted_boolean_round(sumcheck, sparse_values, beta, round, prefix_weight, nullptr, &challenge, &actual_polynomial);
            std::vector<F> reference_polynomial(4, F_ZERO);
            for (std::size_t pair = 0; pair < dense.size() / 2; ++pair) {
                F suffix_weight = F_ONE;
                for (std::size_t bit = round + 1; bit < beta.size(); ++bit)
                    suffix_weight *= ((pair >> (bit - round - 1)) & 1U) != 0 ? beta[bit] : F_ONE - beta[bit];
                for (std::size_t evaluation = 0; evaluation < reference_polynomial.size(); ++evaluation) {
                    const F x(static_cast<int>(evaluation));
                    const F matrix_value = dense[2 * pair] + x * (dense[2 * pair + 1] - dense[2 * pair]);
                    const F current_weight = F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * x;
                    reference_polynomial[evaluation] += prefix_before * suffix_weight * current_weight * matrix_value * (matrix_value - F_ONE);
                }
            }
            if (actual_polynomial != reference_polynomial || actual_polynomial[0] + actual_polynomial[1] != running_claim) return false;
            running_claim = interpolate(reference_polynomial, challenge);
            for (std::size_t pair = 0; pair < dense.size() / 2; ++pair)
                dense[pair] += challenge * (dense[2 * pair + 1] - dense[2 * pair]);
            dense.resize(dense.size() / 2);
            prefix_weight *= F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * challenge;
            if (running_claim != dense_claim(dense, round + 1, prefix_weight)) return false;
            if (round == 2 && sparse_values.size() >= previous_support) return false;
        }
        if (!saw_different_column_pair || dense.size() != 1 || sparse_values.size() > 1 || (!sparse_values.empty() && sparse_values.front().index != 0)) return false;
        const F matrix_value = sparse_values.empty() ? F_ZERO : sparse_values.front().value;
        if (dense.front() != matrix_value || running_claim != prefix_weight * matrix_value * (matrix_value - F_ONE)) return false;
        terminal_spec.unmasked_terminal_openings = {scalar_evaluation_opening(scalar_basis, matrix_value), scalar_evaluation_opening(scalar_basis, matrix_value - F_ONE)};
        if (!sumcheck.set_terminal_spec(terminal_spec, nullptr)) return false;
        NativeMaskedSumcheckResult proved;
        if (!sumcheck.finish({matrix_value, matrix_value - F_ONE}, &proved, nullptr) || proved.challenges.size() != beta.size() || proved.unmasked_claim_opening.value != running_claim) return false;
        NativeMaskedTerminalSpec verify_spec = terminal_spec;
        verify_spec.initial_opening = {};
        verify_spec.unmasked_terminal_openings.clear();
        verify_spec.terminal_commitments = proved.terminal_commitments;
        NativeMaskedSumcheckResult verified;
        if (!native_masked_sumcheck_verify(session, "lasso/smoke/booleanity/sum", 3, beta.size(), initial_claim, verify_spec, &verified, nullptr) || verified.challenges != proved.challenges || verified.terminal_values != proved.terminal_values || !verified.external_binding_verified || !session.replay(nullptr)) return false;
        return true;
    } catch (...) {
        return false;
    }
}

bool native_lasso_t2_booleanity_dense_reference_smoke() {
    try {
        const std::vector<std::uint64_t> selected = {1, 0, 1, 0};
        const std::vector<F> beta = {F(2), F(3), F(5)};
        const F challenges[3] = {F(7), F(11), F(13)};
        zkgpt::NativeFieldVector column_one(selected.size());
        std::vector<F> dense(8, F_ZERO);
        for (std::size_t query = 0; query < selected.size(); ++query) {
            column_one[query] = selected[query] == 1 ? F_ONE : F_ZERO;
            dense[query] = selected[query] == 0 ? F_ONE : F_ZERO;
            dense[query + selected.size()] = selected[query] == 1 ? F_ONE : F_ZERO;
        }
        F prefix_weight = F_ONE;
        for (std::size_t round = 0; round < beta.size(); ++round) {
            const std::vector<F> actual = weighted_booleanity_t2_round_polynomial(column_one, beta, round, prefix_weight);
            std::vector<F> reference(4, F_ZERO);
            for (std::size_t pair = 0; pair < dense.size() / 2; ++pair) {
                F suffix_weight = F_ONE;
                for (std::size_t bit = round + 1; bit < beta.size(); ++bit) {
                    const bool set = ((pair >> (bit - round - 1)) & 1U) != 0;
                    suffix_weight *= set ? beta[bit] : F_ONE - beta[bit];
                }
                for (std::size_t evaluation = 0; evaluation < reference.size(); ++evaluation) {
                    const F x(static_cast<int>(evaluation));
                    const F matrix_value = dense[2 * pair] + x * (dense[2 * pair + 1] - dense[2 * pair]);
                    const F current_weight = F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * x;
                    reference[evaluation] += prefix_weight * suffix_weight * current_weight * matrix_value * (matrix_value - F_ONE);
                }
            }
            if (actual != reference) return false;
            if (round + 1 < beta.size()) {
                const std::size_t pair_count = column_one.size() / 2;
                for (std::size_t pair = 0; pair < pair_count; ++pair)
                    column_one[pair] += challenges[round] * (column_one[pair * 2 + 1] - column_one[pair * 2]);
                column_one.resize(pair_count);
            }
            for (std::size_t pair = 0; pair < dense.size() / 2; ++pair)
                dense[pair] += challenges[round] * (dense[pair * 2 + 1] - dense[pair * 2]);
            dense.resize(dense.size() / 2);
            prefix_weight *= F_ONE - beta[round] + (F(2) * beta[round] - F_ONE) * challenges[round];
        }
        const F terminal_matrix_value = F_ONE - column_one[0] + challenges[beta.size() - 1] * (F(2) * column_one[0] - F_ONE);
        return dense.size() == 1 && column_one.size() == 1 && dense[0] == terminal_matrix_value;
    } catch (...) {
        return false;
    }
}

bool native_lasso_prove_bundle(NativeProofSession& session, std::vector<NativeLassoFamilyProver>& families, NativeLassoQueryProver& query,
                               NativeLassoBundleResult* result, std::string* reason, const NativeLassoTensorTailProveHook& tensor_tail_stage) {
    SessionCheckpointGuard checkpoint_guard(session);
    try {
        if (families.empty() || query.tensor_view.full_domain == 0 || !tensor_tail_stage || !validate_query(query, reason)) {
            if (query.tensor_view.full_domain == 0) set_reason(reason, "formal native Lasso requires a typed balanced L_qry tensor");
            else if (!tensor_tail_stage) set_reason(reason, "formal native Lasso requires the shared Aux/L_qry tensor-tail certificate stage");
            return false;
        }
        std::vector<NativeLassoFamilyPublic> identities;
        identities.reserve(families.size());
        for (NativeLassoFamilyProver& family : families) {
            if (!validate_prover_family(family, reason)) return false;
            identities.push_back(public_identity(family));
        }
        if (!check_family_order(identities, reason)) return false;
        const std::uint64_t query_size = query_domain(query);
        const std::string query_digest = native_lasso_tensor_query_digest(query.tensor_view, query.tensor_commitment);
        if (!check_statement_identity_with_l_digest(session, identities, query_digest, query.statement_l_digest, reason)) return false;
        for (const NativeLassoFamilyPublic& family : identities) {
            if (family.segment_offset > query_size || family.padded_queries > query_size - family.segment_offset) {
                set_reason(reason, "native Lasso family exceeds L_qry segment");
                return false;
            }
        }
        std::vector<MaskedFamilyState> states;
        states.reserve(families.size());
        NativeLassoBundleResult local;
        const zkgpt::native_pedersen::PedersenBasis scalar_basis = lasso_scalar_basis(session);
        for (std::size_t index = 0; index < families.size(); ++index) {
            const NativeLassoFamilyProver& family = families[index];
            MaskedFamilyState state;
            state.basis_id = family_basis_id(session, family);
            commit_family_rows(family, state);
            const std::string commitment_label = family_prefix(family) + "/commitment";
            session.absorb(commitment_label, masked_family_payload(family, state.row_commitments, state.basis_id));
            const std::size_t chunk_count = row_commitment_chunk_count(family.matrix_layout.rows);
            const std::size_t last_chunk_count = row_commitment_last_chunk_count(family.matrix_layout.rows);
            for (std::size_t begin = 0, chunk = 0; chunk < chunk_count; ++chunk) {
                const std::size_t count = chunk + 1 == chunk_count ? last_chunk_count : kNativeLassoRowCommitmentChunkEntries;
                session.absorb(commitment_label + "/rows/" + std::to_string(chunk), masked_family_row_chunk_payload(state.row_commitments, begin, count));
                begin += count;
            }
            std::vector<G1>().swap(state.row_commitments);
            states.push_back(std::move(state));
        }

        if (!tensor_tail_stage(session, query, reason)) {
            if (reason != nullptr && reason->empty()) set_reason(reason, "native Lasso shared tensor-tail certificate stage failed");
            return false;
        }

        std::vector<std::vector<F>> booleanity_weights(families.size());
        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyProver& family = families[family_index];
            const std::size_t rounds = relation_rounds(static_cast<std::size_t>(family.matrix_layout.vector_size));
            std::vector<F>& beta = booleanity_weights[family_index];
            beta.reserve(rounds);
            for (std::size_t round = 0; round < rounds; ++round)
                beta.push_back(session.challenge(family_prefix(family) + "/booleanity/weight/" + std::to_string(round)));
        }

        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            NativeLassoFamilyProver& family = families[family_index];
            const std::string prefix = family_prefix(family);
            const std::size_t rows = static_cast<std::size_t>(family.padded_queries);
            const std::size_t cols = family.table.size();
            const std::size_t query_rounds = relation_rounds(rows);
            std::vector<F> row_point;
            row_point.reserve(query_rounds);
            for (std::size_t round = 0; round < query_rounds; ++round) row_point.push_back(session.challenge(prefix + "/a-query/r/" + std::to_string(round)));
            local.family_query_points.push_back(row_point);

            NativeLassoSparseView sparse = native_lasso_sparse_view(family);
            NativeMaskedTerminalSpec boolean_spec;
            boolean_spec.kind = NativeMaskedTerminalKind::Product3Public;
            boolean_spec.public_provider = equality_weight_provider(booleanity_weights[family_index]);
            boolean_spec.has_external_binding = true;
            boolean_spec.public_initial_claim = true;
            boolean_spec.initial_opening = public_scalar_opening(scalar_basis, F_ZERO);
            boolean_spec.initial_commitment = boolean_spec.initial_opening.commitment;
            session.absorb_field(prefix + "/b-boolean/claim", F_ZERO);
            NativeMaskedSumcheckProver boolean_prover(session, prefix + "/b-boolean/sum", 3, F_ZERO, boolean_spec);
            std::vector<F> boolean_point;
            boolean_point.reserve(booleanity_weights[family_index].size());
            F boolean_value = F_ZERO;
            if (family.table.size() == 2) {
                WeightedBooleanityFold folded = append_weighted_booleanity_t2(boolean_prover, family, booleanity_weights[family_index], reason);
                boolean_point = std::move(folded.point);
                boolean_value = folded.matrix_value;
            } else {
                SparseMatrixValues boolean_values = sparse_matrix_values(family, booleanity_weights[family_index]);
                F prefix_weight = F_ONE;
                for (std::size_t round = 0; round < booleanity_weights[family_index].size(); ++round) {
                    F challenge;
                    append_weighted_boolean_round(boolean_prover, boolean_values, booleanity_weights[family_index], round, prefix_weight, reason, &challenge);
                    boolean_point.push_back(challenge);
                }
                if (boolean_values.size() > 1 || (!boolean_values.empty() && boolean_values.front().index != 0))
                    throw std::logic_error("native Lasso weighted Booleanity terminal fold is not scalar");
                boolean_value = boolean_values.empty() ? F_ZERO : boolean_values.front().value;
            }
            const F complement_value = boolean_value - F_ONE;
            const zkgpt::native_pedersen::ScalarOpening boolean_raw_opening = scalar_evaluation_opening(scalar_basis, boolean_value);
            const zkgpt::native_pedersen::ScalarOpening complement_raw_opening = scalar_evaluation_opening(scalar_basis, complement_value);
            boolean_spec.unmasked_terminal_openings = {boolean_raw_opening, complement_raw_opening};
            if (!boolean_prover.set_terminal_spec(boolean_spec, reason)) return false;
            NativeMaskedSumcheckResult boolean_result;
            if (!boolean_prover.finish({boolean_value, complement_value}, &boolean_result, reason)) return false;
            hyrax::MaskedMleOpening boolean_opening;
            {
                MatrixRowFold boolean_fold = fold_family_rows(family, states[family_index], boolean_point);
                if (!hyrax::open_masked_mle_with_claim(boolean_fold.commitment, boolean_fold.view, boolean_fold.values, boolean_result.terminal_values[0], session,
                                                       prefix + "/c-boolean-open", boolean_opening, reason, false, "", &scalar_basis,
                                                       &boolean_result.terminal_openings[0], &boolean_raw_opening)) return false;
            }
            {
                MatrixRowFold complement_fold = fold_family_rows(family, states[family_index], boolean_point, true);
                if (!hyrax::open_masked_mle_with_claim(complement_fold.commitment, complement_fold.view, complement_fold.values,
                                                       boolean_result.terminal_values[1], session, prefix + "/c-boolean-complement-open", boolean_opening,
                                                       reason, false, "", &scalar_basis, &boolean_result.terminal_openings[1], &complement_raw_opening)) return false;
            }

            std::vector<F> onehot_left = selected_table_row_mle(family, row_point);
            session.absorb_field(prefix + "/d-onehot/claim", F_ONE);
            NativeMaskedTerminalSpec onehot_spec;
            onehot_spec.kind = NativeMaskedTerminalKind::Scalar;
            onehot_spec.has_external_binding = true;

            onehot_spec.public_initial_claim = true;
            onehot_spec.initial_opening = public_scalar_opening(scalar_basis, F_ONE);
            onehot_spec.initial_commitment = onehot_spec.initial_opening.commitment;
            NativeMaskedSumcheckProver onehot_prover(session, prefix + "/d-onehot/sum", 1, F_ONE, onehot_spec);
            std::vector<F> onehot_current = onehot_left;
            std::vector<F> onehot_challenges;
            onehot_challenges.reserve(relation_rounds(cols));
            for (std::size_t round = 0; round < relation_rounds(cols); ++round) {
                std::vector<F> next;
                F challenge;
                append_linear_round(onehot_prover, onehot_current, next, reason, &challenge);
                onehot_challenges.push_back(challenge);
                onehot_current.swap(next);
            }
            std::vector<F> onehot_point = row_point;
            onehot_point.insert(onehot_point.end(), onehot_challenges.begin(), onehot_challenges.end());
            const zkgpt::native_pedersen::ScalarOpening onehot_raw_opening = scalar_evaluation_opening(scalar_basis, sparse.evaluate_mle(onehot_point));
            onehot_spec.unmasked_terminal_openings = {onehot_raw_opening};
            if (!onehot_prover.set_terminal_spec(onehot_spec, reason)) return false;
            NativeMaskedSumcheckResult onehot_result;
            if (!onehot_prover.finish({onehot_current.front()}, &onehot_result, reason)) return false;
            hyrax::MaskedMleOpening onehot_opening;
            {
                MatrixRowFold onehot_fold = fold_family_rows(family, states[family_index], onehot_point);
                if (!hyrax::open_masked_mle_with_claim(onehot_fold.commitment, onehot_fold.view, onehot_fold.values, onehot_result.final_claim, session,
                                                       prefix + "/e-onehot-open", onehot_opening, reason, false, "", &scalar_basis,
                                                       &onehot_result.terminal_openings[0], &onehot_raw_opening)) return false;
            }

            const std::vector<F>& left = onehot_left;
            F ell = F_ZERO;
            for (std::size_t col = 0; col < cols; ++col) ell += left[col] * family.table[col];
            const F initial_mask = [&]() {
                F value;
                value.setByCSPRNG();
                return value;
            }();
            const F masked_initial_claim = ell + initial_mask;
            const zkgpt::native_pedersen::ScalarOpening lookup_initial_opening = scalar_evaluation_opening(scalar_basis, ell);

            const zkgpt::native_pedersen::ScalarOpening lookup_initial_mask = zkgpt::native_pedersen::commit_fresh(scalar_basis, initial_mask);
            zkgpt::native_pedersen::ScalarOpening lookup_masked_initial_opening;
            lookup_masked_initial_opening.value = lookup_initial_opening.value + lookup_initial_mask.value;
            lookup_masked_initial_opening.blinding = lookup_initial_opening.blinding + lookup_initial_mask.blinding;
            lookup_masked_initial_opening.commitment =
                zkgpt::native_pedersen::commit(scalar_basis, lookup_masked_initial_opening.value, lookup_masked_initial_opening.blinding);
            session.absorb_field(prefix + "/a-query/claim", masked_initial_claim);
            NativeMaskedTerminalSpec lookup_spec;
            lookup_spec.kind = NativeMaskedTerminalKind::Product2Public;
            lookup_spec.public_vector = family.table;
            lookup_spec.has_external_binding = true;
            lookup_spec.has_masked_initial_binding = true;
            lookup_spec.initial_opening = lookup_initial_opening;
            lookup_spec.initial_commitment = lookup_initial_opening.commitment;
            lookup_spec.masked_initial_opening = lookup_masked_initial_opening;
            lookup_spec.masked_initial_commitment = lookup_masked_initial_opening.commitment;
            NativeMaskedSumcheckProver lookup_prover(session, prefix + "/f-lookup", 2, masked_initial_claim, lookup_spec, &ell);
            std::vector<F> lookup_left = left;
            std::vector<F> lookup_right = family.table;
            std::vector<F> lookup_challenges;
            lookup_challenges.reserve(relation_rounds(cols));
            for (std::size_t round = 0; round < relation_rounds(cols); ++round) {
                std::vector<F> next_left, next_right;
                F challenge;
                append_product_round(lookup_prover, lookup_left, lookup_right, next_left, next_right, reason, &challenge);
                lookup_challenges.push_back(challenge);
                lookup_left.swap(next_left);
                lookup_right.swap(next_right);
            }
            std::vector<F> lookup_point = row_point;
            lookup_point.insert(lookup_point.end(), lookup_challenges.begin(), lookup_challenges.end());
            const zkgpt::native_pedersen::ScalarOpening lookup_raw_opening = scalar_evaluation_opening(scalar_basis, sparse.evaluate_mle(lookup_point));
            lookup_spec.unmasked_terminal_openings = {lookup_raw_opening};

            lookup_spec.public_evaluation = lookup_right.front();
            if (!lookup_prover.set_terminal_spec(lookup_spec, reason)) return false;
            NativeMaskedSumcheckResult lookup_result;
            if (!lookup_prover.finish({lookup_left.front()}, &lookup_result, reason)) return false;
            hyrax::MaskedMleOpening lookup_opening;
            {
                MatrixRowFold lookup_fold = fold_family_rows(family, states[family_index], lookup_point);
                if (!hyrax::open_masked_mle_with_claim(lookup_fold.commitment, lookup_fold.view, lookup_fold.values, lookup_result.terminal_values[0],
                                                       session, prefix + "/g-matrix-open", lookup_opening, reason, false, "", &scalar_basis,
                                                       &lookup_result.terminal_openings[0], &lookup_raw_opening)) return false;
            }

            F query_eval = F_ZERO;
            zkgpt::native_for_each_equality_weight(row_point, rows, [&](std::size_t row, const F& row_weight) {
                const std::uint64_t absolute = family.segment_offset + row;
                const F selected = selected_query_value(family, row);
                if (!query.value_provider.at || absolute >= query.value_provider.size || query.value_provider.at(absolute) != selected)
                    throw std::runtime_error("native Lasso selected query differs from L_qry provider");
                query_eval += row_weight * selected;
            });
            if (lookup_initial_opening.value != query_eval) throw std::runtime_error("native Lasso lookup/query claim mismatch");
            session.absorb_field(prefix + "/h-input/claim", lookup_result.final_claim);
            hyrax::MaskedMleOpening input_opening;
            const std::vector<F> query_point = lift_segment_point(row_point, static_cast<std::size_t>(family.segment_offset), rows,
                                                                  static_cast<std::size_t>(query_size));
            if (!hyrax::open_hyrax_tensor_mle_with_claim(query.tensor_view, query.tensor_commitment, query.tensor_state, query_point,
                                                         query.value_provider, lookup_result.final_claim, session, prefix + "/h-input-open",
                                                         input_opening, reason, "", &scalar_basis, &lookup_result.claim_opening,
                                                         &lookup_initial_opening))
                return false;
            (void)query_eval;
            std::vector<F>().swap(states[family_index].row_blindings);
        }

        const std::size_t final_rounds = relation_rounds(static_cast<std::size_t>(query_size));
        local.final_query_point.reserve(final_rounds);
        for (std::size_t round = 0; round < final_rounds; ++round)
            local.final_query_point.push_back(session.challenge("lasso/i-final/r/" + std::to_string(round)));
        F final_claim;
        final_claim.setByCSPRNG();
        session.absorb_field("lasso/i-final/claim", final_claim);
        hyrax::MaskedMleOpening final_opening;
        if (!hyrax::open_hyrax_tensor_mle_with_claim(query.tensor_view, query.tensor_commitment, query.tensor_state, local.final_query_point,
                                                     query.value_provider, final_claim, session, NativeLassoTerminalOpeningLabels::l_qry(),
                                                     final_opening, reason, "", &scalar_basis))
            return false;
        if (final_opening.masked_evaluation != final_claim) throw std::runtime_error("native Lasso final opening claim mismatch");
        local.final_l_evaluation = final_claim;
        local.final_l_commitment = final_opening.masked_evaluation_opening.commitment;
        local.has_final_l_commitment = true;
        if (!session.replay(reason)) return false;
        if (result != nullptr) *result = std::move(local);
        checkpoint_guard.commit();
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool native_lasso_verify_bundle(const NativeProofSession& session, const std::vector<NativeLassoFamilyPublic>& families, const NativeLassoQueryPublic& query,
                                NativeLassoBundleResult* result, std::string* reason, const NativeLassoTensorTailVerifyHook& tensor_tail_stage) {
    try {
        const std::uint64_t query_size = query_domain(query);
        if (query.tensor_view.full_domain == 0 || !tensor_tail_stage || !check_family_order(families, reason) || query_size > std::numeric_limits<std::size_t>::max() ||
            !validate_query_public(query, static_cast<std::size_t>(query_size), reason) ||
            !check_statement_identity_with_l_digest(session, families,
                                                    native_lasso_tensor_query_digest(query.tensor_view, query.tensor_commitment),
                                                    query.statement_l_digest, reason)) {
            if (query.tensor_view.full_domain == 0) set_reason(reason, "formal native Lasso requires a typed balanced L_qry tensor");
            else if (!tensor_tail_stage) set_reason(reason, "formal native Lasso requires the shared Aux/L_qry tensor-tail certificate verifier");
            return false;
        }
        for (const NativeLassoFamilyPublic& family : families)
            if (family.segment_offset > query_size || family.padded_queries > query_size - family.segment_offset) {
                set_reason(reason, "native Lasso public family exceeds L_qry segment");
                return false;
            }
        if (!session.replay(reason)) return false;
        const zkgpt::native_pedersen::PedersenBasis scalar_basis = lasso_scalar_basis(session);
        NativeLassoBundleResult local;
        std::vector<std::vector<std::size_t>> row_chunk_positions(families.size());
        const std::string first_commitment_label = family_prefix(families.front()) + "/commitment";
        const std::size_t first_commitment = find_message(session, first_commitment_label, NativeProofMessageKind::Data, reason);
        if (first_commitment == session.messages().size()) return false;
        for (std::size_t index = 0; index < first_commitment; ++index) {
            const std::string& label = session.messages()[index].label;
            if (is_lasso_protocol_message(label)) {
                set_reason(reason, "native Lasso scope has a hidden prefix message");
                return false;
            }
        }
        std::size_t position = first_commitment;
        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyPublic& family = families[family_index];
            const std::string expected = family_prefix(family) + "/commitment";
            const NativeProofMessage* message = consume(session, position, NativeProofMessageKind::Data, expected, reason);
            if (message == nullptr) return false;
            const std::string basis_id = family_basis_id(session, family);
            std::vector<G1> row_commitments;
            if (!parse_masked_family_commitment(*message, family, basis_id, row_commitments, reason)) return false;
            const std::size_t chunk_count = row_commitment_chunk_count(family.matrix_layout.rows);
            const std::size_t last_chunk_count = row_commitment_last_chunk_count(family.matrix_layout.rows);
            for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
                const std::size_t count = chunk + 1 == chunk_count ? last_chunk_count : kNativeLassoRowCommitmentChunkEntries;
                const std::string label = expected + "/rows/" + std::to_string(chunk);
                const NativeProofMessage* row_message = consume(session, position, NativeProofMessageKind::Data, label, reason);
                if (row_message == nullptr || !parse_masked_family_row_chunk(*row_message, count, row_commitments, reason)) return false;
                row_chunk_positions[family_index].push_back(position - 1);
            }
            if (row_commitments.size() != family.matrix_layout.rows) {
                set_reason(reason, "native Lasso row commitment count mismatch");
                return false;
            }
        }

        if (!tensor_tail_stage(session, query, position, reason)) {
            if (reason != nullptr && reason->empty()) set_reason(reason, "native Lasso shared tensor-tail certificate verification failed");
            return false;
        }

        std::vector<std::vector<F>> booleanity_weights(families.size());
        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyPublic& family = families[family_index];
            const std::size_t rounds = relation_rounds(static_cast<std::size_t>(family.matrix_layout.vector_size));
            std::vector<F>& beta = booleanity_weights[family_index];
            beta.reserve(rounds);
            for (std::size_t round = 0; round < rounds; ++round) {
                F challenge;
                const std::string label = "challenge/" + family_prefix(family) + "/booleanity/weight/" + std::to_string(round);
                const NativeProofMessage* message = consume(session, position, NativeProofMessageKind::Challenge, label, reason);
                if (message == nullptr || !decode_field(message->payload, challenge)) {
                    set_reason(reason, "malformed native Lasso booleanity weight challenge");
                    return false;
                }
                beta.push_back(challenge);
            }
        }
        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyPublic& family = families[family_index];
            const std::string prefix = family_prefix(family);
            std::vector<G1> row_commitments;
            row_commitments.reserve(static_cast<std::size_t>(family.matrix_layout.rows));
            const std::size_t chunk_count = row_commitment_chunk_count(family.matrix_layout.rows);
            const std::size_t last_chunk_count = row_commitment_last_chunk_count(family.matrix_layout.rows);
            if (row_chunk_positions[family_index].size() != chunk_count) {
                set_reason(reason, "native Lasso row commitment chunk count changed after transcript replay");
                return false;
            }
            for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
                const std::size_t count = chunk + 1 == chunk_count ? last_chunk_count : kNativeLassoRowCommitmentChunkEntries;
                const std::size_t message_index = row_chunk_positions[family_index][chunk];
                if (message_index >= session.messages().size() ||
                    !parse_masked_family_row_chunk(session.messages()[message_index], count, row_commitments, reason)) return false;
            }
            if (row_commitments.size() != family.matrix_layout.rows) {
                set_reason(reason, "native Lasso row commitment count mismatch after transcript replay");
                return false;
            }
            std::vector<F> row_point;
            for (std::size_t round = 0; round < relation_rounds(family.padded_queries); ++round) {
                F challenge;
                const NativeProofMessage* message =
                    consume(session, position, NativeProofMessageKind::Challenge, "challenge/" + prefix + "/a-query/r/" + std::to_string(round), reason);
                if (message == nullptr || !decode_field(message->payload, challenge)) {
                    set_reason(reason, "malformed native Lasso row challenge");
                    return false;
                }
                row_point.push_back(challenge);
            }
            local.family_query_points.push_back(row_point);

            F boolean_claim;
            if (!consume_field_message(session, position, prefix + "/b-boolean/claim", boolean_claim, reason) || !boolean_claim.isZero()) return false;
            NativeMaskedTerminalSpec boolean_spec;
            if (!build_sumcheck_verify_spec(session, prefix + "/b-boolean/sum", NativeMaskedTerminalKind::Product3Public, F_ZERO, scalar_basis, boolean_spec, reason,
                                            true))
                return false;
            boolean_spec.public_provider = equality_weight_provider(booleanity_weights[family_index]);
            NativeMaskedSumcheckResult boolean_result;
            std::string sumcheck_reason;
            if (!native_masked_sumcheck_verify(session, prefix + "/b-boolean/sum", 3,
                                               relation_rounds(static_cast<std::size_t>(family.matrix_layout.vector_size)),
                                               boolean_claim, boolean_spec, &boolean_result, &sumcheck_reason)) {
                set_reason(reason, prefix + "/b-boolean/sum: " + sumcheck_reason);
                return false;
            }
            if (!advance_masked_scope(session, prefix + "/b-boolean/sum", position, reason)) return false;
            const std::string matrix_basis_id = family_basis_id(session, family);
            if (!verify_masked_matrix_opening(session, family, row_commitments, matrix_basis_id, boolean_result.challenges,
                                              prefix + "/c-boolean-open", boolean_result.terminal_values[0], reason, false, &scalar_basis,
                                              &boolean_result.unmasked_terminal_commitments[0], &boolean_result.terminal_commitments[0]))
                return false;
            if (!advance_masked_scope(session, prefix + "/c-boolean-open", position, reason)) return false;
            if (!verify_masked_matrix_opening(session, family, row_commitments, matrix_basis_id, boolean_result.challenges,
                                              prefix + "/c-boolean-complement-open", boolean_result.terminal_values[1], reason, true, &scalar_basis,
                                              &boolean_result.unmasked_terminal_commitments[1], &boolean_result.terminal_commitments[1]))
                return false;
            if (!advance_masked_scope(session, prefix + "/c-boolean-complement-open", position, reason)) return false;

            F onehot_claim;
            if (!consume_field_message(session, position, prefix + "/d-onehot/claim", onehot_claim, reason) || onehot_claim != F_ONE) return false;
            NativeMaskedTerminalSpec onehot_spec;
            if (!build_sumcheck_verify_spec(session, prefix + "/d-onehot/sum", NativeMaskedTerminalKind::Scalar, F_ONE, scalar_basis, onehot_spec, reason,
                                            true))
                return false;
            NativeMaskedSumcheckResult onehot_result;
            sumcheck_reason.clear();
            if (!native_masked_sumcheck_verify(session, prefix + "/d-onehot/sum", 1, relation_rounds(family.table.size()), onehot_claim, onehot_spec,
                                               &onehot_result, &sumcheck_reason)) {
                set_reason(reason, prefix + "/d-onehot/sum: " + sumcheck_reason);
                return false;
            }
            if (!advance_masked_scope(session, prefix + "/d-onehot/sum", position, reason)) return false;
            std::vector<F> onehot_point = row_point;
            onehot_point.insert(onehot_point.end(), onehot_result.challenges.begin(), onehot_result.challenges.end());
            if (!verify_masked_matrix_opening(session, family, row_commitments, matrix_basis_id, onehot_point, prefix + "/e-onehot-open",
                                              onehot_result.final_claim, reason, false, &scalar_basis,
                                              &onehot_result.unmasked_terminal_commitments[0], &onehot_result.terminal_commitments[0]))
                return false;
            if (!advance_masked_scope(session, prefix + "/e-onehot-open", position, reason)) return false;

            F lookup_claim;
            if (!consume_field_message(session, position, prefix + "/a-query/claim", lookup_claim, reason)) return false;
            NativeMaskedTerminalSpec lookup_spec;
            if (!build_sumcheck_verify_spec(session, prefix + "/f-lookup", NativeMaskedTerminalKind::Product2Public, F_ZERO, scalar_basis, lookup_spec, reason,
                                            false))
                return false;
            lookup_spec.public_vector = family.table;
            NativeMaskedSumcheckResult lookup_result;
            sumcheck_reason.clear();
            if (!native_masked_sumcheck_verify(session, prefix + "/f-lookup", 2, relation_rounds(family.table.size()), lookup_claim, lookup_spec,
                                               &lookup_result, &sumcheck_reason)) {
                set_reason(reason, prefix + "/f-lookup: " + sumcheck_reason);
                return false;
            }
            if (!advance_masked_scope(session, prefix + "/f-lookup", position, reason)) return false;
            std::vector<F> lookup_point = row_point;
            lookup_point.insert(lookup_point.end(), lookup_result.challenges.begin(), lookup_result.challenges.end());
            if (!verify_masked_matrix_opening(session, family, row_commitments, matrix_basis_id, lookup_point, prefix + "/g-matrix-open",
                                              lookup_result.terminal_values[0], reason, false, &scalar_basis,
                                              &lookup_result.unmasked_terminal_commitments[0], &lookup_result.terminal_commitments[0]))
                return false;
            if (!advance_masked_scope(session, prefix + "/g-matrix-open", position, reason)) return false;
            F input_claim;
            if (!consume_field_message(session, position, prefix + "/h-input/claim", input_claim, reason) || input_claim != lookup_result.final_claim)
                return false;
            const std::vector<F> query_point = lift_segment_point(row_point, static_cast<std::size_t>(family.segment_offset), family.padded_queries,
                                                                  static_cast<std::size_t>(query_size));
            if (!verify_masked_tensor_opening(session, query.tensor_view, query.tensor_commitment, query_point, prefix + "/h-input-open", input_claim,
                                              reason, true, &scalar_basis, &lookup_result.initial_commitment,
                                              &lookup_result.claim_commitment))
                return false;
            if (!advance_masked_scope(session, prefix + "/h-input-open", position, reason)) return false;
        }

        local.final_query_point.reserve(relation_rounds(static_cast<std::size_t>(query_size)));
        for (std::size_t round = 0; round < relation_rounds(static_cast<std::size_t>(query_size)); ++round) {
            F challenge;
            const NativeProofMessage* message =
                consume(session, position, NativeProofMessageKind::Challenge, "challenge/lasso/i-final/r/" + std::to_string(round), reason);
            if (message == nullptr || !decode_field(message->payload, challenge)) {
                set_reason(reason, "malformed native Lasso final query challenge");
                return false;
            }
            local.final_query_point.push_back(challenge);
        }
        F final_claim;
        if (!consume_field_message(session, position, "lasso/i-final/claim", final_claim, reason)) return false;
        G1 final_mask_commitment;
        F encoded_final_claim;
        zkgpt::native_pedersen::ScalarCommitment final_raw_commitment;
        zkgpt::native_pedersen::ScalarCommitment final_masked_commitment;
        if (!parse_masked_open_claim(session, NativeLassoTerminalOpeningLabels::l_qry(), final_mask_commitment, encoded_final_claim, reason, &scalar_basis,
                                     &final_raw_commitment, &final_masked_commitment) ||
            encoded_final_claim != final_claim) {
            set_reason(reason, "native Lasso final scalar claim binding is malformed");
            return false;
        }
        if (!verify_masked_tensor_opening(session, query.tensor_view, query.tensor_commitment, local.final_query_point,
                                          NativeLassoTerminalOpeningLabels::l_qry(), final_claim, reason, true, &scalar_basis,
                                          &final_raw_commitment, &final_masked_commitment))
            return false;
        if (!advance_masked_scope(session, NativeLassoTerminalOpeningLabels::l_qry(), position, reason)) return false;
        local.final_l_evaluation = final_claim;
        local.final_l_commitment = final_masked_commitment;
        local.has_final_l_commitment = true;
        for (std::size_t index = position; index < session.messages().size(); ++index) {
            const std::string& label = session.messages()[index].label;
            if (is_lasso_protocol_message(label)) {
                set_reason(reason, "native Lasso proof has trailing or inserted messages");
                return false;
            }
        }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}

namespace native_lasso_legacy_clear {
bool native_lasso_verify_bundle(const NativeProofSession& session, const std::vector<NativeLassoFamilyPublic>& families, const NativeLassoQueryPublic& query,
                                NativeLassoBundleResult* result, std::string* reason) {
    try {
        if (!check_family_order(families, reason) || !validate_query_public(query, query.generators.size(), reason) ||
            !check_statement_identity(session, families, query.commitment, query.generators.size(), query.statement_l_digest, reason))
            return false;
        for (const NativeLassoFamilyPublic& family : families) {
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
        for (const NativeLassoFamilyPublic& family : families) {
            const std::string expected = family_prefix(family) + "/commitment";
            const NativeProofMessage* message = consume(session, position, NativeProofMessageKind::Data, expected, reason);
            if (message == nullptr) return false;
            G1 commitment;
            if (!parse_commitment(*message, family, query.G, commitment, reason)) return false;
            local.family_commitments.push_back(commitment);
        }

        for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
            const NativeLassoFamilyPublic& family = families[family_index];
            const std::string prefix = family_prefix(family);
            const std::size_t row_rounds = log2_power(family.padded_queries);
            std::vector<F> row_point;
            for (std::size_t round = 0; round < row_rounds; ++round) {
                F value;
                const NativeProofMessage* challenge =
                    consume(session, position, NativeProofMessageKind::Challenge, "challenge/" + prefix + "/a-query/r/" + std::to_string(round), reason);
                if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                    set_reason(reason, "native Lasso query challenge is malformed");
                    return false;
                }
                row_point.push_back(value);
            }
            local.family_query_points.push_back(row_point);

            const NativeProofMessage* boolean_claim = consume(session, position, NativeProofMessageKind::Field, prefix + "/b-boolean/claim", reason);
            F boolean_initial;
            if (boolean_claim == nullptr || !decode_field(boolean_claim->payload, boolean_initial) || !boolean_initial.isZero()) {
                set_reason(reason, "native Lasso boolean claim is malformed");
                return false;
            }
            const std::string boolean_label = prefix + "/b-boolean/sum";
            if (position >= session.messages().size() || session.messages()[position].label != boolean_label + "/poly/0") {
                set_reason(reason, "native Lasso booleanity Sumcheck is not adjacent");
                return false;
            }
            F boolean_eval, boolean_minus_one_eval;
            if (!native_range_verify_product(session, boolean_label, boolean_initial, &boolean_eval, &boolean_minus_one_eval, reason)) return false;
            const std::size_t boolean_terminal = find_message(session, boolean_label + "/terminal", NativeProofMessageKind::FieldVector, reason);
            if (boolean_terminal == session.messages().size() || boolean_terminal < position) {
                set_reason(reason, "native Lasso booleanity terminal is missing");
                return false;
            }
            position = boolean_terminal + 1;
            if (boolean_minus_one_eval != boolean_eval - F(1)) {
                set_reason(reason, "native Lasso boolean terminal relation failed");
                return false;
            }
            const std::size_t matrix_rounds = log2_power(family.padded_queries * family.table.size());
            std::vector<F> boolean_point;
            if (!collect_challenges(session, boolean_label, matrix_rounds, boolean_point, reason)) return false;
            std::vector<F> boolean_basis = mle_basis(boolean_point);
            const std::string boolean_open_label = prefix + "/c-boolean-open";
            if (find_message(session, boolean_open_label + "/basis", NativeProofMessageKind::Data, reason) != position) {
                set_reason(reason, "native Lasso boolean opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(local.family_commitments[family_index], boolean_eval, query.G, family.legacy_clear_matrix_generators, boolean_basis,
                                                    session, boolean_open_label, reason, false, true))
                return false;
            const std::size_t boolean_open_terminal = find_message(session, boolean_open_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (boolean_open_terminal == session.messages().size() || boolean_open_terminal < position) {
                set_reason(reason, "native Lasso boolean opening terminal is missing");
                return false;
            }
            position = boolean_open_terminal + 1;

            const G1 complement_commit = local.family_commitments[family_index] - commit_all_ones(family.legacy_clear_matrix_generators);
            const std::string complement_open_label = prefix + "/c-boolean-complement-open";
            if (find_message(session, complement_open_label + "/basis", NativeProofMessageKind::Data, reason) != position) {
                set_reason(reason, "native Lasso boolean complement opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(complement_commit, boolean_minus_one_eval, query.G, family.legacy_clear_matrix_generators, boolean_basis, session,
                                                    complement_open_label, reason, false, true))
                return false;
            const std::size_t complement_open_terminal = find_message(session, complement_open_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (complement_open_terminal == session.messages().size() || complement_open_terminal < position) {
                set_reason(reason, "native Lasso boolean complement terminal is missing");
                return false;
            }
            position = complement_open_terminal + 1;

            std::vector<F> onehot_point;
            for (std::size_t round = 0; round < row_rounds; ++round) {
                F value;
                const NativeProofMessage* challenge =
                    consume(session, position, NativeProofMessageKind::Challenge, "challenge/" + prefix + "/d-onehot/r/" + std::to_string(round), reason);
                if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                    set_reason(reason, "native Lasso one-hot challenge is malformed");
                    return false;
                }
                onehot_point.push_back(value);
            }
            const NativeProofMessage* onehot_claim = consume(session, position, NativeProofMessageKind::Field, prefix + "/d-onehot/claim", reason);
            F onehot_initial;
            if (onehot_claim == nullptr || !decode_field(onehot_claim->payload, onehot_initial) || onehot_initial != F_ONE) {
                set_reason(reason, "native Lasso one-hot claim is malformed");
                return false;
            }
            const std::string onehot_label = prefix + "/d-onehot/sum";
            if (position >= session.messages().size() || session.messages()[position].label != onehot_label + "/poly/0") {
                set_reason(reason, "native Lasso one-hot Sumcheck is not adjacent");
                return false;
            }
            F onehot_eval;
            if (!native_range_verify_deg1(session, onehot_label, onehot_initial, &onehot_eval, reason)) return false;
            const std::size_t onehot_terminal = find_message(session, onehot_label + "/terminal", NativeProofMessageKind::Field, reason);
            if (onehot_terminal == session.messages().size() || onehot_terminal < position) {
                set_reason(reason, "native Lasso one-hot terminal is missing");
                return false;
            }
            position = onehot_terminal + 1;
            const std::size_t col_rounds = log2_power(family.table.size());
            std::vector<F> onehot_columns;
            if (!collect_challenges(session, onehot_label, col_rounds, onehot_columns, reason)) return false;
            std::vector<F> onehot_matrix_point;
            onehot_matrix_point.reserve(onehot_point.size() + onehot_columns.size());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot_point.begin(), onehot_point.end());
            onehot_matrix_point.insert(onehot_matrix_point.end(), onehot_columns.begin(), onehot_columns.end());
            std::vector<F> onehot_basis = mle_basis(onehot_matrix_point);
            const std::string onehot_open_label = prefix + "/e-onehot-open";
            if (find_message(session, onehot_open_label + "/basis", NativeProofMessageKind::Data, reason) != position) {
                set_reason(reason, "native Lasso one-hot opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(local.family_commitments[family_index], onehot_eval, query.G, family.legacy_clear_matrix_generators, onehot_basis,
                                                    session, onehot_open_label, reason, false, true))
                return false;
            const std::size_t onehot_open_terminal = find_message(session, onehot_open_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (onehot_open_terminal == session.messages().size() || onehot_open_terminal < position) {
                set_reason(reason, "native Lasso one-hot opening terminal is missing");
                return false;
            }
            position = onehot_open_terminal + 1;

            const NativeProofMessage* query_claim = consume(session, position, NativeProofMessageKind::Field, prefix + "/a-query/claim", reason);
            F ell;
            if (query_claim == nullptr || !decode_field(query_claim->payload, ell)) {
                set_reason(reason, "native Lasso query claim is malformed");
                return false;
            }
            const std::string lookup_label = prefix + "/f-lookup";
            const std::string first_lookup_poly = lookup_label + "/poly/0";
            if (position >= session.messages().size() || session.messages()[position].label != first_lookup_poly) {
                set_reason(reason, "native Lasso lookup Sumcheck is not adjacent");
                return false;
            }
            F matrix_eval, table_eval;
            if (!native_range_verify_product(session, lookup_label, ell, &matrix_eval, &table_eval, reason)) return false;
            const std::size_t lookup_terminal = find_message(session, lookup_label + "/terminal", NativeProofMessageKind::FieldVector, reason);
            if (lookup_terminal == session.messages().size() || lookup_terminal < position) {
                set_reason(reason, "native Lasso lookup terminal is missing");
                return false;
            }
            position = lookup_terminal + 1;
            std::vector<F> y_point;
            if (!collect_challenges(session, lookup_label, col_rounds, y_point, reason)) return false;
            const std::vector<F> table_basis = mle_basis(y_point);
            F expected_table = 0;
            for (std::size_t i = 0; i < family.table.size(); ++i) expected_table += table_basis[i] * family.table[i];
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
            const std::size_t matrix_basis_index = find_message(session, matrix_label + "/basis", NativeProofMessageKind::Data, reason);
            if (matrix_basis_index != position) {
                set_reason(reason, "native Lasso M_f opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(local.family_commitments[family_index], matrix_eval, query.G, family.legacy_clear_matrix_generators, matrix_basis,
                                                    session, matrix_label, reason, false, true))
                return false;
            const std::size_t matrix_terminal = find_message(session, matrix_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (matrix_terminal == session.messages().size() || matrix_terminal < position) {
                set_reason(reason, "native Lasso M_f terminal is missing");
                return false;
            }
            position = matrix_terminal + 1;

            const NativeProofMessage* input_claim = consume(session, position, NativeProofMessageKind::FieldVector, prefix + "/h-input/claim", reason);
            std::vector<F> input_values;
            if (!decode_vector(input_claim, 2, input_values) || input_values[0] != ell || input_values[1] != input_values[0]) {
                set_reason(reason, "native Lasso query/input consistency claim failed");
                return false;
            }
            std::vector<F> selector_basis(query.generators.size(), F_ZERO);
            const std::vector<F> row_basis = mle_basis(row_point);
            for (std::size_t row = 0; row < family.padded_queries; ++row) selector_basis[family.segment_offset + row] = row_basis[row];
            const std::string input_label = prefix + "/h-input-open";
            const std::size_t input_basis_index = find_message(session, input_label + "/basis", NativeProofMessageKind::Data, reason);
            if (input_basis_index != position) {
                set_reason(reason, "native Lasso input opening is not adjacent");
                return false;
            }
            if (!hyrax::verify_opening_with_session(query.commitment, input_values[1], query.G, query.generators, selector_basis, session, input_label, reason,
                                                    true, true))
                return false;
            const std::size_t input_terminal = find_message(session, input_label + "/terminal", NativeProofMessageKind::Data, reason);
            if (input_terminal == session.messages().size() || input_terminal < position) {
                set_reason(reason, "native Lasso input terminal is missing");
                return false;
            }
            position = input_terminal + 1;
        }

        const std::size_t final_rounds = log2_power(query.generators.size());
        for (std::size_t round = 0; round < final_rounds; ++round) {
            F value;
            const NativeProofMessage* challenge =
                consume(session, position, NativeProofMessageKind::Challenge, "challenge/lasso/i-final/r/" + std::to_string(round), reason);
            if (challenge == nullptr || !decode_field(challenge->payload, value)) {
                set_reason(reason, "native Lasso final query challenge is malformed");
                return false;
            }
            local.final_query_point.push_back(value);
        }
        const std::vector<F> final_basis = mle_basis(local.final_query_point);
        const NativeProofMessage* final_claim = consume(session, position, NativeProofMessageKind::Field, "lasso/i-final/claim", reason);
        F final_eval;
        if (final_claim == nullptr || !decode_field(final_claim->payload, final_eval)) {
            set_reason(reason, "native Lasso final L_qry claim is malformed");
            return false;
        }
        local.final_l_evaluation = final_eval;
        const std::size_t final_basis_index =
            find_message(session, std::string(NativeLassoTerminalOpeningLabels::l_qry()) + "/basis", NativeProofMessageKind::Data, reason);
        if (final_basis_index != position) {
            set_reason(reason, "native Open_L is not adjacent to its claim");
            return false;
        }
        if (!hyrax::verify_opening_with_session(query.commitment, final_eval, query.G, query.generators, final_basis, session,
                                                NativeLassoTerminalOpeningLabels::l_qry(), reason, true, false))
            return false;
        const std::size_t final_terminal =
            find_message(session, std::string(NativeLassoTerminalOpeningLabels::l_qry()) + "/terminal", NativeProofMessageKind::Data, reason);
        if (final_terminal == session.messages().size() || final_terminal < position) {
            set_reason(reason, "native Open_L terminal is missing");
            return false;
        }
        position = final_terminal + 1;
        for (std::size_t index = position; index < session.messages().size(); ++index) {
            const std::string& label = session.messages()[index].label;
            if (label.compare(0, sizeof("lasso/") - 1, "lasso/") == 0 || label.compare(0, sizeof("challenge/lasso/") - 1, "challenge/lasso/") == 0) {
                set_reason(reason, "native Lasso proof has trailing or inserted messages");
                return false;
            }
        }
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception& error) {
        set_reason(reason, error.what());
        return false;
    }
}
}
