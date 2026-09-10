#include "gat_native_core.cuh"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace zkdl_gat {
namespace {

using FrWire = zkdl::FrWire;

constexpr const char* kBridgeApi = "zkdl_native_gat_bridge_v1";
constexpr const char* kProofScope = "full_gat_forward";

std::string digest_hex(const std::array<zkdl::byte, 32>& digest) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(64U);
    for (const auto byte : digest) {
        out.push_back(hex[(byte >> 4U) & 0x0fU]);
        out.push_back(hex[byte & 0x0fU]);
    }
    return out;
}

bool parse_digest(const std::string& text, std::array<zkdl::byte, 32>& digest) {
    if (text.size() != 64U) return false;
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < digest.size(); ++i) {
        const int hi = nibble(text[2U * i]);
        const int lo = nibble(text[2U * i + 1U]);
        if (hi < 0 || lo < 0) return false;
        digest[i] = static_cast<zkdl::byte>((hi << 4) | lo);
    }
    return true;
}

bool read_file(const std::filesystem::path& path, std::vector<zkdl::byte>& bytes,
               std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error) *error = "cannot open authenticated resource: " + path.string();
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > (1ULL << 32)) {
        if (error) *error = "authenticated resource has invalid size: " + path.string();
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input && !bytes.empty()) {
        if (error) *error = "cannot read authenticated resource: " + path.string();
        return false;
    }
    return true;
}

bool file_digest(const std::filesystem::path& path,
                 std::array<zkdl::byte, 32>& digest, std::string* error) {
    std::vector<zkdl::byte> bytes;
    if (!read_file(path, bytes, error)) return false;
    digest = zkdl::sha256(bytes);
    return true;
}

bool contains(const std::vector<zkdl::byte>& bytes, const std::string& needle) {
    if (needle.empty() || bytes.size() < needle.size()) return false;
    return std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end()) != bytes.end();
}

bool manifest_string_field(const std::vector<zkdl::byte>& bytes,
                           const std::string& key, std::string& value) {
    const std::string text(bytes.begin(), bytes.end());
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) return false;
    const std::size_t colon = text.find(':', key_pos + marker.size());
    if (colon == std::string::npos) return false;
    const std::size_t first_quote = text.find('"', colon + 1U);
    if (first_quote == std::string::npos) return false;
    const std::size_t last_quote = text.find('"', first_quote + 1U);
    if (last_quote == std::string::npos || last_quote == first_quote + 1U) return false;
    // Resource paths and hexadecimal identities are not allowed to contain
    // escapes in the canonical manifest representation.
    if (text.find('\\', first_quote + 1U) != std::string::npos &&
        text.find('\\', first_quote + 1U) < last_quote) return false;
    value = text.substr(first_quote + 1U, last_quote - first_quote - 1U);
    return true;
}

bool reject(NativeProofResult* result, std::string* error,
            const std::string& message, const Config* config = nullptr,
            const PublicStatement* statement = nullptr, Backend backend = Backend::Cpu) {
    if (result) {
        *result = NativeProofResult{};
        result->envelope.native_core_api = kBridgeApi;
        result->envelope.status = "BLOCKED";
        result->envelope.backend = backend_name(backend);
        result->envelope.proof_scope = kProofScope;
        result->envelope.silent_fallback = false;
        result->envelope.verifier_reads_witness = false;
        result->envelope.output_bound = false;
        result->envelope.gpu_kernel_executed = false;
        result->envelope.error = message;
        if (config) {
            result->envelope.scale_chain_digest = config->scale_chain_digest;
            result->envelope.integer_bound_ledger_id = "UNAVAILABLE";
            result->envelope.integer_bound_ledger_digest = "UNAVAILABLE";
            result->envelope.integer_bound_ledger_descriptor_digest = "UNAVAILABLE";
        }
        if (statement) {
            result->envelope.statement_id = statement->statement_id;
            result->envelope.static_table_descriptor_digest = statement->static_table_descriptor_digest;
            result->envelope.scale_chain_digest = statement->scale_chain_digest;
            result->envelope.integer_bound_ledger_id = statement->integer_bound_ledger_id;
            result->envelope.integer_bound_ledger_digest = statement->integer_bound_ledger_digest;
            result->envelope.integer_bound_ledger_descriptor_digest = statement->integer_bound_ledger_descriptor_digest;
        }
    }
    if (error) *error = message;
    return false;
}

bool resource_admission(const Config& config, const NativeRelationWitness& witness,
                        std::string* error) {
    if (!witness.srs_authenticated || witness.generators.empty()) {
        if (error) *error = "BLOCKED: native relation proof requires authenticated SRS generators";
        return false;
    }
    if (!witness.registry_authenticated || witness.registry_path.empty()) {
        if (error) *error = "BLOCKED: native relation proof requires an authenticated verifier registry path";
        return false;
    }
    const std::filesystem::path srs_root(config.srs_path);
    if (!std::filesystem::is_directory(srs_root)) {
        if (error) *error = "BLOCKED: canonical SRS directory is missing: " + srs_root.string();
        return false;
    }
    const std::filesystem::path manifest = srs_root / "srs_manifest.json";
    const std::filesystem::path coefficient = srs_root / "pk_coef.bin";
    if (!std::filesystem::is_regular_file(manifest) ||
        !std::filesystem::is_regular_file(coefficient)) {
        if (error) *error = "BLOCKED: canonical SRS manifest/pk_coef.bin is missing under " + srs_root.string();
        return false;
    }
    std::vector<zkdl::byte> manifest_bytes;
    if (!read_file(manifest, manifest_bytes, error)) return false;
    if (!contains(manifest_bytes, "\"manifest_format\": \"zkgat_srs_manifest\"" ) ||
        !contains(manifest_bytes, "\"curve_id\": \"bls12_381\"" ) ||
        !contains(manifest_bytes, "\"pk_coef_path\": \"pk_coef.bin\"" ) ||
        !contains(manifest_bytes, "\"srs_id\": \"" + config.srs_id + "\"")) {
        if (error) *error = "BLOCKED: canonical SRS manifest identity/format is not authenticated";
        return false;
    }
    std::array<zkdl::byte, 32> manifest_digest{};
    manifest_digest = zkdl::sha256(manifest_bytes);
    if (!std::equal(manifest_digest.begin(), manifest_digest.end(), witness.srs_digest.begin())) {
        if (error) *error = "BLOCKED: supplied SRS generator identity does not match srs_manifest.json";
        return false;
    }
    std::string pk_coef_hash;
    std::array<zkdl::byte, 32> expected_pk_coef{};
    std::array<zkdl::byte, 32> actual_pk_coef{};
    if (!manifest_string_field(manifest_bytes, "pk_coef_hash", pk_coef_hash) ||
        !parse_digest(pk_coef_hash, expected_pk_coef) ||
        !file_digest(coefficient, actual_pk_coef, error) ||
        actual_pk_coef != expected_pk_coef) {
        if (error && error->empty()) *error = "BLOCKED: pk_coef.bin SHA-256 does not match authenticated SRS manifest";
        return false;
    }
    const std::filesystem::path registry =
        std::filesystem::absolute(witness.registry_path).lexically_normal();
    const auto registry_relative = registry.lexically_relative(
        std::filesystem::absolute(config.baseline_root).lexically_normal());
    if (registry_relative.empty() || registry_relative.string().find("..") == 0U ||
        std::filesystem::is_symlink(registry) || !std::filesystem::is_regular_file(registry)) {
        if (error) *error = "BLOCKED: authenticated verifier registry is missing: " + registry.string();
        return false;
    }
    std::array<zkdl::byte, 32> expected_registry{};
    if (!parse_digest(config.verifier_registry_digest, expected_registry)) {
        if (error) *error = "BLOCKED: verifier_registry_digest is not a canonical SHA-256";
        return false;
    }
    std::array<zkdl::byte, 32> registry_digest{};
    if (!file_digest(registry, registry_digest, error)) return false;
    if (registry_digest != expected_registry || registry_digest != witness.registry_digest) {
        if (error) *error = "BLOCKED: verifier registry digest does not match secure config and witness";
        return false;
    }
    // The restored repository has no canonical compressed-G1 decoder or
    // manifest parser for the SRS byte layout.  Never treat a caller-supplied
    // CpuG1Point vector as an authenticated substitute: the native path stays
    // blocked until the original project publishes this exact loader.
    if (error) *error = "BLOCKED: authenticated SRS compressed-G1/Fr loader is not defined by the restored zkDL tree";
    return false;
}

bool statement_from_adapter(const Config& config, const Graph& graph,
                            const ModelBundle& model, const StaticTables& tables,
                            const RelationPlan& plan, const PublicStatement& statement,
                            const NativeRelationWitness& witness,
                            relation::RelationStatement& out, std::string* error) {
    if (statement.proof_scope != kProofScope || plan.proof_scope != kProofScope) {
        if (error) *error = "relation statement/proof scope mismatch";
        return false;
    }
    if (!tables.authenticated || !plan.integer_bound_ledger.authenticated ||
        statement.static_table_digest != tables.digest ||
        statement.static_table_descriptor_digest != tables.descriptor_digest ||
        statement.scale_chain_digest != config.scale_chain_digest ||
        statement.integer_bound_ledger_digest != plan.integer_bound_ledger.digest ||
        statement.integer_bound_ledger_descriptor_digest != plan.integer_bound_ledger.descriptor_digest ||
        statement.relation_registry_digest != plan.registry_digest ||
        statement.relation_terminal_claims_digest != plan.terminal_claims_digest ||
        statement.relation_mapping_digest != plan.mapping_digest ||
        statement.relation_selector_digest != plan.selector_digest ||
        statement.relation_degree_digest != plan.degree_digest ||
        statement.verifier_registry_digest != config.verifier_registry_digest) {
        if (error) *error = "adapter statement/resource identity is not bound to authenticated tables/ledger/plan";
        return false;
    }
    if (!parse_digest(statement.statement_id, out.statement_digest) ||
        !parse_digest(model.manifest_digest, out.model_digest) ||
        !parse_digest(tables.digest, out.table_digest) ||
        !parse_digest(plan.integer_bound_ledger.digest, out.ledger_digest) ||
        !parse_digest(config.verifier_registry_digest, out.registry_digest)) {
        if (error) *error = "adapter statement/resource identity is not canonical SHA-256";
        return false;
    }
    if (witness.srs_digest == std::array<zkdl::byte, 32>{} ||
        witness.registry_digest != out.registry_digest) {
        if (error) *error = "native witness SRS/registry identity is missing or mismatched";
        return false;
    }
    out.srs_digest = witness.srs_digest;
    const auto public_it = std::find_if(witness.tensors.begin(), witness.tensors.end(),
                                        [](const relation::TensorWitness& tensor) {
                                            return tensor.id == "public_Y" && tensor.visibility == 1U;
                                        });
    if (public_it == witness.tensors.end() || !public_it->valid(error)) {
        if (error && error->empty()) *error = "canonical public_Y tensor witness is missing";
        return false;
    }
    out.public_output = relation::encode_public_field_vector(public_it->values);
    (void)graph;
    return true;
}

void append_unique(std::vector<std::string>& refs, const std::string& value) {
    if (std::find(refs.begin(), refs.end(), value) == refs.end()) refs.push_back(value);
}

std::uint32_t local_ref(const relation::OperationDescriptor& operation,
                        const std::string& ref) {
    const auto found = std::find(operation.tensor_refs.begin(), operation.tensor_refs.end(), ref);
    if (found == operation.tensor_refs.end())
        throw std::invalid_argument("relation descriptor omits referenced tensor: " + ref);
    return static_cast<std::uint32_t>(std::distance(operation.tensor_refs.begin(), found));
}

relation::Monomial monomial(const FrWire& coefficient,
                            std::initializer_list<std::uint32_t> refs) {
    relation::Monomial term;
    term.coefficient = coefficient;
    term.refs.assign(refs.begin(), refs.end());
    return term;
}

relation::ConstraintDescriptor linear_constraint(const std::string& id,
                                                const relation::OperationDescriptor& operation,
                                                const std::vector<std::string>& inputs,
                                                const std::string& output) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 1U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(one, {local_ref(operation, output)}));
    for (const auto& input : inputs)
        constraint.terms.push_back(monomial(zkdl::fr_neg(one), {local_ref(operation, input)}));
    return constraint;
}

relation::ConstraintDescriptor product_constraint(const std::string& id,
                                                 const relation::OperationDescriptor& operation,
                                                 const std::string& left,
                                                 const std::string& right,
                                                 const std::string& output) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(one, {local_ref(operation, output)}));
    constraint.terms.push_back(monomial(zkdl::fr_neg(one), {local_ref(operation, left),
                                                             local_ref(operation, right)}));
    return constraint;
}

std::string view_or_ref(const OperationFamily& plan_operation,
                        const std::map<std::string, std::uint32_t>& domains,
                        const std::string& role, const std::string& fallback) {
    const std::string candidate = plan_operation.id + "/view/" + role;
    return domains.count(candidate) != 0U ? candidate : fallback;
}

void add_ref(relation::OperationDescriptor& operation, const std::string& ref) {
    if (std::find(operation.tensor_refs.begin(), operation.tensor_refs.end(), ref) == operation.tensor_refs.end())
        operation.tensor_refs.push_back(ref);
}

std::vector<std::string> aligned_inputs(
    const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains) {
    std::vector<std::string> refs;
    refs.reserve(plan_operation.inputs.size());
    for (size_t i = 0; i < plan_operation.inputs.size(); ++i)
        refs.push_back(view_or_ref(plan_operation, domains, "input" + std::to_string(i),
                                   plan_operation.inputs[i]));
    return refs;
}

std::string aligned_output(const OperationFamily& plan_operation,
                           const std::map<std::string, std::uint32_t>& domains,
                           size_t index = 0U) {
    if (index >= plan_operation.outputs.size())
        throw std::invalid_argument("relation operation output index is out of range");
    return view_or_ref(plan_operation, domains, "output" + std::to_string(index),
                       plan_operation.outputs[index]);
}

void set_tensor_refs(relation::OperationDescriptor& operation,
                     const std::vector<std::string>& refs) {
    operation.tensor_refs.clear();
    for (const auto& ref : refs) add_ref(operation, ref);
}

relation::ConstraintDescriptor boolean_constraint(const std::string& id,
                                                 const relation::OperationDescriptor& operation,
                                                 const std::string& ref) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const std::uint32_t index = local_ref(operation, ref);
    constraint.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {index}));
    constraint.terms.push_back(monomial(zkdl::fr_from_u32(1U), {index, index}));
    return constraint;
}

relation::ConstraintDescriptor reconstruction_constraint(
    const std::string& id, const relation::OperationDescriptor& operation,
    const std::string& value, const std::vector<std::string>& bits,
    std::uint32_t bit_width, std::int64_t offset = 0) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 1U;
    const FrWire one = zkdl::fr_from_u32(1U);
    const std::uint32_t value_index = local_ref(operation, value);
    constraint.terms.push_back(monomial(zkdl::fr_neg(one), {value_index}));
    if (offset != 0) constraint.constant = relation::fr_from_signed_i64(offset);
    for (std::uint32_t i = 0U; i < bit_width && i < bits.size(); ++i) {
        FrWire coefficient = zkdl::fr_from_u32(i < 31U ? (1U << i) : 0U);
        if (i >= 31U && i < 63U) {
            coefficient = relation::fr_from_signed_i64(static_cast<std::int64_t>(1ULL << i));
        }
        constraint.terms.push_back(monomial(coefficient, {local_ref(operation, bits[i])}));
    }
    return constraint;
}

relation::ConstraintDescriptor signed_round_linear(
    const std::string& id, const relation::OperationDescriptor& operation,
    const std::string& a, const std::string& d, const std::string& q,
    const std::string& r, const std::string& delta) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(2), {local_ref(operation, a)}));
    constraint.terms.push_back(monomial(one, {local_ref(operation, d)}));
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(-2), {local_ref(operation, d), local_ref(operation, q)}));
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(-1), {local_ref(operation, r)}));
    (void)delta;
    return constraint;
}

std::uint32_t ceil_log2(std::size_t value) {
    if (value <= 1U) return 0U;
    std::uint32_t bits = 0U;
    size_t domain = 1U;
    while (domain < value) {
        domain <<= 1U;
        ++bits;
    }
    return bits;
}

const std::vector<std::int64_t>& lookup_table(const StaticTables& tables,
                                              const OperationFamily& operation,
                                              std::int64_t& minimum,
                                              std::string* error) {
    if (operation.id.find("/leaky_relu") != std::string::npos) {
        minimum = tables.lrelu_min;
        return tables.lrelu;
    }
    if (operation.id.find("/exp_map") != std::string::npos) {
        minimum = 0;
        return tables.exp;
    }
    if (operation.id.find("/elu") != std::string::npos) {
        minimum = tables.elu_min;
        return tables.elu;
    }
    if (error) *error = "BLOCKED: no authenticated static table is mapped to " + operation.id;
    static const std::vector<std::int64_t> empty;
    return empty;
}

relation::OperationDescriptor table_operation(
    const Config& config, const StaticTables& tables,
    const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    std::string* error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = plan_operation.id.find("/leaky_relu") != std::string::npos
        ? relation::RelationKind::LeakyRelu
        : (plan_operation.id.find("/exp_map") != std::string::npos
           ? relation::RelationKind::Exp : relation::RelationKind::Elu);
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;masked-from-relation";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    for (const auto& input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.source_refs, output);
    std::int64_t minimum = 0;
    const std::vector<std::int64_t>& selected = lookup_table(tables, plan_operation, minimum, error);
    if (selected.empty()) return relation::OperationDescriptor{};
    if ((selected.size() & (selected.size() - 1U)) != 0U) {
        if (error) *error = "BLOCKED: authenticated table size is not a power of two for " + operation.id;
        return relation::OperationDescriptor{};
    }
    const std::uint32_t depth = ceil_log2(selected.size());
    const std::string query = view_or_ref(plan_operation, domains, "query", plan_operation.inputs.front());
    const std::string index = view_or_ref(plan_operation, domains, "index", plan_operation.inputs.size() > 2U ? plan_operation.inputs[2] : plan_operation.inputs.front());
    const std::string output = aligned_output(plan_operation, domains);
    std::vector<std::string> bits;
    bits.reserve(depth);
    for (std::uint32_t bit = 0U; bit < depth; ++bit) {
        const std::string ref = view_or_ref(plan_operation, domains, "bit/" + std::to_string(bit),
                                            plan_operation.id + "/view/bit/" + std::to_string(bit));
        if (!domains.count(ref)) {
            if (error) *error = "BLOCKED: table lookup is missing canonical bit view " + ref;
            return relation::OperationDescriptor{};
        }
        bits.push_back(ref);
    }
    add_ref(operation, query);
    add_ref(operation, index);
    for (const auto& bit : bits) add_ref(operation, bit);
    std::vector<std::vector<std::string>> chi(depth + 1U);
    for (std::uint32_t level = 0U; level <= depth; ++level) {
        const std::uint32_t width = 1U << level;
        chi[level].reserve(width);
        for (std::uint32_t node = 0U; node < width; ++node) {
            const std::string ref = plan_operation.id + "/view/chi/" + std::to_string(level) + "/" + std::to_string(node);
            if (!domains.count(ref)) {
                if (error) *error = "BLOCKED: table lookup is missing canonical chi-tree view " + ref;
                return relation::OperationDescriptor{};
            }
            chi[level].push_back(ref);
            add_ref(operation, ref);
        }
    }
    add_ref(operation, output);
    // The index is not an adapter-side hint: it is the field representation
    // of the operation query and must be bound before the bit decomposition
    // selects a static-table row.
    operation.constraints.push_back(linear_constraint(
        operation.id + "/query-index", operation, {query}, index));
    // Binary index reconstruction and bit booleanity.
    relation::ConstraintDescriptor index_constraint = reconstruction_constraint(
        operation.id + "/index", operation, index, bits, depth, minimum);
    operation.constraints.push_back(std::move(index_constraint));
    for (std::uint32_t bit = 0U; bit < depth; ++bit)
        operation.constraints.push_back(boolean_constraint(operation.id + "/bit/" + std::to_string(bit), operation, bits[bit]));
    relation::ConstraintDescriptor root;
    root.id = operation.id + "/chi-root";
    root.degree = 1U;
    root.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
    root.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, chi[0][0])}));
    operation.constraints.push_back(std::move(root));
    for (std::uint32_t level = 0U; level < depth; ++level) {
        for (std::uint32_t node = 0U; node < (1U << level); ++node) {
            const std::string& parent = chi[level][node];
            const std::string& zero = chi[level + 1U][2U * node];
            const std::string& one = chi[level + 1U][2U * node + 1U];
            const std::string& bit = bits[level];
            relation::ConstraintDescriptor zero_constraint;
            zero_constraint.id = operation.id + "/chi/" + std::to_string(level) + "/" + std::to_string(node) + "/zero";
            zero_constraint.degree = 2U;
            zero_constraint.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, zero)}));
            zero_constraint.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, parent)}));
            zero_constraint.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, parent), local_ref(operation, bit)}));
            operation.constraints.push_back(std::move(zero_constraint));
            relation::ConstraintDescriptor one_constraint;
            one_constraint.id = operation.id + "/chi/" + std::to_string(level) + "/" + std::to_string(node) + "/one";
            one_constraint.degree = 2U;
            one_constraint.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, one)}));
            one_constraint.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, parent), local_ref(operation, bit)}));
            operation.constraints.push_back(std::move(one_constraint));
        }
    }
    relation::ConstraintDescriptor output_constraint;
    output_constraint.id = operation.id + "/table-output";
    output_constraint.degree = 1U;
    output_constraint.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, output)}));
    for (std::uint32_t i = 0U; i < selected.size(); ++i)
        output_constraint.terms.push_back(monomial(relation::fr_from_signed_i64(selected[i]), {local_ref(operation, chi[depth][i])}));
    operation.constraints.push_back(std::move(output_constraint));
    std::string validation_error;
    if (!operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: table relation descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    (void)config;
    return operation;
}

std::string required_view(const OperationFamily& plan_operation,
                          const std::map<std::string, std::uint32_t>& domains,
                          const std::string& role, const std::string& fallback,
                          std::string* error) {
    const std::string candidate = plan_operation.id + "/view/" + role;
    if (domains.count(candidate) != 0U) return candidate;
    if (!fallback.empty() && domains.count(fallback) != 0U) return fallback;
    if (error) *error = "BLOCKED: relation " + plan_operation.id +
                        " is missing canonical view " + candidate;
    return std::string();
}

relation::OperationDescriptor signed_round_operation(
    const Config& config, const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    std::string* error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = relation::RelationKind::SignedRoundDiv;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;masked-from-relation";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    for (const auto& input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.source_refs, output);
    const std::string numerator = required_view(plan_operation, domains, "signed_numerator",
                                                plan_operation.inputs.empty() ? "" : plan_operation.inputs.front(), error);
    const std::string divisor = required_view(plan_operation, domains, "divisor",
                                              plan_operation.inputs.size() < 2U ? "" : plan_operation.inputs[1], error);
    const std::string signed_q = required_view(plan_operation, domains, "signed_quotient",
                                               plan_operation.outputs.empty() ? "" : plan_operation.outputs.front(), error);
    const std::string a = required_view(plan_operation, domains, "a", "", error);
    const std::string q = required_view(plan_operation, domains, "q_abs", "", error);
    const std::string r = required_view(plan_operation, domains, "r", "", error);
    const std::string delta = required_view(plan_operation, domains, "delta", "", error);
    const std::string sigma = required_view(plan_operation, domains, "sigma", "", error);
    const std::string zero = required_view(plan_operation, domains, "z", "", error);
    const std::string a_inverse = required_view(plan_operation, domains, "a_inverse", "", error);
    const std::string divisor_inverse = required_view(plan_operation, domains, "divisor_inverse", "", error);
    for (const auto& ref : {numerator, divisor, signed_q, a, q, r, delta, sigma, zero,
                            a_inverse, divisor_inverse}) {
        if (ref.empty()) return relation::OperationDescriptor{};
        add_ref(operation, ref);
    }
    const std::vector<std::string> range_values = {a, q, r, delta, divisor};
    std::map<std::string, std::vector<std::string>> range_bits;
    const std::uint32_t width = static_cast<std::uint32_t>(config.range_bits);
    if (width == 0U || width >= 31U) {
        if (error) *error = "BLOCKED: SignedRoundDiv range_bits is not a bounded native width";
        return relation::OperationDescriptor{};
    }
    for (const auto& value : range_values) {
        for (std::uint32_t bit = 0U; bit < width; ++bit) {
            const std::string ref = required_view(plan_operation, domains,
                                                  "range/" + value.substr(value.rfind('/') + 1U) + "/" + std::to_string(bit),
                                                  "", error);
            if (ref.empty()) return relation::OperationDescriptor{};
            range_bits[value].push_back(ref);
            add_ref(operation, ref);
        }
    }
    relation::ConstraintDescriptor signed_n;
    signed_n.id = operation.id + "/signed-numerator";
    signed_n.degree = 2U;
    signed_n.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, numerator)}));
    signed_n.terms.push_back(monomial(relation::fr_from_signed_i64(-1), {local_ref(operation, a)}));
    signed_n.terms.push_back(monomial(relation::fr_from_signed_i64(2), {local_ref(operation, sigma), local_ref(operation, a)}));
    operation.constraints.push_back(std::move(signed_n));
    operation.constraints.push_back(signed_round_linear(operation.id + "/nearest-ties-away",
                                                        operation, a, divisor, q, r, delta));
    relation::ConstraintDescriptor slack;
    slack.id = operation.id + "/slack";
    slack.degree = 1U;
    slack.constant = zkdl::fr_from_u32(1U);
    slack.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, r)}));
    slack.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, delta)}));
    slack.terms.push_back(monomial(relation::fr_from_signed_i64(-2), {local_ref(operation, divisor)}));
    operation.constraints.push_back(std::move(slack));
    relation::ConstraintDescriptor a_inv;
    a_inv.id = operation.id + "/a-inverse";
    a_inv.degree = 2U;
    a_inv.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
    a_inv.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, a), local_ref(operation, a_inverse)}));
    a_inv.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, zero)}));
    operation.constraints.push_back(std::move(a_inv));
    relation::ConstraintDescriptor d_inv;
    d_inv.id = operation.id + "/divisor-inverse";
    d_inv.degree = 2U;
    d_inv.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
    d_inv.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, divisor), local_ref(operation, divisor_inverse)}));
    operation.constraints.push_back(std::move(d_inv));
    relation::ConstraintDescriptor q_sign;
    q_sign.id = operation.id + "/signed-quotient";
    q_sign.degree = 2U;
    q_sign.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, signed_q)}));
    q_sign.terms.push_back(monomial(relation::fr_from_signed_i64(-1), {local_ref(operation, q)}));
    q_sign.terms.push_back(monomial(relation::fr_from_signed_i64(2), {local_ref(operation, sigma), local_ref(operation, q)}));
    operation.constraints.push_back(std::move(q_sign));
    operation.constraints.push_back(boolean_constraint(operation.id + "/sigma-boolean", operation, sigma));
    operation.constraints.push_back(boolean_constraint(operation.id + "/zero-boolean", operation, zero));
    relation::ConstraintDescriptor zero_sign;
    zero_sign.id = operation.id + "/zero-sign";
    zero_sign.degree = 2U;
    zero_sign.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, sigma), local_ref(operation, zero)}));
    operation.constraints.push_back(std::move(zero_sign));
    for (const auto& value : range_values) {
        operation.constraints.push_back(reconstruction_constraint(
            operation.id + "/range/" + value.substr(value.rfind('/') + 1U),
            operation, value, range_bits[value], width));
        for (std::uint32_t bit = 0U; bit < width; ++bit)
            operation.constraints.push_back(boolean_constraint(
                operation.id + "/range-bit/" + value.substr(value.rfind('/') + 1U) + "/" + std::to_string(bit),
                operation, range_bits[value][bit]));
    }
    std::string validation_error;
    if (!operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: SignedRoundDiv descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor canonical_operation(
    const Config& config, const StaticTables& tables,
    const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    std::string* error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.degree = static_cast<std::uint32_t>(plan_operation.degree);
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;masked-from-relation";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    for (const auto& input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.source_refs, output);
    for (const auto& input : plan_operation.inputs) append_unique(operation.tensor_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.tensor_refs, output);
    for (const auto& auxiliary : plan_operation.auxiliary) operation.selector_refs.push_back(auxiliary);

    const auto unsupported = [&](const std::string& reason) {
        if (error) *error = "BLOCKED: relation family " + plan_operation.id + " requires " + reason;
        return relation::OperationDescriptor{};
    };
    try {
        if (plan_operation.kind == "field_addition" ||
            plan_operation.kind == "tensor_sumcheck" ||
            plan_operation.kind == "tensor_reindex_sumcheck" ||
            plan_operation.kind == "output_bias") {
            operation.kind = plan_operation.kind == "output_bias"
                ? relation::RelationKind::Bias
                : (plan_operation.kind == "tensor_reindex_sumcheck"
                   ? relation::RelationKind::Reindex
                   : (plan_operation.kind == "tensor_sumcheck"
                      ? relation::RelationKind::HeadSum : relation::RelationKind::Add));
            if (plan_operation.outputs.size() != 1U || plan_operation.inputs.empty())
                return unsupported("one output and aligned input MLE tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            std::vector<std::string> refs = inputs;
            refs.push_back(output);
            set_tensor_refs(operation, refs);
            operation.constraints.push_back(linear_constraint(
                operation.id + "/linear", operation, inputs, output));
        } else if (plan_operation.kind == "tensor_product_sumcheck" &&
                   (plan_operation.id.find("/consistency") != std::string::npos ||
                    plan_operation.inputs.size() >= 3U)) {
            operation.kind = relation::RelationKind::FieldIntegerConsistency;
            if (plan_operation.inputs.size() < 3U || plan_operation.outputs.size() != 1U)
                return unsupported("denominator/field/integer aggregate and residual field tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            set_tensor_refs(operation, {inputs[0], inputs[1], inputs[2], output});
            relation::ConstraintDescriptor consistency;
            consistency.id = operation.id + "/field-integer";
            consistency.degree = 2U;
            consistency.terms.push_back(monomial(
                zkdl::fr_from_u32(1U), {local_ref(operation, inputs[0]),
                                        local_ref(operation, inputs[1])}));
            consistency.terms.push_back(monomial(
                relation::fr_from_signed_i64(-1), {local_ref(operation, inputs[2])}));
            consistency.terms.push_back(monomial(
                relation::fr_from_signed_i64(-1), {local_ref(operation, output)}));
            operation.constraints.push_back(std::move(consistency));
        } else if (plan_operation.kind == "tensor_product_sumcheck") {
            operation.kind = relation::RelationKind::Product;
            if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() != 1U)
                return unsupported("two aligned input tensors and one output tensor");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            set_tensor_refs(operation, {inputs[0], inputs[1], output});
            operation.constraints.push_back(product_constraint(
                operation.id + "/product", operation, inputs[0], inputs[1], output));
        } else if (plan_operation.kind == "field_inverse_product_sumcheck") {
            operation.kind = relation::RelationKind::Alpha;
            if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() < 2U)
                return unsupported("denominator/exp/inverse/alpha aligned field tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string inverse_view = aligned_output(plan_operation, domains, 0U);
            const std::string alpha_view = aligned_output(plan_operation, domains, 1U);
            set_tensor_refs(operation, {inputs[0], inputs[1], inverse_view, alpha_view});
            relation::ConstraintDescriptor inverse;
            inverse.id = operation.id + "/inverse";
            inverse.degree = 2U;
            inverse.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
            inverse.terms.push_back(monomial(
                zkdl::fr_from_u32(1U), {local_ref(operation, inputs[0]),
                                        local_ref(operation, inverse_view)}));
            operation.constraints.push_back(std::move(inverse));
            operation.constraints.push_back(product_constraint(
                operation.id + "/alpha", operation, inputs[1], inverse_view,
                alpha_view));
        } else if (plan_operation.kind == "public_selector_reindex") {
            operation.kind = relation::RelationKind::SparseRoute;
            if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() != 1U)
                return unsupported("canonical route selector and aligned reindexed witness");
            // The public route tensor is not a pointwise field value.  The
            // native runner must provide the already reindexed input view;
            // this copy equation then binds that view to the output.
            const std::string source_view = view_or_ref(plan_operation, domains, "route_source",
                                                        plan_operation.inputs.back());
            const std::string output_view = aligned_output(plan_operation, domains);
            set_tensor_refs(operation, {source_view, output_view});
            operation.constraints.push_back(linear_constraint(
                operation.id + "/route", operation, {source_view}, output_view));
        } else if (plan_operation.kind == "hyrax_terminal_evaluation") {
            operation.kind = relation::RelationKind::PublicOutput;
            if (plan_operation.inputs.size() != 1U || plan_operation.outputs.size() != 1U ||
                plan_operation.inputs.front() != plan_operation.outputs.front())
                return unsupported("a single public_Y tensor");
            relation::ConstraintDescriptor constraint;
            constraint.id = operation.id + "/terminal";
            constraint.degree = 1U;
            operation.constraints.push_back(std::move(constraint));
        } else if (plan_operation.kind == "indexed_public_table_sumcheck") {
            return table_operation(config, tables, plan_operation, domains, error);
        } else if (plan_operation.kind == "segmented_sumcheck") {
            return unsupported("canonical segmented MLE/reindex witness views");
        } else if (plan_operation.kind == "signed_round_div_sumcheck") {
            return signed_round_operation(config, plan_operation, domains, error);
        } else if (plan_operation.kind == "tensor_matmul_sumcheck" ||
                   plan_operation.kind == "inner_product_sumcheck") {
            return unsupported("zkMatMul/inner-product native shaped witness adapter");
        } else {
            return unsupported("an authenticated operation-family mapping");
        }
    } catch (const std::exception& exception) {
        if (error) *error = "BLOCKED: relation family " + plan_operation.id + " descriptor construction failed: " + exception.what();
        return relation::OperationDescriptor{};
    }
    std::string validation_error;
    if (!operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: canonical relation descriptor rejected for " + plan_operation.id + ": " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

bool build_canonical_operations(const Config& config, const StaticTables& tables,
                                const RelationPlan& relation,
                                const std::vector<relation::TensorWitness>& tensors,
                                std::vector<relation::OperationDescriptor>& operations,
                                std::string* error) {
    std::map<std::string, std::uint32_t> domains;
    for (const auto& tensor : tensors) domains.emplace(tensor.id, tensor.domain_size);
    operations.clear();
    std::set<std::string> ids;
    for (const auto& plan_operation : relation.operation_families) {
        relation::OperationDescriptor operation = canonical_operation(config, tables, plan_operation, domains, error);
        if (operation.id.empty()) return false;
        for (const auto& ref : operation.tensor_refs) {
            const auto found = domains.find(ref);
            if (found == domains.end()) {
                if (error) *error = "BLOCKED: native relation witness is missing tensor " + ref +
                                    " required by " + operation.id;
                return false;
            }
        }
        if (operation.tensor_refs.empty()) return false;
        const std::uint32_t domain = domains.at(operation.tensor_refs.front());
        for (const auto& ref : operation.tensor_refs)
            if (domains.at(ref) != domain) {
                if (error) *error = "BLOCKED: operation " + operation.id +
                                    " has non-aligned MLE domains; explicit FAC4DNN reindex witness is required";
                return false;
            }
        if (!ids.insert(operation.id).second) {
            if (error) *error = "duplicate canonical native operation id: " + operation.id;
            return false;
        }
        operations.push_back(std::move(operation));
    }
    return !operations.empty();
}

bool operation_set_matches_plan(const RelationPlan& plan,
                                const std::vector<relation::OperationDescriptor>& operations,
                                std::string* error) {
    std::map<std::string, const relation::OperationDescriptor*> supplied;
    for (const auto& operation : operations) {
        if (!operation.valid(error) || !supplied.emplace(operation.id, &operation).second) {
            if (error && error->empty()) *error = "duplicate/invalid stage-qualified native relation descriptor";
            return false;
        }
    }
    if (supplied.size() != plan.operation_families.size()) {
        if (error) *error = "BLOCKED: native witness does not carry one descriptor for every RelationPlan operation";
        return false;
    }
    for (const auto& expected : plan.operation_families) {
        const auto found = supplied.find(expected.id);
        if (found == supplied.end()) {
            if (error) *error = "BLOCKED: native relation descriptor is missing: " + expected.id;
            return false;
        }
        const auto& actual = *found->second;
        const bool low_degree_tree = expected.kind.find("indexed_public_table") != std::string::npos ||
                                     expected.kind.find("signed_round_div") != std::string::npos ||
                                     expected.kind.find("segmented_") != std::string::npos;
        if ((!low_degree_tree && actual.degree != expected.degree) ||
            (low_degree_tree && (actual.degree < expected.degree || actual.degree > 2U)) ||
            actual.mapping != expected.mapping ||
            actual.terminal_point != expected.terminal_point ||
            actual.terminal_claim != expected.terminal_claim ||
            actual.requires_reindex != expected.requires_reindex ||
            actual.public_route != expected.public_route) {
            if (error) *error = "native relation descriptor does not match authenticated plan: " + expected.id;
            return false;
        }
        std::set<std::string> refs(actual.source_refs.begin(), actual.source_refs.end());
        for (const auto& ref : expected.inputs)
            if (!refs.count(ref)) {
                if (error) *error = "native relation descriptor omits plan input: " + expected.id + " -> " + ref;
                return false;
            }
        for (const auto& ref : expected.outputs)
            if (!refs.count(ref)) {
                if (error) *error = "native relation descriptor omits plan output: " + expected.id + " -> " + ref;
                return false;
            }
        std::set<std::string> selectors(actual.selector_refs.begin(), actual.selector_refs.end());
        for (const auto& selector : expected.auxiliary)
            if (!selectors.count(selector)) {
                if (error) *error = "native relation descriptor omits plan selector: " + expected.id + " -> " + selector;
                return false;
            }
    }
    return true;
}

bool context_from_request(const NativeProveRequest& request,
                          relation::ProverContext& context, std::string* error) {
    if (!request.config || !request.graph || !request.model || !request.tables ||
        !request.relation || !request.trace || !request.statement || !request.witness) {
        if (error) *error = "BLOCKED: native prove request lacks canonical tensors/witness/selectors/public Y";
        return false;
    }
    if (!resource_admission(*request.config, *request.witness, error)) return false;
    if (!statement_from_adapter(*request.config, *request.graph, *request.model,
                                *request.tables, *request.relation, *request.statement,
                                *request.witness, context.statement, error)) return false;
    context.tensors = request.witness->tensors;
    if (!build_canonical_operations(*request.config, *request.tables, *request.relation,
                                    context.tensors,
                                    context.operations, error)) return false;
    // If the adapter supplied descriptors, treat them only as a consistency
    // witness.  Equations used by the proof are reconstructed above from the
    // authenticated plan and never taken from this optional vector.
    if (!request.witness->operations.empty()) {
        if (!operation_set_matches_plan(*request.relation, request.witness->operations, error)) return false;
        std::map<std::string, const relation::OperationDescriptor*> supplied;
        for (const auto& operation : request.witness->operations) supplied.emplace(operation.id, &operation);
        for (const auto& operation : context.operations) {
            const auto found = supplied.find(operation.id);
            std::vector<zkdl::byte> expected_bytes, actual_bytes;
            relation::append_operation(expected_bytes, operation);
            relation::append_operation(actual_bytes, *found->second);
            if (expected_bytes != actual_bytes) {
                if (error) *error = "native supplied relation descriptor differs from plan-derived descriptor: " + operation.id;
                return false;
            }
        }
    }
    context.generators = request.witness->generators;
    std::string shape_error;
    for (const auto& tensor : context.tensors)
        if (!tensor.valid(&shape_error)) {
            if (error) *error = "native relation tensor rejected: " + shape_error;
            return false;
        }
    return true;
}

bool context_from_verify(const NativeVerifyRequest& request,
                         relation::VerifierContext& context, std::string* error) {
    if (!request.config || !request.statement || !request.public_context ||
        !request.relation || !request.tables) {
        if (error) *error = "BLOCKED: independent verifier lacks public RelationPlan/context/SRS/registry";
        return false;
    }
    const auto& supplied = *request.public_context;
    if (!supplied.srs_authenticated || !supplied.registry_authenticated || supplied.generators.empty()) {
        if (error) *error = "BLOCKED: independent verifier context has no authenticated SRS/registry";
        return false;
    }
    if (request.statement->proof_scope != kProofScope || request.relation->proof_scope != kProofScope) {
        if (error) *error = "independent verifier proof scope is not full_gat_forward";
        return false;
    }
    if (!request.tables->authenticated || !request.relation->integer_bound_ledger.authenticated ||
        request.statement->static_table_digest != request.tables->digest ||
        request.statement->static_table_descriptor_digest != request.tables->descriptor_digest ||
        request.statement->scale_chain_digest != request.config->scale_chain_digest ||
        request.statement->integer_bound_ledger_digest != request.relation->integer_bound_ledger_digest ||
        request.statement->integer_bound_ledger_descriptor_digest !=
            request.relation->integer_bound_ledger_descriptor_digest ||
        request.statement->relation_registry_digest != request.relation->registry_digest ||
        request.statement->relation_terminal_claims_digest != request.relation->terminal_claims_digest ||
        request.statement->relation_mapping_digest != request.relation->mapping_digest ||
        request.statement->relation_selector_digest != request.relation->selector_digest ||
        request.statement->relation_degree_digest != request.relation->degree_digest ||
        request.statement->verifier_registry_digest != request.config->verifier_registry_digest) {
        if (error) *error = "independent verifier statement/resource identity is not bound to the authenticated plan";
        return false;
    }
    // This performs the same path, manifest, SHA-256, and registry admission
    // as proving. It intentionally ends in an explicit BLOCKED result until
    // the restored tree provides its canonical compressed-G1/Fr loader.
    if (!resource_admission(*request.config, supplied, error)) return false;
    for (const auto& tensor : supplied.tensors)
        if (tensor.visibility != 1U) {
            if (error) *error = "independent verifier context contains private witness tensor";
            return false;
        }
    std::vector<relation::TensorWitness> shape_tensors;
    shape_tensors.reserve(supplied.tensor_shapes.size());
    for (const auto& shape : supplied.tensor_shapes) {
        relation::TensorWitness tensor;
        tensor.id = shape.id;
        tensor.domain_size = shape.domain_size;
        tensor.visibility = shape.visibility;
        shape_tensors.push_back(std::move(tensor));
    }
    // A public context may carry public values for Y, but all other tensors
    // are shape-only.  The canonical plan builder uses only ids/domains here;
    // it never asks the independent verifier for private values.
    for (const auto& tensor : supplied.tensors) {
        if (std::find_if(shape_tensors.begin(), shape_tensors.end(),
                         [&](const relation::TensorWitness& shape) { return shape.id == tensor.id; }) == shape_tensors.end()) {
            relation::TensorWitness shape = tensor;
            shape.values.clear();
            shape_tensors.push_back(std::move(shape));
        }
    }
    const std::filesystem::path registry(supplied.registry_path);
    if (!std::filesystem::is_regular_file(registry)) {
        if (error) *error = "BLOCKED: independent verifier registry is missing: " + registry.string();
        return false;
    }
    std::array<zkdl::byte, 32> expected_registry{};
    if (!parse_digest(request.config->verifier_registry_digest, expected_registry) ||
        supplied.registry_digest != expected_registry) {
        if (error) *error = "independent verifier registry identity mismatch";
        return false;
    }
    if (!parse_digest(request.statement->statement_id, context.statement.statement_digest) ||
        !parse_digest(request.statement->checkpoint_manifest_digest, context.statement.model_digest) ||
        !parse_digest(request.statement->static_table_digest, context.statement.table_digest) ||
        !parse_digest(request.statement->integer_bound_ledger_digest, context.statement.ledger_digest) ||
        !parse_digest(request.statement->verifier_registry_digest, context.statement.registry_digest)) {
        if (error) *error = "independent verifier statement identity is not canonical SHA-256";
        return false;
    }
    context.statement.srs_digest = supplied.srs_digest;
    const auto public_y = std::find_if(request.public_context->tensors.begin(),
                                       request.public_context->tensors.end(),
                                       [](const relation::TensorWitness& tensor) {
                                           return tensor.id == "public_Y" && tensor.visibility == 1U;
                                       });
    context.statement.public_output = public_y == request.public_context->tensors.end()
        ? std::vector<zkdl::byte>()
        : relation::encode_public_field_vector(public_y->values);
    if (context.statement.public_output.empty()) {
        if (error) *error = "BLOCKED: independent verifier lacks public_Y field witness";
        return false;
    }
    if (!build_canonical_operations(*request.config, *request.tables, *request.relation,
                                    shape_tensors,
                                    context.operations, error)) return false;
    if (!supplied.operations.empty()) {
        if (!operation_set_matches_plan(*request.relation, supplied.operations, error)) return false;
        std::map<std::string, const relation::OperationDescriptor*> supplied_operations;
        for (const auto& operation : supplied.operations) supplied_operations.emplace(operation.id, &operation);
        for (const auto& operation : context.operations) {
            const auto found = supplied_operations.find(operation.id);
            std::vector<zkdl::byte> expected_bytes, actual_bytes;
            relation::append_operation(expected_bytes, operation);
            relation::append_operation(actual_bytes, *found->second);
            if (expected_bytes != actual_bytes) {
                if (error) *error = "independent verifier descriptor differs from RelationPlan: " + operation.id;
                return false;
            }
        }
    }
    context.generators = supplied.generators;
    context.tensor_ids.clear();
    context.tensor_ids.reserve(shape_tensors.size());
    context.tensor_visibility.clear();
    for (const auto& tensor : shape_tensors) {
        if (tensor.id.empty()) {
            if (error) *error = "independent verifier context contains an empty tensor id";
            return false;
        }
        if (context.tensor_visibility.emplace(tensor.id, tensor.visibility).second) {
            context.tensor_ids.push_back(tensor.id);
        } else if (context.tensor_visibility.at(tensor.id) != tensor.visibility) {
            if (error) *error = "independent verifier context has duplicate tensor ids with different visibility";
            return false;
        }
    }
    return relation::descriptor_vector_valid(context.operations, error);
}

}  // namespace

bool zkdl_gat_relation_prove(const relation::ProverContext& context,
                             relation::Proof* proof, std::string* error) {
    if (!proof) {
        if (error) *error = "relation proof output pointer is null";
        return false;
    }
    return relation::prove(context, *proof, error);
}

bool zkdl_gat_relation_verify(const relation::Proof& proof,
                              const relation::VerifierContext& context,
                              std::string* error) {
    return relation::verify(proof, context, error);
}

bool zkdl_gat_native_core_prove(const NativeProveRequest& request,
                                NativeProofResult* result,
                                std::string* error) {
    if (!result) {
        if (error) *error = "native prove result pointer is null";
        return false;
    }
    if (request.backend == Backend::Gpu)
        return reject(result, error,
                      "BLOCKED: native GPU GAT proof provenance is not implemented; CPU fallback is forbidden",
                      request.config, request.statement, request.backend);
    relation::ProverContext context;
    if (!context_from_request(request, context, error))
        return reject(result, error, error ? *error : "native relation context rejected",
                      request.config, request.statement, request.backend);
    relation::Proof proof;
    if (!relation::prove(context, proof, error))
        return reject(result, error, error ? *error : "native relation prover rejected witness",
                      request.config, request.statement, request.backend);
    result->proof_bytes = proof.serialize();
    result->envelope.native_core_api = kBridgeApi;
    result->envelope.statement_id = request.statement->statement_id;
    result->envelope.proof_id = digest_hex(zkdl::sha256(result->proof_bytes));
    result->envelope.backend = backend_name(request.backend);
    result->envelope.proof_scope = request.statement->proof_scope;
    result->envelope.status = "VERIFIED";
    result->envelope.static_table_descriptor_digest = request.statement->static_table_descriptor_digest;
    result->envelope.scale_chain_digest = request.statement->scale_chain_digest;
    result->envelope.integer_bound_ledger_id = request.statement->integer_bound_ledger_id;
    result->envelope.integer_bound_ledger_digest = request.statement->integer_bound_ledger_digest;
    result->envelope.integer_bound_ledger_descriptor_digest = request.statement->integer_bound_ledger_descriptor_digest;
    result->envelope.proof_size_bytes = result->proof_bytes.size();
    result->envelope.commitment_size_bytes = proof.commitments.size();
    result->envelope.verifier_reads_witness = false;
    result->envelope.gpu_kernel_executed = false;
    result->envelope.silent_fallback = false;
    result->envelope.output_bound = true;
    result->envelope.error.clear();
    return true;
}

bool zkdl_gat_native_core_verify(const NativeVerifyRequest& request,
                                 std::string* error) {
    if (!request.config || !request.statement || !request.envelope || !request.proof_bytes)
        return reject(nullptr, error, "native verify request lacks proof envelope/bytes",
                      request.config, request.statement, request.backend);
    if (request.backend == Backend::Gpu)
        return reject(nullptr, error,
                      "BLOCKED: native GPU GAT verifier provenance is not implemented",
                      request.config, request.statement, request.backend);
    if (request.envelope->schema_id != "zkDL_gat_proof_envelope_v1" ||
        request.envelope->statement_id != request.statement->statement_id ||
        request.envelope->backend != backend_name(request.backend) ||
        request.envelope->proof_scope != kProofScope ||
        request.envelope->status != "VERIFIED" || request.envelope->silent_fallback ||
        request.envelope->verifier_reads_witness || !request.envelope->output_bound ||
        request.envelope->native_core_api != kBridgeApi ||
        request.envelope->static_table_descriptor_digest != request.statement->static_table_descriptor_digest ||
        request.envelope->scale_chain_digest != request.statement->scale_chain_digest ||
        request.envelope->integer_bound_ledger_id != request.statement->integer_bound_ledger_id ||
        request.envelope->integer_bound_ledger_digest != request.statement->integer_bound_ledger_digest ||
        request.envelope->integer_bound_ledger_descriptor_digest !=
            request.statement->integer_bound_ledger_descriptor_digest) {
        return reject(nullptr, error, "proof envelope is not an accepted native relation proof",
                      request.config, request.statement, request.backend);
    }
    relation::VerifierContext context;
    if (!context_from_verify(request, context, error))
        return reject(nullptr, error, error ? *error : "independent verifier context rejected",
                      request.config, request.statement, request.backend);
    relation::Proof proof;
    std::string parse_error;
    if (!relation::Proof::deserialize(*request.proof_bytes, proof, &parse_error))
        return reject(nullptr, error, "native GAT relation proof parse failed: " + parse_error,
                      request.config, request.statement, request.backend);
    if (request.envelope->proof_size_bytes != request.proof_bytes->size() ||
        request.envelope->proof_id != digest_hex(zkdl::sha256(*request.proof_bytes))) {
        return reject(nullptr, error, "proof envelope size/id is not bound to native proof bytes",
                      request.config, request.statement, request.backend);
    }
    if (!relation::verify(proof, context, error))
        return reject(nullptr, error, error ? *error : "native relation verifier rejected proof",
                      request.config, request.statement, request.backend);
    return true;
}

}  // namespace zkdl_gat
