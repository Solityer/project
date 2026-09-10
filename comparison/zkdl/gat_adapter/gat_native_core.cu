#include "gat_native_core.cuh"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

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

std::size_t serialized_commitment_section_size(const relation::Proof& proof) {
    std::vector<zkdl::byte> section;
    zkdl::append_u64(section, static_cast<std::uint64_t>(proof.commitments.size()));
    for (const auto& commitment : proof.commitments)
        relation::append_commitment(section, commitment);
    return section.size();
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

std::string graph_digest(const Graph& graph) {
    std::vector<zkdl::byte> encoded;
    const auto append_u64 = [&](std::uint64_t value) { zkdl::append_u64(encoded, value); };
    append_u64(graph.total_nodes);
    append_u64(graph.nodes);
    append_u64(graph.edges);
    append_u64(graph.graphs);
    append_u64(graph.features);
    append_u64(graph.absolute_ids.size());
    for (const auto value : graph.absolute_ids) append_u64(value);
    append_u64(graph.node_ptr.size());
    for (const auto value : graph.node_ptr) append_u64(value);
    append_u64(graph.edge_ptr.size());
    for (const auto value : graph.edge_ptr) append_u64(value);
    append_u64(graph.node_edge_ptr.size());
    for (const auto value : graph.node_edge_ptr) append_u64(value);
    append_u64(graph.src.size());
    for (std::size_t i = 0U; i < graph.src.size(); ++i) {
        append_u64(graph.src[i]);
        if (i >= graph.dst.size()) return std::string();
        append_u64(graph.dst[i]);
    }
    if (graph.dst.size() != graph.src.size()) return std::string();
    return digest_hex(zkdl::sha256(encoded));
}

std::string graph_boundary_digest(const Graph& graph) {
    std::vector<zkdl::byte> encoded;
    const auto append_u64 = [&](std::uint64_t value) { zkdl::append_u64(encoded, value); };
    const std::string tag = "zkDL_gat_graph_boundaries_v1";
    append_u64(tag.size());
    encoded.insert(encoded.end(), tag.begin(), tag.end());
    const auto append_vector = [&](const std::vector<std::uint64_t>& values) {
        append_u64(values.size());
        for (const auto value : values) append_u64(value);
    };
    append_vector(graph.node_ptr);
    append_vector(graph.edge_ptr);
    append_vector(graph.node_edge_ptr);
    return digest_hex(zkdl::sha256(encoded));
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
    const std::string expected_topology = graph_digest(graph);
    const std::string expected_boundaries = graph_boundary_digest(graph);
    if (expected_topology.empty() || statement.topology_id != expected_topology ||
        statement.topology_boundary_digest != expected_boundaries) {
        if (error) *error = "adapter graph src/dst or boundary digest is not statement-bound";
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
    out.public_output_digest = zkdl::sha256(out.public_output);
    if (statement.public_output_digest.empty() ||
        statement.public_output_digest != digest_hex(out.public_output_digest)) {
        if (error) *error = "public_Y digest is not bound to the canonical field vector";
        return false;
    }
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

bool add_fac_binding_spec(
    relation::OperationDescriptor& operation,
    const std::string& view_ref, const std::string& canonical_ref,
    const std::map<std::string, std::uint32_t>& domains,
    std::uint32_t view_outer_domain, std::uint32_t canonical_outer_domain,
    std::uint32_t view_feature_domain, std::uint32_t canonical_feature_domain,
    std::vector<std::uint32_t> source_outer_index,
    std::vector<std::uint32_t> source_feature_index,
    std::vector<std::uint8_t> selector_active,
    std::string* error) {
    if (view_ref.empty() || canonical_ref.empty() || view_ref == canonical_ref) return true;
    const auto view = domains.find(view_ref);
    const auto canonical = domains.find(canonical_ref);
    if (view == domains.end() || canonical == domains.end()) {
        if (error) *error = "BLOCKED: FAC4DNN binding omits a canonical tensor domain: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    const auto power_of_two = [](std::uint32_t value) {
        return value != 0U && (value & (value - 1U)) == 0U;
    };
    if (!power_of_two(view_outer_domain) || !power_of_two(canonical_outer_domain) ||
        !power_of_two(view_feature_domain) || !power_of_two(canonical_feature_domain) ||
        static_cast<std::uint64_t>(view_outer_domain) * view_feature_domain != view->second ||
        static_cast<std::uint64_t>(canonical_outer_domain) * canonical_feature_domain != canonical->second ||
        source_outer_index.size() != view_outer_domain ||
        source_feature_index.size() != view_feature_domain ||
        selector_active.size() != view_outer_domain) {
        if (error) *error = "BLOCKED: FAC4DNN selector map shape is not canonical for " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::FacBindingDescriptor binding;
    binding.view_ref = view_ref;
    binding.canonical_ref = canonical_ref;
    binding.view_outer_domain = view_outer_domain;
    binding.canonical_outer_domain = canonical_outer_domain;
    binding.view_feature_domain = view_feature_domain;
    binding.canonical_feature_domain = canonical_feature_domain;
    binding.source_outer_index = std::move(source_outer_index);
    binding.source_feature_index = std::move(source_feature_index);
    binding.selector_active = std::move(selector_active);
    if (!binding.valid(operation.source_refs, error)) {
        // The canonical source may be carried by tensor_refs rather than the
        // logical source set for legacy operation families.
        std::vector<std::string> refs = operation.source_refs;
        refs.insert(refs.end(), operation.tensor_refs.begin(), operation.tensor_refs.end());
        if (!binding.valid(refs, error)) return false;
    }
    append_unique(operation.source_refs, view_ref);
    append_unique(operation.source_refs, canonical_ref);
    operation.fac_bindings.push_back(std::move(binding));
    return true;
}

bool add_fac_binding(relation::OperationDescriptor& operation,
                     const std::string& view_ref, const std::string& canonical_ref,
                     const std::map<std::string, std::uint32_t>& domains,
                     std::string* error) {
    if (view_ref.empty() || canonical_ref.empty() || view_ref == canonical_ref) return true;
    const auto view = domains.find(view_ref);
    const auto canonical = domains.find(canonical_ref);
    if (view == domains.end() || canonical == domains.end()) {
        if (error) *error = "BLOCKED: FAC4DNN binding omits a canonical tensor domain: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    if (view->second != canonical->second) {
        if (error) *error = "BLOCKED: FAC4DNN cross-domain selector requires an explicit public map: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view->second, canonical->second, 1U, 1U,
                                [&]() {
                                    std::vector<std::uint32_t> values(view->second);
                                    for (std::uint32_t i = 0U; i < view->second; ++i) values[i] = i;
                                    return values;
                                }(), {0U}, std::vector<std::uint8_t>(view->second, 1U), error);
}

void sort_fac_bindings(relation::OperationDescriptor& operation) {
    std::sort(operation.fac_bindings.begin(), operation.fac_bindings.end(),
              [](const relation::FacBindingDescriptor& left,
                 const relation::FacBindingDescriptor& right) {
                  return std::tie(left.view_ref, left.canonical_ref) <
                         std::tie(right.view_ref, right.canonical_ref);
              });
}

bool bind_aligned_views(relation::OperationDescriptor& operation,
                        const OperationFamily& plan_operation,
                        const std::vector<std::string>& views,
                        const std::map<std::string, std::uint32_t>& domains,
                        std::string* error) {
    if (views.size() != plan_operation.inputs.size()) {
        if (error) *error = "BLOCKED: FAC4DNN input view count does not match RelationPlan";
        return false;
    }
    for (std::size_t i = 0U; i < views.size(); ++i)
        if (!add_fac_binding(operation, views[i], plan_operation.inputs[i], domains, error)) return false;
    return true;
}

bool bind_output_view(relation::OperationDescriptor& operation,
                      const OperationFamily& plan_operation,
                      const std::string& view,
                      std::size_t output_index,
                      const std::map<std::string, std::uint32_t>& domains,
                      std::string* error) {
    if (output_index >= plan_operation.outputs.size()) {
        if (error) *error = "BLOCKED: FAC4DNN output view index is outside RelationPlan";
        return false;
    }
    return add_fac_binding(operation, view, plan_operation.outputs[output_index], domains, error);
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

std::uint32_t padded_domain(std::uint64_t value, std::string* error,
                            const std::string& label) {
    if (value == 0U || value > (1ULL << 30U)) {
        if (error) *error = "BLOCKED: " + label + " has no bounded power-of-two domain";
        return 0U;
    }
    std::uint64_t domain = 1U;
    while (domain < value) domain <<= 1U;
    return static_cast<std::uint32_t>(domain);
}

std::uint64_t flattened_feature_domain(const TensorFamily& family,
                                       std::string* error,
                                       const std::string& label) {
    if (family.padded_shape.size() < 2U || family.padded_shape.size() > 3U ||
        family.padded_shape[0] == 0U) {
        if (error) *error = "BLOCKED: " + label + " has no canonical outer/feature shape";
        return 0U;
    }
    std::uint64_t feature = family.padded_shape[1];
    if (family.padded_shape.size() == 3U) {
        if (family.padded_shape[2] == 0U ||
            feature > std::numeric_limits<std::uint64_t>::max() / family.padded_shape[2]) {
            if (error) *error = "BLOCKED: " + label + " feature shape overflows";
            return 0U;
        }
        feature *= family.padded_shape[2];
    }
    if (feature == 0U || (feature & (feature - 1U)) != 0U) {
        if (error) *error = "BLOCKED: " + label + " feature domain is not a power of two";
        return 0U;
    }
    return feature;
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

bool add_fac_identity_from_families(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error);

bool add_fac_feature_column(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::uint32_t column, std::string* error);

relation::OperationDescriptor table_operation(
    const Config& config, const StaticTables& tables,
    const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
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
    operation.batch_family = "indexed_public_table";
    if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() != 1U) {
        if (error) *error = "BLOCKED: indexed table relation lacks query/table inputs and one output";
        return relation::OperationDescriptor{};
    }
    operation.public_table_ref = plan_operation.inputs.size() > 1U
        ? plan_operation.inputs[1] : std::string();
    for (const auto& input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.source_refs, output);
    std::int64_t minimum = 0;
    const std::vector<std::int64_t>& selected = lookup_table(tables, plan_operation, minimum, error);
    if (selected.empty()) return relation::OperationDescriptor{};
    if ((selected.size() & (selected.size() - 1U)) != 0U) {
        if (error) *error = "BLOCKED: authenticated table size is not a power of two for " + operation.id;
        return relation::OperationDescriptor{};
    }
    // The fixed multiplication-tree family is intentionally bounded by the
    // relation wire cap.  Reject a table whose tree would exceed that cap;
    // do not fall back to a row-by-row or metadata-only lookup proof.
    if (selected.size() > relation::kMaxRelationConstraints / 4U) {
        if (error) *error = "BLOCKED: authenticated table domain exceeds native lookup-family cap for " + operation.id;
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
    if (!add_fac_identity_from_families(operation, query,
                                        plan_operation.inputs.empty() ? std::string() : plan_operation.inputs[0],
                                        domains, families, error) ||
        (plan_operation.inputs.size() > 2U &&
         !add_fac_identity_from_families(operation, index, plan_operation.inputs[2],
                                         domains, families, error)) ||
        !add_fac_identity_from_families(operation, output, plan_operation.outputs[0],
                                        domains, families, error))
        return relation::OperationDescriptor{};
    if (plan_operation.inputs.size() > 3U) {
        for (std::uint32_t bit = 0U; bit < depth; ++bit)
            if (!add_fac_feature_column(operation, bits[bit], plan_operation.inputs[3],
                                        domains, families, bit, error))
                return relation::OperationDescriptor{};
    }
    sort_fac_bindings(operation);
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

const TensorFamily* tensor_family(
    const std::map<std::string, const TensorFamily*>& families,
    const std::string& id) {
    const auto found = families.find(id);
    return found == families.end() ? nullptr : found->second;
}

bool fac_axes(const std::map<std::string, const TensorFamily*>& families,
              const std::string& ref, std::uint32_t& outer,
              std::uint32_t& feature, std::string* error) {
    const TensorFamily* family = tensor_family(families, ref);
    if (family == nullptr || family->padded_shape.empty()) {
        if (error) *error = "BLOCKED: FAC4DNN tensor family is missing for " + ref;
        return false;
    }
    outer = padded_domain(family->padded_shape.front(), error, ref + "/outer");
    if (family->padded_shape.size() == 1U) {
        feature = 1U;
        return outer != 0U;
    }
    const std::uint64_t flattened = flattened_feature_domain(*family, error, ref);
    if (outer == 0U || flattened == 0U || flattened > std::numeric_limits<std::uint32_t>::max())
        return false;
    feature = static_cast<std::uint32_t>(flattened);
    return true;
}

bool add_fac_identity_from_families(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error)) return false;
    if (view_outer != canonical_outer || view_feature != canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN identity map has incompatible family axes: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer);
    for (std::uint32_t index = 0U; index < view_outer; ++index) outer_map[index] = index;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                [&]() {
                                    std::vector<std::uint32_t> feature_map(view_feature);
                                    for (std::uint32_t index = 0U; index < view_feature; ++index)
                                        feature_map[index] = index;
                                    return feature_map;
                                }(), std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_broadcast_from_graph(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families, const Graph& graph,
    bool destination, std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error)) return false;
    const std::uint32_t expected_view_outer = padded_domain(graph.edges, error, view_ref + "/edge");
    const std::uint32_t expected_source_outer = padded_domain(graph.nodes, error, canonical_ref + "/node");
    if (expected_view_outer == 0U || expected_source_outer == 0U ||
        view_outer != expected_view_outer || canonical_outer != expected_source_outer ||
        (view_feature != canonical_feature && canonical_feature != 1U)) {
        if (error) *error = "BLOCKED: FAC4DNN graph route axes do not match public graph: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer, 0U);
    std::vector<std::uint8_t> active(view_outer, 0U);
    for (std::uint64_t edge = 0U; edge < graph.edges; ++edge) {
        if (graph.src[edge] >= graph.nodes || graph.dst[edge] >= graph.nodes) {
            if (error) *error = "BLOCKED: FAC4DNN graph route index is out of bounds";
            return false;
        }
        outer_map[edge] = destination ? graph.dst[edge] : graph.src[edge];
        active[edge] = 1U;
    }
    std::vector<std::uint32_t> feature_map(view_feature, 0U);
    if (canonical_feature == 1U) {
        std::fill(feature_map.begin(), feature_map.end(), 0U);
    } else {
        if (view_feature != canonical_feature) {
            if (error) *error = "BLOCKED: FAC4DNN graph route feature map is not canonical";
            return false;
        }
        for (std::uint32_t feature = 0U; feature < view_feature; ++feature)
            feature_map[feature] = feature;
    }
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool add_fac_feature_broadcast(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error)) return false;
    if (view_outer != canonical_outer || canonical_feature != 1U) {
        if (error) *error = "BLOCKED: FAC4DNN feature broadcast axes are not canonical: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer);
    for (std::uint32_t index = 0U; index < view_outer; ++index) outer_map[index] = index;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                std::vector<std::uint32_t>(view_feature, 0U),
                                std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_selector_column(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::uint32_t column, std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        view_outer != canonical_outer || view_feature != 1U ||
        canonical_feature == 0U || column >= canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN public selector-column axes are not canonical: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer);
    for (std::uint32_t index = 0U; index < view_outer; ++index) outer_map[index] = index;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                {column}, std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_feature_block(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::uint32_t block, std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        view_outer != canonical_outer || view_feature == 0U ||
        block > std::numeric_limits<std::uint32_t>::max() / view_feature ||
        static_cast<std::uint64_t>(block) * view_feature + view_feature > canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN feature-block axes are not canonical: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer);
    for (std::uint32_t index = 0U; index < view_outer; ++index) outer_map[index] = index;
    std::vector<std::uint32_t> feature_map(view_feature);
    for (std::uint32_t index = 0U; index < view_feature; ++index)
        feature_map[index] = block * view_feature + index;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                std::move(feature_map),
                                std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_scalar_broadcast(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        canonical_outer != 1U || canonical_feature != 1U) {
        if (error) *error = "BLOCKED: FAC4DNN scalar broadcast axes are not canonical: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature,
                                std::vector<std::uint32_t>(view_outer, 0U),
                                std::vector<std::uint32_t>(view_feature, 0U),
                                std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_vector_broadcast(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error)) return false;
    const TensorFamily* canonical = tensor_family(families, canonical_ref);
    if (canonical == nullptr || canonical->padded_shape.size() != 1U ||
        canonical->padded_shape.front() == 0U ||
        canonical->padded_shape.front() != view_feature) {
        if (error) *error = "BLOCKED: FAC4DNN vector broadcast axes are not canonical: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer, 0U);
    std::vector<std::uint32_t> feature_map(view_feature);
    for (std::uint32_t index = 0U; index < view_feature; ++index) feature_map[index] = index;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, 1U, view_feature, view_feature,
                                std::move(outer_map), std::move(feature_map),
                                std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_feature_column(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::uint32_t column, std::string* error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) ||
        !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        view_outer != canonical_outer || view_feature != 1U ||
        canonical_feature == 0U || column >= canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN feature-column axes are not canonical: " +
                            view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> outer_map(view_outer);
    for (std::uint32_t index = 0U; index < view_outer; ++index) outer_map[index] = index;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                {column}, std::vector<std::uint8_t>(view_outer, 1U), error);
}

bool add_fac_concat_binding(
    relation::OperationDescriptor& operation, const std::string& view_ref,
    const std::string& canonical_ref, const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::uint32_t head, std::string* error) {
    const TensorFamily* view = tensor_family(families, view_ref);
    const TensorFamily* canonical = tensor_family(families, canonical_ref);
    if (view == nullptr || canonical == nullptr || view->padded_shape.size() != 3U ||
        canonical->padded_shape.size() < 2U || view->padded_shape[1] == 0U ||
        view->padded_shape[2] != canonical->padded_shape.back() ||
        head >= view->padded_shape[1]) {
        if (error) *error = "BLOCKED: FAC4DNN concat layout is not authenticated";
        return false;
    }
    const std::uint32_t view_outer = padded_domain(
        view->padded_shape[0] * view->padded_shape[1], error, view_ref + "/outer");
    const std::uint32_t canonical_outer = padded_domain(
        canonical->padded_shape.front(), error, canonical_ref + "/outer");
    const std::uint32_t view_feature = padded_domain(view->padded_shape[2], error,
                                                     view_ref + "/feature");
    const std::uint32_t canonical_feature = padded_domain(canonical->padded_shape.back(), error,
                                                          canonical_ref + "/feature");
    if (view_outer == 0U || canonical_outer == 0U || view_feature == 0U ||
        canonical_feature == 0U || view_feature != canonical_feature) return false;
    std::vector<std::uint32_t> outer_map(view_outer, 0U);
    std::vector<std::uint8_t> active(view_outer, 0U);
    const std::uint32_t head_domain = static_cast<std::uint32_t>(view->padded_shape[1]);
    for (std::uint32_t slot = 0U; slot < view_outer; ++slot) {
        const std::uint32_t slot_head = slot % head_domain;
        const std::uint32_t node = slot / head_domain;
        if (slot_head == head) {
            outer_map[slot] = node;
            active[slot] = 1U;
        }
    }
    std::vector<std::uint32_t> feature_map(view_feature);
    for (std::uint32_t feature = 0U; feature < view_feature; ++feature)
        feature_map[feature] = feature;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains,
                                view_outer, canonical_outer, view_feature,
                                canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool padded_matrix_shape(
    const std::map<std::string, const TensorFamily*>& families,
    const std::string& input_id, const std::string& kernel_id,
    const std::string& output_id, relation::PrimitiveDescriptor& primitive,
    std::string* error) {
    const TensorFamily* input = tensor_family(families, input_id);
    const TensorFamily* kernel = tensor_family(families, kernel_id);
    const TensorFamily* output = tensor_family(families, output_id);
    if (input == nullptr || kernel == nullptr || output == nullptr ||
        (input->padded_shape.size() != 1U && input->padded_shape.size() != 2U &&
         input->padded_shape.size() != 3U) ||
        (kernel->padded_shape.size() != 1U && kernel->padded_shape.size() != 3U) ||
        (output->padded_shape.size() != 1U && output->padded_shape.size() != 2U)) {
        if (error) *error = "BLOCKED: matrix relation lacks canonical padded input/kernel/output shapes";
        return false;
    }
    if (input->padded_shape.size() == 1U || output->padded_shape.size() == 1U) {
        if (input->padded_shape.size() != 2U && input->padded_shape.size() != 3U) {
            if (error) *error = "BLOCKED: matrix input view must expose row and inner dimensions";
            return false;
        }
    }
    const std::uint64_t kernel_inner = kernel->padded_shape.size() == 1U
        ? kernel->padded_shape[0] : kernel->padded_shape[1];
    const std::uint64_t kernel_output = kernel->padded_shape.size() == 1U
        ? 1U : kernel->padded_shape[2];
    const std::uint64_t output_rows = output->padded_shape[0];
    const std::uint64_t output_cols = output->padded_shape.size() == 1U
        ? 1U : output->padded_shape[1];
    if (kernel->padded_shape.size() == 3U && kernel->padded_shape[0] != 1U) {
        if (error) *error = "BLOCKED: matrix kernel batch dimension is not one";
        return false;
    }
    if (input->padded_shape.size() == 3U &&
        (input->padded_shape[1] == 0U ||
         input->padded_shape[2] > std::numeric_limits<std::uint64_t>::max() /
                                     input->padded_shape[1])) {
        if (error) *error = "BLOCKED: canonical padded matrix inner dimension overflows";
        return false;
    }
    const std::uint64_t input_inner = input->padded_shape.size() == 2U
        ? input->padded_shape[1]
        : input->padded_shape[1] * input->padded_shape[2];
    if (input_inner != kernel_inner ||
        output_rows != input->padded_shape[0] || output_cols != kernel_output ||
        input_inner > std::numeric_limits<std::uint32_t>::max()) {
        if (error) *error = "BLOCKED: canonical padded matrix inner dimension does not match kernel";
        return false;
    }
    primitive.kind = 1U;
    primitive.matmul_shape.num = 1U;
    primitive.matmul_shape.m = static_cast<std::uint32_t>(input->padded_shape[0]);
    primitive.matmul_shape.n = static_cast<std::uint32_t>(input_inner);
    primitive.matmul_shape.k = static_cast<std::uint32_t>(output_cols);
    primitive.tensor_refs = {input_id, kernel_id, output_id};
    std::string shape_error;
    if (!primitive.matmul_shape.valid(&shape_error)) {
        if (error) *error = "BLOCKED: canonical padded matrix shape is invalid: " + shape_error;
        return false;
    }
    return true;
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

bool same_domain_refs(const std::map<std::string, std::uint32_t>& domains,
                      const std::vector<std::string>& refs,
                      std::uint32_t* domain, std::string* error,
                      const std::string& operation_id) {
    if (refs.empty()) {
        if (error) *error = "BLOCKED: relation " + operation_id + " has no canonical views";
        return false;
    }
    const auto first = domains.find(refs.front());
    if (first == domains.end()) {
        if (error) *error = "BLOCKED: relation " + operation_id + " is missing view " + refs.front();
        return false;
    }
    for (const auto& ref : refs) {
        const auto found = domains.find(ref);
        if (found == domains.end() || found->second != first->second) {
            if (error) *error = "BLOCKED: relation " + operation_id +
                                " canonical views have different MLE domains";
            return false;
        }
    }
    if (domain) *domain = first->second;
    return true;
}

relation::ConstraintDescriptor recurrence_constraint(
    const relation::OperationDescriptor& operation, const std::string& id,
    const std::string& prefix, const std::string& prefix_prev,
    const std::string& input, const std::string& group_new) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(one, {local_ref(operation, prefix)}));
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(-1),
                                        {local_ref(operation, prefix_prev)}));
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(-1),
                                        {local_ref(operation, input)}));
    constraint.terms.push_back(monomial(one, {local_ref(operation, group_new),
                                              local_ref(operation, prefix_prev)}));
    return constraint;
}

relation::OperationDescriptor grouped_sum_operation(
    const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    const Graph* graph,
    std::string* error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = relation::RelationKind::GroupSum;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;group-boundary-masked";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    for (const auto& ref : plan_operation.inputs) append_unique(operation.source_refs, ref);
    for (const auto& ref : plan_operation.outputs) append_unique(operation.source_refs, ref);
    const std::string group_new = required_view(plan_operation, domains, "group_new", "", error);
    const std::string group_end = required_view(plan_operation, domains, "group_end", "", error);
    if (group_new.empty() || group_end.empty()) return relation::OperationDescriptor{};
    add_ref(operation, group_new);
    add_ref(operation, group_end);
    std::vector<std::string> all_refs{group_new, group_end};
    if (plan_operation.inputs.empty() || plan_operation.outputs.empty()) {
        if (error) *error = "BLOCKED: grouped sum requires canonical input/output views";
        return relation::OperationDescriptor{};
    }
    for (std::size_t index = 0U; index < plan_operation.outputs.size(); ++index) {
        const std::string input = required_view(plan_operation, domains,
                                                "input" + std::to_string(index),
                                                plan_operation.inputs[std::min(index, plan_operation.inputs.size() - 1U)], error);
        const std::string output = required_view(plan_operation, domains,
                                                 "output" + std::to_string(index),
                                                 plan_operation.outputs[index], error);
        const std::string prefix = required_view(plan_operation, domains,
                                                 "prefix" + std::to_string(index), "", error);
        const std::string prefix_prev = required_view(plan_operation, domains,
                                                      "prefix_prev" + std::to_string(index), "", error);
        if (input.empty() || output.empty() || prefix.empty() || prefix_prev.empty())
            return relation::OperationDescriptor{};
        add_ref(operation, input); add_ref(operation, output);
        add_ref(operation, prefix); add_ref(operation, prefix_prev);
        all_refs.insert(all_refs.end(), {input, output, prefix, prefix_prev});
        operation.constraints.push_back(recurrence_constraint(
            operation, operation.id + "/prefix/" + std::to_string(index),
            prefix, prefix_prev, input, group_new));
        relation::ConstraintDescriptor reset;
        reset.id = operation.id + "/prefix-reset/" + std::to_string(index);
        reset.degree = 2U;
        reset.terms.push_back(monomial(zkdl::fr_from_u32(1U),
                                       {local_ref(operation, group_new), local_ref(operation, prefix_prev)}));
        operation.constraints.push_back(std::move(reset));
        relation::ConstraintDescriptor terminal;
        terminal.id = operation.id + "/group-terminal/" + std::to_string(index);
        terminal.degree = 2U;
        terminal.terms.push_back(monomial(zkdl::fr_from_u32(1U),
                                          {local_ref(operation, group_end), local_ref(operation, output)}));
        terminal.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                                          {local_ref(operation, group_end), local_ref(operation, prefix)}));
        operation.constraints.push_back(std::move(terminal));
    }
    if (graph == nullptr || plan_operation.inputs.empty() ||
        plan_operation.outputs.empty()) {
        if (error) *error = "BLOCKED: grouped sum lacks authenticated graph/input/output mapping";
        return relation::OperationDescriptor{};
    }
    const std::string input0 = required_view(plan_operation, domains, "input0",
                                             plan_operation.inputs[0], error);
    const std::string input1 = plan_operation.inputs.size() > 1U
        ? required_view(plan_operation, domains, "input1", plan_operation.inputs[1], error)
        : std::string();
    const std::string output0 = required_view(plan_operation, domains, "output0",
                                              plan_operation.outputs[0], error);
    const std::string output1 = plan_operation.outputs.size() > 1U
        ? required_view(plan_operation, domains, "output1", plan_operation.outputs[1], error)
        : std::string();
    if (input0.empty() || output0.empty() ||
        (plan_operation.inputs.size() > 1U && input1.empty()) ||
        (plan_operation.outputs.size() > 1U && output1.empty()))
        return relation::OperationDescriptor{};
    // Group-sum views are edge-major.  The first input is a scalar edge
    // weight broadcast over feature coordinates; the second input already
    // carries the feature axis.  Outputs use the public destination CSR
    // route, with scalar denominator broadcast where required.
    const bool input0_scalar = [&]() {
        std::uint32_t outer = 0U, feature = 0U;
        return fac_axes(families, input0, outer, feature, nullptr) && feature == 1U;
    }();
    if ((input0_scalar
         ? !add_fac_feature_broadcast(operation, input0, plan_operation.inputs[0],
                                      domains, families, error)
         : !add_fac_identity_from_families(operation, input0, plan_operation.inputs[0],
                                           domains, families, error)) ||
        (plan_operation.inputs.size() > 1U &&
         !add_fac_identity_from_families(operation, input1, plan_operation.inputs[1],
                                         domains, families, error)) ||
        !add_fac_broadcast_from_graph(operation, output0, plan_operation.outputs[0],
                                      domains, families, *graph, true, error) ||
        (plan_operation.outputs.size() > 1U &&
         !add_fac_broadcast_from_graph(operation, output1, plan_operation.outputs[1],
                                       domains, families, *graph, true, error)))
        return relation::OperationDescriptor{};
    append_unique(operation.source_refs, "group_selectors");
    if (!add_fac_selector_column(operation, group_new, "group_selectors",
                                 domains, families, 0U, error) ||
        !add_fac_selector_column(operation, group_end, "group_selectors",
                                 domains, families, 1U, error))
        return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    std::uint32_t domain = 0U;
    if (!same_domain_refs(domains, all_refs, &domain, error, operation.id))
        return relation::OperationDescriptor{};
    (void)domain;
    std::string validation_error;
    if (!operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: grouped sum descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor max_range_operation(
    const Config& config, const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    const Graph* graph,
    std::string* error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = relation::RelationKind::MaxSelector;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;group-boundary-masked";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    for (const auto& ref : plan_operation.inputs) append_unique(operation.source_refs, ref);
    for (const auto& ref : plan_operation.outputs) append_unique(operation.source_refs, ref);
    if (plan_operation.outputs.size() < 3U || plan_operation.inputs.empty()) {
        if (error) *error = "BLOCKED: max/range relation requires value, maximum, selector and delta";
        return relation::OperationDescriptor{};
    }
    const std::string value = required_view(plan_operation, domains, "value", plan_operation.inputs.front(), error);
    const std::string maximum = required_view(plan_operation, domains, "maximum_broadcast", "", error);
    const std::string selector = required_view(plan_operation, domains, "selector", "", error);
    const std::string delta = required_view(plan_operation, domains, "delta", "", error);
    const std::string group_new = required_view(plan_operation, domains, "group_new", "", error);
    const std::string group_end = required_view(plan_operation, domains, "group_end", "", error);
    const std::string selector_prefix = required_view(plan_operation, domains, "selector_prefix", "", error);
    const std::string selector_prev = required_view(plan_operation, domains, "selector_prev", "", error);
    for (const auto& ref : {value, maximum, selector, delta, group_new, group_end,
                            selector_prefix, selector_prev}) {
        if (ref.empty()) return relation::OperationDescriptor{};
        add_ref(operation, ref);
    }
    std::uint32_t width = static_cast<std::uint32_t>(config.range_bits);
    if (width == 0U || width >= 31U) {
        if (error) *error = "BLOCKED: max/range relation has an unbounded range width";
        return relation::OperationDescriptor{};
    }
    std::vector<std::string> all_refs{value, maximum, selector, delta, group_new,
                                      group_end, selector_prefix, selector_prev};
    relation::ConstraintDescriptor max_delta;
    max_delta.id = operation.id + "/delta";
    max_delta.degree = 1U;
    max_delta.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, maximum)}));
    max_delta.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, value)}));
    max_delta.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, delta)}));
    operation.constraints.push_back(std::move(max_delta));
    relation::ConstraintDescriptor selected;
    selected.id = operation.id + "/selector-delta";
    selected.degree = 2U;
    selected.terms.push_back(monomial(zkdl::fr_from_u32(1U),
                                      {local_ref(operation, selector), local_ref(operation, delta)}));
    operation.constraints.push_back(std::move(selected));
    operation.constraints.push_back(boolean_constraint(operation.id + "/selector-boolean", operation, selector));
    operation.constraints.push_back(recurrence_constraint(
        operation, operation.id + "/one-hot-prefix", selector_prefix, selector_prev,
        selector, group_new));
    relation::ConstraintDescriptor prefix_reset;
    prefix_reset.id = operation.id + "/one-hot-reset";
    prefix_reset.degree = 2U;
    prefix_reset.terms.push_back(monomial(zkdl::fr_from_u32(1U),
                                          {local_ref(operation, group_new), local_ref(operation, selector_prev)}));
    operation.constraints.push_back(std::move(prefix_reset));
    relation::ConstraintDescriptor one_hot;
    one_hot.id = operation.id + "/one-hot-terminal";
    one_hot.degree = 2U;
    one_hot.terms.push_back(monomial(zkdl::fr_from_u32(1U),
                                     {local_ref(operation, group_end), local_ref(operation, selector_prefix)}));
    one_hot.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                                     {local_ref(operation, group_end)}));
    operation.constraints.push_back(std::move(one_hot));
    std::vector<std::string> bits;
    bits.reserve(width);
    for (std::uint32_t bit = 0U; bit < width; ++bit) {
        const std::string ref = required_view(plan_operation, domains,
                                              "range_bit/" + std::to_string(bit), "", error);
        if (ref.empty()) return relation::OperationDescriptor{};
        bits.push_back(ref); add_ref(operation, ref); all_refs.push_back(ref);
        operation.constraints.push_back(boolean_constraint(
            operation.id + "/range-bit/" + std::to_string(bit), operation, ref));
    }
    operation.constraints.push_back(reconstruction_constraint(
        operation.id + "/range", operation, delta, bits, width));
    if (graph == nullptr) {
        if (error) *error = "BLOCKED: max/range relation lacks authenticated graph selector map";
        return relation::OperationDescriptor{};
    }
    if (!add_fac_identity_from_families(operation, value, plan_operation.inputs.front(),
                                         domains, families, error) ||
        !add_fac_broadcast_from_graph(operation, maximum, plan_operation.outputs[0],
                                      domains, families, *graph, true, error) ||
        !add_fac_identity_from_families(operation, selector, plan_operation.outputs[1],
                                         domains, families, error) ||
        !add_fac_identity_from_families(operation, delta, plan_operation.outputs[2],
                                         domains, families, error))
        return relation::OperationDescriptor{};
    const std::size_t max_suffix = operation.id.rfind("/max_range");
    const std::string canonical_range_bits = max_suffix == std::string::npos
        ? std::string() : operation.id.substr(0U, max_suffix) + "/trace/range_bits";
    if (canonical_range_bits.empty() || families.count(canonical_range_bits) == 0U) {
        if (error) *error = "BLOCKED: max/range canonical range_bits family is missing";
        return relation::OperationDescriptor{};
    }
    append_unique(operation.source_refs, canonical_range_bits);
    for (std::uint32_t bit = 0U; bit < width; ++bit)
        if (!add_fac_feature_column(operation, bits[bit], canonical_range_bits,
                                    domains, families, bit, error))
            return relation::OperationDescriptor{};
    append_unique(operation.source_refs, "group_selectors");
    if (!add_fac_selector_column(operation, group_new, "group_selectors",
                                 domains, families, 0U, error) ||
        !add_fac_selector_column(operation, group_end, "group_selectors",
                                 domains, families, 1U, error))
        return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    if (!same_domain_refs(domains, all_refs, nullptr, error, operation.id))
        return relation::OperationDescriptor{};
    std::string validation_error;
    if (!operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: max/range descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor signed_round_operation(
    const Config& config, const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = plan_operation.id == "final_average"
        ? relation::RelationKind::FinalAverage : relation::RelationKind::SignedRoundDiv;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;masked-from-relation";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    operation.batch_family = "signed_round_div";
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
    if (plan_operation.inputs.size() >= 2U) {
        bool divisor_is_public_broadcast = false;
        if (domains.count(divisor) != 0U && domains.count(plan_operation.inputs[1]) != 0U &&
            domains.at(divisor) != domains.at(plan_operation.inputs[1])) {
            // final_average broadcasts the public head count from a scalar
            // plan tensor over the node/class MLE domain.
            if (plan_operation.id == "final_average") {
                operation.has_public_divisor = true;
                operation.public_divisor = relation::fr_from_signed_i64(
                    static_cast<std::int64_t>(config.architecture.output_heads));
                divisor_is_public_broadcast = true;
            } else {
                if (error) *error = "BLOCKED: SignedRoundDiv divisor view is cross-domain without a public scale route";
                return relation::OperationDescriptor{};
            }
        }
        if (!add_fac_identity_from_families(operation, numerator, plan_operation.inputs[0],
                                            domains, families, error) ||
            (!divisor_is_public_broadcast &&
             !add_fac_feature_broadcast(operation, divisor, plan_operation.inputs[1],
                                        domains, families, error)) ||
            (divisor_is_public_broadcast &&
             !add_fac_scalar_broadcast(operation, divisor, plan_operation.inputs[1],
                                       domains, families, error)) ||
            !add_fac_identity_from_families(operation, signed_q, plan_operation.outputs[0],
                                            domains, families, error))
            return relation::OperationDescriptor{};
    } else {
        // output_scale_alignment carries a public scalar scale in Config, not
        // a second plan tensor.  Bind that broadcast value as an equation in
        // the batched family instead of accepting an unconstrained divisor.
        if (plan_operation.id.find("/output_scale_alignment") == std::string::npos) {
            if (error) *error = "BLOCKED: SignedRoundDiv has no canonical divisor input";
            return relation::OperationDescriptor{};
        }
        operation.has_public_divisor = true;
        operation.public_divisor = relation::fr_from_signed_i64(
            static_cast<std::int64_t>(config.quantization.output_pre));
        if (!add_fac_identity_from_families(operation, numerator, plan_operation.inputs.front(),
                                            domains, families, error) ||
            !add_fac_identity_from_families(operation, signed_q, plan_operation.outputs[0],
                                            domains, families, error))
            return relation::OperationDescriptor{};
    }
    sort_fac_bindings(operation);
    const std::vector<std::string> range_values = {a, q, r, delta, divisor};
    std::map<std::string, std::vector<std::string>> range_bits;
    // Keep the family stack bounded before allocating any per-bit view list.
    // The shipped secure configs use at most 21 bits; a larger range must
    // arrive with a separately reviewed native wire budget, never by silently
    // expanding the proof envelope.
    constexpr std::uint32_t kMaxSignedRoundRangeBits = 24U;
    if (config.range_bits == 0U || config.range_bits > kMaxSignedRoundRangeBits) {
        if (error) *error = "BLOCKED: SignedRoundDiv range_bits is not a bounded native width";
        return relation::OperationDescriptor{};
    }
    const std::uint32_t width = static_cast<std::uint32_t>(config.range_bits);
    const std::uint64_t estimated_constraints = 16U +
        5U * static_cast<std::uint64_t>(width);
    if (width >= 31U ||
        estimated_constraints > relation::kMaxRelationConstraints) {
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
    // The auxiliary stack is the canonical node-major [sigma,z,a,q,r,
    // delta,divisor,reserved] view emitted by the runner.  Each operation
    // view is a public feature block of that authenticated tensor, so every
    // quotient/remainder/sign witness is linked before the batched family is
    // opened.
    if (plan_operation.outputs.size() > 1U) {
        const std::string aux = plan_operation.outputs[1];
        if (!add_fac_feature_block(operation, sigma, aux, domains, families, 0U, error) ||
            !add_fac_feature_block(operation, zero, aux, domains, families, 1U, error) ||
            !add_fac_feature_block(operation, a, aux, domains, families, 2U, error) ||
            !add_fac_feature_block(operation, q, aux, domains, families, 3U, error) ||
            !add_fac_feature_block(operation, r, aux, domains, families, 4U, error) ||
            !add_fac_feature_block(operation, delta, aux, domains, families, 5U, error) ||
            !add_fac_feature_block(operation, divisor, aux, domains, families, 6U, error))
            return relation::OperationDescriptor{};
    }
    const std::string aux_prefix = plan_operation.id;
    const std::string a_inverse_canonical = aux_prefix.find("/signed_round_div") != std::string::npos
        ? aux_prefix.substr(0U, aux_prefix.find("/signed_round_div")) + "/trace/rounddiv_a_inverse"
        : std::string();
    const std::string divisor_inverse_canonical = aux_prefix.find("/signed_round_div") != std::string::npos
        ? aux_prefix.substr(0U, aux_prefix.find("/signed_round_div")) + "/trace/rounddiv_divisor_inverse"
        : std::string();
    if (!a_inverse_canonical.empty() && domains.count(a_inverse_canonical) != 0U &&
        !add_fac_identity_from_families(operation, a_inverse, a_inverse_canonical,
                                        domains, families, error)) return relation::OperationDescriptor{};
    if (!divisor_inverse_canonical.empty() && domains.count(divisor_inverse_canonical) != 0U &&
        !add_fac_identity_from_families(operation, divisor_inverse, divisor_inverse_canonical,
                                        domains, families, error)) return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    relation::ConstraintDescriptor signed_n;
    signed_n.id = operation.id + "/signed-numerator";
    signed_n.degree = 2U;
    signed_n.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, numerator)}));
    signed_n.terms.push_back(monomial(relation::fr_from_signed_i64(-1), {local_ref(operation, a)}));
    signed_n.terms.push_back(monomial(relation::fr_from_signed_i64(2), {local_ref(operation, sigma), local_ref(operation, a)}));
    operation.constraints.push_back(std::move(signed_n));
    if (operation.has_public_divisor) {
        relation::ConstraintDescriptor public_divisor;
        public_divisor.id = operation.id + "/public-divisor";
        public_divisor.degree = 1U;
        public_divisor.constant = zkdl::fr_neg(operation.public_divisor);
        public_divisor.terms.push_back(monomial(
            zkdl::fr_from_u32(1U), {local_ref(operation, divisor)}));
        operation.constraints.push_back(std::move(public_divisor));
    }
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

relation::OperationDescriptor sparse_route_operation(
    const Graph& graph, const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    relation::OperationDescriptor operation;
    if (plan_operation.inputs.size() != 2U || plan_operation.outputs.size() != 1U) {
        if (error) *error = "BLOCKED: sparse route requires public route input, canonical source, and one output view";
        return operation;
    }
    const std::string source_ref = plan_operation.inputs.back();
    const std::string output_ref = plan_operation.outputs.front();
    const TensorFamily* source_family = tensor_family(families, source_ref);
    const TensorFamily* output_family = tensor_family(families, output_ref);
    if (source_family == nullptr || output_family == nullptr) {
        if (error) *error = "BLOCKED: sparse route source/output family is not in the authenticated RelationPlan";
        return operation;
    }
    if (graph.nodes == 0U || graph.edges == 0U || graph.src.size() != graph.edges ||
        graph.dst.size() != graph.edges) {
        if (error) *error = "BLOCKED: sparse route graph has incomplete public src/dst arrays";
        return operation;
    }
    const std::uint32_t node_domain = padded_domain(source_family->padded_shape.front(), error,
                                                     plan_operation.id + "/node");
    const std::uint32_t edge_domain = padded_domain(output_family->padded_shape.front(), error,
                                                     plan_operation.id + "/edge");
    const std::uint64_t source_feature = flattened_feature_domain(*source_family, error,
                                                                  plan_operation.id + "/source");
    const std::uint64_t output_feature = flattened_feature_domain(*output_family, error,
                                                                  plan_operation.id + "/output");
    if (node_domain == 0U || edge_domain == 0U || source_feature == 0U ||
        output_feature == 0U || (source_feature != output_feature && source_feature != 1U) ||
        source_feature > std::numeric_limits<std::uint32_t>::max() ||
        output_feature > std::numeric_limits<std::uint32_t>::max() ||
        graph.nodes > node_domain || graph.edges > edge_domain ||
        domains.find(source_ref) == domains.end() || domains.find(output_ref) == domains.end() ||
        domains.at(source_ref) != static_cast<std::uint64_t>(node_domain) * source_feature ||
        domains.at(output_ref) != static_cast<std::uint64_t>(edge_domain) * output_feature) {
        if (error) *error = "BLOCKED: sparse route source/output shape is not fixed zero-padded and aligned";
        return operation;
    }
    operation.id = plan_operation.id;
    operation.kind = relation::RelationKind::SparseRoute;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping + ";public-selector-mle:src-dst";
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "fixed-zero-padding;public-equality-selector-mle;fac4dnn-line8";
    operation.public_route = true;
    operation.requires_reindex = true;
    for (const auto& input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.source_refs, output);
    for (const auto& auxiliary : plan_operation.auxiliary) operation.selector_refs.push_back(auxiliary);
    operation.selector_edge_domain = edge_domain;
    operation.selector_node_domain = node_domain;
    operation.selector_feature_domain = static_cast<std::uint32_t>(output_feature);
    operation.selector_source_feature_domain = static_cast<std::uint32_t>(source_feature);
    operation.selector_src.assign(edge_domain, 0U);
    operation.selector_dst.assign(edge_domain, 0U);
    operation.selector_active.assign(edge_domain, 0U);
    operation.selector_feature_map.resize(static_cast<std::size_t>(output_feature));
    for (std::uint32_t feature = 0U; feature < output_feature; ++feature)
        operation.selector_feature_map[feature] = source_feature == 1U ? 0U : feature;
    for (std::uint64_t edge = 0U; edge < graph.edges; ++edge) {
        if (graph.src[edge] >= graph.nodes || graph.dst[edge] >= graph.nodes) {
            if (error) *error = "BLOCKED: sparse route graph src/dst index is out of bounds";
            return relation::OperationDescriptor{};
        }
        operation.selector_src[edge] = graph.src[edge];
        operation.selector_dst[edge] = graph.dst[edge];
        operation.selector_active[edge] = 1U;
    }
    operation.selector_source_ref = source_ref;
    operation.selector_output_ref = output_ref;
    // The route relation itself owns the sparse selector polynomial.  The
    // source/output commitments are bound by the route proof's canonical
    // Hyrax openings; no tautological anchor constraint is emitted.
    set_tensor_refs(operation, {output_ref});
    return operation;
}

relation::OperationDescriptor inverse_broadcast_operation(
    const Graph& graph, const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    std::string* error) {
    if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() < 2U) {
        if (error) *error = "BLOCKED: inverse/alpha relation lacks denominator, edge weight, and outputs";
        return relation::OperationDescriptor{};
    }
    const std::string denominator_edge = required_view(
        plan_operation, domains, "input0", "", error);
    const std::string edge_weight = required_view(
        plan_operation, domains, "input1", "", error);
    const std::string inverse_edge = required_view(
        plan_operation, domains, "output0", "", error);
    const std::string alpha_edge = required_view(
        plan_operation, domains, "output1", "", error);
    if (denominator_edge.empty() || edge_weight.empty() || inverse_edge.empty() ||
        alpha_edge.empty()) return relation::OperationDescriptor{};

    // Reuse the canonical selector builder with a synthetic two-input route:
    // denominator(node) -> denominator_edge(dst[e]).  The original plan
    // remains in source_refs below, so the route descriptor cannot be
    // replaced by a prover-selected mapping.
    OperationFamily route_plan = plan_operation;
    route_plan.inputs = {"sparse_route", plan_operation.inputs.front()};
    route_plan.outputs = {denominator_edge};
    relation::OperationDescriptor operation = sparse_route_operation(
        graph, route_plan, domains, families, error);
    if (operation.id.empty()) return operation;
    operation.id = plan_operation.id;
    operation.kind = relation::RelationKind::Alpha;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping + ";public-selector-mle:dst";
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    operation.route_by_destination = true;
    operation.source_refs.clear();
    for (const auto& input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto& output : plan_operation.outputs) append_unique(operation.source_refs, output);
    // Generic Alpha constraints are all on the edge domain.  The node->edge
    // denominator broadcast is proved by the route proof above.
    set_tensor_refs(operation, {denominator_edge, edge_weight, inverse_edge, alpha_edge});
    if (!add_fac_broadcast_from_graph(operation, denominator_edge,
                                      plan_operation.inputs.front(), domains,
                                      families, graph, true, error) ||
        !add_fac_identity_from_families(operation, edge_weight, plan_operation.inputs[1],
                                        domains, families, error) ||
        !add_fac_identity_from_families(operation, inverse_edge, plan_operation.outputs[0],
                                        domains, families, error) ||
        !add_fac_identity_from_families(operation, alpha_edge, plan_operation.outputs[1],
                                        domains, families, error))
        return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    operation.constraints.clear();
    relation::ConstraintDescriptor inverse;
    inverse.id = operation.id + "/inverse-edge";
    inverse.degree = 2U;
    inverse.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
    inverse.terms.push_back(monomial(
        zkdl::fr_from_u32(1U), {local_ref(operation, denominator_edge),
                                local_ref(operation, inverse_edge)}));
    operation.constraints.push_back(std::move(inverse));
    operation.constraints.push_back(product_constraint(
        operation.id + "/alpha-edge", operation, edge_weight, inverse_edge, alpha_edge));
    operation.padding = "fixed-zero-padding;public-dst-broadcast;field-inverse-alpha";
    std::uint32_t edge_domain = 0U;
    if (!same_domain_refs(domains, operation.tensor_refs, &edge_domain, error, operation.id))
        return relation::OperationDescriptor{};
    (void)edge_domain;
    std::string validation_error;
    if (!operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: inverse/alpha descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor canonical_operation(
    const Config& config, const StaticTables& tables,
    const OperationFamily& plan_operation,
    const std::map<std::string, std::uint32_t>& domains,
    const std::map<std::string, const TensorFamily*>& families,
    const Graph* graph,
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
            plan_operation.kind == "output_bias") {
            operation.kind = plan_operation.kind == "output_bias"
                ? relation::RelationKind::Bias
                : (plan_operation.kind == "tensor_sumcheck"
                   ? relation::RelationKind::HeadSum : relation::RelationKind::Add);
            if (plan_operation.outputs.size() != 1U || plan_operation.inputs.empty())
                return unsupported("one output and aligned input MLE tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            std::vector<std::string> refs = inputs;
            refs.push_back(output);
            set_tensor_refs(operation, refs);
            const bool bias_vector = plan_operation.kind == "output_bias" && inputs.size() == 2U;
            if ((!bias_vector &&
                 !bind_aligned_views(operation, plan_operation, inputs, domains, error)) ||
                (bias_vector &&
                 (!add_fac_identity_from_families(operation, inputs[0], plan_operation.inputs[0],
                                                  domains, families, error) ||
                  !add_fac_vector_broadcast(operation, inputs[1], plan_operation.inputs[1],
                                            domains, families, error))) ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.constraints.push_back(linear_constraint(
                operation.id + "/linear", operation, inputs, output));
        } else if (plan_operation.kind == "tensor_reindex_sumcheck") {
            operation.kind = plan_operation.id.find("/concat") != std::string::npos
                ? relation::RelationKind::Concat : relation::RelationKind::Reindex;
            if (plan_operation.outputs.size() != 1U || plan_operation.inputs.empty())
                return unsupported("canonical reindex input/output views");
            std::vector<std::string> inputs;
            inputs.reserve(plan_operation.inputs.size());
            for (std::size_t index = 0U; index < plan_operation.inputs.size(); ++index) {
                const std::string input = required_view(
                    plan_operation, domains, "input" + std::to_string(index), "", error);
                if (input.empty()) return relation::OperationDescriptor{};
                inputs.push_back(input);
            }
            const std::string output = required_view(plan_operation, domains, "output", "", error);
            if (output.empty()) return relation::OperationDescriptor{};
            std::vector<std::string> refs = inputs;
            refs.push_back(output);
            set_tensor_refs(operation, refs);
            bool fac_ok = true;
            for (std::size_t index = 0U; index < inputs.size(); ++index) {
                if (operation.kind == relation::RelationKind::Concat) {
                    fac_ok = add_fac_concat_binding(operation, inputs[index],
                                                    plan_operation.inputs[index], domains,
                                                    families, static_cast<std::uint32_t>(index), error);
                } else {
                    fac_ok = add_fac_binding(operation, inputs[index],
                                             plan_operation.inputs[index], domains, error);
                }
                if (!fac_ok) break;
            }
            if (!fac_ok ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.padding = "fixed-zero-padding;explicit-fac4dnn-reindex-view";
            operation.constraints.push_back(linear_constraint(
                operation.id + "/reindex", operation, inputs, output));
        } else if (plan_operation.kind == "tensor_product_sumcheck" &&
                   (plan_operation.id.find("/consistency") != std::string::npos ||
                    plan_operation.inputs.size() >= 3U)) {
            operation.kind = relation::RelationKind::FieldIntegerConsistency;
            if (plan_operation.inputs.size() < 3U || plan_operation.outputs.size() != 1U)
                return unsupported("denominator/field/integer aggregate and residual field tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            set_tensor_refs(operation, {inputs[0], inputs[1], inputs[2], output});
            if (!bind_aligned_views(operation, plan_operation, inputs, domains, error) ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
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
            if (!bind_aligned_views(operation, plan_operation, inputs, domains, error) ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.constraints.push_back(product_constraint(
                operation.id + "/product", operation, inputs[0], inputs[1], output));
        } else if (plan_operation.kind == "field_inverse_product_sumcheck") {
            if (graph == nullptr) return unsupported("public destination route for denominator broadcast");
            operation = inverse_broadcast_operation(*graph, plan_operation, domains, families, error);
            if (operation.id.empty()) return operation;
        } else if (plan_operation.kind == "public_selector_reindex") {
            if (graph == nullptr) return unsupported("public canonical graph src/dst route descriptor");
            operation = sparse_route_operation(*graph, plan_operation, domains, families, error);
            if (operation.id.empty()) return operation;
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
            return table_operation(config, tables, plan_operation, domains, families, error);
        } else if (plan_operation.kind == "segmented_max_range_sumcheck") {
            return max_range_operation(config, plan_operation, domains, families, graph, error);
        } else if (plan_operation.kind == "segmented_sumcheck") {
            return grouped_sum_operation(plan_operation, domains, families, graph, error);
        } else if (plan_operation.kind == "signed_round_div_sumcheck") {
            return signed_round_operation(config, plan_operation, domains, families, error);
        } else if (plan_operation.kind == "tensor_matmul_sumcheck") {
            if (plan_operation.inputs.size() != 2U || plan_operation.outputs.size() != 1U)
                return unsupported("one input, model kernel, and output tensor");
            relation::PrimitiveDescriptor primitive;
            if (!padded_matrix_shape(families, plan_operation.inputs[0],
                                     plan_operation.inputs[1], plan_operation.outputs[0],
                                     primitive, error)) return relation::OperationDescriptor{};
            operation.kind = relation::RelationKind::Projection;
            operation.degree = 2U;
            const std::string output = plan_operation.outputs[0];
            set_tensor_refs(operation, {output});
            for (const auto& ref : primitive.tensor_refs) append_unique(operation.source_refs, ref);
            operation.primitives.push_back(std::move(primitive));
            operation.padding = "fixed-zero-padding;zkMatMul;shape-bound";
        } else if (plan_operation.kind == "inner_product_sumcheck") {
            if (plan_operation.inputs.size() != 3U || plan_operation.outputs.size() != 2U)
                return unsupported("src/dst routed matrices, kernels, and two edge outputs");
            const std::string src_matrix = required_view(plan_operation, domains, "src_matrix", "", error);
            const std::string dst_matrix = required_view(plan_operation, domains, "dst_matrix", "", error);
            const std::string src_kernel = required_view(plan_operation, domains, "src_kernel",
                                                         plan_operation.inputs[1], error);
            const std::string dst_kernel = required_view(plan_operation, domains, "dst_kernel",
                                                         plan_operation.inputs[2], error);
            const std::string src_output = required_view(plan_operation, domains, "src_output", "", error);
            const std::string dst_output = required_view(plan_operation, domains, "dst_output", "", error);
            if (src_matrix.empty() || dst_matrix.empty() || src_kernel.empty() || dst_kernel.empty() ||
                src_output.empty() || dst_output.empty())
                return relation::OperationDescriptor{};
            relation::PrimitiveDescriptor src_primitive, dst_primitive;
            if (!padded_matrix_shape(families, src_matrix, src_kernel, src_output,
                                     src_primitive, error) ||
                !padded_matrix_shape(families, dst_matrix, dst_kernel, dst_output,
                                     dst_primitive, error)) return relation::OperationDescriptor{};
            if (src_primitive.matmul_shape.k != 1U || dst_primitive.matmul_shape.k != 1U)
                return unsupported("scalar inner-product output shape");
            operation.kind = relation::RelationKind::AttentionInnerProduct;
            operation.degree = 2U;
            set_tensor_refs(operation, {plan_operation.outputs[0], plan_operation.outputs[1]});
            for (const auto& ref : src_primitive.tensor_refs) append_unique(operation.source_refs, ref);
            for (const auto& ref : dst_primitive.tensor_refs) append_unique(operation.source_refs, ref);
            operation.primitives.push_back(std::move(src_primitive));
            operation.primitives.push_back(std::move(dst_primitive));
            operation.padding = "fixed-zero-padding;zkMatMul;shape-bound;src-dst-route";
            if (graph == nullptr ||
                !add_fac_broadcast_from_graph(operation, src_matrix, plan_operation.inputs[0],
                                              domains, families, *graph, false, error) ||
                !add_fac_broadcast_from_graph(operation, dst_matrix, plan_operation.inputs[0],
                                              domains, families, *graph, true, error) ||
                !add_fac_identity_from_families(operation, src_output, plan_operation.outputs[0],
                                                domains, families, error) ||
                !add_fac_identity_from_families(operation, dst_output, plan_operation.outputs[1],
                                                domains, families, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
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
                                const Graph* graph,
                                std::string* error) {
    std::map<std::string, std::uint32_t> domains;
    for (const auto& tensor : tensors) domains.emplace(tensor.id, tensor.domain_size);
    std::map<std::string, const TensorFamily*> families;
    for (const auto& family : relation.tensor_families)
        if (!families.emplace(family.id, &family).second) {
            if (error) *error = "duplicate canonical tensor family: " + family.id;
            return false;
        }
    operations.clear();
    std::set<std::string> ids;
    for (const auto& plan_operation : relation.operation_families) {
        relation::OperationDescriptor operation = canonical_operation(
            config, tables, plan_operation, domains, families, graph, error);
        if (operation.id.empty()) return false;
        for (const auto& ref : operation.tensor_refs) {
            const auto found = domains.find(ref);
            if (found == domains.end()) {
                if (error) *error = "BLOCKED: native relation witness is missing tensor " + ref +
                                    " required by " + operation.id;
                return false;
            }
        }
        for (const auto& primitive : operation.primitives) {
            for (const auto& ref : primitive.tensor_refs) {
                const auto found = domains.find(ref);
                if (found == domains.end()) {
                    if (error) *error = "BLOCKED: native relation witness is missing primitive tensor " + ref +
                                        " required by " + operation.id;
                    return false;
                }
            }
            const auto& shape = primitive.matmul_shape;
            if (domains.at(primitive.tensor_refs[0]) != shape.a_size() ||
                domains.at(primitive.tensor_refs[1]) != shape.b_size() ||
                domains.at(primitive.tensor_refs[2]) != shape.c_size()) {
                if (error) *error = "BLOCKED: primitive tensor domains do not match fixed padded zkMatMul shape for " +
                                    operation.id;
                return false;
            }
        }
        if (operation.tensor_refs.empty()) return false;
        // SparseRoute is deliberately allowed to span source-node and
        // edge-view domains; its public selector-MLE proof carries both
        // canonical openings.  All other relation families still require
        // same-domain MLE views here.
        const bool has_route = operation.kind == relation::RelationKind::SparseRoute ||
                               !operation.selector_source_ref.empty();
        if (!has_route) {
            const std::uint32_t domain = domains.at(operation.tensor_refs.front());
            for (const auto& ref : operation.tensor_refs)
                if (domains.at(ref) != domain) {
                    if (error) *error = "BLOCKED: operation " + operation.id +
                                        " has non-aligned MLE domains; explicit FAC4DNN reindex witness is required";
                    return false;
                }
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
        const bool canonical_root = expected.id == "head_sum" ||
                                    expected.id == "final_average" ||
                                    expected.id == "terminal_opening";
        if ((expected.id.find('/') == std::string::npos && !canonical_root) ||
            (expected.id == "head_sum" && expected.kind != "tensor_sumcheck") ||
            (expected.id == "final_average" && expected.kind != "signed_round_div_sumcheck") ||
            (expected.id == "terminal_opening" && expected.kind != "hyrax_terminal_evaluation")) {
            if (error) *error = "BLOCKED: RelationPlan operation id is not a canonical stage/root id: " + expected.id;
            return false;
        }
        const auto found = supplied.find(expected.id);
        if (found == supplied.end()) {
            if (error) *error = "BLOCKED: native relation descriptor is missing: " + expected.id;
            return false;
        }
        const auto& actual = *found->second;
        const bool low_degree_tree = expected.kind.find("indexed_public_table") != std::string::npos ||
                                     expected.kind.find("signed_round_div") != std::string::npos ||
                                     expected.kind.find("segmented_") != std::string::npos ||
                                     expected.kind == "public_selector_reindex";
        const bool route_mapping = expected.kind == "public_selector_reindex" &&
                                   (actual.mapping == expected.mapping ||
                                    actual.mapping == expected.mapping + ";public-selector-mle:src-dst");
        const bool broadcast_mapping = expected.kind == "field_inverse_product_sumcheck" &&
                                       (actual.mapping == expected.mapping ||
                                        actual.mapping == expected.mapping + ";public-selector-mle:dst");
        if ((!low_degree_tree && actual.degree != expected.degree) ||
            (low_degree_tree && (actual.degree < expected.degree || actual.degree > 2U)) ||
            (!route_mapping && !broadcast_mapping && actual.mapping != expected.mapping) ||
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

std::uint32_t descriptor_visibility(const std::string& visibility) {
    return visibility == "public" ? 1U : (visibility == "witness" ? 3U : 0U);
}

bool shape_contract_from_plan(
    const RelationPlan& plan, const std::vector<NativeRelationWitness::TensorShape>& supplied,
    std::vector<relation::TensorDescriptor>& descriptors, std::string* error) {
    std::map<std::string, const TensorFamily*> families;
    for (const auto& family : plan.tensor_families)
        if (!families.emplace(family.id, &family).second) {
            if (error) *error = "duplicate tensor family in authenticated RelationPlan: " + family.id;
            return false;
        }
    std::set<std::string> seen;
    descriptors.clear();
    for (const auto& shape : supplied) {
        const auto found = families.find(shape.id);
        if (found == families.end() || !seen.insert(shape.id).second) {
            if (error) *error = "public tensor shape is not an authenticated RelationPlan family: " + shape.id;
            return false;
        }
        relation::TensorDescriptor descriptor;
        descriptor.id = shape.id;
        descriptor.visibility = shape.visibility;
        descriptor.domain_size = shape.domain_size;
        descriptor.logical_shape = shape.logical_shape;
        descriptor.padded_shape = shape.padded_shape;
        descriptor.mle_order = shape.mle_order;
        descriptor.stack_axis = shape.stack_axis;
        descriptor.mapping = shape.mapping;
        descriptor.padding_rule = shape.padding_rule;
        descriptor.registry_id = shape.registry_id;
        std::string descriptor_error;
        if (!descriptor.valid(&descriptor_error)) {
            if (error) *error = "public tensor shape contract rejected: " + shape.id + ": " + descriptor_error;
            return false;
        }
        const TensorFamily& family = *found->second;
        const std::uint32_t family_visibility = descriptor_visibility(family.visibility);
        if (descriptor.logical_shape != family.logical_shape ||
            descriptor.padded_shape != family.padded_shape ||
            descriptor.mle_order != family.mle_order ||
            descriptor.stack_axis != family.stack_axis ||
            descriptor.mapping != family.mapping ||
            descriptor.padding_rule != family.padding_rule ||
            (family_visibility == 1U && descriptor.visibility != 1U) ||
            (family_visibility == 3U && (descriptor.visibility != 2U && descriptor.visibility != 3U))) {
            if (error) *error = "public tensor shape/mapping/padding differs from authenticated RelationPlan: " + shape.id;
            return false;
        }
        descriptors.push_back(std::move(descriptor));
    }
    for (const auto& family : plan.tensor_families)
        if (!seen.count(family.id)) {
            if (error) *error = "public tensor shape contract omits RelationPlan family: " + family.id;
            return false;
        }
    return true;
}

bool shape_contract_from_witness(
    const RelationPlan& plan, const NativeRelationWitness& witness,
    std::vector<relation::TensorDescriptor>& descriptors, std::string* error) {
    if (witness.tensor_shapes.empty()) {
        if (error) *error = "BLOCKED: native relation witness lacks canonical tensor shape/padding descriptors";
        return false;
    }
    return shape_contract_from_plan(plan, witness.tensor_shapes, descriptors, error);
}

bool graph_public_descriptor_id(const std::string& id) {
    const auto suffix = [&](const std::string& value) {
        return id.size() >= value.size() &&
               id.compare(id.size() - value.size(), value.size(), value) == 0U;
    };
    return id == "input_features" || id == "sparse_route" || id == "group_selectors" ||
           id == "node_ptr" || id == "edge_ptr" || id == "node_edge_ptr" ||
           suffix("/view/group_new") || suffix("/view/group_end");
}

bool graph_public_values(const Graph& graph, const relation::TensorDescriptor& descriptor,
                         std::vector<FrWire>& values, std::string* error) {
    values.assign(descriptor.domain_size, FrWire{});
    const auto suffix = [&](const std::string& value) {
        return descriptor.id.size() >= value.size() &&
               descriptor.id.compare(descriptor.id.size() - value.size(), value.size(), value) == 0U;
    };
    const bool csr_boundary_view = suffix("/view/group_new") ||
                                   suffix("/view/group_end");
    const auto from_u64 = [&](std::uint64_t value, FrWire& out) {
        if (value > std::numeric_limits<std::uint32_t>::max()) return false;
        out = zkdl::fr_from_u32(static_cast<std::uint32_t>(value));
        return true;
    };
    if (descriptor.id == "input_features") {
        if (descriptor.padded_shape.size() != 2U ||
            descriptor.padded_shape[0] < graph.feature_values.rows ||
            descriptor.padded_shape[1] < graph.feature_values.cols ||
            graph.feature_values.rows != graph.nodes ||
            graph.feature_values.cols != graph.features ||
            descriptor.domain_size !=
                descriptor.padded_shape[0] * descriptor.padded_shape[1]) {
            if (error) *error = "public input_features shape is incompatible with graph feature material";
            return false;
        }
        for (std::uint64_t row = 0U; row < graph.feature_values.rows; ++row)
            for (std::uint64_t column = 0U; column < graph.feature_values.cols; ++column) {
                const std::int64_t value = graph.feature_values.at(row, column);
                values[static_cast<std::size_t>(row * descriptor.padded_shape[1] + column)] =
                    relation::fr_from_signed_i64(value);
            }
        return true;
    }
    if (descriptor.id == "sparse_route" || descriptor.id == "group_selectors") {
        if (descriptor.padded_shape.size() != 2U || descriptor.padded_shape[1] != 2U ||
            descriptor.padded_shape[0] < graph.edges || graph.src.size() != graph.edges ||
            graph.dst.size() != graph.edges) {
            if (error) *error = "public graph tensor shape is incompatible with src/dst selector material";
            return false;
        }
        for (std::uint64_t edge = 0U; edge < graph.edges; ++edge) {
            const std::uint64_t base = edge * 2U;
            if (descriptor.id == "sparse_route") {
                if (!from_u64(graph.src[edge], values[base]) ||
                    !from_u64(graph.dst[edge], values[base + 1U])) return false;
            } else {
                const std::uint32_t destination = graph.dst[edge];
                if (destination >= graph.nodes || graph.node_edge_ptr.size() != graph.nodes + 1U) {
                    if (error) *error = "public group selector graph boundary is malformed";
                    return false;
                }
                const bool first = edge == graph.node_edge_ptr[destination];
                const bool last = edge + 1U == graph.node_edge_ptr[destination + 1U];
                values[base] = zkdl::fr_from_u32(first ? 1U : 0U);
                values[base + 1U] = zkdl::fr_from_u32(last ? 1U : 0U);
            }
        }
        return true;
    }
    if (csr_boundary_view) {
        if (graph.dst.size() != graph.edges || graph.node_edge_ptr.size() != graph.nodes + 1U ||
            graph.node_edge_ptr.empty() || graph.node_edge_ptr.front() != 0U ||
            graph.node_edge_ptr.back() != graph.edges || descriptor.padded_shape.empty() ||
            descriptor.padded_shape.front() < graph.edges) {
            if (error) *error = "public CSR group-boundary view has an incompatible graph/domain";
            return false;
        }
        for (std::size_t node = 0U; node + 1U < graph.node_edge_ptr.size(); ++node) {
            if (graph.node_edge_ptr[node] > graph.node_edge_ptr[node + 1U] ||
                graph.node_edge_ptr[node + 1U] > graph.edges) {
                if (error) *error = "public CSR node_edge_ptr is not monotone or bounded";
                return false;
            }
        }
        std::uint64_t feature_width = 1U;
        for (std::size_t axis = 1U; axis < descriptor.padded_shape.size(); ++axis) {
            if (descriptor.padded_shape[axis] == 0U ||
                feature_width > std::numeric_limits<std::uint64_t>::max() /
                                    descriptor.padded_shape[axis]) {
                if (error) *error = "public CSR group-boundary view feature domain overflows";
                return false;
            }
            feature_width *= descriptor.padded_shape[axis];
        }
        if (feature_width == 0U ||
            feature_width > std::numeric_limits<std::uint32_t>::max() ||
            feature_width * descriptor.padded_shape.front() != descriptor.domain_size) {
            if (error) *error = "public CSR group-boundary view domain does not match its shape";
            return false;
        }
        const bool want_end = suffix("/view/group_end");
        for (std::uint64_t edge = 0U; edge < graph.edges; ++edge) {
            const std::uint64_t destination = graph.dst[edge];
            if (destination >= graph.nodes ||
                edge < graph.node_edge_ptr[destination] ||
                edge >= graph.node_edge_ptr[destination + 1U]) {
                if (error) *error = "public CSR edge is outside its destination segment";
                return false;
            }
            const bool boundary = want_end
                ? edge + 1U == graph.node_edge_ptr[destination + 1U]
                : edge == graph.node_edge_ptr[destination];
            for (std::uint64_t feature = 0U; feature < feature_width; ++feature)
                values[static_cast<std::size_t>(edge * feature_width + feature)] =
                    zkdl::fr_from_u32(boundary ? 1U : 0U);
        }
        return true;
    }
    const std::vector<std::uint64_t>* source = nullptr;
    if (descriptor.id == "node_ptr") source = &graph.node_ptr;
    else if (descriptor.id == "edge_ptr") source = &graph.edge_ptr;
    else if (descriptor.id == "node_edge_ptr") source = &graph.node_edge_ptr;
    if (source == nullptr) return false;
    if (source->size() > values.size()) {
        if (error) *error = "public graph boundary tensor exceeds its padded domain";
        return false;
    }
    for (std::size_t i = 0U; i < source->size(); ++i)
        if (!from_u64((*source)[i], values[i])) {
            if (error) *error = "public graph boundary exceeds field wire range";
            return false;
        }
    return true;
}

bool authenticated_table_values(const StaticTables& tables,
                                const relation::TensorDescriptor& descriptor,
                                std::vector<FrWire>& values,
                                std::string* error) {
    const std::vector<std::int64_t>* table = nullptr;
    if (descriptor.id == "static_lrelu") table = &tables.lrelu;
    else if (descriptor.id == "static_elu") table = &tables.elu;
    else if (descriptor.id == "static_exp") table = &tables.exp;
    else return false;
    if (table->empty() || table->size() > descriptor.domain_size ||
        descriptor.logical_shape.size() != 1U ||
        descriptor.logical_shape.front() != table->size() ||
        descriptor.padded_shape.size() != 1U ||
        descriptor.padded_shape.front() != descriptor.domain_size) {
        if (error) *error = "authenticated static table shape does not match RelationPlan: " + descriptor.id;
        return false;
    }
    values.assign(descriptor.domain_size, FrWire{});
    for (std::size_t row = 0U; row < table->size(); ++row)
        values[row] = relation::fr_from_signed_i64((*table)[row]);
    return true;
}

bool bind_authenticated_public_material(const StaticTables& tables,
                                        const std::vector<relation::TensorDescriptor>& descriptors,
                                        std::map<std::string, std::vector<FrWire>>& values,
                                        std::string* error) {
    for (const auto& descriptor : descriptors) {
        if (descriptor.visibility != 1U) continue;
        std::vector<FrWire> expected;
        if (descriptor.id == "input_features") {
            // Graph-derived public inputs are checked by graph_public_values at
            // the call site because the Graph object is required there.
            continue;
        }
        if (!authenticated_table_values(tables, descriptor, expected, error)) {
            if (error && !error->empty() && descriptor.id.rfind("static_", 0U) == 0U)
                return false;
            continue;
        }
        const auto found = values.find(descriptor.id);
        if (found == values.end()) values.emplace(descriptor.id, std::move(expected));
        else if (found->second != expected) {
            if (error) *error = "public static table witness differs from authenticated table descriptor: " + descriptor.id;
            return false;
        }
    }
    return true;
}

// Decode the proof-stack-independent bridge payload exactly once at the
// native-core boundary.  The runner and bridge never carry C++ proof objects;
// all operation, commitment, hiding-record, generator, and field payloads are
// canonical bytes here and are converted into the official relation types
// before any prover/verifier code is entered.
bool decode_native_witness(const NativeRelationWitnessInput& input,
                           NativeRelationWitness& output, std::string* error) {
    output = NativeRelationWitness{};
    output.registry_path = input.registry_path;
    output.srs_digest = input.srs_digest;
    output.registry_digest = input.registry_digest;
    output.srs_authenticated = input.srs_authenticated;
    output.registry_authenticated = input.registry_authenticated;

    output.tensors.reserve(input.tensors.size());
    for (const auto& supplied : input.tensors) {
        relation::TensorWitness tensor;
        tensor.id = supplied.id;
        tensor.visibility = supplied.visibility;
        tensor.domain_size = supplied.domain_size;
        tensor.values.assign(supplied.values.begin(), supplied.values.end());
        output.tensors.push_back(std::move(tensor));
    }
    output.tensor_shapes.reserve(input.tensor_shapes.size());
    for (const auto& supplied : input.tensor_shapes) {
        NativeRelationWitness::TensorShape shape;
        shape.id = supplied.id;
        shape.domain_size = supplied.domain_size;
        shape.visibility = supplied.visibility;
        shape.logical_shape = supplied.logical_shape;
        shape.padded_shape = supplied.padded_shape;
        shape.mle_order = supplied.mle_order;
        shape.stack_axis = supplied.stack_axis;
        shape.mapping = supplied.mapping;
        shape.padding_rule = supplied.padding_rule;
        shape.registry_id = supplied.registry_id;
        output.tensor_shapes.push_back(std::move(shape));
    }
    output.operations.reserve(input.operations.size());
    for (const auto& encoded : input.operations) {
        relation::OperationDescriptor operation;
        std::size_t offset = 0U;
        if (!relation::read_operation_strict(encoded, offset, operation) || offset != encoded.size()) {
            if (error) *error = "BLOCKED: native relation operation payload is not canonical";
            return false;
        }
        output.operations.push_back(std::move(operation));
    }
    output.registered_model_commitments.reserve(input.registered_model_commitments.size());
    for (const auto& supplied : input.registered_model_commitments) {
        relation::CommitmentWire commitment;
        commitment.tensor_id = supplied.tensor_id;
        commitment.visibility = supplied.visibility;
        commitment.domain_size = supplied.domain_size;
        std::size_t offset = 0U;
        if (!zkdl::read_g1(supplied.commitment, offset, commitment.commitment) ||
            offset != supplied.commitment.size()) {
            if (error) *error = "BLOCKED: native model commitment payload is not canonical";
            return false;
        }
        output.registered_model_commitments.push_back(std::move(commitment));
    }
    output.hiding_commitment_records.reserve(input.hiding_commitment_records.size());
    for (const auto& supplied : input.hiding_commitment_records) {
        zkdl::CommitmentRecord record;
        std::size_t offset = 0U;
        if (!relation::read_hiding_record(supplied.encoded, offset, record) ||
            offset != supplied.encoded.size()) {
            if (error) *error = "BLOCKED: native hiding commitment record is not canonical";
            return false;
        }
        output.hiding_commitment_records.push_back(std::move(record));
    }
    output.generators.reserve(input.generators.size());
    for (const auto& encoded : input.generators) {
        G1Jacobian_t generator{};
        std::size_t offset = 0U;
        if (!zkdl::read_g1(encoded, offset, generator) || offset != encoded.size()) {
            if (error) *error = "BLOCKED: native SRS generator payload is not canonical";
            return false;
        }
        output.generators.push_back(zkdl::cpu_g1_from_native(generator));
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
    NativeRelationWitness witness;
    if (!decode_native_witness(*request.witness, witness, error)) return false;
    if (!resource_admission(*request.config, witness, error)) return false;
    if (!statement_from_adapter(*request.config, *request.graph, *request.model,
                                *request.tables, *request.relation, *request.statement,
                                witness, context.statement, error)) return false;
    context.tensors = witness.tensors;
    if (!build_canonical_operations(*request.config, *request.tables, *request.relation,
                                    context.tensors,
                                    context.operations, request.graph, error)) return false;
    if (!shape_contract_from_witness(*request.relation, witness,
                                     context.tensor_descriptors, error)) return false;
    // If the adapter supplied descriptors, treat them only as a consistency
    // witness.  Equations used by the proof are reconstructed above from the
    // authenticated plan and never taken from this optional vector.
    if (!witness.operations.empty()) {
        if (!operation_set_matches_plan(*request.relation, witness.operations, error)) return false;
        std::map<std::string, const relation::OperationDescriptor*> supplied;
        for (const auto& operation : witness.operations) supplied.emplace(operation.id, &operation);
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
    context.generators = witness.generators;
    context.registered_model_commitments = witness.registered_model_commitments;
    context.hiding_commitment_records = witness.hiding_commitment_records;
    for (const auto& tensor : context.tensors)
        if (tensor.visibility == 1U) context.public_tensor_values.emplace(tensor.id, tensor.values);
    for (const auto& descriptor : context.tensor_descriptors) {
        const bool graph_view = graph_public_descriptor_id(descriptor.id);
        if (descriptor.visibility != 1U && !graph_view) continue;
        std::vector<FrWire> expected;
        if (!graph_public_values(*request.graph, descriptor, expected, error)) {
            if (graph_view) return false;
            continue;
        }
        if (graph_view) context.graph_public_tensor_ids.insert(descriptor.id);
        const auto found = context.public_tensor_values.find(descriptor.id);
        if (found == context.public_tensor_values.end()) {
            context.public_tensor_values.emplace(descriptor.id, std::move(expected));
        } else if (found->second != expected) {
            if (error) *error = "public graph selector tensor witness differs from Graph::src/dst/CSR";
            return false;
        }
    }
    if (!bind_authenticated_public_material(*request.tables, context.tensor_descriptors,
                                            context.public_tensor_values, error)) return false;
    for (const auto& tensor : context.tensors) {
        const relation::TensorDescriptor* descriptor = relation::find_tensor_descriptor(
            context.tensor_descriptors, tensor.id);
        if (descriptor == nullptr || descriptor->domain_size != tensor.domain_size ||
            descriptor->visibility != tensor.visibility) {
            if (error) *error = "native tensor witness does not match its authenticated shape descriptor: " + tensor.id;
            return false;
        }
    }
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
    if (!request.config || !request.graph || !request.statement || !request.public_context ||
        !request.relation || !request.tables) {
        if (error) *error = "BLOCKED: independent verifier lacks public graph/RelationPlan/context/SRS/registry";
        return false;
    }
    NativeRelationWitness supplied;
    if (!decode_native_witness(*request.public_context, supplied, error)) return false;
    if (!supplied.srs_authenticated || !supplied.registry_authenticated || supplied.generators.empty()) {
        if (error) *error = "BLOCKED: independent verifier context has no authenticated SRS/registry";
        return false;
    }
    if (request.statement->proof_scope != kProofScope || request.relation->proof_scope != kProofScope) {
        if (error) *error = "independent verifier proof scope is not full_gat_forward";
        return false;
    }
    const std::string expected_topology = graph_digest(*request.graph);
    const std::string expected_boundaries = graph_boundary_digest(*request.graph);
    if (expected_topology.empty() || request.statement->topology_id != expected_topology ||
        request.statement->topology_boundary_digest != expected_boundaries) {
        if (error) *error = "independent verifier graph src/dst or boundary digest is not statement-bound";
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
    if (supplied.tensor_shapes.empty()) {
        if (error) *error = "BLOCKED: independent verifier lacks canonical tensor shape/padding descriptors";
        return false;
    }
    if (!shape_contract_from_plan(*request.relation, supplied.tensor_shapes,
                                  context.tensor_descriptors, error)) return false;
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
        const relation::TensorDescriptor* descriptor = relation::find_tensor_descriptor(
            context.tensor_descriptors, tensor.id);
        if (descriptor == nullptr || descriptor->visibility != tensor.visibility ||
            descriptor->domain_size != tensor.domain_size) {
            if (error) *error = "independent verifier public tensor does not match shape contract: " + tensor.id;
            return false;
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
    const auto public_y = std::find_if(supplied.tensors.begin(),
                                       supplied.tensors.end(),
                                       [](const relation::TensorWitness& tensor) {
                                           return tensor.id == "public_Y" && tensor.visibility == 1U;
                                       });
    context.statement.public_output = public_y == supplied.tensors.end()
        ? std::vector<zkdl::byte>()
        : relation::encode_public_field_vector(public_y->values);
    if (context.statement.public_output.empty()) {
        if (error) *error = "BLOCKED: independent verifier lacks public_Y field witness";
        return false;
    }
    context.statement.public_output_digest = zkdl::sha256(context.statement.public_output);
    if (request.statement->public_output_digest.empty() ||
        request.statement->public_output_digest != digest_hex(context.statement.public_output_digest)) {
        if (error) *error = "independent verifier public_Y digest mismatch";
        return false;
    }
    if (!build_canonical_operations(*request.config, *request.tables, *request.relation,
                                    shape_tensors,
                                    context.operations, request.graph, error)) return false;
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
    context.registered_model_commitments = supplied.registered_model_commitments;
    context.hiding_commitment_records = supplied.hiding_commitment_records;
    for (const auto& tensor : supplied.tensors)
        context.public_tensor_values.emplace(tensor.id, tensor.values);
    for (const auto& descriptor : context.tensor_descriptors) {
        const bool graph_view = graph_public_descriptor_id(descriptor.id);
        if (descriptor.visibility != 1U && !graph_view) continue;
        std::vector<FrWire> expected;
        if (!graph_public_values(*request.graph, descriptor, expected, error)) {
            if (graph_view) return false;
            continue;
        }
        if (graph_view) context.graph_public_tensor_ids.insert(descriptor.id);
        const auto found = context.public_tensor_values.find(descriptor.id);
        if (found == context.public_tensor_values.end()) {
            context.public_tensor_values.emplace(descriptor.id, std::move(expected));
        } else if (found->second != expected) {
            if (error) *error = "independent verifier graph selector values differ from public Graph::src/dst/CSR";
            return false;
        }
    }
    if (!bind_authenticated_public_material(*request.tables, context.tensor_descriptors,
                                            context.public_tensor_values, error)) return false;
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
    const auto prove_start = std::chrono::steady_clock::now();
    relation::ProverContext context;
    if (!context_from_request(request, context, error))
        return reject(result, error, error ? *error : "native relation context rejected",
                      request.config, request.statement, request.backend);
    double commitment_open_seconds = 0.0;
    context.commitment_open_time_seconds = &commitment_open_seconds;
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
    result->envelope.status = "PROVED";
    result->envelope.static_table_descriptor_digest = request.statement->static_table_descriptor_digest;
    result->envelope.scale_chain_digest = request.statement->scale_chain_digest;
    result->envelope.integer_bound_ledger_id = request.statement->integer_bound_ledger_id;
    result->envelope.integer_bound_ledger_digest = request.statement->integer_bound_ledger_digest;
    result->envelope.integer_bound_ledger_descriptor_digest = request.statement->integer_bound_ledger_descriptor_digest;
    result->envelope.proof_size_bytes = result->proof_bytes.size();
    result->envelope.commitment_size_bytes = serialized_commitment_section_size(proof);
    result->envelope.commitment_time_seconds = commitment_open_seconds;
    result->envelope.prove_time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prove_start).count();
    result->envelope.verify_time_seconds = 0.0;
    result->envelope.verifier_reads_witness = false;
    result->envelope.gpu_kernel_executed = false;
    result->envelope.silent_fallback = false;
    result->envelope.output_bound = true;
    result->envelope.error.clear();
    return true;
}

bool zkdl_gat_native_core_verify(const NativeVerifyRequest& request,
                                 std::string* error) {
    const auto verify_start = std::chrono::steady_clock::now();
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
        request.envelope->status != "PROVED" || request.envelope->silent_fallback ||
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
    request.envelope->verify_time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - verify_start).count();
    return true;
}

}  // namespace zkdl_gat
