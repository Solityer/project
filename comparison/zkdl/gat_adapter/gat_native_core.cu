#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "gat_native_resources.hpp"
#include "gat_native_core.cuh"
#include "lookup_table_index_bits.hpp"
#include "signed_round_bounds.hpp"
#include "gat_native_resources.cu"

namespace zkdl_gat {
namespace {

using FrWire = zkdl::FrWire;

constexpr const char *kBridgeApi = "zkdl_native_gat_bridge_v1";
constexpr const char *kProofScope = "full_gat_forward";
constexpr std::uint64_t kMaxNativeDomain = 1ULL << 30U;

class NativeProgressScope {
public:
    explicit NativeProgressScope(const char *phase)
        : phase_(phase), started_(std::chrono::steady_clock::now()) {
        emit("START");
    }

    ~NativeProgressScope() {
        if (!completed_) emit("FAIL");
    }

    void pass(const std::string &detail = std::string()) {
        completed_ = true;
        emit("PASS", detail);
    }

private:
    const char *phase_;
    std::chrono::steady_clock::time_point started_;
    bool completed_ = false;

    void emit(const char *status, const std::string &detail = std::string()) const {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_).count();
        std::clog << "zkdl_native_progress phase=" << phase_ << " status=" << status
                  << " elapsed_wall_seconds=" << elapsed;
        if (!detail.empty()) std::clog << ' ' << detail;
        std::clog << '\n' << std::flush;
    }
};

std::uint32_t checked_native_u32(std::uint64_t value, const std::string &label, std::string *error = nullptr, bool allow_zero = false) {
    if ((!allow_zero && value == 0U) || value > kMaxNativeDomain || value > std::numeric_limits<std::uint32_t>::max()) {
        const std::string message = "BLOCKED: " + label + " exceeds native u32 domain bounds";
        if (error) *error = message;
        throw std::runtime_error(message);
    }
    return static_cast<std::uint32_t>(value);
}

bool checked_mul_u64(std::uint64_t left, std::uint64_t right, std::uint64_t &product) {
    if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) return false;
    product = left * right;
    return true;
}

bool checked_add_u64(std::uint64_t left, std::uint64_t right, std::uint64_t &sum) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) return false;
    sum = left + right;
    return true;
}

bool native_power_of_two(std::uint64_t value) { return value != 0U && (value & (value - 1U)) == 0U; }

std::string digest_hex(const std::array<zkdl::byte, 32> &digest) {
    static const char *hex = "0123456789abcdef";
    std::string out;
    out.reserve(64U);
    for (const auto byte : digest) {
        out.push_back(hex[(byte >> 4U) & 0x0fU]);
        out.push_back(hex[byte & 0x0fU]);
    }
    return out;
}

std::size_t serialized_commitment_section_size(const relation::Proof &proof) {
    std::vector<zkdl::byte> section;
    zkdl::append_u64(section, static_cast<std::uint64_t>(proof.commitments.size()));
    for (const auto &commitment : proof.commitments) relation::append_commitment(section, commitment);
    return section.size();
}

bool parse_digest(const std::string &text, std::array<zkdl::byte, 32> &digest) {
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

bool native_commitment_to_relation(const NativeCommitmentInput &supplied, relation::CommitmentWire &commitment, std::string *error) {
    commitment = relation::CommitmentWire{};
    commitment.tensor_id = supplied.tensor_id;
    commitment.visibility = supplied.visibility;
    commitment.domain_size = supplied.domain_size;
    commitment.segment_binding_version = supplied.segment_binding_version;
    commitment.segment_size = supplied.segment_size;
    commitment.segment_count = supplied.segment_count;
    commitment.srs_manifest_digest = supplied.srs_manifest_digest;
    commitment.basis_digest = supplied.basis_digest;
    commitment.segment_commitment_digest = supplied.segment_commitment_digest;
    std::size_t offset = 0U;
    if (!zkdl::read_g1(supplied.commitment, offset, commitment.commitment) || offset != supplied.commitment.size()) {
        if (error) *error = "BLOCKED: native model commitment payload is not canonical";
        return false;
    }
    commitment.segment_commitments.reserve(supplied.segment_commitments.size());
    for (const auto &supplied_segment : supplied.segment_commitments) {
        relation::SegmentCommitmentWire segment;
        segment.segment_index = supplied_segment.segment_index;
        segment.logical_offset = supplied_segment.logical_offset;
        segment.actual_length = supplied_segment.actual_length;
        segment.padded_length = supplied_segment.padded_length;
        segment.padding_length = supplied_segment.padding_length;
        segment.frame_digest = supplied_segment.frame_digest;
        std::size_t segment_offset = 0U;
        if (!zkdl::read_g1(supplied_segment.commitment, segment_offset, segment.commitment) || segment_offset != supplied_segment.commitment.size()) {
            if (error) *error = "BLOCKED: native model segment commitment payload is not canonical";
            return false;
        }
        commitment.segment_commitments.push_back(std::move(segment));
    }
    if (!relation::segment_commitment_wire_shape_valid(commitment, error) ||
        relation::segment_commitment_digest(commitment) != commitment.segment_commitment_digest) {
        if (error && error->empty()) *error = "BLOCKED: native model segment commitment vector is not canonical";
        return false;
    }
    return true;
}

std::string graph_digest(const Graph &graph) {
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

std::string graph_boundary_digest(const Graph &graph) {
    std::vector<zkdl::byte> encoded;
    const auto append_u64 = [&](std::uint64_t value) { zkdl::append_u64(encoded, value); };
    const std::string tag = "zkDL_gat_graph_boundaries_v1";
    append_u64(tag.size());
    encoded.insert(encoded.end(), tag.begin(), tag.end());
    const auto append_vector = [&](const std::vector<std::uint64_t> &values) {
        append_u64(values.size());
        for (const auto value : values) append_u64(value);
    };
    append_vector(graph.node_ptr);
    append_vector(graph.edge_ptr);
    append_vector(graph.node_edge_ptr);
    return digest_hex(zkdl::sha256(encoded));
}

bool read_file(const std::filesystem::path &path, std::vector<zkdl::byte> &bytes, std::string *error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error) *error = "cannot open authenticated resource: " + path.string();
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        if (error) *error = "authenticated resource has invalid size: " + path.string();
        return false;
    }
    const std::uint64_t byte_count = static_cast<std::uint64_t>(size);
    if (byte_count > (1ULL << 32) || byte_count > std::numeric_limits<std::size_t>::max() ||
        byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        if (error) *error = "authenticated resource has invalid size: " + path.string();
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(byte_count));
    if (!bytes.empty()) input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(byte_count));
    if (!input && !bytes.empty()) {
        if (error) *error = "cannot read authenticated resource: " + path.string();
        return false;
    }
    return true;
}

bool file_digest(const std::filesystem::path &path, std::array<zkdl::byte, 32> &digest, std::string *error) {
    std::vector<zkdl::byte> bytes;
    if (!read_file(path, bytes, error)) return false;
    digest = zkdl::sha256(bytes);
    return true;
}

bool streaming_file_digest(const std::filesystem::path &path, std::array<zkdl::byte, 32> &digest, std::string *error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error) *error = "cannot open authenticated scratch for hashing: " + path.string();
        return false;
    }
    zkdl::Sha256 hash;
    std::array<zkdl::byte, 1U << 16U> buffer{};
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof()) {
        if (error) *error = "cannot read authenticated scratch for hashing: " + path.string();
        return false;
    }
    digest = hash.final();
    return true;
}

bool contains(const std::vector<zkdl::byte> &bytes, const std::string &needle) {
    if (needle.empty() || bytes.size() < needle.size()) return false;
    return std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end()) != bytes.end();
}

bool manifest_string_field(const std::vector<zkdl::byte> &bytes, const std::string &key, std::string &value) {
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

    if (text.find('\\', first_quote + 1U) != std::string::npos && text.find('\\', first_quote + 1U) < last_quote) return false;
    value = text.substr(first_quote + 1U, last_quote - first_quote - 1U);
    return true;
}

bool reject(NativeProofResult *result, std::string *error, const std::string &message, const Config *config = nullptr,
            const PublicStatement *statement = nullptr, Backend backend = Backend::Cpu) {
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

bool resource_admission(const Config &config, NativeRelationWitness &witness, bool require_model_blinders, std::string *error) {
    AuthenticatedSrsResource srs;
    if (!load_authenticated_srs(config, &srs, error)) return false;
    if (!witness.srs_authenticated || witness.srs_digest != srs.manifest_digest || witness.generators.size() != srs.generators.size() ||
        witness.ipa_generators.size() != srs.generators.size()) {
        if (error) *error = "BLOCKED: native witness SRS basis is not the authenticated GATZKG1C resource";
        return false;
    }

    std::map<std::string, std::uint64_t> required_domains;
    for (const auto &tensor : witness.tensors) {
        const auto found = required_domains.find(tensor.id);
        if (found != required_domains.end() && found->second != tensor.domain_size) {
            if (error) *error = "BLOCKED: relation family has inconsistent SRS domain: " + tensor.id;
            return false;
        }
        required_domains[tensor.id] = tensor.domain_size;
    }
    for (const auto &shape : witness.tensor_shapes) {
        const auto found = required_domains.find(shape.id);
        if (found != required_domains.end() && found->second != shape.domain_size) {
            if (error) *error = "BLOCKED: tensor shape has inconsistent SRS domain: " + shape.id;
            return false;
        }
        required_domains[shape.id] = shape.domain_size;
    }
    if (required_domains.empty()) {
        if (error) *error = "BLOCKED: native witness has no relation-family SRS domain requests";
        return false;
    }
    for (const auto &requirement : required_domains) {
        const AuthenticatedSrsResource::Chunk *selected = nullptr;
        if (!select_authenticated_srs_chunk(&srs, requirement.first, requirement.second, &selected, error)) return false;
        if (!selected || selected->logical_domain < requirement.second || selected->basis_domain != srs.basis_domain ||
            selected->generators.size() < selected->basis_domain || selected->domain != srs.domain || srs.generators.size() < selected->basis_domain ||
            selected->generator_id != srs.generator_id || selected->basis_digest != srs.basis_digest ||
            selected->blinding_generator != srs.blinding_generator || selected->u_generator != srs.u_generator) {
            if (error) *error = "BLOCKED: SRS family/chunk is not compatible with the common public chunk: " + requirement.first;
            return false;
        }
        for (std::uint64_t index = 0U; index < selected->basis_domain; ++index) {
            if (selected->generators[static_cast<std::size_t>(index)] != srs.generators[static_cast<std::size_t>(index)]) {
                if (error) *error = "BLOCKED: SRS family/chunk generator prefix differs from common public chunk: " + requirement.first;
                return false;
            }
        }
    }
    if (srs.count != srs.domain || srs.generators.size() != srs.count || srs.domain == 0U || !native_power_of_two(srs.domain) ||
        (srs.domain > 1U && srs.degree < srs.domain - 1U)) {
        if (error) *error = "BLOCKED: authenticated common SRS chunk has invalid point/domain contract";
        return false;
    }
    const auto encode_point = [&](const zkdl::CpuG1Point &point, std::vector<zkdl::byte> &encoded) {
        std::array<zkdl::byte, kNativeG1WireBytes> wire{};
        std::string point_error;
        if (!zkdl_g1_wire::encode(zkdl::cpu_g1_to_native(point), wire, &point_error)) return false;
        encoded.assign(wire.begin(), wire.end());
        return true;
    };
    for (std::size_t index = 0U; index < srs.generators.size(); ++index) {
        std::vector<zkdl::byte> generator, ipa_generator;
        if (!encode_point(witness.generators[index], generator) || !encode_point(witness.ipa_generators[index], ipa_generator) ||
            generator != srs.generators[index] || ipa_generator != srs.generators[index]) {
            if (error) *error = "BLOCKED: supplied SRS/IPA generator differs from authenticated G.bin";
            return false;
        }
    }
    if ((srs.degree == 0U && srs.domain != 1U) || srs.domain == 0U || srs.count != srs.domain || !native_power_of_two(srs.domain) ||
        (srs.domain > 1U && srs.degree < srs.domain - 1U)) {
        if (error) *error = "BLOCKED: authenticated SRS degree/domain contract is invalid";
        return false;
    }
    for (const auto &tensor : witness.tensors) {
        const auto selected = std::find_if(srs.chunks.begin(), srs.chunks.end(), [&](const AuthenticatedSrsResource::Chunk &chunk) {
            return chunk.family_id == tensor.id || chunk.family_id == "*";
        });
        if (tensor.domain_size == 0U || selected == srs.chunks.end() || selected->logical_domain < tensor.domain_size ||
            (tensor.domain_size & (tensor.domain_size - 1U)) != 0U) {
            if (error) *error = "BLOCKED: native tensor domain exceeds authenticated logical SRS domain: " + tensor.id;
            return false;
        }
    }
    for (const auto &shape : witness.tensor_shapes) {
        const auto selected = std::find_if(srs.chunks.begin(), srs.chunks.end(),
                                           [&](const AuthenticatedSrsResource::Chunk &chunk) { return chunk.family_id == shape.id || chunk.family_id == "*"; });
        if (shape.domain_size == 0U || selected == srs.chunks.end() || selected->logical_domain < shape.domain_size ||
            (shape.domain_size & (shape.domain_size - 1U)) != 0U) {
            if (error) *error = "BLOCKED: native tensor shape domain exceeds authenticated logical SRS domain: " + shape.id;
            return false;
        }
    }
    std::vector<zkdl::byte> h, u;
    if (!encode_point(witness.ipa_blinding_generator, h) || !encode_point(witness.ipa_u_generator, u) || h != srs.blinding_generator || u != srs.u_generator) {
        if (error) *error = "BLOCKED: supplied IPA H/U differs from authenticated H.bin/U.bin";
        return false;
    }
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(witness.ipa_generators), witness.ipa_blinding_generator,
            witness.ipa_u_generator, witness.authenticated_ipa_basis, error))
        return false;
    AuthenticatedModelRegistry registry;
    if (!load_authenticated_model_registry(config, srs, &registry, error)) return false;
    const std::filesystem::path supplied_registry = std::filesystem::absolute(witness.registry_path).lexically_normal();
    if (!witness.registry_authenticated || witness.registry_digest != registry.digest ||
        supplied_registry != std::filesystem::absolute(registry.path).lexically_normal() ||
        witness.registered_model_commitments.size() != registry.model_commitments.size()) {
        if (error) *error = "BLOCKED: native model registry identity/commitments are not authenticated";
        return false;
    }
    std::set<std::string> supplied_ids;
    std::uint32_t supplied_generator_count = 0U;
    try {
        supplied_generator_count = checked_native_u32(witness.authenticated_ipa_basis.size(), "supplied IPA generator count");
    } catch (const std::runtime_error &exception) {
        if (error) *error = exception.what();
        return false;
    }
    const auto authenticated_basis_digest = zkdl::cpu_ipa_basis_digest(
        witness.authenticated_ipa_basis, supplied_generator_count,
        zkdl::cpu_ipa_padded_size(supplied_generator_count));
    for (const auto &expected : registry.model_commitments) {
        const auto found = std::find_if(witness.registered_model_commitments.begin(), witness.registered_model_commitments.end(),
                                        [&](const relation::CommitmentWire &actual) { return actual.tensor_id == expected.tensor_id; });
        if (found == witness.registered_model_commitments.end() || !supplied_ids.insert(found->tensor_id).second || found->visibility != 2U ||
            found->domain_size != expected.domain_size) {
            if (error) *error = "BLOCKED: native model registry commitment set is incomplete";
            return false;
        }
        relation::CommitmentWire expected_wire;
        if (!native_commitment_to_relation(expected, expected_wire, error) || !relation::commitment_equal(*found, expected_wire)) {
            if (error && error->empty()) *error = "BLOCKED: native model commitment differs from authenticated registry";
            return false;
        }
        if (expected.srs_manifest_digest != srs.manifest_digest || expected.basis_digest != authenticated_basis_digest) {
            if (error) *error = "BLOCKED: model registry segment basis is not bound to authenticated SRS";
            return false;
        }
        const AuthenticatedSrsResource::Chunk *selected = nullptr;
        if (!select_authenticated_srs_chunk(&srs, expected.tensor_id, expected.domain_size, &selected, error) || selected == nullptr) {
            if (error && error->empty()) *error = "BLOCKED: model registry segment layout is not covered by authenticated SRS";
            return false;
        }

        const std::uint64_t expected_segment_size = std::min<std::uint64_t>(expected.domain_size, srs.basis_domain);
        const std::uint64_t expected_segment_count = (expected.domain_size + expected_segment_size - 1U) / expected_segment_size;
        const bool common_chunk_valid = selected->segment_size == srs.basis_domain && selected->basis_digest == srs.basis_digest &&
                                        selected->generator_digest == digest_hex(srs.generator_digest) && selected->generators.size() >= srs.basis_domain &&
                                        selected->blinding_generator.size() == kNativeG1WireBytes && selected->u_generator.size() == kNativeG1WireBytes;
        if (expected.segment_size != expected_segment_size || expected.segment_count != expected_segment_count || !common_chunk_valid) {
            if (error && error->empty()) *error = "BLOCKED: model registry segment layout is not covered by authenticated SRS";
            return false;
        }
        const auto authenticated_frame_digest = zkdl::cpu_ipa_basis_digest(
            witness.authenticated_ipa_basis,
            checked_native_u32(expected.segment_size, "model segment generator count"),
            zkdl::cpu_ipa_padded_size(checked_native_u32(
                expected.segment_size, "model segment generator count")));
        for (std::size_t segment_index = 0U; segment_index < expected.segment_commitments.size(); ++segment_index) {
            const auto &declared = expected.segment_commitments[segment_index];
            const std::uint64_t expected_offset = static_cast<std::uint64_t>(segment_index) * expected.segment_size;
            std::uint32_t expected_actual = 0U;
            try {
                expected_actual = checked_native_u32(std::min<std::uint64_t>(expected.segment_size, expected.domain_size - expected_offset),
                                                     "expected model segment actual length");
            } catch (const std::runtime_error &exception) {
                if (error) *error = exception.what();
                return false;
            }
            if (declared.segment_index != segment_index || declared.logical_offset != expected_offset || declared.actual_length != expected_actual ||
                declared.padded_length != expected.segment_size || declared.frame_digest != authenticated_frame_digest) {
                if (error) *error = "BLOCKED: model registry segment metadata/frame is not SRS-bound";
                return false;
            }
        }
    }
    if (require_model_blinders) {
        AuthenticatedModelBlinders blinders;
        if (!load_authenticated_model_blinders(config, registry, &blinders, error)) return false;
        if (!witness.model_blinders_authenticated || witness.model_blinder_digest != blinders.digest ||
            witness.blinding_scalars.size() != blinders.entries.size()) {
            if (error) *error = "BLOCKED: native model blinder sidecar identity is not authenticated";
            return false;
        }
        std::set<std::string> blinder_ids;
        for (const auto &entry : blinders.entries) {
            const auto found = witness.blinding_scalars.find(entry.tensor_id);
            if (found == witness.blinding_scalars.end() || !blinder_ids.insert(entry.tensor_id).second || found->second != entry.blinder) {
                if (error) *error = "BLOCKED: native model blinder does not match authenticated sidecar: " + entry.tensor_id;
                return false;
            }
        }
    } else if (witness.model_blinders_present || witness.model_blinders_authenticated) {
        if (error) *error = "BLOCKED: independent verifier context carries prover-only model blinders";
        return false;
    }
    return true;
}

bool statement_from_adapter(const Config &config, const Graph &graph, const ModelBundle &model, const StaticTables &tables, const RelationPlan &plan,
                            const PublicStatement &statement, const NativeRelationWitness &witness, relation::RelationStatement &out, std::string *error) {
    if (statement.proof_scope != kProofScope || plan.proof_scope != kProofScope) {
        if (error) *error = "relation statement/proof scope mismatch";
        return false;
    }
    const std::string expected_topology = graph_digest(graph);
    const std::string expected_boundaries = graph_boundary_digest(graph);
    if (expected_topology.empty() || statement.topology_id != expected_topology || statement.topology_boundary_digest != expected_boundaries) {
        if (error) *error = "adapter graph src/dst or boundary digest is not statement-bound";
        return false;
    }
    if (!tables.authenticated || !plan.integer_bound_ledger.authenticated || statement.static_table_digest != tables.digest ||
        statement.static_table_descriptor_digest != tables.descriptor_digest || statement.scale_chain_digest != config.scale_chain_digest ||
        statement.integer_bound_ledger_digest != plan.integer_bound_ledger.digest ||
        statement.integer_bound_ledger_descriptor_digest != plan.integer_bound_ledger.descriptor_digest ||
        statement.relation_registry_digest != plan.registry_digest || statement.relation_terminal_claims_digest != plan.terminal_claims_digest ||
        statement.relation_mapping_digest != plan.mapping_digest || statement.relation_selector_digest != plan.selector_digest ||
        statement.relation_degree_digest != plan.degree_digest || statement.verifier_registry_digest != config.verifier_registry_digest) {
        if (error) *error = "adapter statement/resource identity is not bound to authenticated tables/ledger/plan";
        return false;
    }
    if (!parse_digest(statement.statement_id, out.statement_digest) || !parse_digest(model.manifest_digest, out.model_digest) ||
        !parse_digest(tables.digest, out.table_digest) || !parse_digest(plan.integer_bound_ledger.digest, out.ledger_digest) ||
        !parse_digest(config.verifier_registry_digest, out.registry_digest)) {
        if (error) *error = "adapter statement/resource identity is not canonical SHA-256";
        return false;
    }
    if (witness.srs_digest == std::array<zkdl::byte, 32>{} || witness.registry_digest != out.registry_digest) {
        if (error) *error = "native witness SRS/registry identity is missing or mismatched";
        return false;
    }
    out.srs_digest = witness.srs_digest;
    const auto public_it = std::find_if(witness.tensors.begin(), witness.tensors.end(),
                                        [](const relation::TensorWitness &tensor) { return tensor.id == "public_Y" && tensor.visibility == 1U; });
    if (public_it == witness.tensors.end() || !public_it->valid(error)) {
        if (error && error->empty()) *error = "canonical public_Y tensor witness is missing";
        return false;
    }
    out.public_output = relation::encode_public_field_vector(public_it->values);
    out.public_output_digest = zkdl::sha256(out.public_output);
    if (statement.public_output_digest.empty() || statement.public_output_digest != digest_hex(out.public_output_digest)) {
        if (error) *error = "public_Y digest is not bound to the canonical field vector";
        return false;
    }
    return true;
}

void append_unique(std::vector<std::string> &refs, const std::string &value) {
    if (std::find(refs.begin(), refs.end(), value) == refs.end()) refs.push_back(value);
}

std::uint32_t local_ref(const relation::OperationDescriptor &operation, const std::string &ref) {
    const auto found = std::find(operation.tensor_refs.begin(), operation.tensor_refs.end(), ref);
    if (found == operation.tensor_refs.end()) throw std::invalid_argument("relation descriptor omits referenced tensor: " + ref);
    const auto distance = std::distance(operation.tensor_refs.begin(), found);
    if (distance < 0) throw std::invalid_argument("relation tensor index is negative: " + ref);
    return checked_native_u32(static_cast<std::uint64_t>(distance), "relation tensor index", nullptr, true);
}

relation::Monomial monomial(const FrWire &coefficient, std::initializer_list<std::uint32_t> refs) {
    relation::Monomial term;
    term.coefficient = coefficient;
    term.refs.assign(refs.begin(), refs.end());
    return term;
}

relation::ConstraintDescriptor linear_constraint(const std::string &id, const relation::OperationDescriptor &operation, const std::vector<std::string> &inputs,
                                                 const std::string &output) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 1U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(one, {local_ref(operation, output)}));
    for (const auto &input : inputs) constraint.terms.push_back(monomial(zkdl::fr_neg(one), {local_ref(operation, input)}));
    return constraint;
}

relation::ConstraintDescriptor product_constraint(const std::string &id, const relation::OperationDescriptor &operation, const std::string &left,
                                                  const std::string &right, const std::string &output) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(one, {local_ref(operation, output)}));
    constraint.terms.push_back(monomial(zkdl::fr_neg(one), {local_ref(operation, left), local_ref(operation, right)}));
    return constraint;
}

std::string view_or_ref(const OperationFamily &plan_operation, const std::map<std::string, std::uint32_t> &domains, const std::string &role,
                        const std::string &fallback) {
    const std::string candidate = plan_operation.id + "/view/" + role;
    return domains.count(candidate) != 0U ? candidate : fallback;
}

// Canonical public selector arrays are immutable for one authenticated
// RelationPlan/graph build.  Reuse their expanded values across operation
// descriptors while leaving every descriptor's dimensions and wire encoding
// independent.  This pool is stack-owned by build_canonical_operations, so
// entries cannot outlive the authenticated context that supplied their key.
class DescriptorBackingInterner {
public:
    DescriptorBackingInterner(const std::string &registry_digest, const Graph *graph,
                              bool analysis_only = false)
        : registry_digest_(registry_digest), graph_(graph), analysis_only_(analysis_only) {}

    bool analysis_only() const noexcept { return analysis_only_; }

    void begin_shape_operation() noexcept {
        shape_operation_wire_payload_delta_ = 0U;
    }

    std::uint64_t shape_operation_wire_payload_delta() const noexcept {
        return shape_operation_wire_payload_delta_;
    }

    std::uint64_t shape_fac_binding_count() const noexcept {
        return shape_fac_binding_count_;
    }

    std::uint64_t shape_route_count() const noexcept {
        return shape_route_count_;
    }

    std::uint64_t expected_unique_outer_map_bytes() const noexcept {
        return expected_unique_outer_map_bytes_;
    }

    std::uint64_t expected_unique_active_mask_bytes() const noexcept {
        return expected_unique_active_mask_bytes_;
    }

    std::uint64_t expected_feature_map_bytes() const noexcept {
        return expected_feature_map_bytes_;
    }

    std::uint64_t received_selector_array_bytes() const noexcept {
        return received_selector_array_bytes_;
    }

    std::uint64_t shape_identity_outer_entries() const noexcept { return shape_identity_outer_entries_; }
    std::uint64_t shape_logical_outer_entries() const noexcept { return shape_logical_outer_entries_; }
    std::uint64_t shape_graph_route_entries() const noexcept { return shape_graph_route_entries_; }
    std::uint64_t shape_concat_outer_entries() const noexcept { return shape_concat_outer_entries_; }
    std::uint64_t shape_prefix_active_entries() const noexcept { return shape_prefix_active_entries_; }
    std::uint64_t shape_unique_outer_map_count() const noexcept { return shape_outer_map_keys_.size(); }
    std::uint64_t shape_unique_active_mask_count() const noexcept { return shape_active_mask_keys_.size(); }

    bool record_fac_binding_shape(std::uint32_t outer_domain,
                                  std::uint32_t feature_domain,
                                  std::string *error) {
        std::uint64_t outer_u32_bytes = 0U, feature_u32_bytes = 0U;
        std::uint64_t active_wire_bytes = 0U, received_outer_bytes = 0U;
        if (!checked_mul_u64(outer_domain, sizeof(std::uint32_t), outer_u32_bytes) ||
            !checked_mul_u64(feature_domain, sizeof(std::uint32_t), feature_u32_bytes) ||
            !checked_mul_u64(outer_domain, sizeof(std::uint8_t), active_wire_bytes) ||
            !checked_add_u64(outer_u32_bytes, active_wire_bytes, received_outer_bytes) ||
            !checked_add_u64(received_outer_bytes, feature_u32_bytes, received_outer_bytes) ||
            !add_shape_wire_delta(outer_u32_bytes, error) ||
            !add_shape_wire_delta(outer_u32_bytes, error) ||
            !add_shape_wire_delta(feature_u32_bytes, error) ||
            !checked_add_u64(expected_feature_map_bytes_, feature_u32_bytes,
                             expected_feature_map_bytes_) ||
            !checked_add_u64(received_selector_array_bytes_, received_outer_bytes,
                             received_selector_array_bytes_) ||
            !increment_shape_counter(shape_fac_binding_count_, error)) {
            if (error && error->empty()) *error = "BLOCKED: analysis-only descriptor shape footprint overflows";
            return false;
        }
        return true;
    }

    bool record_route_shape(std::uint32_t edge_domain,
                            std::uint32_t output_feature_domain,
                            std::string *error) {
        std::uint64_t edge_u32_bytes = 0U, feature_u32_bytes = 0U;
        std::uint64_t received_bytes = 0U, edge_maps_wire_bytes = 0U;
        if (!checked_mul_u64(edge_domain, sizeof(std::uint32_t), edge_u32_bytes) ||
            !checked_mul_u64(output_feature_domain, sizeof(std::uint32_t), feature_u32_bytes) ||
            !checked_mul_u64(edge_u32_bytes, 2U, edge_maps_wire_bytes) ||
            !checked_add_u64(edge_maps_wire_bytes, edge_domain, received_bytes) ||
            !checked_add_u64(received_bytes, feature_u32_bytes, received_bytes) ||
            !add_shape_wire_delta(edge_maps_wire_bytes, error) ||
            !add_shape_wire_delta(edge_u32_bytes, error) ||
            !add_shape_wire_delta(feature_u32_bytes, error) ||
            !checked_add_u64(expected_feature_map_bytes_, feature_u32_bytes,
                             expected_feature_map_bytes_) ||
            !checked_add_u64(received_selector_array_bytes_, received_bytes,
                             received_selector_array_bytes_) ||
            !increment_shape_counter(shape_route_count_, error)) {
            if (error && error->empty()) *error = "BLOCKED: analysis-only route shape footprint overflows";
            return false;
        }
        return true;
    }

    bool logical_outer(const TensorFamily &family, std::uint32_t outer_domain, bool source_zero,
                       relation::IndexMapU32 &outer_map, relation::CowByteVector &active,
                       std::string *error) {
        if (registry_digest_.empty() || family.logical_shape.empty() || family.padded_shape.empty() ||
            family.logical_shape.size() != family.padded_shape.size() ||
            family.padded_shape.front() != outer_domain || family.logical_shape.front() == 0U ||
            family.logical_shape.front() > outer_domain) {
            if (error) *error = "BLOCKED: FAC4DNN logical outer selector shape is not authenticated: " + family.id;
            return false;
        }
        const auto key = std::make_tuple(family.logical_shape.front(), outer_domain, source_zero);
        const auto found = logical_outer_.find(key);
        if (found != logical_outer_.end()) {
            outer_map = found->second.outer_map;
            active = found->second.active;
            return true;
        }
        const bool full_identity = !source_zero && family.logical_shape.front() == outer_domain;
        if (full_identity) outer_map.set_identity(outer_domain);
        else if (!analysis_only_) outer_map.assign(outer_domain, 0U);
        if (analysis_only_ && !full_identity &&
            !record_unique_outer_map("logical:" + std::to_string(family.logical_shape.front()) + ":" +
                                     std::to_string(outer_domain) + ":" + (source_zero ? "zero" : "prefix"),
                                     outer_domain, error)) return false;
        if (!prefix_active(static_cast<std::uint32_t>(family.logical_shape.front()), outer_domain, active, error)) return false;
        if (!analysis_only_) {
            for (std::uint32_t row = 0U; row < family.logical_shape.front(); ++row) {
                if (!source_zero && !full_identity) outer_map[row] = row;
            }
        }
        logical_outer_.emplace(key, OuterSelectorBacking{outer_map, active});
        if (analysis_only_ && !increment_shape_counter(shape_logical_outer_entries_, error)) return false;
        return true;
    }

    bool identity_outer(std::uint32_t outer_domain, relation::IndexMapU32 &outer_map,
                        relation::CowByteVector &active, std::string *error) {
        if (registry_digest_.empty() || outer_domain == 0U) {
            if (error) *error = "BLOCKED: FAC4DNN identity selector lacks an authenticated domain";
            return false;
        }
        const auto found = identity_outer_.find(outer_domain);
        if (found != identity_outer_.end()) {
            outer_map = found->second.outer_map;
            active = found->second.active;
            return true;
        }
        outer_map.set_identity(outer_domain);
        if (!prefix_active(outer_domain, outer_domain, active, error)) return false;
        identity_outer_.emplace(outer_domain, OuterSelectorBacking{outer_map, active});
        if (analysis_only_ && !increment_shape_counter(shape_identity_outer_entries_, error)) return false;
        return true;
    }

    bool graph_route(const Graph &graph, std::uint32_t edge_domain, bool destination,
                     relation::IndexMapU32 &outer_map, relation::CowByteVector &active,
                     std::string *error) {
        if (registry_digest_.empty() || &graph != graph_ || graph.nodes == 0U || graph.edges == 0U ||
            (!analysis_only_ && (graph.src.size() != graph.edges || graph.dst.size() != graph.edges)) ||
            graph.edges > edge_domain) {
            if (error) *error = "BLOCKED: graph selector interner is outside the authenticated graph context";
            return false;
        }
        const auto key = std::make_pair(edge_domain, destination);
        const auto found = graph_routes_.find(key);
        if (found != graph_routes_.end()) {
            outer_map = found->second;
            return prefix_active(static_cast<std::uint32_t>(graph.edges), edge_domain, active, error);
        }
        if (analysis_only_) {
            if (!record_unique_outer_map(std::string("graph:") + std::to_string(edge_domain) +
                                         (destination ? ":dst" : ":src"), edge_domain, error) ||
                !prefix_active(static_cast<std::uint32_t>(graph.edges), edge_domain, active, error)) return false;
            graph_routes_.emplace(key, outer_map);
            if (!increment_shape_counter(shape_graph_route_entries_, error)) return false;
            return true;
        }
        std::vector<std::uint32_t> indices(edge_domain, 0U);
        if (!prefix_active(static_cast<std::uint32_t>(graph.edges), edge_domain, active, error)) return false;
        for (std::uint64_t edge = 0U; edge < graph.edges; ++edge) {
            if (graph.src[edge] >= graph.nodes || graph.dst[edge] >= graph.nodes) {
                if (error) *error = "BLOCKED: FAC4DNN graph route index is out of bounds";
                return false;
            }
            indices[edge] = checked_native_u32(destination ? graph.dst[edge] : graph.src[edge],
                                               "FAC graph-route", error, true);
        }
        outer_map.assign(std::move(indices));
        graph_routes_.emplace(key, outer_map);
        return true;
    }

    bool concat_outer(std::uint32_t outer_domain, std::uint32_t head_domain, std::uint32_t head,
                      relation::IndexMapU32 &outer_map, relation::CowByteVector &active,
                      std::string *error) {
        if (registry_digest_.empty() || outer_domain == 0U || head_domain == 0U || head >= head_domain) {
            if (error) *error = "BLOCKED: FAC4DNN concat selector dimensions are invalid";
            return false;
        }
        const auto key = std::make_tuple(outer_domain, head_domain, head);
        const auto found = concat_outer_.find(key);
        if (found != concat_outer_.end()) {
            outer_map = found->second.outer_map;
            active = found->second.active;
            return true;
        }
        if (analysis_only_) {
            if (head_domain > 1U &&
                !record_unique_outer_map("concat:" + std::to_string(outer_domain) + ":" +
                                         std::to_string(head_domain) + ":" + std::to_string(head),
                                         outer_domain, error)) return false;
            if (!record_unique_active_mask("concat:" + std::to_string(outer_domain) + ":" +
                                           std::to_string(head_domain) + ":" + std::to_string(head),
                                           outer_domain, error)) return false;
            concat_outer_.emplace(key, OuterSelectorBacking{outer_map, active});
            if (!increment_shape_counter(shape_concat_outer_entries_, error)) return false;
            return true;
        }
        std::vector<std::uint32_t> indices(outer_domain, 0U);
        active.assign(outer_domain, 0U);
        for (std::uint32_t slot = 0U; slot < outer_domain; ++slot) {
            if (slot % head_domain == head) {
                indices[slot] = slot / head_domain;
                active[slot] = 1U;
            }
        }
        outer_map.assign(std::move(indices));
        concat_outer_.emplace(key, OuterSelectorBacking{outer_map, active});
        return true;
    }

private:
    struct OuterSelectorBacking {
        relation::IndexMapU32 outer_map;
        relation::CowByteVector active;
    };

    bool prefix_active(std::uint32_t logical_rows, std::uint32_t outer_domain,
                       relation::CowByteVector &active, std::string *error) {
        if (logical_rows == 0U || logical_rows > outer_domain) {
            if (error) *error = "BLOCKED: selector active-prefix dimensions are invalid";
            return false;
        }
        const auto key = std::make_pair(logical_rows, outer_domain);
        const auto found = prefix_active_.find(key);
        if (found != prefix_active_.end()) {
            active = found->second;
            return true;
        }
        if (analysis_only_) {
            if (!record_unique_active_mask("prefix:" + std::to_string(logical_rows) + ":" +
                                           std::to_string(outer_domain), outer_domain, error)) return false;
            prefix_active_.emplace(key, active);
            if (!increment_shape_counter(shape_prefix_active_entries_, error)) return false;
            return true;
        }
        active.assign(outer_domain, 0U);
        for (std::uint32_t row = 0U; row < logical_rows; ++row) active[row] = 1U;
        prefix_active_.emplace(key, active);
        return true;
    }

    bool add_shape_wire_delta(std::uint64_t bytes, std::string *error) {
        if (!checked_add_u64(shape_operation_wire_payload_delta_, bytes,
                             shape_operation_wire_payload_delta_)) {
            if (error) *error = "BLOCKED: analysis-only descriptor wire delta overflows";
            return false;
        }
        return true;
    }

    bool increment_shape_counter(std::uint64_t &counter, std::string *error) {
        if (counter == std::numeric_limits<std::uint64_t>::max()) {
            if (error) *error = "BLOCKED: analysis-only descriptor counter overflows";
            return false;
        }
        ++counter;
        return true;
    }

    bool record_unique_outer_map(const std::string &key, std::uint32_t domain,
                                 std::string *error) {
        if (!analysis_only_ || !shape_outer_map_keys_.insert(key).second) return true;
        std::uint64_t bytes = 0U;
        if (!checked_mul_u64(domain, sizeof(std::uint32_t), bytes) ||
            !checked_add_u64(expected_unique_outer_map_bytes_, bytes,
                             expected_unique_outer_map_bytes_)) {
            if (error) *error = "BLOCKED: analysis-only unique outer-map capacity overflows";
            return false;
        }
        return true;
    }

    bool record_unique_active_mask(const std::string &key, std::uint32_t domain,
                                   std::string *error) {
        if (!analysis_only_ || !shape_active_mask_keys_.insert(key).second) return true;
        if (!checked_add_u64(expected_unique_active_mask_bytes_, domain,
                             expected_unique_active_mask_bytes_)) {
            if (error) *error = "BLOCKED: analysis-only unique active-mask capacity overflows";
            return false;
        }
        return true;
    }

    const std::string registry_digest_;
    const Graph *const graph_;
    const bool analysis_only_;
    std::uint64_t expected_unique_outer_map_bytes_ = 0U;
    std::uint64_t expected_unique_active_mask_bytes_ = 0U;
    std::uint64_t expected_feature_map_bytes_ = 0U;
    std::uint64_t received_selector_array_bytes_ = 0U;
    std::uint64_t shape_operation_wire_payload_delta_ = 0U;
    std::uint64_t shape_fac_binding_count_ = 0U;
    std::uint64_t shape_route_count_ = 0U;
    std::uint64_t shape_identity_outer_entries_ = 0U;
    std::uint64_t shape_logical_outer_entries_ = 0U;
    std::uint64_t shape_graph_route_entries_ = 0U;
    std::uint64_t shape_concat_outer_entries_ = 0U;
    std::uint64_t shape_prefix_active_entries_ = 0U;
    std::set<std::string> shape_outer_map_keys_;
    std::set<std::string> shape_active_mask_keys_;
    std::map<std::pair<std::uint32_t, std::uint32_t>, relation::CowByteVector> prefix_active_;
    std::map<std::uint32_t, OuterSelectorBacking> identity_outer_;
    std::map<std::tuple<std::uint64_t, std::uint32_t, bool>, OuterSelectorBacking> logical_outer_;
    std::map<std::pair<std::uint32_t, bool>, relation::IndexMapU32> graph_routes_;
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, OuterSelectorBacking> concat_outer_;
};

void add_ref(relation::OperationDescriptor &operation, const std::string &ref) {
    if (std::find(operation.tensor_refs.begin(), operation.tensor_refs.end(), ref) == operation.tensor_refs.end()) operation.tensor_refs.push_back(ref);
}

bool add_fac_binding_spec(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                          const std::map<std::string, std::uint32_t> &domains, DescriptorBackingInterner &interner,
                          std::uint32_t view_outer_domain, std::uint32_t canonical_outer_domain,
                          std::uint32_t view_feature_domain, std::uint32_t canonical_feature_domain, relation::IndexMapU32 source_outer_index,
                          relation::IndexMapU32 source_feature_index, relation::CowByteVector selector_active, std::string *error) {
    if (view_ref.empty() || canonical_ref.empty() || view_ref == canonical_ref) return true;
    const auto view = domains.find(view_ref);
    const auto canonical = domains.find(canonical_ref);
    if (view == domains.end() || canonical == domains.end()) {
        if (error) *error = "BLOCKED: FAC4DNN binding omits a canonical tensor domain: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    const auto power_of_two = [](std::uint32_t value) { return value != 0U && (value & (value - 1U)) == 0U; };
    if (!power_of_two(view_outer_domain) || !power_of_two(canonical_outer_domain) || !power_of_two(view_feature_domain) ||
        !power_of_two(canonical_feature_domain) || static_cast<std::uint64_t>(view_outer_domain) * view_feature_domain != view->second ||
        static_cast<std::uint64_t>(canonical_outer_domain) * canonical_feature_domain != canonical->second ||
        (!interner.analysis_only() &&
         (source_outer_index.size() != view_outer_domain ||
          source_feature_index.size() != view_feature_domain ||
          selector_active.size() != view_outer_domain))) {
        if (error) *error = "BLOCKED: FAC4DNN selector map shape is not canonical for " + view_ref + " -> " + canonical_ref;
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
    if (!interner.analysis_only() && !binding.valid(operation.source_refs, error)) {
        std::vector<std::string> refs = operation.source_refs;
        refs.insert(refs.end(), operation.tensor_refs.begin(), operation.tensor_refs.end());
        if (!binding.valid(refs, error)) return false;
    }
    if (interner.analysis_only() &&
        !interner.record_fac_binding_shape(view_outer_domain, view_feature_domain, error)) return false;
    append_unique(operation.source_refs, view_ref);
    append_unique(operation.source_refs, canonical_ref);
    operation.fac_bindings.push_back(std::move(binding));
    return true;
}

bool add_fac_binding(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                     const std::map<std::string, std::uint32_t> &domains, DescriptorBackingInterner &interner, std::string *error) {
    if (view_ref.empty() || canonical_ref.empty() || view_ref == canonical_ref) return true;
    const auto view = domains.find(view_ref);
    const auto canonical = domains.find(canonical_ref);
    if (view == domains.end() || canonical == domains.end()) {
        if (error) *error = "BLOCKED: FAC4DNN binding omits a canonical tensor domain: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    if (view->second != canonical->second) {
        if (error) *error = "BLOCKED: FAC4DNN cross-domain selector requires an explicit public map: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!interner.identity_outer(view->second, outer_map, active, error)) return false;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view->second, canonical->second, 1U, 1U,
                                std::move(outer_map), relation::IndexMapU32::identity(1U), std::move(active), error);
}

void sort_fac_bindings(relation::OperationDescriptor &operation) {
    std::sort(operation.fac_bindings.begin(), operation.fac_bindings.end(),
              [](const relation::FacBindingDescriptor &left, const relation::FacBindingDescriptor &right) {
                  return std::tie(left.view_ref, left.canonical_ref) < std::tie(right.view_ref, right.canonical_ref);
              });
}

bool bind_aligned_views(relation::OperationDescriptor &operation, const OperationFamily &plan_operation, const std::vector<std::string> &views,
                        const std::map<std::string, std::uint32_t> &domains, DescriptorBackingInterner &interner, std::string *error) {
    if (views.size() != plan_operation.inputs.size()) {
        if (error) *error = "BLOCKED: FAC4DNN input view count does not match RelationPlan";
        return false;
    }
    for (std::size_t i = 0U; i < views.size(); ++i)
        if (!add_fac_binding(operation, views[i], plan_operation.inputs[i], domains, interner, error)) return false;
    return true;
}

bool bind_output_view(relation::OperationDescriptor &operation, const OperationFamily &plan_operation, const std::string &view, std::size_t output_index,
                      const std::map<std::string, std::uint32_t> &domains, DescriptorBackingInterner &interner, std::string *error) {
    if (output_index >= plan_operation.outputs.size()) {
        if (error) *error = "BLOCKED: FAC4DNN output view index is outside RelationPlan";
        return false;
    }
    return add_fac_binding(operation, view, plan_operation.outputs[output_index], domains, interner, error);
}

std::vector<std::string> aligned_inputs(const OperationFamily &plan_operation, const std::map<std::string, std::uint32_t> &domains) {
    std::vector<std::string> refs;
    refs.reserve(plan_operation.inputs.size());
    for (size_t i = 0; i < plan_operation.inputs.size(); ++i)
        refs.push_back(view_or_ref(plan_operation, domains, "input" + std::to_string(i), plan_operation.inputs[i]));
    return refs;
}

std::string aligned_output(const OperationFamily &plan_operation, const std::map<std::string, std::uint32_t> &domains, size_t index = 0U) {
    if (index >= plan_operation.outputs.size()) throw std::invalid_argument("relation operation output index is out of range");
    return view_or_ref(plan_operation, domains, "output" + std::to_string(index), plan_operation.outputs[index]);
}

void set_tensor_refs(relation::OperationDescriptor &operation, const std::vector<std::string> &refs) {
    operation.tensor_refs.clear();
    for (const auto &ref : refs) add_ref(operation, ref);
}

relation::ConstraintDescriptor boolean_constraint(const std::string &id, const relation::OperationDescriptor &operation, const std::string &ref) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const std::uint32_t index = local_ref(operation, ref);
    constraint.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {index}));
    constraint.terms.push_back(monomial(zkdl::fr_from_u32(1U), {index, index}));
    return constraint;
}

relation::ConstraintDescriptor reconstruction_constraint(const std::string &id, const relation::OperationDescriptor &operation, const std::string &value,
                                                         const std::vector<std::string> &bits, std::uint32_t bit_width, std::int64_t offset = 0) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 1U;
    const FrWire one = zkdl::fr_from_u32(1U);
    const std::uint32_t value_index = local_ref(operation, value);
    constraint.terms.push_back(monomial(zkdl::fr_neg(one), {value_index}));
    if (offset != 0) constraint.constant = relation::fr_from_signed_i64(offset);
    for (std::uint32_t i = 0U; i < bit_width && i < bits.size(); ++i) {
        FrWire coefficient = zkdl::fr_from_u32(i < 31U ? (1U << i) : 0U);
        if (i >= 31U && i < 63U) { coefficient = relation::fr_from_signed_i64(static_cast<std::int64_t>(1ULL << i)); }
        constraint.terms.push_back(monomial(coefficient, {local_ref(operation, bits[i])}));
    }
    return constraint;
}

relation::ConstraintDescriptor signed_round_linear(const std::string &id, const relation::OperationDescriptor &operation, const std::string &a,
                                                   const std::string &d, const std::string &q, const std::string &r, const std::string &delta) {
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
        if (domain > std::numeric_limits<size_t>::max() / 2U) throw std::overflow_error("native MLE domain overflows size_t");
        domain <<= 1U;
        ++bits;
    }
    return bits;
}

std::uint32_t padded_domain(std::uint64_t value, std::string *error, const std::string &label) {
    if (value == 0U || value > kMaxNativeDomain) {
        if (error) *error = "BLOCKED: " + label + " has no bounded power-of-two domain";
        return 0U;
    }
    std::uint64_t domain = 1U;
    while (domain < value) domain <<= 1U;
    try {
        return checked_native_u32(domain, label, error);
    } catch (const std::runtime_error &) {
        return 0U;
    }
}

std::uint64_t flattened_feature_domain(const TensorFamily &family, std::string *error, const std::string &label) {
    if (family.padded_shape.size() < 2U || family.padded_shape.size() > 3U || family.padded_shape[0] == 0U) {
        if (error) *error = "BLOCKED: " + label + " has no canonical outer/feature shape";
        return 0U;
    }
    std::uint64_t feature = family.padded_shape[1];
    if (family.padded_shape.size() == 3U) {
        if (family.padded_shape[2] == 0U || feature > std::numeric_limits<std::uint64_t>::max() / family.padded_shape[2]) {
            if (error) *error = "BLOCKED: " + label + " feature shape overflows";
            return 0U;
        }
        feature *= family.padded_shape[2];
    }
    if (feature == 0U || feature > kMaxNativeDomain || (feature & (feature - 1U)) != 0U) {
        if (error) *error = "BLOCKED: " + label + " feature domain is not a power of two";
        return 0U;
    }
    return feature;
}

const std::vector<std::int64_t> &lookup_table(const StaticTables &tables, const OperationFamily &operation, std::int64_t &minimum, std::string *error) {
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

bool add_fac_identity_from_families(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                                    const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                                    DescriptorBackingInterner &interner, std::string *error);

bool add_fac_feature_column(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                            const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                            std::uint32_t column, DescriptorBackingInterner &interner, std::string *error);

const TensorFamily *tensor_family(const std::map<std::string, const TensorFamily *> &families, const std::string &id);

relation::OperationDescriptor sorted_lookup_operation(const Config &config, const StaticTables &tables, const OperationFamily &plan_operation,
                                                      const std::map<std::string, std::uint32_t> &domains,
                                                      const std::map<std::string, const TensorFamily *> &families,
                                                      DescriptorBackingInterner &interner, std::string *error) {
    (void)config;
    (void)tables;
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = relation::RelationKind::TableLookup;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "fixed-zero-padding;authenticated-sorted-lookup-v1";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    operation.batch_family = "authenticated_sorted_lookup";
    operation.selector_refs = plan_operation.auxiliary;
    if (plan_operation.inputs.size() != 4U || plan_operation.outputs.size() != 1U) {
        if (error) *error = "BLOCKED: sorted lookup relation must have query/table/index/bit inputs and one output";
        return {};
    }
    operation.public_table_ref = plan_operation.inputs[1];
    for (const auto &input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto &output : plan_operation.outputs) append_unique(operation.source_refs, output);
    const std::string query = view_or_ref(plan_operation, domains, "query", plan_operation.inputs.front());
    const std::string index = view_or_ref(plan_operation, domains, "index", plan_operation.inputs[2]);
    const std::string output = aligned_output(plan_operation, domains);
    if (query.empty() || index.empty() || output.empty() || !domains.count(query) || !domains.count(index) || !domains.count(output) ||
        domains.at(query) != domains.at(index) || domains.at(query) != domains.at(output)) {
        if (error) *error = "BLOCKED: sorted lookup Q/index/output domains are not aligned";
        return {};
    }
    operation.tensor_refs = {query, index, output};

    if (!add_fac_identity_from_families(operation, query, plan_operation.inputs[0], domains, families, interner, error) ||
        !add_fac_identity_from_families(operation, index, plan_operation.inputs[2], domains, families, interner, error) ||
        !add_fac_identity_from_families(operation, output, plan_operation.outputs[0], domains, families, interner, error)) {
        return {};
    }
    sort_fac_bindings(operation);
    std::string validation_error;
    if (!interner.analysis_only() && !operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: sorted lookup descriptor rejected: " + validation_error;
        return {};
    }
    return operation;
}
const TensorFamily *tensor_family(const std::map<std::string, const TensorFamily *> &families, const std::string &id) {
    const auto found = families.find(id);
    return found == families.end() ? nullptr : found->second;
}

bool fac_axes(const std::map<std::string, const TensorFamily *> &families, const std::string &ref, std::uint32_t &outer, std::uint32_t &feature,
              std::string *error) {
    const TensorFamily *family = tensor_family(families, ref);
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
    if (outer == 0U || flattened == 0U) return false;
    try {
        feature = checked_native_u32(flattened, ref + "/feature", error);
    } catch (const std::runtime_error &) {
        return false;
    }
    return true;
}

bool fac_logical_outer_mask(const TensorFamily &family, std::uint32_t outer_domain, bool source_zero,
                            relation::IndexMapU32 &outer_map, relation::CowByteVector &active,
                            DescriptorBackingInterner &interner, std::string *error) {
    return interner.logical_outer(family, outer_domain, source_zero, outer_map, active, error);
}

bool add_fac_identity_from_families(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                                    const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                                    DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error))
        return false;
    if (view_outer != canonical_outer || view_feature != canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN identity map has incompatible family axes: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!interner.identity_outer(view_outer, outer_map, active, error)) return false;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature,
                                std::move(outer_map), relation::IndexMapU32::identity(view_feature), std::move(active), error);
}

bool add_fac_broadcast_from_graph(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                                  const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                                  const Graph &graph, bool destination, DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error))
        return false;
    const std::uint32_t expected_view_outer = padded_domain(graph.edges, error, view_ref + "/edge");
    const std::uint32_t expected_source_outer = padded_domain(graph.nodes, error, canonical_ref + "/node");
    if (expected_view_outer == 0U || expected_source_outer == 0U || view_outer != expected_view_outer || canonical_outer != expected_source_outer ||
        (view_feature != canonical_feature && canonical_feature != 1U)) {
        if (error) *error = "BLOCKED: FAC4DNN graph route axes do not match public graph: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!interner.graph_route(graph, view_outer, destination, outer_map, active, error)) return false;
    std::vector<std::uint32_t> feature_map;
    if (canonical_feature == 1U) {
        if (!interner.analysis_only()) feature_map.assign(view_feature, 0U);
    } else {
        if (view_feature != canonical_feature) {
            if (error) *error = "BLOCKED: FAC4DNN graph route feature map is not canonical";
            return false;
        }
        if (!interner.analysis_only()) {
            feature_map.resize(view_feature);
            for (std::uint32_t feature = 0U; feature < view_feature; ++feature) feature_map[feature] = feature;
        }
    }
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool add_fac_feature_broadcast(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                               const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                               DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error))
        return false;
    if (view_outer != canonical_outer || canonical_feature != 1U) {
        if (error) *error = "BLOCKED: FAC4DNN feature broadcast axes are not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    const TensorFamily *view_family = tensor_family(families, view_ref);
    const TensorFamily *canonical_family = tensor_family(families, canonical_ref);
    if (view_family == nullptr || canonical_family == nullptr || view_family->logical_shape.empty() || canonical_family->logical_shape.empty() ||
        view_family->logical_shape.front() != canonical_family->logical_shape.front()) {
        if (error) *error = "BLOCKED: FAC4DNN feature broadcast logical outer rows are not aligned: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    if (view_feature > 1U && (view_family->logical_shape.size() != 2U || view_family->padded_shape.size() != 2U ||
                              view_family->logical_shape[1] != view_family->padded_shape[1])) {
        if (error) *error = "BLOCKED: FAC4DNN scalar-feature view must contain every padded feature lane: " + view_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!fac_logical_outer_mask(*view_family, view_outer, false, outer_map, active, interner, error)) return false;
    std::vector<std::uint32_t> feature_map;
    if (!interner.analysis_only()) feature_map.assign(view_feature, 0U);
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool add_fac_selector_column(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                             const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                             std::uint32_t column, DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        view_outer != canonical_outer || view_feature == 0U || canonical_feature == 0U || column >= canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN public selector-column axes are not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    const TensorFamily *view_family = tensor_family(families, view_ref);
    const TensorFamily *canonical_family = tensor_family(families, canonical_ref);
    if (view_family == nullptr || canonical_family == nullptr || view_family->logical_shape.empty() || canonical_family->logical_shape.empty() ||
        view_family->logical_shape.front() != canonical_family->logical_shape.front()) {
        if (error) *error = "BLOCKED: FAC4DNN selector-column logical outer rows are not aligned: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!fac_logical_outer_mask(*view_family, view_outer, false, outer_map, active, interner, error)) return false;
    // These are auxiliary operation views with a full padded feature axis.
    // Their producer repeats the selected public selector column at every
    // feature coordinate, including lanes that are padding for the related
    // canonical matrix.  The auxiliary view is logically [E,Fpad], so those
    // repeated selector lanes are active; canonical product/output tensors
    // retain their own zero-padding Qpad contract.
    std::vector<std::uint32_t> feature_map;
    if (!interner.analysis_only()) feature_map.assign(view_feature, column);
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool add_fac_feature_block(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                           const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                           std::uint32_t block, DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        view_outer != canonical_outer || view_feature == 0U) {
        if (error) *error = "BLOCKED: FAC4DNN feature-block axes are not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    const std::uint64_t block_start = static_cast<std::uint64_t>(block) * view_feature;
    if (block_start > canonical_feature || view_feature > static_cast<std::uint64_t>(canonical_feature) - block_start) {
        if (error) *error = "BLOCKED: FAC4DNN feature-block extent exceeds canonical feature axis: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    std::vector<std::uint32_t> feature_map;
    const TensorFamily *canonical_family = tensor_family(families, canonical_ref);
    const TensorFamily *view_family = tensor_family(families, view_ref);
    if (canonical_family == nullptr || view_family == nullptr || view_family->logical_shape.empty() || canonical_family->logical_shape.empty() ||
        view_family->logical_shape.front() != canonical_family->logical_shape.front()) {
        if (error) *error = "BLOCKED: FAC4DNN feature-block canonical family is missing: " + canonical_ref;
        return false;
    }

    if (canonical_family->mapping == "rounddiv_auxiliary:node-column-channel") {
        const bool padded_divisor_broadcast = view_family->mapping == "signed_round_div:divisor_aux:full_padded_feature_broadcast";
        if (canonical_family->padded_shape.size() != 3U || canonical_family->logical_shape.size() != 3U ||
            canonical_family->logical_shape[2U] != 8U || canonical_family->padded_shape[2U] != 8U ||
            view_family->logical_shape.size() != 2U || view_family->padded_shape.size() != 2U || canonical_family->padded_shape.back() == 0U ||
            block >= canonical_family->padded_shape.back() ||
            (padded_divisor_broadcast && (block != 6U || view_family->logical_shape[1] != view_family->padded_shape[1] ||
                                          view_family->padded_shape[1] != canonical_family->padded_shape[1] ||
                                          canonical_family->logical_shape[1] > view_family->logical_shape[1])) ||
            (!padded_divisor_broadcast && view_family->logical_shape[1] != canonical_family->logical_shape[1])) {
            if (error) *error = "BLOCKED: rounddiv_auxiliary channel descriptor is malformed: " + canonical_ref;
            return false;
        }
        std::uint32_t channels = 0U;
        try {
            channels = checked_native_u32(canonical_family->padded_shape.back(), canonical_ref + "/channels", error);
        } catch (const std::runtime_error &) {
            return false;
        }
        if (!interner.analysis_only()) {
            feature_map.resize(view_feature);
            for (std::uint32_t index = 0U; index < view_feature; ++index) {
                const std::uint32_t logical_feature = padded_divisor_broadcast && index >= canonical_family->logical_shape[1] ? 0U : index;
                const std::uint64_t mapped = static_cast<std::uint64_t>(logical_feature) * channels + block;
                if (mapped >= canonical_feature || mapped > std::numeric_limits<std::uint32_t>::max()) {
                    if (error) *error = "BLOCKED: FAC4DNN feature-block mapping exceeds canonical feature domain: " + canonical_ref;
                    return false;
                }
                feature_map[index] = checked_native_u32(mapped, canonical_ref + "/feature-map", nullptr, true);
            }
        }
    } else {
        if (!interner.analysis_only()) {
            feature_map.resize(view_feature);
            for (std::uint32_t index = 0U; index < view_feature; ++index)
                feature_map[index] = checked_native_u32(block_start + index, canonical_ref + "/feature-map", error, true);
        }
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!fac_logical_outer_mask(*view_family, view_outer, false, outer_map, active, interner, error)) return false;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool add_fac_scalar_broadcast(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                              const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                              DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        canonical_outer != 1U || canonical_feature != 1U) {
        if (error) *error = "BLOCKED: FAC4DNN scalar broadcast axes are not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    const TensorFamily *view_family = tensor_family(families, view_ref);
    const TensorFamily *canonical_family = tensor_family(families, canonical_ref);
    if (view_family == nullptr || canonical_family == nullptr || canonical_family->logical_shape != std::vector<std::uint64_t>{1U} ||
        view_family->logical_shape.empty() ||
        (view_family->logical_shape.size() == 2U && view_family->logical_shape[1] != view_family->padded_shape[1])) {
        if (error) *error = "BLOCKED: FAC4DNN scalar broadcast logical shape is not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!fac_logical_outer_mask(*view_family, view_outer, true, outer_map, active, interner, error)) return false;
    std::vector<std::uint32_t> feature_map;
    if (!interner.analysis_only()) feature_map.assign(view_feature, 0U);
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature,
                                std::move(outer_map), std::move(feature_map), std::move(active), error);
}

bool add_fac_vector_broadcast(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                              const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                              DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error)) return false;
    const TensorFamily *view_family = tensor_family(families, view_ref);
    const TensorFamily *canonical = tensor_family(families, canonical_ref);
    if (view_family == nullptr || canonical == nullptr || view_family->logical_shape.size() != 2U || canonical->logical_shape.size() != 1U ||
        canonical->padded_shape.size() != 1U || canonical->padded_shape.front() == 0U || canonical->padded_shape.front() != view_feature ||
        view_family->logical_shape[1] != canonical->logical_shape.front() || view_family->padded_shape[1] != canonical->padded_shape.front()) {
        if (error) *error = "BLOCKED: FAC4DNN vector broadcast axes are not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!fac_logical_outer_mask(*view_family, view_outer, true, outer_map, active, interner, error)) return false;
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, 1U, view_feature, view_feature, std::move(outer_map),
                                relation::IndexMapU32::identity(view_feature), std::move(active), error);
}

bool add_fac_feature_column(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                            const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                            std::uint32_t column, DescriptorBackingInterner &interner, std::string *error) {
    std::uint32_t view_outer = 0U, view_feature = 0U;
    std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
    if (!fac_axes(families, view_ref, view_outer, view_feature, error) || !fac_axes(families, canonical_ref, canonical_outer, canonical_feature, error) ||
        view_outer != canonical_outer || view_feature != 1U || canonical_feature == 0U || column >= canonical_feature) {
        if (error) *error = "BLOCKED: FAC4DNN feature-column axes are not canonical: " + view_ref + " -> " + canonical_ref;
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!interner.identity_outer(view_outer, outer_map, active, error)) return false;
    relation::IndexMapU32 feature_map;
    if (interner.analysis_only()) feature_map.clear();
    else feature_map.assign({column});
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature,
                                std::move(outer_map), std::move(feature_map), std::move(active), error);
}

bool add_fac_concat_binding(relation::OperationDescriptor &operation, const std::string &view_ref, const std::string &canonical_ref,
                            const std::map<std::string, std::uint32_t> &domains, const std::map<std::string, const TensorFamily *> &families,
                            std::uint32_t head, DescriptorBackingInterner &interner, std::string *error) {
    const TensorFamily *view = tensor_family(families, view_ref);
    const TensorFamily *canonical = tensor_family(families, canonical_ref);
    if (view == nullptr || canonical == nullptr || view->padded_shape.size() != 3U || canonical->padded_shape.size() < 2U || view->padded_shape[1] == 0U ||
        view->padded_shape[2] != canonical->padded_shape.back() || head >= view->padded_shape[1]) {
        if (error) *error = "BLOCKED: FAC4DNN concat layout is not authenticated";
        return false;
    }
    std::uint64_t flattened_outer = 0U;
    if (!checked_mul_u64(view->padded_shape[0], view->padded_shape[1], flattened_outer) || flattened_outer > kMaxNativeDomain) {
        if (error) *error = "BLOCKED: FAC4DNN concat outer shape product exceeds native domain: " + view_ref;
        return false;
    }
    const std::uint32_t view_outer = padded_domain(flattened_outer, error, view_ref + "/outer");
    const std::uint32_t canonical_outer = padded_domain(canonical->padded_shape.front(), error, canonical_ref + "/outer");
    const std::uint32_t view_feature = padded_domain(view->padded_shape[2], error, view_ref + "/feature");
    const std::uint32_t canonical_feature = padded_domain(canonical->padded_shape.back(), error, canonical_ref + "/feature");
    if (view_outer == 0U || canonical_outer == 0U || view_feature == 0U || canonical_feature == 0U || view_feature != canonical_feature) return false;
    std::uint32_t head_domain = 0U;
    try {
        head_domain = checked_native_u32(view->padded_shape[1], view_ref + "/heads", error);
    } catch (const std::runtime_error &) {
        return false;
    }
    relation::IndexMapU32 outer_map;
    relation::CowByteVector active;
    if (!interner.concat_outer(view_outer, head_domain, head, outer_map, active, error)) return false;
    std::vector<std::uint32_t> feature_map;
    if (!interner.analysis_only()) {
        feature_map.resize(view_feature);
        for (std::uint32_t feature = 0U; feature < view_feature; ++feature) feature_map[feature] = feature;
    }
    return add_fac_binding_spec(operation, view_ref, canonical_ref, domains, interner, view_outer, canonical_outer, view_feature, canonical_feature, std::move(outer_map),
                                std::move(feature_map), std::move(active), error);
}

bool padded_matrix_shape(const std::map<std::string, const TensorFamily *> &families, const std::string &input_id, const std::string &kernel_id,
                         const std::string &output_id, relation::PrimitiveDescriptor &primitive, std::string *error) {
    const TensorFamily *input = tensor_family(families, input_id);
    const TensorFamily *kernel = tensor_family(families, kernel_id);
    const TensorFamily *output = tensor_family(families, output_id);
    if (input == nullptr || kernel == nullptr || output == nullptr ||
        (input->padded_shape.size() != 1U && input->padded_shape.size() != 2U && input->padded_shape.size() != 3U) ||
        (kernel->padded_shape.size() != 1U && kernel->padded_shape.size() != 3U) || (output->padded_shape.size() != 1U && output->padded_shape.size() != 2U)) {
        if (error) *error = "BLOCKED: matrix relation lacks canonical padded input/kernel/output shapes";
        return false;
    }
    if (input->padded_shape.size() == 1U || output->padded_shape.size() == 1U) {
        if (input->padded_shape.size() != 2U && input->padded_shape.size() != 3U) {
            if (error) *error = "BLOCKED: matrix input view must expose row and inner dimensions";
            return false;
        }
    }
    const std::uint64_t kernel_inner = kernel->padded_shape.size() == 1U ? kernel->padded_shape[0] : kernel->padded_shape[1];
    const std::uint64_t kernel_output = kernel->padded_shape.size() == 1U ? 1U : kernel->padded_shape[2];
    const std::uint64_t output_rows = output->padded_shape[0];
    const std::uint64_t output_cols = output->padded_shape.size() == 1U ? 1U : output->padded_shape[1];
    if (kernel->padded_shape.size() == 3U && kernel->padded_shape[0] != 1U) {
        if (error) *error = "BLOCKED: matrix kernel batch dimension is not one";
        return false;
    }
    if (input->padded_shape.size() == 3U &&
        (input->padded_shape[1] == 0U || input->padded_shape[2] > std::numeric_limits<std::uint64_t>::max() / input->padded_shape[1])) {
        if (error) *error = "BLOCKED: canonical padded matrix inner dimension overflows";
        return false;
    }
    const std::uint64_t input_inner = input->padded_shape.size() == 2U ? input->padded_shape[1] : input->padded_shape[1] * input->padded_shape[2];
    if (input_inner != kernel_inner || output_rows != input->padded_shape[0] || output_cols != kernel_output ||
        input_inner > kMaxNativeDomain) {
        if (error) *error = "BLOCKED: canonical padded matrix inner dimension does not match kernel";
        return false;
    }

    primitive.kind = 2U;
    primitive.matmul_shape.num = 1U;
    try {
        primitive.matmul_shape.m = checked_native_u32(input->padded_shape[0], input_id + "/rows", error);
        primitive.matmul_shape.n = checked_native_u32(input_inner, input_id + "/inner", error);
        primitive.matmul_shape.k = checked_native_u32(output_cols, output_id + "/columns", error);
    } catch (const std::runtime_error &) {
        return false;
    }
    primitive.tensor_refs = {input_id, kernel_id, output_id};
    std::string shape_error;
    if (!primitive.matmul_shape.valid(&shape_error)) {
        if (error) *error = "BLOCKED: canonical padded matrix shape is invalid: " + shape_error;
        return false;
    }
    return true;
}

std::string required_view(const OperationFamily &plan_operation, const std::map<std::string, std::uint32_t> &domains, const std::string &role,
                          const std::string &fallback, std::string *error) {
    const std::string candidate = plan_operation.id + "/view/" + role;
    if (domains.count(candidate) != 0U) return candidate;
    if (!fallback.empty() && domains.count(fallback) != 0U) return fallback;
    if (error) *error = "BLOCKED: relation " + plan_operation.id + " is missing canonical view " + candidate;
    return std::string();
}

bool same_domain_refs(const std::map<std::string, std::uint32_t> &domains, const std::vector<std::string> &refs, std::uint32_t *domain, std::string *error,
                      const std::string &operation_id) {
    if (refs.empty()) {
        if (error) *error = "BLOCKED: relation " + operation_id + " has no canonical views";
        return false;
    }
    const auto first = domains.find(refs.front());
    if (first == domains.end()) {
        if (error) *error = "BLOCKED: relation " + operation_id + " is missing view " + refs.front();
        return false;
    }
    for (const auto &ref : refs) {
        const auto found = domains.find(ref);
        if (found == domains.end() || found->second != first->second) {
            if (error) *error = "BLOCKED: relation " + operation_id + " canonical views have different MLE domains";
            return false;
        }
    }
    if (domain) *domain = first->second;
    return true;
}

relation::ConstraintDescriptor recurrence_constraint(const relation::OperationDescriptor &operation, const std::string &id, const std::string &prefix,
                                                     const std::string &prefix_prev, const std::string &input, const std::string &group_new) {
    relation::ConstraintDescriptor constraint;
    constraint.id = id;
    constraint.degree = 2U;
    const FrWire one = zkdl::fr_from_u32(1U);
    constraint.terms.push_back(monomial(one, {local_ref(operation, prefix)}));
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(-1), {local_ref(operation, prefix_prev)}));
    constraint.terms.push_back(monomial(relation::fr_from_signed_i64(-1), {local_ref(operation, input)}));
    constraint.terms.push_back(monomial(one, {local_ref(operation, group_new), local_ref(operation, prefix_prev)}));
    return constraint;
}

relation::OperationDescriptor grouped_sum_operation(const OperationFamily &plan_operation, const std::map<std::string, std::uint32_t> &domains,
                                                    const std::map<std::string, const TensorFamily *> &families, const Graph *graph,
                                                    DescriptorBackingInterner &interner, std::string *error) {
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
    operation.selector_refs = plan_operation.auxiliary;
    for (const auto &ref : plan_operation.inputs) append_unique(operation.source_refs, ref);
    for (const auto &ref : plan_operation.outputs) append_unique(operation.source_refs, ref);
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
        const std::string input = required_view(plan_operation, domains, "input" + std::to_string(index),
                                                plan_operation.inputs[std::min(index, plan_operation.inputs.size() - 1U)], error);
        const std::string output = required_view(plan_operation, domains, "output" + std::to_string(index), plan_operation.outputs[index], error);
        const std::string prefix = required_view(plan_operation, domains, "prefix" + std::to_string(index), "", error);
        const std::string prefix_prev = required_view(plan_operation, domains, "prefix_prev" + std::to_string(index), "", error);
        if (input.empty() || output.empty() || prefix.empty() || prefix_prev.empty()) return relation::OperationDescriptor{};
        add_ref(operation, input);
        add_ref(operation, output);
        add_ref(operation, prefix);
        add_ref(operation, prefix_prev);
        all_refs.insert(all_refs.end(), {input, output, prefix, prefix_prev});
        operation.constraints.push_back(
            recurrence_constraint(operation, operation.id + "/prefix/" + std::to_string(index), prefix, prefix_prev, input, group_new));
        relation::ConstraintDescriptor reset;
        reset.id = operation.id + "/prefix-reset/" + std::to_string(index);
        reset.degree = 2U;
        reset.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, group_new), local_ref(operation, prefix_prev)}));
        operation.constraints.push_back(std::move(reset));
        relation::ConstraintDescriptor terminal;
        terminal.id = operation.id + "/group-terminal/" + std::to_string(index);
        terminal.degree = 2U;
        terminal.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, group_end), local_ref(operation, output)}));
        terminal.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, group_end), local_ref(operation, prefix)}));
        operation.constraints.push_back(std::move(terminal));
    }
    if (graph == nullptr || plan_operation.inputs.empty() || plan_operation.outputs.empty()) {
        if (error) *error = "BLOCKED: grouped sum lacks authenticated graph/input/output mapping";
        return relation::OperationDescriptor{};
    }
    const std::string input0 = required_view(plan_operation, domains, "input0", plan_operation.inputs[0], error);
    const std::string input1 = plan_operation.inputs.size() > 1U ? required_view(plan_operation, domains, "input1", plan_operation.inputs[1], error) :
                                                                   std::string();
    const std::string output0 = required_view(plan_operation, domains, "output0", plan_operation.outputs[0], error);
    const std::string output1 = plan_operation.outputs.size() > 1U ? required_view(plan_operation, domains, "output1", plan_operation.outputs[1], error) :
                                                                     std::string();
    if (input0.empty() || output0.empty() || (plan_operation.inputs.size() > 1U && input1.empty()) || (plan_operation.outputs.size() > 1U && output1.empty()))
        return relation::OperationDescriptor{};

    const bool input0_scalar = [&]() {
        std::uint32_t outer = 0U, feature = 0U;
        return fac_axes(families, plan_operation.inputs[0], outer, feature, nullptr) && feature == 1U;
    }();
    if ((input0_scalar ? !add_fac_feature_broadcast(operation, input0, plan_operation.inputs[0], domains, families, interner, error) :
                         !add_fac_identity_from_families(operation, input0, plan_operation.inputs[0], domains, families, interner, error)) ||
        (plan_operation.inputs.size() > 1U && !add_fac_identity_from_families(operation, input1, plan_operation.inputs[1], domains, families, interner, error)) ||
        !add_fac_broadcast_from_graph(operation, output0, plan_operation.outputs[0], domains, families, *graph, true, interner, error) ||
        (plan_operation.outputs.size() > 1U &&
         !add_fac_broadcast_from_graph(operation, output1, plan_operation.outputs[1], domains, families, *graph, true, interner, error)))
        return relation::OperationDescriptor{};
    append_unique(operation.source_refs, "group_selectors");
    if (!add_fac_selector_column(operation, group_new, "group_selectors", domains, families, 0U, interner, error) ||
        !add_fac_selector_column(operation, group_end, "group_selectors", domains, families, 1U, interner, error))
        return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    std::uint32_t domain = 0U;
    if (!same_domain_refs(domains, all_refs, &domain, error, operation.id)) return relation::OperationDescriptor{};
    (void)domain;
    std::string validation_error;
    if (!interner.analysis_only() && !operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: grouped sum descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor max_range_operation(const StaticTables &tables, const OperationFamily &plan_operation,
                                                  const std::map<std::string, std::uint32_t> &domains,
                                                  const std::map<std::string, const TensorFamily *> &families, const Graph *graph,
                                                  DescriptorBackingInterner &interner, std::string *error) {
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
    operation.selector_refs = plan_operation.auxiliary;
    for (const auto &ref : plan_operation.inputs) append_unique(operation.source_refs, ref);
    for (const auto &ref : plan_operation.outputs) append_unique(operation.source_refs, ref);
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
    for (const auto &ref : {value, maximum, selector, delta, group_new, group_end, selector_prefix, selector_prev}) {
        if (ref.empty()) return relation::OperationDescriptor{};
        add_ref(operation, ref);
    }
    std::uint32_t width = 0U;
    try {
        width = table_index_bit_width(tables.exp_row_count());
    } catch (const std::exception &) {
        if (error) *error = "BLOCKED: max/range relation lacks an authenticated ExpMap-derived delta width";
        return relation::OperationDescriptor{};
    }
    if (width == 0U || width >= 31U) {
        if (error) *error = "BLOCKED: max/range relation has an unbounded range width";
        return relation::OperationDescriptor{};
    }
    std::vector<std::string> all_refs{value, maximum, selector, delta, group_new, group_end, selector_prefix, selector_prev};
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
    selected.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, selector), local_ref(operation, delta)}));
    operation.constraints.push_back(std::move(selected));
    operation.constraints.push_back(boolean_constraint(operation.id + "/selector-boolean", operation, selector));
    operation.constraints.push_back(recurrence_constraint(operation, operation.id + "/one-hot-prefix", selector_prefix, selector_prev, selector, group_new));
    relation::ConstraintDescriptor prefix_reset;
    prefix_reset.id = operation.id + "/one-hot-reset";
    prefix_reset.degree = 2U;
    prefix_reset.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, group_new), local_ref(operation, selector_prev)}));
    operation.constraints.push_back(std::move(prefix_reset));
    relation::ConstraintDescriptor one_hot;
    one_hot.id = operation.id + "/one-hot-terminal";
    one_hot.degree = 2U;
    one_hot.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, group_end), local_ref(operation, selector_prefix)}));
    one_hot.terms.push_back(monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {local_ref(operation, group_end)}));
    operation.constraints.push_back(std::move(one_hot));
    std::vector<std::string> bits;
    bits.reserve(width);
    for (std::uint32_t bit = 0U; bit < width; ++bit) {
        const std::string ref = required_view(plan_operation, domains, "range_bit/" + std::to_string(bit), "", error);
        if (ref.empty()) return relation::OperationDescriptor{};
        bits.push_back(ref);
        add_ref(operation, ref);
        all_refs.push_back(ref);
        operation.constraints.push_back(boolean_constraint(operation.id + "/range-bit/" + std::to_string(bit), operation, ref));
    }
    operation.constraints.push_back(reconstruction_constraint(operation.id + "/range", operation, delta, bits, width));
    if (graph == nullptr) {
        if (error) *error = "BLOCKED: max/range relation lacks authenticated graph selector map";
        return relation::OperationDescriptor{};
    }
    if (!add_fac_identity_from_families(operation, value, plan_operation.inputs.front(), domains, families, interner, error) ||
        !add_fac_broadcast_from_graph(operation, maximum, plan_operation.outputs[0], domains, families, *graph, true, interner, error) ||
        !add_fac_identity_from_families(operation, selector, plan_operation.outputs[1], domains, families, interner, error) ||
        !add_fac_identity_from_families(operation, delta, plan_operation.outputs[2], domains, families, interner, error))
        return relation::OperationDescriptor{};
    const std::size_t max_suffix = operation.id.rfind("/max_range");
    const std::string canonical_range_bits = max_suffix == std::string::npos ? std::string() : operation.id.substr(0U, max_suffix) + "/trace/range_bits";
    if (canonical_range_bits.empty() || families.count(canonical_range_bits) == 0U) {
        if (error) *error = "BLOCKED: max/range canonical range_bits family is missing";
        return relation::OperationDescriptor{};
    }
    append_unique(operation.source_refs, canonical_range_bits);
    for (std::uint32_t bit = 0U; bit < width; ++bit)
        if (!add_fac_feature_column(operation, bits[bit], canonical_range_bits, domains, families, bit, interner, error)) return relation::OperationDescriptor{};
    append_unique(operation.source_refs, "group_selectors");
    if (!add_fac_selector_column(operation, group_new, "group_selectors", domains, families, 0U, interner, error) ||
        !add_fac_selector_column(operation, group_end, "group_selectors", domains, families, 1U, interner, error))
        return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    if (!same_domain_refs(domains, all_refs, nullptr, error, operation.id)) return relation::OperationDescriptor{};
    std::string validation_error;
    if (!interner.analysis_only() && !operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: max/range descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor signed_round_operation(const Config &config, const OperationFamily &plan_operation,
                                                     const IntegerBoundLedger &ledger,
                                                     const std::map<std::string, std::uint32_t> &domains,
                                                     const std::map<std::string, const TensorFamily *> &families,
                                                     DescriptorBackingInterner &interner, std::string *error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    operation.kind = plan_operation.id == "final_average" ? relation::RelationKind::FinalAverage : relation::RelationKind::SignedRoundDiv;
    operation.degree = 2U;
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;masked-from-relation";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    operation.batch_family = "signed_round_div";
    operation.selector_refs = plan_operation.auxiliary;
    for (const auto &input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto &output : plan_operation.outputs) append_unique(operation.source_refs, output);
    const std::string numerator = required_view(plan_operation, domains, "signed_numerator", plan_operation.inputs.empty() ? "" : plan_operation.inputs.front(),
                                                error);
    const std::string divisor = required_view(plan_operation, domains, "divisor", plan_operation.inputs.size() < 2U ? "" : plan_operation.inputs[1], error);
    signed_round_bounds::Widths widths;
    try {
        widths = ledger.analysis_only_planned && interner.analysis_only()
            ? signed_round_bounds::derive_planned(config, ledger, plan_operation.id)
            : signed_round_bounds::derive(config, ledger, plan_operation.id);
    } catch (const std::exception &exception) {
        if (error) *error = "BLOCKED: signed-round authenticated bounds are invalid: " + std::string(exception.what());
        return relation::OperationDescriptor{};
    }
    const std::string divisor_aux = widths.binds_raw_denominator ?
        required_view(plan_operation, domains, "divisor_aux", "", error) : std::string();
    const std::string scaled_divisor = widths.binds_raw_denominator ? divisor_aux : divisor;
    const std::string signed_q = required_view(plan_operation, domains, "signed_quotient", plan_operation.outputs.empty() ? "" : plan_operation.outputs.front(),
                                               error);
    const std::string a = required_view(plan_operation, domains, "a", "", error);
    const std::string q = required_view(plan_operation, domains, "q_abs", "", error);
    const std::string r = required_view(plan_operation, domains, "r", "", error);
    const std::string delta = required_view(plan_operation, domains, "delta", "", error);
    const std::string sigma = required_view(plan_operation, domains, "sigma", "", error);
    const std::string zero = required_view(plan_operation, domains, "z", "", error);
    const std::string a_inverse = required_view(plan_operation, domains, "a_inverse", "", error);
    const std::string divisor_inverse = required_view(plan_operation, domains, "divisor_inverse", "", error);
    for (const auto &ref : {numerator, divisor, signed_q, a, q, r, delta, sigma, zero, a_inverse, divisor_inverse}) {
        if (ref.empty()) return relation::OperationDescriptor{};
        add_ref(operation, ref);
    }
    if (widths.binds_raw_denominator) {
        if (divisor_aux.empty()) return relation::OperationDescriptor{};
        add_ref(operation, divisor_aux);
    }
    const auto has_node_feature_broadcast_divisor = [&]() {
        std::uint32_t view_outer = 0U, view_feature = 0U;
        std::uint32_t canonical_outer = 0U, canonical_feature = 0U;
        return fac_axes(families, divisor, view_outer, view_feature, nullptr) &&
               fac_axes(families, plan_operation.inputs[1], canonical_outer, canonical_feature, nullptr) &&
               view_outer == canonical_outer && canonical_feature == 1U;
    };
    if (plan_operation.inputs.size() >= 2U) {
        bool divisor_is_public_broadcast = false;
        if (domains.count(divisor) != 0U && domains.count(plan_operation.inputs[1]) != 0U && domains.at(divisor) != domains.at(plan_operation.inputs[1])) {

            if (plan_operation.id == "final_average") {
                operation.has_public_divisor = true;
                operation.public_divisor = relation::fr_from_signed_i64(static_cast<std::int64_t>(config.architecture.output_heads));
                divisor_is_public_broadcast = true;
            } else if (has_node_feature_broadcast_divisor()) {
                // Per-node denominator is explicitly broadcast into the
                // node/feature division view.  FAC binds each feature lane
                // back to that same canonical node scalar.
            } else {
                if (error) *error = "BLOCKED: SignedRoundDiv divisor view is cross-domain without a public scale route";
                return relation::OperationDescriptor{};
            }
        }
        if (!add_fac_identity_from_families(operation, numerator, plan_operation.inputs[0], domains, families, interner, error) ||
            (!divisor_is_public_broadcast && !add_fac_feature_broadcast(operation, divisor, plan_operation.inputs[1], domains, families, interner, error)) ||
            (divisor_is_public_broadcast && !add_fac_scalar_broadcast(operation, divisor, plan_operation.inputs[1], domains, families, interner, error)) ||
            !add_fac_identity_from_families(operation, signed_q, plan_operation.outputs[0], domains, families, interner, error))
            return relation::OperationDescriptor{};
    } else {

        if (plan_operation.id.find("/output_scale_alignment") == std::string::npos) {
            if (error) *error = "BLOCKED: SignedRoundDiv has no canonical divisor input";
            return relation::OperationDescriptor{};
        }
        operation.has_public_divisor = true;
        operation.public_divisor = relation::fr_from_signed_i64(static_cast<std::int64_t>(config.quantization.output_pre));
        if (!add_fac_identity_from_families(operation, numerator, plan_operation.inputs.front(), domains, families, interner, error) ||
            !add_fac_identity_from_families(operation, signed_q, plan_operation.outputs[0], domains, families, interner, error))
            return relation::OperationDescriptor{};
    }
    sort_fac_bindings(operation);
    const std::vector<std::pair<std::string, std::string>> range_values = {
        {a, "a"}, {q, "q_abs"}, {r, "r"}, {delta, "delta"}, {scaled_divisor, "divisor"},
    };
    std::map<std::string, std::vector<std::string>> range_bits;
    const std::map<std::string, std::uint32_t> range_widths = {
        {a, widths.a_bits}, {q, widths.q_bits}, {r, widths.r_bits}, {delta, widths.delta_bits}, {scaled_divisor, widths.divisor_bits},
    };
    std::uint64_t total_range_bits = 0U;
    for (const auto &entry : range_values) {
        const auto &value = entry.first;
        const auto &role = entry.second;
        const std::uint32_t width = range_widths.at(value);
        if (width == 0U || width > 63U) {
            if (error) *error = "BLOCKED: SignedRoundDiv authenticated role width is outside native bounds: " + value;
            return relation::OperationDescriptor{};
        }
        total_range_bits += width;
        for (std::uint32_t bit = 0U; bit < width; ++bit) {
            const std::string ref = required_view(plan_operation, domains, "range/" + role + "/" + std::to_string(bit), "",
                                                  error);
            if (ref.empty()) return relation::OperationDescriptor{};
            range_bits[value].push_back(ref);
            add_ref(operation, ref);
        }
    }
    if (total_range_bits > (relation::kMaxRelationConstraints - 32U) / 2U) {
        if (error) *error = "BLOCKED: SignedRoundDiv authenticated range roles exceed native relation capacity";
        return relation::OperationDescriptor{};
    }

    if (plan_operation.outputs.size() > 1U) {
        const std::string aux = plan_operation.outputs[1];
        if (!add_fac_feature_block(operation, sigma, aux, domains, families, 0U, interner, error) ||
            !add_fac_feature_block(operation, zero, aux, domains, families, 1U, interner, error) ||
            !add_fac_feature_block(operation, a, aux, domains, families, 2U, interner, error) ||
            !add_fac_feature_block(operation, q, aux, domains, families, 3U, interner, error) ||
            !add_fac_feature_block(operation, r, aux, domains, families, 4U, interner, error) ||
            !add_fac_feature_block(operation, delta, aux, domains, families, 5U, interner, error) ||
            !add_fac_feature_block(operation, scaled_divisor, aux, domains, families, 6U, interner, error))
            return relation::OperationDescriptor{};
    }
    const std::string aux_prefix = plan_operation.id;
    const std::string a_inverse_canonical = aux_prefix.find("/signed_round_div") != std::string::npos ?
                                                aux_prefix.substr(0U, aux_prefix.find("/signed_round_div")) + "/trace/rounddiv_a_inverse" :
                                                std::string();
    const std::string divisor_inverse_canonical = aux_prefix.find("/signed_round_div") != std::string::npos ?
                                                      aux_prefix.substr(0U, aux_prefix.find("/signed_round_div")) + "/trace/rounddiv_divisor_inverse" :
                                                      std::string();
    if (!a_inverse_canonical.empty() && domains.count(a_inverse_canonical) != 0U) {
        append_unique(operation.source_refs, a_inverse_canonical);
        if (!add_fac_identity_from_families(operation, a_inverse, a_inverse_canonical, domains, families, interner, error))
            return relation::OperationDescriptor{};
    }
    if (!divisor_inverse_canonical.empty() && domains.count(divisor_inverse_canonical) != 0U) {
        append_unique(operation.source_refs, divisor_inverse_canonical);
        if (!add_fac_identity_from_families(operation, divisor_inverse, divisor_inverse_canonical, domains, families, interner, error))
            return relation::OperationDescriptor{};
    }
    sort_fac_bindings(operation);
    if (widths.binds_raw_denominator) {
        relation::ConstraintDescriptor scale_relation;
        scale_relation.id = operation.id + "/scaled-divisor";
        scale_relation.degree = 1U;
        scale_relation.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, scaled_divisor)}));
        scale_relation.terms.push_back(monomial(relation::fr_from_signed_i64(-widths.denominator_scale), {local_ref(operation, divisor)}));
        operation.constraints.push_back(std::move(scale_relation));
    }
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
        public_divisor.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, divisor)}));
        operation.constraints.push_back(std::move(public_divisor));
    }
    operation.constraints.push_back(signed_round_linear(operation.id + "/nearest-ties-away", operation, a, scaled_divisor, q, r, delta));
    relation::ConstraintDescriptor slack;
    slack.id = operation.id + "/slack";
    slack.degree = 1U;
    slack.constant = zkdl::fr_from_u32(1U);
    slack.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, r)}));
    slack.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, delta)}));
    slack.terms.push_back(monomial(relation::fr_from_signed_i64(-2), {local_ref(operation, scaled_divisor)}));
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
    d_inv.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, scaled_divisor), local_ref(operation, divisor_inverse)}));
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
    for (const auto &entry : range_values) {
        const auto &value = entry.first;
        const auto &role = entry.second;
        const std::uint32_t width = range_widths.at(value);
        operation.constraints.push_back(
            reconstruction_constraint(operation.id + "/range/" + role, operation, value, range_bits[value], width));
        for (std::uint32_t bit = 0U; bit < width; ++bit)
            operation.constraints.push_back(boolean_constraint(operation.id + "/range-bit/" + role + "/" + std::to_string(bit),
                                                               operation, range_bits[value][bit]));
    }
    std::string validation_error;
    if (!interner.analysis_only() && !operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: SignedRoundDiv descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor sparse_route_operation(const Graph &graph, const OperationFamily &plan_operation,
                                                     const std::map<std::string, std::uint32_t> &domains,
                                                     const std::map<std::string, const TensorFamily *> &families,
                                                     DescriptorBackingInterner &interner, std::string *error) {
    relation::OperationDescriptor operation;
    if (plan_operation.inputs.size() != 2U || plan_operation.outputs.size() != 1U) {
        if (error) *error = "BLOCKED: sparse route requires public route input, canonical source, and one output view";
        return operation;
    }
    const std::string source_ref = plan_operation.inputs.back();
    const std::string output_ref = plan_operation.outputs.front();
    const TensorFamily *source_family = tensor_family(families, source_ref);
    const TensorFamily *output_family = tensor_family(families, output_ref);
    if (source_family == nullptr || output_family == nullptr) {
        if (error) *error = "BLOCKED: sparse route source/output family is not in the authenticated RelationPlan";
        return operation;
    }
    if (graph.nodes == 0U || graph.edges == 0U ||
        (!interner.analysis_only() && (graph.src.size() != graph.edges || graph.dst.size() != graph.edges))) {
        if (error) *error = "BLOCKED: sparse route graph has incomplete public src/dst arrays";
        return operation;
    }
    const std::uint32_t node_domain = padded_domain(source_family->padded_shape.front(), error, plan_operation.id + "/node");
    const std::uint32_t edge_domain = padded_domain(output_family->padded_shape.front(), error, plan_operation.id + "/edge");
    const auto route_feature_domain = [&](const TensorFamily &family, const std::string &label) {
        if (family.padded_shape.size() == 1U) return std::uint64_t{1U};
        return flattened_feature_domain(family, error, label);
    };
    const std::uint64_t source_feature = route_feature_domain(*source_family, plan_operation.id + "/source");
    const std::uint64_t output_feature = route_feature_domain(*output_family, plan_operation.id + "/output");
    if (node_domain == 0U || edge_domain == 0U || source_feature == 0U || output_feature == 0U || (source_feature != output_feature && source_feature != 1U) ||
        source_feature > kMaxNativeDomain || output_feature > kMaxNativeDomain || graph.nodes > node_domain ||
        graph.edges > edge_domain || domains.find(source_ref) == domains.end() || domains.find(output_ref) == domains.end() ||
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
    for (const auto &input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto &output : plan_operation.outputs) append_unique(operation.source_refs, output);
    for (const auto &auxiliary : plan_operation.auxiliary) operation.selector_refs.push_back(auxiliary);
    operation.selector_edge_domain = edge_domain;
    operation.selector_node_domain = node_domain;
    try {
        operation.selector_feature_domain = checked_native_u32(output_feature, plan_operation.id + "/output-feature", error);
    operation.selector_source_feature_domain = checked_native_u32(source_feature, plan_operation.id + "/source-feature", error);
    } catch (const std::runtime_error &) {
        return relation::OperationDescriptor{};
    }
    relation::IndexMapU32 selector_src;
    relation::IndexMapU32 selector_dst;
    relation::CowByteVector selector_active;
    if (!interner.graph_route(graph, edge_domain, false, selector_src, selector_active, error))
        return relation::OperationDescriptor{};
    relation::CowByteVector destination_active;
    if (!interner.graph_route(graph, edge_domain, true, selector_dst, destination_active, error))
        return relation::OperationDescriptor{};
    if (selector_active.backing_identity() != destination_active.backing_identity()) {
        if (error) *error = "BLOCKED: graph route src/dst selectors do not share their authenticated active mask";
        return relation::OperationDescriptor{};
    }
    if (interner.analysis_only() && !interner.record_route_shape(edge_domain, operation.selector_feature_domain, error))
        return relation::OperationDescriptor{};
    operation.selector_src = std::move(selector_src);
    operation.selector_dst = std::move(selector_dst);
    operation.selector_active = std::move(selector_active);
    if (!interner.analysis_only()) {
        operation.selector_feature_map.resize(static_cast<std::size_t>(output_feature));
        for (std::uint32_t feature = 0U; feature < output_feature; ++feature) operation.selector_feature_map[feature] = source_feature == 1U ? 0U : feature;
    }
    operation.selector_source_ref = source_ref;
    operation.selector_output_ref = output_ref;

    set_tensor_refs(operation, {output_ref});
    return operation;
}

relation::OperationDescriptor inverse_broadcast_operation(const Graph &graph, const OperationFamily &plan_operation,
                                                          const std::map<std::string, std::uint32_t> &domains,
                                                          const std::map<std::string, const TensorFamily *> &families,
                                                          DescriptorBackingInterner &interner, std::string *error) {
    if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() < 2U) {
        if (error) *error = "BLOCKED: inverse/alpha relation lacks denominator, edge weight, and outputs";
        return relation::OperationDescriptor{};
    }
    const std::string denominator_edge = required_view(plan_operation, domains, "input0", "", error);
    const std::string edge_weight = required_view(plan_operation, domains, "input1", "", error);
    const std::string inverse_edge = required_view(plan_operation, domains, "output0", "", error);
    const std::string alpha_edge = required_view(plan_operation, domains, "output1", "", error);
    if (denominator_edge.empty() || edge_weight.empty() || inverse_edge.empty() || alpha_edge.empty()) return relation::OperationDescriptor{};

    OperationFamily route_plan = plan_operation;
    route_plan.inputs = {"sparse_route", plan_operation.inputs.front()};
    route_plan.outputs = {denominator_edge};
    relation::OperationDescriptor operation = sparse_route_operation(graph, route_plan, domains, families, interner, error);
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
    for (const auto &input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto &output : plan_operation.outputs) append_unique(operation.source_refs, output);

    set_tensor_refs(operation, {denominator_edge, edge_weight, inverse_edge, alpha_edge});
    if (!add_fac_broadcast_from_graph(operation, denominator_edge, plan_operation.inputs.front(), domains, families, graph, true, interner, error) ||
        !add_fac_identity_from_families(operation, edge_weight, plan_operation.inputs[1], domains, families, interner, error) ||
        !add_fac_identity_from_families(operation, inverse_edge, plan_operation.outputs[0], domains, families, interner, error) ||
        !add_fac_identity_from_families(operation, alpha_edge, plan_operation.outputs[1], domains, families, interner, error))
        return relation::OperationDescriptor{};
    sort_fac_bindings(operation);
    operation.constraints.clear();
    relation::ConstraintDescriptor inverse;
    inverse.id = operation.id + "/inverse-edge";
    inverse.degree = 2U;
    inverse.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
    inverse.terms.push_back(monomial(zkdl::fr_from_u32(1U), {local_ref(operation, denominator_edge), local_ref(operation, inverse_edge)}));
    operation.constraints.push_back(std::move(inverse));
    operation.constraints.push_back(product_constraint(operation.id + "/alpha-edge", operation, edge_weight, inverse_edge, alpha_edge));
    operation.padding = "fixed-zero-padding;public-dst-broadcast;field-inverse-alpha";
    std::uint32_t edge_domain = 0U;
    if (!same_domain_refs(domains, operation.tensor_refs, &edge_domain, error, operation.id)) return relation::OperationDescriptor{};
    (void)edge_domain;
    std::string validation_error;
    if (!interner.analysis_only() && !operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: inverse/alpha descriptor rejected: " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

relation::OperationDescriptor canonical_operation(const Config &config, const StaticTables &tables, const IntegerBoundLedger &ledger,
                                                  const OperationFamily &plan_operation,
                                                  const std::map<std::string, std::uint32_t> &domains,
                                                  const std::map<std::string, const TensorFamily *> &families, const Graph *graph,
                                                  DescriptorBackingInterner &interner, std::string *error) {
    relation::OperationDescriptor operation;
    operation.id = plan_operation.id;
    try {
        operation.degree = checked_native_u32(plan_operation.degree, plan_operation.id + "/degree", error);
    } catch (const std::runtime_error &) {
        return relation::OperationDescriptor{};
    }
    operation.mapping = plan_operation.mapping;
    operation.terminal_claim = plan_operation.terminal_claim;
    operation.terminal_point = plan_operation.terminal_point;
    operation.padding = "all-padding-values-zero;masked-from-relation";
    operation.public_route = plan_operation.public_route;
    operation.requires_reindex = plan_operation.requires_reindex;
    for (const auto &input : plan_operation.inputs) append_unique(operation.source_refs, input);
    for (const auto &output : plan_operation.outputs) append_unique(operation.source_refs, output);
    for (const auto &input : plan_operation.inputs) append_unique(operation.tensor_refs, input);
    for (const auto &output : plan_operation.outputs) append_unique(operation.tensor_refs, output);
    for (const auto &auxiliary : plan_operation.auxiliary) operation.selector_refs.push_back(auxiliary);

    const auto unsupported = [&](const std::string &reason) {
        if (error) *error = "BLOCKED: relation family " + plan_operation.id + " requires " + reason;
        return relation::OperationDescriptor{};
    };
    try {
        const bool output_bias = plan_operation.kind == "output_bias" ||
                                 (plan_operation.id.size() >= 12U &&
                                  plan_operation.id.compare(plan_operation.id.size() - 12U, 12U, "/output_bias") == 0);
        if (plan_operation.kind == "field_addition" || plan_operation.kind == "tensor_sumcheck" || output_bias) {
            operation.kind = output_bias ?
                                 relation::RelationKind::Bias :
                                 (plan_operation.kind == "tensor_sumcheck" ? relation::RelationKind::HeadSum : relation::RelationKind::Add);
            if (plan_operation.outputs.size() != 1U || plan_operation.inputs.empty()) return unsupported("one output and aligned input MLE tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            std::vector<std::string> refs = inputs;
            refs.push_back(output);
            set_tensor_refs(operation, refs);
            const bool bias_vector = output_bias && inputs.size() == 2U;
            if ((!bias_vector && !bind_aligned_views(operation, plan_operation, inputs, domains, interner, error)) ||
                (bias_vector && (!add_fac_identity_from_families(operation, inputs[0], plan_operation.inputs[0], domains, families, interner, error) ||
                                 !add_fac_vector_broadcast(operation, inputs[1], plan_operation.inputs[1], domains, families, interner, error))) ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, interner, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.constraints.push_back(linear_constraint(operation.id + "/linear", operation, inputs, output));
        } else if (plan_operation.kind == "tensor_reindex_sumcheck") {
            operation.kind = plan_operation.id.find("/concat") != std::string::npos ? relation::RelationKind::Concat : relation::RelationKind::Reindex;
            if (plan_operation.outputs.size() != 1U || plan_operation.inputs.empty()) return unsupported("canonical reindex input/output views");
            std::vector<std::string> inputs;
            inputs.reserve(plan_operation.inputs.size());
            for (std::size_t index = 0U; index < plan_operation.inputs.size(); ++index) {
                const std::string input = required_view(plan_operation, domains, "input" + std::to_string(index), "", error);
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
                    fac_ok = add_fac_concat_binding(operation, inputs[index], plan_operation.inputs[index], domains, families,
                                                    checked_native_u32(index, operation.id + "/concat-input-index", error, true), interner, error);
                } else {
                    fac_ok = add_fac_binding(operation, inputs[index], plan_operation.inputs[index], domains, interner, error);
                }
                if (!fac_ok) break;
            }
            if (!fac_ok || !bind_output_view(operation, plan_operation, output, 0U, domains, interner, error)) return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.padding = "fixed-zero-padding;explicit-fac4dnn-reindex-view";
            operation.constraints.push_back(linear_constraint(operation.id + "/reindex", operation, inputs, output));
        } else if (plan_operation.kind == "tensor_product_sumcheck" &&
                   (plan_operation.id.find("/consistency") != std::string::npos || plan_operation.inputs.size() >= 3U)) {
            operation.kind = relation::RelationKind::FieldIntegerConsistency;
            if (plan_operation.inputs.size() < 3U || plan_operation.outputs.size() != 1U)
                return unsupported("denominator/field/integer aggregate and residual field tensors");
            const std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            set_tensor_refs(operation, {inputs[0], inputs[1], inputs[2], output});
            std::uint32_t input0_outer = 0U, input0_feature = 0U;
            const bool input0_feature_broadcast =
                fac_axes(families, plan_operation.inputs[0], input0_outer, input0_feature, error) && input0_feature == 1U;
            const bool input0_binding_ok = input0_feature_broadcast ?
                add_fac_feature_broadcast(operation, inputs[0], plan_operation.inputs[0], domains, families, interner, error) :
                add_fac_identity_from_families(operation, inputs[0], plan_operation.inputs[0], domains, families, interner, error);
            if (!input0_binding_ok ||
                !add_fac_identity_from_families(operation, inputs[1], plan_operation.inputs[1], domains, families, interner, error) ||
                !add_fac_identity_from_families(operation, inputs[2], plan_operation.inputs[2], domains, families, interner, error) ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, interner, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.constraints = relation::field_integer_consistency_constraints(operation.id, local_ref(operation, inputs[0]), local_ref(operation, inputs[1]), local_ref(operation, inputs[2]), local_ref(operation, output));
        } else if (plan_operation.kind == "tensor_product_sumcheck") {
            operation.kind = relation::RelationKind::Product;
            if (plan_operation.inputs.size() < 2U || plan_operation.outputs.size() != 1U) return unsupported("two aligned input tensors and one output tensor");
            std::vector<std::string> inputs = aligned_inputs(plan_operation, domains);
            const std::string output = aligned_output(plan_operation, domains);
            const auto ends_with = [](const std::string &value, const std::string &suffix) {
                return value.size() >= suffix.size() &&
                       value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            bool first_input_feature_broadcast = false;
            if (ends_with(plan_operation.id, "/u_h_prime") || ends_with(plan_operation.id, "/u_y_prime")) {
                const std::size_t operation_start = plan_operation.id.rfind('/');
                const std::size_t stage_end = operation_start == std::string::npos ? 0U : operation_start;
                inputs[0] = plan_operation.id.substr(0U, stage_end) + "/group_sum/view/input0";
                first_input_feature_broadcast = true;
            } else if (ends_with(plan_operation.id, "/alpha_h_prime") ||
                       ends_with(plan_operation.id, "/alpha_y_prime")) {
                inputs[0] = plan_operation.id + "/view/input0";
                first_input_feature_broadcast = true;
            }
            set_tensor_refs(operation, {inputs[0], inputs[1], output});
            const bool input_bindings_ok = first_input_feature_broadcast ?
                add_fac_feature_broadcast(operation, inputs[0], plan_operation.inputs[0], domains, families, interner, error) :
                bind_aligned_views(operation, plan_operation, inputs, domains, interner, error);
            if (!input_bindings_ok ||
                (first_input_feature_broadcast &&
                 !add_fac_identity_from_families(operation, inputs[1], plan_operation.inputs[1], domains, families, interner, error)) ||
                !bind_output_view(operation, plan_operation, output, 0U, domains, interner, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
            operation.constraints.push_back(product_constraint(operation.id + "/product", operation, inputs[0], inputs[1], output));
        } else if (plan_operation.kind == "field_inverse_product_sumcheck") {
            if (graph == nullptr) return unsupported("public destination route for denominator broadcast");
            operation = inverse_broadcast_operation(*graph, plan_operation, domains, families, interner, error);
            if (operation.id.empty()) return operation;
        } else if (plan_operation.kind == "public_selector_reindex") {
            if (graph == nullptr) return unsupported("public canonical graph src/dst route descriptor");
            operation = sparse_route_operation(*graph, plan_operation, domains, families, interner, error);
            if (operation.id.empty()) return operation;
        } else if (plan_operation.kind == "hyrax_terminal_evaluation") {
            operation.kind = relation::RelationKind::PublicOutput;
            if (plan_operation.inputs.size() != 1U || plan_operation.outputs.size() != 1U || plan_operation.inputs.front() != plan_operation.outputs.front())
                return unsupported("a single public_Y tensor");
        } else if (plan_operation.kind == "indexed_public_table_sumcheck") {
            return sorted_lookup_operation(config, tables, plan_operation, domains, families, interner, error);
        } else if (plan_operation.kind == "segmented_max_range_sumcheck") {
            return max_range_operation(tables, plan_operation, domains, families, graph, interner, error);
        } else if (plan_operation.kind == "segmented_sumcheck") {
            return grouped_sum_operation(plan_operation, domains, families, graph, interner, error);
        } else if (plan_operation.kind == "signed_round_div_sumcheck") {
            return signed_round_operation(config, plan_operation, ledger, domains, families, interner, error);
        } else if (plan_operation.kind == "tensor_matmul_sumcheck") {
            if (plan_operation.inputs.size() != 2U || plan_operation.outputs.size() != 1U) return unsupported("one input, model kernel, and output tensor");
            relation::PrimitiveDescriptor primitive;
            if (!padded_matrix_shape(families, plan_operation.inputs[0], plan_operation.inputs[1], plan_operation.outputs[0], primitive, error))
                return relation::OperationDescriptor{};
            operation.kind = relation::RelationKind::Projection;
            operation.degree = 2U;
            const std::string output = plan_operation.outputs[0];
            set_tensor_refs(operation, {output});
            for (const auto &ref : primitive.tensor_refs) append_unique(operation.source_refs, ref);
            operation.primitives.push_back(std::move(primitive));
            operation.padding = "fixed-zero-padding;zkMatMul;shape-bound";
        } else if (plan_operation.kind == "inner_product_sumcheck") {
            if (plan_operation.inputs.size() != 3U || plan_operation.outputs.size() != 2U)
                return unsupported("src/dst routed matrices, kernels, and two edge outputs");
            const std::string src_matrix = required_view(plan_operation, domains, "src_matrix", "", error);
            const std::string dst_matrix = required_view(plan_operation, domains, "dst_matrix", "", error);
            const std::string src_kernel = required_view(plan_operation, domains, "src_kernel", plan_operation.inputs[1], error);
            const std::string dst_kernel = required_view(plan_operation, domains, "dst_kernel", plan_operation.inputs[2], error);
            const std::string src_output = required_view(plan_operation, domains, "src_output", "", error);
            const std::string dst_output = required_view(plan_operation, domains, "dst_output", "", error);
            if (src_matrix.empty() || dst_matrix.empty() || src_kernel.empty() || dst_kernel.empty() || src_output.empty() || dst_output.empty())
                return relation::OperationDescriptor{};
            relation::PrimitiveDescriptor src_primitive, dst_primitive;
            if (!padded_matrix_shape(families, src_matrix, src_kernel, src_output, src_primitive, error) ||
                !padded_matrix_shape(families, dst_matrix, dst_kernel, dst_output, dst_primitive, error))
                return relation::OperationDescriptor{};
            if (src_primitive.matmul_shape.k != 1U || dst_primitive.matmul_shape.k != 1U) return unsupported("scalar inner-product output shape");
            operation.kind = relation::RelationKind::AttentionInnerProduct;
            operation.degree = 2U;
            set_tensor_refs(operation, {plan_operation.outputs[0], plan_operation.outputs[1]});
            for (const auto &ref : src_primitive.tensor_refs) append_unique(operation.source_refs, ref);
            for (const auto &ref : dst_primitive.tensor_refs) append_unique(operation.source_refs, ref);
            operation.primitives.push_back(std::move(src_primitive));
            operation.primitives.push_back(std::move(dst_primitive));
            operation.padding = "fixed-zero-padding;zkMatMul;shape-bound;src-dst-route";
            if (graph == nullptr || !add_fac_broadcast_from_graph(operation, src_matrix, plan_operation.inputs[0], domains, families, *graph, false, interner, error) ||
                !add_fac_broadcast_from_graph(operation, dst_matrix, plan_operation.inputs[0], domains, families, *graph, true, interner, error) ||
                !add_fac_identity_from_families(operation, src_output, plan_operation.outputs[0], domains, families, interner, error) ||
                !add_fac_identity_from_families(operation, dst_output, plan_operation.outputs[1], domains, families, interner, error))
                return relation::OperationDescriptor{};
            sort_fac_bindings(operation);
        } else {
            return unsupported("an authenticated operation-family mapping");
        }
    } catch (const std::exception &exception) {
        if (error) *error = "BLOCKED: relation family " + plan_operation.id + " descriptor construction failed: " + exception.what();
        return relation::OperationDescriptor{};
    }
    std::string validation_error;
    if (!interner.analysis_only() && !operation.valid(&validation_error)) {
        if (error) *error = "BLOCKED: canonical relation descriptor rejected for " + plan_operation.id + ": " + validation_error;
        return relation::OperationDescriptor{};
    }
    return operation;
}

bool build_canonical_operations(const Config &config, const StaticTables &tables, const RelationPlan &relation,
                                const std::vector<relation::TensorWitness> &tensors, std::vector<relation::OperationDescriptor> &operations, const Graph *graph,
                                std::string *error) {
    std::map<std::string, std::uint32_t> domains;
    for (const auto &tensor : tensors) domains.emplace(tensor.id, tensor.domain_size);
    std::map<std::string, const TensorFamily *> families;
    for (const auto &family : relation.tensor_families)
        if (!families.emplace(family.id, &family).second) {
            if (error) *error = "duplicate canonical tensor family: " + family.id;
            return false;
        }
    operations.clear();
    DescriptorBackingInterner interner(relation.registry_digest, graph);
    std::set<std::string> ids;
    std::vector<std::string> contract_errors;
    for (const auto &plan_operation : relation.operation_families) {
        std::string operation_error;
        relation::OperationDescriptor operation = canonical_operation(config, tables, relation.integer_bound_ledger, plan_operation, domains, families, graph, interner,
                                                                        &operation_error);
        if (operation.id.empty()) {
            contract_errors.push_back(plan_operation.id + ": " +
                                      (operation_error.empty() ? "canonical descriptor construction failed" : operation_error));
            continue;
        }
        bool operation_valid = true;
        bool all_operation_refs_present = true;
        for (const auto &ref : operation.tensor_refs) {
            const auto found = domains.find(ref);
            if (found == domains.end()) {
                contract_errors.push_back(operation.id + ": missing tensor " + ref);
                operation_valid = false;
                all_operation_refs_present = false;
            }
        }
        for (const auto &primitive : operation.primitives) {
            for (const auto &ref : primitive.tensor_refs) {
                const auto found = domains.find(ref);
                if (found == domains.end()) {
                    contract_errors.push_back(operation.id + ": missing primitive tensor " + ref);
                    operation_valid = false;
                    all_operation_refs_present = false;
                }
            }
            const auto &shape = primitive.matmul_shape;
            if (primitive.tensor_refs.size() == 3U &&
                std::all_of(primitive.tensor_refs.begin(), primitive.tensor_refs.end(),
                            [&](const std::string &ref) { return domains.count(ref) != 0U; }) &&
                (domains.at(primitive.tensor_refs[0]) != shape.a_size() ||
                 domains.at(primitive.tensor_refs[1]) != shape.b_size() ||
                 domains.at(primitive.tensor_refs[2]) != shape.c_size())) {
                contract_errors.push_back(operation.id + ": primitive tensor domains do not match fixed padded zkMatMul shape");
                operation_valid = false;
            }
        }
        if (operation.tensor_refs.empty()) {
            contract_errors.push_back(operation.id + ": descriptor has no tensor references");
            operation_valid = false;
            all_operation_refs_present = false;
        }

        const bool has_route = operation.kind == relation::RelationKind::SparseRoute || !operation.selector_source_ref.empty();
        if (!has_route && all_operation_refs_present && !operation.tensor_refs.empty()) {
            const std::uint32_t domain = domains.at(operation.tensor_refs.front());
            for (const auto &ref : operation.tensor_refs)
                if (domains.at(ref) != domain) {
                    contract_errors.push_back(operation.id + ": tensor references have non-aligned MLE domains; explicit FAC4DNN reindex witness is required");
                    operation_valid = false;
                    break;
                }
        }
        if (!ids.insert(operation.id).second) {
            contract_errors.push_back(operation.id + ": duplicate canonical native operation id");
            operation_valid = false;
        }
        if (!operation.constraints.empty()) {
            const auto shape = families.find(operation.tensor_refs.front());
            if (shape == families.end() || shape->second == nullptr) {
                contract_errors.push_back(operation.id + ": pointwise operation is missing authenticated active shape");
                operation_valid = false;
            } else {
                for (auto& constraint : operation.constraints) {
                    constraint.logical_shape = shape->second->logical_shape;
                    constraint.padded_shape = shape->second->padded_shape;
                }
            }
        }
        if (!operation_valid) continue;
        operations.push_back(std::move(operation));
    }
    if (!contract_errors.empty()) {
        if (error) {
            std::ostringstream message;
            message << "BLOCKED: native relation contract audit found " << contract_errors.size() << " issue(s)";
            for (const auto &item : contract_errors) message << "\n - " << item;
            *error = message.str();
        }
        return false;
    }
    return !operations.empty();
}

bool operation_set_matches_plan(const RelationPlan &plan, const std::vector<relation::OperationDescriptor> &operations, std::string *error) {
    std::map<std::string, const relation::OperationDescriptor *> supplied;
    for (const auto &operation : operations) {
        if (!operation.valid(error) || !supplied.emplace(operation.id, &operation).second) {
            if (error && error->empty()) *error = "duplicate/invalid stage-qualified native relation descriptor";
            return false;
        }
    }
    if (supplied.size() != plan.operation_families.size()) {
        if (error) *error = "BLOCKED: native witness does not carry one descriptor for every RelationPlan operation";
        return false;
    }
    for (const auto &expected : plan.operation_families) {
        const bool canonical_root = expected.id == "head_sum" || expected.id == "final_average" || expected.id == "terminal_opening";
        if ((expected.id.find('/') == std::string::npos && !canonical_root) || (expected.id == "head_sum" && expected.kind != "tensor_sumcheck") ||
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
        const auto &actual = *found->second;
        const bool low_degree_tree = expected.kind.find("indexed_public_table") != std::string::npos ||
                                     expected.kind.find("signed_round_div") != std::string::npos || expected.kind.find("segmented_") != std::string::npos ||
                                     expected.kind == "public_selector_reindex";
        const bool route_mapping = expected.kind == "public_selector_reindex" &&
                                   (actual.mapping == expected.mapping || actual.mapping == expected.mapping + ";public-selector-mle:src-dst");
        const bool broadcast_mapping = expected.kind == "field_inverse_product_sumcheck" &&
                                       (actual.mapping == expected.mapping || actual.mapping == expected.mapping + ";public-selector-mle:dst");
        if ((!low_degree_tree && actual.degree != expected.degree) || (low_degree_tree && (actual.degree < expected.degree || actual.degree > 2U)) ||
            (!route_mapping && !broadcast_mapping && actual.mapping != expected.mapping) || actual.terminal_point != expected.terminal_point ||
            actual.terminal_claim != expected.terminal_claim || actual.requires_reindex != expected.requires_reindex ||
            actual.public_route != expected.public_route) {
            if (error) *error = "native relation descriptor does not match authenticated plan: " + expected.id;
            return false;
        }
        std::set<std::string> refs(actual.source_refs.begin(), actual.source_refs.end());
        for (const auto &ref : expected.inputs)
            if (!refs.count(ref)) {
                if (error) *error = "native relation descriptor omits plan input: " + expected.id + " -> " + ref;
                return false;
            }
        for (const auto &ref : expected.outputs)
            if (!refs.count(ref)) {
                if (error) *error = "native relation descriptor omits plan output: " + expected.id + " -> " + ref;
                return false;
            }
        std::set<std::string> selectors(actual.selector_refs.begin(), actual.selector_refs.end());
        for (const auto &selector : expected.auxiliary)
            if (!selectors.count(selector)) {
                if (error) *error = "native relation descriptor omits plan selector: " + expected.id + " -> " + selector;
                return false;
            }
    }
    return true;
}

std::uint32_t descriptor_visibility(const std::string &visibility) { return visibility == "public" ? 1U : (visibility == "witness" ? 3U : 0U); }

bool shape_contract_from_plan(const RelationPlan &plan, const std::vector<NativeRelationWitness::TensorShape> &supplied,
                              std::vector<relation::TensorDescriptor> &descriptors, std::string *error) {
    std::map<std::string, const TensorFamily *> families;
    for (const auto &family : plan.tensor_families)
        if (!families.emplace(family.id, &family).second) {
            if (error) *error = "duplicate tensor family in authenticated RelationPlan: " + family.id;
            return false;
        }
    std::set<std::string> seen;
    descriptors.clear();
    for (const auto &shape : supplied) {
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
        const TensorFamily &family = *found->second;
        const std::uint32_t family_visibility = descriptor_visibility(family.visibility);
        if (descriptor.logical_shape != family.logical_shape || descriptor.padded_shape != family.padded_shape || descriptor.mle_order != family.mle_order ||
            descriptor.stack_axis != family.stack_axis || descriptor.mapping != family.mapping || descriptor.padding_rule != family.padding_rule ||
            (family_visibility == 1U && descriptor.visibility != 1U) ||
            (family_visibility == 3U && (descriptor.visibility != 2U && descriptor.visibility != 3U))) {
            if (error) *error = "public tensor shape/mapping/padding differs from authenticated RelationPlan: " + shape.id;
            return false;
        }
        descriptors.push_back(std::move(descriptor));
    }
    for (const auto &family : plan.tensor_families)
        if (!seen.count(family.id)) {
            if (error) *error = "public tensor shape contract omits RelationPlan family: " + family.id;
            return false;
        }
    return true;
}

bool shape_contract_from_witness(const RelationPlan &plan, const NativeRelationWitness &witness, std::vector<relation::TensorDescriptor> &descriptors,
                                 std::string *error) {
    if (witness.tensor_shapes.empty()) {
        if (error) *error = "BLOCKED: native relation witness lacks canonical tensor shape/padding descriptors";
        return false;
    }
    return shape_contract_from_plan(plan, witness.tensor_shapes, descriptors, error);
}

bool graph_public_descriptor_id(const std::string &id) {
    const auto suffix = [&](const std::string &value) { return id.size() >= value.size() && id.compare(id.size() - value.size(), value.size(), value) == 0U; };
    return id == "input_features" || (id.rfind("hidden/layer0/head", 0U) == 0U && suffix("/trace/input")) || id == "sparse_route" || id == "group_selectors" || id == "node_ptr" || id == "edge_ptr" || id == "node_edge_ptr" ||
           suffix("/view/group_new") || suffix("/view/group_end");
}

bool public_logical_provider(const relation::TensorDescriptor &descriptor, std::size_t basis_size, std::function<FrWire(std::uint64_t)> reader, relation::SegmentedTensorProvider &provider, std::string *error) {
    std::uint32_t segment_size = 0U, segment_count = 0U;
    if (!descriptor.valid(error) || !relation::SegmentedTensorProvider::shape(descriptor.domain_size, basis_size, segment_size, segment_count, error)) return false;
    provider = relation::SegmentedTensorProvider(descriptor.domain_size, segment_size, [logical = descriptor.logical_shape, padded = descriptor.padded_shape, reader, domain = descriptor.domain_size](std::uint64_t offset, std::uint32_t count, std::vector<FrWire> &values, std::string *read_error) {
        if (offset > domain || count > domain - offset) { if (read_error) *read_error = "native tensor range exceeds authenticated domain"; return false; }
        values.assign(count, FrWire{});
        for (std::uint32_t index = 0U; index < count; ++index) {
            std::uint64_t remainder = offset + index, logical_index = 0U, stride = 1U;
            bool active = true;
            for (std::size_t axis = logical.size(); axis-- > 0U;) {
                const auto coordinate = remainder % padded[axis];
                remainder /= padded[axis];
                if (coordinate >= logical[axis]) { active = false; break; }
                if (coordinate != 0U && stride > std::numeric_limits<std::uint64_t>::max() / coordinate) return false;
                const std::uint64_t term = coordinate * stride;
                if (logical_index > std::numeric_limits<std::uint64_t>::max() - term ||
                    (logical[axis] != 0U && stride > std::numeric_limits<std::uint64_t>::max() / logical[axis])) return false;
                logical_index += term;
                stride *= logical[axis];
            }
            if (active) values[index] = reader(logical_index);
        }
        return true;
    });
    return true;
}

bool graph_public_provider(const Graph &graph, const relation::TensorDescriptor &descriptor, std::size_t basis_size, relation::SegmentedTensorProvider &provider, std::string *error) {
    const auto from_u64 = [](std::uint64_t value) {
        return zkdl::fr_from_u32(checked_native_u32(value, "public graph index", nullptr, true));
    };
    if (descriptor.id == "input_features" || (descriptor.id.rfind("hidden/layer0/head", 0U) == 0U && descriptor.id.size() >= 12U && descriptor.id.compare(descriptor.id.size() - 12U, 12U, "/trace/input") == 0U)) {
        if (descriptor.logical_shape != std::vector<std::uint64_t>{graph.nodes, graph.features} || graph.feature_values.rows != graph.nodes || graph.feature_values.cols != graph.features) { if (error) *error = "public input_features shape differs from graph feature material"; return false; }
        return public_logical_provider(descriptor, basis_size, [&graph](std::uint64_t index) { return relation::fr_from_signed_i64(graph.feature_values.at(index / graph.features, index % graph.features)); }, provider, error);
    }
    const auto suffix = [&](const std::string &value) { return descriptor.id.size() >= value.size() && descriptor.id.compare(descriptor.id.size() - value.size(), value.size(), value) == 0U; };
    const bool boundary_view = suffix("/view/group_new") || suffix("/view/group_end");
    if (descriptor.id == "sparse_route" || descriptor.id == "group_selectors" || boundary_view) {
        if (graph.nodes == std::numeric_limits<std::uint64_t>::max() || graph.nodes > kMaxNativeDomain || graph.edges > kMaxNativeDomain ||
            descriptor.logical_shape.empty() || descriptor.logical_shape.front() != graph.edges || graph.src.size() != graph.edges || graph.dst.size() != graph.edges || graph.node_edge_ptr.size() != graph.nodes + 1U || graph.node_edge_ptr.front() != 0U || graph.node_edge_ptr.back() != graph.edges) { if (error) *error = "public route/CSR tensor shape is incompatible with graph"; return false; }
        if (!boundary_view && descriptor.logical_shape != std::vector<std::uint64_t>{graph.edges, 2U}) { if (error) *error = "public route tensor must have two logical columns"; return false; }
        for (std::size_t node = 0U; node + 1U < graph.node_edge_ptr.size(); ++node) if (graph.node_edge_ptr[node] > graph.node_edge_ptr[node + 1U] || graph.node_edge_ptr[node + 1U] > graph.edges) { if (error) *error = "public CSR pointers are not monotone and bounded"; return false; }
        for (std::size_t edge = 0U; edge < graph.dst.size(); ++edge) { const auto node = graph.dst[edge]; if (node >= graph.nodes || graph.src[edge] >= graph.nodes || edge < graph.node_edge_ptr[node] || edge >= graph.node_edge_ptr[node + 1U]) { if (error) *error = "public edge is outside its destination CSR segment"; return false; } }
        std::uint64_t width = 1U;
        for (std::size_t axis = 1U; axis < descriptor.logical_shape.size(); ++axis) {
            if (!checked_mul_u64(width, descriptor.logical_shape[axis], width)) { if (error) *error = "public graph route width overflows uint64"; return false; }
        }
        const bool route = descriptor.id == "sparse_route", end = suffix("/view/group_end");
        return public_logical_provider(descriptor, basis_size, [&graph, width, route, end, boundary_view, from_u64](std::uint64_t index) { const auto edge = index / width, column = index % width; if (route) return from_u64((column == 0U ? graph.src : graph.dst).at(edge)); const auto node = graph.dst.at(edge); const bool want_end = boundary_view ? end : column == 1U; return zkdl::fr_from_u32((want_end ? edge + 1U == graph.node_edge_ptr.at(node + 1U) : edge == graph.node_edge_ptr.at(node)) ? 1U : 0U); }, provider, error);
    }
    const std::vector<std::uint64_t> *source = descriptor.id == "node_ptr" ? &graph.node_ptr : descriptor.id == "edge_ptr" ? &graph.edge_ptr : descriptor.id == "node_edge_ptr" ? &graph.node_edge_ptr : nullptr;
    if (source == nullptr) return false;
    if (descriptor.logical_shape != std::vector<std::uint64_t>{source->size()}) { if (error) *error = "public graph boundary tensor logical shape differs from Graph"; return false; }
    return public_logical_provider(descriptor, basis_size, [source, from_u64](std::uint64_t index) { return from_u64(source->at(index)); }, provider, error);
}

bool authenticated_table_values(const StaticTables &tables, const relation::TensorDescriptor &descriptor, std::vector<FrWire> &values, std::string *error) {
    const std::vector<std::int64_t> *table = nullptr;
    if (descriptor.id == "static_lrelu")
        table = &tables.lrelu;
    else if (descriptor.id == "static_elu")
        table = &tables.elu;
    else if (descriptor.id == "static_exp")
        table = &tables.exp;
    else
        return false;
    if (table->empty() || table->size() > descriptor.domain_size || descriptor.logical_shape.size() != 1U ||
        descriptor.logical_shape.front() != table->size() || descriptor.padded_shape.size() != 1U ||
        descriptor.padded_shape.front() != descriptor.domain_size) {
        if (error) *error = "authenticated static table shape does not match RelationPlan: " + descriptor.id;
        return false;
    }
    values.assign(descriptor.domain_size, FrWire{});
    for (std::size_t row = 0U; row < table->size(); ++row) values[row] = relation::fr_from_signed_i64((*table)[row]);
    return true;
}

bool bind_authenticated_public_material(const StaticTables &tables, const std::vector<relation::TensorDescriptor> &descriptors,
                                        std::map<std::string, std::vector<FrWire>> &values, std::string *error) {
    for (const auto &descriptor : descriptors) {
        if (descriptor.visibility != 1U) continue;
        std::vector<FrWire> expected;
        if (descriptor.id == "input_features") { continue; }
        if (!authenticated_table_values(tables, descriptor, expected, error)) {
            if (error && !error->empty() && descriptor.id.rfind("static_", 0U) == 0U) return false;
            continue;
        }
        const auto found = values.find(descriptor.id);
        if (found == values.end())
            values.emplace(descriptor.id, std::move(expected));
        else if (found->second != expected) {
            if (error) *error = "public static table witness differs from authenticated table descriptor: " + descriptor.id;
            return false;
        }
    }
    return true;
}

struct SortedLookupScratchLayout {
    std::filesystem::path path;
    std::string roles;
    std::string key_max_u128;
    std::uint64_t table_offset = 0U;
    std::uint64_t query_offset = 0U;
    std::uint64_t merge_offset = 0U;
    std::uint32_t table_domain = 0U;
    std::uint32_t query_domain = 0U;
    std::uint32_t merge_domain = 0U;
    std::uint32_t segment_size = 0U;
    std::uint32_t table_logical = 0U;
    std::uint32_t query_logical = 0U;
    std::uint32_t merge_logical = 0U;
    std::uint32_t x_bits = 0U;
    std::uint32_t y_bits = 0U;
    static constexpr std::uint64_t kRecordBytes = 107U;
};

bool sorted_scratch_u64(std::ifstream &input, std::uint64_t &value) {
    std::array<std::uint8_t, 8U> bytes{};
    if (!input.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) return false;
    value = 0U;
    for (unsigned index = 0U; index < bytes.size(); ++index) value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
    return true;
}

bool sorted_scratch_u32(std::ifstream &input, std::uint32_t &value) {
    std::array<std::uint8_t, 4U> bytes{};
    if (!input.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) return false;
    value = 0U;
    for (unsigned index = 0U; index < bytes.size(); ++index) value |= static_cast<std::uint32_t>(bytes[index]) << (8U * index);
    return true;
}

bool sorted_scratch_string(std::ifstream &input, std::string &value) {
    std::uint64_t length = 0U;
    if (!sorted_scratch_u64(input, length) || length > (1ULL << 20U) || length > std::numeric_limits<std::size_t>::max() ||
        length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    value.assign(static_cast<std::size_t>(length), '\0');
    return length == 0U || static_cast<bool>(input.read(value.data(), static_cast<std::streamsize>(length)));
}

bool sorted_lookup_parse_u128(const std::string &text, unsigned __int128 &value, std::string *error) {
    if (text.empty()) {
        if (error) *error = "BLOCKED: sorted-lookup u128 decimal is empty";
        return false;
    }
    value = 0U;
    const unsigned __int128 maximum = ~static_cast<unsigned __int128>(0U);
    for (const char digit : text) {
        if (digit < '0' || digit > '9' || value > (maximum - static_cast<unsigned __int128>(digit - '0')) / 10U) {
            if (error) *error = "BLOCKED: sorted-lookup u128 decimal is not canonical";
            return false;
        }
        value = value * 10U + static_cast<unsigned __int128>(digit - '0');
    }
    return true;
}

bool sorted_scratch_layout(const Config &config, const NativeSortedLookupProviderInput &input, SortedLookupScratchLayout &layout, std::string *error) {
    layout = SortedLookupScratchLayout{};
    layout.path = std::filesystem::path(input.scratch_path);
    if (layout.path.is_relative()) layout.path = config.baseline_root / layout.path;
    std::ifstream stream(layout.path, std::ios::binary);
    if (!stream) {
        if (error) *error = "BLOCKED: sorted-lookup scratch is missing: " + layout.path.string();
        return false;
    }
    std::array<char, 8U> magic{};
    if (!stream.read(magic.data(), magic.size()) || std::string(magic.data(), magic.size()) != "ZKDLQTS1") {
        if (error) *error = "BLOCKED: sorted-lookup scratch magic mismatch";
        return false;
    }
    std::uint32_t version = 0U;
    std::string operation, roles, key_max;
    std::uint64_t table_domain = 0U, query_domain = 0U, merge_domain = 0U, segment_size = 0U, table_logical = 0U, query_logical = 0U, merge_logical = 0U,
                  active_length = 0U;
    if (!sorted_scratch_u32(stream, version) || version != 1U || !sorted_scratch_string(stream, operation) || !sorted_scratch_string(stream, roles) ||
        !sorted_scratch_u64(stream, table_domain) || !sorted_scratch_u64(stream, query_domain) || !sorted_scratch_u64(stream, merge_domain) ||
        !sorted_scratch_u64(stream, segment_size) || !sorted_scratch_u64(stream, table_logical) || !sorted_scratch_u64(stream, query_logical) ||
        !sorted_scratch_u64(stream, merge_logical)) {
        if (error) *error = "BLOCKED: sorted-lookup scratch header is truncated";
        return false;
    }
    if (!sorted_scratch_u32(stream, layout.x_bits) || !sorted_scratch_u32(stream, layout.y_bits) || !sorted_scratch_string(stream, key_max) ||
        !sorted_scratch_u64(stream, active_length)) {
        if (error) *error = "BLOCKED: sorted-lookup scratch key header is truncated";
        return false;
    }
    layout.roles = roles;
    layout.key_max_u128 = key_max;
    const auto fits_u32 = [](std::uint64_t value) { return value != 0U && value <= kMaxNativeDomain; };
    if (!fits_u32(table_domain) || !fits_u32(query_domain) || !fits_u32(merge_domain) || !fits_u32(segment_size) || !fits_u32(table_logical) ||
        !fits_u32(query_logical) || !fits_u32(merge_logical) || table_domain % segment_size != 0U || query_domain % segment_size != 0U ||
        merge_domain % segment_size != 0U || !native_power_of_two(table_domain) || !native_power_of_two(query_domain) ||
        !native_power_of_two(merge_domain) || !native_power_of_two(segment_size) || operation != input.operation_id || roles.empty() ||
        active_length != table_logical || table_logical > table_domain || query_logical > query_domain || merge_logical > merge_domain ||
        table_logical > kMaxNativeDomain - query_logical || merge_logical != table_logical + query_logical ||
        layout.x_bits == 0U || layout.y_bits == 0U || layout.x_bits > 127U || layout.y_bits > 127U || layout.x_bits > 127U - layout.y_bits ||
        input.table_domain != table_domain || input.query_domain != query_domain || input.merge_domain != merge_domain || input.segment_size != segment_size ||
        input.table_logical_rows != table_logical || input.query_logical_rows != query_logical || input.merge_logical_rows != merge_logical ||
        input.x_bits != layout.x_bits || input.y_bits != layout.y_bits) {
        if (error) *error = "BLOCKED: sorted-lookup scratch metadata does not match provider descriptor";
        return false;
    }
    try {
        layout.table_domain = checked_native_u32(table_domain, "sorted lookup table domain", error);
        layout.query_domain = checked_native_u32(query_domain, "sorted lookup query domain", error);
        layout.merge_domain = checked_native_u32(merge_domain, "sorted lookup merge domain", error);
        layout.segment_size = checked_native_u32(segment_size, "sorted lookup segment size", error);
        layout.table_logical = checked_native_u32(table_logical, "sorted lookup table row count", error);
        layout.query_logical = checked_native_u32(query_logical, "sorted lookup query row count", error);
        layout.merge_logical = checked_native_u32(merge_logical, "sorted lookup merge row count", error);
    } catch (const std::runtime_error &) {
        return false;
    }
    const auto checked_section = [&](std::uint64_t count, std::uint64_t &offset) {
        const auto position = stream.tellg();
        if (position < 0 || count > std::numeric_limits<std::uint64_t>::max() / SortedLookupScratchLayout::kRecordBytes) return false;
        offset = static_cast<std::uint64_t>(position);
        const std::uint64_t bytes = count * SortedLookupScratchLayout::kRecordBytes;
        if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
        stream.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        return static_cast<bool>(stream);
    };
    if (!checked_section(table_domain, layout.table_offset) || !sorted_scratch_u64(stream, active_length) || active_length != query_logical ||
        !checked_section(query_domain, layout.query_offset) || !sorted_scratch_u64(stream, active_length) || active_length != merge_logical ||
        !checked_section(merge_domain, layout.merge_offset)) {
        if (error) *error = "BLOCKED: sorted-lookup scratch record sections are truncated";
        return false;
    }
    std::error_code size_error;
    const auto file_size = std::filesystem::file_size(layout.path, size_error);
    if (size_error || layout.merge_offset > file_size || merge_domain > (file_size - layout.merge_offset) / SortedLookupScratchLayout::kRecordBytes) {
        if (error) *error = "BLOCKED: sorted-lookup scratch record coverage is invalid";
        return false;
    }
    return true;
}

bool sorted_scratch_provider(const SortedLookupScratchLayout &layout, std::uint64_t section_offset, std::uint32_t domain, std::uint32_t field,
                             relation::SegmentedTensorProvider &provider, std::string *error,
                             double *explicit_io_decode_seconds = nullptr) {
    const auto path = layout.path;
    provider = relation::SegmentedTensorProvider(domain, layout.segment_size,
                                                 [path, section_offset, domain, field, explicit_io_decode_seconds](std::uint64_t offset, std::uint32_t count, std::vector<FrWire> &output,
                                                                                       std::string *read_error) {
                                                     constexpr std::uint64_t record_bytes = SortedLookupScratchLayout::kRecordBytes;
                                                     if (offset > domain || count > domain - offset || offset > std::numeric_limits<std::uint64_t>::max() / record_bytes ||
                                                         count > std::numeric_limits<std::uint64_t>::max() / record_bytes) {
                                                         if (read_error) *read_error = "sorted-lookup scratch range overflow";
                                                         return false;
                                                     }
                                                     const std::uint64_t section_delta = offset * record_bytes;
                                                     const std::uint64_t byte_count = static_cast<std::uint64_t>(count) * record_bytes;
                                                     if (section_offset > std::numeric_limits<std::uint64_t>::max() - section_delta ||
                                                         byte_count > std::numeric_limits<std::size_t>::max() ||
                                                         byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
                                                         if (read_error) *read_error = "sorted-lookup scratch byte range is not representable";
                                                         return false;
                                                     }
                                                     const std::uint64_t byte_offset = section_offset + section_delta;
                                                     if (byte_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
                                                         if (read_error) *read_error = "sorted-lookup scratch offset exceeds stream range";
                                                         return false;
                                                     }
                                                     const auto explicit_io_decode_start = std::chrono::steady_clock::now();
                                                     std::ifstream stream(path, std::ios::binary);
                                                     if (!stream) {
                                                         if (read_error) *read_error = "cannot open sorted-lookup scratch";
                                                         return false;
                                                     }
                                                     stream.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
                                                     std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byte_count));
                                                     if (!bytes.empty() &&
                                                         !stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
                                                         if (read_error) *read_error = "cannot read sorted-lookup scratch range";
                                                         return false;
                                                     }
                                                     const auto read_u128 = [&](std::size_t base) {
                                                         unsigned __int128 value = 0U;
                                                         for (unsigned index = 0U; index < 16U; ++index)
                                                             value |= static_cast<unsigned __int128>(bytes[base + index]) << (8U * index);
                                                         return relation::sorted_lookup_field_from_u128(value);
                                                     };
                                                     const auto read_field = [&](std::size_t base) {
                                                         FrWire value{};
                                                         for (unsigned index = 0U; index < value.size(); ++index)
                                                             value[index] = static_cast<std::uint32_t>(bytes[base + 4U * index]) |
                                                                            (static_cast<std::uint32_t>(bytes[base + 4U * index + 1U]) << 8U) |
                                                                            (static_cast<std::uint32_t>(bytes[base + 4U * index + 2U]) << 16U) |
                                                                            (static_cast<std::uint32_t>(bytes[base + 4U * index + 3U]) << 24U);
                                                         return value;
                                                     };
                                                     const auto read_u64 = [&](std::size_t base) {
                                                         std::uint64_t value = 0U;
                                                         for (unsigned index = 0U; index < 8U; ++index)
                                                             value |= static_cast<std::uint64_t>(bytes[base + index]) << (8U * index);
                                                         return relation::sorted_lookup_field_from_u128(value);
                                                     };
                                                     output.resize(count);
                                                     for (std::uint32_t index = 0U; index < count; ++index) {
                                                         const std::size_t base = static_cast<std::size_t>(index) * SortedLookupScratchLayout::kRecordBytes;
                                                         switch (field) {
                                                             case 0U:
                                                                 output[index] = read_u128(base);
                                                                 break;
                                                             case 1U:
                                                                 output[index] = read_u128(base + 16U);
                                                                 break;
                                                             case 2U:
                                                                 output[index] = read_field(base + 32U);
                                                                 break;
                                                             case 3U:
                                                                 output[index] = read_u128(base + 64U);
                                                                 break;
                                                             case 4U:
                                                                 output[index] = read_u128(base + 80U);
                                                                 break;
                                                             case 5U:
                                                                 output[index] = read_u64(base + 96U);
                                                                 break;
                                                             case 6U:
                                                                 output[index] = zkdl::fr_from_u32(bytes[base + 104U]);
                                                                 break;
                                                             case 7U:
                                                                 output[index] = zkdl::fr_from_u32(bytes[base + 105U]);
                                                                 break;
                                                             default:
                                                                 output[index] = zkdl::fr_from_u32(bytes[base + 106U]);
                                                                 break;
                                                         }
                                                     }
                                                     if (explicit_io_decode_seconds != nullptr)
                                                         *explicit_io_decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - explicit_io_decode_start).count();
                                                     return true;
                                                 });
    return provider.valid(error);
}

const OperationFamily *sorted_lookup_plan_operation(const RelationPlan &relation, const std::string &operation_id) {
    const auto found = std::find_if(relation.operation_families.begin(), relation.operation_families.end(),
                                    [&](const OperationFamily &operation) { return operation.id == operation_id; });
    return found == relation.operation_families.end() ? nullptr : &*found;
}

bool sorted_lookup_descriptor_from_plan(const RelationPlan &relation, const PublicStatement &statement, const relation::OperationDescriptor &operation,
                                        const SortedLookupQueryPlan &query, zkdl::AuthenticatedSortedLookupDescriptor &descriptor, std::string *error) {
    if (statement.sorted_lookup_descriptor_digest != relation.sorted_lookup.descriptor_digest ||
        statement.sorted_lookup_plan_digest != relation.sorted_lookup.plan_digest ||
        statement.sorted_lookup_query_count != relation.sorted_lookup.queries.size() ||
        relation.sorted_lookup.table_digest != statement.static_table_descriptor_digest || operation.id != query.operation_id ||
        operation.batch_family != "authenticated_sorted_lookup" || operation.public_table_ref != query.table_ref) {
        if (error) *error = "BLOCKED: sorted-lookup descriptor is not bound to RelationPlan/statement";
        return false;
    }
    if (query.table_domain == 0U || query.table_domain > kMaxNativeDomain || !native_power_of_two(query.table_domain) ||
        query.query_domain == 0U || query.query_domain > kMaxNativeDomain || !native_power_of_two(query.query_domain) ||
        query.merge_domain == 0U || query.merge_domain > kMaxNativeDomain || !native_power_of_two(query.merge_domain) ||
        query.table_logical_rows == 0U || query.table_logical_rows > query.table_domain ||
        query.query_logical_rows == 0U || query.query_logical_rows > query.query_domain ||
        query.merge_logical_rows == 0U || query.merge_logical_rows > query.merge_domain ||
        query.table_logical_rows > kMaxNativeDomain - query.query_logical_rows ||
        query.merge_logical_rows != query.table_logical_rows + query.query_logical_rows ||
        query.segment_size == 0U || query.segment_size > kMaxNativeDomain || !native_power_of_two(query.segment_size) ||
        query.x_bits == 0U || query.y_bits == 0U || query.x_bits > 127U || query.y_bits > 127U || query.x_bits > 127U - query.y_bits) {
        if (error) *error = "BLOCKED: sorted-lookup descriptor exceeds native wire bounds";
        return false;
    }
    std::array<zkdl::byte, 32U> table_digest{}, descriptor_digest{};
    unsigned __int128 key_max = 0U;
    if (!parse_digest(statement.static_table_digest, table_digest) || !parse_digest(query.layout_digest, descriptor_digest) ||
        !sorted_lookup_parse_u128(query.key_max_u128, key_max, error)) {
        if (error && error->empty()) *error = "BLOCKED: sorted-lookup descriptor digest/key bound is not canonical";
        return false;
    }
    descriptor = zkdl::AuthenticatedSortedLookupDescriptor{};
    descriptor.operation_id = query.operation_id;
    descriptor.table_ref = query.table_ref;
    descriptor.query_ref = query.query_ref;
    descriptor.merge_ref = query.merge_ref;
    descriptor.table_digest = table_digest;
    descriptor.descriptor_digest = descriptor_digest;
    try {
        descriptor.table_domain = checked_native_u32(query.table_domain, "sorted lookup table domain", error);
        descriptor.query_domain = checked_native_u32(query.query_domain, "sorted lookup query domain", error);
        descriptor.merge_domain = checked_native_u32(query.merge_domain, "sorted lookup merge domain", error);
        descriptor.table_active_length = checked_native_u32(query.table_logical_rows, "sorted lookup table row count", error);
        descriptor.query_active_length = checked_native_u32(query.query_logical_rows, "sorted lookup query row count", error);
        descriptor.merge_active_length = checked_native_u32(query.merge_logical_rows, "sorted lookup merge row count", error);
        descriptor.segment_size = checked_native_u32(query.segment_size, "sorted lookup segment size", error);
        descriptor.table_segment_count = checked_native_u32(query.table_segment_count, "sorted lookup table segment count", error);
        descriptor.query_segment_count = checked_native_u32(query.query_segment_count, "sorted lookup query segment count", error);
        descriptor.merge_segment_count = checked_native_u32(query.merge_segment_count, "sorted lookup merge segment count", error);
    } catch (const std::runtime_error &) {
        return false;
    }
    descriptor.x_bits = query.x_bits;
    descriptor.y_bits = query.y_bits;
    descriptor.key_max = relation::sorted_lookup_field_from_u128(key_max);
    descriptor.table_unique_rows = descriptor.table_active_length;
    return descriptor.valid(error);
}

bool sorted_lookup_public_table_providers(const StaticTables &tables, const OperationFamily &plan_operation, const SortedLookupQueryPlan &query,
                                          relation::SegmentedTensorProvider &table_keys, relation::SegmentedTensorProvider &table_active, std::string *error) {
    std::int64_t ignored_minimum = 0;
    const auto &table = lookup_table(tables, plan_operation, ignored_minimum, error);
    if (table.empty() || table.size() != query.table_logical_rows || query.table_domain == 0U || query.segment_size == 0U ||
        query.table_domain % query.segment_size != 0U || query.segment_size > (1ULL << 18U)) {
        if (error && error->empty()) *error = "BLOCKED: authenticated sorted-lookup table shape mismatch";
        return false;
    }
    unsigned __int128 key_max = 0U;
    if (!sorted_lookup_parse_u128(query.key_max_u128, key_max, error)) return false;
    const std::int64_t output_min = *std::min_element(table.begin(), table.end());
    const std::uint32_t key_bits = query.x_bits + query.y_bits;
    const auto *source = &table;
    const auto key_at = [source, output_min, key_max, key_bits, x_bits = query.x_bits, y_bits = query.y_bits](std::uint64_t row, FrWire &key,
                                                                                                              std::string *range_error) {
        if (row >= source->size()) {
            if (range_error) *range_error = "sorted-lookup public table row is out of bounds";
            return false;
        }
        const __int128 difference = static_cast<__int128>((*source)[static_cast<std::size_t>(row)]) - static_cast<__int128>(output_min);
        if (difference < 0 || x_bits == 0U || y_bits == 0U) {
            if (range_error) *range_error = "BLOCKED: authenticated sorted-lookup table key is negative";
            return false;
        }
        const unsigned __int128 y = static_cast<unsigned __int128>(difference);
        const unsigned __int128 candidate = (static_cast<unsigned __int128>(row) << y_bits) | y;
        if (candidate > key_max || (key_bits < 128U && (candidate >> key_bits) != 0U)) {
            if (range_error) *range_error = "BLOCKED: authenticated sorted-lookup table key exceeds descriptor bound";
            return false;
        }
        key = relation::sorted_lookup_field_from_u128(candidate);
        return true;
    };

    FrWire previous_key{};
    for (std::size_t row = 0U; row < table.size(); ++row) {
        FrWire key{};
        if (!key_at(row, key, error)) return false;
        if (row != 0U && key <= previous_key) {
            if (error) *error = "BLOCKED: authenticated sorted-lookup table keys are not unique/sorted";
            return false;
        }
        previous_key = key;
    }
    std::uint32_t table_domain_u32 = 0U, segment_size_u32 = 0U;
    try {
        table_domain_u32 = checked_native_u32(query.table_domain, query.operation_id + "/table-domain", error);
        segment_size_u32 = checked_native_u32(query.segment_size, query.operation_id + "/segment-size", error);
    } catch (const std::runtime_error &) {
        return false;
    }
    const auto range_provider = [&](std::uint32_t field, relation::SegmentedTensorProvider &result) {
        result = relation::SegmentedTensorProvider(table_domain_u32, segment_size_u32,
                                                   [source, key_at, field, table_domain = query.table_domain,
                                                    segment_size = query.segment_size](std::uint64_t offset, std::uint32_t count, std::vector<FrWire> &output,
                                                                                       std::string *range_error) {
                                                       if (count > segment_size || offset > table_domain || count > table_domain - offset) {
                                                           if (range_error) *range_error = "sorted-lookup public table range exceeds authenticated segment";
                                                           return false;
                                                       }
                                                       output.assign(count, FrWire{});
                                                       for (std::uint32_t index = 0U; index < count; ++index) {
                                                           const std::uint64_t row = offset + index;
                                                           if (row < source->size()) {
                                                               if (field == 1U) {
                                                                   output[index] = zkdl::fr_from_u32(1U);
                                                               } else if (!key_at(row, output[index], range_error)) {
                                                                   return false;
                                                               }
                                                           } else if (field != 1U) {

                                                               output[index] = FrWire{};
                                                           }
                                                       }
                                                       return true;
                                                   });
        return result.valid(error);
    };
    return range_provider(0U, table_keys) && range_provider(1U, table_active);
}

bool sorted_lookup_prover_providers(const zkdl::AuthenticatedSortedLookupDescriptor &descriptor, const SortedLookupScratchLayout &layout,
                                    const relation::SegmentedTensorProvider &table_keys, const relation::SegmentedTensorProvider &table_active,
                                    relation::AuthenticatedSortedLookupProviderSet &providers, std::string *error,
                                    double *explicit_io_decode_seconds = nullptr) {
    providers = relation::AuthenticatedSortedLookupProviderSet{};
    providers.table_keys = table_keys;
    providers.table_active = table_active;
    const auto scratch = [&](std::uint64_t offset, std::uint32_t domain, std::uint32_t field, relation::SegmentedTensorProvider &result) {
        return sorted_scratch_provider(layout, offset, domain, field, result, error, explicit_io_decode_seconds);
    };
    if (!scratch(layout.query_offset, descriptor.query_domain, 0U, providers.query_keys) ||
        !scratch(layout.query_offset, descriptor.query_domain, 3U, providers.query_x) ||
        !scratch(layout.query_offset, descriptor.query_domain, 4U, providers.query_y) ||
        !scratch(layout.query_offset, descriptor.query_domain, 5U, providers.query_prefix) ||
        !scratch(layout.query_offset, descriptor.query_domain, 6U, providers.query_active) ||
        !scratch(layout.merge_offset, descriptor.merge_domain, 0U, providers.merge_keys) ||
        !scratch(layout.merge_offset, descriptor.merge_domain, 1U, providers.merge_delta) ||
        !scratch(layout.merge_offset, descriptor.merge_domain, 2U, providers.merge_inverse) ||
        !scratch(layout.merge_offset, descriptor.merge_domain, 5U, providers.merge_prefix) ||
        !scratch(layout.merge_offset, descriptor.merge_domain, 6U, providers.merge_active) ||
        !scratch(layout.merge_offset, descriptor.merge_domain, 7U, providers.merge_marker) ||
        !relation::make_authenticated_sorted_lookup_shifted_key_provider(
            providers.merge_keys, providers.merge_active,
            providers.merge_previous_key, error) ||
        !relation::make_authenticated_sorted_lookup_nonzero_provider(providers.merge_delta, providers.merge_nonzero, error) ||
        !relation::make_authenticated_sorted_lookup_selector_provider(descriptor.merge_domain, descriptor.segment_size, 0U, providers.merge_first, error) ||
        !relation::make_authenticated_sorted_lookup_selector_provider(descriptor.merge_domain, descriptor.segment_size, descriptor.merge_active_length - 1U,
                                                                      providers.merge_last, error) ||
        !relation::make_authenticated_sorted_lookup_selector_provider(descriptor.query_domain, descriptor.segment_size, 0U, providers.query_first, error) ||
        !relation::make_authenticated_sorted_lookup_selector_provider(descriptor.query_domain, descriptor.segment_size, descriptor.query_active_length - 1U,
                                                                      providers.query_last, error) ||
        !relation::make_authenticated_sorted_lookup_constant_provider(descriptor.query_domain, descriptor.segment_size, FrWire{}, providers.query_prefix_next,
                                                                      error) ||
        !relation::make_authenticated_sorted_lookup_constant_provider(descriptor.merge_domain, descriptor.segment_size, FrWire{}, providers.merge_prefix_next,
                                                                      error))
        return false;
    for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit) {
        relation::SegmentedTensorProvider query_bit, merge_bit, delta_bit;
        if (!relation::make_authenticated_sorted_lookup_bit_provider(providers.query_keys, descriptor.x_bits + descriptor.y_bits, bit, query_bit, error) ||
            !relation::make_authenticated_sorted_lookup_bit_provider(providers.merge_keys, descriptor.x_bits + descriptor.y_bits, bit, merge_bit, error) ||
            !relation::make_authenticated_sorted_lookup_bit_provider(providers.merge_delta, descriptor.x_bits + descriptor.y_bits, bit, delta_bit, error))
            return false;
        providers.query_bits.push_back(std::move(query_bit));
        providers.merge_bits.push_back(std::move(merge_bit));
        providers.merge_delta_bits.push_back(std::move(delta_bit));
    }
    return providers.valid(descriptor, error);
}

std::string sorted_lookup_join_roles(const std::vector<std::string> &roles) {
    std::ostringstream joined;
    for (std::size_t index = 0U; index < roles.size(); ++index) {
        if (index != 0U) joined << ',';
        joined << roles[index];
    }
    return joined.str();
}

bool sorted_lookup_build_prover_context(const Config &config, const StaticTables &tables, const RelationPlan &plan, const PublicStatement &statement,
                                        const relation::RelationStatement &relation_statement, const NativeRelationWitness &witness,
                                        relation::ProverContext &context, std::string *error,
                                        double *outer_commitment_wall_seconds,
                                        double *outer_commitment_file_io_seconds) {
    std::map<std::string, const relation::OperationDescriptor *> operations;
    for (const auto &operation : context.operations)
        if (operation.batch_family == "authenticated_sorted_lookup" && !operations.emplace(operation.id, &operation).second) {
            if (error) *error = "BLOCKED: duplicate sorted-lookup operation in RelationPlan";
            return false;
        }
    if (operations.size() != plan.sorted_lookup.queries.size() || witness.sorted_lookup_providers.size() != plan.sorted_lookup.queries.size()) {
        if (error) *error = "BLOCKED: sorted-lookup operation/provider count does not match RelationPlan";
        return false;
    }
    std::map<std::string, const NativeSortedLookupProviderInput *> supplied;
    for (const auto &provider : witness.sorted_lookup_providers)
        if (!supplied.emplace(provider.operation_id, &provider).second) {
            if (error) *error = "BLOCKED: duplicate sorted-lookup provider operation id";
            return false;
        }
    if (supplied.size() != plan.sorted_lookup.queries.size()) {
        if (error) *error = "BLOCKED: sorted-lookup provider set contains an extra operation";
        return false;
    }
    context.sorted_lookup_descriptors.clear();
    context.sorted_lookup_providers.clear();
    context.sorted_lookup_commitment_refs.clear();
    context.sorted_lookup_outer_commitments.clear();
    context.sorted_lookup_outer_blinders.clear();
    context.sorted_lookup_outer_segment_blinders.clear();
    context.sorted_lookup_outer_segment_commitments.clear();
    context.sorted_lookup_commitment_ids.clear();
    // Reuse one authenticated-basis MSM session for every outer lookup
    // segment. Count its preparation as commitment time, and include it in the
    // outer-commitment wall bucket so context-only proving excludes it.
    const auto msm_session_prepare_start = std::chrono::steady_clock::now();
    zkdl::CpuG1MsmSession msm_session(context.ipa_generators);
    const double msm_session_prepare_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - msm_session_prepare_start).count();
    if (context.commitment_open_time_seconds != nullptr)
        *context.commitment_open_time_seconds += msm_session_prepare_seconds;
    if (outer_commitment_wall_seconds != nullptr)
        *outer_commitment_wall_seconds += msm_session_prepare_seconds;
    for (const auto &query : plan.sorted_lookup.queries) {
        const auto operation_it = operations.find(query.operation_id);
        const auto provider_it = supplied.find(query.operation_id);
        if (operation_it == operations.end() || provider_it == supplied.end()) {
            if (error) *error = "BLOCKED: sorted-lookup operation/provider is missing: " + query.operation_id;
            return false;
        }
        const auto &input = *provider_it->second;
        if (input.table_ref != query.table_ref || input.query_ref != query.query_ref || input.merge_ref != query.merge_ref ||
            input.key_max_u128 != query.key_max_u128 || sorted_lookup_join_roles(input.commitment_roles) != query.commitment_role_map) {
            if (error) *error = "BLOCKED: sorted-lookup provider metadata differs from RelationPlan: " + query.operation_id;
            return false;
        }
        zkdl::AuthenticatedSortedLookupDescriptor descriptor;
        if (!sorted_lookup_descriptor_from_plan(plan, statement, *operation_it->second, query, descriptor, error)) return false;
        SortedLookupScratchLayout layout;
        if (!sorted_scratch_layout(config, input, layout, error) || layout.roles != query.commitment_role_map || layout.key_max_u128 != query.key_max_u128 ||
            input.layout_digest != query.layout_digest) {
            if (error && error->empty()) *error = "BLOCKED: sorted-lookup scratch layout is not plan-bound";
            return false;
        }
        std::array<zkdl::byte, 32U> expected_scratch{}, actual_scratch{};
        if (!parse_digest(input.scratch_digest, expected_scratch)) {
            if (error && error->empty()) *error = "BLOCKED: sorted-lookup scratch digest mismatch";
            return false;
        }
        const auto scratch_digest_start = std::chrono::steady_clock::now();
        const bool scratch_digest_read = streaming_file_digest(layout.path, actual_scratch, error);
        if (context.provider_file_io_decode_time_seconds != nullptr)
            *context.provider_file_io_decode_time_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - scratch_digest_start).count();
        if (!scratch_digest_read || expected_scratch != actual_scratch) {
            if (error && error->empty()) *error = "BLOCKED: sorted-lookup scratch digest mismatch";
            return false;
        }
        relation::SegmentedTensorProvider table_keys, table_active;
        const auto plan_operation = sorted_lookup_plan_operation(plan, query.operation_id);
        if (plan_operation == nullptr || !sorted_lookup_public_table_providers(tables, *plan_operation, query, table_keys, table_active, error)) return false;
        relation::AuthenticatedSortedLookupProviderSet providers;
        if (!sorted_lookup_prover_providers(descriptor, layout, table_keys, table_active, providers, error,
                                            context.provider_file_io_decode_time_seconds)) return false;

        std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef> refs;
        std::vector<zkdl::CpuG1Point> outer;
        std::vector<FrWire> blinders;
        std::vector<std::vector<FrWire>> segment_blinders;
        std::vector<std::vector<zkdl::CpuG1Point>> segment_commitments;
        relation::RelationSegmentProgressReporter outer_commitment_progress(
            "sorted_lookup_outer_commitment");
        const auto add_outer = [&](std::uint32_t role, std::uint32_t column, const std::string &name, const relation::SegmentedTensorProvider &provider) {
            const auto commitment_operation_start = std::chrono::steady_clock::now();
            const double provider_io_before_commitment = context.provider_file_io_decode_time_seconds != nullptr
                ? *context.provider_file_io_decode_time_seconds : 0.0;
            FrWire blinder = zkdl::cpu_fresh_csprng_scalar();
            while (blinder == FrWire{}) blinder = zkdl::cpu_fresh_csprng_scalar();
            relation::CommitmentWire wire;
            relation::SegmentCommitmentMaterial material;
            const std::string tensor_id = "sorted_lookup/" + query.operation_id + "/" + name;
            double commitment_crypto_seconds = 0.0;
            const relation::SegmentCommitmentProgressCallback segment_progress =
                [&tensor_id, &outer_commitment_progress](std::uint32_t completed,
                                                        std::uint32_t total) {
                    outer_commitment_progress.update(tensor_id, completed, total);
                };
            const bool commitment_built = relation::zktr_build_segment_commitment_material(
                tensor_id, 3U, provider, context.ipa_generators, context.ipa_blinding_generator,
                context.ipa_u_generator, blinder, relation_statement, wire, material,
                error, &commitment_crypto_seconds, descriptor.segment_size,
                &segment_progress, &context.ipa_generators, &msm_session);
            if (!commitment_built) return false;
            const auto point_conversion_start = std::chrono::steady_clock::now();
            const zkdl::CpuG1Point commitment = zkdl::cpu_g1_from_native(wire.commitment);
            const double point_conversion_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - point_conversion_start).count();
            if (context.commitment_open_time_seconds != nullptr)
                *context.commitment_open_time_seconds += commitment_crypto_seconds + point_conversion_seconds;
            if (outer_commitment_wall_seconds != nullptr)
                *outer_commitment_wall_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - commitment_operation_start).count();
            if (outer_commitment_file_io_seconds != nullptr && context.provider_file_io_decode_time_seconds != nullptr) {
                const double provider_io_delta = *context.provider_file_io_decode_time_seconds - provider_io_before_commitment;
                if (provider_io_delta < 0.0) {
                    if (error) *error = "sorted-lookup commitment provider I/O timing moved backwards";
                    return false;
                }
                *outer_commitment_file_io_seconds += provider_io_delta;
            }
            if (commitment.is_zero()) {
                if (error) *error = "sorted-lookup outer commitment handle is zero after shared-segment construction";
                return false;
            }
            zkdl::AuthenticatedSortedLookupCommitmentRef ref;
            ref.role = role;
            ref.column_index = column;
            ref.logical_domain = provider.logical_domain();
            ref.segment_size = provider.segment_size();
            ref.segment_count = provider.segment_count();
            refs.push_back(ref);
            outer.push_back(commitment);
            blinders.push_back(blinder);
            segment_blinders.push_back(std::move(material.blinders));
            segment_commitments.push_back(std::move(material.commitments));
            context.sorted_lookup_commitment_ids.push_back(tensor_id);
            return true;
        };
        if (!add_outer(zkdl::kSortedLookupCommitmentQueryX, 0U, "query_x", providers.query_x) ||
            !add_outer(zkdl::kSortedLookupCommitmentQueryY, 0U, "query_y", providers.query_y) ||
            !add_outer(zkdl::kSortedLookupCommitmentQueryKey, 0U, "query_key", providers.query_keys) ||
            !add_outer(zkdl::kSortedLookupCommitmentQueryPrefix, 0U, "query_prefix", providers.query_prefix) ||
            !add_outer(zkdl::kSortedLookupCommitmentMergeKey, 0U, "merge_key", providers.merge_keys) ||
            !add_outer(zkdl::kSortedLookupCommitmentMergeDelta, 0U, "merge_delta", providers.merge_delta) ||
            !add_outer(zkdl::kSortedLookupCommitmentMergeMarker, 0U, "merge_marker", providers.merge_marker) ||
            !add_outer(zkdl::kSortedLookupCommitmentMergeInverse, 0U, "merge_inverse", providers.merge_inverse) ||
            !add_outer(zkdl::kSortedLookupCommitmentMergePrefix, 0U, "merge_prefix", providers.merge_prefix)) {
            if (error && error->empty()) *error = "BLOCKED: sorted-lookup outer commitment construction failed";
            return false;
        }
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit)
            if (!add_outer(zkdl::kSortedLookupCommitmentMergeBit, bit, "merge_bit_" + std::to_string(bit), providers.merge_bits[bit])) {
                if (error && error->empty()) *error = "BLOCKED: sorted-lookup merge-bit commitment construction failed";
                return false;
            }
        context.sorted_lookup_descriptors.push_back(descriptor);
        context.sorted_lookup_providers.push_back(std::move(providers));
        context.sorted_lookup_commitment_refs.push_back(std::move(refs));
        context.sorted_lookup_outer_commitments.push_back(std::move(outer));
        context.sorted_lookup_outer_blinders.push_back(std::move(blinders));
        context.sorted_lookup_outer_segment_blinders.push_back(std::move(segment_blinders));
        context.sorted_lookup_outer_segment_commitments.push_back(std::move(segment_commitments));
    }
    return true;
}

bool sorted_lookup_build_verifier_context(const StaticTables &tables, const RelationPlan &plan, const PublicStatement &statement,
                                          const std::vector<relation::OperationDescriptor> &operations, relation::VerifierContext &context,
                                          std::string *error) {
    std::map<std::string, const relation::OperationDescriptor *> sorted_operations;
    for (const auto &operation : operations)
        if (operation.batch_family == "authenticated_sorted_lookup" && !sorted_operations.emplace(operation.id, &operation).second) {
            if (error) *error = "BLOCKED: duplicate sorted-lookup operation in verifier plan";
            return false;
        }
    if (sorted_operations.size() != plan.sorted_lookup.queries.size()) {
        if (error) *error = "BLOCKED: verifier sorted-lookup operation count does not match RelationPlan";
        return false;
    }
    context.sorted_lookup_descriptors.clear();
    context.sorted_lookup_table_keys.clear();
    context.sorted_lookup_table_active.clear();
    context.sorted_lookup_commitment_refs.clear();
    context.sorted_lookup_outer_commitments.clear();
    context.sorted_lookup_commitment_ids.clear();
    for (const auto &query : plan.sorted_lookup.queries) {
        const auto operation_it = sorted_operations.find(query.operation_id);
        const auto plan_operation = sorted_lookup_plan_operation(plan, query.operation_id);
        if (operation_it == sorted_operations.end() || plan_operation == nullptr) {
            if (error) *error = "BLOCKED: verifier sorted-lookup operation is missing: " + query.operation_id;
            return false;
        }
        zkdl::AuthenticatedSortedLookupDescriptor descriptor;
        if (!sorted_lookup_descriptor_from_plan(plan, statement, *operation_it->second, query, descriptor, error)) return false;
        relation::SegmentedTensorProvider table_keys, table_active;
        if (!sorted_lookup_public_table_providers(tables, *plan_operation, query, table_keys, table_active, error)) return false;
        context.sorted_lookup_descriptors.push_back(std::move(descriptor));
        context.sorted_lookup_table_keys.push_back(std::move(table_keys));
        context.sorted_lookup_table_active.push_back(std::move(table_active));
        const auto add_id = [&](const std::string &name) {
            context.sorted_lookup_commitment_ids.push_back("sorted_lookup/" + query.operation_id + "/" + name);
        };
        add_id("query_x");
        add_id("query_y");
        add_id("query_key");
        add_id("query_prefix");
        add_id("merge_key");
        add_id("merge_delta");
        add_id("merge_marker");
        add_id("merge_inverse");
        add_id("merge_prefix");
        for (std::uint32_t bit = 0U; bit < query.x_bits + query.y_bits; ++bit) add_id("merge_bit_" + std::to_string(bit));
    }
    return true;
}

bool decode_native_witness(const NativeRelationWitnessInput &input, NativeRelationWitness &output, std::string *error,
                           double *provider_file_io_decode_seconds = nullptr) {
    output = NativeRelationWitness{};
    output.registry_path = input.registry_path;
    output.srs_digest = input.srs_digest;
    output.registry_digest = input.registry_digest;
    output.model_blinder_digest = input.model_blinder_digest;
    output.srs_authenticated = input.srs_authenticated;
    output.registry_authenticated = input.registry_authenticated;
    output.model_blinders_authenticated = input.model_blinders_authenticated;
    output.model_blinders_present = !input.model_blinders.empty();

    if (!input.hiding_commitment_records.empty()) {
        if (error) *error = "BLOCKED: formal GATR rejects legacy hiding-record ABI payloads";
        return false;
    }

    output.tensors.reserve(input.tensors.size());
    for (const auto &supplied : input.tensors) {
        relation::TensorWitness tensor;
        tensor.id = supplied.id;
        tensor.visibility = supplied.visibility;
        tensor.domain_size = supplied.domain_size;
        if (supplied.storage) {
            if (!supplied.values.empty() || supplied.logical_shape.empty() || supplied.logical_shape.size() != supplied.padded_shape.size()) { if (error) *error = "native compact tensor ABI is ambiguous"; return false; }
            std::uint32_t segment_size = 0U, segment_count = 0U;
            if (!relation::SegmentedTensorProvider::shape(tensor.domain_size, 1U << 18U, segment_size, segment_count, error)) return false;
            tensor.stored_provider = relation::SegmentedTensorProvider(tensor.domain_size, segment_size, [storage = supplied.storage, logical = supplied.logical_shape, padded = supplied.padded_shape, provider_file_io_decode_seconds](std::uint64_t offset, std::uint32_t count, std::vector<FrWire>& output, std::string* reader_error) { return storage->read_padded(logical, padded, offset, count, output, reader_error, provider_file_io_decode_seconds); });
        } else tensor.values.assign(supplied.values.begin(), supplied.values.end());
        output.tensors.push_back(std::move(tensor));
    }
    output.tensor_shapes.reserve(input.tensor_shapes.size());
    for (const auto &supplied : input.tensor_shapes) {
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
    for (const auto &encoded : input.operations) {
        relation::OperationDescriptor operation;
        std::size_t offset = 0U;
        if (!relation::read_operation_strict(encoded, offset, operation) || offset != encoded.size()) {
            if (error) *error = "BLOCKED: native relation operation payload is not canonical";
            return false;
        }
        output.operations.push_back(std::move(operation));
    }
    output.sorted_lookup_providers = input.sorted_lookup_providers;
    for (const auto &provider : output.sorted_lookup_providers) {
        if (provider.operation_id.empty() || provider.scratch_path.empty() || provider.scratch_digest.empty() || provider.layout_digest.empty() ||
            provider.segment_size == 0U || provider.table_domain == 0U || provider.query_domain == 0U || provider.merge_domain == 0U ||
            provider.commitment_roles.empty() || !provider.owner_only || !provider.active_prefix || !provider.canonical_padding) {
            if (error) *error = "BLOCKED: native sorted-lookup provider metadata is incomplete";
            return false;
        }
    }
    output.registered_model_commitments.reserve(input.registered_model_commitments.size());
    for (const auto &supplied : input.registered_model_commitments) {
        relation::CommitmentWire commitment;
        if (!native_commitment_to_relation(supplied, commitment, error)) return false;
        output.registered_model_commitments.push_back(std::move(commitment));
    }
    output.blinding_scalars.clear();
    for (const auto &supplied : input.model_blinders) {
        if (supplied.tensor_id.empty() || supplied.registry_digest != output.registry_digest || !zkdl::fr_canonical(supplied.blinder) ||
            zkdl::fr_compare(supplied.blinder, FrWire{}) == 0 || !output.blinding_scalars.emplace(supplied.tensor_id, supplied.blinder).second) {
            if (error) *error = "BLOCKED: native model blinder payload is not canonical or registry-bound";
            return false;
        }
    }
    output.generators.reserve(input.generators.size());
    for (const auto &encoded : input.generators) {
        G1Jacobian_t generator{};
        std::size_t offset = 0U;
        if (!zkdl::read_g1(encoded, offset, generator) || offset != encoded.size()) {
            if (error) *error = "BLOCKED: native SRS generator payload is not canonical";
            return false;
        }
        output.generators.push_back(zkdl::cpu_g1_from_native(generator));
    }
    output.ipa_generators.reserve(input.ipa_generators.size());
    for (const auto &encoded : input.ipa_generators) {
        G1Jacobian_t generator{};
        std::size_t offset = 0U;
        if (!zkdl::read_g1(encoded, offset, generator) || offset != encoded.size()) {
            if (error) *error = "BLOCKED: native IPA generator payload is not canonical";
            return false;
        }
        output.ipa_generators.push_back(zkdl::cpu_g1_from_native(generator));
    }
    auto decode_single_ipa_point = [&](const std::vector<std::uint8_t> &encoded, zkdl::CpuG1Point &point, const char *label) -> bool {
        if (encoded.empty()) {
            if (error) *error = std::string("BLOCKED: authenticated IPA ") + label + " is missing";
            return false;
        }
        G1Jacobian_t native{};
        std::size_t offset = 0U;
        if (!zkdl::read_g1(encoded, offset, native) || offset != encoded.size()) {
            if (error) *error = std::string("BLOCKED: authenticated IPA ") + label + " is not canonical";
            return false;
        }
        point = zkdl::cpu_g1_from_native(native);
        if (point.is_zero()) {
            if (error) *error = std::string("BLOCKED: authenticated IPA ") + label + " is zero";
            return false;
        }
        return true;
    };
    if (!input.ipa_generators.empty() || !input.ipa_blinding_generator.empty() || !input.ipa_u_generator.empty()) {
        if (input.ipa_generators.empty() || !decode_single_ipa_point(input.ipa_blinding_generator, output.ipa_blinding_generator, "blinding generator") ||
            !decode_single_ipa_point(input.ipa_u_generator, output.ipa_u_generator, "U generator"))
            return false;
    }
    return true;
}

struct NativeProveContextTiming {
    double relation_context_wall_seconds = 0.0;
    double blinder_sampling_seconds = 0.0;
    double sorted_lookup_outer_commitment_wall_seconds = 0.0;
    double sorted_lookup_outer_commitment_file_io_seconds = 0.0;
    double explicit_file_io_seconds = 0.0;
};

bool context_from_request(const NativeProveRequest &request, relation::ProverContext &context,
                          NativeProveContextTiming &timing, std::string *error) {
    if (!request.config || !request.graph || !request.model || !request.tables || !request.relation || !request.trace || !request.statement ||
        !request.witness) {
        if (error) *error = "BLOCKED: native prove request lacks canonical tensors/witness/selectors/public Y";
        return false;
    }
    context.sumcheck_spill_directory =
        (request.config->run_root / ".private_native").string();
    NativeRelationWitness witness;
    if (!decode_native_witness(*request.witness, witness, error,
                               context.provider_file_io_decode_time_seconds)) return false;
    const double provider_io_after_witness_decode = context.provider_file_io_decode_time_seconds != nullptr
        ? *context.provider_file_io_decode_time_seconds : 0.0;
    const auto relation_context_start = std::chrono::steady_clock::now();
    if (!resource_admission(*request.config, witness, true, error)) return false;
    if (!statement_from_adapter(*request.config, *request.graph, *request.model, *request.tables, *request.relation, *request.statement, witness,
                                context.statement, error))
        return false;
    context.tensors = std::move(witness.tensors);
    const auto blinder_sampling_start = std::chrono::steady_clock::now();
    for (const auto &tensor : context.tensors) {
        if (tensor.visibility == 1U) continue;
        if (tensor.visibility == 2U) {
            if (witness.blinding_scalars.find(tensor.id) == witness.blinding_scalars.end()) {
                if (error) *error = "BLOCKED: model tensor has no authenticated prover-only blinder: " + tensor.id;
                return false;
            }
            continue;
        }
        if (tensor.visibility != 3U) {
            if (error) *error = "BLOCKED: relation tensor has unsupported visibility: " + tensor.id;
            return false;
        }
        try {

            witness.blinding_scalars.emplace(tensor.id, zkdl::cpu_fresh_csprng_scalar());
        } catch (const std::exception &exception) {
            if (error) *error = exception.what();
            return false;
        }
    }
    timing.blinder_sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - blinder_sampling_start).count();
    if (!build_canonical_operations(*request.config, *request.tables, *request.relation, context.tensors, context.operations, request.graph, error))
        return false;
    if (!shape_contract_from_witness(*request.relation, witness, context.tensor_descriptors, error)) return false;

    if (!witness.operations.empty()) {
        if (!operation_set_matches_plan(*request.relation, witness.operations, error)) return false;
        std::map<std::string, const relation::OperationDescriptor *> supplied;
        for (const auto &operation : witness.operations) supplied.emplace(operation.id, &operation);
        for (const auto &operation : context.operations) {
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
    context.ipa_generators = witness.authenticated_ipa_basis;
    context.ipa_blinding_generator = witness.authenticated_ipa_basis.h();
    context.ipa_u_generator = witness.authenticated_ipa_basis.u();
    if (!sorted_lookup_build_prover_context(*request.config, *request.tables, *request.relation, *request.statement, context.statement, witness, context,
                                            error, &timing.sorted_lookup_outer_commitment_wall_seconds,
                                            &timing.sorted_lookup_outer_commitment_file_io_seconds))
        return false;
    context.registered_model_commitments = witness.registered_model_commitments;
    context.hiding_commitment_records = witness.hiding_commitment_records;
    context.blinding_scalars = witness.blinding_scalars;
    for (const auto &tensor : context.tensors)
        if (tensor.visibility == 1U && !tensor.values.empty() &&
            tensor.id != "public_Y")
            context.public_tensor_values.emplace(tensor.id, tensor.values);
    for (const auto &descriptor : context.tensor_descriptors) {
        if (graph_public_descriptor_id(descriptor.id)) {
            relation::SegmentedTensorProvider provider;
            if (!graph_public_provider(*request.graph, descriptor, context.ipa_generators.size(), provider, error)) return false;
            context.public_tensor_providers.emplace(descriptor.id, provider);
            context.graph_public_tensor_ids.insert(descriptor.id);
        } else if (descriptor.id == "output_head_count") {
            relation::SegmentedTensorProvider provider;
            if (descriptor.logical_shape != std::vector<std::uint64_t>{1U} || !public_logical_provider(descriptor, context.ipa_generators.size(), [heads = request.config->architecture.output_heads](std::uint64_t) { return relation::fr_from_signed_i64(static_cast<std::int64_t>(heads)); }, provider, error)) return false;
            context.public_tensor_providers.emplace(descriptor.id, provider);
        }
    }
    if (!bind_authenticated_public_material(*request.tables, context.tensor_descriptors, context.public_tensor_values, error)) return false;
    for (const auto &tensor : context.tensors) {
        const relation::TensorDescriptor *descriptor = relation::find_tensor_descriptor(context.tensor_descriptors, tensor.id);
        if (descriptor == nullptr || descriptor->domain_size != tensor.domain_size || descriptor->visibility != tensor.visibility) {
            if (error) *error = "native tensor witness does not match its authenticated shape descriptor: " + tensor.id;
            return false;
        }
    }
    std::string shape_error;
    for (const auto &tensor : context.tensors)
        if (!tensor.valid(&shape_error)) {
            if (error) *error = "native relation tensor rejected: " + shape_error;
            return false;
        }
    timing.relation_context_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - relation_context_start).count();
    if (context.provider_file_io_decode_time_seconds != nullptr)
        timing.explicit_file_io_seconds = *context.provider_file_io_decode_time_seconds - provider_io_after_witness_decode;
    return true;
}

bool context_from_verify(const NativeVerifyRequest &request, relation::VerifierContext &context, std::string *error) {
    if (!request.config || !request.graph || !request.statement || !request.public_context || !request.relation || !request.tables) {
        if (error) *error = "BLOCKED: independent verifier lacks public graph/RelationPlan/context/SRS/registry";
        return false;
    }
    NativeRelationWitness supplied;
    if (!decode_native_witness(*request.public_context, supplied, error)) return false;
    if (!supplied.srs_authenticated || !supplied.registry_authenticated || supplied.generators.empty() || supplied.ipa_generators.empty() ||
        supplied.ipa_generators.size() != supplied.generators.size() || supplied.ipa_blinding_generator.is_zero() || supplied.ipa_u_generator.is_zero()) {
        if (error) *error = "BLOCKED: independent verifier context has no authenticated SRS/IPA G/H/U basis";
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
        request.statement->integer_bound_ledger_descriptor_digest != request.relation->integer_bound_ledger_descriptor_digest ||
        request.statement->relation_registry_digest != request.relation->registry_digest ||
        request.statement->relation_terminal_claims_digest != request.relation->terminal_claims_digest ||
        request.statement->relation_mapping_digest != request.relation->mapping_digest ||
        request.statement->relation_selector_digest != request.relation->selector_digest ||
        request.statement->relation_degree_digest != request.relation->degree_digest ||
        request.statement->verifier_registry_digest != request.config->verifier_registry_digest) {
        if (error) *error = "independent verifier statement/resource identity is not bound to the authenticated plan";
        return false;
    }

    if (!resource_admission(*request.config, supplied, false, error)) return false;
    if (supplied.tensor_shapes.empty()) {
        if (error) *error = "BLOCKED: independent verifier lacks canonical tensor shape/padding descriptors";
        return false;
    }
    if (!shape_contract_from_plan(*request.relation, supplied.tensor_shapes, context.tensor_descriptors, error)) return false;
    for (const auto &tensor : supplied.tensors)
        if (tensor.visibility != 1U) {
            if (error) *error = "independent verifier context contains private witness tensor";
            return false;
        }
    std::vector<relation::TensorWitness> shape_tensors;
    shape_tensors.reserve(supplied.tensor_shapes.size());
    for (const auto &shape : supplied.tensor_shapes) {
        relation::TensorWitness tensor;
        tensor.id = shape.id;
        tensor.domain_size = shape.domain_size;
        tensor.visibility = shape.visibility;
        shape_tensors.push_back(std::move(tensor));
    }

    for (const auto &tensor : supplied.tensors) {
        const relation::TensorDescriptor *descriptor = relation::find_tensor_descriptor(context.tensor_descriptors, tensor.id);
        if (descriptor == nullptr || descriptor->visibility != tensor.visibility || descriptor->domain_size != tensor.domain_size) {
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
    if (!parse_digest(request.config->verifier_registry_digest, expected_registry) || supplied.registry_digest != expected_registry) {
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
    const auto public_y = std::find_if(supplied.tensors.begin(), supplied.tensors.end(),
                                       [](const relation::TensorWitness &tensor) { return tensor.id == "public_Y" && tensor.visibility == 1U; });
    context.statement.public_output = public_y == supplied.tensors.end() ? std::vector<zkdl::byte>() : relation::encode_public_field_vector(public_y->values);
    if (context.statement.public_output.empty()) {
        if (error) *error = "BLOCKED: independent verifier lacks public_Y field witness";
        return false;
    }
    context.statement.public_output_digest = zkdl::sha256(context.statement.public_output);
    if (request.statement->public_output_digest.empty() || request.statement->public_output_digest != digest_hex(context.statement.public_output_digest)) {
        if (error) *error = "independent verifier public_Y digest mismatch";
        return false;
    }
    if (!build_canonical_operations(*request.config, *request.tables, *request.relation, shape_tensors, context.operations, request.graph, error)) return false;
    if (!supplied.operations.empty()) {
        if (!operation_set_matches_plan(*request.relation, supplied.operations, error)) return false;
        std::map<std::string, const relation::OperationDescriptor *> supplied_operations;
        for (const auto &operation : supplied.operations) supplied_operations.emplace(operation.id, &operation);
        for (const auto &operation : context.operations) {
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
    context.ipa_generators = supplied.authenticated_ipa_basis;
    context.ipa_blinding_generator = supplied.authenticated_ipa_basis.h();
    context.ipa_u_generator = supplied.authenticated_ipa_basis.u();
    if (!supplied.sorted_lookup_providers.empty()) {
        if (error) *error = "BLOCKED: independent verifier context carries private sorted-lookup scratch";
        return false;
    }
    if (!sorted_lookup_build_verifier_context(*request.tables, *request.relation, *request.statement, context.operations, context, error)) return false;
    context.registered_model_commitments = supplied.registered_model_commitments;
    context.hiding_commitment_records = supplied.hiding_commitment_records;
    for (const auto &tensor : supplied.tensors)
        if (!tensor.values.empty() && tensor.id != "public_Y")
            context.public_tensor_values.emplace(tensor.id, tensor.values);
    for (const auto &descriptor : context.tensor_descriptors) {
        if (graph_public_descriptor_id(descriptor.id)) {
            relation::SegmentedTensorProvider provider;
            if (!graph_public_provider(*request.graph, descriptor, context.ipa_generators.size(), provider, error)) return false;
            context.public_tensor_providers.emplace(descriptor.id, provider);
            context.graph_public_tensor_ids.insert(descriptor.id);
        } else if (descriptor.id == "output_head_count") {
            relation::SegmentedTensorProvider provider;
            if (descriptor.logical_shape != std::vector<std::uint64_t>{1U} || !public_logical_provider(descriptor, context.ipa_generators.size(), [heads = request.config->architecture.output_heads](std::uint64_t) { return relation::fr_from_signed_i64(static_cast<std::int64_t>(heads)); }, provider, error)) return false;
            context.public_tensor_providers.emplace(descriptor.id, provider);
        }
    }
    if (!bind_authenticated_public_material(*request.tables, context.tensor_descriptors, context.public_tensor_values, error)) return false;
    context.tensor_ids.clear();
    context.tensor_ids.reserve(shape_tensors.size());
    context.tensor_visibility.clear();
    for (const auto &tensor : shape_tensors) {
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

}

bool zkdl_gat_native_relation_plan_metrics(const Config &config, const Graph &graph,
                                            const StaticTables &tables, const RelationPlan &plan,
                                            NativeRelationPlanMetrics *metrics, std::string *error,
                                            bool analysis_only_shape) {
    if (metrics == nullptr) {
        if (error) *error = "native relation admission metrics output is null";
        return false;
    }
    *metrics = NativeRelationPlanMetrics{};
    std::vector<relation::TensorWitness> shape_tensors;
    std::map<std::string, std::uint64_t> domains;
    DescriptorBackingInterner shape_interner(plan.registry_digest, &graph,
                                              analysis_only_shape);
    std::vector<std::uint64_t> shape_operation_wire_deltas;
    constexpr std::uint64_t kSegmentSize = 1ULL << 18U;
    const auto add = [&](std::uint64_t *target, std::uint64_t value, const char *label) -> bool {
        if (value > std::numeric_limits<std::uint64_t>::max() - *target) {
            if (error) *error = std::string("native relation admission metric overflows: ") + label;
            return false;
        }
        *target += value;
        return true;
    };
    const auto checked_add_wire = [&](std::uint64_t left, std::uint64_t right,
                                      std::uint64_t &sum, const char *label) -> bool {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            if (error) *error = std::string("native typed proof-wire bound overflows: ") + label;
            return false;
        }
        sum = left + right;
        return true;
    };
    const auto checked_mul_wire = [&](std::uint64_t left, std::uint64_t right,
                                      std::uint64_t &product, const char *label) -> bool {
        if (!checked_mul_u64(left, right, product)) {
            if (error) *error = std::string("native typed proof-wire bound overflows: ") + label;
            return false;
        }
        return true;
    };
    const auto log2_domain = [](std::uint64_t domain, std::uint32_t &bits) {
        if (!native_power_of_two(domain)) return false;
        bits = 0U;
        while (domain > 1U) { domain >>= 1U; ++bits; }
        return true;
    };
    std::uint64_t ordinary_segmented_sumcheck_peak = 0U;
    using HyraxShapeKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<HyraxShapeKey, NativeRelationPlanMetrics::HyraxOpeningSegmentBucket>
        hyrax_opening_shape_counts;
    using SumcheckShapeKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<SumcheckShapeKey, NativeRelationPlanMetrics::SumcheckShapeBucket>
        sumcheck_shape_counts;
    const auto record_segmented_sumcheck_peak = [&](std::uint64_t tensor_count,
                                                     std::uint64_t domain,
                                                     std::uint64_t segment_size) -> bool {
        std::uint64_t peak = 0U;
        if (!relation::zksc_selected_sumcheck_peak_bytes(
                tensor_count, domain, segment_size, peak)) {
            if (error) *error = "native segmented Sumcheck selected live shape overflows";
            return false;
        }
        ordinary_segmented_sumcheck_peak =
            std::max(ordinary_segmented_sumcheck_peak, peak);
        return true;
    };
    metrics->sumcheck_folded_state_budget_bytes =
        relation::kZkSumcheckFoldedStateBudgetBytes;
    const auto record_sumcheck_shape = [&](std::uint64_t provider_count,
                                            std::uint64_t domain,
                                            std::uint64_t segment_size,
                                            bool force_cached_state) -> bool {
        if (provider_count == 0U || domain < 2U || !native_power_of_two(domain) ||
            segment_size == 0U || !native_power_of_two(segment_size) ||
            segment_size > domain || domain % segment_size != 0U ||
            segment_size > kSegmentSize) {
            if (error) *error = "native Sumcheck work profile received a non-canonical shape";
            return false;
        }
        std::uint64_t folded_peak = 0U, streaming_peak = 0U;
        if (!relation::zksc_folded_state_peak_bytes(
                provider_count, domain, segment_size, folded_peak) ||
            !relation::zksc_segmented_streaming_peak_bytes(
                provider_count, segment_size, streaming_peak)) {
            if (error) *error = "native Sumcheck work profile overflows its checked source formula";
            return false;
        }
        const bool cached = force_cached_state ||
            folded_peak <= relation::kZkSumcheckFoldedStateBudgetBytes;
        std::uint32_t bits = 0U;
        if (!log2_domain(domain, bits)) {
            if (error) *error = "native Sumcheck work profile could not derive logical rounds";
            return false;
        }
        std::uint64_t provider_values = 0U, passes = 2U, scan_values = 0U;
        if (!checked_mul_wire(provider_count, domain, provider_values,
                              "Sumcheck provider value slots") ||
            !checked_mul_wire(provider_values, passes, scan_values,
                              "Sumcheck provider scan value slots")) return false;
        std::uint64_t scan_bytes = 0U;
        if (!checked_mul_wire(scan_values, sizeof(FrWire), scan_bytes,
                              "Sumcheck provider scan bytes")) return false;
        std::uint64_t spill_disk_peak = 0U, spill_ram_peak = 0U;
        std::uint64_t spill_state_io_bytes = 0U, spill_source_bytes = 0U;
        if (!cached &&
            (!relation::zksc_spilled_state_disk_peak_bytes(
                 provider_count, domain, spill_disk_peak) ||
             !relation::zksc_spilled_state_ram_peak_bytes(
                 provider_count, domain, segment_size, spill_ram_peak) ||
             !relation::zksc_spilled_state_io_bytes_upper_bound(
                 provider_count, domain, spill_state_io_bytes) ||
             !checked_mul_wire(scan_values,
                 SortedLookupScratchLayout::kRecordBytes, spill_source_bytes,
                 "Sumcheck spill provider source bytes"))) {
            if (error) *error = "native Sumcheck spill source envelope overflows";
            return false;
        }
        if (cached) {
            if (!add(&metrics->sumcheck_cached_call_count, 1U,
                     "Sumcheck cached call count") ||
                (force_cached_state &&
                 !add(&metrics->sumcheck_forced_cached_call_count, 1U,
                      "Sumcheck forced-cache call count"))) return false;
            if (!add(&metrics->sumcheck_cached_provider_value_slots_upper_bound,
                     scan_values, "cached Sumcheck provider scan slots") ||
                !add(&metrics->sumcheck_cached_provider_scan_bytes_upper_bound,
                     scan_bytes, "cached Sumcheck provider scan bytes")) return false;
        } else if (!add(&metrics->sumcheck_spill_call_count, 1U,
                        "Sumcheck spill call count") ||
                   !add(&metrics->sumcheck_spill_provider_value_slots_upper_bound,
                        scan_values, "spill Sumcheck provider scan slots") ||
                   !add(&metrics->sumcheck_spill_provider_scan_bytes_upper_bound,
                        scan_bytes, "spill Sumcheck provider field bytes") ||
                   !add(&metrics->sumcheck_spill_provider_source_bytes_upper_bound,
                        spill_source_bytes, "spill Sumcheck provider source bytes") ||
                   !add(&metrics->sumcheck_spill_storage_io_bytes_upper_bound,
                        spill_state_io_bytes, "Sumcheck spill state storage I/O bytes")) return false;
        metrics->sumcheck_max_spill_disk_peak_bytes_upper_bound = std::max(
            metrics->sumcheck_max_spill_disk_peak_bytes_upper_bound, spill_disk_peak);
        metrics->sumcheck_max_spill_ram_peak_bytes_upper_bound = std::max(
            metrics->sumcheck_max_spill_ram_peak_bytes_upper_bound, spill_ram_peak);
        metrics->sumcheck_max_folded_state_peak_bytes_upper_bound =
            std::max(metrics->sumcheck_max_folded_state_peak_bytes_upper_bound,
                     folded_peak);
        metrics->sumcheck_max_streaming_peak_bytes_upper_bound =
            std::max(metrics->sumcheck_max_streaming_peak_bytes_upper_bound,
                     streaming_peak);
        ordinary_segmented_sumcheck_peak = std::max(
            ordinary_segmented_sumcheck_peak,
            cached ? folded_peak : std::max(streaming_peak, spill_ram_peak));
        const SumcheckShapeKey key{domain, segment_size, provider_count};
        auto [entry, inserted] = sumcheck_shape_counts.emplace(key,
            NativeRelationPlanMetrics::SumcheckShapeBucket{});
        auto &bucket = entry->second;
        if (inserted) {
            bucket.logical_domain = domain;
            bucket.segment_size = segment_size;
            bucket.provider_count = provider_count;
        }
        if (force_cached_state) {
            if (!add(&bucket.cached_call_count, 1U,
                     "Sumcheck histogram cached count") ||
                !add(&bucket.forced_cached_call_count, 1U,
                     "Sumcheck histogram forced-cache count")) return false;
        } else if (cached) {
            if (!add(&bucket.cached_call_count, 1U,
                     "Sumcheck histogram cached count")) return false;
        } else if (!add(&bucket.spill_call_count, 1U,
                        "Sumcheck histogram spill count") ||
                   !add(&bucket.spill_provider_value_slots_upper_bound,
                        scan_values, "Sumcheck histogram spill provider scan slots"))
            return false;
        std::uint64_t *bucket_values = cached
            ? &bucket.cached_provider_value_slots_upper_bound
            : &bucket.spill_provider_value_slots_upper_bound;
        if (!cached) return true;
        return add(bucket_values, scan_values,
                   "Sumcheck histogram provider scan slots");
    };
    const auto record_hyrax_opening_shape = [&](std::uint64_t logical_domain,
                                                 std::uint64_t segment_size,
                                                 std::uint64_t computed_openings,
                                                 std::uint64_t retained_objects,
                                                 std::uint64_t terminal_mirror_copies) -> bool {
        if (logical_domain == 0U || !native_power_of_two(logical_domain) ||
            segment_size == 0U || !native_power_of_two(segment_size) ||
            segment_size > logical_domain || logical_domain % segment_size != 0U ||
            segment_size > kSegmentSize || retained_objects < computed_openings ||
            terminal_mirror_copies > retained_objects) {
            if (error) *error = "native Hyrax opening profile received a non-canonical shape";
            return false;
        }
        std::uint32_t domain_bits = 0U, inner_bits = 0U;
        if (!log2_domain(logical_domain, domain_bits) ||
            !log2_domain(segment_size, inner_bits)) {
            if (error) *error = "native Hyrax opening profile could not derive point lengths";
            return false;
        }
        const std::uint32_t padded_generator_size = zkdl::cpu_ipa_padded_size(
            checked_native_u32(segment_size, "Hyrax segment generator size"));
        std::uint32_t fold_bits = 0U;
        for (std::uint32_t value = padded_generator_size; value > 1U; value >>= 1U)
            ++fold_bits;
        std::uint64_t round_points_per_opening = 0U, point_values_per_opening = 0U;
        if (!checked_mul_wire(fold_bits, 2U, round_points_per_opening,
                              "Hyrax retained round points") ||
            !checked_add_wire(inner_bits, domain_bits, point_values_per_opening,
                              "Hyrax retained evaluation-point elements")) return false;
        std::uint64_t retained_round_points = 0U, retained_point_values = 0U;
        std::uint64_t round_bytes = 0U, point_bytes = 0U, object_bytes = 0U;
        if (!checked_mul_wire(retained_objects, round_points_per_opening,
                              retained_round_points,
                              "Hyrax retained round-point count") ||
            !checked_mul_wire(retained_objects, point_values_per_opening,
                              retained_point_values,
                              "Hyrax retained evaluation-point count") ||
            !checked_mul_wire(retained_round_points, sizeof(G1Jacobian_t),
                              round_bytes, "Hyrax retained round-vector bytes") ||
            !checked_mul_wire(retained_point_values, sizeof(Fr_t), point_bytes,
                              "Hyrax retained point-vector bytes") ||
            !checked_mul_wire(retained_objects,
                              sizeof(zkdl::HyraxOpeningProofV3), object_bytes,
                              "Hyrax retained opening-object bytes")) return false;
        if (!add(&metrics->computed_hyrax_opening_count, computed_openings,
                 "computed Hyrax opening count") ||
            !add(&metrics->retained_hyrax_opening_object_count, retained_objects,
                 "retained Hyrax opening object count") ||
            !add(&metrics->retained_hyrax_terminal_mirror_copy_count,
                 terminal_mirror_copies, "retained Hyrax terminal mirror count") ||
            !add(&metrics->hyrax_round_vector_capacity_bytes_upper_bound,
                 round_bytes, "Hyrax round-vector resident capacity") ||
            !add(&metrics->hyrax_point_vector_capacity_bytes_upper_bound,
                 point_bytes, "Hyrax point-vector resident capacity") ||
            !add(&metrics->hyrax_opening_object_header_bytes_upper_bound,
                 object_bytes, "Hyrax opening-object resident headers")) return false;
        std::uint64_t one_round_bytes = 0U, one_point_bytes = 0U, one_opening_bytes = 0U;
        if (!checked_mul_wire(round_points_per_opening, sizeof(G1Jacobian_t),
                              one_round_bytes, "one Hyrax opening round vectors") ||
            !checked_mul_wire(point_values_per_opening, sizeof(Fr_t),
                              one_point_bytes, "one Hyrax opening point vectors") ||
            !checked_add_wire(one_round_bytes, one_point_bytes, one_opening_bytes,
                              "one Hyrax opening vector payload")) return false;
        metrics->hyrax_maximum_single_opening_vector_bytes_upper_bound = std::max(
            metrics->hyrax_maximum_single_opening_vector_bytes_upper_bound,
            one_opening_bytes);
        metrics->total_hyrax_opening_segment_count =
            metrics->computed_hyrax_opening_count;
        const HyraxShapeKey key{logical_domain, segment_size, padded_generator_size};
        auto [entry, inserted] = hyrax_opening_shape_counts.emplace(key,
            NativeRelationPlanMetrics::HyraxOpeningSegmentBucket{});
        auto &bucket = entry->second;
        if (inserted) {
            bucket.logical_domain = logical_domain;
            bucket.segment_size = segment_size;
            bucket.padded_generator_size = padded_generator_size;
        }
        return add(&bucket.computed_opening_count, computed_openings,
                   "Hyrax histogram computed openings") &&
               add(&bucket.retained_object_count, retained_objects,
                   "Hyrax histogram retained objects") &&
               add(&bucket.terminal_mirror_copy_count, terminal_mirror_copies,
                   "Hyrax histogram terminal mirrors") &&
               add(&bucket.retained_round_vector_bytes_upper_bound, round_bytes,
                   "Hyrax histogram round-vector bytes") &&
               add(&bucket.retained_point_vector_bytes_upper_bound, point_bytes,
                   "Hyrax histogram point-vector bytes");
    };
    const auto record_terminal_hyrax_openings = [&](std::uint64_t logical_domain,
                                                     std::uint64_t segment_size,
                                                     std::uint64_t provider_count,
                                                     std::uint64_t *category_counter,
                                                     const char *label) -> bool {
        if (provider_count == 0U || segment_size == 0U ||
            logical_domain % segment_size != 0U) {
            if (error) *error = std::string("native terminal opening shape is invalid: ") + label;
            return false;
        }
        const std::uint64_t segments = logical_domain / segment_size;
        std::uint64_t computed = 0U, retained_per_provider = 0U, retained = 0U;
        if (!checked_mul_wire(provider_count, segments, computed,
                              "terminal Hyrax computed openings") ||
            !checked_add_wire(segments, 1U, retained_per_provider,
                              "terminal Hyrax mirrored opening count") ||
            !checked_mul_wire(provider_count, retained_per_provider, retained,
                              "terminal Hyrax retained objects") ||
            category_counter == nullptr ||
            !add(category_counter, computed, label)) return false;
        return record_hyrax_opening_shape(
            logical_domain, segment_size, computed, retained, provider_count);
    };
    const auto hyrax_wire_bytes_for_segment = [&](std::uint64_t domain,
                                      std::uint64_t segment_size,
                                      std::uint64_t &bytes,
                                      std::uint64_t &segments) -> bool {
        if (domain == 0U || !native_power_of_two(domain) || domain > kMaxNativeDomain ||
            segment_size == 0U || !native_power_of_two(segment_size) ||
            segment_size > domain || domain % segment_size != 0U ||
            segment_size > kSegmentSize) {
            if (error) *error = "native Hyrax wire profile received a non-canonical domain";
            return false;
        }
        segments = domain / segment_size;
        std::uint32_t domain_bits = 0U, inner_bits = 0U;
        if (!log2_domain(domain, domain_bits) || !log2_domain(segment_size, inner_bits)) {
            if (error) *error = "native Hyrax wire profile could not derive point lengths";
            return false;
        }
        // The v3 codec pads the IPA fold domain to next_pow2(segment_size + 1).
        std::uint32_t fold_bits = 0U;
        std::uint64_t padded = 1U;
        while (padded < segment_size + 1U) { padded <<= 1U; ++fold_bits; }
        std::uint64_t point_bits = 0U, round_bytes = 0U;
        if (!checked_add_wire(inner_bits, domain_bits - inner_bits, point_bits,
                              "Hyrax point bits") ||
            !checked_mul_wire(32U, point_bits, round_bytes, "Hyrax point bytes") ||
            !checked_add_wire(556U, round_bytes, bytes, "Hyrax fixed/point bytes")) return false;
        std::uint64_t folds = 0U;
        if (!checked_mul_wire(96U, fold_bits, folds, "Hyrax fold bytes") ||
            !checked_add_wire(bytes, folds, bytes, "Hyrax total bytes")) return false;
        return true;
    };
    const auto hyrax_wire_bytes = [&](std::uint64_t domain, std::uint64_t &bytes,
                                      std::uint64_t &segments) -> bool {
        return hyrax_wire_bytes_for_segment(domain, std::min(domain, kSegmentSize),
                                            bytes, segments);
    };
    const auto sumcheck_wire_bytes = [&](const std::string &label, std::uint64_t domain,
                                         std::uint32_t degree, std::uint64_t &bytes) -> bool {
        if (domain < 2U || !native_power_of_two(domain) || degree == 0U || degree >= 4096U) {
            if (error) *error = "native Sumcheck wire profile received an unsupported domain/degree";
            return false;
        }
        std::uint32_t rounds = 0U;
        if (!log2_domain(domain, rounds) || rounds == 0U || rounds > 30U) {
            if (error) *error = "native Sumcheck wire profile received an unsupported round count";
            return false;
        }
        std::uint64_t vector_payload = 0U, first_round = 0U, later_round = 0U;
        if (!checked_mul_wire(128U, static_cast<std::uint64_t>(degree) + 1U,
                              vector_payload, "Sumcheck vector-linear proof") ||
            !checked_add_wire(1364U, vector_payload, first_round,
                              "Sumcheck initial round") ||
            !checked_add_wire(1412U, vector_payload, later_round,
                              "Sumcheck chained round")) return false;
        std::uint64_t body = 472U + label.size();
        if (!checked_add_wire(body, first_round, body, "Sumcheck first round") ||
            (rounds > 1U && !checked_mul_wire(rounds - 1U, later_round,
                                               vector_payload, "Sumcheck later rounds")) ||
            (rounds > 1U && !checked_add_wire(body, vector_payload, body,
                                               "Sumcheck later rounds"))) return false;
        bytes = body;
        if (bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes) {
            if (error) *error = "native typed Sumcheck body exceeds the strict 4 MiB nested parser limit: " + label;
            return false;
        }
        return true;
    };
    const auto terminal_wire_bytes_for_segment = [&](std::uint64_t domain,
                                         std::uint64_t segment_size,
                                         std::uint64_t references,
                                         std::uint64_t product_nodes,
                                         const std::string &label,
                                         std::uint64_t &bytes) -> bool {
        std::uint64_t opening = 0U, segments = 0U;
        if (references == 0U || references > zkdl::ZkTerminalRelationProof::kMaxTerminalCommitments ||
            product_nodes > zkdl::ZkTerminalRelationProof::kMaxProductNodes ||
            !hyrax_wire_bytes_for_segment(domain, segment_size, opening, segments) ||
            segments > (1U << 12U)) {
            if (error && error->empty()) *error =
                "native terminal relation typed shape exceeds parser limits: " + label +
                " (domain=" + std::to_string(domain) +
                ", segment_size=" + std::to_string(segment_size) +
                ", references=" + std::to_string(references) +
                ", segments=" + std::to_string(segments) +
                ", product_nodes=" + std::to_string(product_nodes) + ")";
            return false;
        }
        // ZkTerminalRelationProof serializes one mirrored first opening and
        // every segment opening per provider, plus the product DAG and link.
        std::uint64_t per_reference = 64U, segment_frame = 8U + opening;
        if (!checked_add_wire(per_reference, opening, per_reference,
                              "terminal mirrored opening") ||
            !checked_mul_wire(segments, segment_frame, segment_frame,
                              "terminal segment openings") ||
            !checked_add_wire(per_reference, segment_frame, per_reference,
                              "terminal segment openings")) return false;
        std::uint64_t references_bytes = 0U, products_bytes = 0U;
        if (!checked_mul_wire(references, per_reference, references_bytes,
                              "terminal provider records") ||
            !checked_mul_wire(product_nodes, 604U, products_bytes,
                              "terminal product nodes") ||
            !checked_add_wire(644U, references_bytes, bytes,
                              "terminal fixed/provider bytes") ||
            !checked_add_wire(bytes, products_bytes, bytes,
                              "terminal product bytes")) return false;
        return true;
    };
    const auto terminal_wire_bytes = [&](std::uint64_t domain,
                                         std::uint64_t references,
                                         std::uint64_t product_nodes,
                                         const std::string &label,
                                         std::uint64_t &bytes) -> bool {
        return terminal_wire_bytes_for_segment(
            domain, std::min(domain, kSegmentSize), references,
            product_nodes, label, bytes);
    };
    const auto product_node_count = [](const relation::ConstraintDescriptor &descriptor,
                                       std::uint64_t &count) {
        count = 0U;
        for (const auto &term : descriptor.terms) {
            if (term.refs.empty() || term.refs.size() - 1U >
                    std::numeric_limits<std::uint64_t>::max() - count) return false;
            count += term.refs.size() - 1U;
        }
        return true;
    };
    const auto constraint_wire_bytes = [&](const relation::ConstraintDescriptor &descriptor,
                                           std::uint64_t domain, std::uint64_t references,
                                           const std::string &operation_context,
                                           std::uint64_t &bytes) -> bool {
        std::vector<zkdl::byte> encoded_descriptor;
        relation::append_constraint(encoded_descriptor, descriptor);
        std::uint64_t products = 0U, sumcheck = 0U, terminal = 0U;
        const std::string typed_label = operation_context + "/" + descriptor.id;
        if (!product_node_count(descriptor, products) ||
            !sumcheck_wire_bytes(operation_context, domain, descriptor.degree + 2U, sumcheck) ||
            !terminal_wire_bytes(domain, references, products, typed_label, terminal)) return false;
        std::uint64_t indices = 0U, body = encoded_descriptor.size();
        if (!checked_mul_wire(references, 4U, indices, "constraint opening-index vectors") ||
            !checked_add_wire(body, 16U, body, "constraint opening-index counts") ||
            !checked_add_wire(body, indices, body, "constraint opening tensor indices") ||
            !checked_add_wire(body, indices, body, "constraint opening commitment indices") ||
            !checked_add_wire(body, 16U, body, "constraint nested-proof framing") ||
            !checked_add_wire(body, sumcheck, body, "constraint Sumcheck body") ||
            !checked_add_wire(body, terminal, body, "constraint terminal body")) return false;
        bytes = body;
        metrics->sumcheck_body_max_bytes = std::max(metrics->sumcheck_body_max_bytes, sumcheck);
        metrics->terminal_relation_body_max_bytes =
            std::max(metrics->terminal_relation_body_max_bytes, terminal);
        metrics->constraint_body_max_bytes =
            std::max(metrics->constraint_body_max_bytes, bytes);
        return true;
    };
    const auto route_wire_bytes = [&](const std::string &route_id,
                                      std::uint64_t edge_domain, std::uint64_t node_domain,
                                      std::uint64_t feature_domain, std::uint64_t source_feature_domain,
                                      std::uint64_t &bytes) -> bool {
        std::uint64_t source_domain = 0U, output_domain = 0U;
        if (!checked_mul_wire(node_domain, source_feature_domain, source_domain,
                              "sparse-route source domain") ||
            !checked_mul_wire(edge_domain, feature_domain, output_domain,
                              "sparse-route output domain")) return false;
        std::uint32_t edge_bits = 0U, node_bits = 0U, feature_bits = 0U, source_bits = 0U;
        std::uint64_t sumcheck = 0U, source_opening = 0U, output_opening = 0U;
        std::uint64_t source_segments = 0U, output_segments = 0U;
        if (!log2_domain(edge_domain, edge_bits) || !log2_domain(node_domain, node_bits) ||
            !log2_domain(feature_domain, feature_bits) ||
            !log2_domain(source_feature_domain, source_bits) ||
            !sumcheck_wire_bytes(route_id + "/hidden-route/" + std::string(64U, '0'),
                                 std::max<std::uint64_t>(2U, source_domain), 2U, sumcheck) ||
            !hyrax_wire_bytes(source_domain, source_opening, source_segments) ||
            !hyrax_wire_bytes(output_domain, output_opening, output_segments)) return false;
        std::uint64_t point_bits = 0U, point_bytes = 0U;
        if (!checked_add_wire(edge_bits, node_bits, point_bits, "route point lengths") ||
            !checked_add_wire(point_bits, feature_bits, point_bits, "route point lengths") ||
            !checked_add_wire(point_bits, source_bits, point_bits, "route point lengths") ||
            !checked_mul_wire(point_bits, 32U, point_bytes, "route point bytes") ||
            !checked_add_wire(424U, point_bytes, bytes, "route fixed point bytes") ||
            !checked_add_wire(bytes, 8U, bytes, "route Sumcheck framing") ||
            !checked_add_wire(bytes, sumcheck, bytes, "route Sumcheck bytes") ||
            !checked_add_wire(bytes, source_opening, bytes, "route source opening") ||
            !checked_add_wire(bytes, output_opening, bytes, "route output opening") ||
            !checked_add_wire(bytes, 352U, bytes, "route terminal equality")) return false;
        return true;
    };
    const auto hidden_matmul_wire_bytes = [&](const relation::OperationDescriptor &operation,
                                              const relation::PrimitiveDescriptor &primitive,
                                              std::size_t primitive_index,
                                              std::uint64_t &bytes) -> bool {
        std::array<std::uint64_t, 3U> sizes{};
        if (!primitive.matmul_shape.tensor_sizes(sizes)) {
            if (error) *error = "native hidden-matmul wire profile has invalid tensor shapes: " + operation.id;
            return false;
        }
        std::uint64_t opening_bytes = 0U;
        for (const auto domain : sizes) {
            std::uint64_t one_opening = 0U, segments = 0U, one_bytes = 0U;
            if (!hyrax_wire_bytes(domain, one_opening, segments) ||
                segments > (1U << 12U) ||
                !checked_mul_wire(one_opening, segments, one_bytes,
                                  "hidden-matmul segmented openings") ||
                !checked_add_wire(opening_bytes, one_bytes, opening_bytes,
                                  "hidden-matmul segmented openings")) {
                if (error && error->empty()) *error = "native hidden-matmul opening exceeds strict segment limits: " + operation.id;
                return false;
            }
        }
        const auto ceil_bits = [](std::uint32_t value) {
            std::uint64_t domain = 1U;
            std::uint32_t bits = 0U;
            while (domain < value) { domain <<= 1U; ++bits; }
            return bits;
        };
        const std::uint64_t point_bits = ceil_bits(primitive.matmul_shape.num) +
                                         ceil_bits(primitive.matmul_shape.m) +
                                         ceil_bits(primitive.matmul_shape.k);
        std::uint64_t point_bytes = 0U, sumcheck = 0U;
        const std::string label = "gat.relation." + operation.id + "/hidden-matmul/" +
                                  std::to_string(primitive_index);
        if (!checked_mul_wire(point_bits, 32U, point_bytes,
                              "hidden-matmul challenge points") ||
            !sumcheck_wire_bytes(label, primitive.matmul_shape.n, 2U, sumcheck) ||
            !checked_add_wire(1120U, point_bytes, bytes,
                              "hidden-matmul fixed/point bytes") ||
            !checked_add_wire(bytes, 8U, bytes, "hidden-matmul Sumcheck framing") ||
            !checked_add_wire(bytes, sumcheck, bytes, "hidden-matmul Sumcheck body") ||
            !checked_add_wire(bytes, 156U, bytes, "hidden-matmul opening headers") ||
            !checked_add_wire(bytes, opening_bytes, bytes,
                              "hidden-matmul opening bodies") ||
            !checked_add_wire(bytes, 8U + 504U + 8U + 344U, bytes,
                              "hidden-matmul terminal bindings")) return false;
        metrics->sumcheck_body_max_bytes =
            std::max(metrics->sumcheck_body_max_bytes, sumcheck);
        if (bytes > relation::HiddenMatMulProof::kMaxProofBytes) {
            if (error) *error = "native typed hidden-matmul proof exceeds the strict 256 MiB parser limit: " + operation.id;
            return false;
        }
        metrics->hidden_matmul_body_max_bytes =
            std::max(metrics->hidden_matmul_body_max_bytes, bytes);
        return true;
    };
    for (const auto &family : plan.tensor_families) {
        if (family.id.empty() || family.logical_shape.empty() || family.logical_shape.size() != family.padded_shape.size()) {
            if (error) *error = "native relation admission found an incomplete tensor-family shape";
            return false;
        }
        std::uint64_t domain = 1U;
        bool padded = false;
        for (std::size_t axis = 0U; axis < family.padded_shape.size(); ++axis) {
            if (family.logical_shape[axis] == 0U || family.padded_shape[axis] < family.logical_shape[axis] ||
                !checked_mul_u64(domain, family.padded_shape[axis], domain) || domain > kMaxNativeDomain) {
                if (error) *error = "native relation admission tensor-family domain is invalid: " + family.id;
                return false;
            }
            padded = padded || family.logical_shape[axis] != family.padded_shape[axis];
        }
        if (!domains.emplace(family.id, domain).second) {
            if (error) *error = "native relation admission found a duplicate tensor family: " + family.id;
            return false;
        }
        relation::TensorWitness tensor;
        tensor.id = family.id;
        tensor.visibility = 1U;
        tensor.domain_size = static_cast<std::uint32_t>(domain);
        shape_tensors.push_back(std::move(tensor));
        const std::uint64_t segments = (domain + kSegmentSize - 1U) / kSegmentSize;
        if (!add(&metrics->family_segment_count, segments, "family segment count") ||
            (relation::qpad_record_required(
                 family.logical_shape, family.padded_shape,
                 family.visibility == "public") &&
             !add(&metrics->padding_family_count, 1U, "padding family count"))) return false;
    }
    std::vector<relation::OperationDescriptor> operations;
    if (analysis_only_shape) {
        std::map<std::string, std::uint32_t> narrow_domains;
        std::map<std::string, const TensorFamily *> families;
        for (const auto &entry : domains) {
            if (entry.second > std::numeric_limits<std::uint32_t>::max()) {
                if (error) *error = "analysis-only native relation domain exceeds descriptor dimensions";
                return false;
            }
            narrow_domains.emplace(entry.first, static_cast<std::uint32_t>(entry.second));
        }
        for (const auto &family : plan.tensor_families)
            if (!families.emplace(family.id, &family).second) {
                if (error) *error = "analysis-only native relation plan has duplicate tensor families";
                return false;
            }
        operations.reserve(plan.operation_families.size());
        shape_operation_wire_deltas.reserve(plan.operation_families.size());
        for (const auto &plan_operation : plan.operation_families) {
            shape_interner.begin_shape_operation();
            std::string operation_error;
            auto operation = canonical_operation(config, tables,
                plan.integer_bound_ledger, plan_operation, narrow_domains, families,
                &graph, shape_interner, &operation_error);
            if (operation.id.empty()) {
                if (error) *error = "BLOCKED: analysis-only canonical shape dispatch failed for " +
                    plan_operation.id + (operation_error.empty() ? std::string() : ": " + operation_error);
                return false;
            }
            shape_operation_wire_deltas.push_back(
                shape_interner.shape_operation_wire_payload_delta());
            operations.push_back(std::move(operation));
        }
        if (!relation::formal_relation_components_supported(operations, error)) return false;
    } else if (!build_canonical_operations(config, tables, plan, shape_tensors, operations, &graph, error) ||
               !operation_set_matches_plan(plan, operations, error) ||
               !relation::descriptor_vector_valid(operations, error) ||
               !relation::formal_relation_components_supported(operations, error)) return false;
    metrics->operation_count = operations.size();
    std::uint64_t operation_section_wire_bytes = 8U;
    std::uint64_t padding_section_wire_bytes = 8U;
    std::uint64_t sorted_lookup_section_wire_bytes = 8U;
    std::uint64_t commitment_section_wire_bytes = 8U;
    for (std::size_t operation_index = 0U; operation_index < operations.size(); ++operation_index) {
        const auto &operation = operations[operation_index];
        std::uint64_t encoded_bytes = 0U;
        if (analysis_only_shape) {
            zkdl::WireSizeCounter counter;
            relation::append_operation(counter, operation);
            if (operation_index >= shape_operation_wire_deltas.size() ||
                !checked_add_u64(counter.size(), shape_operation_wire_deltas[operation_index],
                                 encoded_bytes)) {
                if (error) *error = "BLOCKED: analysis-only canonical descriptor wire size overflows";
                return false;
            }
        } else {
            std::vector<zkdl::byte> encoded;
            relation::append_operation(encoded, operation);
            encoded_bytes = encoded.size();
        }
        if (!add(&metrics->operation_descriptor_bytes, encoded_bytes, "operation descriptor bytes") ||
            !add(&metrics->primitive_count, operation.primitives.size(), "primitive count") ||
            !add(&metrics->constraint_count, operation.constraints.size(), "constraint count") ||
            !add(&metrics->fac_binding_count, operation.fac_bindings.size(), "FAC binding count")) return false;
        metrics->operation_descriptor_max_bytes =
            std::max<std::uint64_t>(metrics->operation_descriptor_max_bytes, encoded_bytes);
        if (operation.kind == relation::RelationKind::SparseRoute || !operation.selector_source_ref.empty())
            if (!add(&metrics->route_count, 1U, "route count")) return false;
        if (operation.kind == relation::RelationKind::SparseRoute || !operation.selector_source_ref.empty()) {
            std::uint64_t route_source_domain = 0U, route_output_domain = 0U;
            std::uint64_t route_factor_elements = 0U;
            if (!checked_mul_u64(operation.selector_node_domain,
                                 operation.selector_source_feature_domain,
                                 route_source_domain) ||
                !checked_mul_u64(operation.selector_edge_domain,
                                 operation.selector_feature_domain,
                                 route_output_domain) ||
                !checked_add_u64(operation.selector_node_domain,
                                 operation.selector_source_feature_domain,
                                 route_factor_elements) ||
                route_source_domain == 0U || route_output_domain == 0U ||
                domains.find(operation.selector_source_ref) == domains.end() ||
                domains.find(operation.selector_output_ref) == domains.end() ||
                domains.at(operation.selector_source_ref) != route_source_domain ||
                domains.at(operation.selector_output_ref) != route_output_domain) {
                if (error) *error = "native route source/output dimensions do not match canonical families: " + operation.id;
                return false;
            }
            metrics->maximum_route_source_domain =
                std::max(metrics->maximum_route_source_domain, route_source_domain);
            metrics->maximum_route_edge_domain =
                std::max(metrics->maximum_route_edge_domain, route_output_domain);
            metrics->maximum_route_selector_factor_elements =
                std::max(metrics->maximum_route_selector_factor_elements, route_factor_elements);
            const std::uint64_t source_segment =
                std::min<std::uint64_t>(route_source_domain, kSegmentSize);
            const std::uint64_t output_segment =
                std::min<std::uint64_t>(route_output_domain, kSegmentSize);
            std::uint64_t route_input_segments = 0U;
            if (!record_hyrax_opening_shape(route_source_domain, source_segment,
                                             1U, 1U, 0U) ||
                !record_hyrax_opening_shape(route_output_domain, output_segment,
                                             1U, 1U, 0U) ||
                !add(&metrics->sparse_route_opening_count, 2U,
                     "sparse route Hyrax opening count") ||
                !checked_add_u64(route_source_domain / source_segment,
                                 route_output_domain / output_segment,
                                 route_input_segments) ||
                !add(&metrics->route_aggregate_input_segment_count,
                     route_input_segments, "sparse route aggregate input segments") ||
                !record_sumcheck_shape(4U, std::max<std::uint64_t>(2U, route_source_domain),
                    std::min<std::uint64_t>(std::max<std::uint64_t>(2U, route_source_domain),
                                            kSegmentSize), true)) return false;
        }
        for (const auto &binding : operation.fac_bindings) {
            std::uint64_t source_domain = 0U, output_domain = 0U;
            std::uint64_t factor_elements = 0U;
            if (!checked_mul_u64(binding.canonical_outer_domain,
                                 binding.canonical_feature_domain, source_domain) ||
                !checked_mul_u64(binding.view_outer_domain,
                                 binding.view_feature_domain, output_domain) ||
                !checked_add_u64(binding.canonical_outer_domain,
                                 binding.canonical_feature_domain, factor_elements) ||
                source_domain == 0U || output_domain == 0U ||
                domains.find(binding.canonical_ref) == domains.end() ||
                domains.find(binding.view_ref) == domains.end() ||
                domains.at(binding.canonical_ref) != source_domain ||
                domains.at(binding.view_ref) != output_domain) {
                if (error) *error = "native FAC route dimensions do not match canonical families: " + operation.id;
                return false;
            }
            metrics->maximum_route_source_domain =
                std::max(metrics->maximum_route_source_domain, source_domain);
            metrics->maximum_route_edge_domain =
                std::max(metrics->maximum_route_edge_domain, output_domain);
            metrics->maximum_route_selector_factor_elements =
                std::max(metrics->maximum_route_selector_factor_elements, factor_elements);
            const std::uint64_t source_segment =
                std::min<std::uint64_t>(source_domain, kSegmentSize);
            const std::uint64_t output_segment =
                std::min<std::uint64_t>(output_domain, kSegmentSize);
            std::uint64_t route_input_segments = 0U;
            if (!record_hyrax_opening_shape(source_domain, source_segment,
                                             1U, 1U, 0U) ||
                !record_hyrax_opening_shape(output_domain, output_segment,
                                             1U, 1U, 0U) ||
                !add(&metrics->fac_selector_opening_count, 2U,
                     "FAC selector Hyrax opening count") ||
                !checked_add_u64(source_domain / source_segment,
                                 output_domain / output_segment,
                                 route_input_segments) ||
                !add(&metrics->route_aggregate_input_segment_count,
                     route_input_segments, "FAC aggregate input segments") ||
                !record_sumcheck_shape(4U, std::max<std::uint64_t>(2U, source_domain),
                    std::min<std::uint64_t>(std::max<std::uint64_t>(2U, source_domain),
                                            kSegmentSize), true)) return false;
        }
        for (const auto &primitive : operation.primitives) {
            for (const auto &tensor_id : primitive.tensor_refs) {
                const auto found = domains.find(tensor_id);
                if (found == domains.end()) {
                    if (error) *error = "native relation admission primitive refers to an unplanned tensor: " + tensor_id;
                    return false;
                }
                const std::uint64_t segments = (found->second + kSegmentSize - 1U) / kSegmentSize;
                const std::uint64_t segment_size =
                    std::min<std::uint64_t>(found->second, kSegmentSize);
                if (!add(&metrics->primitive_opening_segment_count, segments, "primitive opening segment count") ||
                    !record_hyrax_opening_shape(found->second, segment_size,
                                                segments, segments, 0U)) return false;
            }
        }

        // Calculate the same typed body lengths used by the production
        // serializers.  These are source-domain/segment-derived bounds, not
        // repeated parser-cap allowances.
        std::uint64_t operation_wire_bytes = encoded_bytes + 8U; // primitive count
        for (std::size_t primitive_index = 0U;
             primitive_index < operation.primitives.size(); ++primitive_index) {
            std::uint64_t primitive_bytes = 0U, framed = 0U;
            if (!hidden_matmul_wire_bytes(operation, operation.primitives[primitive_index],
                                          primitive_index, primitive_bytes) ||
                !checked_add_wire(8U, primitive_bytes, framed,
                                  "operation primitive frame") ||
                !checked_add_wire(operation_wire_bytes, framed, operation_wire_bytes,
                                  "operation primitive section")) return false;
        }
        if (!checked_add_wire(operation_wire_bytes, 8U, operation_wire_bytes,
                              "operation primitive-binding count")) return false;

        const bool has_route = operation.kind == relation::RelationKind::SparseRoute ||
                               !operation.selector_source_ref.empty();
        std::uint64_t route_bytes = 0U;
        if (has_route) {
            if (!route_wire_bytes(operation.id, operation.selector_edge_domain,
                                  operation.selector_node_domain,
                                  operation.selector_feature_domain,
                                  operation.selector_source_feature_domain, route_bytes)) return false;
            metrics->sparse_route_body_max_bytes =
                std::max(metrics->sparse_route_body_max_bytes, route_bytes);
        }
        if (!checked_add_wire(operation_wire_bytes, 8U, operation_wire_bytes,
                              "operation route frame") ||
            !checked_add_wire(operation_wire_bytes, route_bytes, operation_wire_bytes,
                              "operation route body")) return false;

        std::uint64_t fac_bytes = 0U;
        if (!operation.fac_bindings.empty()) {
            fac_bytes = 368U;
            for (std::size_t binding_index = 0U;
                 binding_index < operation.fac_bindings.size(); ++binding_index) {
                const auto &binding = operation.fac_bindings[binding_index];
                const std::string selector_id = operation.id +
                    "/fac4dnn-line8/selector/" + std::to_string(binding_index);
                std::uint64_t binding_route = 0U, framed = 0U;
                if (!route_wire_bytes(selector_id, binding.view_outer_domain,
                                      binding.canonical_outer_domain,
                                      binding.view_feature_domain,
                                      binding.canonical_feature_domain, binding_route) ||
                    binding_route > zkdl::ZkSumcheckProof::kMaxNestedProofBytes ||
                    !checked_add_wire(8U, binding_route, framed,
                                      "FAC selector-route frame") ||
                    !checked_add_wire(fac_bytes, framed, fac_bytes,
                                      "FAC selector-route records")) {
                    if (error && error->empty()) *error = "native FAC selector route exceeds the strict 4 MiB nested parser limit";
                    return false;
                }
                metrics->sparse_route_body_max_bytes =
                    std::max(metrics->sparse_route_body_max_bytes, binding_route);
            }
            metrics->fac_line8_body_max_bytes =
                std::max(metrics->fac_line8_body_max_bytes, fac_bytes);
        }
        if (!checked_add_wire(operation_wire_bytes, 8U, operation_wire_bytes,
                              "operation FAC frame") ||
            !checked_add_wire(operation_wire_bytes, fac_bytes, operation_wire_bytes,
                              "operation FAC body")) return false;

        std::uint64_t relation_domain = 0U;
        const auto operation_domain = [&]() -> bool {
            for (const auto &ref : operation.tensor_refs) {
                const auto found = domains.find(ref);
                if (found == domains.end()) return false;
                if (relation_domain == 0U) relation_domain = found->second;
                else if (relation_domain != found->second) return false;
            }
            return relation_domain >= 2U;
        };
        const bool batched_relation = !operation.batch_family.empty() &&
                                      operation.batch_family != "authenticated_sorted_lookup";
        std::uint64_t batch_bytes = 0U, constraint_section_bytes = 8U;
        if (batched_relation) {
            if (!operation_domain()) {
                if (error) *error = "native batch relation tensors do not share a supported domain: " + operation.id;
                return false;
            }
            std::uint64_t batched_tensor_count = 0U;
            if (!checked_add_u64(operation.tensor_refs.size(), 2U,
                                 batched_tensor_count) ||
                !record_segmented_sumcheck_peak(
                    batched_tensor_count, relation_domain,
                    std::min<std::uint64_t>(relation_domain, kSegmentSize)) ||
                !record_sumcheck_shape(
                    batched_tensor_count, relation_domain,
                    std::min<std::uint64_t>(relation_domain, kSegmentSize), false)) {
                if (error && error->empty())
                    *error = "native batched Sumcheck tensor count overflows: " + operation.id;
                return false;
            }
            relation::ConstraintDescriptor batched;
            batched.id = operation.id + "/batch/" + operation.batch_family;
            batched.degree = 1U;
            if (!operation.constraints.empty()) {
                batched.logical_shape = operation.constraints.front().logical_shape;
                batched.padded_shape = operation.constraints.front().padded_shape;
            }
            for (const auto &constraint : operation.constraints) {
                batched.degree = std::max(batched.degree, constraint.degree);
                batched.terms.insert(batched.terms.end(), constraint.terms.begin(),
                                     constraint.terms.end());
            }
            std::uint64_t constraint_bytes = 0U;
            if (!constraint_wire_bytes(batched, relation_domain,
                                       operation.tensor_refs.size(),
                                       "gat.relation." + batched.id,
                                       constraint_bytes) ||
                !checked_add_wire(4U, constraint_bytes, batch_bytes,
                                  "batched constraint payload")) return false;
            if (!record_terminal_hyrax_openings(
                    relation_domain,
                    std::min<std::uint64_t>(relation_domain, kSegmentSize),
                    operation.tensor_refs.size(),
                    &metrics->constraint_opening_segment_count,
                    "batched constraint Hyrax opening segments")) return false;
        }
        if (!checked_add_wire(operation_wire_bytes, 8U, operation_wire_bytes,
                              "operation batch frame") ||
            !checked_add_wire(operation_wire_bytes, batch_bytes, operation_wire_bytes,
                              "operation batch body")) return false;
        if (!batched_relation) {
            if (!operation.constraints.empty() && !operation_domain()) {
                if (error) *error = "native relation tensors do not share a supported proof domain: " + operation.id;
                return false;
            }
            if (!operation.constraints.empty()) {
                std::uint64_t tensor_count = 0U;
                if (!checked_add_u64(operation.tensor_refs.size(), 2U, tensor_count) ||
                    !record_segmented_sumcheck_peak(
                        tensor_count, relation_domain,
                        std::min<std::uint64_t>(relation_domain, kSegmentSize))) {
                    if (error && error->empty())
                        *error = "native constraint Sumcheck tensor count overflows: " + operation.id;
                    return false;
                }
                for (std::size_t index = 0U; index < operation.constraints.size(); ++index)
                    if (!record_sumcheck_shape(
                            tensor_count, relation_domain,
                            std::min<std::uint64_t>(relation_domain, kSegmentSize), false))
                        return false;
                const std::uint64_t segment_size =
                    std::min<std::uint64_t>(relation_domain, kSegmentSize);
                std::uint64_t terminal_provider_count = 0U;
                if (!checked_mul_u64(operation.constraints.size(),
                                     operation.tensor_refs.size(),
                                     terminal_provider_count)) {
                    if (error) *error = "native constraint terminal provider count overflows: " + operation.id;
                    return false;
                }
                if (!record_terminal_hyrax_openings(
                        relation_domain, segment_size,
                        terminal_provider_count,
                        &metrics->constraint_opening_segment_count,
                        "constraint Hyrax opening segments")) return false;
            }
            for (const auto &constraint : operation.constraints) {
                std::uint64_t constraint_bytes = 0U;
                if (!constraint_wire_bytes(constraint, relation_domain,
                                           operation.tensor_refs.size(),
                                           "gat.relation." + constraint.id,
                                           constraint_bytes) ||
                    !checked_add_wire(constraint_section_bytes, constraint_bytes,
                                      constraint_section_bytes,
                                      "operation constraint bodies")) return false;
            }
        }
        if (!checked_add_wire(operation_wire_bytes, constraint_section_bytes,
                              operation_wire_bytes, "operation constraint section") ||
            !checked_add_wire(operation_section_wire_bytes, operation_wire_bytes,
                              operation_section_wire_bytes,
                              "total operation proof section")) return false;
    }

    // Commitment records carry the authenticated segment vector itself.  Its
    // size is determined by each canonical family and the fixed protocol
    // segment size; no SRS point is counted as a native sizeof(G1) estimate.
    for (const auto &family : plan.tensor_families) {
        const auto found = domains.find(family.id);
        if (found == domains.end()) {
            if (error) *error = "native commitment wire profile lost a canonical family: " + family.id;
            return false;
        }
        const std::uint64_t segments = (found->second + kSegmentSize - 1U) / kSegmentSize;
        std::uint64_t segment_bytes = 0U, commitment_bytes = 180U + family.id.size();
        if (!checked_mul_wire(segments, 104U, segment_bytes,
                              "family segment commitment records") ||
            !checked_add_wire(commitment_bytes, segment_bytes, commitment_bytes,
                              "family commitment wire") ||
            !checked_add_wire(commitment_section_wire_bytes, commitment_bytes,
                              commitment_section_wire_bytes,
                              "family commitment section")) return false;
    }

    const auto sorted_relation_wire_bytes = [&](std::uint32_t kind,
                                                const std::string &operation_id,
                                                const std::vector<std::string> &tags,
                                                std::uint64_t domain,
                                                std::uint64_t segment_size,
                                                std::uint64_t product_nodes,
                                                std::uint64_t &bytes) -> bool {
        if (tags.empty() || tags.size() > 128U) {
            if (error) *error = "native sorted-lookup relation exceeds the strict 128-reference parser limit";
            return false;
        }
        const std::string context = "gat.sorted_lookup.relation." + std::to_string(kind);
        const std::string typed_label = operation_id + "/relation." + std::to_string(kind);
        std::uint64_t tag_bytes = 0U, sumcheck = 0U, terminal = 0U;
        for (const auto &tag : tags)
            if (!checked_add_wire(tag_bytes, 8U + tag.size(), tag_bytes,
                                  "sorted-lookup relation tags")) return false;
        if (!sumcheck_wire_bytes(context, domain, 4U, sumcheck) ||
            !terminal_wire_bytes_for_segment(domain, segment_size,
                                              tags.size(), product_nodes, typed_label,
                                              terminal)) return false;
        std::uint64_t ref_bytes = 0U;
        if (!checked_mul_wire(tags.size(), 8U, ref_bytes,
                              "sorted-lookup relation index arrays") ||
            !checked_add_wire(92U, ref_bytes, bytes,
                              "sorted-lookup relation fixed/index bytes") ||
            !checked_add_wire(bytes, tag_bytes, bytes,
                              "sorted-lookup relation provider tags") ||
            !checked_add_wire(bytes, sumcheck, bytes,
                              "sorted-lookup relation Sumcheck") ||
            !checked_add_wire(bytes, terminal, bytes,
                              "sorted-lookup relation terminal")) return false;
        metrics->sumcheck_body_max_bytes =
            std::max(metrics->sumcheck_body_max_bytes, sumcheck);
        metrics->terminal_relation_body_max_bytes =
            std::max(metrics->terminal_relation_body_max_bytes, terminal);
        metrics->sorted_lookup_relation_body_max_bytes =
            std::max(metrics->sorted_lookup_relation_body_max_bytes, bytes);
        return true;
    };

    for (const auto &query : plan.sorted_lookup.queries) {
        const std::uint64_t bit_count = static_cast<std::uint64_t>(query.x_bits) + query.y_bits;
        if (query.table_domain == 0U || query.query_domain == 0U || query.merge_domain == 0U ||
            query.segment_size > kSegmentSize || !native_power_of_two(query.segment_size) ||
            query.table_domain % query.segment_size != 0U ||
            query.query_domain % query.segment_size != 0U ||
            query.merge_domain % query.segment_size != 0U ||
            8U + 2U * bit_count > 128U || 9U + bit_count > 256U) {
            if (error) *error = "native sorted-lookup typed profile disagrees with its authenticated segmented descriptor: " + query.operation_id;
            return false;
        }
        const auto tags_for_kind = [&](std::uint32_t kind,
                                       std::vector<std::string> &tags,
                                       std::uint64_t &products,
                                       std::uint64_t &domain) -> bool {
            products = 0U;
            if (kind == 1U) {
                tags = {"query_x", "query_y", "query_key", "query_active", "query_prefix",
                        "query_prefix_next", "query_first", "query_last", "query_endpoint"};
                products = 7U; domain = query.query_domain;
            } else if (kind == 2U) {
                tags = {"merge_prefix", "merge_prefix_next", "merge_key", "merge_active", "merge_first"};
                products = 4U; domain = query.merge_domain;
            } else if (kind == 3U) {
                tags = {"merge_key", "merge_previous_key", "merge_delta", "merge_active"};
                for (std::uint64_t bit = 0U; bit < bit_count; ++bit) {
                    tags.push_back("merge_bit_" + std::to_string(bit));
                    tags.push_back("merge_delta_bit_" + std::to_string(bit));
                }
                tags.insert(tags.end(), {"merge_inverse", "merge_nonzero", "merge_marker", "merge_first"});
                products = 2U * bit_count + 9U; domain = query.merge_domain;
            } else if (kind == 4U) {
                tags = {"merge_marker", "merge_active"};
                products = 1U; domain = query.merge_domain;
            } else if (kind == 5U) {
                tags = {"merge_prefix_next", "merge_last", "merge_endpoint"};
                products = 2U; domain = query.merge_domain;
            } else return false;
            return true;
        };
        std::uint64_t relation_section_bytes = 8U;
        for (std::uint32_t kind = 1U; kind <= 5U; ++kind) {
            std::vector<std::string> tags;
            std::uint64_t products = 0U, domain = 0U, body = 0U, framed = 0U;
            std::uint64_t weighted_tensor_count = 0U;
            if (!tags_for_kind(kind, tags, products, domain) ||
                !checked_add_u64(tags.size(), 2U, weighted_tensor_count) ||
                !record_segmented_sumcheck_peak(
                    weighted_tensor_count, domain, query.segment_size) ||
                !record_sumcheck_shape(
                    weighted_tensor_count, domain, query.segment_size, false) ||
                !record_terminal_hyrax_openings(
                    domain, query.segment_size, tags.size(),
                    &metrics->sorted_lookup_opening_segment_count,
                    "sorted-lookup Hyrax opening segments") ||
                !sorted_relation_wire_bytes(kind, query.operation_id, tags, domain, query.segment_size,
                                            products, body) ||
                !checked_add_wire(8U, body, framed, "sorted-lookup relation proof frame") ||
                !checked_add_wire(relation_section_bytes, framed,
                                  relation_section_bytes,
                                  "sorted-lookup relation section")) return false;
        }
        const std::uint64_t commitment_refs = 9U + bit_count;
        const std::uint64_t descriptor_bytes = 220U +
            std::string("authenticated_sorted_lookup_v1").size() +
            query.operation_id.size() + query.table_ref.size() +
            query.query_ref.size() + query.merge_ref.size();
        std::uint64_t ref_bytes = 0U, lookup_bytes = descriptor_bytes;
        if (!checked_mul_wire(commitment_refs, 24U, ref_bytes,
                              "sorted-lookup outer commitment references") ||
            !checked_add_wire(8U, ref_bytes, ref_bytes,
                              "sorted-lookup commitment vector") ||
            !checked_add_wire(lookup_bytes, ref_bytes, lookup_bytes,
                              "sorted-lookup descriptor/reference section") ||
            !checked_add_wire(lookup_bytes, relation_section_bytes, lookup_bytes,
                              "sorted-lookup relation section") ||
            !checked_add_wire(lookup_bytes, 1056U, lookup_bytes,
                              "sorted-lookup endpoint binding tail")) return false;
        metrics->sorted_lookup_proof_body_max_bytes =
            std::max(metrics->sorted_lookup_proof_body_max_bytes, lookup_bytes);
        if (lookup_bytes > zkdl::AuthenticatedSortedLookupProof::kMaxProofBytes) {
            if (error) *error = "native typed authenticated sorted-lookup proof exceeds the strict 256 MiB parser limit: " + query.operation_id;
            return false;
        }
        if (!checked_add_wire(8U, lookup_bytes, lookup_bytes,
                              "sorted-lookup proof vector frame") ||
            !checked_add_wire(sorted_lookup_section_wire_bytes, lookup_bytes,
                              sorted_lookup_section_wire_bytes,
                              "sorted-lookup proof section")) return false;

        const auto add_sorted_outer = [&](const std::string &role,
                                          std::uint64_t domain) -> bool {
            const std::uint64_t segments = domain / query.segment_size;
            std::uint64_t record_bytes = 180U +
                std::string("sorted_lookup/").size() + query.operation_id.size() +
                1U + role.size(), segment_bytes = 0U;
            if (!checked_mul_wire(segments, 104U, segment_bytes,
                                  "sorted-lookup outer segment commitments") ||
                !checked_add_wire(record_bytes, segment_bytes, record_bytes,
                                  "sorted-lookup outer commitment") ||
                !checked_add_wire(commitment_section_wire_bytes, record_bytes,
                                  commitment_section_wire_bytes,
                                  "sorted-lookup commitment section") ||
                metrics->sorted_lookup_outer_commitment_count ==
                    std::numeric_limits<std::uint64_t>::max()) return false;
            ++metrics->sorted_lookup_outer_commitment_count;
            return true;
        };
        for (const auto *role : {"query_x", "query_y", "query_key", "query_prefix"})
            if (!add_sorted_outer(role, query.query_domain)) return false;
        for (const auto *role : {"merge_key", "merge_delta", "merge_marker",
                                 "merge_inverse", "merge_prefix"})
            if (!add_sorted_outer(role, query.merge_domain)) return false;
        for (std::uint64_t bit = 0U; bit < bit_count; ++bit)
            if (!add_sorted_outer("merge_bit_" + std::to_string(bit),
                                  query.merge_domain)) return false;
    }

    // Qpad records exist only for non-public padded families.  The two
    // providers are the actual family and its public selector, both on this
    // family's authenticated padded domain.
    for (const auto &family : plan.tensor_families) {
        if (!relation::qpad_record_required(
                family.logical_shape, family.padded_shape,
                family.visibility == "public")) continue;
        const std::uint64_t domain = domains.at(family.id);
        const std::uint64_t segment_size =
            std::min<std::uint64_t>(domain, kSegmentSize);
        if (!record_segmented_sumcheck_peak(
                4U, domain, segment_size) ||
            !record_sumcheck_shape(4U, domain, segment_size, false) ||
            !record_terminal_hyrax_openings(
                domain, segment_size, 2U,
                &metrics->padding_opening_segment_count,
                "Qpad Hyrax opening segments"))
            return false;
        const auto descriptor = relation::qpad_constraint_descriptor(family.id);
        std::uint64_t relation_bytes = 0U, record_bytes = 64U + family.id.size();
        if (!constraint_wire_bytes(descriptor, domain, 2U,
                                   "gat.relation.padding." + descriptor.id,
                                   relation_bytes) ||
            !checked_add_wire(record_bytes, relation_bytes, record_bytes,
                              "Qpad record body") ||
            !checked_add_wire(8U, record_bytes, record_bytes,
                              "Qpad record frame") ||
            !checked_add_wire(padding_section_wire_bytes, record_bytes,
                              padding_section_wire_bytes,
                              "Qpad proof section")) return false;
        metrics->padding_record_body_max_bytes =
            std::max(metrics->padding_record_body_max_bytes, record_bytes - 8U);
        if (record_bytes - 8U > zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes) {
            if (error) *error = "native typed Qpad record exceeds the strict 4 MiB parser limit: " + family.id;
            return false;
        }
    }

    if (metrics->padding_family_count > (1U << 20U) ||
        shape_tensors.size() + metrics->sorted_lookup_outer_commitment_count > (1U << 20U) ||
        plan.sorted_lookup.queries.size() > (1U << 16U)) {
        if (error) *error = "native typed proof sections exceed a strict top-level parser item-count limit "
            "(qpad=" + std::to_string(metrics->padding_family_count) + "/" +
            std::to_string(1U << 20U) + ", commitments=" +
            std::to_string(shape_tensors.size() + metrics->sorted_lookup_outer_commitment_count) +
            "/1048576, sorted_lookup=" + std::to_string(plan.sorted_lookup.queries.size()) +
            "/65536)";
        return false;
    }

    metrics->commitment_section_wire_bytes_upper_bound = commitment_section_wire_bytes;
    if (!checked_add_wire(padding_section_wire_bytes, operation_section_wire_bytes,
                          metrics->relation_proof_sections_wire_bytes_upper_bound,
                          "relation proof sections") ||
        !checked_add_wire(metrics->relation_proof_sections_wire_bytes_upper_bound,
                          sorted_lookup_section_wire_bytes,
                          metrics->relation_proof_sections_wire_bytes_upper_bound,
                          "sorted-lookup proof section")) return false;
    std::uint64_t public_y_domain = 0U, public_y_opening = 0U, public_y_segments = 0U;
    const auto public_y = domains.find("public_Y");
    if (public_y == domains.end() ||
        !hyrax_wire_bytes(public_y->second, public_y_opening, public_y_segments)) {
        if (error) *error = "native proof-wire profile lacks the canonical public_Y opening domain";
        return false;
    }
    public_y_domain = public_y->second;
    std::uint64_t public_y_openings = 0U, public_output_values = 0U;
    if (!checked_mul_wire(public_y_opening, public_y_segments, public_y_openings,
                          "public_Y segmented opening") ||
        !record_hyrax_opening_shape(
            public_y_domain, std::min<std::uint64_t>(public_y_domain, kSegmentSize),
            public_y_segments, public_y_segments, 0U) ||
        !add(&metrics->public_output_opening_segment_count, public_y_segments,
             "public output Hyrax opening segments") ||
        !checked_mul_wire(public_y_domain, 32U, public_output_values,
                          "public_Y padded canonical field bytes") ||
        !checked_add_wire(public_output_values, 8U, public_output_values,
                          "public_Y field vector framing")) return false;
    std::uint64_t proof_bytes = 8U + 268U; // GATR header/version + RelationStatement fixed fields.
    if (!checked_add_wire(proof_bytes, public_output_values, proof_bytes,
                          "proof statement public output") ||
        !checked_add_wire(proof_bytes, commitment_section_wire_bytes, proof_bytes,
                          "proof commitment section") ||
        !checked_add_wire(proof_bytes, metrics->relation_proof_sections_wire_bytes_upper_bound,
                          proof_bytes, "proof relation sections") ||
        !checked_add_wire(proof_bytes, 8U, proof_bytes,
                          "proof hiding commitment section") ||
        !checked_add_wire(proof_bytes, 392U, proof_bytes,
                          "proof public output opening/binding framing") ||
        !checked_add_wire(proof_bytes, public_y_openings, proof_bytes,
                          "proof public output opening") ||
        !checked_add_wire(proof_bytes, 32U, proof_bytes,
                          "proof transcript digest")) return false;
    metrics->proof_wire_bytes_upper_bound = proof_bytes;

    // The canonical context owns interned selector backing while deserialization
    // builds a separate, unshared received descriptor tree.  Count ordinary
    // nested vectors/strings in both trees, shared selector arrays once for the
    // expected context, and every selector occurrence for the received proof.
    // Verification compares canonical and received descriptor bytes at the
    // same time, so also retain two maximum descriptor-wire buffers.
    std::uint64_t descriptor_expected_backing_bytes = 0U;
    std::uint64_t descriptor_received_backing_bytes = 0U;
    std::set<const void *> unique_index_map_backings;
    std::set<const void *> unique_active_mask_backings;
    const auto add_expected_capacity = [&](std::uint64_t count,
                                           std::uint64_t element_size,
                                           const char *label) -> bool {
        std::uint64_t bytes = 0U;
        return checked_mul_wire(count, element_size, bytes, label) &&
               checked_add_wire(descriptor_expected_backing_bytes, bytes,
                                descriptor_expected_backing_bytes, label);
    };
    const auto add_received_capacity = [&](std::uint64_t count,
                                           std::uint64_t element_size,
                                           const char *label) -> bool {
        std::uint64_t bytes = 0U;
        return checked_mul_wire(count, element_size, bytes, label) &&
               checked_add_wire(descriptor_received_backing_bytes, bytes,
                                descriptor_received_backing_bytes, label);
    };
    const auto add_descriptor_capacity = [&](std::uint64_t count,
                                              std::uint64_t element_size,
                                              const char *label) -> bool {
        return add_expected_capacity(count, element_size, label) &&
               add_received_capacity(count, element_size, label);
    };
    const auto add_index_map = [&](const relation::IndexMapU32 &values,
                                   const char *label) -> bool {
        if (!add_received_capacity(values.capacity(), sizeof(std::uint32_t), label)) return false;
        const void *const backing = values.backing_identity();
        return backing == nullptr || !unique_index_map_backings.insert(backing).second ||
               add_expected_capacity(values.capacity(), sizeof(std::uint32_t), label);
    };
    const auto add_active_mask = [&](const relation::CowByteVector &values,
                                     const char *label) -> bool {
        if (!add_received_capacity(values.capacity(), sizeof(std::uint8_t), label)) return false;
        const void *const backing = values.backing_identity();
        return backing == nullptr || !unique_active_mask_backings.insert(backing).second ||
               add_expected_capacity(values.capacity(), sizeof(std::uint8_t), label);
    };
    const auto add_descriptor_string = [&](const std::string &value) -> bool {
        std::uint64_t allocated = 0U;
        return checked_add_wire(value.capacity(), 1U, allocated,
                                "descriptor string capacity") &&
               add_descriptor_capacity(allocated, sizeof(char),
                                       "descriptor string backing capacity");
    };
    const auto add_descriptor_string_vector = [&](const std::vector<std::string> &values,
                                                   const char *label) -> bool {
        if (!add_descriptor_capacity(values.capacity(), sizeof(std::string), label))
            return false;
        for (const auto &value : values)
            if (!add_descriptor_string(value)) return false;
        return true;
    };
    for (const auto &operation : operations) {
        const std::string *const operation_strings[] = {
            &operation.id, &operation.mapping, &operation.padding,
            &operation.terminal_claim, &operation.terminal_point,
            &operation.selector_source_ref, &operation.selector_output_ref,
            &operation.batch_family, &operation.public_table_ref};
        for (const auto *value : operation_strings)
            if (!add_descriptor_string(*value)) return false;
        if (!add_descriptor_string_vector(operation.source_refs,
                                          "operation source-reference capacity") ||
            !add_descriptor_string_vector(operation.tensor_refs,
                                          "operation tensor-reference capacity") ||
            !add_descriptor_string_vector(operation.selector_refs,
                                          "operation selector-reference capacity") ||
            !add_descriptor_capacity(operation.constants.capacity(), sizeof(FrWire),
                                     "operation constant capacity") ||
            !add_descriptor_capacity(operation.constraints.capacity(),
                                     sizeof(relation::ConstraintDescriptor),
                                     "operation constraint capacity") ||
            !add_descriptor_capacity(operation.primitives.capacity(),
                                     sizeof(relation::PrimitiveDescriptor),
                                     "operation primitive capacity") ||
            !add_index_map(operation.selector_src, "operation selector source capacity") ||
            !add_index_map(operation.selector_dst, "operation selector destination capacity") ||
            !add_active_mask(operation.selector_active, "operation selector active capacity") ||
            !add_index_map(operation.selector_feature_map,
                           "operation selector feature-map capacity") ||
            !add_descriptor_capacity(operation.fac_bindings.capacity(),
                                     sizeof(relation::FacBindingDescriptor),
                                     "operation FAC binding capacity") ||
            !add_descriptor_capacity(operation.public_table_selector.capacity(), sizeof(FrWire),
                                     "operation public-table selector capacity")) return false;
        for (const auto &constraint : operation.constraints) {
            if (!add_descriptor_string(constraint.id) ||
                !add_descriptor_capacity(constraint.terms.capacity(),
                                         sizeof(relation::Monomial),
                                         "constraint monomial capacity") ||
                !add_descriptor_capacity(constraint.logical_shape.capacity(),
                                         sizeof(std::uint64_t),
                                         "constraint logical-shape capacity") ||
                !add_descriptor_capacity(constraint.padded_shape.capacity(),
                                         sizeof(std::uint64_t),
                                         "constraint padded-shape capacity")) return false;
            for (const auto &term : constraint.terms)
                if (!add_descriptor_capacity(term.refs.capacity(), sizeof(std::uint32_t),
                                             "constraint reference capacity")) return false;
        }
        for (const auto &primitive : operation.primitives)
            if (!add_descriptor_string_vector(primitive.tensor_refs,
                                              "primitive tensor-reference capacity"))
                return false;
        for (const auto &binding : operation.fac_bindings) {
            if (!add_descriptor_string(binding.view_ref) ||
                !add_descriptor_string(binding.canonical_ref) ||
                !add_index_map(binding.source_outer_index, "FAC source-outer capacity") ||
                !add_index_map(binding.source_feature_index, "FAC source-feature capacity") ||
                !add_active_mask(binding.selector_active, "FAC selector-active capacity")) return false;
        }
    }
    std::uint64_t context_header_bytes = 0U, proof_header_bytes = 0U;
    std::uint64_t descriptor_backing_reserve = 0U;
    std::uint64_t descriptor_compare_wire_bytes = 0U;
    if (!checked_mul_wire(operations.capacity(), sizeof(relation::OperationDescriptor),
                          context_header_bytes, "context operation header capacity") ||
        !checked_mul_wire(operations.size(), sizeof(relation::OperationProof),
                          proof_header_bytes, "proof operation header capacity") ||
        !checked_add_wire(proof_header_bytes, sizeof(relation::OperationProof),
                          proof_header_bytes, "active proof operation header") ||
        !checked_add_wire(descriptor_expected_backing_bytes,
                          descriptor_received_backing_bytes,
                          descriptor_backing_reserve,
                          "expected and received descriptor backing") ||
        !checked_mul_wire(metrics->operation_descriptor_max_bytes, 2U,
                          descriptor_compare_wire_bytes,
                          "expected/actual descriptor comparison buffers") ||
        !checked_add_wire(context_header_bytes, proof_header_bytes,
                          metrics->operation_descriptor_resident_bytes_upper_bound,
                          "descriptor context/proof headers") ||
        !checked_add_wire(metrics->operation_descriptor_resident_bytes_upper_bound,
                          descriptor_backing_reserve,
                          metrics->operation_descriptor_resident_bytes_upper_bound,
                          "descriptor backing and allocator reserve") ||
        !checked_add_wire(metrics->operation_descriptor_resident_bytes_upper_bound,
                          descriptor_compare_wire_bytes,
                          metrics->operation_descriptor_resident_bytes_upper_bound,
                          "expected/actual descriptor comparison buffers")) return false;

    if (analysis_only_shape) {
        NativeDescriptorShapeFootprint footprint;
        if (!zkdl_gat_native_descriptor_shape_footprint(
                config, graph, tables, plan, &footprint, error)) return false;
        if (footprint.operation_descriptor_wire_bytes != metrics->operation_descriptor_bytes ||
            footprint.operation_descriptor_max_bytes != metrics->operation_descriptor_max_bytes) {
            if (error) *error = "BLOCKED: analysis-only typed metrics disagree with the canonical descriptor shape wire count";
            return false;
        }
        metrics->operation_descriptor_resident_bytes_upper_bound =
            footprint.operation_descriptor_resident_bytes_upper_bound;
    }

    // Typed source payload for the one GatTrace held by produce().  These
    // formulas follow the owning vectors in HiddenHeadTrace,
    // OutputHeadTrace and SignedRoundDivWitness; edge gathers are E x D/C,
    // while aggregates are N x D/C.  They deliberately count the native
    // trace, not operation views (which are streamed by NativeFamilyStorage).
    std::uint64_t trace_i64_elements = 0U;
    std::uint64_t trace_field_elements = 0U;
    std::uint64_t trace_u8_elements = 0U;
    const auto add_shape_elements = [&](std::uint64_t *target, std::uint64_t rows,
                                        std::uint64_t columns, std::uint64_t replicas,
                                        const char *label) -> bool {
        std::uint64_t cells = 0U, total = 0U;
        if (!checked_mul_u64(rows, columns, cells) || !checked_mul_u64(cells, replicas, total)) {
            if (error) *error = std::string("native trace typed shape overflows: ") + label;
            return false;
        }
        return add(target, total, label);
    };
    const auto add_elements = [&](std::uint64_t *target, std::uint64_t count,
                                  std::uint64_t replicas, const char *label) -> bool {
        std::uint64_t total = 0U;
        if (!checked_mul_u64(count, replicas, total)) {
            if (error) *error = std::string("native trace typed count overflows: ") + label;
            return false;
        }
        return add(target, total, label);
    };
    std::uint64_t hidden_head_count = 0U;
    std::uint64_t maximum_hidden_dim = 0U;
    std::uint64_t maximum_hidden_width = graph.features;
    for (std::size_t layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const std::uint64_t dim = config.architecture.hidden_dims[layer];
        const std::uint64_t heads = config.architecture.hidden_heads.at(layer);
        std::uint64_t hidden_width = 0U;
        if (dim == 0U || heads == 0U || !checked_mul_u64(dim, heads, hidden_width) ||
            !add(&hidden_head_count, heads, "trace hidden head count")) {
            if (error && error->empty()) *error = "native trace architecture shape is invalid";
            return false;
        }
        maximum_hidden_dim = std::max(maximum_hidden_dim, dim);
        maximum_hidden_width = std::max(maximum_hidden_width, hidden_width);
        const std::uint64_t edge_matrix_count = 2U;
        const std::uint64_t hidden_node_i64_matrices = 5U;
        const std::uint64_t hidden_round_i64_arrays = 5U;
        // h_prime, aggregate_pre/num, elu_input, aggregate; routed/product
        // edge matrices; six edge vectors; maximum/denominator node vectors;
        // SignedRoundDiv's five int64 arrays.
        if (!add_shape_elements(&trace_i64_elements, graph.nodes, dim,
                                hidden_node_i64_matrices + hidden_round_i64_arrays,
                                "hidden node and round-div int64 arrays") ||
            !add_shape_elements(&trace_i64_elements, graph.edges, dim, edge_matrix_count,
                                "hidden edge matrices") ||
            !add_elements(&trace_i64_elements, graph.edges, 6U, "hidden edge vectors") ||
            !add_elements(&trace_i64_elements, graph.nodes, 2U, "hidden node vectors") ||
            // E-sized inverse/alpha fields, E x D product, two N x D
            // aggregate fields and two N x D SignedRoundDiv fields.
            !add_elements(&trace_field_elements, graph.edges, 2U, "hidden edge fields") ||
            !add_shape_elements(&trace_field_elements, graph.edges, dim, 1U,
                                "hidden edge field matrix") ||
            !add_shape_elements(&trace_field_elements, graph.nodes, dim, 4U,
                                "hidden node fields") ||
            !add_shape_elements(&trace_u8_elements, graph.nodes, dim, 2U,
                                "hidden round-div byte arrays")) return false;
    }
    const std::uint64_t output_dim = config.architecture.classes;
    const std::uint64_t output_heads = config.architecture.output_heads;
    if (output_dim == 0U || output_heads == 0U ||
        !add_shape_elements(&trace_i64_elements, graph.nodes, graph.features, 1U,
                            "trace first input") ||
        !add_shape_elements(&trace_i64_elements, graph.nodes, output_dim, 2U,
                            "trace output sum/logits") ||
        // Final-average SignedRoundDiv witness.
        !add_shape_elements(&trace_i64_elements, graph.nodes, output_dim, 5U,
                            "trace final-average int64 arrays") ||
        !add_shape_elements(&trace_field_elements, graph.nodes, output_dim, 2U,
                            "trace final-average field arrays") ||
        !add_shape_elements(&trace_u8_elements, graph.nodes, output_dim, 2U,
                            "trace final-average byte arrays")) {
        if (error && error->empty()) *error = "native trace output shape is invalid";
        return false;
    }
    for (std::size_t layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        const std::uint64_t width = config.architecture.hidden_heads[layer] * config.architecture.hidden_dims[layer];
        const std::uint64_t input_width = layer == 0U ? graph.features :
            config.architecture.hidden_heads[layer - 1U] * config.architecture.hidden_dims[layer - 1U];
        if (!add_shape_elements(&trace_i64_elements, graph.nodes, input_width, 1U,
                                "hidden layer input matrix") ||
            !add_shape_elements(&trace_i64_elements, graph.nodes, width, 1U,
                                "hidden layer concat matrix")) return false;
    }
    // Output heads own six node matrices (y_prime, aggregate_pre/num,
    // y_quantized, aligned and output), two edge matrices, 6 edge vectors,
    // two node vectors, one class bias, two SignedRoundDiv records (five
    // int64, two field and two byte vectors apiece), and typed field arrays.
    if (!add_shape_elements(&trace_i64_elements, graph.nodes, output_dim,
                            output_heads * 16U, "output head matrices and round-div arrays") ||
        !add_shape_elements(&trace_i64_elements, graph.edges, output_dim,
                            output_heads * 2U, "output head edge matrices") ||
        !add_elements(&trace_i64_elements, graph.edges, output_heads * 6U,
                      "output head edge vectors") ||
        !add_elements(&trace_i64_elements, graph.nodes, output_heads * 2U,
                      "output head node vectors") ||
        !add_elements(&trace_i64_elements, output_dim, output_heads,
                      "output head bias vectors") ||
        !add_shape_elements(&trace_field_elements, graph.nodes, output_dim,
                            output_heads * 6U, "output head field matrices and round-div arrays") ||
        !add_shape_elements(&trace_field_elements, graph.edges, output_dim,
                            output_heads, "output head edge field matrix") ||
        !add_elements(&trace_field_elements, graph.edges, output_heads * 2U,
                      "output head edge fields") ||
        !add_shape_elements(&trace_u8_elements, graph.nodes, output_dim,
                            output_heads * 4U, "output head round-div byte arrays")) return false;

    if (!checked_mul_u64(trace_i64_elements, sizeof(std::int64_t), metrics->trace_i64_payload_bytes) ||
        !checked_mul_u64(trace_field_elements, sizeof(FieldValue), metrics->trace_field_payload_bytes) ||
        !checked_mul_u64(trace_u8_elements, sizeof(std::uint8_t), metrics->trace_u8_payload_bytes)) {
        if (error) *error = "native trace typed payload byte count overflows";
        return false;
    }
    // Object metadata includes all trace containers and the contiguous
    // storage for architecture-sized layer/head objects.  Per-value vectors
    // above are constructed from exact shape on empty vectors in this source.
    std::uint64_t hidden_layer_object_bytes = 0U, hidden_head_object_bytes = 0U;
    std::uint64_t output_head_object_bytes = 0U, trace_metadata = sizeof(GatTrace);
    if (!checked_mul_u64(config.architecture.hidden_dims.size(), sizeof(HiddenLayerTrace), hidden_layer_object_bytes) ||
        !checked_mul_u64(hidden_head_count, sizeof(HiddenHeadTrace), hidden_head_object_bytes) ||
        !checked_mul_u64(output_heads, sizeof(OutputHeadTrace), output_head_object_bytes) ||
        !add(&trace_metadata, hidden_layer_object_bytes, "trace layer object metadata") ||
        !add(&trace_metadata, hidden_head_object_bytes, "trace hidden-head object metadata") ||
        !add(&trace_metadata, output_head_object_bytes, "trace output-head object metadata")) return false;
    metrics->trace_object_metadata_bytes = trace_metadata;

    std::uint64_t trace_payload = metrics->trace_i64_payload_bytes;
    if (!add(&trace_payload, metrics->trace_field_payload_bytes, "trace payload bytes") ||
        !add(&trace_payload, metrics->trace_u8_payload_bytes, "trace payload bytes") ||
        !add(&trace_payload, trace_metadata, "trace payload object metadata")) return false;
    // run_forward's actual overlap: current is a copy of graph input / the
    // previous concat; at concat, head_outputs copies every aggregate and
    // concat/current coexist; an active head retains projected while its
    // h_prime copy is in trace and attention_scores returns one E-vector.
    std::uint64_t current_elements = 0U, current_bytes = 0U;
    std::uint64_t max_active_projection_bytes = 0U, max_attention_return_bytes = 0U;
    std::uint64_t max_hidden_head_output_bytes = 0U;
    std::uint64_t max_output_projection_bytes = 0U, node_temporary_bytes = 0U;
    std::uint64_t matrix_header_bytes = 0U, max_head_output_count = 0U;
    if (!checked_mul_u64(graph.nodes, maximum_hidden_width, current_elements) ||
        !checked_mul_u64(current_elements, sizeof(std::int64_t), current_bytes) ||
        !checked_mul_u64(graph.nodes, maximum_hidden_dim, max_active_projection_bytes) ||
        !checked_mul_u64(max_active_projection_bytes, sizeof(std::int64_t), max_active_projection_bytes) ||
        !checked_mul_u64(graph.edges, sizeof(std::int64_t), max_attention_return_bytes) ||
        !checked_mul_u64(graph.nodes, maximum_hidden_dim, max_hidden_head_output_bytes) ||
        !checked_mul_u64(max_hidden_head_output_bytes, sizeof(std::int64_t), max_hidden_head_output_bytes) ||
        !checked_mul_u64(graph.nodes, output_dim, max_output_projection_bytes) ||
        !checked_mul_u64(max_output_projection_bytes, sizeof(std::int64_t), max_output_projection_bytes) ||
        !checked_mul_u64(graph.nodes, 2U, node_temporary_bytes) ||
        !checked_mul_u64(node_temporary_bytes, sizeof(std::int64_t), node_temporary_bytes) ||
        !checked_mul_u64(sizeof(Matrix), config.architecture.hidden_heads.empty() ? 0U :
                             *std::max_element(config.architecture.hidden_heads.begin(), config.architecture.hidden_heads.end()),
                         matrix_header_bytes)) {
        if (error) *error = "native trace live-shape upper bound overflows";
        return false;
    }
    for (std::size_t layer = 0U; layer < config.architecture.hidden_dims.size(); ++layer) {
        std::uint64_t elements = 0U, bytes = 0U;
        if (!checked_mul_u64(graph.nodes, config.architecture.hidden_dims[layer], elements) ||
            !checked_mul_u64(elements, config.architecture.hidden_heads[layer], elements) ||
            !checked_mul_u64(elements, sizeof(std::int64_t), bytes)) {
            if (error) *error = "native trace concat live shape overflows";
            return false;
        }
        max_hidden_head_output_bytes = std::max(max_hidden_head_output_bytes, bytes);
        max_head_output_count = std::max(max_head_output_count, config.architecture.hidden_heads[layer]);
    }
    std::uint64_t hidden_forward_temp = current_bytes;
    if (!add(&hidden_forward_temp, max_active_projection_bytes, "trace active hidden projection temp") ||
        !add(&hidden_forward_temp, max_attention_return_bytes, "trace hidden attention return temp") ||
        !add(&hidden_forward_temp, node_temporary_bytes, "trace hidden denominator temp") ||
        !add(&hidden_forward_temp, max_hidden_head_output_bytes, "trace concat head-output copies") ||
        !add(&hidden_forward_temp, current_bytes, "trace previous current/concat overlap") ||
        !add(&hidden_forward_temp, current_bytes, "trace current copy of concat") ||
        !add(&hidden_forward_temp, matrix_header_bytes, "trace head-output matrix headers") ||
        !add(&hidden_forward_temp, sizeof(Matrix) * 4U, "trace active matrix headers")) return false;
    std::uint64_t output_forward_temp = current_bytes;
    if (!add(&output_forward_temp, max_output_projection_bytes, "trace active output projection temp") ||
        !add(&output_forward_temp, max_attention_return_bytes, "trace output attention return temp") ||
        !add(&output_forward_temp, node_temporary_bytes, "trace output denominator copies") ||
        !add(&output_forward_temp, sizeof(Matrix) * 4U, "trace active output matrix headers")) return false;
    (void)max_head_output_count;
    metrics->trace_forward_live_peak_bytes_upper_bound = trace_payload;
    if (!add(&metrics->trace_forward_live_peak_bytes_upper_bound,
             std::max(hidden_forward_temp, output_forward_temp),
             "trace forward live peak")) return false;

    // Sparse-route Sumcheck keeps the validated source vector and, after the
    // first challenge, four folded states whose largest combined length is
    // two source domains.  Selector, zero and constant-output providers are
    // virtual; their only retained public data are the node/feature factors.
    // The source vector remains live for its authenticated opening after the
    // Sumcheck, while all provider/fold buffers stay segment-bounded.
    const std::uint64_t route_domain = metrics->maximum_route_source_domain;
    const std::uint64_t route_sumcheck_domain = std::max<std::uint64_t>(2U, route_domain);
    std::uint64_t route_source_bytes = 0U, route_folded_values = 0U;
    std::uint64_t route_folded_state_bytes = 0U, route_segment_bytes = 0U;
    std::uint64_t route_selector_factor_bytes = 0U;
    std::uint64_t route_selector_weight_values = 0U;
    std::uint64_t route_selector_weight_bytes = 0U;
    std::uint64_t route_output_segment_bytes = 0U;
    std::uint64_t route_source_opening_segment_bytes = 0U;
    std::uint64_t route_opening_value_bytes = 0U;
    std::uint64_t route_opening_generator_count = 0U;
    std::uint64_t route_opening_generator_bytes = 0U;
    const std::uint64_t route_weight_domain = std::max<std::uint64_t>(
        1U, metrics->maximum_route_edge_domain);
    const std::uint64_t route_low_weight_count = std::min<std::uint64_t>(
        route_weight_domain, 1U << 18U);
    const std::uint64_t route_high_weight_count =
        (route_weight_domain + route_low_weight_count - 1U) /
        route_low_weight_count;
    if (!checked_mul_u64(route_sumcheck_domain, sizeof(FrWire),
                         route_source_bytes) ||
        !checked_mul_u64(route_sumcheck_domain / 2U, 4U,
                         route_folded_values) ||
        !checked_mul_u64(route_folded_values, sizeof(FrWire),
                         route_folded_state_bytes) ||
        !checked_mul_u64(std::min<std::uint64_t>(
                             std::max(route_domain, metrics->maximum_route_edge_domain),
                             kSegmentSize), sizeof(FrWire) * 8U, route_segment_bytes) ||
        !checked_mul_u64(metrics->maximum_route_selector_factor_elements,
                         sizeof(FrWire),
                         route_selector_factor_bytes) ||
        !checked_add_u64(route_low_weight_count, route_high_weight_count,
                         route_selector_weight_values) ||
        !checked_mul_u64(route_selector_weight_values, sizeof(FrWire),
                         route_selector_weight_bytes) ||
        !checked_mul_u64(std::min<std::uint64_t>(metrics->maximum_route_edge_domain,
                                                kSegmentSize),
                         sizeof(FrWire), route_output_segment_bytes)) {
        if (error) *error = "native route folded-state live shape overflows";
        return false;
    }
    if (!checked_mul_u64(std::min<std::uint64_t>(route_sumcheck_domain, kSegmentSize),
                         sizeof(FrWire), route_source_opening_segment_bytes) ||
        !checked_add_u64(route_source_opening_segment_bytes,
                         route_output_segment_bytes, route_opening_value_bytes) ||
        !checked_mul_u64(std::min<std::uint64_t>(
                             std::max(route_sumcheck_domain, metrics->maximum_route_edge_domain),
                             kSegmentSize), 2U, route_opening_generator_count) ||
        !checked_mul_u64(route_opening_generator_count, sizeof(zkdl::CpuG1Point),
                         route_opening_generator_bytes)) {
        if (error) *error = "native route opening live shape overflows";
        return false;
    }
    // The output-domain upper bound also bounds each edge/feature weight
    // domain; weights are split at 2^18 and never allocate a full MLE table.
    // Factor storage uses the maximum node+source-feature count of one route
    // or FAC binding, not an unrelated source-domain upper bound.
    // Selector construction overlaps the retained source, node/feature
    // factors, one bounded weight pair, and the output opening segment.
    std::uint64_t selector_peak = route_source_bytes;
    if (!add(&selector_peak, route_selector_factor_bytes,
             "route selector factors") ||
        !add(&selector_peak, route_selector_weight_bytes,
             "route selector bounded MLE weights") ||
        !add(&selector_peak, route_output_segment_bytes,
             "route output opening segment")) return false;
    std::uint64_t sumcheck_peak = route_source_bytes;
    if (!add(&sumcheck_peak, route_folded_state_bytes,
             "route cached folded Sumcheck states") ||
        !add(&sumcheck_peak, route_segment_bytes,
             "route Sumcheck segment buffers") ||
        !add(&sumcheck_peak, route_output_segment_bytes,
             "route Sumcheck retained output segment") ||
        !add(&sumcheck_peak, route_selector_factor_bytes,
             "route Sumcheck retained selector factors") ||
        !add(&sumcheck_peak, 12U * sizeof(std::vector<FrWire>),
             "route Sumcheck vector metadata") ||
        !add(&sumcheck_peak, sizeof(relation::SparseRouteProof),
             "route proof object metadata")) return false;
    // After the fold-state function returns, both segment generator prefixes
    // and the two aggregated opening-value vectors coexist.  Keep this phase
    // separate from the 8 GiB cache phase so the bound reflects actual
    // lifetimes while still covering the authenticated source opening.
    std::uint64_t opening_peak = route_source_bytes;
    if (!add(&opening_peak, route_selector_factor_bytes,
             "route opening retained selector factors") ||
        !add(&opening_peak, route_opening_value_bytes,
             "route source/output aggregate values") ||
        !add(&opening_peak, route_opening_generator_bytes,
             "route source/output segment generators") ||
        !add(&opening_peak, sizeof(relation::SparseRouteProof),
             "route opening proof metadata")) return false;
    metrics->route_sumcheck_live_peak_bytes_upper_bound =
        std::max(selector_peak, std::max(sumcheck_peak, opening_peak));
    metrics->segmented_sumcheck_live_peak_bytes_upper_bound = std::max(
        metrics->route_sumcheck_live_peak_bytes_upper_bound,
        ordinary_segmented_sumcheck_peak);
    metrics->hyrax_opening_segment_domain_histogram.reserve(
        hyrax_opening_shape_counts.size());
    for (const auto &entry : hyrax_opening_shape_counts)
        metrics->hyrax_opening_segment_domain_histogram.push_back(entry.second);
    metrics->sumcheck_shape_histogram.reserve(sumcheck_shape_counts.size());
    for (const auto &entry : sumcheck_shape_counts)
        metrics->sumcheck_shape_histogram.push_back(entry.second);
    return true;
}

bool zkdl_gat_native_descriptor_shape_footprint(const Config &config, const Graph &graph,
                                                const StaticTables &tables, const RelationPlan &plan,
                                                NativeDescriptorShapeFootprint *footprint,
                                                std::string *error) {
    if (footprint == nullptr) {
        if (error) *error = "analysis-only descriptor shape footprint output is null";
        return false;
    }
    *footprint = NativeDescriptorShapeFootprint{};
    if (plan.registry_digest.empty() || plan.integer_bound_ledger_digest.empty() ||
        graph.nodes == 0U || graph.edges == 0U || graph.graphs == 0U ||
        plan.operation_families.empty() || plan.tensor_families.empty()) {
        if (error) *error = "BLOCKED: analysis-only descriptor footprint lacks checked plan/graph metadata";
        return false;
    }

    std::map<std::string, std::uint32_t> domains;
    std::map<std::string, const TensorFamily *> families;
    for (const auto &family : plan.tensor_families) {
        if (family.id.empty() || family.logical_shape.empty() ||
            family.logical_shape.size() != family.padded_shape.size() ||
            !families.emplace(family.id, &family).second) {
            if (error) *error = "BLOCKED: analysis-only descriptor footprint encountered an unknown family shape";
            return false;
        }
        std::uint64_t domain = 1U;
        for (std::size_t axis = 0U; axis < family.padded_shape.size(); ++axis) {
            if (family.logical_shape[axis] == 0U ||
                family.padded_shape[axis] < family.logical_shape[axis] ||
                !checked_mul_u64(domain, family.padded_shape[axis], domain) ||
                domain > kMaxNativeDomain) {
                if (error) *error = "BLOCKED: analysis-only descriptor footprint has invalid family axes: " + family.id;
                return false;
            }
        }
        if (!domains.emplace(family.id, static_cast<std::uint32_t>(domain)).second) {
            if (error) *error = "BLOCKED: analysis-only descriptor footprint found duplicate family identifiers";
            return false;
        }
    }

    // The analysis-only interner tracks scalar selector shapes and canonical
    // backing keys. It never creates graph-sized selector arrays. Each
    // operation is constructed as a small shell so FAC/constraint/string
    // capacities and the exact descriptor serializer can be measured from the
    // same construction branches as formal admission.
    DescriptorBackingInterner shape_interner(plan.registry_digest, &graph, true);
    std::uint64_t ordinary_per_side = 0U;
    std::uint64_t descriptor_wire_total = 0U;
    std::uint64_t descriptor_wire_max = 0U;

    const auto add_bytes = [&](std::uint64_t count, std::uint64_t element_size,
                               std::uint64_t &target, const char *label) -> bool {
        std::uint64_t bytes = 0U;
        if (!checked_mul_u64(count, element_size, bytes) ||
            !checked_add_u64(target, bytes, target)) {
            if (error) *error = std::string("BLOCKED: analysis-only descriptor capacity overflows: ") + label;
            return false;
        }
        return true;
    };
    const auto add_string_capacity = [&](const std::string &value,
                                         std::uint64_t &target) -> bool {
        return add_bytes(value.capacity() + 1U, sizeof(char), target,
                         "descriptor string");
    };
    const auto add_string_vector_capacity = [&](const std::vector<std::string> &values,
                                                std::uint64_t &target) -> bool {
        if (!add_bytes(values.capacity(), sizeof(std::string), target,
                       "descriptor string vector")) return false;
        for (const auto &value : values)
            if (!add_string_capacity(value, target)) return false;
        return true;
    };
    const auto add_operation_ordinary_capacity = [&](const relation::OperationDescriptor &operation,
                                                     std::uint64_t &target) -> bool {
        const std::string *const operation_strings[] = {
            &operation.id, &operation.mapping, &operation.padding,
            &operation.terminal_claim, &operation.terminal_point,
            &operation.selector_source_ref, &operation.selector_output_ref,
            &operation.batch_family, &operation.public_table_ref};
        for (const auto *value : operation_strings)
            if (!add_string_capacity(*value, target)) return false;
        if (!add_string_vector_capacity(operation.source_refs, target) ||
            !add_string_vector_capacity(operation.tensor_refs, target) ||
            !add_string_vector_capacity(operation.selector_refs, target) ||
            !add_bytes(operation.constants.capacity(), sizeof(FrWire), target,
                       "operation constants") ||
            !add_bytes(operation.constraints.capacity(), sizeof(relation::ConstraintDescriptor), target,
                       "operation constraints") ||
            !add_bytes(operation.primitives.capacity(), sizeof(relation::PrimitiveDescriptor), target,
                       "operation primitives") ||
            !add_bytes(operation.fac_bindings.capacity(), sizeof(relation::FacBindingDescriptor), target,
                       "operation FAC binding headers") ||
            !add_bytes(operation.public_table_selector.capacity(), sizeof(FrWire), target,
                       "public-table selector")) return false;
        for (const auto &constraint : operation.constraints) {
            if (!add_string_capacity(constraint.id, target) ||
                !add_bytes(constraint.terms.capacity(), sizeof(relation::Monomial), target,
                           "constraint monomials") ||
                !add_bytes(constraint.logical_shape.capacity(), sizeof(std::uint64_t), target,
                           "constraint logical shape") ||
                !add_bytes(constraint.padded_shape.capacity(), sizeof(std::uint64_t), target,
                           "constraint padded shape")) return false;
            for (const auto &term : constraint.terms)
                if (!add_bytes(term.refs.capacity(), sizeof(std::uint32_t), target,
                               "constraint term references")) return false;
        }
        for (const auto &primitive : operation.primitives)
            if (!add_string_vector_capacity(primitive.tensor_refs, target)) return false;
        for (const auto &binding : operation.fac_bindings) {
            if (!add_string_capacity(binding.view_ref, target) ||
                !add_string_capacity(binding.canonical_ref, target)) return false;
        }
        return true;
    };

    for (const auto &plan_operation : plan.operation_families) {
        shape_interner.begin_shape_operation();
        std::string operation_error;
        relation::OperationDescriptor operation = canonical_operation(
            config, tables, plan.integer_bound_ledger, plan_operation, domains,
            families, &graph, shape_interner, &operation_error);
        if (operation.id.empty()) {
            if (error) *error = "BLOCKED: analysis-only descriptor shape dispatch failed for " +
                plan_operation.id + (operation_error.empty() ? std::string() : ": " + operation_error);
            return false;
        }
        if (!add_operation_ordinary_capacity(operation, ordinary_per_side)) return false;
        zkdl::WireSizeCounter wire_counter;
        try {
            relation::append_operation(wire_counter, operation);
        } catch (const std::exception &exception) {
            if (error) *error = "BLOCKED: analysis-only descriptor wire counter failed for " +
                plan_operation.id + ": " + exception.what();
            return false;
        }
        std::uint64_t descriptor_bytes = 0U;
        if (!checked_add_u64(wire_counter.size(),
                             shape_interner.shape_operation_wire_payload_delta(),
                             descriptor_bytes) ||
            !checked_add_u64(descriptor_wire_total, descriptor_bytes,
                             descriptor_wire_total)) {
            if (error) *error = "BLOCKED: analysis-only descriptor wire total overflows";
            return false;
        }
        descriptor_wire_max = std::max(descriptor_wire_max, descriptor_bytes);
    }

    if (shape_interner.shape_fac_binding_count() == 0U ||
        descriptor_wire_total == 0U || descriptor_wire_max == 0U) {
        if (error) *error = "BLOCKED: analysis-only descriptor shape traversal omitted selector/descriptor arrays";
        return false;
    }

    std::uint64_t context_operation_capacity = 1U;
    while (context_operation_capacity < plan.operation_families.size()) {
        if (context_operation_capacity > std::numeric_limits<std::uint64_t>::max() / 2U) {
            if (error) *error = "BLOCKED: analysis-only operation header capacity overflows";
            return false;
        }
        context_operation_capacity *= 2U;
    }
    std::uint64_t context_header_bytes = 0U, proof_header_bytes = 0U;
    std::uint64_t compare_wire_bytes = 0U, expected_selector_bytes = 0U;
    std::uint64_t interner_metadata_bytes = sizeof(shape_interner);
    std::uint64_t selector_allocator_metadata_bytes = 0U;
    std::uint64_t resident_bytes = 0U;
    const auto add_interner_nodes = [&](std::uint64_t entries, std::uint64_t key_bytes,
                                        std::uint64_t value_bytes) -> bool {
        std::uint64_t node_bytes = 0U;
        if (!checked_add_u64(key_bytes, value_bytes, node_bytes) ||
            !checked_add_u64(node_bytes, 4U * sizeof(void *), node_bytes) ||
            !add_bytes(entries, node_bytes, interner_metadata_bytes,
                       "interner backing map nodes")) return false;
        return true;
    };
    const std::uint64_t outer_selector_object_bytes =
        sizeof(relation::IndexMapU32) + sizeof(relation::CowByteVector);
    if (!checked_add_u64(interner_metadata_bytes, plan.registry_digest.capacity() + 1U,
                         interner_metadata_bytes) ||
        !add_interner_nodes(shape_interner.shape_prefix_active_entries(),
                            sizeof(std::pair<std::uint32_t, std::uint32_t>),
                            sizeof(relation::CowByteVector)) ||
        !add_interner_nodes(shape_interner.shape_identity_outer_entries(),
                            sizeof(std::uint32_t), outer_selector_object_bytes) ||
        !add_interner_nodes(shape_interner.shape_logical_outer_entries(),
                            sizeof(std::tuple<std::uint64_t, std::uint32_t, bool>),
                            outer_selector_object_bytes) ||
        !add_interner_nodes(shape_interner.shape_graph_route_entries(),
                            sizeof(std::pair<std::uint32_t, bool>),
                            sizeof(relation::IndexMapU32)) ||
        !add_interner_nodes(shape_interner.shape_concat_outer_entries(),
                            sizeof(std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>),
                            outer_selector_object_bytes)) {
        if (error && error->empty()) *error = "BLOCKED: analysis-only interner bookkeeping overflows";
        return false;
    }
    // Each explicit shared_ptr-backed vector has one allocator/control-block
    // envelope.  64 bytes covers its shared control block plus vector-buffer
    // allocator bookkeeping; this does not alter any authenticated wire size.
    std::uint64_t expected_backing_allocations = 0U;
    std::uint64_t received_backing_allocations = 0U;
    std::uint64_t feature_backing_allocations = 0U, direct_route_backing_allocations = 0U;
    if (!checked_add_u64(shape_interner.shape_unique_outer_map_count(),
                         shape_interner.shape_unique_active_mask_count(),
                         expected_backing_allocations) ||
        !checked_add_u64(shape_interner.shape_fac_binding_count(),
                         shape_interner.shape_route_count(), feature_backing_allocations) ||
        !checked_add_u64(expected_backing_allocations, feature_backing_allocations,
                         expected_backing_allocations) ||
        !checked_mul_u64(shape_interner.shape_fac_binding_count(), 3U,
                         received_backing_allocations) ||
        !checked_mul_u64(shape_interner.shape_route_count(), 4U,
                         direct_route_backing_allocations) ||
        !checked_add_u64(received_backing_allocations, direct_route_backing_allocations,
                         received_backing_allocations) ||
        !checked_add_u64(expected_backing_allocations, received_backing_allocations,
                         expected_backing_allocations) ||
        !checked_mul_u64(expected_backing_allocations, 64U,
                         selector_allocator_metadata_bytes)) {
        if (error) *error = "BLOCKED: analysis-only selector allocator metadata overflows";
        return false;
    }
    if (!checked_mul_u64(context_operation_capacity,
                         sizeof(relation::OperationDescriptor), context_header_bytes) ||
        !checked_mul_u64(plan.operation_families.size(),
                         sizeof(relation::OperationProof), proof_header_bytes) ||
        !checked_add_u64(proof_header_bytes, sizeof(relation::OperationProof), proof_header_bytes) ||
        !checked_mul_u64(descriptor_wire_max, 2U, compare_wire_bytes) ||
        !checked_add_u64(shape_interner.expected_unique_outer_map_bytes(),
                         shape_interner.expected_unique_active_mask_bytes(), expected_selector_bytes) ||
        !checked_add_u64(expected_selector_bytes, shape_interner.expected_feature_map_bytes(),
                         expected_selector_bytes) ||
        !checked_add_u64(context_header_bytes, proof_header_bytes, resident_bytes) ||
        !checked_add_u64(resident_bytes, interner_metadata_bytes, resident_bytes) ||
        !checked_add_u64(resident_bytes, ordinary_per_side, resident_bytes) ||
        !checked_add_u64(resident_bytes, ordinary_per_side, resident_bytes) ||
        !checked_add_u64(resident_bytes, expected_selector_bytes, resident_bytes) ||
        !checked_add_u64(resident_bytes, shape_interner.received_selector_array_bytes(), resident_bytes) ||
        !checked_add_u64(resident_bytes, selector_allocator_metadata_bytes, resident_bytes) ||
        !checked_add_u64(resident_bytes, compare_wire_bytes, resident_bytes)) {
        if (error) *error = "BLOCKED: analysis-only descriptor resident capacity overflows";
        return false;
    }

    footprint->operation_count = plan.operation_families.size();
    footprint->fac_binding_count = shape_interner.shape_fac_binding_count();
    footprint->route_count = shape_interner.shape_route_count();
    footprint->expected_unique_outer_map_bytes = shape_interner.expected_unique_outer_map_bytes();
    footprint->expected_unique_active_mask_bytes = shape_interner.expected_unique_active_mask_bytes();
    footprint->expected_feature_map_bytes = shape_interner.expected_feature_map_bytes();
    footprint->received_selector_array_bytes = shape_interner.received_selector_array_bytes();
    if (!checked_mul_u64(ordinary_per_side, 2U,
                         footprint->ordinary_descriptor_capacity_bytes)) {
        if (error) *error = "BLOCKED: analysis-only ordinary descriptor capacity overflows";
        return false;
    }
    if (!checked_add_u64(context_header_bytes, proof_header_bytes,
                         footprint->descriptor_object_header_bytes)) {
        if (error) *error = "BLOCKED: analysis-only descriptor header capacity overflows";
        return false;
    }
    footprint->interner_metadata_bytes = interner_metadata_bytes;
    footprint->selector_allocator_metadata_bytes = selector_allocator_metadata_bytes;
    footprint->descriptor_compare_buffer_bytes = compare_wire_bytes;
    footprint->operation_descriptor_wire_bytes = descriptor_wire_total;
    footprint->operation_descriptor_max_bytes = descriptor_wire_max;
    footprint->operation_descriptor_resident_bytes_upper_bound = resident_bytes;
    return true;
}

bool zkdl_gat_relation_prove(const relation::ProverContext &context, relation::Proof *proof, std::string *error) {
    if (!proof) {
        if (error) *error = "relation proof output pointer is null";
        return false;
    }
    return relation::prove(context, *proof, error);
}

bool zkdl_gat_relation_verify(const relation::Proof &proof, const relation::VerifierContext &context, std::string *error) {
    return relation::verify(proof, context, error);
}

bool zkdl_gat_native_core_prove(const NativeProveRequest &request, NativeProofResult *result, std::string *error) {
    if (!result) {
        if (error) *error = "native prove result pointer is null";
        return false;
    }
    relation::ProverContext context;
    NativeProveContextTiming context_timing;
    double commitment_open_seconds = 0.0;
    double relation_commitment_seconds = 0.0;
    double provider_file_io_decode_seconds = 0.0;
    double relation_commitment_file_io_decode_seconds = 0.0;
    context.commitment_open_time_seconds = &commitment_open_seconds;
    context.relation_commitment_time_seconds = &relation_commitment_seconds;
    context.provider_file_io_decode_time_seconds = &provider_file_io_decode_seconds;
    context.relation_commitment_file_io_decode_time_seconds = &relation_commitment_file_io_decode_seconds;
    if (!context_from_request(request, context, context_timing, error))
        return reject(result, error, error ? *error : "native relation context rejected", request.config, request.statement, request.backend);
    context.gpu_backend = request.backend == Backend::Gpu;
    const std::uint64_t gpu_kernel_before = zkdl::gpu_crypto_kernel_invocations();
    const double provider_io_before_prove = provider_file_io_decode_seconds;
    const auto prove_start = std::chrono::steady_clock::now();
    relation::Proof proof;
    if (!relation::prove_consuming(context, proof, error))
        return reject(result, error, error ? *error : "native relation prover rejected witness", request.config, request.statement, request.backend);
    const double relation_prove_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prove_start).count();
    const double provider_io_during_prove = provider_file_io_decode_seconds - provider_io_before_prove;
    const double provider_io_outside_commitments = provider_io_during_prove - relation_commitment_file_io_decode_seconds;
    const double context_file_io_outside_commitments = context_timing.explicit_file_io_seconds -
        context_timing.sorted_lookup_outer_commitment_file_io_seconds;
    const double relation_context_prove_seconds = context_timing.relation_context_wall_seconds -
        context_timing.blinder_sampling_seconds - context_timing.sorted_lookup_outer_commitment_wall_seconds -
        context_file_io_outside_commitments;
    const double sorted_lookup_generation_seconds = request.witness->sorted_lookup_protocol_construction_seconds;
    const double proof_auxiliary_construction_seconds = request.witness->proof_auxiliary_construction_seconds;
    const std::uint64_t gpu_kernel_after = zkdl::gpu_crypto_kernel_invocations();
    if (context.gpu_backend && gpu_kernel_after <= gpu_kernel_before)
        return reject(result, error, "BLOCKED: formal GPU prover completed without a native cryptographic kernel", request.config, request.statement,
                      request.backend);
    const auto serialization_start = std::chrono::steady_clock::now();
    NativeProgressScope wire_progress("proof_wire_serialize");
    result->proof_bytes = proof.serialize();
    wire_progress.pass("proof_bytes=" + std::to_string(result->proof_bytes.size()));
    const double serialization_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - serialization_start).count();
    if (relation_commitment_seconds > relation_prove_seconds ||
        provider_io_during_prove < relation_commitment_file_io_decode_seconds ||
        provider_io_during_prove > relation_prove_seconds ||
        relation_prove_seconds - relation_commitment_seconds - provider_io_outside_commitments < 0.0 ||
        context_timing.sorted_lookup_outer_commitment_wall_seconds < context_timing.sorted_lookup_outer_commitment_file_io_seconds ||
        context_file_io_outside_commitments < 0.0 || relation_context_prove_seconds < 0.0 ||
        sorted_lookup_generation_seconds < 0.0 || proof_auxiliary_construction_seconds < 0.0 ||
        !std::isfinite(context_timing.relation_context_wall_seconds) ||
        !std::isfinite(context_timing.blinder_sampling_seconds) ||
        !std::isfinite(context_timing.sorted_lookup_outer_commitment_wall_seconds) ||
        !std::isfinite(context_timing.sorted_lookup_outer_commitment_file_io_seconds) ||
        !std::isfinite(context_timing.explicit_file_io_seconds) || !std::isfinite(sorted_lookup_generation_seconds) ||
        !std::isfinite(proof_auxiliary_construction_seconds)) {
        return reject(result, error, "native proof metric accounting exceeded its measured stage boundary",
                      request.config, request.statement, request.backend);
    }
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
    result->envelope.proof_auxiliary_construction_seconds = proof_auxiliary_construction_seconds;
    result->envelope.sorted_lookup_protocol_construction_seconds = sorted_lookup_generation_seconds;
    result->envelope.relation_context_prove_seconds = relation_context_prove_seconds;
    result->envelope.relation_prove_wall_seconds = relation_prove_seconds;
    result->envelope.relation_commitment_time_seconds = relation_commitment_seconds;
    result->envelope.prove_provider_file_io_excluded_seconds = provider_io_outside_commitments;
    result->envelope.proof_serialization_seconds = serialization_seconds;
    result->envelope.proof_auxiliary_construction_scope =
        "signed_round_aux_witness_and_relation_views;steady_clock_wall;exclude_family_storage_write_hash_and_explicit_file_io";
    result->envelope.prove_time_seconds = proof_auxiliary_construction_seconds + sorted_lookup_generation_seconds + relation_context_prove_seconds +
        relation_prove_seconds - relation_commitment_seconds - provider_io_outside_commitments + serialization_seconds;
    if (!std::isfinite(result->envelope.commitment_time_seconds) || result->envelope.commitment_time_seconds < 0.0 ||
        !std::isfinite(result->envelope.prove_time_seconds) || result->envelope.prove_time_seconds < 0.0)
        return reject(result, error, "native proof commitment/prove timing is negative or non-finite",
                      request.config, request.statement, request.backend);
    result->envelope.verify_time_seconds = 0.0;
    result->envelope.verifier_reads_witness = false;
    result->envelope.gpu_kernel_executed = context.gpu_backend;
    result->envelope.silent_fallback = false;
    result->envelope.output_bound = true;
    result->envelope.error.clear();
    return true;
}

bool zkdl_gat_native_core_verify(const NativeVerifyRequest &request, std::string *error) {
    if (!request.config || !request.statement || !request.envelope || !request.proof_bytes)
        return reject(nullptr, error, "native verify request lacks proof envelope/bytes", request.config, request.statement, request.backend);
    if (request.envelope->schema_id != "zkDL_gat_proof_envelope_v1" || request.envelope->statement_id != request.statement->statement_id ||
        request.envelope->backend != backend_name(request.backend) || request.envelope->proof_scope != kProofScope || request.envelope->status != "PROVED" ||
        request.envelope->silent_fallback || request.envelope->verifier_reads_witness || !request.envelope->output_bound ||
        request.envelope->native_core_api != kBridgeApi ||
        request.envelope->static_table_descriptor_digest != request.statement->static_table_descriptor_digest ||
        request.envelope->scale_chain_digest != request.statement->scale_chain_digest ||
        request.envelope->integer_bound_ledger_id != request.statement->integer_bound_ledger_id ||
        request.envelope->integer_bound_ledger_digest != request.statement->integer_bound_ledger_digest ||
        request.envelope->integer_bound_ledger_descriptor_digest != request.statement->integer_bound_ledger_descriptor_digest) {
        return reject(nullptr, error, "proof envelope is not an accepted native relation proof", request.config, request.statement, request.backend);
    }
    relation::VerifierContext context;
    if (!context_from_verify(request, context, error))
        return reject(nullptr, error, error ? *error : "independent verifier context rejected", request.config, request.statement, request.backend);
    NativeRelationPlanMetrics typed_plan_metrics;
    std::string typed_plan_error;
    if (request.graph == nullptr || request.tables == nullptr || request.relation == nullptr ||
        !zkdl_gat_native_relation_plan_metrics(*request.config, *request.graph,
                                               *request.tables, *request.relation,
                                               &typed_plan_metrics, &typed_plan_error)) {
        return reject(nullptr, error,
                      typed_plan_error.empty()
                          ? "independent verifier lacks its authenticated typed relation plan"
                          : typed_plan_error,
                      request.config, request.statement, request.backend);
    }
    if (request.proof_bytes->size() > typed_plan_metrics.proof_wire_bytes_upper_bound) {
        return reject(nullptr, error,
                      "independent proof bytes exceed the authenticated typed wire upper bound",
                      request.config, request.statement, request.backend);
    }
    std::vector<std::string> expected_qpad_ids;
    for (const auto &descriptor : context.tensor_descriptors) {
        if (relation::qpad_record_required(
                descriptor.logical_shape, descriptor.padded_shape,
                descriptor.visibility == 1U))
            expected_qpad_ids.push_back(descriptor.id);
    }
    if (expected_qpad_ids.size() != typed_plan_metrics.padding_family_count) {
        return reject(nullptr, error,
                      "independent verifier Qpad context disagrees with authenticated plan metrics",
                      request.config, request.statement, request.backend);
    }
    context.gpu_backend = request.backend == Backend::Gpu;
    const std::uint64_t gpu_kernel_before = zkdl::gpu_crypto_kernel_invocations();
    // Include verifier-facing proof parsing/canonical decoding and package
    // identity checks.  Resource loading and verifier-context construction
    // above remain outside this cryptographic verification interval.
    const auto verify_start = std::chrono::steady_clock::now();
    NativeProgressScope parse_progress("proof_wire_parse");
    relation::Proof proof;
    std::string parse_error;
    if (!relation::Proof::deserialize(*request.proof_bytes, proof,
                                      expected_qpad_ids,
                                      typed_plan_metrics.padding_record_body_max_bytes,
                                      typed_plan_metrics.terminal_relation_body_max_bytes,
                                      typed_plan_metrics.sorted_lookup_relation_body_max_bytes,
                                      &parse_error))
        return reject(nullptr, error, "native GAT relation proof parse failed: " + parse_error, request.config, request.statement, request.backend);
    parse_progress.pass("proof_bytes=" + std::to_string(request.proof_bytes->size()));
    if (request.envelope->proof_size_bytes != request.proof_bytes->size() || request.envelope->proof_id != digest_hex(zkdl::sha256(*request.proof_bytes))) {
        return reject(nullptr, error, "proof envelope size/id is not bound to native proof bytes", request.config, request.statement, request.backend);
    }
    NativeProgressScope verify_progress("independent_relation_verify");
    const bool verified = relation::verify(
        proof, context, error,
        typed_plan_metrics.terminal_relation_body_max_bytes);
    if (verified) verify_progress.pass();
    const auto verify_end = std::chrono::steady_clock::now();
    if (!verified)
        return reject(nullptr, error, error ? *error : "native relation verifier rejected proof", request.config, request.statement, request.backend);
    const std::uint64_t gpu_kernel_after = zkdl::gpu_crypto_kernel_invocations();
    if (context.gpu_backend && gpu_kernel_after <= gpu_kernel_before)
        return reject(nullptr, error, "BLOCKED: formal GPU verifier completed without a native cryptographic kernel", request.config, request.statement,
                      request.backend);
    if ((!context.gpu_backend && request.envelope->gpu_kernel_executed) || (context.gpu_backend && !request.envelope->gpu_kernel_executed))
        return reject(nullptr, error, "GPU cryptographic provenance flag is inconsistent", request.config, request.statement, request.backend);
    request.envelope->verify_time_seconds = std::chrono::duration<double>(verify_end - verify_start).count();
    return true;
}

}
