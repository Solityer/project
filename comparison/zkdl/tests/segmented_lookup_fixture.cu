#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define ZKDL_SORTED_LOOKUP_FIXTURE_INCLUDE
#include "sorted_lookup_prefix_cache_check.cu"
#undef ZKDL_SORTED_LOOKUP_FIXTURE_INCLUDE

#include "../zkrelation-proof.cuh"
#include "../gat_adapter/gat_native_resources.hpp"
#include "../gat_adapter/gat_native_resources.cu"

namespace zkdl_gat::relation::fixture {

bool fixture_fail(const char *stage, const std::string &reason, std::string *error = nullptr) {
    std::string detail = reason;
    if (error != nullptr && !error->empty() && *error != reason) {
        detail += ": ";
        detail += *error;
    }
    if (error != nullptr) *error = detail;
    std::cerr << "segmented_fixture stage=" << stage << " status=FAIL reason=" << detail << '\n';
    return false;
}

void fixture_pass(const char *stage, const char *reason) { std::cerr << "segmented_fixture stage=" << stage << " status=PASS reason=" << reason << '\n'; }

bool descriptor_backing_cow_fixture() {
    OperationDescriptor descriptor;
    descriptor.id = "fixture/cow-selector";
    descriptor.kind = RelationKind::SparseRoute;
    descriptor.degree = 2U;
    descriptor.source_refs = {"input", "output"};
    descriptor.tensor_refs = descriptor.source_refs;
    descriptor.mapping = "fixture-selector";
    descriptor.padding = "zero-mask";
    descriptor.terminal_claim = "output";
    descriptor.terminal_point = "output";
    descriptor.selector_edge_domain = 2U;
    descriptor.selector_node_domain = 2U;
    descriptor.selector_feature_domain = 1U;
    descriptor.selector_source_feature_domain = 1U;
    descriptor.selector_src = {1U, 0U};
    descriptor.selector_dst = {1U, 0U};
    descriptor.selector_active = {1U, 0U};
    descriptor.selector_feature_map = {0U};
    descriptor.selector_source_ref = "input";
    descriptor.selector_output_ref = "output";

    OperationDescriptor shared = descriptor;
    if (shared.selector_active.backing_identity() !=
            descriptor.selector_active.backing_identity() ||
        shared.selector_src.backing_identity() !=
            descriptor.selector_src.backing_identity())
        return fixture_fail("descriptor-cow", "descriptor copy did not share selector backing");
    std::vector<byte> descriptor_wire, shared_wire;
    append_operation(descriptor_wire, descriptor);
    append_operation(shared_wire, shared);
    if (descriptor_wire != shared_wire)
        return fixture_fail("descriptor-cow", "shared descriptor serialization changed canonical bytes");

    shared.selector_active[1U] = 1U;
    shared.selector_src[0U] = 0U;
    if (descriptor.selector_active[1U] != 0U ||
        static_cast<std::uint32_t>(descriptor.selector_src[0U]) != 1U ||
        shared.selector_active.backing_identity() ==
            descriptor.selector_active.backing_identity() ||
        shared.selector_src.backing_identity() ==
            descriptor.selector_src.backing_identity())
        return fixture_fail("descriptor-cow", "mutable selector access did not detach safely");
    OperationDescriptor independent = descriptor;
    independent.selector_active.assign({1U, 1U});
    independent.selector_src.assign({0U, 0U});
    std::vector<byte> detached_wire, independent_wire;
    append_operation(detached_wire, shared);
    append_operation(independent_wire, independent);
    if (detached_wire != independent_wire)
        return fixture_fail("descriptor-cow", "detached descriptor serialization differs from equal values");
    fixture_pass("descriptor-cow", "descriptor copies share backing, writes detach, and wire bytes stay canonical");
    return true;
}

std::string hex_digest(const std::array<zkdl::byte, 32U> &digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string text;
    text.reserve(64U);
    for (const auto value : digest) {
        text.push_back(kHex[(value >> 4U) & 0x0fU]);
        text.push_back(kHex[value & 0x0fU]);
    }
    return text;
}

std::string trim_config_text(const std::string &value) {
    std::size_t begin = 0U;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1U]))) --end;
    return value.substr(begin, end - begin);
}

std::string quote_config_text(const std::string &value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(byte));
        } else if (byte == '\n') {
            result += "\\n";
        } else if (byte == '\r') {
            result += "\\r";
        } else if (byte == '\t') {
            result += "\\t";
        } else {
            result.push_back(static_cast<char>(byte));
        }
    }
    result.push_back('"');
    return result;
}

bool load_authenticated_test_config(const std::filesystem::path &path,
                                    Config *config, std::string *error) {
    if (config == nullptr) {
        if (error) *error = "test config output is null";
        return false;
    }
    std::ifstream input(path);
    if (!input) {
        if (error) *error = "cannot read authenticated SRS test config: " + path.string();
        return false;
    }
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        const std::string normalized = trim_config_text(line);
        if (normalized.empty()) continue;
        const std::size_t separator = normalized.find('=');
        if (separator == std::string::npos) {
            if (error) *error = "malformed authenticated SRS test config line";
            return false;
        }
        const std::string key = trim_config_text(normalized.substr(0U, separator));
        const std::string value = trim_config_text(normalized.substr(separator + 1U));
        if (key.empty() || value.empty() || !values.emplace(key, value).second) {
            if (error) *error = "duplicate or empty authenticated SRS test config key/value";
            return false;
        }
    }
    for (const char *key : {"srs_path", "srs_manifest_digest", "srs_id", "srs_type", "chunking_rule"}) {
        if (values.find(key) == values.end()) {
            if (error) *error = std::string("authenticated SRS test config lacks ") + key;
            return false;
        }
    }
    std::string canonical = "{";
    bool first = true;
    for (const auto &entry : values) {
        if (entry.first == "srs_manifest_digest" || entry.first == "verifier_registry_digest") continue;
        if (!first) canonical.push_back(',');
        first = false;
        canonical += quote_config_text(entry.first) + ":" + quote_config_text(entry.second);
    }
    canonical.push_back('}');
    config->baseline_root = std::filesystem::current_path();
    config->config_path = std::filesystem::absolute(path).lexically_normal();
    config->srs_path = values.at("srs_path");
    config->srs_manifest_digest = values.at("srs_manifest_digest");
    config->srs_id = values.at("srs_id");
    config->srs_type = values.at("srs_type");
    config->chunking_rule = values.at("chunking_rule");
    config->config_binding_digest = hex_digest(zkdl::sha256(
        std::vector<zkdl::byte>(canonical.begin(), canonical.end())));
    return true;
}

bool write_authenticated_resource_fixture(const std::filesystem::path &root, std::string *error) {
    const std::string manifest = "{\"manifest_format\":\"zkgat_srs_manifest\","
                                 "\"manifest_version\":2,\"basis_domain\":8,"
                                 "\"logical_domain\":32,\"segment_size\":8,"
                                 "\"segment_count\":4,\"domain\":8,"
                                 "\"chunk_id\":\"fixture-4seg\","
                                 "\"generator_digest\":\"fixture-generator\"}\n";
    const auto manifest_bytes = std::vector<zkdl::byte>(manifest.begin(), manifest.end());
    const auto manifest_digest = zkdl::sha256(manifest_bytes);
    const std::string registry = "{\"schema\":\"zkDL_gat_model_registry_v3\","
                                 "\"srs_manifest_digest\":\"" +
                                 hex_digest(manifest_digest) +
                                 "\","
                                 "\"basis_domain\":8,\"logical_domain\":32,"
                                 "\"segment_size\":8,\"segment_count\":4}\n";
    std::ofstream manifest_file(root / "srs_manifest.json", std::ios::binary);
    std::ofstream registry_file(root / "model_registry.json", std::ios::binary);
    if (!manifest_file || !registry_file) {
        if (error) *error = "fixture cannot create authenticated resource files";
        return fixture_fail("resource-admission", "cannot create authenticated resource files", error);
    }
    manifest_file << manifest;
    registry_file << registry;
    manifest_file.close();
    registry_file.close();

    AuthenticatedSrsResource::Chunk chunk;
    chunk.domain = 8U;
    chunk.point_count = 8U;
    chunk.logical_domain = 32U;
    chunk.basis_domain = 8U;
    chunk.segment_size = 8U;
    chunk.segment_count = 4U;
    for (std::uint64_t segment = 0U; segment < 4U; ++segment) {
        AuthenticatedSrsResource::Chunk::Segment frame;
        frame.segment_index = segment;
        frame.logical_offset = segment * 8U;
        frame.actual_length = 8U;
        frame.padded_length = 8U;
        frame.basis_offset = 0U;
        frame.basis_count = 8U;
        chunk.segments.push_back(frame);
    }
    const bool frame_admission = chunk.logical_domain > chunk.basis_domain && chunk.segment_count == chunk.logical_domain / chunk.segment_size &&
                                 chunk.segments.size() == chunk.segment_count;
    std::ifstream manifest_check(root / "srs_manifest.json", std::ios::binary);
    std::ifstream registry_check(root / "model_registry.json", std::ios::binary);
    std::string manifest_read((std::istreambuf_iterator<char>(manifest_check)), {});
    std::string registry_read((std::istreambuf_iterator<char>(registry_check)), {});
    if (!frame_admission || manifest_read != manifest || registry_read.find(hex_digest(manifest_digest)) == std::string::npos ||
        registry_read.find("\"logical_domain\":32") == std::string::npos) {
        if (error) *error = "fixture authenticated resource metadata is not canonical";
        return fixture_fail("resource-admission", "manifest/registry/frame metadata is not canonical", error);
    }
    fixture_pass("resource-admission", "logical32 is admitted over four physical frame8 segments");
    return true;
}

bool prepared_msm_chunk_fixture(
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
    std::string* error);


bool qpad_same_mask_group_fixture(std::string* error = nullptr) {
    constexpr std::uint32_t kDomain = 16U;
    constexpr std::uint32_t kBasisSize = 8U;
    std::vector<zkdl::CpuG1Point> generators;
    generators.reserve(kBasisSize);
    for (std::uint32_t index = 0U; index < kBasisSize; ++index)
        generators.push_back(zkdl::cpu_g1_mul(
            zkdl::cpu_g1_generator(), zkdl::fr_from_u32(index + 2U)));
    const auto h = zkdl::cpu_g1_mul(
        zkdl::cpu_g1_generator(), zkdl::fr_from_u32(101U));
    const auto u = zkdl::cpu_g1_mul(
        zkdl::cpu_g1_generator(), zkdl::fr_from_u32(103U));
    zkdl::CpuIpaBasisContext basis;
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(generators), h, u, basis, error))
        return fixture_fail("qpad-basis", "small authenticated basis was rejected", error);
    const auto& owned_generators = basis.generators();
    zkdl::CpuG1MsmSession msm_session(basis);

    RelationStatement base_statement;
    base_statement.statement_digest = zkdl::sha256(std::vector<byte>{'s'});
    base_statement.model_digest = zkdl::sha256(std::vector<byte>{'m'});
    base_statement.table_digest = zkdl::sha256(std::vector<byte>{'t'});
    base_statement.ledger_digest = zkdl::sha256(std::vector<byte>{'l'});
    base_statement.srs_digest = zkdl::sha256(std::vector<byte>{'p'});
    base_statement.registry_digest = zkdl::sha256(std::vector<byte>{'r'});

    TensorDescriptor descriptor_a;
    descriptor_a.id = "fixture/qpad-rlc/a";
    descriptor_a.visibility = 3U;
    descriptor_a.domain_size = kDomain;
    descriptor_a.logical_shape = {5U};
    descriptor_a.padded_shape = {kDomain};
    descriptor_a.mle_order = "row-major";
    descriptor_a.stack_axis = "first-axis";
    descriptor_a.mapping = "input-a";
    descriptor_a.padding_rule = "all-padding-zero-mask";
    TensorDescriptor descriptor_b = descriptor_a;
    descriptor_b.id = "fixture/qpad-rlc/b";
    descriptor_b.mle_order = "different-text-order";
    descriptor_b.stack_axis = "different-text-axis";
    descriptor_b.mapping = "different-text-mapping";
    TensorDescriptor descriptor_c = descriptor_a;
    descriptor_c.id = "fixture/qpad-rlc/c";
    descriptor_c.logical_shape = {6U};
    descriptor_c.mapping = "different-mask";
    const std::vector<TensorDescriptor> descriptors{
        descriptor_a, descriptor_b, descriptor_c};

    for (const auto& descriptor : descriptors) {
        SegmentedTensorProvider selector;
        if (!qpad_selector_provider_unhashed(
                descriptor, owned_generators.size(), selector, error))
            return fixture_fail("qpad-public-selector", "public selector construction failed", error);
        std::uint64_t offset = 0U;
        for (std::uint32_t segment = 0U; segment < selector.segment_count(); ++segment) {
            std::vector<FrWire> mask;
            if (!selector.read_segment(segment, mask, error))
                return fixture_fail("qpad-public-selector", "public selector segment read failed", error);
            for (const FrWire& value : mask) {
                const FrWire expected = offset < descriptor.logical_shape.front()
                    ? FrWire{} : zkdl::fr_from_u32(1U);
                if (value != expected)
                    return fixture_fail("qpad-public-selector", "row-major mask enumeration differs from the logical prefix", error);
                std::vector<FrWire> msb_point(log2_exact(kDomain), FrWire{});
                for (std::size_t bit = 0U; bit < msb_point.size(); ++bit) {
                    const std::size_t lsb = msb_point.size() - 1U - bit;
                    if (((offset >> lsb) & 1U) != 0U)
                        msb_point[bit] = zkdl::fr_from_u32(1U);
                }
                if (qpad_selector_mle_from_point(descriptor, msb_point) != expected)
                    return fixture_fail("qpad-public-selector", "verifier Boolean MLE disagrees with the enumerated row-major mask", error);
                ++offset;
            }
        }
        if (offset != kDomain)
            return fixture_fail("qpad-public-selector", "selector enumeration did not cover the padded domain", error);
    }
    fixture_pass("qpad-public-selector", "enumerated selector and verifier Boolean MLE agree over the complete padded domain");

    const auto make_witnesses = [](const std::vector<TensorDescriptor>& members,
                                   bool cancel_padding) {
        std::vector<TensorWitness> witnesses;
        witnesses.reserve(members.size());
        for (std::size_t member = 0U; member < members.size(); ++member) {
            TensorWitness tensor;
            tensor.id = members[member].id;
            tensor.visibility = members[member].visibility;
            tensor.domain_size = members[member].domain_size;
            tensor.values.reserve(tensor.domain_size);
            for (std::uint32_t index = 0U; index < tensor.domain_size; ++index) {
                if (index < members[member].logical_shape.front()) {
                    tensor.values.push_back(
                        zkdl::fr_from_u32(index + 1U + 10U * member));
                } else if (cancel_padding && index == tensor.domain_size - 1U &&
                           member < 2U) {
                    const FrWire value = zkdl::fr_from_u32(17U);
                    tensor.values.push_back(member == 0U
                        ? value : zkdl::fr_neg(value));
                } else {
                    tensor.values.push_back(FrWire{});
                }
            }
            witnesses.push_back(std::move(tensor));
        }
        return witnesses;
    };

    const auto build_set = [&](const std::vector<TensorDescriptor>& members,
                               const std::vector<TensorWitness>& witnesses,
                               RelationStatement& statement,
                               std::vector<CommitmentWire>& commitments,
                               std::map<std::string, const TensorWitness*>& tensor_map,
                               std::map<std::string, FrWire>& blinders,
                               std::map<std::string, std::vector<FrWire>>& segment_blinders,
                               std::map<std::string, SegmentCommitmentValidationReceipt>& receipts) {
        if (members.size() != witnesses.size()) {
            if (error) *error = "Qpad fixture witness/descriptor count mismatch";
            return false;
        }
        commitments.clear();
        tensor_map.clear();
        blinders.clear();
        segment_blinders.clear();
        receipts.clear();
        commitments.reserve(members.size());
        for (std::size_t index = 0U; index < members.size(); ++index) {
            const auto& descriptor = members[index];
            const auto& tensor = witnesses[index];
            if (!descriptor.valid(error) || !tensor.metadata_valid(error) ||
                tensor.id != descriptor.id || tensor.domain_size != descriptor.domain_size ||
                tensor.visibility != descriptor.visibility) {
                if (error && error->empty())
                    *error = "Qpad fixture witness does not match its descriptor";
                return false;
            }
            SegmentedTensorProvider provider;
            if (!tensor.segmented(owned_generators.size(), provider, error)) return false;
            const FrWire blinder = zkdl::fr_from_u32(
                static_cast<std::uint32_t>(193U + 2U * index));
            CommitmentWire wire;
            SegmentCommitmentMaterial material;
            SegmentCommitmentValidationReceipt receipt;
            if (!zktr_build_segment_commitment_material(
                    tensor.id, tensor.visibility, provider, owned_generators, h, u,
                    blinder, statement, wire, material, error, nullptr, 0U, nullptr,
                    &basis, &msm_session, &receipt) ||
                !receipt.valid()) return false;
            tensor_map.emplace(tensor.id, &tensor);
            blinders.emplace(tensor.id, blinder);
            segment_blinders.emplace(tensor.id, material.blinders);
            receipts.emplace(tensor.id, std::move(receipt));
            commitments.push_back(std::move(wire));
        }
        statement.segment_commitments_digest =
            relation_commitment_vector_digest(statement, commitments);
        return relation_segment_commitments_valid(statement, commitments, error);
    };

    const auto make_prover_context = [&](const std::vector<TensorDescriptor>& members,
                                         const std::map<std::string, FrWire>& blinders) {
        ProverContext context;
        context.tensor_descriptors = members;
        context.ipa_generators = basis;
        context.ipa_blinding_generator = h;
        context.ipa_u_generator = u;
        context.blinding_scalars = blinders;
        return context;
    };
    const auto make_verifier_context = [&](const std::vector<TensorDescriptor>& members) {
        VerifierContext context;
        context.tensor_descriptors = members;
        context.ipa_generators = basis;
        context.ipa_blinding_generator = h;
        context.ipa_u_generator = u;
        return context;
    };

    auto valid_witnesses = make_witnesses(descriptors, false);
    RelationStatement statement = base_statement;
    std::vector<CommitmentWire> commitments;
    std::map<std::string, const TensorWitness*> tensor_map;
    std::map<std::string, FrWire> blinders;
    std::map<std::string, std::vector<FrWire>> segment_blinders;
    std::map<std::string, SegmentCommitmentValidationReceipt> receipts;
    if (!build_set(descriptors, valid_witnesses, statement, commitments,
                   tensor_map, blinders, segment_blinders, receipts))
        return fixture_fail("qpad-original-commitments", "original input commitments or receipts failed", error);

    std::vector<QpadGroupPlan> groups;
    if (!qpad_build_group_plans(
            descriptors, owned_generators.size(), &statement, &owned_generators,
            &h, &u, &basis, groups, error) ||
        groups.size() != 2U ||
        !std::any_of(groups.begin(), groups.end(), [](const QpadGroupPlan& group) {
            return group.members.size() == 2U &&
                   group.members[0]->id == "fixture/qpad-rlc/a" &&
                   group.members[1]->id == "fixture/qpad-rlc/b";
        }))
        return fixture_fail("qpad-group-plan", "same-mask tensors did not group or distinct mask did not split", error);
    const std::uint64_t family_count = descriptors.size();
    std::uint64_t opening_segments = 0U;
    for (const auto& group : groups) opening_segments += group.segment_count;
    if (family_count != 3U || groups.size() != 2U || opening_segments != 4U ||
        std::any_of(groups.begin(), groups.end(), [](const QpadGroupPlan& group) {
            return group.segment_count != 2U;
        }))
        return fixture_fail("qpad-group-count", "Qpad fixture did not create two groups over two source segments", error);

    ProverContext prover_context = make_prover_context(descriptors, blinders);
    const std::string transcript_domain = "fixture.qpad.same-mask-rlc.v2";
    zkdl::Transcript prover_transcript(transcript_domain);
    absorb_statement(prover_transcript, statement, commitments);
    std::vector<PaddingProofRecord> records;
    if (!prove_padding_records(
            prover_context, statement, tensor_map, commitments, prover_transcript,
            records, error, &segment_blinders, &msm_session, &receipts,
            groups.size()) || records.size() != groups.size())
        return fixture_fail("qpad-group-prove", "same-mask aggregate proof generation failed", error);
    std::uint64_t record_opening_segments = 0U;
    for (const auto& record : records) {
        if (record.member_tensor_ids.empty() ||
            record.relation.terminal_relation.terminal_segment_openings.size() != 1U)
            return fixture_fail("qpad-group-opening-count", "a Qpad group did not emit exactly one terminal tensor opening", error);
        record_opening_segments += record.relation.terminal_relation
            .terminal_segment_openings.front().size();
    }
    if (record_opening_segments != groups.size() ||
        record_opening_segments >= family_count)
        return fixture_fail("qpad-group-opening-count", "multi-segment groups did not reduce terminal openings to one each", error);

    Proof proof;
    proof.statement = statement;
    proof.commitments = commitments;
    for (const auto& record : records) {
        const std::vector<byte> encoded = record.serialize();
        PaddingProofRecord decoded;
        if (!PaddingProofRecord::deserialize(encoded, decoded, error) ||
            decoded.serialize() != encoded)
            return fixture_fail("qpad-record-wire", "PAD2 group record did not survive its wire roundtrip", error);
        proof.padding_records.push_back(std::move(decoded));
    }
    if (proof.padding_records.size() != records.size())
        return fixture_fail("qpad-record-wire", "serialized group records were not retained");

    VerifierContext verifier_context = make_verifier_context(descriptors);
    zkdl::Transcript verifier_transcript(transcript_domain);
    absorb_statement(verifier_transcript, statement, commitments);
    if (!verify_padding_records(
            verifier_context, proof, verifier_transcript, error, &msm_session))
        return fixture_fail("qpad-serialized-independent-verify", "fresh verifier rejected serialized PAD2 group records", error);
    fixture_pass("qpad-serialized-independent-verify", "fresh verifier accepted serialized aggregate records against all original commitment wires");

    const auto rejects_record = [&](const PaddingProofRecord& candidate,
                                    const std::vector<CommitmentWire>& candidate_commitments,
                                    const char* stage) {
        Proof altered = proof;
        altered.commitments = candidate_commitments;
        altered.padding_records[0] = candidate;
        zkdl::Transcript transcript(transcript_domain);
        absorb_statement(transcript, altered.statement, altered.commitments);
        std::string tamper_error;
        if (verify_padding_records(
                verifier_context, altered, transcript, &tamper_error, &msm_session))
            return fixture_fail(stage, "tampered Qpad group record was accepted", &tamper_error);
        return true;
    };
    PaddingProofRecord changed = proof.padding_records[0];
    std::swap(changed.member_tensor_ids[0], changed.member_tensor_ids[1]);
    std::swap(changed.member_commitment_indices[0], changed.member_commitment_indices[1]);
    if (!rejects_record(changed, commitments, "qpad-member-order")) return false;
    changed = proof.padding_records[0];
    changed.member_tensor_ids[1] = changed.member_tensor_ids[0];
    if (!rejects_record(changed, commitments, "qpad-member-duplicate")) return false;
    changed = proof.padding_records[0];
    changed.member_tensor_ids.pop_back();
    changed.member_commitment_indices.pop_back();
    if (!rejects_record(changed, commitments, "qpad-member-omission")) return false;
    changed = proof.padding_records[0];
    changed.group_key[0U] ^= 1U;
    if (!rejects_record(changed, commitments, "qpad-group-key-tamper")) return false;
    changed = proof.padding_records[0];
    auto opening = zkdl::cpu_g1_from_native(
        changed.relation.terminal_relation.terminal_segment_openings[0][0].xi);
    auto changed_xi = zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(opening, h));
    changed.relation.terminal_relation.terminal_segment_openings[0][0].xi = changed_xi;
    changed.relation.terminal_relation.terminal_openings[0].xi = changed_xi;
    if (!rejects_record(changed, commitments, "qpad-terminal-xi-tamper")) return false;
    auto changed_commitments = commitments;
    changed_commitments[0].segment_commitment_digest[0U] ^= 1U;
    if (!rejects_record(proof.padding_records[0], changed_commitments,
                        "qpad-original-segment-binding")) return false;

    std::vector<byte> old_record = proof.padding_records[0].serialize();
    old_record[4U] = 1U;
    old_record[5U] = old_record[6U] = old_record[7U] = 0U;
    PaddingProofRecord old_decoded;
    std::string old_error;
    if (PaddingProofRecord::deserialize(old_record, old_decoded, &old_error))
        return fixture_fail("qpad-old-record-version", "PAD1 record version was accepted", error);
    const std::vector<byte> old_outer{'G', 'A', 'T', 'R', 7U, 0U, 0U, 0U};
    Proof old_proof;
    old_error.clear();
    if (Proof::deserialize(old_outer, old_proof, &old_error))
        return fixture_fail("qpad-old-proof-version", "Proof7 was accepted by the Proof8 parser", error);
    fixture_pass("qpad-wire-tamper", "member order/omission/duplication, group key, Xi, original wire, PAD1, and Proof7 are rejected");

    const std::vector<TensorDescriptor> cancel_descriptors{
        descriptor_a, descriptor_b};
    auto invalid_witnesses = make_witnesses(cancel_descriptors, true);
    RelationStatement invalid_statement = base_statement;
    std::vector<CommitmentWire> invalid_commitments;
    std::map<std::string, const TensorWitness*> invalid_tensor_map;
    std::map<std::string, FrWire> invalid_blinders;
    std::map<std::string, std::vector<FrWire>> invalid_segment_blinders;
    std::map<std::string, SegmentCommitmentValidationReceipt> invalid_receipts;
    if (!build_set(cancel_descriptors, invalid_witnesses, invalid_statement,
                   invalid_commitments, invalid_tensor_map, invalid_blinders,
                   invalid_segment_blinders, invalid_receipts))
        return fixture_fail("qpad-canceling-invalid", "invalid padded witness commitments failed", error);
    std::vector<QpadGroupPlan> invalid_groups;
    if (!qpad_build_group_plans(
            cancel_descriptors, owned_generators.size(), &invalid_statement,
            &owned_generators, &h, &u, &basis, invalid_groups, error) ||
        invalid_groups.size() != 1U || invalid_groups.front().members.size() != 2U)
        return fixture_fail("qpad-canceling-invalid", "invalid fixture did not form a two-member group", error);
    bool distinct_alpha_domain_found = false;
    std::string invalid_transcript_domain;
    for (std::uint32_t attempt = 0U; attempt < 16U; ++attempt) {
        const std::string candidate_domain =
            "fixture.qpad.canceling-invalid.v2." + std::to_string(attempt);
        zkdl::Transcript candidate_transcript(candidate_domain);
        absorb_statement(candidate_transcript, invalid_statement, invalid_commitments);
        std::vector<std::uint32_t> indices;
        std::vector<FrWire> alphas;
        if (!qpad_group_randomizers(
                invalid_groups.front(), invalid_commitments, invalid_statement,
                candidate_transcript,
                qpad_group_operation_context(invalid_groups.front()),
                indices, alphas, error) || alphas.size() != 2U)
            return fixture_fail("qpad-canceling-invalid", "cannot derive the cancellation test coefficients", error);
        if (alphas[0] != alphas[1]) {
            invalid_transcript_domain = candidate_domain;
            distinct_alpha_domain_found = true;
            break;
        }
    }
    if (!distinct_alpha_domain_found)
        return fixture_fail("qpad-canceling-invalid", "test could not obtain distinct Fiat-Shamir coefficients", error);
    ProverContext invalid_prover = make_prover_context(
        cancel_descriptors, invalid_blinders);
    zkdl::Transcript invalid_prover_transcript(invalid_transcript_domain);
    absorb_statement(invalid_prover_transcript, invalid_statement,
                     invalid_commitments);
    std::vector<PaddingProofRecord> invalid_records;
    std::string invalid_error;
    const bool invalid_proved = prove_padding_records(
        invalid_prover, invalid_statement, invalid_tensor_map,
        invalid_commitments, invalid_prover_transcript, invalid_records,
        &invalid_error, &invalid_segment_blinders, &msm_session,
        &invalid_receipts, invalid_groups.size());
    if (invalid_proved) {
        Proof invalid_proof;
        invalid_proof.statement = invalid_statement;
        invalid_proof.commitments = invalid_commitments;
        invalid_proof.padding_records = std::move(invalid_records);
        VerifierContext invalid_verifier = make_verifier_context(cancel_descriptors);
        zkdl::Transcript invalid_verifier_transcript(invalid_transcript_domain);
        absorb_statement(invalid_verifier_transcript, invalid_statement,
                         invalid_commitments);
        std::string verify_error;
        if (verify_padding_records(
                invalid_verifier, invalid_proof, invalid_verifier_transcript,
                &verify_error, &msm_session))
            return fixture_fail("qpad-canceling-invalid", "+/- padding violations canceled under aggregation and were accepted", &verify_error);
    }
    fixture_pass("qpad-canceling-invalid", "opposite nonzero padding values cannot cancel under distinct nonzero Fiat-Shamir coefficients");
    return true;
}

bool prepared_msm_receipt_fixture(
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
    const SegmentedTensorProvider& provider,
    const RelationStatement& statement, std::string* error) {
    std::vector<zkdl::CpuG1Point> authenticated_points = generators;
    zkdl::CpuIpaBasisContext authenticated_basis;
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(authenticated_points), h, u, authenticated_basis, error))
        return fixture_fail("prepared-msm-basis", "small authenticated basis rejected", error);
    const auto& owned_generators = authenticated_basis.generators();
    zkdl::CpuG1MsmSession session(authenticated_basis);

    CommitmentWire cached_wire;
    SegmentCommitmentMaterial material;
    SegmentCommitmentValidationReceipt receipt;
    if (!zktr_build_segment_commitment_material(
            "fixture/cached-public", 1U, provider, owned_generators, h, u,
            FrWire{}, statement, cached_wire, material, error, nullptr, 0U,
            nullptr, &authenticated_basis, &session, &receipt) ||
        !receipt.valid())
        return fixture_fail("prepared-msm-receipt", "cached commitment/receipt build failed", error);
    CommitmentWire raw_wire;
    if (!zktr_commitment_wire_from_material(
            "fixture/cached-public", 1U, provider, owned_generators, h, u,
            statement, material, raw_wire, error, 0U, nullptr,
            &authenticated_basis, nullptr, nullptr))
        return fixture_fail("prepared-msm-equality", "raw replay failed", error);
    std::vector<byte> cached_encoding, raw_encoding;
    append_commitment(cached_encoding, cached_wire);
    append_commitment(raw_encoding, raw_wire);
    if (cached_encoding != raw_encoding ||
        !receipt.matches_context(
            cached_wire.tensor_id, cached_wire.visibility, provider,
            owned_generators, h, u, &authenticated_basis, cached_wire,
            material, error))
        return fixture_fail("prepared-msm-equality", "cached and raw commitment bytes differ", error);
    std::vector<FrWire> segment_values;
    if (!provider.read_segment(0U, segment_values, error) ||
        !receipt.matches_segment(0U, segment_values, material, error))
        return fixture_fail("prepared-msm-receipt", "unchanged segment did not match receipt", error);
    std::vector<FrWire> changed_values = segment_values;
    changed_values[0U] = zkdl::fr_add(changed_values[0U], zkdl::fr_from_u32(1U));
    std::string mutation_error;
    if (receipt.matches_segment(0U, changed_values, material, &mutation_error))
        return fixture_fail("prepared-msm-mutation", "mutated scalar was accepted by receipt", error);
    SegmentCommitmentMaterial changed_material = material;
    changed_material.blinders[0U] = zkdl::fr_from_u32(1U);
    mutation_error.clear();
    if (receipt.matches_context(
            cached_wire.tensor_id, cached_wire.visibility, provider,
            owned_generators, h, u, &authenticated_basis, cached_wire,
            changed_material, &mutation_error))
        return fixture_fail("prepared-msm-mutation", "mutated material was accepted by receipt", error);
    fixture_pass("prepared-msm-receipt", "raw bytes match; scalar and material mutations are rejected");

    return prepared_msm_chunk_fixture(h, u, error);
}

bool prepared_msm_chunk_fixture(
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
    std::string* error) {
    constexpr std::size_t count =
        4U * zkdl::kCpuG1MsmChunkPoints + 17U;
    const auto first_point = zkdl::cpu_g1_mul(
        zkdl::cpu_g1_generator(), zkdl::fr_from_u32(211U));
    const auto second_point = zkdl::cpu_g1_mul(
        zkdl::cpu_g1_generator(), zkdl::fr_from_u32(223U));
    std::vector<zkdl::CpuG1Point> large_points(count, first_point);
    for (std::size_t index = 4U * zkdl::kCpuG1MsmChunkPoints;
         index < count; ++index)
        large_points[index] = second_point;
    zkdl::CpuIpaBasisContext large_basis;
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(large_points), h, u, large_basis, error))
        return fixture_fail("prepared-msm-chunks", "chunked authenticated basis rejected", error);
    const auto& large_generators = large_basis.generators();
    std::vector<FrWire> scalars(count);
    for (std::size_t index = 0U; index < count; ++index) {
        if (index % 19U == 0U) scalars[index] = FrWire{};
        else if (index % 19U == 1U) scalars[index] = zkdl::fr_from_u32(1U);
        else if (index % 19U == 2U) scalars[index] = zkdl::fr_neg(zkdl::fr_from_u32(1U));
        else scalars[index] = zkdl::fr_from_u32(static_cast<std::uint32_t>(index % 97U + 1U));
    }
    const auto prepare_start = std::chrono::steady_clock::now();
    zkdl::CpuG1MsmSession large_session(large_basis);
    const double prepare_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prepare_start).count();
    const auto raw_start = std::chrono::steady_clock::now();
    const zkdl::CpuG1Point raw = zkdl::cpu_g1_sum(
        large_generators.data(), scalars.data(), count);
    const double raw_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - raw_start).count();
    const auto cached_start = std::chrono::steady_clock::now();
    const zkdl::CpuG1Point cached = large_session.sum(
        large_basis, large_generators, h, u, scalars.data(), count);
    const double cached_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - cached_start).count();
    if (!zkdl::cpu_g1_equal(raw, cached))
        return fixture_fail("prepared-msm-chunks", "raw and prepared MSM outputs differ", error);
    const std::size_t active_workspace_lanes = large_session.workspace_count();
    if (active_workspace_lanes < 5U)
        return fixture_fail("prepared-msm-chunks", "prepared MSM did not allocate five chunk workspaces", error);
    const std::size_t pool_workers = zkdl::cpu_bounded_thread_pool().size();
    if (pool_workers < 4U)
        return fixture_fail("prepared-msm-chunks", "bounded pool did not expose four background workers", error);
    std::clog << "segmented_fixture stage=prepared-msm-timing status=PASS basis_points="
              << count << " pippenger_chunks="
              << (count + zkdl::kCpuG1MsmChunkPoints - 1U) /
                     zkdl::kCpuG1MsmChunkPoints
              << " active_workspace_lanes=" << active_workspace_lanes
              << " pool_workers=" << pool_workers
              << " raw_seconds=" << raw_seconds
              << " prepared_build_seconds=" << prepare_seconds
              << " prepared_basis_bytes=" << large_session.prepared_basis_bytes()
              << " workspace_bytes=" << large_session.workspace_bytes()
              << " prepared_sum_seconds=" << cached_seconds << '\n';
    fixture_pass("prepared-msm-chunks", "five-workspace prepared MSM matches serial raw output");
    return true;
}

bool relation_artifact_roundtrip(const std::vector<zkdl::CpuG1Point> &generators, const zkdl::CpuG1Point &h, const zkdl::CpuG1Point &u,
                                 const std::filesystem::path &artifact_path, std::string *error) {
    const auto scalar = [](std::uint32_t value) { return zkdl::fr_from_u32(value); };
    const std::vector<FrWire> public_values{scalar(1U), scalar(2U), scalar(3U), scalar(4U), scalar(5U), scalar(6U), scalar(7U), scalar(8U)};
    OperationDescriptor operation;
    operation.id = "fixture/output";
    operation.kind = RelationKind::PublicOutput;
    operation.degree = 1U;
    operation.source_refs = {"public_Y"};
    operation.tensor_refs = {"public_Y"};
    operation.mapping = "fixture-public-output";
    operation.padding = "none";
    operation.terminal_claim = "public_Y";
    operation.terminal_point = "public_Y";
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.public_output = encode_public_field_vector(public_values);

    ProverContext prover;
    prover.statement = statement;
    prover.tensors.push_back(TensorWitness{"public_Y", 1U, 8U, public_values});
    prover.operations = {operation};
    prover.generators = generators;
    prover.ipa_blinding_generator = h;
    prover.ipa_u_generator = u;
    std::vector<zkdl::CpuG1Point> prover_ipa_generators = generators;
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(prover_ipa_generators), h, u, prover.ipa_generators,
            error))
        return fixture_fail("prover-ipa-basis", "cannot authenticate prover IPA basis", error);
    Proof proof;
    std::cerr << "segmented_fixture stage=prove status=START reason=CPU native tiny relation\n";
    if (!prove(prover, proof, error)) return fixture_fail("prove", "CPU native tiny relation rejected", error);
    fixture_pass("prove", "CPU native tiny relation produced a proof");
    const std::vector<byte> artifact = proof.serialize();
    std::ofstream output(artifact_path, std::ios::binary);
    if (!output) {
        if (error) *error = "fixture cannot write proof artifact";
        return fixture_fail("artifact-roundtrip", "cannot write proof artifact", error);
    }
    output.write(reinterpret_cast<const char *>(artifact.data()), static_cast<std::streamsize>(artifact.size()));
    output.close();

    Proof decoded;
    if (!Proof::deserialize(artifact, decoded, error) || decoded.serialize() != artifact) {
        if (error && error->empty()) *error = "fixture proof artifact is not canonical";
        return fixture_fail("artifact-roundtrip", "proof artifact failed canonical deserialize/serialize", error);
    }
    fixture_pass("artifact-roundtrip", "proof artifact is canonical after deserialize/serialize");
    VerifierContext verifier;
    verifier.statement = decoded.statement;
    verifier.operations = {operation};
    verifier.tensor_ids = {"public_Y"};
    verifier.tensor_visibility.emplace("public_Y", 1U);
    verifier.generators = generators;
    verifier.ipa_blinding_generator = h;
    verifier.ipa_u_generator = u;
    std::vector<zkdl::CpuG1Point> verifier_ipa_generators = generators;
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(verifier_ipa_generators), h, u, verifier.ipa_generators,
            error))
        return fixture_fail("verifier-ipa-basis", "cannot authenticate verifier IPA basis", error);
    std::cerr << "segmented_fixture stage=independent-verify status=START reason=fresh verifier context\n";
    if (!verify(decoded, verifier, error)) return fixture_fail("independent-verify", "fresh verifier rejected valid artifact", error);
    fixture_pass("independent-verify", "fresh verifier accepted serialized artifact");

    Proof swapped = decoded;
    const auto original_xi = zkdl::cpu_g1_from_native(swapped.commitments[0].segment_commitments[0].commitment);
    swapped.commitments[0].segment_commitments[0].commitment = zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(original_xi, h));
    swapped.commitments[0].segment_commitment_digest = segment_commitment_digest(swapped.commitments[0]);
    std::string tamper_error;
    if (verify(swapped, verifier, &tamper_error)) {
        if (error) *error = "fixture verifier accepted a swapped commitment tamper";
        return fixture_fail("tamper", "single-segment Xi mutation was accepted", error);
    }
    fixture_pass("tamper-swap", "single-segment Xi mutation rejected; four-segment swap covered");
    Proof old_version = decoded;
    old_version.commitments[0].segment_binding_version = 4U;
    tamper_error.clear();
    if (verify(old_version, verifier, &tamper_error)) {
        if (error) *error = "fixture verifier accepted a version tamper";
        return fixture_fail("tamper-version", "version downgrade was accepted", error);
    }
    fixture_pass("tamper-version", "version downgrade rejected");

    ProverContext gpu_prover = prover;
    gpu_prover.gpu_backend = true;
    Proof ignored;
    tamper_error.clear();
    if (prove(gpu_prover, ignored, &tamper_error)) {
        if (error) *error = "fixture prover did not fail closed for GPU backend";
        return fixture_fail("cpu-only", "GPU prover did not fail closed", error);
    }
    VerifierContext gpu_verifier = verifier;
    gpu_verifier.gpu_backend = true;
    tamper_error.clear();
    if (verify(decoded, gpu_verifier, &tamper_error)) {
        if (error) *error = "fixture verifier did not fail closed for GPU backend";
        return fixture_fail("cpu-only", "GPU verifier did not fail closed", error);
    }
    fixture_pass("cpu-only", "GPU prover/verifier fail closed");
    return true;
}

bool small_shared_segment_terminal_fixture() {
    constexpr std::uint32_t kDomain = 16U;
    constexpr std::uint32_t kSharedSegment = 8U;
    constexpr std::uint32_t kBasis = 8U;
    std::vector<zkdl::CpuG1Point> generators;
    generators.reserve(kBasis);
    for (std::uint32_t index = 0U; index < kBasis; ++index)
        generators.push_back(zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(),
                                               zkdl::fr_from_u32(index + 2U)));
    const auto h = zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(101U));
    const auto u = zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(103U));

    std::vector<FrWire> values;
    values.reserve(kDomain);
    for (std::uint32_t index = 0U; index < kDomain; ++index)
        values.push_back(zkdl::fr_from_u32(index + 1U));
    SegmentedTensorProvider provider(
        kDomain, kSharedSegment,
        [&values](std::uint64_t offset, std::uint32_t count,
                  std::vector<FrWire> &output, std::string *read_error) {
            if (offset > values.size() || count > values.size() - offset) {
                if (read_error) *read_error = "fixture range is outside the tensor";
                return false;
            }
            output.assign(values.begin() + static_cast<std::ptrdiff_t>(offset),
                          values.begin() + static_cast<std::ptrdiff_t>(offset + count));
            return true;
        });
    if (provider.logical_domain() != kDomain || provider.segment_size() != kSharedSegment ||
        provider.segment_count() != 2U)
        return fixture_fail("shared-small-layout", "small provider does not match the declared two-segment layout");
    ZkRelationSegmentLayout layout;
    std::string error;
    if (!zktr_segment_layout_for_size(kDomain, generators.size(), kSharedSegment, layout, &error) ||
        layout.segment_count != 2U || layout.segment_size != kSharedSegment)
        return fixture_fail("shared-small-layout", "canonical shared-S layout was rejected", &error);
    if (zktr_segment_layout_for_size(kDomain, generators.size(), 3U, layout, &error))
        return fixture_fail("shared-small-negative", "non-power-of-two shared segment was accepted", &error);
    fixture_pass("shared-small-layout", "logical domain 16 uses canonical shared segment size 8 across two segments");

    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    const FrWire tensor_blinder = zkdl::fr_from_u32(193U);
    std::vector<zkdl::CpuG1Point> authenticated_points = generators;
    zkdl::CpuIpaBasisContext authenticated_basis;
    if (!zkdl::CpuIpaBasisContext::authenticate(
            std::move(authenticated_points), h, u, authenticated_basis, &error))
        return fixture_fail("shared-small-basis", "authenticated shared-S basis was rejected", &error);
    const auto& owned_generators = authenticated_basis.generators();
    zkdl::CpuG1MsmSession msm_session(authenticated_basis);
    CommitmentWire wire;
    SegmentCommitmentMaterial material;
    SegmentCommitmentValidationReceipt validation_receipt;
    if (!zktr_build_segment_commitment_material(
            "fixture/small-shared", 3U, provider, owned_generators, h, u,
            tensor_blinder, statement, wire, material, &error, nullptr,
            kSharedSegment, nullptr, &authenticated_basis, &msm_session,
            &validation_receipt) || !validation_receipt.valid() ||
        wire.segment_size != kSharedSegment || wire.segment_count != 2U ||
        wire.segment_commitments.size() != 2U)
        return fixture_fail("shared-small-commitment", "small shared-S commitment failed", &error);
    const std::vector<CommitmentWire> commitment_wires{wire};
    statement.segment_commitments_digest =
        relation_commitment_vector_digest(statement, commitment_wires);

    SegmentedTensorProvider natural_provider;
    if (!SegmentedTensorProvider::from_values(values, generators.size(), natural_provider, &error))
        return fixture_fail("shared-small-negative", "natural provider setup failed", &error);
    CommitmentWire rejected_wire;
    SegmentCommitmentMaterial rejected_material;
    error.clear();
    if (zktr_build_segment_commitment_material(
            "fixture/small-shared-mismatch", 3U, natural_provider, generators,
            h, u, tensor_blinder, statement, rejected_wire, rejected_material,
            &error, nullptr, kSharedSegment / 2U) ||
        error.find("authenticated shared layout") == std::string::npos)
        return fixture_fail("shared-small-negative", "natural provider was not rejected for the shared-S layout", &error);
    fixture_pass("shared-small-commitment", "both segments committed using the declared common S");

    ConstraintDescriptor descriptor;
    descriptor.id = "fixture/small-shared-terminal";
    descriptor.degree = 1U;
    descriptor.terms.push_back(Monomial{zkdl::fr_from_u32(1U), {0U}});
    std::vector<FrWire> point{zkdl::fr_from_u32(2U), zkdl::fr_from_u32(3U),
                              zkdl::fr_from_u32(5U), zkdl::fr_from_u32(7U)};
    FrWire expected_value{};
    if (!provider.evaluate_mle(point, expected_value, &error))
        return fixture_fail("shared-small-terminal", "small shared-S MLE evaluation failed", &error);
    const FrWire external_blinder = zkdl::fr_from_u32(197U);
    const auto external_commitment = zkdl::cpu_pedersen_scalar_commit(
        expected_value, external_blinder, h, u);
    const std::vector<zkdl::CpuG1Point> canonical_commitments{
        zkdl::cpu_g1_from_native(wire.commitment)};
    const std::vector<FrWire> tensor_blinders{tensor_blinder};
    const std::vector<std::uint32_t> commitment_indices{0U};
    const std::vector<SegmentedTensorProvider> terminal_providers{provider};
    const std::vector<SegmentCommitmentMaterial> terminal_materials{material};
    const std::vector<const SegmentCommitmentValidationReceipt*> terminal_receipts{
        &validation_receipt};
    const std::string context = "fixture.small-shared-terminal";
    zkdl::Transcript prover_transcript(context);
    ZkTerminalRelationProverResult terminal;
    if (!prove_zk_terminal_relation_segmented(
            descriptor, terminal_providers, canonical_commitments, tensor_blinders,
            commitment_indices, point, owned_generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            external_blinder, FrWire{}, context, prover_transcript, terminal,
            &error, &terminal_materials, kSharedSegment, nullptr,
            &authenticated_basis, &msm_session, &terminal_receipts,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate))
        return fixture_fail("shared-small-prove", "small shared-S terminal proof failed", &error);
    if (terminal.proof.terminal_segment_openings.size() != 1U ||
        terminal.proof.version != zkdl::ZkTerminalRelationProof::kAggregateVersion ||
        terminal.proof.terminal_segment_openings.front().size() != 1U ||
        terminal.proof.terminal_segment_openings.front().front().logical_domain != kDomain ||
        terminal.proof.terminal_segment_openings.front().front().logical_offset != 0U ||
        terminal.proof.terminal_segment_openings.front().front().actual_length != kSharedSegment ||
        terminal.proof.terminal_segment_openings.front().front().padded_length != kSharedSegment ||
        terminal.proof.terminal_segment_openings.front().front().generator_size != kSharedSegment ||
        terminal.proof.terminal_segment_openings.front().front().padded_generator_size !=
            zkdl::cpu_ipa_padded_size(kSharedSegment))
        return fixture_fail("shared-small-prove", "terminal proof did not aggregate both source segments into one virtual opening");
    std::vector<zkdl::byte> encoded;
    zkdl::serialize_zk_terminal_relation(terminal.proof, encoded);
    zkdl::ZkTerminalRelationProof fresh;
    if (!zkdl::deserialize_zk_terminal_relation(encoded, fresh, &error))
        return fixture_fail("shared-small-wire", "canonical shared-S proof wire roundtrip failed", &error);
    zkdl::Transcript verifier_transcript(context);
    ZkTerminalRelationVerifierResult verified;
    if (!verify_zk_terminal_relation(
            fresh, descriptor, canonical_commitments, commitment_indices, point,
            generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, verifier_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-independent-verify", "fresh shared-S verifier rejected valid wire", &error);
    fixture_pass("shared-small-independent-verify", "canonical shared-S terminal proof survived fresh serialization and independent verification");

    zkdl::ZkTerminalRelationProof tampered = fresh;
    const auto xi = zkdl::cpu_g1_from_native(
        tampered.terminal_segment_openings[0][0].xi);
    const auto changed_xi = zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(xi, h));
    tampered.terminal_segment_openings[0][0].xi = changed_xi;
    tampered.terminal_openings[0].xi = changed_xi;
    zkdl::Transcript tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            tampered, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "tampered shared-S segment opening was accepted");
    fixture_pass("shared-small-tamper", "mutated first-segment Hyrax commitment rejected");

    tampered = fresh;
    const auto aggregate_commitment = zkdl::cpu_g1_from_native(
        tampered.terminal_commitments[0]);
    tampered.terminal_commitments[0] = zkdl::cpu_g1_to_native(
        zkdl::cpu_g1_add(aggregate_commitment, h));
    zkdl::Transcript aggregate_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            tampered, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, aggregate_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "tampered aggregate terminal commitment was accepted");
    fixture_pass("shared-small-tamper", "aggregate terminal commitment must match all weighted segment taus");

    tampered = fresh;
    const auto aggregate_tau = zkdl::cpu_g1_from_native(
        tampered.terminal_segment_openings[0][0].tau);
    const auto changed_tau = zkdl::cpu_g1_to_native(
        zkdl::cpu_g1_add(aggregate_tau, h));
    tampered.terminal_segment_openings[0][0].tau = changed_tau;
    tampered.terminal_openings[0].tau = changed_tau;
    tampered.terminal_commitments[0] = changed_tau;
    zkdl::Transcript segment_tau_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            tampered, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, segment_tau_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "tampered aggregate tau opening was accepted");
    fixture_pass("shared-small-tamper", "mutated aggregate tau rejected by fresh verifier");

    tampered = fresh;
    tampered.version = zkdl::ZkTerminalRelationProof::kLegacyVersion;
    zkdl::Transcript version_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            tampered, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, version_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "aggregate proof with legacy version was accepted");
    fixture_pass("shared-small-tamper", "nested version is bound to the expected aggregate mode");

    zkdl::Transcript mode_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            fresh, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, mode_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::LegacyPerSegment, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "aggregate proof was accepted when caller required legacy mode");
    fixture_pass("shared-small-tamper", "proof cannot choose the verifier opening mode");

    tampered = fresh;
    tampered.terminal_openings[0].logical_domain = 32U;
    tampered.terminal_segment_openings[0][0].logical_domain = 32U;
    zkdl::Transcript domain_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            tampered, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, domain_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "virtual opening with changed source logical domain was accepted");
    fixture_pass("shared-small-tamper", "virtual logical domain remains tied to trusted source layout");

    std::vector<FrWire> changed_point = point;
    changed_point.front() = zkdl::fr_add(changed_point.front(), zkdl::fr_from_u32(1U));
    zkdl::Transcript weight_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            fresh, descriptor, canonical_commitments, commitment_indices,
            changed_point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, weight_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "aggregate opening reused with changed high/low evaluation point was accepted");
    fixture_pass("shared-small-tamper", "aggregate weights and opening point remain transcript-bound");

    std::vector<CommitmentWire> reordered_commitments = commitment_wires;
    std::swap(reordered_commitments[0].segment_commitments[0].commitment,
              reordered_commitments[0].segment_commitments[1].commitment);
    reordered_commitments[0].segment_commitment_digest =
        segment_commitment_digest(reordered_commitments[0]);
    RelationStatement reordered_statement = statement;
    reordered_statement.segment_commitments_digest =
        relation_commitment_vector_digest(reordered_statement, reordered_commitments);
    const std::vector<zkdl::CpuG1Point> reordered_canonical{
        zkdl::cpu_g1_from_native(reordered_commitments[0].commitment)};
    zkdl::Transcript order_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            fresh, descriptor, reordered_canonical, commitment_indices,
            point, generators, h, u, reordered_statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, order_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &reordered_commitments, kSharedSegment))
        return fixture_fail("shared-small-tamper", "aggregate opening accepted a reordered source Xi vector");
    fixture_pass("shared-small-tamper", "ordered original segment Xi vector is authenticated before aggregate verification");

    std::vector<FrWire> changed_values = values;
    changed_values[0U] = zkdl::fr_add(changed_values[0U], zkdl::fr_from_u32(1U));
    SegmentedTensorProvider changed_provider;
    CommitmentWire changed_wire;
    SegmentCommitmentMaterial changed_material;
    if (!SegmentedTensorProvider::from_values(
            changed_values, owned_generators.size(), changed_provider, &error) ||
        !zktr_build_segment_commitment_material(
            "fixture/small-shared", 3U, changed_provider, owned_generators,
            h, u, tensor_blinder, statement, changed_wire, changed_material,
            &error, nullptr, kSharedSegment, nullptr, &authenticated_basis,
            &msm_session))
        return fixture_fail("shared-small-tamper", "changed-source-value commitment setup failed", &error);
    std::vector<CommitmentWire> changed_value_wires{changed_wire};
    RelationStatement changed_value_statement = statement;
    changed_value_statement.segment_commitments_digest =
        relation_commitment_vector_digest(changed_value_statement, changed_value_wires);
    const std::vector<zkdl::CpuG1Point> changed_value_canonical{
        zkdl::cpu_g1_from_native(changed_wire.commitment)};
    zkdl::Transcript value_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            fresh, descriptor, changed_value_canonical, commitment_indices,
            point, generators, h, u, changed_value_statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, value_tamper_transcript, verified,
            zkdl::ZkTerminalOpeningMode::WeightedAggregate, &error,
            &changed_value_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "aggregate opening accepted a changed original segment value commitment");
    fixture_pass("shared-small-tamper", "changed source value commitment is rejected by the authenticated aggregate context");

    std::vector<FrWire> second_values(values.size());
    for (std::size_t index = 0U; index < second_values.size(); ++index)
        second_values[index] = zkdl::fr_from_u32(
            static_cast<std::uint32_t>(17U + 3U * index));
    SegmentedTensorProvider second_provider;
    if (!SegmentedTensorProvider::from_values(
            second_values, owned_generators.size(), second_provider, &error))
        return fixture_fail("cross-provider-setup", "second provider construction failed", &error);
    const FrWire second_blinder = zkdl::fr_from_u32(199U);
    RelationStatement cross_commitment_statement = statement;
    cross_commitment_statement.segment_commitments_digest = {};
    CommitmentWire second_wire;
    SegmentCommitmentMaterial second_material;
    SegmentCommitmentValidationReceipt second_receipt;
    if (!zktr_build_segment_commitment_material(
            "fixture/small-shared-second", 3U, second_provider,
            owned_generators, h, u, second_blinder, cross_commitment_statement,
            second_wire, second_material, &error, nullptr, kSharedSegment,
            nullptr, &authenticated_basis, &msm_session, &second_receipt) ||
        !second_receipt.valid())
        return fixture_fail("cross-provider-setup", "second provider commitment failed", &error);
    std::vector<CommitmentWire> cross_wires{wire, second_wire};
    RelationStatement cross_statement = statement;
    cross_statement.segment_commitments_digest =
        relation_commitment_vector_digest(cross_statement, cross_wires);
    ConstraintDescriptor cross_descriptor;
    cross_descriptor.id = "fixture/small-cross-provider-terminal";
    cross_descriptor.degree = 1U;
    cross_descriptor.terms = {
        Monomial{zkdl::fr_from_u32(1U), {0U}},
        Monomial{zkdl::fr_from_u32(1U), {1U}}};
    FrWire second_expected{};
    if (!second_provider.evaluate_mle(point, second_expected, &error))
        return fixture_fail("cross-provider-setup", "second provider evaluation failed", &error);
    const FrWire cross_expected = zkdl::fr_add(expected_value, second_expected);
    const FrWire cross_external_blinder = zkdl::fr_from_u32(211U);
    const auto cross_external_commitment = zkdl::cpu_pedersen_scalar_commit(
        cross_expected, cross_external_blinder, h, u);
    const std::vector<zkdl::CpuG1Point> cross_canonical{
        zkdl::cpu_g1_from_native(wire.commitment),
        zkdl::cpu_g1_from_native(second_wire.commitment)};
    const std::vector<FrWire> cross_blinders{tensor_blinder, second_blinder};
    const std::vector<std::uint32_t> cross_indices{0U, 1U};
    const std::vector<SegmentedTensorProvider> cross_providers{
        provider, second_provider};
    const std::vector<SegmentCommitmentMaterial> cross_materials{
        material, second_material};
    const std::vector<const SegmentCommitmentValidationReceipt*> cross_receipts{
        &validation_receipt, &second_receipt};
    const std::string cross_context = "fixture.small-cross-provider-terminal";
    zkdl::Transcript cross_prover_transcript(cross_context);
    ZkTerminalRelationProverResult cross_terminal;
    if (!prove_zk_terminal_relation_segmented(
            cross_descriptor, cross_providers, cross_canonical, cross_blinders,
            cross_indices, point, owned_generators, h, u, cross_statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            cross_external_commitment, cross_external_blinder, FrWire{},
            cross_context, cross_prover_transcript, cross_terminal, &error,
            &cross_materials, kSharedSegment, nullptr, &authenticated_basis,
            &msm_session, &cross_receipts,
            zkdl::ZkTerminalOpeningMode::CrossProviderAggregate))
        return fixture_fail("cross-provider-prove", "two-provider v4 opening failed", &error);
    if (cross_terminal.proof.version !=
            zkdl::ZkTerminalRelationProof::kCrossProviderAggregateVersion ||
        cross_terminal.proof.terminal_commitments.size() != 2U ||
        cross_terminal.proof.terminal_openings.size() != 1U ||
        cross_terminal.proof.terminal_segment_openings.size() != 1U ||
        cross_terminal.proof.terminal_segment_openings.front().size() != 1U ||
        cross_terminal.proof.terminal_segment_openings.front().front().logical_domain != kDomain ||
        cross_terminal.proof.terminal_segment_openings.front().front().generator_size !=
            kSharedSegment)
        return fixture_fail("cross-provider-prove", "v4 did not preserve N tau values and emit one opening");
    std::vector<zkdl::byte> cross_encoded;
    zkdl::serialize_zk_terminal_relation(cross_terminal.proof, cross_encoded);
    zkdl::ZkTerminalRelationProof cross_fresh;
    if (!zkdl::deserialize_zk_terminal_relation(cross_encoded, cross_fresh, &error))
        return fixture_fail("cross-provider-wire", "v4 wire roundtrip failed", &error);
    zkdl::Transcript cross_verifier_transcript(cross_context);
    ZkTerminalRelationVerifierResult cross_verified;
    if (!verify_zk_terminal_relation(
            cross_fresh, cross_descriptor, cross_canonical, cross_indices, point,
            owned_generators, h, u, cross_statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            cross_external_commitment, FrWire{}, cross_context,
            cross_verifier_transcript, cross_verified,
            zkdl::ZkTerminalOpeningMode::CrossProviderAggregate, &error,
            &cross_wires, kSharedSegment, &authenticated_basis, &msm_session))
        return fixture_fail("cross-provider-independent-verify",
                            "fresh verifier rejected two-provider v4 wire", &error);
    fixture_pass("cross-provider-independent-verify",
                 "v4 Prove/wire/fresh Verify combined two providers across two source segments into one opening");

    std::vector<FrWire> changed_second_values = second_values;
    changed_second_values[0U] = zkdl::fr_add(
        changed_second_values[0U], zkdl::fr_from_u32(1U));
    SegmentedTensorProvider changed_second_provider;
    if (!SegmentedTensorProvider::from_values(
            changed_second_values, owned_generators.size(),
            changed_second_provider, &error))
        return fixture_fail("cross-provider-value-tamper",
                            "changed second source provider setup failed", &error);
    const std::vector<SegmentedTensorProvider> changed_cross_providers{
        provider, changed_second_provider};
    zkdl::Transcript changed_source_transcript(cross_context);
    ZkTerminalRelationProverResult changed_source_result;
    error.clear();
    if (prove_zk_terminal_relation_segmented(
            cross_descriptor, changed_cross_providers, cross_canonical,
            cross_blinders, cross_indices, point, owned_generators, h, u,
            cross_statement, zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            cross_external_commitment, cross_external_blinder, FrWire{},
            cross_context, changed_source_transcript, changed_source_result,
            &error, &cross_materials, kSharedSegment, nullptr,
            &authenticated_basis, &msm_session, &cross_receipts,
            zkdl::ZkTerminalOpeningMode::CrossProviderAggregate))
        return fixture_fail("cross-provider-value-tamper",
                            "v4 prover accepted changed source values under original authenticated commitments");
    fixture_pass("cross-provider-value-tamper",
                 "v4 first/second pass rejects source-value changes against original commitments");
    return true;
}

bool basis8_logical32_four_segment_tamper_fixture() {
    std::vector<zkdl::CpuG1Point> generators;
    generators.reserve(8U);
    for (std::uint32_t index = 0U; index < 8U; ++index) generators.push_back(zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(index + 2U)));
    const auto h = zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(101U));
    const auto u = zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), zkdl::fr_from_u32(103U));
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});

    std::vector<FrWire> values;
    values.reserve(32U);
    for (std::uint32_t index = 0U; index < 32U; ++index) values.push_back(zkdl::fr_from_u32(index + 1U));
    SegmentedTensorProvider provider;
    std::string error;
    if (!SegmentedTensorProvider::from_values(values, generators.size(), provider, &error) || provider.logical_domain() != 32U ||
        provider.segment_size() != 8U || provider.segment_count() != 4U)
        return fixture_fail("range", "logical32/frame8 provider shape or admission failed", &error);
    if (!prepared_msm_receipt_fixture(generators, h, u, provider, statement,
                                      &error))
        return false;
    if (!::zkdl_run_tiny_sorted_lookup_receipt_fixture(generators, h, u))
        return fixture_fail("sorted-receipt", "tiny authenticated sorted lookup with receipts failed");
    fixture_pass("range", "cross-segment provider shape is logical32/frame8/4");
    std::vector<FrWire> first_segment, last_segment;
    if (!provider.read_segment(0U, first_segment, &error) || !provider.read_segment(3U, last_segment, &error) || first_segment.size() != 8U ||
        last_segment.size() != 8U || zkdl::fr_compare(first_segment.front(), zkdl::fr_from_u32(1U)) != 0 ||
        zkdl::fr_compare(last_segment.front(), zkdl::fr_from_u32(25U)) != 0)
        return fixture_fail("range", "cross-segment range read returned unexpected values", &error);
    fixture_pass("range", "segment0/segment3 range reads returned values 1/25");

    CommitmentWire wire;
    SegmentCommitmentMaterial material;
    if (!zktr_build_segment_commitment_material("fixture/lookup", 3U, provider, generators, h, u, zkdl::fr_from_u32(17U), statement, wire, material, &error) ||
        wire.segment_count != 4U || wire.segment_size != 8U)
        return fixture_fail("v5-digest", "ordered v5 segment material construction failed", &error);
    fixture_pass("v5-digest", "ordered Xi vector and digest constructed");

    CommitmentWire swapped = wire;
    std::swap(swapped.segment_commitments[0].commitment, swapped.segment_commitments[1].commitment);
    swapped.segment_commitment_digest = segment_commitment_digest(swapped);
    if (!segment_commitment_wire_shape_valid(swapped, &error) || zktr_segment_material_matches(swapped, material, &error))
        return fixture_fail("tamper-swap", "ordered segment swap was not rejected", &error);
    fixture_pass("tamper-swap", "ordered segment swap rejected");

    CommitmentWire cancelled = wire;
    const FrWire delta = zkdl::fr_from_u32(19U);
    const auto plus = zkdl::cpu_g1_add(zkdl::cpu_g1_from_native(cancelled.segment_commitments[0].commitment), zkdl::cpu_g1_mul(h, delta));
    const auto minus = zkdl::cpu_g1_add(zkdl::cpu_g1_from_native(cancelled.segment_commitments[1].commitment), zkdl::cpu_g1_mul(h, zkdl::fr_neg(delta)));
    cancelled.segment_commitments[0].commitment = zkdl::cpu_g1_to_native(plus);
    cancelled.segment_commitments[1].commitment = zkdl::cpu_g1_to_native(minus);
    cancelled.segment_commitment_digest = segment_commitment_digest(cancelled);
    if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(cancelled.commitment), zkdl::cpu_g1_from_native(wire.commitment)) ||
        zktr_segment_material_matches(cancelled, material, &error))
        return fixture_fail("tamper-delta", "+delta/-delta cancellation was not rejected", &error);
    fixture_pass("tamper-delta", "legacy sum cancellation does not pass ordered material");

    CommitmentWire padding = wire;
    padding.segment_commitments.back().actual_length = 7U;
    padding.segment_commitments.back().padding_length = 1U;
    padding.segment_commitment_digest = segment_commitment_digest(padding);
    if (segment_commitment_wire_shape_valid(padding, &error)) return fixture_fail("tamper-padding", "forged last-segment padding was accepted", &error);
    fixture_pass("tamper-padding", "forged last-segment padding rejected");

    CommitmentWire version = wire;
    version.segment_binding_version = 4U;
    version.segment_commitment_digest = segment_commitment_digest(version);
    if (segment_commitment_wire_shape_valid(version, &error)) return fixture_fail("tamper-version", "v4 segment binding was accepted", &error);
    fixture_pass("tamper-version", "v4 segment binding rejected");

    const std::vector<FrWire> zero_high_point{zkdl::fr_from_u32(0U), zkdl::fr_from_u32(1U)};
    const auto zero_weights = zkdl::cpu_ipa_mle_weights(zero_high_point);
    if (!std::any_of(zero_weights.begin(), zero_weights.end(), [](const FrWire &weight) { return zkdl::fr_compare(weight, FrWire{}) == 0; }))
        return fixture_fail("zero-weight", "fixture did not produce a zero high-point weight", &error);
    fixture_pass("zero-weight", "zero high-point weight is present for fail-closed coverage");

    const auto fixture_root = std::filesystem::temp_directory_path() /
                              ("zkdl-segmented-fixture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(fixture_root);
    if (!write_authenticated_resource_fixture(fixture_root, &error)) return false;
    if (!relation_artifact_roundtrip(generators, h, u, fixture_root / "tiny-proof.artifact", &error))
        return fixture_fail("artifact-roundtrip", "tiny proof/artifact roundtrip failed", &error);
    std::cerr
        << "segmented_fixture stage=collector/final status=NOT_RUN reason=fixture emits bounded artifact evidence; formal result collector/final validator remains external\n";
    std::cout << "segmented_fixture artifact=" << (fixture_root / "tiny-proof.artifact").string()
              << " independent_verify=PASS resource_logical32_frame8=PASS"
                 " cross_segment_range=PASS tamper_rejection=PASS cpu_only_fail_closed=PASS\n";
    return true;
}

bool authenticated_shared_segment_proof_fixture(const std::filesystem::path &config_path) {
    std::string error;
    Config config;
    if (!load_authenticated_test_config(config_path, &config, &error))
        return fixture_fail("authenticated-srs-config", "cannot load official Cora SRS identity", &error);
    AuthenticatedSrsResource srs;
    if (!load_authenticated_srs(config, &srs, &error) || !srs.authenticated ||
        srs.basis_domain != (1U << 18U) || srs.generators.size() != srs.basis_domain ||
        srs.blinding_generator.size() != 48U || srs.u_generator.size() != 48U) {
        return fixture_fail("authenticated-srs", "official Cora SRS did not authenticate the expected basis", &error);
    }
    fixture_pass("authenticated-srs", "official Cora manifest and complete generator basis authenticated");

    std::vector<zkdl::CpuG1Point> generators;
    generators.reserve(srs.generators.size());
    for (std::size_t index = 0U; index < srs.generators.size(); ++index) {
        if (srs.generators[index].size() != 48U)
            return fixture_fail("authenticated-basis", "authenticated generator has the wrong encoding width", &error);
        std::array<zkdl::byte, 48U> encoded{};
        std::copy(srs.generators[index].begin(), srs.generators[index].end(), encoded.begin());
        G1Jacobian_t native{};
        if (!zkdl_g1_wire::decode(encoded, native, &error))
            return fixture_fail("authenticated-basis", "authenticated generator failed canonical decode", &error);
        const auto point = zkdl::cpu_g1_from_native(native);
        if (point.is_zero())
            return fixture_fail("authenticated-basis", "authenticated SRS contains a zero generator", &error);
        generators.push_back(point);
    }
    std::array<zkdl::byte, 48U> h_encoded{}, u_encoded{};
    std::copy(srs.blinding_generator.begin(), srs.blinding_generator.end(), h_encoded.begin());
    std::copy(srs.u_generator.begin(), srs.u_generator.end(), u_encoded.begin());
    G1Jacobian_t h_native{}, u_native{};
    if (!zkdl_g1_wire::decode(h_encoded, h_native, &error) ||
        !zkdl_g1_wire::decode(u_encoded, u_native, &error))
        return fixture_fail("authenticated-basis", "authenticated H/U failed canonical decode", &error);
    const auto h = zkdl::cpu_g1_from_native(h_native);
    const auto u = zkdl::cpu_g1_from_native(u_native);

    constexpr std::uint32_t kDomain = 1U << 15U;
    constexpr std::uint32_t kSharedSegment = 1U << 14U;
    ZkRelationSegmentLayout layout;
    if (!zktr_segment_layout_for_size(kDomain, generators.size(), kSharedSegment,
                                      layout, &error) ||
        layout.segment_count != 2U || layout.segment_size != kSharedSegment)
        return fixture_fail("shared-layout", "authenticated two-segment layout was rejected", &error);
    if (zktr_segment_layout_for_size(kDomain, generators.size(), 12288U, layout, &error))
        return fixture_fail("shared-layout-negative", "non-power-of-two shared segment was accepted", &error);
    error.clear();
    if (zktr_segment_layout_for_size(kDomain, generators.size(), 1U << 19U, layout, &error))
        return fixture_fail("shared-layout-negative", "segment larger than authenticated basis was accepted", &error);

    std::vector<FrWire> values;
    values.reserve(kDomain);
    for (std::uint32_t index = 0U; index < kDomain; ++index)
        values.push_back(zkdl::fr_from_u32((index % 251U) + 1U));
    SegmentedTensorProvider provider(
        kDomain, kSharedSegment,
        [&values](std::uint64_t offset, std::uint32_t count,
                  std::vector<FrWire> &output, std::string *read_error) {
            if (offset > values.size() || count > values.size() - offset) {
                if (read_error) *read_error = "fixture range is outside the tensor";
                return false;
            }
            output.assign(values.begin() + static_cast<std::ptrdiff_t>(offset),
                           values.begin() + static_cast<std::ptrdiff_t>(offset + count));
            return true;
        });
    SegmentedTensorProvider natural_provider;
    if (!SegmentedTensorProvider::from_values(values, generators.size(), natural_provider, &error) ||
        natural_provider.segment_size() != kDomain || natural_provider.segment_count() != 1U)
        return fixture_fail("provider-partition-negative", "natural provider fixture shape is invalid", &error);
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    statement.srs_digest = srs.manifest_digest;
    const FrWire tensor_blinder = zkdl::fr_from_u32(193U);
    CommitmentWire rejected_wire;
    SegmentCommitmentMaterial rejected_material;
    error.clear();
    if (zktr_build_segment_commitment_material(
            "fixture/shared-segment-mismatch", 3U, natural_provider,
            generators, h, u, tensor_blinder, statement, rejected_wire,
            rejected_material, &error, nullptr, kSharedSegment) ||
        error.find("authenticated shared layout") == std::string::npos)
        return fixture_fail("provider-partition-negative", "provider partition mismatch was not rejected explicitly", &error);
    fixture_pass("provider-partition-negative", "natural 32768 provider rejected for authenticated 16384x2 layout");

    CommitmentWire wire;
    SegmentCommitmentMaterial material;
    if (!zktr_build_segment_commitment_material(
            "fixture/authenticated-cora-shared-segment", 3U, provider,
            generators, h, u, tensor_blinder, statement, wire, material,
            &error, nullptr, kSharedSegment) ||
        wire.segment_size != kSharedSegment || wire.segment_count != 2U ||
        wire.segment_commitments.size() != 2U)
        return fixture_fail("shared-commitment", "authenticated two-segment commitment failed", &error);
    std::vector<CommitmentWire> commitment_wires{wire};
    statement.segment_commitments_digest =
        relation_commitment_vector_digest(statement, commitment_wires);
    fixture_pass("shared-commitment", "authenticated Cora prefix basis committed as two ordered 16384-point segments");

    ConstraintDescriptor descriptor;
    descriptor.id = "fixture/shared-segment-terminal";
    descriptor.degree = 1U;
    descriptor.terms.push_back(Monomial{zkdl::fr_from_u32(1U), {0U}});
    std::vector<FrWire> point;
    point.reserve(15U);
    for (std::uint32_t index = 0U; index < 15U; ++index)
        point.push_back(zkdl::fr_from_u32(index + 2U));
    FrWire expected_value{};
    if (!provider.evaluate_mle(point, expected_value, &error))
        return fixture_fail("segmented-mle", "shared two-segment MLE evaluation failed", &error);
    FrWire external_blinder = zkdl::cpu_fresh_csprng_scalar();
    while (zkdl::fr_compare(external_blinder, FrWire{}) == 0)
        external_blinder = zkdl::cpu_fresh_csprng_scalar();
    const auto external_commitment = zkdl::cpu_pedersen_scalar_commit(
        expected_value, external_blinder, h, u);
    const std::vector<zkdl::CpuG1Point> canonical_commitments{
        zkdl::cpu_g1_from_native(wire.commitment)};
    const std::vector<FrWire> tensor_blinders{tensor_blinder};
    const std::vector<std::uint32_t> commitment_indices{0U};
    const std::vector<SegmentedTensorProvider> terminal_providers{provider};
    const std::vector<SegmentCommitmentMaterial> terminal_materials{material};
    const std::string context = "fixture.authenticated-shared-segment-terminal";
    zkdl::Transcript prover_transcript(context);
    ZkTerminalRelationProverResult terminal;
    if (!prove_zk_terminal_relation_segmented(
            descriptor, terminal_providers, canonical_commitments, tensor_blinders,
            commitment_indices, point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            external_blinder, FrWire{}, context, prover_transcript, terminal,
            &error, &terminal_materials,
            kSharedSegment))
        return fixture_fail("segmented-opening-prove", "native terminal relation failed on shared layout", &error);
    if (terminal.proof.terminal_segment_openings.size() != 1U ||
        terminal.proof.terminal_segment_openings.front().size() != 2U)
        return fixture_fail("segmented-opening-shape", "terminal relation did not emit both segment openings", &error);
    std::vector<zkdl::byte> encoded_terminal;
    zkdl::serialize_zk_terminal_relation(terminal.proof, encoded_terminal);
    zkdl::ZkTerminalRelationProof fresh_terminal;
    if (!zkdl::deserialize_zk_terminal_relation(encoded_terminal, fresh_terminal, &error))
        return fixture_fail("segmented-opening-wire", "terminal opening did not survive canonical wire roundtrip", &error);
    zkdl::Transcript verifier_transcript(context);
    ZkTerminalRelationVerifierResult verified;
    if (!verify_zk_terminal_relation(
            fresh_terminal, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, verifier_transcript, verified,
            zkdl::ZkTerminalOpeningMode::LegacyPerSegment, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("segmented-opening-independent-verify", "fresh verifier rejected the serialized two-segment opening", &error);
    fixture_pass("segmented-opening-independent-verify", "serialized two-segment MLE opening verified against authenticated Cora SRS and commitment wire");
    std::cerr << "segmented_fixture stage=collector/final status=NOT_RUN reason=bounded native proof fixture; official comparison collector remains external\n";
    return true;
}

}

int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--authenticated-srs-config") {
            return zkdl_gat::relation::fixture::authenticated_shared_segment_proof_fixture(argv[2]) ? 0 : 1;
        }
        if (argc == 2 && std::string(argv[1]) == "--qpad-rlc-only") {
            std::string error;
            return zkdl_gat::relation::fixture::qpad_same_mask_group_fixture(
                       &error)
                       ? 0
                       : 1;
        }
        if (argc == 2 && std::string(argv[1]) == "--sorted-receipt-only") {
            std::vector<zkdl::CpuG1Point> generators;
            generators.reserve(8U);
            for (std::uint32_t index = 0U; index < 8U; ++index)
                generators.push_back(zkdl::cpu_g1_mul(
                    zkdl::cpu_g1_generator(), zkdl::fr_from_u32(index + 2U)));
            const auto h = zkdl::cpu_g1_mul(
                zkdl::cpu_g1_generator(), zkdl::fr_from_u32(101U));
            const auto u = zkdl::cpu_g1_mul(
                zkdl::cpu_g1_generator(), zkdl::fr_from_u32(103U));
            return ::zkdl_run_tiny_sorted_lookup_receipt_fixture(
                       generators, h, u)
                       ? 0
                       : 1;
        }
        if (argc == 2 && std::string(argv[1]) == "--msm-only") {
            const auto h = zkdl::cpu_g1_mul(
                zkdl::cpu_g1_generator(), zkdl::fr_from_u32(101U));
            const auto u = zkdl::cpu_g1_mul(
                zkdl::cpu_g1_generator(), zkdl::fr_from_u32(103U));
            std::string error;
            return zkdl_gat::relation::fixture::prepared_msm_chunk_fixture(
                       h, u, &error)
                       ? 0
                       : 1;
        }
        if (argc != 1) {
            std::cerr << "usage: segmented_lookup_fixture [--authenticated-srs-config CONFIG | --qpad-rlc-only | --sorted-receipt-only | --msm-only]\n";
            return 2;
        }
        if (!zkdl_gat::relation::fixture::descriptor_backing_cow_fixture()) return 1;
        if (!zkdl_gat::relation::fixture::small_shared_segment_terminal_fixture()) return 1;
        if (!zkdl_gat::relation::fixture::basis8_logical32_four_segment_tamper_fixture()) return 1;
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "segmented_fixture FAIL: " << exception.what() << '\n';
        return 1;
    }
}
