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
        if (entry.first == "srs_manifest_digest" || entry.first == "verifier_registry_digest" ||
            entry.first == "model_blinder_sidecar_digest") continue;
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
    prover.ipa_generators = generators;
    prover.ipa_blinding_generator = h;
    prover.ipa_u_generator = u;
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
    verifier.ipa_generators = generators;
    verifier.ipa_blinding_generator = h;
    verifier.ipa_u_generator = u;
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
    constexpr std::uint32_t kDomain = 8U;
    constexpr std::uint32_t kSharedSegment = 4U;
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
    fixture_pass("shared-small-layout", "logical domain 8 uses canonical shared segment size 4 across two segments");

    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    const FrWire tensor_blinder = zkdl::fr_from_u32(193U);
    CommitmentWire wire;
    SegmentCommitmentMaterial material;
    if (!zktr_build_segment_commitment_material(
            "fixture/small-shared", 3U, provider, generators, h, u,
            tensor_blinder, statement, wire, material, &error, nullptr,
            kSharedSegment) ||
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
            &error, nullptr, kSharedSegment) ||
        error.find("authenticated shared layout") == std::string::npos)
        return fixture_fail("shared-small-negative", "natural provider was not rejected for the shared-S layout", &error);
    fixture_pass("shared-small-commitment", "both segments committed using the declared common S");

    ConstraintDescriptor descriptor;
    descriptor.id = "fixture/small-shared-terminal";
    descriptor.degree = 1U;
    descriptor.terms.push_back(Monomial{zkdl::fr_from_u32(1U), {0U}});
    std::vector<FrWire> point{zkdl::fr_from_u32(2U), zkdl::fr_from_u32(3U),
                              zkdl::fr_from_u32(5U)};
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
    const std::string context = "fixture.small-shared-terminal";
    zkdl::Transcript prover_transcript(context);
    ZkTerminalRelationProverResult terminal;
    if (!prove_zk_terminal_relation_segmented(
            descriptor, terminal_providers, canonical_commitments, tensor_blinders,
            commitment_indices, point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            external_blinder, FrWire{}, context, prover_transcript, terminal,
            &error, &terminal_materials, kSharedSegment))
        return fixture_fail("shared-small-prove", "small shared-S terminal proof failed", &error);
    if (terminal.proof.terminal_segment_openings.size() != 1U ||
        terminal.proof.terminal_segment_openings.front().size() != 2U)
        return fixture_fail("shared-small-prove", "terminal proof omitted a shared-S segment opening");
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
            FrWire{}, context, verifier_transcript, verified, &error,
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
            FrWire{}, context, tamper_transcript, verified, &error,
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
            FrWire{}, context, aggregate_tamper_transcript, verified, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "tampered aggregate terminal commitment was accepted");
    fixture_pass("shared-small-tamper", "aggregate terminal commitment must match all weighted segment taus");

    tampered = fresh;
    const auto second_segment_tau = zkdl::cpu_g1_from_native(
        tampered.terminal_segment_openings[0][1].tau);
    tampered.terminal_segment_openings[0][1].tau = zkdl::cpu_g1_to_native(
        zkdl::cpu_g1_add(second_segment_tau, h));
    zkdl::Transcript segment_tau_tamper_transcript(context);
    error.clear();
    if (verify_zk_terminal_relation(
            tampered, descriptor, canonical_commitments, commitment_indices,
            point, generators, h, u, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality, external_commitment,
            FrWire{}, context, segment_tau_tamper_transcript, verified, &error,
            &commitment_wires, kSharedSegment))
        return fixture_fail("shared-small-tamper", "tampered second-segment tau opening was accepted");
    fixture_pass("shared-small-tamper", "mutated nonzero-index segment tau rejected by fresh verifier");
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
            FrWire{}, context, verifier_transcript, verified, &error,
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
        if (argc != 1) {
            std::cerr << "usage: segmented_lookup_fixture [--authenticated-srs-config CONFIG]\n";
            return 2;
        }
        if (!zkdl_gat::relation::fixture::small_shared_segment_terminal_fixture()) return 1;
        if (!zkdl_gat::relation::fixture::basis8_logical32_four_segment_tamper_fixture()) return 1;
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "segmented_fixture FAIL: " << exception.what() << '\n';
        return 1;
    }
}
