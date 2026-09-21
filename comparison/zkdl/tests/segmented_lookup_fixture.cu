#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "../zkrelation-proof.cuh"
#include "../gat_adapter/gat_native_resources.hpp"

// Small algebra-only fixture for the authenticated lookup segment contract.
// It is intentionally not wired into a formal runner and is not a resource
// or SRS generator.  The production gate remains the independent verifier.
namespace zkdl_gat::relation::fixture {

bool fixture_fail(const char* stage, const std::string& reason,
                  std::string* error = nullptr) {
    std::string detail = reason;
    if (error != nullptr && !error->empty() && *error != reason) {
        detail += ": ";
        detail += *error;
    }
    if (error != nullptr) *error = detail;
    std::cerr << "segmented_fixture stage=" << stage
              << " status=FAIL reason=" << detail << '\n';
    return false;
}

void fixture_pass(const char* stage, const char* reason) {
    std::cerr << "segmented_fixture stage=" << stage
              << " status=PASS reason=" << reason << '\n';
}

std::string hex_digest(const std::array<zkdl::byte, 32U>& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string text;
    text.reserve(64U);
    for (const auto value : digest) {
        text.push_back(kHex[(value >> 4U) & 0x0fU]);
        text.push_back(kHex[value & 0x0fU]);
    }
    return text;
}

bool write_authenticated_resource_fixture(const std::filesystem::path& root,
                                          std::string* error) {
    const std::string manifest =
        "{\"manifest_format\":\"zkgat_srs_manifest\","
        "\"manifest_version\":2,\"basis_domain\":8,"
        "\"logical_domain\":32,\"segment_size\":8,"
        "\"segment_count\":4,\"domain\":8,"
        "\"chunk_id\":\"fixture-4seg\","
        "\"generator_digest\":\"fixture-generator\"}\n";
    const auto manifest_bytes = std::vector<zkdl::byte>(manifest.begin(), manifest.end());
    const auto manifest_digest = zkdl::sha256(manifest_bytes);
    const std::string registry =
        "{\"schema\":\"zkDL_gat_model_registry_v3\","
        "\"srs_manifest_digest\":\"" + hex_digest(manifest_digest) + "\","
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

    // This mirrors the authenticated loader's logical/basis admission rule:
    // one physical frame is reused across four logical segments.
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
    const bool frame_admission = chunk.logical_domain > chunk.basis_domain &&
                                 chunk.segment_count ==
                                     chunk.logical_domain / chunk.segment_size &&
                                 chunk.segments.size() == chunk.segment_count;
    std::ifstream manifest_check(root / "srs_manifest.json", std::ios::binary);
    std::ifstream registry_check(root / "model_registry.json", std::ios::binary);
    std::string manifest_read((std::istreambuf_iterator<char>(manifest_check)), {});
    std::string registry_read((std::istreambuf_iterator<char>(registry_check)), {});
    if (!frame_admission || manifest_read != manifest ||
        registry_read.find(hex_digest(manifest_digest)) == std::string::npos ||
        registry_read.find("\"logical_domain\":32") == std::string::npos) {
        if (error) *error = "fixture authenticated resource metadata is not canonical";
        return fixture_fail("resource-admission", "manifest/registry/frame metadata is not canonical", error);
    }
    fixture_pass("resource-admission", "logical32 is admitted over four physical frame8 segments");
    return true;
}

bool relation_artifact_roundtrip(
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
    const std::filesystem::path& artifact_path, std::string* error) {
    const auto scalar = [](std::uint32_t value) { return zkdl::fr_from_u32(value); };
    const std::vector<FrWire> public_values{
        scalar(1U), scalar(2U), scalar(3U), scalar(4U),
        scalar(5U), scalar(6U), scalar(7U), scalar(8U)};
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
    if (!prove(prover, proof, error))
        return fixture_fail("prove", "CPU native tiny relation rejected", error);
    fixture_pass("prove", "CPU native tiny relation produced a proof");
    const std::vector<byte> artifact = proof.serialize();
    std::ofstream output(artifact_path, std::ios::binary);
    if (!output) {
        if (error) *error = "fixture cannot write proof artifact";
        return fixture_fail("artifact-roundtrip", "cannot write proof artifact", error);
    }
    output.write(reinterpret_cast<const char*>(artifact.data()),
                 static_cast<std::streamsize>(artifact.size()));
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
    if (!verify(decoded, verifier, error))
        return fixture_fail("independent-verify", "fresh verifier rejected valid artifact", error);
    fixture_pass("independent-verify", "fresh verifier accepted serialized artifact");

    // A single-segment wire is naturally compatible: segment[0] equals the
    // legacy handle, so swapping those two fields would be an identity.  Use
    // a real Xi mutation here; the ordered four-segment swap is covered by
    // the algebra fixture above.
    Proof swapped = decoded;
    const auto original_xi = zkdl::cpu_g1_from_native(
        swapped.commitments[0].segment_commitments[0].commitment);
    swapped.commitments[0].segment_commitments[0].commitment =
        zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(original_xi, h));
    swapped.commitments[0].segment_commitment_digest =
        segment_commitment_digest(swapped.commitments[0]);
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

bool basis8_logical32_four_segment_tamper_fixture() {
    std::vector<zkdl::CpuG1Point> generators;
    generators.reserve(8U);
    for (std::uint32_t index = 0U; index < 8U; ++index)
        generators.push_back(zkdl::cpu_g1_mul(
            zkdl::cpu_g1_generator(), zkdl::fr_from_u32(index + 2U)));
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
    for (std::uint32_t index = 0U; index < 32U; ++index)
        values.push_back(zkdl::fr_from_u32(index + 1U));
    SegmentedTensorProvider provider;
    std::string error;
    if (!SegmentedTensorProvider::from_values(values, generators.size(), provider, &error) ||
        provider.logical_domain() != 32U || provider.segment_size() != 8U ||
        provider.segment_count() != 4U)
        return fixture_fail("range", "logical32/frame8 provider shape or admission failed", &error);
    fixture_pass("range", "cross-segment provider shape is logical32/frame8/4");
    std::vector<FrWire> first_segment, last_segment;
    if (!provider.read_segment(0U, first_segment, &error) ||
        !provider.read_segment(3U, last_segment, &error) ||
        first_segment.size() != 8U || last_segment.size() != 8U ||
        zkdl::fr_compare(first_segment.front(), zkdl::fr_from_u32(1U)) != 0 ||
        zkdl::fr_compare(last_segment.front(), zkdl::fr_from_u32(25U)) != 0)
        return fixture_fail("range", "cross-segment range read returned unexpected values", &error);
    fixture_pass("range", "segment0/segment3 range reads returned values 1/25");

    CommitmentWire wire;
    SegmentCommitmentMaterial material;
    if (!zktr_build_segment_commitment_material(
            "fixture/lookup", 3U, provider, generators, h, u,
            zkdl::fr_from_u32(17U), statement, wire, material, &error) ||
        wire.segment_count != 4U || wire.segment_size != 8U)
        return fixture_fail("v5-digest", "ordered v5 segment material construction failed", &error);
    fixture_pass("v5-digest", "ordered Xi vector and digest constructed");

    // Swapping Xi points preserves the unweighted handle only if the points
    // are swapped as a pair; the authenticated ordered material still rejects.
    CommitmentWire swapped = wire;
    std::swap(swapped.segment_commitments[0].commitment,
              swapped.segment_commitments[1].commitment);
    swapped.segment_commitment_digest = segment_commitment_digest(swapped);
    if (!segment_commitment_wire_shape_valid(swapped, &error) ||
        zktr_segment_material_matches(swapped, material, &error))
        return fixture_fail("tamper-swap", "ordered segment swap was not rejected", &error);
    fixture_pass("tamper-swap", "ordered segment swap rejected");

    // A +delta/-delta pair leaves legacy Sigma Xi unchanged, but cannot pass
    // the ordered material check.
    CommitmentWire cancelled = wire;
    const FrWire delta = zkdl::fr_from_u32(19U);
    const auto plus = zkdl::cpu_g1_add(
        zkdl::cpu_g1_from_native(cancelled.segment_commitments[0].commitment),
        zkdl::cpu_g1_mul(h, delta));
    const auto minus = zkdl::cpu_g1_add(
        zkdl::cpu_g1_from_native(cancelled.segment_commitments[1].commitment),
        zkdl::cpu_g1_mul(h, zkdl::fr_neg(delta)));
    cancelled.segment_commitments[0].commitment = zkdl::cpu_g1_to_native(plus);
    cancelled.segment_commitments[1].commitment = zkdl::cpu_g1_to_native(minus);
    cancelled.segment_commitment_digest = segment_commitment_digest(cancelled);
    if (!zkdl::cpu_g1_equal(
            zkdl::cpu_g1_from_native(cancelled.commitment),
            zkdl::cpu_g1_from_native(wire.commitment)) ||
        zktr_segment_material_matches(cancelled, material, &error))
        return fixture_fail("tamper-delta", "+delta/-delta cancellation was not rejected", &error);
    fixture_pass("tamper-delta", "legacy sum cancellation does not pass ordered material");

    // The last segment has full coverage in logical32/basis8; a forged one-
    // element padding claim must fail closed even after digest recomputation.
    CommitmentWire padding = wire;
    padding.segment_commitments.back().actual_length = 7U;
    padding.segment_commitments.back().padding_length = 1U;
    padding.segment_commitment_digest = segment_commitment_digest(padding);
    if (segment_commitment_wire_shape_valid(padding, &error))
        return fixture_fail("tamper-padding", "forged last-segment padding was accepted", &error);
    fixture_pass("tamper-padding", "forged last-segment padding rejected");

    CommitmentWire version = wire;
    version.segment_binding_version = 4U;
    version.segment_commitment_digest = segment_commitment_digest(version);
    if (segment_commitment_wire_shape_valid(version, &error))
        return fixture_fail("tamper-version", "v4 segment binding was accepted", &error);
    fixture_pass("tamper-version", "v4 segment binding rejected");

    const std::vector<FrWire> zero_high_point{
        zkdl::fr_from_u32(0U), zkdl::fr_from_u32(1U)};
    const auto zero_weights = zkdl::cpu_ipa_mle_weights(zero_high_point);
    if (!std::any_of(zero_weights.begin(), zero_weights.end(), [](const FrWire& weight) {
            return zkdl::fr_compare(weight, FrWire{}) == 0;
        }))
        return fixture_fail("zero-weight", "fixture did not produce a zero high-point weight", &error);
    fixture_pass("zero-weight", "zero high-point weight is present for fail-closed coverage");

    const auto fixture_root = std::filesystem::temp_directory_path() /
        ("zkdl-segmented-fixture-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(fixture_root);
    if (!write_authenticated_resource_fixture(fixture_root, &error)) return false;
    if (!relation_artifact_roundtrip(
            generators, h, u, fixture_root / "tiny-proof.artifact", &error))
        return fixture_fail("artifact-roundtrip", "tiny proof/artifact roundtrip failed", &error);
    std::cerr << "segmented_fixture stage=collector/final status=NOT_RUN reason=fixture emits bounded artifact evidence; formal result collector/final validator remains external\n";
    std::cout << "segmented_fixture artifact="
              << (fixture_root / "tiny-proof.artifact").string()
              << " independent_verify=PASS resource_logical32_frame8=PASS"
                 " cross_segment_range=PASS tamper_rejection=PASS cpu_only_fail_closed=PASS\n";
    return true;
}

}  // namespace zkdl_gat::relation::fixture

int main() {
    try {
        if (!zkdl_gat::relation::fixture::basis8_logical32_four_segment_tamper_fixture())
            return 1;
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "segmented_fixture FAIL: " << exception.what() << '\n';
        return 1;
    }
}
