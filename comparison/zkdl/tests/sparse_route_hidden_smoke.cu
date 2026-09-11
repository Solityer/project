#include <iostream>
#include <type_traits>
#include "../zkrelation-proof.cuh"

int main() {
    using namespace zkdl_gat::relation;
    const auto scalar = [](std::uint32_t value) { return zkdl::fr_from_u32(value); };
    // Algebra-only fixture, deliberately confined to tests.  These known-DL
    // points and digest labels are not a trusted production SRS or registry.
    std::vector<zkdl::CpuG1Point> generators;
    for (std::uint32_t i = 0U; i < 8U; ++i)
        generators.push_back(zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), scalar(i + 2U)));
    const auto h = zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), scalar(103U));
    const auto u = zkdl::cpu_g1_mul(zkdl::cpu_g1_generator(), scalar(109U));
    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});
    TensorWitness source{"layer1/source", 3U, 2U, {scalar(3U), scalar(5U)}};
    TensorWitness output{"layer1/output", 3U, 8U,
        {scalar(5U), scalar(5U), scalar(3U), scalar(3U),
         scalar(5U), scalar(5U), scalar(0U), scalar(0U)}};
    const std::map<std::string, const TensorWitness*> witnesses{
        {source.id, &source}, {output.id, &output}};
    const std::map<std::string, FrWire> blinders{
        {source.id, scalar(17U)}, {output.id, scalar(19U)}};
    std::vector<CommitmentWire> commitments;
    for (const auto* tensor : {&source, &output}) {
        const auto commitment = zkdl::cpu_g1_add(zkdl::cpu_g1_sum(
            std::vector<zkdl::CpuG1Point>(generators.begin(),
                generators.begin() + tensor->domain_size), tensor->values),
            zkdl::cpu_g1_mul(h, blinders.at(tensor->id)));
        commitments.push_back(CommitmentWire{tensor->id, tensor->visibility,
            tensor->domain_size, zkdl::cpu_g1_to_native(commitment)});
    }
    OperationDescriptor descriptor;
    descriptor.id = "layer1/route";
    descriptor.kind = RelationKind::SparseRoute;
    descriptor.degree = 2U;
    descriptor.source_refs = {source.id, output.id};
    descriptor.tensor_refs = {source.id, output.id};
    descriptor.mapping = "public-dst-and-feature-broadcast";
    descriptor.padding = "inactive-edge-zero";
    descriptor.terminal_claim = "hidden-canonical-evaluations";
    descriptor.terminal_point = "feature-before-outer";
    descriptor.public_route = true;
    descriptor.route_by_destination = true;
    descriptor.selector_edge_domain = 4U;
    descriptor.selector_node_domain = 2U;
    descriptor.selector_feature_domain = 2U;
    descriptor.selector_source_feature_domain = 1U;
    descriptor.selector_src = {0U, 1U, 0U, 0U};
    descriptor.selector_dst = {1U, 0U, 1U, 0U};
    descriptor.selector_active = {1U, 1U, 1U, 0U};
    descriptor.selector_feature_map = {0U, 0U};
    descriptor.selector_source_ref = source.id;
    descriptor.selector_output_ref = output.id;
    std::string error;
    const auto require = [&](bool condition, const char* label) {
        if (!condition) throw std::runtime_error(std::string(label) + ": " + error);
    };
    try {
        require(descriptor.valid(&error), "descriptor");
        require(formal_relation_components_supported({descriptor}, &error), "route admission");
        zkdl::Transcript prover_transcript("sparse-route-hidden-smoke");
        SparseRouteProof proof;
        static_assert(std::is_same<decltype(proof.sumcheck), zkdl::ZkSumcheckProof>::value,
                      "route must use hidden Sumcheck");
        static_assert(std::is_same<decltype(proof.source_opening), zkdl::HyraxOpeningProofV3>::value,
                      "route must use a hidden evaluation opening");
        require(prove_sparse_route(descriptor, witnesses, blinders, generators, h, u,
            commitments, statement, prover_transcript, proof, &error), "prove");
        const auto encoded = proof.serialize();
        SparseRouteProof decoded;
        require(SparseRouteProof::deserialize(encoded, decoded, &error), "decode");
        require(decoded.serialize() == encoded, "canonical wire");
        zkdl::Transcript verifier_transcript("sparse-route-hidden-smoke");
        require(verify_sparse_route(descriptor, commitments, generators, h, u,
            statement, verifier_transcript, decoded, &error), "independent verify");
        require(fr_compare_mixed(prover_transcript.challenge("after-route"),
            verifier_transcript.challenge("after-route")) == 0, "transcript replay");

        auto changed_descriptor = descriptor;
        changed_descriptor.selector_dst[0] = 0U;
        zkdl::Transcript changed_transcript("sparse-route-hidden-smoke");
        require(!verify_sparse_route(changed_descriptor, commitments, generators, h, u,
            statement, changed_transcript, decoded, &error), "reject changed public route");
        auto changed_commitments = commitments;
        changed_commitments[0].commitment = zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(
            zkdl::cpu_g1_from_native(commitments[0].commitment), generators[0]));
        zkdl::Transcript replaced_transcript("sparse-route-hidden-smoke");
        require(!verify_sparse_route(descriptor, changed_commitments, generators, h, u,
            statement, replaced_transcript, decoded, &error), "reject replaced canonical source");
        auto old_version = encoded;
        old_version[4] = 2U;
        require(!SparseRouteProof::deserialize(old_version, decoded, &error), "reject old revealing wire");
        auto trailing = encoded;
        trailing.push_back(0U);
        require(!SparseRouteProof::deserialize(trailing, decoded, &error), "reject trailing wire");
        decoded = proof;
        decoded.terminal_binding.schnorr_s = zkdl::fr_value(zkdl::fr_add(
            zkdl::fr_wire(decoded.terminal_binding.schnorr_s), scalar(1U)));
        zkdl::Transcript tampered_transcript("sparse-route-hidden-smoke");
        require(!verify_sparse_route(descriptor, commitments, generators, h, u,
            statement, tampered_transcript, decoded, &error), "reject terminal binding tamper");
        std::cout << "sparse_route_hidden_smoke PASS: cross-domain route, destination/feature "
                     "selectors, inactive padding, hidden wire, independent verifier, tamper\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
