#include <iostream>
#include <type_traits>
#include "../zkrelation-proof.cuh"

int main() {
    using namespace zkdl_gat::relation;
    const auto scalar = [](std::uint32_t value) { return zkdl::fr_from_u32(value); };
    const auto require = [](bool condition, const char *label,
                            const std::string& detail) {
        if (!condition)
            throw std::runtime_error(std::string(label) + ": " + detail);
    };

    std::vector<zkdl::CpuG1Point> generator_points;
    const auto seed = zkdl::cpu_g1_generator();
    for (std::uint32_t index = 0U; index < 4U; ++index)
        generator_points.push_back(zkdl::cpu_g1_mul(seed, scalar(index + 2U)));
    const auto h = zkdl::cpu_g1_mul(seed, scalar(103U));
    const auto u = zkdl::cpu_g1_mul(seed, scalar(109U));
    zkdl::CpuIpaBasisContext prover_basis;
    std::string error;
    require(zkdl::CpuIpaBasisContext::authenticate(
                std::move(generator_points), h, u, prover_basis, &error),
            "prover basis authentication", error);
    const auto& generators = prover_basis.generators();
    zkdl::CpuG1MsmSession prover_session(prover_basis);

    TensorWitness canonical{"layer1/fac/canonical", 3U, 2U,
                            {scalar(3U), scalar(5U)}};
    TensorWitness view{"layer1/fac/view", 3U, 8U,
                       {scalar(3U), scalar(3U), scalar(3U), scalar(3U),
                        scalar(5U), scalar(5U), scalar(5U), scalar(5U)}};
    const std::map<std::string, const TensorWitness*> witnesses{
        {canonical.id, &canonical}, {view.id, &view}};
    const std::map<std::string, FrWire> blinders{
        {canonical.id, scalar(29U)}, {view.id, scalar(31U)}};

    RelationStatement statement;
    statement.statement_digest = zkdl::sha256(std::vector<zkdl::byte>{'s'});
    statement.model_digest = zkdl::sha256(std::vector<zkdl::byte>{'m'});
    statement.table_digest = zkdl::sha256(std::vector<zkdl::byte>{'t'});
    statement.ledger_digest = zkdl::sha256(std::vector<zkdl::byte>{'l'});
    statement.srs_digest = zkdl::sha256(std::vector<zkdl::byte>{'p'});
    statement.registry_digest = zkdl::sha256(std::vector<zkdl::byte>{'r'});

    OperationDescriptor fac;
    fac.id = "layer1/fac-smoke";
    fac.kind = RelationKind::Add;
    fac.degree = 2U;
    fac.source_refs = {canonical.id, view.id};
    fac.tensor_refs = fac.source_refs;
    fac.mapping = "tiny-fac-nested-cross-segment-route";
    fac.padding = "zero-mask";
    fac.terminal_claim = "fac-terminal";
    fac.terminal_point = "fac-point";
    ConstraintDescriptor constraint;
    constraint.id = fac.id + "/shape";
    constraint.degree = 1U;
    constraint.terms = {Monomial{scalar(1U), {0U}}};
    constraint.logical_shape = {2U};
    constraint.padded_shape = {2U};
    fac.constraints.push_back(std::move(constraint));
    FacBindingDescriptor binding;
    binding.view_ref = view.id;
    binding.canonical_ref = canonical.id;
    binding.view_outer_domain = 2U;
    binding.canonical_outer_domain = 2U;
    binding.view_feature_domain = 4U;
    binding.canonical_feature_domain = 1U;
    binding.source_outer_index = {0U, 1U};
    binding.source_feature_index = {0U, 0U, 0U, 0U};
    binding.selector_active = {1U, 1U};
    fac.fac_bindings.push_back(std::move(binding));
    require(fac.valid(&error), "FAC descriptor", error);

    std::vector<CommitmentWire> commitments;
    std::map<std::string, std::vector<FrWire>> segment_blinders;
    for (const TensorWitness* tensor : {&canonical, &view}) {
        SegmentedTensorProvider provider;
        SegmentCommitmentMaterial material;
        CommitmentWire commitment;
        require(SegmentedTensorProvider::from_values(
                    tensor->values, generators.size(), provider, &error),
                "segmented provider", error);
        require(zktr_build_segment_commitment_material(
                    tensor->id, tensor->visibility, provider, generators, h, u,
                    blinders.at(tensor->id), statement, commitment, material,
                    &error, nullptr, 0U, nullptr, &prover_basis,
                    &prover_session),
                "segmented commitment material", error);
        if (tensor->id == view.id &&
            (commitment.segment_size != 4U || commitment.segment_count != 2U))
            throw std::runtime_error("view commitment did not cross the expected 4+4 segments");
        segment_blinders.emplace(tensor->id, std::move(material.blinders));
        commitments.push_back(std::move(commitment));
    }
    statement.segment_commitments_digest =
        relation_commitment_vector_digest(statement, commitments);

    const auto raw_source_digest = zkdl::cpu_ipa_basis_digest(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + 2),
        h, u, 4U);
    const auto cached_source_digest =
        zkdl::cpu_ipa_basis_digest(prover_basis, 2U, 4U);
    const auto raw_view_digest = zkdl::cpu_ipa_basis_digest(generators, h, u, 8U);
    const auto cached_view_digest =
        zkdl::cpu_ipa_basis_digest(prover_basis, 4U, 8U);
    require(raw_source_digest == cached_source_digest &&
                raw_view_digest == cached_view_digest,
            "legacy prefix basis digests", "prepared and raw basis digest differ");

    static_assert(std::is_same<decltype(std::declval<SparseRouteProof>().source_opening),
                               zkdl::HyraxOpeningProofV3>::value,
                  "nested FAC selector must use hidden Hyrax openings");
    FacLine8Proof proof;
    zkdl::Transcript prover_transcript("fac-line8-cross-segment-smoke");
    require(prove_fac_line8(
                fac, witnesses, blinders, generators, h, u, commitments,
                statement, prover_transcript, proof, &error, &segment_blinders,
                &prover_basis, &prover_session),
            "prove nested FAC sparse route", error);
    const std::vector<zkdl::byte> encoded = proof.serialize();
    FacLine8Proof decoded;
    require(FacLine8Proof::deserialize(encoded, decoded, &error) &&
                decoded.serialize() == encoded,
            "canonical FAC wire", error);

    zkdl::CpuIpaBasisContext verifier_basis;
    require(zkdl::CpuIpaBasisContext::authenticate(
                std::vector<zkdl::CpuG1Point>(generators), h, u,
                verifier_basis, &error),
            "fresh verifier basis authentication", error);
    zkdl::CpuG1MsmSession verifier_session(verifier_basis);
    zkdl::Transcript verifier_transcript("fac-line8-cross-segment-smoke");
    require(verify_fac_line8(
                fac, commitments, verifier_basis.generators(), h, u,
                statement, verifier_transcript, decoded, &error,
                &verifier_basis, &verifier_session),
            "fresh independent FAC verification", error);
    require(fr_compare_mixed(
                prover_transcript.challenge("after-fac"),
                verifier_transcript.challenge("after-fac")) == 0,
            "transcript replay", "prover and fresh verifier transcripts differ");

    decoded.selector_proofs[0U].output_opening.z1.val[0] ^= 1U;
    zkdl::Transcript tampered_transcript("fac-line8-cross-segment-smoke");
    require(!verify_fac_line8(
                fac, commitments, verifier_basis.generators(), h, u,
                statement, tampered_transcript, decoded, &error,
                &verifier_basis, &verifier_session),
            "FAC opening tamper rejection", "tampered opening was accepted");

    std::cout << "sparse_route_hidden_smoke PASS nested_FAC=1 view_domain=8"
              << " segment_size=4 segment_count=2 canonical_domain=2 basis=4"
              << " raw_prefix_digest=PREPARED_EQUAL fresh_verify=PASS"
              << " tamper=REJECTED full_datasets=0\n";
    return 0;
}
