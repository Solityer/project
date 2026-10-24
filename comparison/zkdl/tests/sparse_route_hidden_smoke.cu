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

    std::vector<FrWire> canonical_values;
    canonical_values.reserve(16U);
    for (std::uint32_t index = 0U; index < 16U; ++index)
        canonical_values.push_back(scalar(index + 3U));
    TensorWitness canonical{"layer1/fac/canonical", 3U, 16U,
                            std::move(canonical_values)};
    const auto make_view = [&](const std::string& id,
                               const std::array<std::uint32_t, 4U>& feature_map) {
        (void)id;
        std::vector<FrWire> values;
        values.reserve(16U);
        for (std::uint32_t outer = 0U; outer < 4U; ++outer)
            for (const auto source_feature : feature_map)
                values.push_back(canonical.values[outer * 4U + source_feature]);
        return TensorWitness{id, 3U, 16U, std::move(values)};
    };
    TensorWitness view{"layer1/fac/view-a", 3U, 16U,
                       make_view("layer1/fac/view-a", {0U, 1U, 2U, 3U}).values};
    TensorWitness second_view{"layer1/fac/view-b", 3U, 16U,
                              make_view("layer1/fac/view-b", {1U, 2U, 3U, 0U}).values};
    const std::map<std::string, const TensorWitness*> witnesses{
        {canonical.id, &canonical}, {view.id, &view},
        {second_view.id, &second_view}};
    const std::map<std::string, FrWire> blinders{
        {canonical.id, scalar(29U)}, {view.id, scalar(31U)},
        {second_view.id, scalar(37U)}};

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
    fac.source_refs = {canonical.id, view.id, second_view.id};
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
    binding.view_outer_domain = 4U;
    binding.canonical_outer_domain = 4U;
    binding.view_feature_domain = 4U;
    binding.canonical_feature_domain = 4U;
    binding.source_outer_index = {0U, 1U, 2U, 3U};
    binding.source_feature_index = {0U, 1U, 2U, 3U};
    binding.selector_active = {1U, 1U, 1U, 1U};
    fac.fac_bindings.push_back(std::move(binding));
    FacBindingDescriptor second_binding = fac.fac_bindings.front();
    second_binding.view_ref = second_view.id;
    second_binding.source_feature_index = {1U, 2U, 3U, 0U};
    fac.fac_bindings.push_back(std::move(second_binding));
    require(fac.valid(&error), "FAC descriptor", error);
    std::vector<FacLine8GroupPlan> fac_groups;
    require(fac_line8_build_groups(fac, fac_groups, &error) &&
                fac_groups.size() == 1U &&
                fac_groups[0U].member_indices ==
                    std::vector<std::size_t>({0U, 1U}),
            "FAC feature-RLC deterministic grouping",
            "same-source bindings with distinct feature maps were not grouped");
    const auto require_separate_fac_groups = [&](
            const OperationDescriptor& candidate, const char* label) {
        std::vector<FacLine8GroupPlan> candidate_groups;
        error.clear();
        require(fac_line8_build_groups(candidate, candidate_groups, &error) &&
                    candidate_groups.size() == 2U &&
                    candidate_groups[0U].member_indices.size() == 1U &&
                    candidate_groups[1U].member_indices.size() == 1U,
                label, error.empty()
                    ? "incompatible FAC bindings were merged"
                    : error);
    };
    auto cross_outer_fac = fac;
    cross_outer_fac.fac_bindings[1U].source_outer_index[0U] = 1U;
    require_separate_fac_groups(cross_outer_fac,
                                "FAC cross-outer bindings stay separate");
    auto different_active_fac = fac;
    different_active_fac.fac_bindings[1U].selector_active[0U] = 0U;
    different_active_fac.fac_bindings[1U].source_outer_index[0U] = 0U;
    require_separate_fac_groups(different_active_fac,
                                "FAC active-mask bindings stay separate");
    auto different_layout_fac = fac;
    different_layout_fac.fac_bindings[1U].view_outer_domain = 2U;
    different_layout_fac.fac_bindings[1U].view_feature_domain = 8U;
    different_layout_fac.fac_bindings[1U].source_outer_index = {0U, 1U};
    different_layout_fac.fac_bindings[1U].selector_active = {1U, 1U};
    different_layout_fac.fac_bindings[1U].source_feature_index =
        {1U, 2U, 3U, 0U, 1U, 2U, 3U, 0U};
    require_separate_fac_groups(different_layout_fac,
                                "FAC target-layout bindings stay separate");

    std::vector<CommitmentWire> commitments;
    std::map<std::string, std::vector<FrWire>> segment_blinders;
    std::map<std::string, SegmentCommitmentValidationReceipt> receipts;
    for (const TensorWitness* tensor : {&canonical, &view, &second_view}) {
        SegmentedTensorProvider provider;
        SegmentCommitmentMaterial material;
        CommitmentWire commitment;
        SegmentCommitmentValidationReceipt receipt;
        require(SegmentedTensorProvider::from_values(
                    tensor->values, generators.size(), provider, &error),
                "segmented provider", error);
        require(zktr_build_segment_commitment_material(
                    tensor->id, tensor->visibility, provider, generators, h, u,
                    blinders.at(tensor->id), statement, commitment, material,
                    &error, nullptr, 0U, nullptr, &prover_basis,
                    &prover_session, &receipt),
                "segmented commitment material", error);
        if (commitment.segment_size != 4U || commitment.segment_count != 4U ||
            !receipt.valid() || receipt.segment_count() != 4U)
            throw std::runtime_error("FAC fixture commitment did not cover four authenticated segments");
        segment_blinders.emplace(tensor->id, std::move(material.blinders));
        receipts.emplace(tensor->id, std::move(receipt));
        commitments.push_back(std::move(commitment));
    }
    statement.segment_commitments_digest =
        relation_commitment_vector_digest(statement, commitments);

    const std::uint32_t fixture_padded_basis_size =
        zkdl::cpu_ipa_padded_size(static_cast<std::uint32_t>(generators.size()));
    const auto raw_segment_digest = zkdl::cpu_ipa_basis_digest(
        std::vector<zkdl::CpuG1Point>(generators), h, u,
        fixture_padded_basis_size);
    const auto cached_segment_digest =
        zkdl::cpu_ipa_basis_digest(
            prover_basis, static_cast<std::uint32_t>(generators.size()),
            fixture_padded_basis_size);
    require(raw_segment_digest == cached_segment_digest,
            "legacy prefix basis digests", "prepared and raw basis digest differ");

    const std::vector<FrWire> aggregate_point{
        scalar(5U), scalar(7U), scalar(11U), scalar(13U)};
    const auto aggregate_probe = [&](
            const TensorWitness& tensor,
            const std::vector<CommitmentWire>& wires,
            const std::map<std::string, std::vector<FrWire>>* shares,
            const SegmentCommitmentValidationReceipt* receipt,
            std::vector<FrWire>& aggregate_values,
            FrWire& aggregate_blinder, FrWire& aggregate_value,
            zkdl::CpuG1Point& aggregate_xi, std::string& detail) {
        detail.clear();
        const auto commitment = std::find_if(
            wires.begin(), wires.end(), [&](const CommitmentWire& item) {
                return item.tensor_id == tensor.id;
            });
        if (commitment == wires.end()) {
            detail = "fixture aggregate commitment is missing";
            return false;
        }
        const auto commitment_index = static_cast<std::uint32_t>(
            std::distance(wires.begin(), commitment));
        return sparse_route_aggregate_opening_inputs(
            tensor, aggregate_point, generators, h, u, wires,
            commitment_index, blinders, shares, aggregate_values,
            aggregate_blinder, aggregate_value, aggregate_xi, &detail,
            nullptr, nullptr, &prover_basis, &prover_session, receipt);
    };
    std::vector<FrWire> aggregate_probe_values;
    FrWire aggregate_probe_blinder{}, aggregate_probe_value{};
    zkdl::CpuG1Point aggregate_probe_xi;
    require(aggregate_probe(view, commitments, &segment_blinders,
                            &receipts.at(view.id),
                            aggregate_probe_values, aggregate_probe_blinder,
                            aggregate_probe_value, aggregate_probe_xi, error),
            "FAC receipt aggregate path", error);
    require(aggregate_probe(view, commitments, &segment_blinders, nullptr,
                            aggregate_probe_values,
                            aggregate_probe_blinder, aggregate_probe_value,
                            aggregate_probe_xi, error),
            "FAC missing-receipt fallback", error);

    TensorWitness changed_view = view;
    changed_view.values[0U] = zkdl::fr_add(changed_view.values[0U], scalar(1U));
    require(!aggregate_probe(changed_view, commitments, &segment_blinders,
                             &receipts.at(view.id),
                             aggregate_probe_values, aggregate_probe_blinder,
                             aggregate_probe_value, aggregate_probe_xi, error),
            "FAC changed-witness receipt rejection",
            "receipt accepted a changed original segment value");

    auto changed_shares = segment_blinders;
    changed_shares.at(view.id)[0U] = zkdl::fr_add(
        changed_shares.at(view.id)[0U], scalar(1U));
    require(!aggregate_probe(view, commitments, &changed_shares,
                             &receipts.at(view.id),
                             aggregate_probe_values, aggregate_probe_blinder,
                             aggregate_probe_value, aggregate_probe_xi, error),
            "FAC changed-blinder receipt rejection",
            "receipt accepted a changed segment blinder/material");

    require(!aggregate_probe(view, commitments, &segment_blinders,
                             &receipts.at(canonical.id),
                             aggregate_probe_values, aggregate_probe_blinder,
                             aggregate_probe_value, aggregate_probe_xi, error),
            "FAC wrong-context receipt rejection",
            "receipt accepted a different tensor context");

    auto view_commitment = std::find_if(
        commitments.begin(), commitments.end(),
        [&](const CommitmentWire& item) { return item.tensor_id == view.id; });
    const auto view_commitment_index = static_cast<std::uint32_t>(
        std::distance(commitments.begin(), view_commitment));
    const auto wrong_u = zkdl::cpu_g1_mul(seed, scalar(127U));
    require(!sparse_route_aggregate_opening_inputs(
                view, aggregate_point, generators, h, wrong_u, commitments,
                view_commitment_index, blinders, &segment_blinders,
                aggregate_probe_values, aggregate_probe_blinder,
                aggregate_probe_value, aggregate_probe_xi, &error,
                nullptr, nullptr, &prover_basis, &prover_session,
                &receipts.at(view.id)),
            "FAC changed-basis receipt rejection",
            "receipt accepted a different H/U context");

    auto changed_xi_wires = commitments;
    auto changed_xi = std::find_if(
        changed_xi_wires.begin(), changed_xi_wires.end(),
        [&](const CommitmentWire& item) { return item.tensor_id == view.id; });
    changed_xi->segment_commitments[0U].commitment = zkdl::cpu_g1_to_native(
        zkdl::cpu_g1_add(
            zkdl::cpu_g1_from_native(
                changed_xi->segment_commitments[0U].commitment),
            generators[0U]));
    zkdl::CpuG1Point changed_xi_aggregate = zkdl::CpuG1Point::zero();
    for (const auto& segment : changed_xi->segment_commitments)
        changed_xi_aggregate = zkdl::cpu_g1_add(
            changed_xi_aggregate,
            zkdl::cpu_g1_from_native(segment.commitment));
    changed_xi->commitment = zkdl::cpu_g1_to_native(changed_xi_aggregate);
    changed_xi->segment_commitment_digest = segment_commitment_digest(*changed_xi);
    require(!aggregate_probe(view, changed_xi_wires, &segment_blinders,
                             &receipts.at(view.id),
                             aggregate_probe_values, aggregate_probe_blinder,
                             aggregate_probe_value, aggregate_probe_xi, error),
            "FAC changed-Xi receipt rejection",
            "receipt accepted a changed segment commitment");

    RelationStatement max_statement = statement;
    max_statement.segment_commitments_digest = {};
    std::vector<FrWire> public_output_values{
        scalar(17U), scalar(19U), scalar(23U), scalar(29U)};
    public_output_values.resize(16U, FrWire{});
    max_statement.public_output = encode_public_field_vector(public_output_values);
    const std::vector<std::string> max_refs{
        "layer1/max/argmax", "layer1/max/bit0", "layer1/max/bit1",
        "layer1/max/unused"};
    std::vector<FrWire> delta_values{
        scalar(0U), scalar(1U), scalar(2U), scalar(3U)};
    std::vector<FrWire> bit0_values{
        scalar(0U), scalar(1U), scalar(0U), scalar(1U)};
    std::vector<FrWire> bit1_values{
        scalar(0U), scalar(0U), scalar(1U), scalar(1U)};
    std::vector<FrWire> unused_values{
        scalar(31U), scalar(32U), scalar(33U), scalar(34U)};
    delta_values.resize(16U, FrWire{});
    bit0_values.resize(16U, FrWire{});
    bit1_values.resize(16U, FrWire{});
    unused_values.resize(16U, FrWire{});
    std::vector<zkdl::CpuG1Point> compact_generators;
    compact_generators.reserve(8U);
    for (std::uint32_t index = 0U; index < 8U; ++index)
        compact_generators.push_back(zkdl::cpu_g1_mul(seed, scalar(index + 2U)));
    const auto compact_h = zkdl::cpu_g1_mul(seed, scalar(103U));
    const auto compact_u = zkdl::cpu_g1_mul(seed, scalar(109U));
    ConstraintDescriptor boolean_constraint;
    boolean_constraint.id = "layer1/ordinary/compact-smoke/boolean-bit0";
    boolean_constraint.degree = 2U;
    boolean_constraint.terms = {
        Monomial{zkdl::fr_neg(scalar(1U)), {1U}},
        Monomial{scalar(1U), {1U, 1U}}};
    boolean_constraint.logical_shape = {16U};
    boolean_constraint.padded_shape = {16U};
    ConstraintDescriptor reconstruction_constraint;
    reconstruction_constraint.id = "layer1/ordinary/compact-smoke/reconstruct";
    reconstruction_constraint.degree = 1U;
    reconstruction_constraint.terms = {
        Monomial{zkdl::fr_neg(scalar(1U)), {0U}},
        Monomial{scalar(1U), {1U}},
        Monomial{scalar(2U), {2U}}};
    reconstruction_constraint.logical_shape = {16U};
    reconstruction_constraint.padded_shape = {16U};
    ConstraintDescriptor compact_boolean, compact_reconstruction;
    std::vector<std::string> compact_boolean_refs, compact_reconstruction_refs;
    require(compact_constraint_tensor_refs(
                boolean_constraint, max_refs, compact_boolean,
                compact_boolean_refs, &error),
            "ordinary boolean constraint compaction", error);
    require(compact_constraint_tensor_refs(
                reconstruction_constraint, max_refs, compact_reconstruction,
                compact_reconstruction_refs, &error),
            "ordinary reconstruction constraint compaction", error);
    require(compact_boolean_refs == std::vector<std::string>{max_refs[1U]} &&
                compact_boolean.terms.size() == 2U &&
                compact_boolean.terms[0U].refs == std::vector<std::uint32_t>{0U} &&
                compact_boolean.terms[1U].refs ==
                    std::vector<std::uint32_t>({0U, 0U}) &&
                compact_reconstruction_refs ==
                    std::vector<std::string>({max_refs[0U], max_refs[1U], max_refs[2U]}) &&
                compact_reconstruction.terms[0U].refs ==
                    std::vector<std::uint32_t>{0U} &&
                compact_reconstruction.terms[1U].refs ==
                    std::vector<std::uint32_t>{1U} &&
                compact_reconstruction.terms[2U].refs ==
                    std::vector<std::uint32_t>{2U},
            "ordinary constraint ascending compact maps",
            "compact refs did not preserve ascending original indices and duplicate factors");
    ConstraintDescriptor invalid_max_constraint = reconstruction_constraint;
    invalid_max_constraint.terms[0U].refs = {4U};
    error.clear();
    require(!compact_constraint_tensor_refs(
                invalid_max_constraint, max_refs, compact_reconstruction,
                compact_reconstruction_refs, &error),
            "ordinary constraint invalid reference rejection",
            "out-of-bounds original tensor reference was accepted");
    error.clear();
    const std::vector<TensorWitness> compact_tensors{
        TensorWitness{max_refs[0U], 3U, 16U, delta_values},
        TensorWitness{max_refs[1U], 3U, 16U, bit0_values},
        TensorWitness{max_refs[2U], 3U, 16U, bit1_values},
        TensorWitness{max_refs[3U], 3U, 16U, unused_values},
        TensorWitness{"public_Y", 1U, 16U, public_output_values}};
    const auto evaluate_constraint_slot = [&](
            const ConstraintDescriptor& value,
            const std::vector<std::string>& refs, std::uint32_t slot) {
        FrWire result = value.constant;
        for (const auto& term : value.terms) {
            FrWire product = term.coefficient;
            for (const std::uint32_t ref : term.refs) {
                const auto tensor = std::find_if(
                    compact_tensors.begin(), compact_tensors.end(),
                    [&](const TensorWitness& item) { return item.id == refs.at(ref); });
                if (tensor == compact_tensors.end())
                    throw std::runtime_error("ordinary slot-evaluation tensor is missing");
                product = zkdl::fr_mul(product, tensor->values.at(slot));
            }
            result = zkdl::fr_add(result, product);
        }
        return result;
    };
    for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
        require(evaluate_constraint_slot(boolean_constraint, max_refs, slot) ==
                    evaluate_constraint_slot(compact_boolean,
                                             compact_boolean_refs, slot) &&
                    evaluate_constraint_slot(reconstruction_constraint, max_refs, slot) ==
                    evaluate_constraint_slot(compact_reconstruction,
                                             compact_reconstruction_refs, slot),
                "ordinary constraint compact slot equivalence",
                "compact descriptor changed a pointwise relation value");
    }
    const std::array<std::pair<RelationKind, const char*>, 3U> compact_kinds{{
        {RelationKind::MaxSelector, "max_selector"},
        {RelationKind::GroupSum, "group_sum"},
        {RelationKind::Alpha, "alpha"}}};
    for (const auto& kind : compact_kinds) {
        OperationDescriptor operation;
        operation.id = std::string("layer1/") + kind.second + "/compact-smoke";
        operation.kind = kind.first;
        operation.degree = 2U;
        operation.source_refs = max_refs;
        operation.tensor_refs = max_refs;
        operation.mapping = "ordinary-constraint-compact-smoke";
        operation.padding = "fixed-zero-padding";
        operation.terminal_claim = "ordinary-constraint-smoke";
        operation.terminal_point = "ordinary-constraint-point";
        operation.constraints = {boolean_constraint, reconstruction_constraint};
        require(operation.valid(&error), "ordinary constraint descriptor", error);

        ProverContext prover_context;
        prover_context.tensors = compact_tensors;
        prover_context.statement = max_statement;
        prover_context.operations = {operation};
        prover_context.generators = compact_generators;
        require(zkdl::CpuIpaBasisContext::authenticate(
                    std::vector<zkdl::CpuG1Point>(compact_generators),
                    compact_h, compact_u,
                    prover_context.ipa_generators, &error),
                "ordinary constraint prover basis authentication", error);
        prover_context.ipa_blinding_generator = compact_h;
        prover_context.ipa_u_generator = compact_u;
        prover_context.blinding_scalars = {
            {max_refs[0U], scalar(201U)}, {max_refs[1U], scalar(203U)},
            {max_refs[2U], scalar(207U)}, {max_refs[3U], scalar(211U)}};

        error.clear();
        Proof compact_proof;
        require(prove(prover_context, compact_proof, &error),
                "ordinary constraint compact full prove", error);
        require(compact_proof.operations.size() == 1U &&
                    compact_proof.operations[0U].constraints.size() == 2U,
                "ordinary constraint compact proof shape",
                "proof has an unexpected operation/constraint count");
        const auto& proved_constraints = compact_proof.operations[0U].constraints;
        require(proved_constraints.size() == 2U &&
                    proved_constraints[0U].opening_tensor_indices ==
                        std::vector<std::uint32_t>{0U} &&
                    proved_constraints[0U].opening_commitment_indices ==
                        std::vector<std::uint32_t>{1U} &&
                    proved_constraints[1U].opening_tensor_indices ==
                        std::vector<std::uint32_t>({0U, 1U, 2U}) &&
                    proved_constraints[1U].opening_commitment_indices ==
                        std::vector<std::uint32_t>({0U, 1U, 2U}) &&
                    compact_proof.commitments.size() == 5U,
                "ordinary constraint compact proof openings",
                "proof did not retain all commitments and open exactly used tensor commitments");
        for (const auto& constraint_proof : proved_constraints) {
            const auto& terminal = constraint_proof.terminal_relation;
            require(terminal.version ==
                            zkdl::ZkTerminalRelationProof::kCrossProviderAggregateVersion &&
                        terminal.terminal_commitments.size() ==
                            constraint_proof.opening_commitment_indices.size() &&
                        terminal.terminal_openings.size() == 1U &&
                        terminal.terminal_segment_openings.size() == 1U &&
                        terminal.terminal_segment_openings.front().size() == 1U,
                    "ordinary terminal cross-provider aggregate",
                    "full ordinary Prove did not combine all two-segment providers into one opening");
        }
        const std::vector<zkdl::byte> compact_encoded = compact_proof.serialize();
        Proof compact_decoded;
        require(Proof::deserialize(compact_encoded, compact_decoded, &error) &&
                    compact_decoded.serialize() == compact_encoded,
                "ordinary constraint canonical proof wire", error);

        VerifierContext verifier_context;
        verifier_context.statement = max_statement;
        verifier_context.operations = {operation};
        verifier_context.tensor_ids = max_refs;
        verifier_context.tensor_ids.push_back("public_Y");
        for (std::size_t index = 0U; index < max_refs.size(); ++index)
            verifier_context.tensor_visibility.emplace(max_refs[index], 3U);
        verifier_context.tensor_visibility.emplace("public_Y", 1U);
        verifier_context.generators = compact_generators;
        require(zkdl::CpuIpaBasisContext::authenticate(
                    std::vector<zkdl::CpuG1Point>(compact_generators),
                    compact_h, compact_u,
                    verifier_context.ipa_generators, &error),
                "ordinary constraint independent verifier basis authentication", error);
        verifier_context.ipa_blinding_generator = compact_h;
        verifier_context.ipa_u_generator = compact_u;
        require(verify(compact_decoded, verifier_context, &error),
                "ordinary constraint compact independent Verify", error);

        const auto reject_tampered = [&](Proof tampered, const char* label) {
            error.clear();
            require(!verify(tampered, verifier_context, &error), label,
                    "fresh verifier accepted a modified v4 binding");
        };
        {
            Proof tampered = compact_decoded;
            auto& terminal = tampered.operations[0U].constraints[0U].terminal_relation;
            const auto changed = zkdl::cpu_g1_add(
                zkdl::cpu_g1_from_native(terminal.terminal_commitments[0U]), compact_h);
            terminal.terminal_commitments[0U] = zkdl::cpu_g1_to_native(changed);
            reject_tampered(std::move(tampered), "v4 provider tau tamper");
        }
        {
            Proof tampered = compact_decoded;
            auto& terminal = tampered.operations[0U].constraints[0U].terminal_relation;
            auto changed = terminal.terminal_openings.front();
            changed.outer_point.front() = zkdl::fr_value(zkdl::fr_add(
                zkdl::fr_wire(changed.outer_point.front()), zkdl::fr_from_u32(1U)));
            terminal.terminal_openings.front() = changed;
            terminal.terminal_segment_openings.front().front() = changed;
            reject_tampered(std::move(tampered), "v4 point tamper");
        }
        {
            Proof tampered = compact_decoded;
            auto& terminal = tampered.operations[0U].constraints[0U].terminal_relation;
            terminal.version = zkdl::ZkTerminalRelationProof::kAggregateVersion;
            reject_tampered(std::move(tampered), "v4 mode/version tamper");
        }
        {
            Proof tampered = compact_decoded;
            auto& source = tampered.commitments[
                tampered.operations[0U].constraints[0U].opening_commitment_indices[0U]];
            const auto changed = zkdl::cpu_g1_add(
                zkdl::cpu_g1_from_native(
                    source.segment_commitments.front().commitment), compact_h);
            source.segment_commitments.front().commitment =
                zkdl::cpu_g1_to_native(changed);
            zkdl::CpuG1Point canonical = zkdl::CpuG1Point::zero();
            for (const auto& segment : source.segment_commitments)
                canonical = zkdl::cpu_g1_add(
                    canonical, zkdl::cpu_g1_from_native(segment.commitment));
            source.commitment = zkdl::cpu_g1_to_native(canonical);
            source.segment_commitment_digest = segment_commitment_digest(source);
            tampered.statement.segment_commitments_digest =
                relation_commitment_vector_digest(tampered.statement,
                                                  tampered.commitments);
            reject_tampered(std::move(tampered), "v4 source Gamma tamper");
        }
        {
            Proof tampered = compact_decoded;
            auto& constraint = tampered.operations[0U].constraints[1U];
            std::reverse(constraint.opening_commitment_indices.begin(),
                         constraint.opening_commitment_indices.end());
            reject_tampered(std::move(tampered), "v4 provider index/order tamper");
        }
    }

    // Full hidden-matmul primitive proof: A, B, and C have distinct logical
    // domains and are all segmented by the four-generator authenticated
    // basis.  ZMH3 must keep all source segment commitments while emitting
    // exactly one weighted aggregate opening per role.
    const zkdl::MatMulShape aggregate_shape{1U, 2U, 4U, 8U};
    std::vector<FrWire> matmul_a_values{
        scalar(1U), scalar(2U), scalar(0U), scalar(1U),
        scalar(2U), scalar(0U), scalar(3U), scalar(1U)};
    std::vector<FrWire> matmul_b_values(32U);
    for (std::size_t index = 0U; index < matmul_b_values.size(); ++index)
        matmul_b_values[index] = scalar(static_cast<std::uint32_t>(index + 2U));
    std::vector<FrWire> matmul_c_values(16U, FrWire{});
    for (std::uint32_t row = 0U; row < aggregate_shape.m; ++row)
        for (std::uint32_t column = 0U; column < aggregate_shape.k; ++column)
            for (std::uint32_t inner = 0U; inner < aggregate_shape.n; ++inner)
                matmul_c_values[row * aggregate_shape.k + column] = zkdl::fr_add(
                    matmul_c_values[row * aggregate_shape.k + column],
                    zkdl::fr_mul(
                        matmul_a_values[row * aggregate_shape.n + inner],
                        matmul_b_values[inner * aggregate_shape.k + column]));
    TensorWitness matmul_a{"hidden/layer0/head0/matmul-A", 3U, 8U,
                           std::move(matmul_a_values)};
    TensorWitness matmul_b{"hidden/layer0/head0/matmul-B", 3U, 32U,
                           std::move(matmul_b_values)};
    TensorWitness matmul_c{"hidden/layer0/head0/matmul-C", 3U, 16U,
                           std::move(matmul_c_values)};
    const std::vector<std::string> matmul_refs{
        matmul_a.id, matmul_b.id, matmul_c.id};
    std::map<std::string, const TensorWitness*> matmul_witnesses{
        {matmul_a.id, &matmul_a}, {matmul_b.id, &matmul_b},
        {matmul_c.id, &matmul_c}};
    const std::map<std::string, FrWire> matmul_blinders{
        {matmul_a.id, scalar(151U)}, {matmul_b.id, scalar(157U)},
        {matmul_c.id, scalar(163U)}};
    RelationStatement matmul_statement;
    matmul_statement.statement_digest = zkdl::sha256(
        std::vector<zkdl::byte>{'z', 'm', 'h', '3', 's'});
    matmul_statement.model_digest = zkdl::sha256(
        std::vector<zkdl::byte>{'z', 'm', 'h', '3', 'm'});
    matmul_statement.table_digest = zkdl::sha256(
        std::vector<zkdl::byte>{'z', 'm', 'h', '3', 't'});
    matmul_statement.ledger_digest = zkdl::sha256(
        std::vector<zkdl::byte>{'z', 'm', 'h', '3', 'l'});
    matmul_statement.srs_digest = zkdl::sha256(
        std::vector<zkdl::byte>{'z', 'm', 'h', '3', 'p'});
    matmul_statement.registry_digest = zkdl::sha256(
        std::vector<zkdl::byte>{'z', 'm', 'h', '3', 'r'});
    std::vector<CommitmentWire> matmul_commitments;
    std::map<std::string, std::vector<FrWire>> matmul_segment_blinders;
    std::map<std::string, SegmentCommitmentValidationReceipt>
        matmul_receipts;
    for (const TensorWitness* tensor : {&matmul_a, &matmul_b, &matmul_c}) {
        SegmentedTensorProvider provider;
        SegmentCommitmentMaterial material;
        CommitmentWire commitment;
        SegmentCommitmentValidationReceipt receipt;
        require(SegmentedTensorProvider::from_values(
                    tensor->values, generators.size(), provider, &error),
                "ZMH3 segmented tensor provider", error);
        require(zktr_build_segment_commitment_material(
                    tensor->id, tensor->visibility, provider, generators,
                    h, u, matmul_blinders.at(tensor->id), matmul_statement,
                    commitment, material, &error, nullptr, 0U, nullptr,
                    &prover_basis, &prover_session, &receipt),
                "ZMH3 source commitments", error);
        const std::uint32_t expected_segments = tensor == &matmul_a ? 2U
            : tensor == &matmul_b ? 8U : 4U;
        require(commitment.segment_size == 4U &&
                    commitment.segment_count == expected_segments &&
                    receipt.valid(),
                "ZMH3 distinct segmented source shapes",
                "A/B/C commitment layouts did not match the fixture contract");
        matmul_segment_blinders.emplace(tensor->id,
                                        std::move(material.blinders));
        matmul_receipts.emplace(tensor->id, std::move(receipt));
        matmul_commitments.push_back(std::move(commitment));
    }
    matmul_statement.segment_commitments_digest =
        relation_commitment_vector_digest(matmul_statement,
                                          matmul_commitments);
    error.clear();
    require(relation_segment_commitments_valid(matmul_statement,
                                               matmul_commitments, &error),
            "ZMH3 ordered source commitment binding", error);
    OperationDescriptor matmul_operation;
    matmul_operation.id = "hidden/layer0/head0/projection/zmh3-smoke";
    matmul_operation.kind = RelationKind::Projection;
    matmul_operation.degree = 2U;
    matmul_operation.source_refs = matmul_refs;
    matmul_operation.tensor_refs = matmul_refs;
    matmul_operation.mapping = "hidden-matmul-zmh3-aggregate";
    matmul_operation.padding = "fixed-zero-padding";
    matmul_operation.terminal_claim = "hidden-matmul-zmh3-claim";
    matmul_operation.terminal_point = "hidden-matmul-zmh3-point";
    PrimitiveDescriptor matmul_primitive;
    matmul_primitive.kind = 2U;
    matmul_primitive.matmul_shape = aggregate_shape;
    matmul_primitive.tensor_refs = matmul_refs;
    matmul_operation.primitives = {matmul_primitive};
    zkdl::Transcript matmul_prover_transcript(
        "hidden-matmul-zmh3-aggregate-smoke");
    HiddenMatMulProof matmul_proof;
    error.clear();
    require(prove_hidden_matmul(
                matmul_operation, matmul_primitive, 0U, matmul_witnesses,
                matmul_blinders, generators, h, u, matmul_commitments,
                matmul_statement, matmul_prover_transcript, matmul_proof,
                &error, matmul_segment_blinders, &prover_basis,
                &prover_session, &matmul_receipts),
            "ZMH3 full hidden-matmul prove", error);
    require(matmul_proof.version == 3U &&
                matmul_proof.opening_a.segments.size() == 1U &&
                matmul_proof.opening_b.segments.size() == 1U &&
                matmul_proof.opening_c.segments.size() == 1U,
            "ZMH3 one weighted opening per role",
            "a source role still emitted one opening per segment");
    const std::vector<zkdl::byte> matmul_encoded = matmul_proof.serialize();
    require(matmul_encoded.size() >= 8U && matmul_encoded[0U] == 'Z' &&
                matmul_encoded[1U] == 'M' && matmul_encoded[2U] == 'H' &&
                matmul_encoded[3U] == '3',
            "ZMH3 mode wire tag", "serialized hidden matmul is not ZMH3");
    HiddenMatMulProof matmul_decoded;
    require(HiddenMatMulProof::deserialize(matmul_encoded, matmul_decoded,
                                           &error) &&
                matmul_decoded.serialize() == matmul_encoded,
            "ZMH3 canonical serialize/deserialize", error);
    zkdl::CpuIpaBasisContext matmul_verifier_basis;
    require(zkdl::CpuIpaBasisContext::authenticate(
                std::vector<zkdl::CpuG1Point>(generators), h, u,
                matmul_verifier_basis, &error),
            "ZMH3 fresh verifier basis", error);
    zkdl::CpuG1MsmSession matmul_verifier_session(matmul_verifier_basis);
    const auto verify_matmul_fresh = [&](const HiddenMatMulProof& candidate,
                                         const std::vector<CommitmentWire>& wires,
                                         std::string& detail) {
        detail.clear();
        if (!relation_segment_commitments_valid(matmul_statement, wires,
                                                &detail)) return false;
        zkdl::Transcript transcript("hidden-matmul-zmh3-aggregate-smoke");
        return verify_hidden_matmul(
            matmul_operation, matmul_primitive, 0U, candidate, wires,
            matmul_verifier_basis.generators(),
            matmul_verifier_basis.h(), matmul_verifier_basis.u(),
            matmul_statement, transcript, &detail,
            &matmul_verifier_basis, &matmul_verifier_session);
    };
    error.clear();
    require(verify_matmul_fresh(matmul_decoded, matmul_commitments, error),
            "ZMH3 fresh independent Verify", error);
    auto reject_matmul = [&](const HiddenMatMulProof& candidate,
                             const std::vector<CommitmentWire>& wires,
                             const char* label) {
        std::string detail;
        require(!verify_matmul_fresh(candidate, wires, detail), label,
                "tampered hidden matrix aggregate was accepted");
    };
    auto changed_source = matmul_a;
    changed_source.values[0U] = zkdl::fr_add(changed_source.values[0U],
                                             scalar(1U));
    auto changed_witnesses = matmul_witnesses;
    changed_witnesses[changed_source.id] = &changed_source;
    HiddenMatMulProof changed_source_proof;
    zkdl::Transcript changed_source_transcript(
        "hidden-matmul-zmh3-aggregate-smoke");
    error.clear();
    require(!prove_hidden_matmul(
                matmul_operation, matmul_primitive, 0U, changed_witnesses,
                matmul_blinders, generators, h, u, matmul_commitments,
                matmul_statement, changed_source_transcript,
                changed_source_proof, &error, matmul_segment_blinders,
                &prover_basis, &prover_session, &matmul_receipts),
            "ZMH3 changed source value rejection",
            "source values changed while their original commitments/receipts remained fixed");
    auto bad_xi = matmul_decoded;
    bad_xi.opening_a.segments[0U].xi =
        zkdl::cpu_g1_to_native(seed);
    reject_matmul(bad_xi, matmul_commitments,
                  "ZMH3 changed aggregate Xi rejection");
    auto bad_point = matmul_decoded;
    bad_point.opening_b.segments[0U].outer_point[0U] =
        zkdl::fr_value(zkdl::fr_add(
            zkdl::fr_wire(bad_point.opening_b.segments[0U].outer_point[0U]),
            scalar(1U)));
    reject_matmul(bad_point, matmul_commitments,
                  "ZMH3 changed opening point rejection");
    auto bad_tau = matmul_decoded;
    bad_tau.opening_c.tau = zkdl::cpu_g1_to_native(seed);
    reject_matmul(bad_tau, matmul_commitments,
                  "ZMH3 changed aggregate tau rejection");
    auto reordered_commitments = matmul_commitments;
    std::swap(reordered_commitments[0U].segment_commitments[0U],
              reordered_commitments[0U].segment_commitments[1U]);
    reject_matmul(matmul_decoded, reordered_commitments,
                  "ZMH3 reordered source segment rejection");
    auto legacy_mode = matmul_encoded;
    legacy_mode[3U] = '2';
    HiddenMatMulProof legacy_parsed;
    error.clear();
    require(!HiddenMatMulProof::deserialize(legacy_mode, legacy_parsed,
                                             &error),
            "ZMH3 legacy mode rejection",
            "ZMH2 proof bytes were accepted by the aggregate verifier");
    auto legacy_version = matmul_encoded;
    legacy_version[4U] = 2U;
    error.clear();
    require(!HiddenMatMulProof::deserialize(legacy_version, legacy_parsed,
                                             &error),
            "ZMH3 legacy version rejection",
            "nested version 2 was accepted by the v3 parser");

    static_assert(std::is_same<decltype(std::declval<SparseRouteProof>().source_opening),
                               zkdl::HyraxOpeningProofV3>::value,
                  "nested FAC selector must use hidden Hyrax openings");
    FacLine8Proof proof;
    zkdl::Transcript prover_transcript("fac-line8-cross-segment-smoke");
    require(prove_fac_line8(
                fac, witnesses, blinders, generators, h, u, commitments,
                statement, prover_transcript, proof, &error, &segment_blinders,
                &prover_basis, &prover_session, &receipts),
            "prove nested FAC sparse route", error);
    const std::vector<zkdl::byte> encoded = proof.serialize();
    FacLine8Proof decoded;
    require(FacLine8Proof::deserialize(encoded, decoded, &error) &&
                decoded.serialize() == encoded && decoded.groups.size() == 1U &&
                decoded.groups[0U].envelope.ordinal == 0U &&
                decoded.groups[0U].envelope.mode ==
                    FacLine8GroupEnvelope::kFeatureRlcAggregate &&
                decoded.groups[0U].envelope.member_count == 2U &&
                decoded.groups[0U].route.output_commitment_index == 1U &&
                encoded.size() >= 4U && encoded[3U] == '4',
            "canonical FAC wire", error);
    auto trailing_fac_bytes = encoded;
    trailing_fac_bytes.push_back(0U);
    FacLine8Proof trailing_fac;
    error.clear();
    require(!FacLine8Proof::deserialize(trailing_fac_bytes, trailing_fac,
                                         &error),
            "FAC4 trailing-byte rejection",
            "obsolete aggregate-binding/trailing bytes were accepted");

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

    const auto fac_candidate_verifies = [&](
            const OperationDescriptor& operation,
            const std::vector<CommitmentWire>& wires,
            const FacLine8Proof& candidate) {
        std::string detail;
        zkdl::Transcript candidate_transcript(
            "fac-line8-cross-segment-smoke");
        return verify_fac_line8(
            operation, wires, verifier_basis.generators(), h, u, statement,
            candidate_transcript, candidate, &detail, &verifier_basis,
            &verifier_session);
    };
    auto changed_map_fac = fac;
    changed_map_fac.fac_bindings[1U].source_feature_index = {0U, 1U, 2U, 3U};
    require(!fac_candidate_verifies(changed_map_fac, commitments, decoded),
            "FAC feature-map tamper rejection",
            "changed authenticated feature map was accepted");
    auto reordered_fac = fac;
    std::swap(reordered_fac.fac_bindings[0U],
              reordered_fac.fac_bindings[1U]);
    require(!reordered_fac.valid(&error),
            "FAC canonical alpha-member order rejection",
            "noncanonical member order was accepted");
    auto changed_source_gamma = commitments;
    changed_source_gamma[0U].commitment =
        zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(
            zkdl::cpu_g1_from_native(changed_source_gamma[0U].commitment),
            generators[0U]));
    require(!fac_candidate_verifies(fac, changed_source_gamma, decoded),
            "FAC source Gamma tamper rejection",
            "changed canonical source Gamma was accepted");
    auto changed_target_gamma = commitments;
    changed_target_gamma[2U].segment_commitments[0U].commitment =
        zkdl::cpu_g1_to_native(zkdl::cpu_g1_add(
            zkdl::cpu_g1_from_native(
                changed_target_gamma[2U].segment_commitments[0U].commitment),
            generators[0U]));
    require(!fac_candidate_verifies(fac, changed_target_gamma, decoded),
            "FAC non-anchor target Gamma tamper rejection",
            "changed second-member target Gamma was ignored");
    auto changed_point_fac = decoded;
    changed_point_fac.groups[0U].route.feature_point[0U] =
        zkdl::fr_value(zkdl::fr_add(
            zkdl::fr_wire(changed_point_fac.groups[0U].route.feature_point[0U]),
            scalar(1U)));
    require(!fac_candidate_verifies(fac, commitments, changed_point_fac),
            "FAC feature-RLC point tamper rejection",
            "changed grouped feature point was accepted");
    auto changed_mode_fac = decoded;
    changed_mode_fac.groups[0U].envelope.mode =
        FacLine8GroupEnvelope::kOriginalBinding;
    require(!fac_candidate_verifies(fac, commitments, changed_mode_fac),
            "FAC FeatureRlcAggregate mode tamper rejection",
            "single-route mode accepted a multi-member route");

    decoded.groups[0U].route.output_opening.z1.val[0] ^= 1U;
    zkdl::Transcript tampered_transcript("fac-line8-cross-segment-smoke");
    require(!verify_fac_line8(
                fac, commitments, verifier_basis.generators(), h, u,
                statement, tampered_transcript, decoded, &error,
                &verifier_basis, &verifier_session),
            "FAC opening tamper rejection", "tampered opening was accepted");

    std::cout << "sparse_route_hidden_smoke PASS nested_FAC=1 shared_source=1"
              << " view_domain=16 segment_size=4 segment_count=4 basis=4"
              << " receipt=PASS missing_receipt_fallback=PASS"
              << " changed_values=REJECTED changed_blinder=REJECTED"
              << " wrong_context=REJECTED changed_xi=REJECTED"
              << " fresh_verify=PASS tamper=REJECTED full_datasets=0\n";
    return 0;
}
