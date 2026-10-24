#include "zkrelation-proof.cuh"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace zkdl_gat::relation;
using byte = zkdl::byte;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void append_legacy_index_map(std::vector<byte>& out,
                             const IndexMapU32& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) zkdl::append_u32(out, value);
}

void append_legacy_operation(std::vector<byte>& out,
                             const OperationDescriptor& descriptor) {
    zkdl::append_string(out, descriptor.id);
    zkdl::append_u32(out, static_cast<std::uint32_t>(descriptor.kind));
    zkdl::append_u32(out, descriptor.degree);
    append_string_vector(out, descriptor.source_refs);
    append_string_vector(out, descriptor.tensor_refs);
    append_string_vector(out, descriptor.selector_refs);
    zkdl::append_string(out, descriptor.mapping);
    zkdl::append_string(out, descriptor.padding);
    zkdl::append_string(out, descriptor.terminal_claim);
    zkdl::append_string(out, descriptor.terminal_point);
    zkdl::append_u32(out, descriptor.public_route ? 1U : 0U);
    zkdl::append_u32(out, descriptor.requires_reindex ? 1U : 0U);
    append_fr_vector(out, descriptor.constants);
    zkdl::append_u32(out, descriptor.selector_edge_domain);
    zkdl::append_u32(out, descriptor.selector_node_domain);
    zkdl::append_u32(out, descriptor.selector_feature_domain);
    zkdl::append_u32(out, descriptor.selector_source_feature_domain);
    append_legacy_index_map(out, descriptor.selector_src);
    append_legacy_index_map(out, descriptor.selector_dst);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.selector_active.size()));
    for (const auto value : descriptor.selector_active) zkdl::append_u32(out, value);
    append_legacy_index_map(out, descriptor.selector_feature_map);
    zkdl::append_string(out, descriptor.selector_source_ref);
    zkdl::append_string(out, descriptor.selector_output_ref);
    zkdl::append_u32(out, descriptor.route_by_destination ? 1U : 0U);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.fac_bindings.size()));
    for (const auto& binding : descriptor.fac_bindings) {
        zkdl::append_string(out, binding.view_ref);
        zkdl::append_string(out, binding.canonical_ref);
        zkdl::append_u32(out, binding.view_outer_domain);
        zkdl::append_u32(out, binding.canonical_outer_domain);
        zkdl::append_u32(out, binding.view_feature_domain);
        zkdl::append_u32(out, binding.canonical_feature_domain);
        append_legacy_index_map(out, binding.source_outer_index);
        append_legacy_index_map(out, binding.source_feature_index);
        zkdl::append_u64(out, static_cast<std::uint64_t>(binding.selector_active.size()));
        for (const auto value : binding.selector_active) zkdl::append_u32(out, value);
    }
    zkdl::append_string(out, descriptor.batch_family);
    zkdl::append_string(out, descriptor.public_table_ref);
    append_fr_vector(out, descriptor.public_table_selector);
    zkdl::append_u32(out, descriptor.has_public_divisor ? 1U : 0U);
    append_fr_wire(out, descriptor.public_divisor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.primitives.size()));
    for (const auto& primitive : descriptor.primitives) append_primitive(out, primitive);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.constraints.size()));
    for (const auto& constraint : descriptor.constraints) append_constraint(out, constraint);
}

std::vector<byte> legacy_serialize_small_proof(const Proof& proof) {
    // This deliberately retains the pre-counting outer Proof::serialize
    // layout as a small byte oracle.  The fixture below has no nested proof
    // bodies, so this path only materializes a few hundred bytes.
    std::vector<byte> out;
    out.insert(out.end(), {'G', 'A', 'T', 'R'});
    zkdl::append_u32(out, proof.version);
    append_statement(out, proof.statement);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.commitments.size()));
    for (const auto& commitment : proof.commitments) append_commitment(out, commitment);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.padding_records.size()));
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.hiding_commitment_records.size()));
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.operations.size()));
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.sorted_lookup_proofs.size()));
    zkdl::append_u32(out, proof.public_output_commitment);
    serialize_hidden_matmul_opening(proof.public_output_opening, out);
    std::vector<byte> public_binding;
    zkdl::serialize_pedersen_scalar_value(proof.public_output_binding,
                                          public_binding);
    zkdl::append_u64(out, static_cast<std::uint64_t>(public_binding.size()));
    out.insert(out.end(), public_binding.begin(), public_binding.end());
    append_digest(out, proof.transcript_digest);
    return out;
}

Proof small_proof_wire_fixture() {
    Proof proof;
    const G1Jacobian_t infinity =
        zkdl::cpu_g1_to_native(zkdl::CpuG1Point::zero());
    CommitmentWire commitment;
    commitment.tensor_id = "tiny";
    commitment.domain_size = 1U;
    commitment.commitment = infinity;
    commitment.segment_size = 1U;
    commitment.segment_count = 1U;
    SegmentCommitmentWire segment;
    segment.actual_length = 1U;
    segment.padded_length = 1U;
    segment.commitment = infinity;
    commitment.segment_commitments.push_back(segment);
    proof.commitments.push_back(std::move(commitment));

    zkdl::HyraxOpeningProofV3 segment_opening;
    segment_opening.commitment_index = 0U;
    segment_opening.evaluation_commitment_index = 1U;
    segment_opening.generator_size = 1U;
    segment_opening.padded_generator_size = 2U;
    segment_opening.rounds.assign(2U, infinity);
    segment_opening.xi = infinity;
    segment_opening.tau = infinity;
    segment_opening.blinding_generator = infinity;
    segment_opening.u_generator = infinity;
    segment_opening.delta = infinity;
    segment_opening.beta = infinity;
    segment_opening.descriptor_digest[0U] = 1U;
    segment_opening.registry_digest[0U] = 1U;
    segment_opening.srs_digest[0U] = 1U;
    proof.public_output_opening.tau = infinity;
    proof.public_output_opening.segments.push_back(std::move(segment_opening));

    auto& binding = proof.public_output_binding;
    binding.context.commitment_index = 0U;
    binding.context.operation =
        zkdl::PedersenProofContext::kOperationScalarValue;
    binding.context.commitment_a_index = 0U;
    binding.context.degree = 1U;
    binding.context.descriptor_digest[0U] = 1U;
    binding.context.registry_digest[0U] = 1U;
    binding.context.srs_digest[0U] = 1U;
    binding.commitment = infinity;
    binding.schnorr_R = infinity;
    return proof;
}

void check_proof_wire_sizing() {
    Proof fixture = small_proof_wire_fixture();
    const std::vector<byte> legacy = legacy_serialize_small_proof(fixture);
    const std::vector<byte> current = fixture.serialize();
    require(fixture.serialized_size() == legacy.size() && current == legacy,
            "exact-size writer changed legacy proof bytes or counted size");

    std::vector<byte> bounded;
    bounded.reserve(3U);
    const std::size_t reserved_capacity = bounded.capacity();
    zkdl::BoundedWireWriter writer(bounded, 3U);
    writer.push_back(1U);
    writer.push_back(2U);
    writer.push_back(3U);
    bool writer_rejected_overflow = false;
    try {
        writer.push_back(4U);
    } catch (const std::length_error&) {
        writer_rejected_overflow = true;
    }
    require(writer_rejected_overflow && bounded.size() == 3U &&
                bounded.capacity() == reserved_capacity,
            "bounded proof writer grew or accepted bytes beyond its counted size");

    zkdl::WireSizeCounter overflow_counter;
    overflow_counter.add(std::numeric_limits<std::uint64_t>::max());
    bool counter_rejected_overflow = false;
    try {
        overflow_counter.add(1U);
    } catch (const std::overflow_error&) {
        counter_rejected_overflow = true;
    }
    require(counter_rejected_overflow,
            "proof wire counter accepted uint64 size overflow");

    const auto huge_identity = IndexMapU32::identity(std::size_t{1U} << 27U);
    zkdl::WireSizeCounter identity_wire_size;
    append_index_map(identity_wire_size, huge_identity);
    require(identity_wire_size.size() == 8U + (std::uint64_t{1U} << 27U) * 4U &&
                huge_identity.resident_bytes() == 0U,
            "identity map wire size is not O(1) without materializing its entries");
}

void check_index_map_wire_and_capacity() {
    constexpr std::uint32_t kLargeMap = 8192U;
    auto huge_identity = IndexMapU32::identity(1U << 27U);
    require(huge_identity.is_identity() && huge_identity.resident_bytes() == 0U,
            "large identity map is not O(1) resident storage");
    const std::uint32_t identity_read = huge_identity[(1U << 27U) - 1U];
    require(identity_read == (1U << 27U) - 1U && huge_identity.resident_bytes() == 0U,
            "mutable identity read materialized its backing map");

    auto identity_original = IndexMapU32::identity(8U);
    auto identity_copy = identity_original;
    identity_copy[3U] = 19U;
    require(identity_original.is_identity() && identity_original[3U] == 3U &&
                identity_original.resident_bytes() == 0U && identity_copy[3U] == 19U,
            "identity copy-on-write changed the original map");

    std::vector<std::uint32_t> explicit_values(kLargeMap);
    for (std::uint32_t index = 0U; index < kLargeMap; ++index)
        explicit_values[index] = index ^ 1U;
    IndexMapU32 explicit_original(std::move(explicit_values));
    auto explicit_copy = explicit_original;
    const std::uint64_t original_capacity = explicit_original.resident_bytes();
    explicit_copy[0U] = 17U;
    require(explicit_original[0U] == 1U && explicit_copy[0U] == 17U &&
                explicit_original.resident_bytes() == original_capacity,
            "explicit map copy-on-write changed the original map");

    OperationDescriptor descriptor;
    descriptor.id = "fixture/index-map-wire";
    descriptor.kind = RelationKind::SparseRoute;
    descriptor.degree = 2U;
    descriptor.mapping = "public-selector-mle:src-dst";
    descriptor.padding = "fixed-zero-padding;public-equality-selector-mle;fac4dnn-line8";
    descriptor.terminal_claim = "output";
    descriptor.terminal_point = "output";
    descriptor.public_route = true;
    descriptor.requires_reindex = true;
    descriptor.selector_src = IndexMapU32::identity(kLargeMap);
    descriptor.selector_dst = explicit_original;
    descriptor.selector_active.assign(kLargeMap, 1U);
    descriptor.selector_feature_map = IndexMapU32::identity(kLargeMap);
    descriptor.selector_edge_domain = kLargeMap;
    descriptor.selector_node_domain = kLargeMap;
    descriptor.selector_feature_domain = kLargeMap;
    descriptor.selector_source_feature_domain = kLargeMap;
    descriptor.selector_source_ref = "source";
    descriptor.selector_output_ref = "output";
    descriptor.source_refs = {"source", "output", "view"};
    descriptor.tensor_refs = {"output", "view"};
    ConstraintDescriptor route_constraint;
    route_constraint.id = "route/shape";
    route_constraint.degree = 1U;
    route_constraint.terms.push_back(
        Monomial{zkdl::fr_from_u32(1U), {0U}});
    route_constraint.logical_shape = {1U};
    route_constraint.padded_shape = {1U};
    descriptor.constraints.push_back(std::move(route_constraint));
    FacBindingDescriptor binding;
    binding.view_ref = "view";
    binding.canonical_ref = "source";
    binding.view_outer_domain = kLargeMap;
    binding.canonical_outer_domain = kLargeMap;
    binding.view_feature_domain = 1U;
    binding.canonical_feature_domain = 1U;
    binding.source_outer_index = IndexMapU32::identity(kLargeMap);
    binding.source_feature_index = IndexMapU32::identity(1U);
    binding.selector_active.assign(kLargeMap, 1U);
    descriptor.fac_bindings.push_back(std::move(binding));

    std::vector<byte> strict_operation_wire;
    std::string strict_error;
    require(descriptor.valid(&strict_error),
            "large-map operation fixture is not canonical: " + strict_error);
    append_operation(strict_operation_wire, descriptor);
    std::size_t strict_operation_offset = 0U;
    OperationDescriptor decoded_operation;
    require(read_operation_strict(strict_operation_wire,
                                  strict_operation_offset,
                                  decoded_operation) &&
                strict_operation_offset == strict_operation_wire.size() &&
                decoded_operation.selector_src.is_identity() &&
                decoded_operation.fac_bindings.size() == 1U &&
                decoded_operation.fac_bindings[0U].source_outer_index.is_identity(),
            "strict operation reader rejected legal selector/FAC maps above 4096: " +
                strict_error);

    OperationDescriptor empty_route;
    empty_route.id = "hidden/layer0/head0/sparse_route";
    empty_route.kind = RelationKind::SparseRoute;
    empty_route.degree = 2U;
    empty_route.source_refs = {"sparse_route", "source", "output"};
    empty_route.tensor_refs = {"output"};
    empty_route.mapping = "public-selector-mle:src-dst";
    empty_route.padding = "fixed-zero-padding;public-equality-selector-mle;fac4dnn-line8";
    empty_route.terminal_claim = "output";
    empty_route.terminal_point = "output";
    empty_route.public_route = true;
    empty_route.requires_reindex = true;
    empty_route.selector_edge_domain = 4U;
    empty_route.selector_node_domain = 4U;
    empty_route.selector_feature_domain = 2U;
    empty_route.selector_source_feature_domain = 2U;
    empty_route.selector_src = {0U, 1U, 0U, 0U};
    empty_route.selector_dst = {1U, 0U, 0U, 0U};
    empty_route.selector_active = {1U, 1U, 0U, 0U};
    empty_route.selector_feature_map = {0U, 1U};
    empty_route.selector_source_ref = "source";
    empty_route.selector_output_ref = "output";
    require(empty_route.valid(&strict_error),
            "canonical empty-body SparseRoute descriptor rejected: " + strict_error);
    std::vector<byte> empty_route_wire;
    append_operation(empty_route_wire, empty_route);
    std::size_t empty_route_offset = 0U;
    OperationDescriptor decoded_empty_route;
    require(read_operation_strict(empty_route_wire, empty_route_offset,
                                  decoded_empty_route) &&
                empty_route_offset == empty_route_wire.size() &&
                decoded_empty_route.constraints.empty() &&
                decoded_empty_route.primitives.empty(),
            "strict reader rejected a canonical empty-body SparseRoute: " +
                strict_error);

    auto out_of_bounds_route = empty_route;
    out_of_bounds_route.selector_src[0U] = out_of_bounds_route.selector_node_domain;
    std::vector<byte> out_of_bounds_route_wire;
    append_operation(out_of_bounds_route_wire, out_of_bounds_route);
    std::size_t out_of_bounds_route_offset = 0U;
    OperationDescriptor decoded_out_of_bounds_route;
    require(!read_operation_strict(out_of_bounds_route_wire,
                                   out_of_bounds_route_offset,
                                   decoded_out_of_bounds_route),
            "strict reader accepted a route index outside the node domain");

    OperationDescriptor ordinary_empty;
    ordinary_empty.id = "fixture/ordinary-empty";
    ordinary_empty.kind = RelationKind::Projection;
    ordinary_empty.degree = 1U;
    ordinary_empty.source_refs = {"source"};
    ordinary_empty.tensor_refs = {"output"};
    ordinary_empty.mapping = "ordinary";
    ordinary_empty.padding = "fixed-zero-padding";
    ordinary_empty.terminal_claim = "output";
    ordinary_empty.terminal_point = "output";
    std::vector<byte> ordinary_empty_wire;
    append_operation(ordinary_empty_wire, ordinary_empty);
    std::size_t ordinary_empty_offset = 0U;
    OperationDescriptor decoded_ordinary_empty;
    require(!read_operation_strict(ordinary_empty_wire,
                                   ordinary_empty_offset,
                                   decoded_ordinary_empty),
            "strict reader accepted an ordinary empty-body descriptor");

    std::vector<byte> current_wire, legacy_wire;
    append_operation(current_wire, descriptor);
    append_legacy_operation(legacy_wire, descriptor);
    require(current_wire == legacy_wire &&
                zkdl::sha256(current_wire) == zkdl::sha256(legacy_wire),
            "compact maps changed legacy descriptor bytes/digest");

    std::vector<byte> encoded_identity;
    zkdl::append_u64(encoded_identity, kLargeMap);
    for (std::uint32_t index = 0U; index < kLargeMap; ++index)
        zkdl::append_u32(encoded_identity, index);
    std::size_t offset = 0U;
    IndexMapU32 decoded_identity;
    require(read_index_map_exact(encoded_identity, offset, kLargeMap,
                                 decoded_identity) &&
                offset == encoded_identity.size() && decoded_identity.is_identity() &&
                decoded_identity.resident_bytes() == 0U,
            "large legacy identity map failed strict decoding");
    std::vector<byte> reencoded_identity;
    append_legacy_index_map(reencoded_identity, decoded_identity);
    require(reencoded_identity == encoded_identity,
            "decoded large identity map changed legacy wire bytes");

    std::vector<byte> encoded_explicit;
    zkdl::append_u64(encoded_explicit, kLargeMap);
    for (std::uint32_t index = 0U; index < kLargeMap; ++index)
        zkdl::append_u32(encoded_explicit, index ^ 1U);
    offset = 0U;
    IndexMapU32 decoded_explicit;
    require(read_index_map_exact(encoded_explicit, offset, kLargeMap,
                                 decoded_explicit) &&
                offset == encoded_explicit.size() && !decoded_explicit.is_identity() &&
                decoded_explicit.size() == kLargeMap,
            "large explicit map failed strict decoding");
    std::vector<byte> reencoded_explicit;
    append_legacy_index_map(reencoded_explicit, decoded_explicit);
    require(reencoded_explicit == encoded_explicit,
            "decoded large explicit map changed legacy wire bytes");

    std::vector<byte> truncated = encoded_identity;
    truncated.pop_back();
    offset = 0U;
    require(!read_index_map_exact(truncated, offset, kLargeMap, decoded_identity),
            "map reader accepted a truncated payload");
    offset = 0U;
    require(!read_index_map_exact(encoded_identity, offset, kLargeMap - 1U,
                                  decoded_identity),
            "map reader accepted a count beyond its declared shape");
    std::vector<byte> over_domain;
    zkdl::append_u64(over_domain, (std::uint64_t{1} << 30U) + 1U);
    offset = 0U;
    require(!read_index_map_exact(over_domain, offset,
                                  static_cast<std::uint32_t>((std::uint64_t{1} << 30U) + 1U),
                                  decoded_identity),
            "map reader accepted a count above the relation domain bound");
}

OperationDescriptor canonical_sorted_lookup() {
    OperationDescriptor operation;
    operation.id = "hidden/layer0/head0/leaky_relu";
    operation.kind = RelationKind::TableLookup;
    operation.degree = 2U;
    operation.mapping = "operation_view_v1->canonical_mle:" + operation.id;
    operation.padding = "fixed-zero-padding;authenticated-sorted-lookup-v1";
    operation.terminal_claim = "terminal_claim_v1|lookup";
    operation.terminal_point = "FS:operation:" + operation.id;
    operation.requires_reindex = true;
    operation.batch_family = "authenticated_sorted_lookup";
    operation.public_table_ref = "static_lrelu";
    operation.source_refs = {
        "hidden/layer0/head0/trace/score",
        "static_lrelu",
        "hidden/layer0/head0/trace/lrelu_idx",
        "hidden/layer0/head0/trace/lrelu_bits",
        "hidden/layer0/head0/trace/leaky_relu",
        operation.id + "/view/query",
        operation.id + "/view/index",
        operation.id + "/view/output0",
    };
    operation.tensor_refs = {
        operation.id + "/view/query",
        operation.id + "/view/index",
        operation.id + "/view/output0",
    };

    const auto add_identity_binding = [&](const std::string& view, const std::string& canonical) {
        FacBindingDescriptor binding;
        binding.view_ref = view;
        binding.canonical_ref = canonical;
        binding.view_outer_domain = 2U;
        binding.canonical_outer_domain = 2U;
        binding.view_feature_domain = 2U;
        binding.canonical_feature_domain = 2U;
        binding.source_outer_index = {0U, 1U};
        binding.source_feature_index = {0U, 1U};
        binding.selector_active = {1U, 1U};
        operation.fac_bindings.push_back(std::move(binding));
    };
    add_identity_binding(operation.id + "/view/query", operation.source_refs[0U]);
    add_identity_binding(operation.id + "/view/index", operation.source_refs[2U]);
    add_identity_binding(operation.id + "/view/output0", operation.source_refs[4U]);
    std::sort(operation.fac_bindings.begin(), operation.fac_bindings.end(),
              [](const FacBindingDescriptor& lhs, const FacBindingDescriptor& rhs) {
                  return std::pair<std::string, std::string>(lhs.view_ref, lhs.canonical_ref) <
                         std::pair<std::string, std::string>(rhs.view_ref, rhs.canonical_ref);
              });
    return operation;
}

void require_rejected(const OperationDescriptor& operation, const std::string& label) {
    require(!operation.authenticated_sorted_lookup_body_valid(), label + " passed specialized body validation");
    std::string error;
    require(!operation.valid(&error), label + " passed complete descriptor validation");
}

}  // namespace

int main() {
    try {
        check_index_map_wire_and_capacity();
        check_proof_wire_sizing();
        const OperationDescriptor canonical = canonical_sorted_lookup();
        require(canonical.authenticated_sorted_lookup_body_valid(), "canonical specialized body rejected");
        std::string error;
        require(canonical.valid(&error), "canonical specialized descriptor rejected: " + error);

        auto wrong_family = canonical;
        wrong_family.batch_family = "unknown_lookup_family";
        require_rejected(wrong_family, "wrong family");

        auto wrong_kind = canonical;
        wrong_kind.kind = RelationKind::Projection;
        require_rejected(wrong_kind, "wrong relation kind");

        auto missing_table_source = canonical;
        missing_table_source.source_refs.erase(missing_table_source.source_refs.begin() + 1);
        require_rejected(missing_table_source, "missing public table source");

        auto table_as_tensor = canonical;
        table_as_tensor.tensor_refs[0U] = table_as_tensor.public_table_ref;
        require_rejected(table_as_tensor, "public table used as witness tensor");

        auto missing_fac = canonical;
        missing_fac.fac_bindings.pop_back();
        require_rejected(missing_fac, "missing identity FAC");

        auto wrong_fac_map = canonical;
        wrong_fac_map.fac_bindings[0U].source_outer_index = {1U, 0U};
        require_rejected(wrong_fac_map, "non-identity FAC map");

        auto wrong_fac_target = canonical;
        const auto query_binding = std::find_if(
            wrong_fac_target.fac_bindings.begin(), wrong_fac_target.fac_bindings.end(),
            [&](const FacBindingDescriptor& binding) {
                return binding.view_ref == wrong_fac_target.id + "/view/query";
            });
        require(query_binding != wrong_fac_target.fac_bindings.end(), "query FAC binding absent in test fixture");
        query_binding->canonical_ref = wrong_fac_target.source_refs[2U];
        require_rejected(wrong_fac_target, "wrong FAC canonical target");

        auto wrong_view = canonical;
        wrong_view.fac_bindings[0U].view_ref = "foreign/view";
        require_rejected(wrong_view, "wrong FAC view");

        std::cout << "authenticated_sorted_lookup_descriptor=PASS positive=1 negative=8\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "authenticated_sorted_lookup_descriptor=FAIL error=" << exception.what() << "\n";
        return 1;
    }
}
