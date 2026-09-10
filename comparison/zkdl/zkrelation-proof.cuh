#ifndef ZKDL_RELATION_PROOF_CUH
#define ZKDL_RELATION_PROOF_CUH

/*
 * Verifier-facing wire and host implementation for the GAT relation layer.
 *
 * This is intentionally a thin extension of the proof stack already used by
 * zkMatMul/zkProd: field values are FrWire, Fiat--Shamir is proof.cuh's
 * Transcript, the relation proof is proof.cuh's SumcheckProof and every
 * private tensor opening is cpu-backend.cuh's Hyrax opening.  There is no
 * second commitment, field, or transcript implementation here.
 *
 * The relation descriptor is supplied by the authenticated relation plan on
 * both sides.  A serialized proof repeats it only as an auditable wire and
 * the verifier compares the repeated descriptor byte-for-byte with the
 * expected descriptor before using any claim.  Consequently a prover cannot
 * change (for example) a product relation into a copy relation by editing
 * proof metadata.
 */

#include "cpu-backend.cuh"
#include "zkmatmul.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zkdl_gat {
namespace relation {

using byte = zkdl::byte;
using FrWire = zkdl::FrWire;

// Relation-family wire caps are deliberately finite.  They are large enough
// for the configured lookup/range domains, but never become an unbounded
// parser escape hatch for malformed proof bytes.
constexpr std::size_t kMaxRelationConstraints = 4096U;
constexpr std::size_t kMaxRelationTerms = 1U << 16;
constexpr std::size_t kMaxRelationOpenings = 4096U;
// Native tensor domains are admitted at most up to 2^30.  Point vectors are
// consequently bounded to the corresponding number of MLE coordinates; this
// is deliberately separate from the much larger opening-count cap above.
constexpr std::size_t kMaxRelationPointBits = 30U;

// The native CPU backend keeps transcript/opening claims as Fr_t while the
// relation wire format uses FrWire.  Compare through the canonical wire
// representation so mixed verifier checks do not depend on an implicit
// conversion that the field implementation intentionally does not provide.
inline int fr_compare_mixed(const Fr_t& a, const Fr_t& b) noexcept {
    return zkdl::fr_compare(zkdl::fr_wire(a), zkdl::fr_wire(b));
}

inline int fr_compare_mixed(const Fr_t& a, const FrWire& b) noexcept {
    return zkdl::fr_compare(zkdl::fr_wire(a), b);
}

inline int fr_compare_mixed(const FrWire& a, const Fr_t& b) noexcept {
    return zkdl::fr_compare(a, zkdl::fr_wire(b));
}

enum class RelationKind : std::uint32_t {
    Projection = 1U,
    AttentionInnerProduct = 2U,
    Product = 3U,
    SparseRoute = 4U,
    Reindex = 5U,
    Add = 6U,
    Bias = 7U,
    GroupSum = 8U,
    TableLookup = 9U,
    MaxSelector = 10U,
    RangeBits = 11U,
    Exp = 12U,
    LeakyRelu = 13U,
    Elu = 14U,
    Inverse = 15U,
    Alpha = 16U,
    FieldIntegerConsistency = 17U,
    SignedRoundDiv = 18U,
    Concat = 19U,
    HeadSum = 20U,
    FinalAverage = 21U,
    PublicOutput = 22U,
    Boolean = 23U,
    Reconstruction = 24U
};

inline const char* relation_kind_name(RelationKind kind) noexcept {
    switch (kind) {
        case RelationKind::Projection: return "projection";
        case RelationKind::AttentionInnerProduct: return "attention_inner_product";
        case RelationKind::Product: return "product";
        case RelationKind::SparseRoute: return "sparse_route";
        case RelationKind::Reindex: return "reindex";
        case RelationKind::Add: return "add";
        case RelationKind::Bias: return "bias";
        case RelationKind::GroupSum: return "group_sum";
        case RelationKind::TableLookup: return "table_lookup";
        case RelationKind::MaxSelector: return "max_selector";
        case RelationKind::RangeBits: return "range_bits";
        case RelationKind::Exp: return "exp";
        case RelationKind::LeakyRelu: return "leaky_relu";
        case RelationKind::Elu: return "elu";
        case RelationKind::Inverse: return "inverse";
        case RelationKind::Alpha: return "alpha";
        case RelationKind::FieldIntegerConsistency: return "field_integer_consistency";
        case RelationKind::SignedRoundDiv: return "signed_round_div";
        case RelationKind::Concat: return "concat";
        case RelationKind::HeadSum: return "head_sum";
        case RelationKind::FinalAverage: return "final_average";
        case RelationKind::PublicOutput: return "public_output";
        case RelationKind::Boolean: return "boolean";
        case RelationKind::Reconstruction: return "reconstruction";
    }
    return "unknown";
}

inline bool relation_kind_valid(std::uint32_t value) noexcept {
    return value >= static_cast<std::uint32_t>(RelationKind::Projection) &&
           value <= static_cast<std::uint32_t>(RelationKind::Reconstruction);
}

/* A monomial is coefficient * product(tensor[ref][x]).  The refs are local
 * indices into OperationDescriptor::tensor_refs, not global commitment
 * indices.  Repeated refs are allowed and intentionally represent powers. */
struct Monomial {
    FrWire coefficient{};
    std::vector<std::uint32_t> refs;
};

struct ConstraintDescriptor {
    std::string id;
    std::uint32_t degree = 0U;
    FrWire constant{};
    std::vector<Monomial> terms;

    bool valid(std::string* error = nullptr) const {
        if (id.empty() || degree == 0U || degree > 8U) {
            if (error) *error = "relation constraint has invalid id/degree";
            return false;
        }
        if (!zkdl::fr_canonical(constant)) {
            if (error) *error = "relation constraint constant is not canonical";
            return false;
        }
        for (const auto& term : terms) {
            if (term.refs.empty() || term.refs.size() > degree ||
                !zkdl::fr_canonical(term.coefficient)) {
                if (error) *error = "relation monomial is incompatible with degree";
                return false;
            }
        }
        return true;
    }
};

/* A primitive descriptor is part of the public relation plan.  The primitive
 * proof bytes are only an execution witness; the verifier reconstructs this
 * shape/ref tuple from the authenticated descriptor and then replays the
 * original zkMatMul verifier against the bytes. */
struct PrimitiveDescriptor {
    std::uint32_t kind = 0U;  // 1 = official zkMatMul
    zkdl::MatMulShape matmul_shape;
    std::vector<std::string> tensor_refs;  // A, B, C in native layout order

    bool valid(const std::vector<std::string>& /*operation_refs*/,
               std::string* error = nullptr) const {
        if (kind != 1U || !matmul_shape.valid(error) || tensor_refs.size() != 3U) {
            if (error && error->empty()) *error = "relation primitive descriptor is not canonical";
            return false;
        }
        std::set<std::string> refs;
        for (const auto& ref : tensor_refs) {
            if (ref.empty() || !refs.insert(ref).second) {
                if (error) *error = "relation primitive tensor reference is not canonical";
                return false;
            }
        }
        const std::uint64_t a = static_cast<std::uint64_t>(matmul_shape.num) *
                                matmul_shape.m * matmul_shape.n;
        const std::uint64_t b = static_cast<std::uint64_t>(matmul_shape.num) *
                                matmul_shape.n * matmul_shape.k;
        const std::uint64_t c = static_cast<std::uint64_t>(matmul_shape.num) *
                                matmul_shape.m * matmul_shape.k;
        if (a == 0U || b == 0U || c == 0U ||
            a > std::numeric_limits<std::uint32_t>::max() ||
            b > std::numeric_limits<std::uint32_t>::max() ||
            c > std::numeric_limits<std::uint32_t>::max()) {
            if (error) *error = "relation primitive matrix size is not canonical";
            return false;
        }
        return true;
    }
};

struct FacBindingDescriptor {
    // operation-view reference -> canonical plan reference.  Both ids are
    // public RelationPlan material; the prover cannot choose a selector or
    // reorder the claims after descriptor authentication.
    std::string view_ref;
    std::string canonical_ref;
    // Public selector-MLE layout.  The view is indexed by
    // (view_outer, view_feature); every view slot names the canonical source
    // outer/feature coordinate.  These arrays are reconstructed from the
    // authenticated plan/graph, never selected by the prover.
    std::uint32_t view_outer_domain = 0U;
    std::uint32_t canonical_outer_domain = 0U;
    std::uint32_t view_feature_domain = 0U;
    std::uint32_t canonical_feature_domain = 0U;
    std::vector<std::uint32_t> source_outer_index;
    std::vector<std::uint32_t> source_feature_index;
    std::vector<std::uint8_t> selector_active;

    bool valid(const std::vector<std::string>& operation_refs,
               std::string* error = nullptr) const {
        const auto present = [&](const std::string& value) {
            return !value.empty() &&
                   std::find(operation_refs.begin(), operation_refs.end(), value) !=
                       operation_refs.end();
        };
        if (!present(view_ref) || !present(canonical_ref) || view_ref == canonical_ref) {
            if (error) *error = "FAC4DNN view binding references are not canonical";
            return false;
        }
        const auto power_of_two = [](std::uint32_t value) {
            return value != 0U && (value & (value - 1U)) == 0U;
        };
        if (!power_of_two(view_outer_domain) || !power_of_two(canonical_outer_domain) ||
            !power_of_two(view_feature_domain) || !power_of_two(canonical_feature_domain) ||
            source_outer_index.size() != view_outer_domain ||
            source_feature_index.size() != view_feature_domain ||
            selector_active.size() != view_outer_domain) {
            if (error) *error = "FAC4DNN selector layout has incomplete power-of-two domains";
            return false;
        }
        for (std::uint32_t slot = 0U; slot < view_outer_domain; ++slot) {
            if (selector_active[slot] > 1U ||
                (!selector_active[slot] && source_outer_index[slot] != 0U) ||
                (selector_active[slot] && source_outer_index[slot] >= canonical_outer_domain)) {
                if (error) *error = "FAC4DNN selector outer index is out of bounds";
                return false;
            }
        }
        for (const auto source : source_feature_index)
            if (source >= canonical_feature_domain) {
                if (error) *error = "FAC4DNN selector feature index is out of bounds";
                return false;
            }
        return true;
    }
};

struct OperationDescriptor {
    std::string id;                 // stage-qualified, never a basename
    RelationKind kind = RelationKind::Add;
    std::uint32_t degree = 0U;
    // source_refs are the logical plan inputs/outputs.  tensor_refs are the
    // canonical, same-domain MLE views consumed by the relation polynomial;
    // reindex/route operations may require explicit derived views.
    std::vector<std::string> source_refs;
    std::vector<std::string> tensor_refs;
    std::vector<std::string> selector_refs;
    std::string mapping;
    std::string padding;
    std::string terminal_claim;
    std::string terminal_point;
    bool public_route = false;
    bool requires_reindex = false;
    std::vector<FrWire> constants;
    std::vector<ConstraintDescriptor> constraints;
    std::vector<PrimitiveDescriptor> primitives;
    // Public selector/equality-MLE descriptor.  For sparse route this is
    // reconstructed from Graph::src/dst by the adapter on both sides.
    std::uint32_t selector_edge_domain = 0U;
    std::uint32_t selector_node_domain = 0U;
    std::uint32_t selector_feature_domain = 0U;
    std::uint32_t selector_source_feature_domain = 0U;
    std::vector<std::uint32_t> selector_src;
    std::vector<std::uint32_t> selector_dst;
    // Public graph slots.  Inactive slots are fixed zero padding and must
    // carry src=dst=0; they contribute no selector mass.
    std::vector<std::uint8_t> selector_active;
    // Output feature coordinate -> source feature coordinate.  Standard
    // graph routing uses identity; scalar broadcast repeats source 0.
    std::vector<std::uint32_t> selector_feature_map;
    std::string selector_source_ref;
    std::string selector_output_ref;
    // A normal sparse route selects source[src[e]].  A broadcast route (for
    // example inverse denominator -> edge) selects source[dst[e]].
    bool route_by_destination = false;
    // Public FAC4DNN Line8 view bindings.  A single native proof compresses
    // this fixed-order list with one FS rho and opens the resulting claim
    // against the canonical commitments.
    std::vector<FacBindingDescriptor> fac_bindings;
    // A family proof combines this operation's fixed low-degree constraints
    // with FS coefficients into one Sumcheck.  Empty means ordinary
    // constraint proofs are used.
    std::string batch_family;
    // Authenticated static-table tensor id for indexed lookup families.
    std::string public_table_ref;
    // A broadcast divisor that is a public scale/constant rather than a
    // witness tensor (output-scale alignment and final head averaging).
    bool has_public_divisor = false;
    FrWire public_divisor{};

    bool valid(std::string* error = nullptr) const {
        const bool canonical_root = id == "head_sum" || id == "final_average" ||
                                    id == "terminal_opening";
        const bool stage_qualified = id.find('/') != std::string::npos;
        const bool root_kind = (id == "head_sum" && kind == RelationKind::HeadSum) ||
                               (id == "final_average" && kind == RelationKind::FinalAverage) ||
                               (id == "terminal_opening" && kind == RelationKind::PublicOutput);
        const auto power_of_two = [](std::uint32_t value) {
            return value != 0U && (value & (value - 1U)) == 0U;
        };
        if (id.empty() || (!stage_qualified && !canonical_root) ||
            (canonical_root && !root_kind) ||
            !relation_kind_valid(static_cast<std::uint32_t>(kind)) || degree == 0U || degree > 8U) {
            if (error) *error = "relation operation descriptor is not canonical";
            return false;
        }
        if (mapping.empty() || padding.empty() || terminal_claim.empty() || terminal_point.empty()) {
            if (error) *error = "relation operation descriptor is missing mapping/padding/terminal identity";
            return false;
        }
        if (source_refs.empty() || tensor_refs.empty() ||
            (constraints.empty() && primitives.empty() && kind != RelationKind::SparseRoute)) {
            if (error) *error = "relation operation descriptor has no tensor/constraint body";
            return false;
        }
        std::set<std::string> source_ids;
        for (const auto& ref : source_refs)
            if (ref.empty() || !source_ids.insert(ref).second) {
                if (error) *error = "relation operation has duplicate/non-qualified source ref";
                return false;
            }
        std::set<std::string> refs;
        for (const auto& ref : tensor_refs) {
            if (ref.empty() || !refs.insert(ref).second) {
                if (error) *error = "relation operation has duplicate/non-qualified tensor ref";
                return false;
            }
        }
        std::set<std::string> constraint_ids;
        for (const auto& constraint : constraints) {
            if (!constraint.valid(error) || constraint.degree > degree ||
                !constraint_ids.insert(constraint.id).second) return false;
            for (const auto& term : constraint.terms)
                for (const std::uint32_t ref : term.refs)
                    if (ref >= tensor_refs.size()) {
                        if (error) *error = "relation monomial references tensor outside operation";
                        return false;
                    }
        }
        for (const auto& constant : constants)
            if (!zkdl::fr_canonical(constant)) {
                if (error) *error = "relation operation constant is not canonical";
                return false;
            }
        std::vector<std::string> primitive_refs = tensor_refs;
        primitive_refs.insert(primitive_refs.end(), source_refs.begin(), source_refs.end());
        for (const auto& primitive : primitives)
            if (!primitive.valid(primitive_refs, error)) return false;
        std::set<std::string> fac_views;
        std::vector<std::string> binding_refs = tensor_refs;
        binding_refs.insert(binding_refs.end(), source_refs.begin(), source_refs.end());
        std::string previous_view;
        std::string previous_canonical;
        for (const auto& binding : fac_bindings) {
            if (!binding.valid(binding_refs, error) ||
                !fac_views.insert(binding.view_ref).second ||
                (!previous_view.empty() &&
                 std::pair<std::string, std::string>(previous_view, previous_canonical) >=
                     std::pair<std::string, std::string>(binding.view_ref,
                                                          binding.canonical_ref))) {
                if (error && error->empty()) *error = "duplicate FAC4DNN view binding";
                return false;
            }
            previous_view = binding.view_ref;
            previous_canonical = binding.canonical_ref;
        }
        if (!batch_family.empty() && batch_family != "indexed_public_table" &&
            batch_family != "signed_round_div") {
            if (error) *error = "relation operation has an unknown batched family";
            return false;
        }
        if (batch_family == "indexed_public_table" && public_table_ref.empty()) {
            if (error) *error = "indexed table relation has no authenticated table reference";
            return false;
        }
        if (batch_family == "indexed_public_table" &&
            std::find(source_refs.begin(), source_refs.end(), public_table_ref) == source_refs.end()) {
            if (error) *error = "indexed table relation does not expose its table in the public source set";
            return false;
        }
        if (batch_family != "indexed_public_table" && !public_table_ref.empty()) {
            if (error) *error = "non-table relation carries an unexpected public table reference";
            return false;
        }
        if (has_public_divisor && !zkdl::fr_canonical(public_divisor)) {
            if (error) *error = "relation public divisor is not canonical";
            return false;
        }
        if (has_public_divisor && batch_family != "signed_round_div") {
            if (error) *error = "non-round-div relation carries a public divisor";
            return false;
        }
        if (constraints.size() > kMaxRelationConstraints) {
            if (error) *error = "relation operation has too many low-degree constraints";
            return false;
        }
        if (!batch_family.empty() && constraints.empty()) {
            if (error) *error = "batched relation has no fixed public constraint family";
            return false;
        }
        const bool route_relation = kind == RelationKind::SparseRoute ||
                                    !selector_source_ref.empty() ||
                                    !selector_output_ref.empty();
        if (route_relation) {
            if (selector_edge_domain == 0U || selector_node_domain == 0U ||
                selector_feature_domain == 0U || selector_source_feature_domain == 0U ||
                !power_of_two(selector_edge_domain) || !power_of_two(selector_node_domain) ||
                !power_of_two(selector_feature_domain) ||
                !power_of_two(selector_source_feature_domain) ||
                selector_src.size() != selector_edge_domain ||
                selector_dst.size() != selector_edge_domain ||
                selector_active.size() != selector_edge_domain ||
                selector_feature_map.size() != selector_feature_domain ||
                selector_source_ref.empty() || selector_output_ref.empty()) {
                if (error) *error = "sparse route selector descriptor is incomplete";
                return false;
            }
            if (std::find(tensor_refs.begin(), tensor_refs.end(), selector_output_ref) == tensor_refs.end()) {
                if (error) *error = "sparse route output is not a canonical relation tensor";
                return false;
            }
            std::set<std::uint32_t> source_values;
            std::set<std::uint32_t> destination_values;
            for (std::uint32_t edge = 0U; edge < selector_edge_domain; ++edge) {
                if (selector_active[edge] > 1U ||
                    (!selector_active[edge] &&
                     (selector_src[edge] != 0U || selector_dst[edge] != 0U)) ||
                    (selector_active[edge] &&
                     (selector_src[edge] >= selector_node_domain ||
                      selector_dst[edge] >= selector_node_domain))) {
                    if (error) *error = "sparse route selector index is out of bounds";
                    return false;
                }
                if (selector_active[edge]) {
                    source_values.insert(selector_src[edge]);
                    destination_values.insert(selector_dst[edge]);
                }
            }
            for (const auto source : selector_feature_map)
                if (source >= selector_source_feature_domain) {
                    if (error) *error = "sparse route feature selector index is out of bounds";
                    return false;
                }
            (void)source_values;
            (void)destination_values;
        } else if (selector_edge_domain != 0U || selector_node_domain != 0U ||
                   selector_feature_domain != 0U || selector_source_feature_domain != 0U ||
                   !selector_src.empty() ||
                   !selector_dst.empty() || !selector_active.empty() ||
                   !selector_feature_map.empty() ||
                   !selector_source_ref.empty() ||
                   !selector_output_ref.empty() || route_by_destination) {
            if (error) *error = "non-route operation carries an unexpected selector descriptor";
            return false;
        }
        return true;
    }
};

struct TensorWitness {
    std::string id;
    std::uint32_t visibility = 0U;  // 1 public, 2 registered model, 3 private witness
    std::uint32_t domain_size = 0U;
    std::vector<FrWire> values;

    bool valid(std::string* error = nullptr) const {
        if (id.empty() ||
            domain_size == 0U || (domain_size & (domain_size - 1U)) != 0U ||
            values.size() != domain_size || visibility == 0U || visibility > 3U) {
            if (error) *error = "relation tensor witness has invalid canonical shape";
            return false;
        }
        for (const auto& value : values)
            if (!zkdl::fr_canonical(value)) {
                if (error) *error = "relation tensor witness contains non-canonical field residue";
                return false;
            }
        return true;
    }
};

/* Public shape/padding contract used by the independent verifier.  Values are
 * intentionally absent: only the authenticated public graph/table material
 * may be supplied separately through VerifierContext::public_tensor_values. */
struct TensorDescriptor {
    std::string id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> padded_shape;
    std::string mle_order;
    std::string stack_axis;
    std::string mapping;
    std::string padding_rule;
    std::string registry_id;

    bool valid(std::string* error = nullptr) const {
        if (id.empty() || visibility == 0U || visibility > 3U || domain_size == 0U ||
            logical_shape.empty() || logical_shape.size() != padded_shape.size() ||
            mle_order.empty() || stack_axis.empty() || mapping.empty() ||
            padding_rule.empty()) {
            if (error) *error = "relation tensor descriptor is incomplete";
            return false;
        }
        std::uint64_t product = 1U;
        for (std::size_t i = 0U; i < logical_shape.size(); ++i) {
            if (logical_shape[i] == 0U || padded_shape[i] < logical_shape[i] ||
                padded_shape[i] == 0U || (padded_shape[i] & (padded_shape[i] - 1U)) != 0U ||
                product > std::numeric_limits<std::uint64_t>::max() / padded_shape[i]) {
                if (error) *error = "relation tensor descriptor has invalid logical/padded shape";
                return false;
            }
            product *= padded_shape[i];
        }
        if (product != domain_size || (domain_size & (domain_size - 1U)) != 0U) {
            if (error) *error = "relation tensor descriptor domain does not match padded shape";
            return false;
        }
        if (visibility == 2U && registry_id.empty()) {
            if (error) *error = "model tensor descriptor has no authenticated registry id";
            return false;
        }
        return true;
    }
};

struct RelationStatement {
    std::array<byte, 32> statement_digest{};
    std::array<byte, 32> model_digest{};
    std::array<byte, 32> table_digest{};
    std::array<byte, 32> ledger_digest{};
    std::array<byte, 32> srs_digest{};
    std::array<byte, 32> registry_digest{};
    std::array<byte, 32> public_output_digest{};
    std::vector<byte> public_output;
};

struct CommitmentWire {
    std::string tensor_id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    G1Jacobian_t commitment{};
};

struct ConstraintProof {
    ConstraintDescriptor descriptor;
    // Indices into OperationDescriptor::tensor_refs for the tensors that
    // actually occur in descriptor.  Large lookup/SignedRoundDiv operations
    // may carry hundreds of auxiliary views, but each relation proof opens
    // only the canonical references used by that constraint.
    std::vector<std::uint32_t> opening_tensor_indices;
    std::vector<std::uint32_t> opening_commitment_indices;
    zkdl::SumcheckProof sumcheck;
    std::vector<zkdl::EvaluationOpeningProof> openings;
    // Independent Fiat--Shamir pointwise check.  The Sumcheck still carries
    // the native relation reduction; this second point prevents a residual
    // that merely cancels in the full-domain sum from being accepted.
    std::vector<Fr_t> pointwise_point;
    std::vector<zkdl::EvaluationOpeningProof> pointwise_openings;
};

struct OperationProof {
    OperationDescriptor descriptor;
    std::vector<ConstraintProof> constraints;
    // One serialized proof per descriptor primitive, in descriptor order.
    std::vector<std::vector<byte>> primitive_proofs;
    // One binding record per primitive.  Each canonical opening is made
    // against the outer tensor commitment at the exact point used by the
    // official zkMatMul row commitment/opening.
    struct PrimitiveBinding {
        std::uint32_t a_commitment_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t b_commitment_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t c_commitment_index = std::numeric_limits<std::uint32_t>::max();
        zkdl::EvaluationOpeningProof opening_a;
        zkdl::EvaluationOpeningProof opening_b;
        zkdl::EvaluationOpeningProof opening_c;
    };
    std::vector<PrimitiveBinding> primitive_bindings;
    std::vector<byte> route_proof;
    // FAC4DNN Line8 relation for all non-identity same-domain views in this
    // operation.  The bytes are parsed strictly before any claim is used.
    std::vector<byte> fac_proof;
    // One serialized ConstraintProof for a fixed-family FS linear
    // combination.  It replaces per-bit/per-row proof repetition.
    std::vector<byte> batch_proof;
};

/* Proof of the public sparse gather relation

       R(e,f) = S(src[e],f).

 * The selector is a public equality-MLE derived from selector_src.  The
 * Sumcheck runs over the source-node coordinates after the edge/feature
 * verifier challenges are fixed, and the two Hyrax openings close the source
 * and routed-output claims against their canonical commitments. */
struct SparseRouteProof {
    static const std::uint32_t kVersion = 2U;
    std::uint32_t version = kVersion;
    std::uint32_t source_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t output_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t edge_domain = 0U;
    std::uint32_t node_domain = 0U;
    // Output/view feature domain.  The source feature domain may differ for
    // a public scalar broadcast or another authenticated feature map.
    std::uint32_t feature_domain = 0U;
    std::uint32_t source_feature_domain = 0U;
    std::vector<Fr_t> edge_point;
    std::vector<Fr_t> feature_point;
    std::vector<Fr_t> source_feature_point;
    // FS challenges for the source-node Sumcheck variables.  They are part
    // of the wire so the verifier never has to replay or trust prover state.
    std::vector<Fr_t> node_point;
    Fr_t output_claim{};
    Fr_t source_sum_claim{};
    Fr_t dst_selector_claim{};
    Fr_t rho{};
    Fr_t compressed_claim{};
    zkdl::SumcheckProof sumcheck;
    zkdl::EvaluationOpeningProof source_opening;
    zkdl::EvaluationOpeningProof output_opening;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            SparseRouteProof& proof,
                            std::string* error = nullptr);
};

/* FAC4DNN Line8 proof for a fixed, public view-binding list.  Each binding is
 * reduced by a public selector-MLE Sumcheck to one opening of its canonical
 * source commitment.  The binding residual claims are then compressed in the
 * authenticated order with one outer FS rho; no host-side gather is accepted
 * as a substitute. */
struct FacLine8Proof {
    static const std::uint32_t kVersion = 2U;
    std::uint32_t version = kVersion;
    Fr_t rho{};
    Fr_t compressed_claim{};
    std::vector<Fr_t> selector_claims;
    std::vector<SparseRouteProof> selector_proofs;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            FacLine8Proof& proof,
                            std::string* error = nullptr);
};

struct Proof {
    static const std::uint32_t kVersion = 2U;
    std::uint32_t version = kVersion;
    RelationStatement statement;
    std::vector<CommitmentWire> commitments;
    // Hiding records are part of the relation proof wire.  The verifier
    // compares them with its public context and checks the native Schnorr
    // proof before accepting any model/private opening.
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    std::vector<OperationProof> operations;
    std::uint32_t public_output_commitment = std::numeric_limits<std::uint32_t>::max();
    zkdl::EvaluationOpeningProof public_output_opening;
    std::array<byte, 32> transcript_digest{};

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes, Proof& proof,
                            std::string* error = nullptr);
};

struct ProverContext {
    RelationStatement statement;
    std::vector<TensorWitness> tensors;
    std::vector<TensorDescriptor> tensor_descriptors;
    std::vector<OperationDescriptor> operations;
    std::vector<zkdl::CpuG1Point> generators;
    std::vector<CommitmentWire> registered_model_commitments;
    // Private/model tensors are committed through the native hiding
    // commitment object.  The record is public metadata plus a Schnorr
    // knowledge proof; values/randomness never cross this context boundary.
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    // Public graph-derived material may be attached to a non-public view
    // commitment (for example group-boundary bits).  The verifier accepts
    // only ids explicitly reconstructed from Graph/CSR, never arbitrary
    // caller-supplied private values.
    std::set<std::string> graph_public_tensor_ids;
    std::map<std::string, std::vector<FrWire>> public_tensor_values;
};

struct VerifierContext {
    RelationStatement statement;
    std::vector<OperationDescriptor> operations;
    // The independent verifier may carry shape-only ids for every committed
    // public/model/witness tensor.  When present this set is authoritative;
    // it prevents an envelope from smuggling an unplanned commitment into the
    // transcript.  Pure relation callers may leave it empty, in which case
    // the operation references are the conservative expected set.
    std::vector<std::string> tensor_ids;
    std::map<std::string, std::uint32_t> tensor_visibility;
    std::vector<TensorDescriptor> tensor_descriptors;
    std::vector<CommitmentWire> registered_model_commitments;
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    std::set<std::string> graph_public_tensor_ids;
    std::map<std::string, std::vector<FrWire>> public_tensor_values;
    std::vector<zkdl::CpuG1Point> generators;
};

inline bool same_wire(const FrWire& a, const FrWire& b) noexcept {
    return zkdl::fr_compare(a, b) == 0;
}

inline bool same_bytes(const std::array<byte, 32>& a, const std::array<byte, 32>& b) noexcept {
    return a == b;
}

inline std::uint32_t log2_exact(std::uint32_t value) {
    if (value == 0U || (value & (value - 1U)) != 0U) return std::numeric_limits<std::uint32_t>::max();
    std::uint32_t bits = 0U;
    while (value > 1U) { value >>= 1U; ++bits; }
    return bits;
}

inline FrWire fr_from_signed_i64(std::int64_t value) {
    if (value >= 0) return zkdl::fr_from_u32(static_cast<std::uint32_t>(value));
    const std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    FrWire out{};
    out[0] = static_cast<std::uint32_t>(magnitude);
    out[1] = static_cast<std::uint32_t>(magnitude >> 32U);
    return zkdl::fr_neg(out);
}

inline FrWire eval_monomial(const Monomial& term,
                            const std::vector<FrWire>& values) {
    FrWire result = term.coefficient;
    for (const std::uint32_t ref : term.refs) result = zkdl::fr_mul(result, values.at(ref));
    return result;
}

inline FrWire eval_constraint(const ConstraintDescriptor& descriptor,
                              const std::vector<FrWire>& values) {
    FrWire result = descriptor.constant;
    for (const auto& term : descriptor.terms)
        result = zkdl::fr_add(result, eval_monomial(term, values));
    return result;
}

inline std::vector<std::uint32_t> constraint_tensor_indices(
    const ConstraintDescriptor& descriptor, std::size_t tensor_count) {
    std::set<std::uint32_t> seen;
    for (const auto& term : descriptor.terms)
        for (const auto ref : term.refs)
            if (ref < tensor_count) seen.insert(ref);
    // A zero-term identity is used only for terminal-output plumbing.  Keep
    // one canonical opening so that the relation still carries a bounded
    // tensor domain and cannot be accepted from metadata alone.
    if (seen.empty() && tensor_count != 0U) seen.insert(0U);
    return std::vector<std::uint32_t>(seen.begin(), seen.end());
}

inline void append_fr_wire(std::vector<byte>& out, const FrWire& value) {
    for (const std::uint32_t limb : value) zkdl::append_u32(out, limb);
}

inline bool read_fr_wire(const std::vector<byte>& in, size_t& offset, FrWire& value) {
    for (std::uint32_t& limb : value) if (!zkdl::read_u32(in, offset, limb)) return false;
    return zkdl::fr_canonical(value);
}

inline void append_digest(std::vector<byte>& out, const std::array<byte, 32>& value) {
    out.insert(out.end(), value.begin(), value.end());
}

inline bool read_digest(const std::vector<byte>& in, size_t& offset,
                        std::array<byte, 32>& value) {
    if (offset > in.size() || in.size() - offset < value.size()) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + value.size()), value.begin());
    offset += value.size();
    return true;
}

inline void append_string_vector(std::vector<byte>& out,
                                 const std::vector<std::string>& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) zkdl::append_string(out, value);
}

inline bool read_string_vector(const std::vector<byte>& in, size_t& offset,
                               std::vector<std::string>& values) {
    std::uint64_t count = 0;
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    values.clear(); values.reserve(static_cast<size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::string value;
        if (!zkdl::read_string(in, offset, value)) return false;
        values.push_back(std::move(value));
    }
    return true;
}

inline void append_fr_vector(std::vector<byte>& out, const std::vector<FrWire>& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) append_fr_wire(out, value);
}

inline bool read_fr_vector(const std::vector<byte>& in, size_t& offset,
                           std::vector<FrWire>& values) {
    std::uint64_t count = 0;
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    values.resize(static_cast<size_t>(count));
    for (auto& value : values) if (!read_fr_wire(in, offset, value)) return false;
    return true;
}

inline std::vector<byte> encode_public_field_vector(const std::vector<FrWire>& values) {
    std::vector<byte> encoded;
    append_fr_vector(encoded, values);
    return encoded;
}

inline bool decode_public_field_vector(const std::vector<byte>& encoded,
                                      std::vector<FrWire>& values) {
    size_t offset = 0U;
    if (!read_fr_vector(encoded, offset, values) || offset != encoded.size() || values.empty()) return false;
    return true;
}

inline void fold_values(std::vector<FrWire>& values, const FrWire& challenge);

inline FrWire mle_value(std::vector<FrWire> values, const std::vector<FrWire>& point) {
    if (values.empty() || values.size() != (size_t(1U) << point.size()))
        throw std::invalid_argument("public MLE shape mismatch");
    for (const auto& challenge : point) fold_values(values, challenge);
    return values.front();
}

inline void append_monomial(std::vector<byte>& out, const Monomial& term) {
    append_fr_wire(out, term.coefficient);
    zkdl::append_u64(out, static_cast<std::uint64_t>(term.refs.size()));
    for (const std::uint32_t ref : term.refs) zkdl::append_u32(out, ref);
}

inline bool read_monomial(const std::vector<byte>& in, size_t& offset, Monomial& term) {
    std::uint64_t count = 0;
    if (!read_fr_wire(in, offset, term.coefficient) ||
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    term.refs.resize(static_cast<size_t>(count));
    for (std::uint32_t& ref : term.refs) if (!zkdl::read_u32(in, offset, ref)) return false;
    return true;
}

inline void append_constraint(std::vector<byte>& out, const ConstraintDescriptor& descriptor) {
    zkdl::append_string(out, descriptor.id);
    zkdl::append_u32(out, descriptor.degree);
    append_fr_wire(out, descriptor.constant);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.terms.size()));
    for (const auto& term : descriptor.terms) append_monomial(out, term);
}

inline bool read_constraint(const std::vector<byte>& in, size_t& offset,
                            ConstraintDescriptor& descriptor) {
    std::uint64_t count = 0;
    if (!zkdl::read_string(in, offset, descriptor.id) ||
        !zkdl::read_u32(in, offset, descriptor.degree) ||
        !read_fr_wire(in, offset, descriptor.constant) ||
        !zkdl::read_u64(in, offset, count) || count > kMaxRelationTerms) return false;
    descriptor.terms.resize(static_cast<size_t>(count));
    for (auto& term : descriptor.terms) if (!read_monomial(in, offset, term)) return false;
    return true;
}

inline void append_hiding_record(std::vector<byte>& out,
                                 const zkdl::CommitmentRecord& record) {
    zkdl::append_string(out, record.label);
    zkdl::append_u32(out, record.visibility);
    zkdl::append_u64(out, static_cast<std::uint64_t>(record.points.size()));
    for (const auto& point : record.points) zkdl::append_g1(out, point);
    zkdl::append_g1(out, record.value_commitment);
    zkdl::append_g1(out, record.blinding_generator);
    zkdl::append_g1(out, record.commitment);
    zkdl::append_g1(out, record.schnorr_R);
    zkdl::append_fr(out, record.schnorr_response);
}

inline bool read_hiding_record(const std::vector<byte>& in, size_t& offset,
                               zkdl::CommitmentRecord& record) {
    std::uint64_t count = 0U;
    if (!zkdl::read_string(in, offset, record.label) ||
        !zkdl::read_u32(in, offset, record.visibility) ||
        !zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    record.points.resize(static_cast<std::size_t>(count));
    for (auto& point : record.points)
        if (!zkdl::read_g1(in, offset, point)) return false;
    return zkdl::read_g1(in, offset, record.value_commitment) &&
           zkdl::read_g1(in, offset, record.blinding_generator) &&
           zkdl::read_g1(in, offset, record.commitment) &&
           zkdl::read_g1(in, offset, record.schnorr_R) &&
           zkdl::read_fr(in, offset, record.schnorr_response);
}

inline void append_native_fr_vector(std::vector<byte>& out,
                                    const std::vector<Fr_t>& values);
inline bool read_native_fr_vector(const std::vector<byte>& in, size_t& offset,
                                  std::vector<Fr_t>& values);
struct ConstraintProof;
inline std::vector<byte> serialize_constraint_payload(const ConstraintProof& proof);
inline bool deserialize_constraint_payload(const std::vector<byte>& bytes,
                                           ConstraintProof& proof,
                                           std::string* error = nullptr);

inline void append_primitive(std::vector<byte>& out,
                             const PrimitiveDescriptor& primitive) {
    zkdl::append_u32(out, primitive.kind);
    zkdl::matmul_append_shape(out, primitive.matmul_shape);
    append_string_vector(out, primitive.tensor_refs);
}

inline bool read_primitive(const std::vector<byte>& in, size_t& offset,
                           PrimitiveDescriptor& primitive) {
    return zkdl::read_u32(in, offset, primitive.kind) &&
           zkdl::matmul_read_shape(in, offset, primitive.matmul_shape) &&
           read_string_vector(in, offset, primitive.tensor_refs);
}

inline void append_operation(std::vector<byte>& out, const OperationDescriptor& descriptor) {
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
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.selector_src.size()));
    for (const auto value : descriptor.selector_src) zkdl::append_u32(out, value);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.selector_dst.size()));
    for (const auto value : descriptor.selector_dst) zkdl::append_u32(out, value);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.selector_active.size()));
    for (const auto value : descriptor.selector_active) zkdl::append_u32(out, value);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.selector_feature_map.size()));
    for (const auto value : descriptor.selector_feature_map) zkdl::append_u32(out, value);
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
        zkdl::append_u64(out, static_cast<std::uint64_t>(binding.source_outer_index.size()));
        for (const auto value : binding.source_outer_index) zkdl::append_u32(out, value);
        zkdl::append_u64(out, static_cast<std::uint64_t>(binding.source_feature_index.size()));
        for (const auto value : binding.source_feature_index) zkdl::append_u32(out, value);
        zkdl::append_u64(out, static_cast<std::uint64_t>(binding.selector_active.size()));
        for (const auto value : binding.selector_active) zkdl::append_u32(out, value);
    }
    zkdl::append_string(out, descriptor.batch_family);
    zkdl::append_string(out, descriptor.public_table_ref);
    zkdl::append_u32(out, descriptor.has_public_divisor ? 1U : 0U);
    append_fr_wire(out, descriptor.public_divisor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.primitives.size()));
    for (const auto& primitive : descriptor.primitives) append_primitive(out, primitive);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.constraints.size()));
    for (const auto& constraint : descriptor.constraints) append_constraint(out, constraint);
}

/* The operation reader keeps its two boolean fields separate so their
 * canonical representation is unambiguous across languages. */
inline bool read_operation_strict(const std::vector<byte>& in, size_t& offset,
                                  OperationDescriptor& descriptor) {
    std::uint32_t kind = 0U, public_route = 0U, reindex = 0U,
                  route_by_destination = 0U;
    std::uint64_t count = 0U;
    if (!zkdl::read_string(in, offset, descriptor.id) ||
        !zkdl::read_u32(in, offset, kind) || !relation_kind_valid(kind) ||
        !zkdl::read_u32(in, offset, descriptor.degree) ||
        !read_string_vector(in, offset, descriptor.source_refs) ||
        !read_string_vector(in, offset, descriptor.tensor_refs) ||
        !read_string_vector(in, offset, descriptor.selector_refs) ||
        !zkdl::read_string(in, offset, descriptor.mapping) ||
        !zkdl::read_string(in, offset, descriptor.padding) ||
        !zkdl::read_string(in, offset, descriptor.terminal_claim) ||
        !zkdl::read_string(in, offset, descriptor.terminal_point) ||
        !zkdl::read_u32(in, offset, public_route) || public_route > 1U ||
        !zkdl::read_u32(in, offset, reindex) || reindex > 1U ||
        !read_fr_vector(in, offset, descriptor.constants) ||
        !zkdl::read_u32(in, offset, descriptor.selector_edge_domain) ||
        !zkdl::read_u32(in, offset, descriptor.selector_node_domain) ||
        !zkdl::read_u32(in, offset, descriptor.selector_feature_domain) ||
        !zkdl::read_u32(in, offset, descriptor.selector_source_feature_domain) ||
        !zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    descriptor.selector_src.resize(static_cast<size_t>(count));
    for (auto& value : descriptor.selector_src)
        if (!zkdl::read_u32(in, offset, value)) return false;
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    descriptor.selector_dst.resize(static_cast<size_t>(count));
    for (auto& value : descriptor.selector_dst)
        if (!zkdl::read_u32(in, offset, value)) return false;
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    descriptor.selector_active.resize(static_cast<size_t>(count));
    for (auto& value : descriptor.selector_active) {
        std::uint32_t active = 0U;
        if (!zkdl::read_u32(in, offset, active) || active > 1U) return false;
        value = static_cast<std::uint8_t>(active);
    }
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    descriptor.selector_feature_map.resize(static_cast<size_t>(count));
    for (auto& value : descriptor.selector_feature_map)
        if (!zkdl::read_u32(in, offset, value)) return false;
    if (!zkdl::read_string(in, offset, descriptor.selector_source_ref) ||
        !zkdl::read_string(in, offset, descriptor.selector_output_ref) ||
        !zkdl::read_u32(in, offset, route_by_destination) || route_by_destination > 1U ||
        !zkdl::read_u64(in, offset, count) || count > kMaxRelationOpenings) return false;
    descriptor.fac_bindings.resize(static_cast<std::size_t>(count));
    for (auto& binding : descriptor.fac_bindings) {
        if (!zkdl::read_string(in, offset, binding.view_ref) ||
            !zkdl::read_string(in, offset, binding.canonical_ref) ||
            !zkdl::read_u32(in, offset, binding.view_outer_domain) ||
            !zkdl::read_u32(in, offset, binding.canonical_outer_domain) ||
            !zkdl::read_u32(in, offset, binding.view_feature_domain) ||
            !zkdl::read_u32(in, offset, binding.canonical_feature_domain) ||
            !zkdl::read_u64(in, offset, count) || count > kMaxRelationOpenings) return false;
        binding.source_outer_index.resize(static_cast<std::size_t>(count));
        for (auto& value : binding.source_outer_index)
            if (!zkdl::read_u32(in, offset, value)) return false;
        if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationOpenings) return false;
        binding.source_feature_index.resize(static_cast<std::size_t>(count));
        for (auto& value : binding.source_feature_index)
            if (!zkdl::read_u32(in, offset, value)) return false;
        if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationOpenings) return false;
        binding.selector_active.resize(static_cast<std::size_t>(count));
        for (auto& value : binding.selector_active) {
            std::uint32_t active = 0U;
            if (!zkdl::read_u32(in, offset, active) || active > 1U) return false;
            value = static_cast<std::uint8_t>(active);
        }
    }
    std::uint32_t has_public_divisor = 0U;
    if (!zkdl::read_string(in, offset, descriptor.batch_family) ||
        !zkdl::read_string(in, offset, descriptor.public_table_ref) ||
        !zkdl::read_u32(in, offset, has_public_divisor) || has_public_divisor > 1U ||
        !read_fr_wire(in, offset, descriptor.public_divisor) ||
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    descriptor.has_public_divisor = has_public_divisor != 0U;
    descriptor.primitives.resize(static_cast<size_t>(count));
    for (auto& primitive : descriptor.primitives)
        if (!read_primitive(in, offset, primitive)) return false;
    if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationConstraints ||
        (count == 0U && descriptor.primitives.empty())) return false;
    descriptor.kind = static_cast<RelationKind>(kind);
    descriptor.public_route = public_route != 0U;
    descriptor.requires_reindex = reindex != 0U;
    descriptor.route_by_destination = route_by_destination != 0U;
    descriptor.constraints.resize(static_cast<size_t>(count));
    for (auto& constraint : descriptor.constraints)
        if (!read_constraint(in, offset, constraint)) return false;
    return descriptor.valid();
}

inline void append_commitment(std::vector<byte>& out, const CommitmentWire& commitment) {
    zkdl::append_string(out, commitment.tensor_id);
    zkdl::append_u32(out, commitment.visibility);
    zkdl::append_u32(out, commitment.domain_size);
    zkdl::append_g1(out, commitment.commitment);
}

inline bool read_commitment(const std::vector<byte>& in, size_t& offset,
                           CommitmentWire& commitment) {
    return zkdl::read_string(in, offset, commitment.tensor_id) &&
           zkdl::read_u32(in, offset, commitment.visibility) &&
           zkdl::read_u32(in, offset, commitment.domain_size) &&
           zkdl::read_g1(in, offset, commitment.commitment);
}

inline void append_statement(std::vector<byte>& out, const RelationStatement& statement) {
    append_digest(out, statement.statement_digest);
    append_digest(out, statement.model_digest);
    append_digest(out, statement.table_digest);
    append_digest(out, statement.ledger_digest);
    append_digest(out, statement.srs_digest);
    append_digest(out, statement.registry_digest);
    append_digest(out, statement.public_output_digest);
    zkdl::append_u64(out, static_cast<std::uint64_t>(statement.public_output.size()));
    out.insert(out.end(), statement.public_output.begin(), statement.public_output.end());
}

inline bool read_statement(const std::vector<byte>& in, size_t& offset,
                           RelationStatement& statement) {
    std::uint64_t bytes = 0U;
    if (!read_digest(in, offset, statement.statement_digest) ||
        !read_digest(in, offset, statement.model_digest) ||
        !read_digest(in, offset, statement.table_digest) ||
        !read_digest(in, offset, statement.ledger_digest) ||
        !read_digest(in, offset, statement.srs_digest) ||
        !read_digest(in, offset, statement.registry_digest) ||
        !read_digest(in, offset, statement.public_output_digest) ||
        !zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    statement.public_output.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                   in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    return true;
}

inline void append_constraint_proof(std::vector<byte>& out, const ConstraintProof& proof) {
    append_constraint(out, proof.descriptor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.opening_tensor_indices.size()));
    for (const auto index : proof.opening_tensor_indices) zkdl::append_u32(out, index);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.opening_commitment_indices.size()));
    for (const std::uint32_t index : proof.opening_commitment_indices) zkdl::append_u32(out, index);
    std::vector<byte> sumcheck;
    zkdl::serialize_sumcheck(proof.sumcheck, sumcheck);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck.size()));
    out.insert(out.end(), sumcheck.begin(), sumcheck.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.openings.size()));
    for (const auto& opening : proof.openings) zkdl::serialize_opening(opening, out);
    append_native_fr_vector(out, proof.pointwise_point);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.pointwise_openings.size()));
    for (const auto& opening : proof.pointwise_openings) zkdl::serialize_opening(opening, out);
}

inline bool read_constraint_proof(const std::vector<byte>& in, size_t& offset,
                                  ConstraintProof& proof) {
    std::uint64_t count = 0U, proof_bytes = 0U;
    if (!read_constraint(in, offset, proof.descriptor) ||
        !zkdl::read_u64(in, offset, count) || count == 0U || count > kMaxRelationOpenings) return false;
    proof.opening_tensor_indices.resize(static_cast<std::size_t>(count));
    for (auto& index : proof.opening_tensor_indices)
        if (!zkdl::read_u32(in, offset, index)) return false;
    if (!zkdl::read_u64(in, offset, count) || count == 0U || count > kMaxRelationOpenings) return false;
    proof.opening_commitment_indices.resize(static_cast<size_t>(count));
    for (auto& index : proof.opening_commitment_indices)
        if (!zkdl::read_u32(in, offset, index)) return false;
    if (!zkdl::read_u64(in, offset, proof_bytes) || proof_bytes > in.size() - offset) return false;
    std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                              in.begin() + static_cast<std::ptrdiff_t>(offset + proof_bytes));
    offset += static_cast<size_t>(proof_bytes);
    size_t inner = 0U;
    if (!zkdl::deserialize_sumcheck(encoded, inner, proof.sumcheck) || inner != encoded.size()) return false;
    if (!zkdl::read_u64(in, offset, count) || count == 0U || count > kMaxRelationOpenings) return false;
    proof.openings.resize(static_cast<size_t>(count));
    for (auto& opening : proof.openings)
        if (!zkdl::deserialize_opening(in, offset, opening)) return false;
    if (!read_native_fr_vector(in, offset, proof.pointwise_point) ||
        proof.pointwise_point.size() > kMaxRelationPointBits ||
        !zkdl::read_u64(in, offset, count) || count == 0U || count > kMaxRelationOpenings) return false;
    proof.pointwise_openings.resize(static_cast<std::size_t>(count));
    for (auto& opening : proof.pointwise_openings)
        if (!zkdl::deserialize_opening(in, offset, opening)) return false;
    return true;
}

inline void append_primitive_binding(
    std::vector<byte>& out, const OperationProof::PrimitiveBinding& binding) {
    zkdl::append_u32(out, binding.a_commitment_index);
    zkdl::append_u32(out, binding.b_commitment_index);
    zkdl::append_u32(out, binding.c_commitment_index);
    zkdl::serialize_opening(binding.opening_a, out);
    zkdl::serialize_opening(binding.opening_b, out);
    zkdl::serialize_opening(binding.opening_c, out);
}

inline bool read_primitive_binding(
    const std::vector<byte>& in, size_t& offset,
    OperationProof::PrimitiveBinding& binding) {
    return zkdl::read_u32(in, offset, binding.a_commitment_index) &&
           zkdl::read_u32(in, offset, binding.b_commitment_index) &&
           zkdl::read_u32(in, offset, binding.c_commitment_index) &&
           zkdl::deserialize_opening(in, offset, binding.opening_a) &&
           zkdl::deserialize_opening(in, offset, binding.opening_b) &&
           zkdl::deserialize_opening(in, offset, binding.opening_c);
}

inline void append_operation_proof(std::vector<byte>& out, const OperationProof& proof) {
    append_operation(out, proof.descriptor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.primitive_proofs.size()));
    for (const auto& primitive : proof.primitive_proofs) {
        zkdl::append_u64(out, static_cast<std::uint64_t>(primitive.size()));
        out.insert(out.end(), primitive.begin(), primitive.end());
    }
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.primitive_bindings.size()));
    for (const auto& binding : proof.primitive_bindings)
        append_primitive_binding(out, binding);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.route_proof.size()));
    out.insert(out.end(), proof.route_proof.begin(), proof.route_proof.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.fac_proof.size()));
    out.insert(out.end(), proof.fac_proof.begin(), proof.fac_proof.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.batch_proof.size()));
    out.insert(out.end(), proof.batch_proof.begin(), proof.batch_proof.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.constraints.size()));
    for (const auto& constraint : proof.constraints) append_constraint_proof(out, constraint);
}

inline bool read_operation_proof(const std::vector<byte>& in, size_t& offset,
                                 OperationProof& proof) {
    std::uint64_t count = 0U, bytes = 0U;
    if (!read_operation_strict(in, offset, proof.descriptor) ||
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    proof.primitive_proofs.resize(static_cast<size_t>(count));
    for (auto& primitive : proof.primitive_proofs) {
        if (!zkdl::read_u64(in, offset, bytes) || bytes == 0U || bytes > in.size() - offset)
            return false;
        primitive.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                         in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<size_t>(bytes);
    }
    if (!zkdl::read_u64(in, offset, count) || count > 8U) return false;
    proof.primitive_bindings.resize(static_cast<size_t>(count));
    for (auto& binding : proof.primitive_bindings)
        if (!read_primitive_binding(in, offset, binding)) return false;
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    proof.route_proof.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                             in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    if (!proof.route_proof.empty()) {
        SparseRouteProof route;
        if (!SparseRouteProof::deserialize(proof.route_proof, route)) return false;
    }
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    proof.fac_proof.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                           in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    if (!proof.fac_proof.empty()) {
        FacLine8Proof fac;
        if (!FacLine8Proof::deserialize(proof.fac_proof, fac)) return false;
    }
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    proof.batch_proof.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                             in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    if (!proof.batch_proof.empty()) {
        ConstraintProof batch;
        if (!deserialize_constraint_payload(proof.batch_proof, batch)) return false;
    }
    if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationConstraints ||
        (count == 0U && proof.primitive_proofs.empty() && proof.route_proof.empty() &&
         proof.fac_proof.empty() && proof.batch_proof.empty())) return false;
    proof.constraints.resize(static_cast<size_t>(count));
    for (auto& constraint : proof.constraints)
        if (!read_constraint_proof(in, offset, constraint)) return false;
    return true;
}

inline void append_native_fr_vector(std::vector<byte>& out,
                                    const std::vector<Fr_t>& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) zkdl::append_fr(out, value);
}

inline bool read_native_fr_vector(const std::vector<byte>& in, size_t& offset,
                                  std::vector<Fr_t>& values) {
    std::uint64_t count = 0U;
    if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationPointBits) return false;
    values.resize(static_cast<size_t>(count));
    for (auto& value : values)
        if (!zkdl::read_fr(in, offset, value)) return false;
    return true;
}

inline std::vector<byte> FacLine8Proof::serialize() const {
    std::vector<byte> out;
    out.reserve(1024U);
    out.push_back('F'); out.push_back('A'); out.push_back('C'); out.push_back('2');
    zkdl::append_u32(out, version);
    zkdl::append_fr(out, rho);
    zkdl::append_fr(out, compressed_claim);
    zkdl::append_u64(out, static_cast<std::uint64_t>(selector_claims.size()));
    for (const auto& claim : selector_claims) zkdl::append_fr(out, claim);
    zkdl::append_u64(out, static_cast<std::uint64_t>(selector_proofs.size()));
    for (const auto& route : selector_proofs) {
        const std::vector<byte> encoded = route.serialize();
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }
    return out;
}

inline bool FacLine8Proof::deserialize(const std::vector<byte>& bytes,
                                       FacLine8Proof& proof,
                                       std::string* error) {
    FacLine8Proof parsed;
    std::uint64_t count = 0U, route_bytes = 0U;
    size_t offset = 0U;
    constexpr std::uint64_t kMaxWireItems = kMaxRelationOpenings;
    if (bytes.size() < 4U || bytes[0] != 'F' || bytes[1] != 'A' ||
        bytes[2] != 'C' || bytes[3] != '2') {
        if (error) *error = "invalid FAC4DNN Line8 proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, parsed.version) || parsed.version != kVersion ||
        !zkdl::read_fr(bytes, offset, parsed.rho) ||
        !zkdl::read_fr(bytes, offset, parsed.compressed_claim) ||
        !zkdl::read_u64(bytes, offset, count) || count == 0U || count > kMaxWireItems) {
        if (error) *error = "invalid FAC4DNN Line8 descriptor/header";
        return false;
    }
    parsed.selector_claims.resize(static_cast<std::size_t>(count));
    for (auto& claim : parsed.selector_claims)
        if (!zkdl::read_fr(bytes, offset, claim)) return false;
    if (!zkdl::read_u64(bytes, offset, count) || count == 0U || count > kMaxWireItems ||
        count != parsed.selector_claims.size()) return false;
    parsed.selector_proofs.resize(static_cast<std::size_t>(count));
    for (auto& route : parsed.selector_proofs) {
        if (!zkdl::read_u64(bytes, offset, route_bytes) || route_bytes == 0U ||
            route_bytes > bytes.size() - offset) {
            if (error) *error = "invalid FAC4DNN selector route length";
            return false;
        }
        std::vector<byte> encoded(
            bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + route_bytes));
        offset += static_cast<std::size_t>(route_bytes);
        if (!SparseRouteProof::deserialize(encoded, route, error)) return false;
    }
    if (offset != bytes.size()) {
        if (error) *error = "trailing FAC4DNN Line8 proof bytes";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline std::vector<byte> serialize_route_sumcheck(const zkdl::SumcheckProof& proof) {
    std::vector<byte> encoded;
    zkdl::serialize_sumcheck(proof, encoded);
    return encoded;
}

inline std::vector<byte> SparseRouteProof::serialize() const {
    std::vector<byte> out;
    out.reserve(1024U);
    out.push_back('S'); out.push_back('R'); out.push_back('P'); out.push_back('1');
    zkdl::append_u32(out, version);
    zkdl::append_u32(out, source_commitment_index);
    zkdl::append_u32(out, output_commitment_index);
    zkdl::append_u32(out, edge_domain);
    zkdl::append_u32(out, node_domain);
    zkdl::append_u32(out, feature_domain);
    zkdl::append_u32(out, source_feature_domain);
    append_native_fr_vector(out, edge_point);
    append_native_fr_vector(out, feature_point);
    append_native_fr_vector(out, source_feature_point);
    append_native_fr_vector(out, node_point);
    zkdl::append_fr(out, output_claim);
    zkdl::append_fr(out, source_sum_claim);
    zkdl::append_fr(out, dst_selector_claim);
    zkdl::append_fr(out, rho);
    zkdl::append_fr(out, compressed_claim);
    const std::vector<byte> sumcheck_bytes = serialize_route_sumcheck(this->sumcheck);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck_bytes.size()));
    out.insert(out.end(), sumcheck_bytes.begin(), sumcheck_bytes.end());
    zkdl::serialize_opening(source_opening, out);
    zkdl::serialize_opening(output_opening, out);
    return out;
}

inline bool SparseRouteProof::deserialize(const std::vector<byte>& bytes,
                                          SparseRouteProof& proof,
                                          std::string* error) {
    SparseRouteProof parsed;
    size_t offset = 0U;
    std::uint64_t sumcheck_bytes = 0U;
    if (bytes.size() < 4U || bytes[0] != 'S' || bytes[1] != 'R' ||
        bytes[2] != 'P' || bytes[3] != '1') {
        if (error) *error = "invalid sparse route proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, parsed.version) || parsed.version != kVersion ||
        !zkdl::read_u32(bytes, offset, parsed.source_commitment_index) ||
        !zkdl::read_u32(bytes, offset, parsed.output_commitment_index) ||
        !zkdl::read_u32(bytes, offset, parsed.edge_domain) ||
        !zkdl::read_u32(bytes, offset, parsed.node_domain) ||
        !zkdl::read_u32(bytes, offset, parsed.feature_domain) ||
        !zkdl::read_u32(bytes, offset, parsed.source_feature_domain) ||
        !read_native_fr_vector(bytes, offset, parsed.edge_point) ||
        !read_native_fr_vector(bytes, offset, parsed.feature_point) ||
        !read_native_fr_vector(bytes, offset, parsed.source_feature_point) ||
        !read_native_fr_vector(bytes, offset, parsed.node_point) ||
        !zkdl::read_fr(bytes, offset, parsed.output_claim) ||
        !zkdl::read_fr(bytes, offset, parsed.source_sum_claim) ||
        !zkdl::read_fr(bytes, offset, parsed.dst_selector_claim) ||
        !zkdl::read_fr(bytes, offset, parsed.rho) ||
        !zkdl::read_fr(bytes, offset, parsed.compressed_claim) ||
        !zkdl::read_u64(bytes, offset, sumcheck_bytes) ||
        sumcheck_bytes == 0U || sumcheck_bytes > bytes.size() - offset) {
        if (error) *error = "truncated sparse route proof header";
        return false;
    }
    std::vector<byte> sumcheck_bytes_vector(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(offset + sumcheck_bytes));
    offset += static_cast<size_t>(sumcheck_bytes);
    size_t inner = 0U;
    if (!zkdl::deserialize_sumcheck(sumcheck_bytes_vector, inner, parsed.sumcheck) ||
        inner != sumcheck_bytes_vector.size() ||
        !zkdl::deserialize_opening(bytes, offset, parsed.source_opening) ||
        !zkdl::deserialize_opening(bytes, offset, parsed.output_opening) ||
        offset != bytes.size()) {
        if (error) *error = "truncated or trailing sparse route proof";
        return false;
    }
    const auto power_of_two = [](std::uint32_t value) {
        return value != 0U && (value & (value - 1U)) == 0U;
    };
    if (!power_of_two(parsed.edge_domain) || !power_of_two(parsed.node_domain) ||
        !power_of_two(parsed.feature_domain) || !power_of_two(parsed.source_feature_domain) ||
        parsed.edge_point.size() >= 32U || parsed.feature_point.size() >= 32U ||
        parsed.source_feature_point.size() >= 32U || parsed.node_point.size() >= 32U ||
        parsed.edge_point.size() != log2_exact(parsed.edge_domain) ||
        parsed.node_point.size() != log2_exact(parsed.node_domain) ||
        parsed.feature_point.size() != log2_exact(parsed.feature_domain) ||
        parsed.source_feature_point.size() != log2_exact(parsed.source_feature_domain) ||
        parsed.sumcheck.terminal_claims.size() != 2U) {
        if (error) *error = "sparse route proof shape is not canonical";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline std::vector<byte> Proof::serialize() const {
    std::vector<byte> out;
    out.reserve(1024U);
    out.push_back('G'); out.push_back('A'); out.push_back('T'); out.push_back('R');
    zkdl::append_u32(out, version);
    append_statement(out, statement);
    zkdl::append_u64(out, static_cast<std::uint64_t>(commitments.size()));
    for (const auto& commitment : commitments) append_commitment(out, commitment);
    zkdl::append_u64(out, static_cast<std::uint64_t>(hiding_commitment_records.size()));
    for (const auto& record : hiding_commitment_records) append_hiding_record(out, record);
    zkdl::append_u64(out, static_cast<std::uint64_t>(operations.size()));
    for (const auto& operation : operations) append_operation_proof(out, operation);
    zkdl::append_u32(out, public_output_commitment);
    zkdl::serialize_opening(public_output_opening, out);
    append_digest(out, transcript_digest);
    return out;
}

inline bool Proof::deserialize(const std::vector<byte>& bytes, Proof& proof,
                               std::string* error) {
    const std::uint64_t max_items = 1U << 20;
    size_t offset = 0U;
    Proof parsed;
    std::uint32_t version = 0U;
    std::uint64_t count = 0U;
    if (bytes.size() < 4U || bytes[0] != 'G' || bytes[1] != 'A' ||
        bytes[2] != 'T' || bytes[3] != 'R') {
        if (error) *error = "invalid GAT relation proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, version) || version != kVersion ||
        !read_statement(bytes, offset, parsed.statement) ||
        !zkdl::read_u64(bytes, offset, count) || count == 0U || count > max_items) {
        if (error) *error = "invalid GAT relation proof header";
        return false;
    }
    parsed.version = version;
    parsed.commitments.resize(static_cast<size_t>(count));
    std::set<std::string> commitment_ids;
    for (auto& commitment : parsed.commitments)
        if (!read_commitment(bytes, offset, commitment)) {
            if (error) *error = "truncated GAT commitment section";
            return false;
        } else if (commitment.tensor_id.empty() || !commitment_ids.insert(commitment.tensor_id).second) {
            if (error) *error = "duplicate/empty GAT commitment id";
            return false;
        }
    if (!zkdl::read_u64(bytes, offset, count) || count > max_items) {
        if (error) *error = "invalid GAT hiding commitment section";
        return false;
    }
    parsed.hiding_commitment_records.resize(static_cast<std::size_t>(count));
    std::set<std::string> hiding_labels;
    for (auto& record : parsed.hiding_commitment_records) {
        if (!read_hiding_record(bytes, offset, record) || record.label.empty() ||
            !hiding_labels.insert(record.label).second) {
            if (error) *error = "invalid/duplicate GAT hiding commitment record";
            return false;
        }
    }
    if (!zkdl::read_u64(bytes, offset, count) || count == 0U || count > max_items) {
        if (error) *error = "GAT relation proof has no operation section";
        return false;
    }
    parsed.operations.resize(static_cast<size_t>(count));
    std::set<std::string> operation_ids;
    for (auto& operation : parsed.operations)
        if (!read_operation_proof(bytes, offset, operation)) {
            if (error) *error = "truncated GAT operation section";
            return false;
        } else if (!operation_ids.insert(operation.descriptor.id).second) {
            if (error) *error = "duplicate GAT operation id";
            return false;
        }
    if (!zkdl::read_u32(bytes, offset, parsed.public_output_commitment) ||
        !zkdl::deserialize_opening(bytes, offset, parsed.public_output_opening) ||
        !read_digest(bytes, offset, parsed.transcript_digest) || offset != bytes.size()) {
        if (error) *error = "truncated or trailing GAT relation proof bytes";
        return false;
    }
    if (parsed.public_output_commitment >= parsed.commitments.size() ||
        parsed.commitments[parsed.public_output_commitment].tensor_id != "public_Y") {
        if (error) *error = "GAT proof public output commitment is missing or not public_Y";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline void absorb_descriptor(zkdl::Transcript& transcript,
                              const OperationDescriptor& descriptor) {
    std::vector<byte> encoded;
    append_operation(encoded, descriptor);
    transcript.absorb("gat.relation.operation", encoded);
}

inline void absorb_statement(zkdl::Transcript& transcript,
                             const RelationStatement& statement,
                             const std::vector<CommitmentWire>& commitments) {
    std::vector<byte> encoded;
    append_statement(encoded, statement);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(commitments.size()));
    for (const auto& commitment : commitments) append_commitment(encoded, commitment);
    transcript.absorb("gat.relation.statement", encoded);
}

inline void fold_values(std::vector<FrWire>& values, const FrWire& challenge) {
    if (values.empty() || (values.size() & 1U) != 0U)
        throw std::invalid_argument("relation fold requires an even non-empty domain");
    std::vector<FrWire> folded(values.size() / 2U);
    const FrWire one_minus = zkdl::fr_sub(zkdl::fr_from_u32(1U), challenge);
    for (size_t i = 0; i < folded.size(); ++i)
        folded[i] = zkdl::fr_add(zkdl::fr_mul(one_minus, values[2U * i]),
                                 zkdl::fr_mul(challenge, values[2U * i + 1U]));
    values.swap(folded);
}

/* A relation prover interpolates every tensor pair at the round variable
 * before evaluating its residual. */
inline zkdl::SumcheckProof prove_constraint_exact(const ConstraintDescriptor& descriptor,
                                                  const std::vector<const TensorWitness*>& tensors,
                                                  zkdl::Transcript& transcript,
                                                  std::vector<Fr_t>& challenges,
                                                  std::vector<FrWire>& terminal_values) {
    std::string error;
    if (!descriptor.valid(&error)) throw std::invalid_argument(error);
    if (tensors.empty()) throw std::invalid_argument("relation constraint has no tensor witness");
    const std::uint32_t domain = tensors.front()->domain_size;
    const std::uint32_t bits = log2_exact(domain);
    if (bits == std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("relation domain is not a power of two");
    std::vector<std::vector<FrWire>> current;
    for (const auto* tensor : tensors) {
        if (tensor == nullptr || tensor->domain_size != domain)
            throw std::invalid_argument("relation constraint domain mismatch");
        current.push_back(tensor->values);
    }
    zkdl::SumcheckProof proof;
    proof.label = "gat.relation." + descriptor.id;
    proof.degree = descriptor.degree;
    challenges.clear();
    for (std::uint32_t round = 0U; round < bits; ++round) {
        zkdl::SumcheckRound message;
        const size_t next_size = current.front().size() / 2U;
        for (std::uint32_t evaluation = 0U; evaluation <= descriptor.degree; ++evaluation) {
            const FrWire x = zkdl::fr_from_u32(evaluation);
            FrWire total{};
            for (size_t row = 0U; row < next_size; ++row) {
                std::vector<FrWire> values;
                values.reserve(current.size());
                for (const auto& tensor : current) {
                    values.push_back(zkdl::fr_add(
                        tensor[2U * row],
                        zkdl::fr_mul(x, zkdl::fr_sub(tensor[2U * row + 1U], tensor[2U * row]))));
                }
                total = zkdl::fr_add(total, eval_constraint(descriptor, values));
            }
            message.evaluations.push_back(zkdl::fr_value(total));
        }
        proof.rounds.push_back(message);
        transcript.absorb_fr(proof.label + ".round", message.evaluations);
        const Fr_t challenge = transcript.challenge(proof.label + ".challenge");
        challenges.push_back(challenge);
        for (auto& tensor : current) fold_values(tensor, zkdl::fr_wire(challenge));
    }
    terminal_values.clear();
    for (const auto& tensor : current) terminal_values.push_back(tensor.front());
    proof.terminal_claims.reserve(terminal_values.size());
    for (const auto& value : terminal_values) proof.terminal_claims.push_back(zkdl::fr_value(value));
    proof.final_claim = zkdl::fr_value(eval_constraint(descriptor, terminal_values));
    return proof;
}

inline bool verify_constraint(const ConstraintProof& proof,
                             const ConstraintDescriptor& expected,
                             const std::vector<std::string>& expected_tensor_refs,
                             const std::vector<CommitmentWire>& commitments,
                             const std::vector<zkdl::CpuG1Point>& generators,
                             zkdl::Transcript& transcript,
                             std::string* error = nullptr) {
    std::vector<byte> expected_bytes, actual_bytes;
    append_constraint(expected_bytes, expected);
    append_constraint(actual_bytes, proof.descriptor);
    if (expected_bytes != actual_bytes) {
        if (error) *error = "relation constraint descriptor mismatch";
        return false;
    }
    const std::vector<std::uint32_t> expected_indices =
        constraint_tensor_indices(expected, expected_tensor_refs.size());
    if (proof.opening_tensor_indices != expected_indices) {
        if (error) *error = "relation constraint opening tensor map is not canonical";
        return false;
    }
    if (proof.opening_commitment_indices.size() != proof.openings.size() ||
        proof.openings.size() != expected_indices.size() ||
        proof.openings.empty() || proof.sumcheck.terminal_claims.size() != proof.openings.size()) {
        if (error) *error = "relation constraint opening/terminal count mismatch";
        return false;
    }
    if (proof.sumcheck.label != "gat.relation." + expected.id ||
        proof.sumcheck.degree != expected.degree) {
        if (error) *error = "relation Sumcheck label/degree is not bound to the expected constraint";
        return false;
    }
    std::vector<Fr_t> challenges;
    if (!zkdl::verify_sumcheck(proof.sumcheck, zkdl::fr_value(FrWire{}), transcript,
                               &challenges, error)) return false;
    if (challenges.size() != proof.openings.front().inner_point.size()) {
        if (error) *error = "relation Sumcheck/opening point dimension mismatch";
        return false;
    }
    const std::uint32_t domain = proof.openings.front().generator_size;
    if (domain == 0U || generators.size() < domain ||
        log2_exact(domain) == std::numeric_limits<std::uint32_t>::max() ||
        challenges.size() > kMaxRelationPointBits ||
        domain > (std::uint32_t(1U) << kMaxRelationPointBits) ||
        domain != (std::uint32_t(1U) << challenges.size())) {
        if (error) *error = "relation verifier has no authenticated generator domain";
        return false;
    }
    std::vector<FrWire> terminal_values;
    terminal_values.reserve(proof.openings.size());
    for (size_t i = 0U; i < proof.openings.size(); ++i) {
        const std::uint32_t tensor_index = expected_indices[i];
        const std::uint32_t index = proof.opening_commitment_indices[i];
        if (index >= commitments.size() || index >= commitments.size() ||
            tensor_index >= expected_tensor_refs.size() ||
            commitments[index].tensor_id != expected_tensor_refs[tensor_index] ||
            commitments[index].domain_size != domain ||
            proof.openings[i].commitment_index != index ||
            proof.openings[i].generator_size != domain ||
            proof.openings[i].inner_point.size() != challenges.size() ||
            !proof.openings[i].outer_point.empty()) {
            if (error) *error = "relation opening commitment reference mismatch";
            return false;
        }
        for (size_t round = 0U; round < challenges.size(); ++round)
            if (!same_wire(zkdl::fr_wire(proof.openings[i].inner_point[round]),
                           zkdl::fr_wire(challenges[round]))) {
                if (error) *error = "relation opening point is not Fiat-Shamir bound";
                return false;
            }
        const zkdl::CpuG1Point expected_commitment = zkdl::cpu_g1_from_native(commitments[index].commitment);
        if (!zkdl::cpu_verify_opening(proof.openings[i], expected_commitment, error)) return false;
        terminal_values.push_back(zkdl::fr_wire(proof.openings[i].value));
        if (!same_wire(terminal_values.back(), zkdl::fr_wire(proof.sumcheck.terminal_claims[i]))) {
            if (error) *error = "relation terminal claim is not opening-bound";
            return false;
        }
    }
    std::vector<FrWire> full_terminal_values(expected_tensor_refs.size(), FrWire{});
    for (size_t i = 0U; i < expected_indices.size(); ++i)
        full_terminal_values[expected_indices[i]] = terminal_values[i];
    if (!same_wire(zkdl::fr_wire(proof.sumcheck.final_claim),
                   eval_constraint(expected, full_terminal_values))) {
        if (error) *error = "relation terminal equation failed";
        return false;
    }

    const std::uint32_t point_bits = log2_exact(domain);
    if (proof.pointwise_point.size() != point_bits ||
        proof.pointwise_openings.size() != expected_indices.size()) {
        if (error) *error = "relation pointwise residual opening count/shape mismatch";
        return false;
    }
    for (std::uint32_t bit = 0U; bit < point_bits; ++bit) {
        const Fr_t expected_point = transcript.challenge(
            "gat.relation." + expected.id + ".pointwise." + std::to_string(bit));
        if (fr_compare_mixed(expected_point, proof.pointwise_point[bit]) != 0) {
            if (error) *error = "relation pointwise check point is not Fiat-Shamir bound";
            return false;
        }
    }
    std::vector<FrWire> pointwise_values(expected_tensor_refs.size(), FrWire{});
    for (size_t i = 0U; i < expected_indices.size(); ++i) {
        const std::uint32_t tensor_index = expected_indices[i];
        const std::uint32_t index = proof.opening_commitment_indices[i];
        const auto& opening = proof.pointwise_openings[i];
        if (tensor_index >= expected_tensor_refs.size() ||
            index >= commitments.size() ||
            commitments[index].tensor_id != expected_tensor_refs[tensor_index] ||
            opening.commitment_index != index ||
            opening.generator_size != domain ||
            !opening.outer_point.empty() ||
            opening.inner_point.size() != proof.pointwise_point.size()) {
            if (error) *error = "relation pointwise opening commitment reference mismatch";
            return false;
        }
        for (size_t bit = 0U; bit < proof.pointwise_point.size(); ++bit)
            if (!same_wire(zkdl::fr_wire(opening.inner_point[bit]),
                           zkdl::fr_wire(proof.pointwise_point[bit]))) {
                if (error) *error = "relation pointwise opening point mismatch";
                return false;
            }
        if (!zkdl::cpu_verify_opening(
                opening, zkdl::cpu_g1_from_native(commitments[index].commitment), error))
            return false;
        pointwise_values[tensor_index] = zkdl::fr_wire(opening.value);
    }
    if (!same_wire(eval_constraint(expected, pointwise_values), FrWire{})) {
        if (error) *error = "relation pointwise residual equation failed";
        return false;
    }
    std::vector<byte> pointwise_bytes;
    for (const auto& opening : proof.pointwise_openings)
        zkdl::serialize_opening(opening, pointwise_bytes);
    transcript.absorb("gat.relation." + expected.id + ".pointwise.openings",
                      pointwise_bytes);
    return true;
}

inline ConstraintDescriptor fac_line8_descriptor(const OperationDescriptor& operation,
                                                 const Fr_t& rho) {
    ConstraintDescriptor descriptor;
    descriptor.id = operation.id + "/fac4dnn-line8";
    descriptor.degree = 1U;
    FrWire coefficient = zkdl::fr_from_u32(1U);
    const FrWire rho_wire = zkdl::fr_wire(rho);
    for (const auto& binding : operation.fac_bindings) {
        const std::uint32_t view = static_cast<std::uint32_t>(
            std::distance(operation.tensor_refs.begin(),
                          std::find(operation.tensor_refs.begin(), operation.tensor_refs.end(),
                                    binding.view_ref)));
        const std::uint32_t canonical = static_cast<std::uint32_t>(
            std::distance(operation.tensor_refs.begin(),
                          std::find(operation.tensor_refs.begin(), operation.tensor_refs.end(),
                                    binding.canonical_ref)));
        descriptor.terms.push_back(Monomial{coefficient, {view}});
        descriptor.terms.push_back(Monomial{zkdl::fr_neg(coefficient), {canonical}});
        coefficient = zkdl::fr_mul(coefficient, rho_wire);
    }
    return descriptor;
}

inline ConstraintDescriptor batched_family_descriptor(
    const OperationDescriptor& operation, zkdl::Transcript& transcript,
    std::string* error = nullptr) {
    ConstraintDescriptor descriptor;
    descriptor.id = operation.id + "/batch/" + operation.batch_family;
    descriptor.degree = 1U;
    for (const auto& constraint : operation.constraints)
        descriptor.degree = std::max(descriptor.degree, constraint.degree);
    if (operation.batch_family.empty() || operation.constraints.empty() ||
        operation.constraints.size() > kMaxRelationConstraints) {
        if (error) *error = "relation batched family has an invalid base constraint stack";
        return ConstraintDescriptor{};
    }
    FrWire weight = zkdl::fr_from_u32(1U);
    for (std::size_t index = 0U; index < operation.constraints.size(); ++index) {
        const Fr_t challenge = transcript.challenge(
            "gat.relation." + operation.id + ".batch." + operation.batch_family +
            ".rho." + std::to_string(index));
        weight = zkdl::fr_wire(challenge);
        const auto& constraint = operation.constraints[index];
        descriptor.constant = zkdl::fr_add(
            descriptor.constant, zkdl::fr_mul(weight, constraint.constant));
        for (const auto& term : constraint.terms) {
            Monomial combined = term;
            combined.coefficient = zkdl::fr_mul(weight, term.coefficient);
            descriptor.terms.push_back(std::move(combined));
        }
    }
    if (descriptor.terms.empty() || descriptor.terms.size() > kMaxRelationTerms ||
        !descriptor.valid(error)) return ConstraintDescriptor{};
    return descriptor;
}

inline bool fill_constraint_openings(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::string>& operation_refs,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<CommitmentWire>& commitments,
    zkdl::Transcript& transcript, ConstraintProof& proof,
    std::string* error = nullptr) {
    if (operation_refs.empty()) {
        if (error) *error = "relation batched proof has no operation tensor references";
        return false;
    }
    std::vector<const TensorWitness*> witnesses;
    witnesses.reserve(operation_refs.size());
    for (const auto& ref : operation_refs) {
        const auto found = tensor_map.find(ref);
        if (found == tensor_map.end() || found->second == nullptr) {
            if (error) *error = "relation batched proof tensor reference is missing: " + ref;
            return false;
        }
        witnesses.push_back(found->second);
    }
    try {
        std::vector<Fr_t> challenges;
        std::vector<FrWire> terminal_values;
        proof.descriptor = descriptor;
        proof.sumcheck = prove_constraint_exact(descriptor, witnesses, transcript,
                                                challenges, terminal_values);
        if (challenges.size() > kMaxRelationPointBits) {
            if (error) *error = "relation batched proof exceeds the native MLE point cap";
            return false;
        }
        proof.opening_tensor_indices = constraint_tensor_indices(
            descriptor, operation_refs.size());
        if (proof.opening_tensor_indices.empty()) {
            if (error) *error = "relation batched proof has no canonical tensor opening";
            return false;
        }
        std::vector<FrWire> point;
        point.reserve(challenges.size());
        for (const auto& challenge : challenges) point.push_back(zkdl::fr_wire(challenge));
        proof.opening_commitment_indices.reserve(proof.opening_tensor_indices.size());
        proof.openings.reserve(proof.opening_tensor_indices.size());
        for (const auto tensor_index : proof.opening_tensor_indices) {
            if (tensor_index >= operation_refs.size()) {
                if (error) *error = "relation batched proof tensor opening index is out of bounds";
                return false;
            }
            const std::string& ref = operation_refs[tensor_index];
            const auto commitment = std::find_if(
                commitments.begin(), commitments.end(),
                [&](const CommitmentWire& value) { return value.tensor_id == ref; });
            if (commitment == commitments.end()) {
                if (error) *error = "relation batched proof commitment reference is missing: " + ref;
                return false;
            }
            const TensorWitness* tensor = tensor_map.at(ref);
            if (tensor->domain_size != (std::size_t(1U) << challenges.size()) ||
                generators.size() < tensor->domain_size) {
                if (error) *error = "relation batched proof opening domain is not authenticated";
                return false;
            }
            const std::uint32_t commitment_index = static_cast<std::uint32_t>(
                std::distance(commitments.begin(), commitment));
            proof.opening_commitment_indices.push_back(commitment_index);
            auto opening = zkdl::cpu_open(
                std::vector<zkdl::CpuG1Point>(generators.begin(),
                                              generators.begin() + tensor->domain_size),
                tensor->values, point);
            opening.commitment_index = commitment_index;
            proof.openings.push_back(std::move(opening));
        }
        proof.sumcheck.terminal_claims.clear();
        for (const auto tensor_index : proof.opening_tensor_indices)
            proof.sumcheck.terminal_claims.push_back(
                zkdl::fr_value(terminal_values.at(tensor_index)));
        const std::uint32_t point_bits = log2_exact(
            tensor_map.at(operation_refs.front())->domain_size);
        if (point_bits == std::numeric_limits<std::uint32_t>::max()) {
            if (error) *error = "relation batched proof pointwise domain is not a power of two";
            return false;
        }
        if (point_bits > kMaxRelationPointBits) {
            if (error) *error = "relation batched proof pointwise domain exceeds the native MLE cap";
            return false;
        }
        proof.pointwise_point.reserve(point_bits);
        for (std::uint32_t bit = 0U; bit < point_bits; ++bit)
            proof.pointwise_point.push_back(transcript.challenge(
                "gat.relation." + descriptor.id + ".pointwise." + std::to_string(bit)));
        std::vector<FrWire> pointwise;
        pointwise.reserve(proof.pointwise_point.size());
        for (const auto& challenge : proof.pointwise_point)
            pointwise.push_back(zkdl::fr_wire(challenge));
        for (const auto tensor_index : proof.opening_tensor_indices) {
            const std::string& ref = operation_refs[tensor_index];
            const TensorWitness* tensor = tensor_map.at(ref);
            const auto commitment = std::find_if(
                commitments.begin(), commitments.end(),
                [&](const CommitmentWire& value) { return value.tensor_id == ref; });
            if (commitment == commitments.end()) return false;
            const std::uint32_t commitment_index = static_cast<std::uint32_t>(
                std::distance(commitments.begin(), commitment));
            auto opening = zkdl::cpu_open(
                std::vector<zkdl::CpuG1Point>(generators.begin(),
                                              generators.begin() + tensor->domain_size),
                tensor->values, pointwise);
            opening.commitment_index = commitment_index;
            proof.pointwise_openings.push_back(std::move(opening));
        }
        std::vector<byte> pointwise_bytes;
        for (const auto& opening : proof.pointwise_openings)
            zkdl::serialize_opening(opening, pointwise_bytes);
        transcript.absorb("gat.relation." + descriptor.id + ".pointwise.openings",
                          pointwise_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline std::vector<byte> serialize_constraint_payload(const ConstraintProof& proof) {
    std::vector<byte> encoded{'B', 'C', 'H', '1'};
    append_constraint_proof(encoded, proof);
    return encoded;
}

inline bool deserialize_constraint_payload(const std::vector<byte>& bytes,
                                           ConstraintProof& proof,
                                           std::string* error) {
    if (bytes.size() < 4U || bytes[0] != 'B' || bytes[1] != 'C' ||
        bytes[2] != 'H' || bytes[3] != '1') {
        if (error) *error = "invalid batched relation proof magic";
        return false;
    }
    size_t offset = 4U;
    if (!read_constraint_proof(bytes, offset, proof) || offset != bytes.size()) {
        if (error) *error = "truncated or trailing batched relation proof";
        return false;
    }
    return true;
}

inline bool prove_sparse_route(
    const OperationDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<CommitmentWire>& commitments,
    zkdl::Transcript& transcript, SparseRouteProof& route,
    std::string* error);

inline bool verify_sparse_route(
    const OperationDescriptor& descriptor,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    zkdl::Transcript& transcript, const SparseRouteProof& route,
    std::string* error);

inline bool prove_fac_line8(
    const OperationDescriptor& operation,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<CommitmentWire>& commitments,
    zkdl::Transcript& transcript, FacLine8Proof& proof,
    std::string* error = nullptr) {
    if (operation.fac_bindings.empty()) {
        if (error) *error = "FAC4DNN proof requested with an empty public binding list";
        return false;
    }
    proof = FacLine8Proof{};
    proof.selector_claims.reserve(operation.fac_bindings.size());
    proof.selector_proofs.reserve(operation.fac_bindings.size());
    for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index) {
        const FacBindingDescriptor& binding = operation.fac_bindings[index];
        OperationDescriptor selector;
        selector.id = operation.id + "/fac4dnn-line8/selector/" + std::to_string(index);
        selector.kind = RelationKind::SparseRoute;
        selector.degree = 2U;
        selector.source_refs = {binding.canonical_ref, binding.view_ref};
        selector.tensor_refs = {binding.canonical_ref, binding.view_ref};
        selector.mapping = operation.mapping + ";fac4dnn-public-selector";
        selector.padding = operation.padding + ";fac4dnn-selector-masked";
        selector.terminal_claim = operation.terminal_claim;
        selector.terminal_point = operation.terminal_point;
        selector.public_route = true;
        selector.requires_reindex = true;
        selector.selector_edge_domain = binding.view_outer_domain;
        selector.selector_node_domain = binding.canonical_outer_domain;
        selector.selector_feature_domain = binding.view_feature_domain;
        selector.selector_source_feature_domain = binding.canonical_feature_domain;
        selector.selector_src = binding.source_outer_index;
        selector.selector_dst = binding.source_outer_index;
        selector.selector_active = binding.selector_active;
        selector.selector_feature_map = binding.source_feature_index;
        selector.selector_source_ref = binding.canonical_ref;
        selector.selector_output_ref = binding.view_ref;
        SparseRouteProof route;
        if (!prove_sparse_route(selector, tensor_map, generators, commitments,
                                transcript, route, error)) return false;
        const std::vector<byte> encoded = route.serialize();
        transcript.absorb("gat.fac4dnn.line8." + operation.id + ".selector." +
                          std::to_string(index), encoded);
        proof.selector_claims.push_back(zkdl::fr_value(zkdl::fr_sub(
            zkdl::fr_wire(route.output_claim), zkdl::fr_wire(route.source_sum_claim))));
        proof.selector_proofs.push_back(std::move(route));
    }
    proof.rho = transcript.challenge("gat.fac4dnn.line8." + operation.id + ".rho");
    FrWire compressed{};
    FrWire power = zkdl::fr_from_u32(1U);
    for (const auto& claim : proof.selector_claims) {
        compressed = zkdl::fr_add(compressed,
                                  zkdl::fr_mul(power, zkdl::fr_wire(claim)));
        power = zkdl::fr_mul(power, zkdl::fr_wire(proof.rho));
    }
    proof.compressed_claim = zkdl::fr_value(compressed);
    transcript.absorb_fr("gat.fac4dnn.line8." + operation.id + ".eq30",
                         {proof.compressed_claim});
    return true;
}

inline bool verify_fac_line8(
    const OperationDescriptor& operation,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    zkdl::Transcript& transcript, const FacLine8Proof& proof,
    std::string* error = nullptr) {
    if (operation.fac_bindings.empty()) {
        if (error) *error = "FAC4DNN proof carries no expected public binding list";
        return false;
    }
    if (proof.version != FacLine8Proof::kVersion ||
        proof.selector_claims.size() != operation.fac_bindings.size() ||
        proof.selector_proofs.size() != operation.fac_bindings.size()) {
        if (error) *error = "FAC4DNN selector proof count/version mismatch";
        return false;
    }
    for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index) {
        const FacBindingDescriptor& binding = operation.fac_bindings[index];
        OperationDescriptor selector;
        selector.id = operation.id + "/fac4dnn-line8/selector/" + std::to_string(index);
        selector.kind = RelationKind::SparseRoute;
        selector.degree = 2U;
        selector.source_refs = {binding.canonical_ref, binding.view_ref};
        selector.tensor_refs = {binding.canonical_ref, binding.view_ref};
        selector.mapping = operation.mapping + ";fac4dnn-public-selector";
        selector.padding = operation.padding + ";fac4dnn-selector-masked";
        selector.terminal_claim = operation.terminal_claim;
        selector.terminal_point = operation.terminal_point;
        selector.public_route = true;
        selector.requires_reindex = true;
        selector.selector_edge_domain = binding.view_outer_domain;
        selector.selector_node_domain = binding.canonical_outer_domain;
        selector.selector_feature_domain = binding.view_feature_domain;
        selector.selector_source_feature_domain = binding.canonical_feature_domain;
        selector.selector_src = binding.source_outer_index;
        selector.selector_dst = binding.source_outer_index;
        selector.selector_active = binding.selector_active;
        selector.selector_feature_map = binding.source_feature_index;
        selector.selector_source_ref = binding.canonical_ref;
        selector.selector_output_ref = binding.view_ref;
        if (!verify_sparse_route(selector, commitments, generators, transcript,
                                 proof.selector_proofs[index], error)) return false;
        const std::vector<byte> encoded = proof.selector_proofs[index].serialize();
        transcript.absorb("gat.fac4dnn.line8." + operation.id + ".selector." +
                          std::to_string(index), encoded);
        const Fr_t expected_claim = zkdl::fr_value(zkdl::fr_sub(
            zkdl::fr_wire(proof.selector_proofs[index].output_claim),
            zkdl::fr_wire(proof.selector_proofs[index].source_sum_claim)));
        if (fr_compare_mixed(expected_claim, proof.selector_claims[index]) != 0) {
            if (error) *error = "FAC4DNN selector residual claim mismatch";
            return false;
        }
    }
    const Fr_t expected_rho = transcript.challenge(
        "gat.fac4dnn.line8." + operation.id + ".rho");
    if (fr_compare_mixed(expected_rho, proof.rho) != 0) {
        if (error) *error = "FAC4DNN rho is not transcript-bound";
        return false;
    }
    FrWire compressed{};
    FrWire power = zkdl::fr_from_u32(1U);
    for (const auto& claim : proof.selector_claims) {
        compressed = zkdl::fr_add(compressed,
                                  zkdl::fr_mul(power, zkdl::fr_wire(claim)));
        power = zkdl::fr_mul(power, zkdl::fr_wire(expected_rho));
    }
    if (fr_compare_mixed(zkdl::fr_value(compressed), proof.compressed_claim) != 0) {
        if (error) *error = "FAC4DNN compressed claim does not match terminal equation";
        return false;
    }
    transcript.absorb_fr("gat.fac4dnn.line8." + operation.id + ".eq30",
                         {proof.compressed_claim});
    return true;
}

inline bool descriptor_vector_valid(const std::vector<OperationDescriptor>& operations,
                                   std::string* error = nullptr) {
    std::set<std::string> ids;
    for (const auto& operation : operations) {
        if (!operation.valid(error) || !ids.insert(operation.id).second) {
            if (error && error->empty()) *error = "duplicate relation operation id";
            return false;
        }
    }
    return !operations.empty();
}

inline const TensorDescriptor* find_tensor_descriptor(
    const std::vector<TensorDescriptor>& descriptors, const std::string& id) {
    const auto found = std::find_if(descriptors.begin(), descriptors.end(),
                                    [&](const TensorDescriptor& descriptor) {
                                        return descriptor.id == id;
                                    });
    return found == descriptors.end() ? nullptr : &*found;
}

inline bool commitment_equal(const CommitmentWire& left,
                             const CommitmentWire& right) noexcept {
    return left.tensor_id == right.tensor_id && left.visibility == right.visibility &&
           left.domain_size == right.domain_size &&
           zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(left.commitment),
                              zkdl::cpu_g1_from_native(right.commitment));
}

inline bool verify_tensor_contract(const VerifierContext& context,
                                   const std::vector<CommitmentWire>& commitments,
                                   std::string* error = nullptr) {
    std::set<std::string> descriptor_ids;
    for (const auto& descriptor : context.tensor_descriptors) {
        if (!descriptor.valid(error) || !descriptor_ids.insert(descriptor.id).second) {
            if (error && error->empty()) *error = "duplicate/invalid verifier tensor descriptor";
            return false;
        }
    }
    if (!context.tensor_descriptors.empty()) {
        for (const auto& commitment : commitments) {
            const TensorDescriptor* descriptor = find_tensor_descriptor(
                context.tensor_descriptors, commitment.tensor_id);
            if (descriptor == nullptr || descriptor->domain_size != commitment.domain_size ||
                descriptor->visibility != commitment.visibility) {
                if (error) *error = "commitment does not match public tensor shape/padding contract";
                return false;
            }
        }
        for (const auto& descriptor : context.tensor_descriptors) {
            const auto found = std::find_if(commitments.begin(), commitments.end(),
                                            [&](const CommitmentWire& commitment) {
                                                return commitment.tensor_id == descriptor.id;
                                            });
            if (found == commitments.end()) {
                if (error) *error = "proof omits a tensor from the public shape contract: " + descriptor.id;
                return false;
            }
        }
    }
    for (const auto& operation : context.operations) {
        if (operation.batch_family == "indexed_public_table") {
            const TensorDescriptor* table = find_tensor_descriptor(
                context.tensor_descriptors, operation.public_table_ref);
            const auto table_values = context.public_tensor_values.find(
                operation.public_table_ref);
            if (table == nullptr || table->visibility != 1U ||
                table_values == context.public_tensor_values.end()) {
                if (error) *error = "indexed table relation lacks an authenticated public table descriptor/value map";
                return false;
            }
        }
        std::uint32_t aligned_domain = 0U;
        for (const auto& ref : operation.tensor_refs) {
            const TensorDescriptor* descriptor = find_tensor_descriptor(
                context.tensor_descriptors, ref);
            if (descriptor == nullptr) continue;
            if (aligned_domain == 0U) aligned_domain = descriptor->domain_size;
            const bool has_route = operation.kind == RelationKind::SparseRoute ||
                                   !operation.selector_source_ref.empty();
            if (!has_route &&
                descriptor->domain_size != aligned_domain) {
                if (error) *error = "operation tensor views violate authenticated MLE padding/domain contract";
                return false;
            }
        }
        for (const auto& primitive : operation.primitives) {
            const std::array<std::uint32_t, 3U> expected = {
                primitive.matmul_shape.a_size(), primitive.matmul_shape.b_size(),
                primitive.matmul_shape.c_size()};
            for (std::size_t i = 0U; i < primitive.tensor_refs.size(); ++i) {
                const TensorDescriptor* descriptor = find_tensor_descriptor(
                    context.tensor_descriptors, primitive.tensor_refs[i]);
                if (descriptor == nullptr || descriptor->domain_size != expected[i]) {
                    if (error) *error = "zkMatMul primitive shape is not bound to public tensor descriptor";
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool verify_model_registry(
    const std::vector<CommitmentWire>& commitments,
    const std::vector<CommitmentWire>& registered_model_commitments,
    std::string* error = nullptr) {
    std::set<std::string> ids;
    for (const auto& registered : registered_model_commitments) {
        if (registered.visibility != 2U || registered.tensor_id.empty() ||
            !ids.insert(registered.tensor_id).second) {
            if (error) *error = "authenticated model registry contains duplicate/non-model commitment";
            return false;
        }
        const auto found = std::find_if(commitments.begin(), commitments.end(),
                                       [&](const CommitmentWire& commitment) {
                                           return commitment.tensor_id == registered.tensor_id;
                                       });
        if (found == commitments.end() || !commitment_equal(*found, registered)) {
            if (error) *error = "model tensor commitment does not match authenticated registry/VK map";
            return false;
        }
    }
    for (const auto& commitment : commitments) {
        if (commitment.visibility != 2U) continue;
        const auto found = std::find_if(registered_model_commitments.begin(),
                                       registered_model_commitments.end(),
                                       [&](const CommitmentWire& registered) {
                                           return registered.tensor_id == commitment.tensor_id;
                                       });
        if (found == registered_model_commitments.end()) {
            if (error) *error = "model tensor is not present in authenticated registry/VK map";
            return false;
        }
    }
    return true;
}

inline bool verify_hiding_records(
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CommitmentRecord>& records,
    const std::vector<zkdl::CpuG1Point>& generators,
    std::string* error = nullptr) {
    std::set<std::string> labels;
    for (const auto& record : records) {
        if (!zkdl::CpuHidingCommitment::verify_record(record, error) ||
            record.label.empty() || !labels.insert(record.label).second) {
            if (error && error->empty()) *error = "invalid/duplicate native hiding commitment record";
            return false;
        }
        const auto found = std::find_if(commitments.begin(), commitments.end(),
                                       [&](const CommitmentWire& commitment) {
                                           return commitment.tensor_id == record.label;
                                       });
        const std::uint32_t expected_record_visibility =
            found == commitments.end() ? 0U : (found->visibility == 2U ? 1U :
                                               (found->visibility == 3U ? 2U : 0U));
        if (found == commitments.end() || expected_record_visibility == 0U ||
            record.visibility != expected_record_visibility ||
            record.points.size() != found->domain_size ||
            generators.size() < found->domain_size ||
            !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(record.value_commitment),
                                zkdl::cpu_g1_from_native(found->commitment))) {
            if (error) *error = "hiding commitment record is not bound to the canonical tensor commitment";
            return false;
        }
        for (std::size_t point = 0U; point < record.points.size(); ++point)
            if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(record.points[point]),
                                    generators[point])) {
                if (error) *error = "hiding commitment record generator vector is not authenticated";
                return false;
            }
    }
    for (const auto& commitment : commitments) {
        if (commitment.visibility == 1U) continue;
        if (!labels.count(commitment.tensor_id)) {
            if (error) *error = "private/model tensor is missing a native hiding commitment record";
            return false;
        }
    }
    return true;
}

inline bool hiding_records_equal(const std::vector<zkdl::CommitmentRecord>& left,
                                 const std::vector<zkdl::CommitmentRecord>& right) {
    if (left.size() != right.size()) return false;
    std::map<std::string, std::vector<byte>> lhs, rhs;
    for (const auto& record : left) {
        std::vector<byte> encoded;
        append_hiding_record(encoded, record);
        if (!lhs.emplace(record.label, std::move(encoded)).second) return false;
    }
    for (const auto& record : right) {
        std::vector<byte> encoded;
        append_hiding_record(encoded, record);
        if (!rhs.emplace(record.label, std::move(encoded)).second) return false;
    }
    return lhs == rhs;
}

inline void absorb_hiding_records(
    zkdl::Transcript& transcript,
    const std::vector<zkdl::CommitmentRecord>& records) {
    std::map<std::string, const zkdl::CommitmentRecord*> ordered;
    for (const auto& record : records) ordered.emplace(record.label, &record);
    std::vector<byte> encoded;
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(ordered.size()));
    for (const auto& entry : ordered) append_hiding_record(encoded, *entry.second);
    transcript.absorb("gat.relation.hiding-records", encoded);
}

inline bool verify_public_tensor_values(
    const VerifierContext& context, const std::vector<CommitmentWire>& commitments,
    std::string* error = nullptr) {
    for (const auto& entry : context.public_tensor_values) {
        const auto found = std::find_if(commitments.begin(), commitments.end(),
                                       [&](const CommitmentWire& commitment) {
                                           return commitment.tensor_id == entry.first;
                                       });
        const bool canonical = std::all_of(entry.second.begin(), entry.second.end(),
                                           [](const FrWire& value) {
                                               return zkdl::fr_canonical(value);
                                           });
        const bool graph_public = context.graph_public_tensor_ids.count(entry.first) != 0U;
        if (found == commitments.end() ||
            (!graph_public && found->visibility != 1U) ||
            entry.second.size() != found->domain_size || !canonical) {
            if (error) *error = "public tensor value material does not match commitment domain";
            return false;
        }
        if (found->domain_size > context.generators.size()) {
            if (error) *error = "public tensor commitment exceeds authenticated generator domain";
            return false;
        }
        const std::vector<zkdl::CpuG1Point> generators(
            context.generators.begin(), context.generators.begin() + found->domain_size);
        const G1Jacobian_t expected = zkdl::cpu_g1_to_native(
            zkdl::cpu_g1_sum(generators, entry.second));
        if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(found->commitment),
                                zkdl::cpu_g1_from_native(expected))) {
            if (error) *error = "public tensor commitment is not graph/table-derived";
            return false;
        }
    }
    return true;
}

inline bool primitive_tensor_values(
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::string& id, std::uint64_t expected_size,
    std::vector<Fr_t>& values, std::string* error) {
    const auto found = tensor_map.find(id);
    if (found == tensor_map.end() || found->second == nullptr ||
        expected_size > std::numeric_limits<std::uint32_t>::max() ||
        found->second->domain_size != expected_size ||
        found->second->values.size() != expected_size) {
        if (error) *error = "zkMatMul primitive tensor shape/value mismatch: " + id;
        return false;
    }
    values.clear();
    values.reserve(found->second->values.size());
    for (const auto& value : found->second->values) values.push_back(zkdl::fr_value(value));
    return true;
}

inline bool prove_matmul_primitive(
    const PrimitiveDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    std::vector<byte>& encoded, std::string* error) {
    if (descriptor.kind != 1U || descriptor.tensor_refs.size() != 3U) {
        if (error) *error = "unsupported native relation primitive";
        return false;
    }
    const zkdl::MatMulShape& shape = descriptor.matmul_shape;
    std::string shape_error;
    if (!shape.valid(&shape_error)) {
        if (error) *error = "zkMatMul primitive shape rejected: " + shape_error;
        return false;
    }
    const std::uint32_t generator_count = std::max(shape.n, std::max(shape.k, 1U));
    if (generators.size() < generator_count) {
        if (error) *error = "authenticated SRS generator domain is too small for zkMatMul";
        return false;
    }
    std::vector<Fr_t> a_values, b_values, c_values;
    if (!primitive_tensor_values(tensor_map, descriptor.tensor_refs[0], shape.a_size(),
                                 a_values, error) ||
        !primitive_tensor_values(tensor_map, descriptor.tensor_refs[1], shape.b_size(),
                                 b_values, error) ||
        !primitive_tensor_values(tensor_map, descriptor.tensor_refs[2], shape.c_size(),
                                 c_values, error)) return false;
    std::vector<G1Jacobian_t> native_generators;
    native_generators.reserve(generator_count);
    for (std::uint32_t i = 0U; i < generator_count; ++i)
        native_generators.push_back(zkdl::cpu_g1_to_native(generators[i]));
    try {
        Commitment gen_a(shape.n, native_generators.data());
        Commitment gen_b(shape.k, native_generators.data());
        Commitment gen_c(shape.k, native_generators.data());
        FrTensor a(shape.a_size(), a_values.data());
        FrTensor b(shape.b_size(), b_values.data());
        FrTensor c(shape.c_size(), c_values.data());
        const zkdl::MatMulProof proof = zkdl::matmul_prove(
            a, b, c, shape, gen_a, gen_b, gen_c);
        encoded = proof.serialize();
        if (encoded.empty()) {
            if (error) *error = "zkMatMul primitive prover returned an empty proof";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = std::string("zkMatMul primitive prover failed: ") + exception.what();
        return false;
    }
}

inline bool verify_matmul_primitive(
    const PrimitiveDescriptor& descriptor, const std::vector<byte>& encoded,
    const std::vector<zkdl::CpuG1Point>& generators, std::string* error) {
    if (descriptor.kind != 1U || descriptor.tensor_refs.size() != 3U || encoded.empty()) {
        if (error) *error = "invalid native relation primitive proof";
        return false;
    }
    const zkdl::MatMulShape& shape = descriptor.matmul_shape;
    std::string shape_error;
    if (!shape.valid(&shape_error)) {
        if (error) *error = "zkMatMul primitive shape rejected: " + shape_error;
        return false;
    }
    const std::uint32_t generator_count = std::max(shape.n, std::max(shape.k, 1U));
    if (generators.size() < generator_count) {
        if (error) *error = "authenticated SRS generator domain is too small for zkMatMul";
        return false;
    }
    zkdl::MatMulProof proof;
    if (!zkdl::MatMulProof::deserialize(encoded, proof, error)) return false;
    if (proof.shape != shape || proof.c_binding_mode != zkdl::MatMulProof::kCCommitment) {
        if (error) *error = "zkMatMul primitive proof shape/binding mismatch";
        return false;
    }
    zkdl::MatMulCpuCommitmentView a_view, b_view, c_view;
    a_view.generators.assign(generators.begin(), generators.begin() + shape.n);
    b_view.generators.assign(generators.begin(), generators.begin() + shape.k);
    c_view.generators.assign(generators.begin(), generators.begin() + shape.k);
    a_view.commitments.reserve(proof.commitment_a.size());
    b_view.commitments.reserve(proof.commitment_b.size());
    c_view.commitments.reserve(proof.commitment_c.size());
    for (const auto& point : proof.commitment_a) a_view.commitments.push_back(zkdl::cpu_g1_from_native(point));
    for (const auto& point : proof.commitment_b) b_view.commitments.push_back(zkdl::cpu_g1_from_native(point));
    for (const auto& point : proof.commitment_c) c_view.commitments.push_back(zkdl::cpu_g1_from_native(point));
    return zkdl::matmul_verify_cpu(proof, shape, a_view, b_view, &c_view, nullptr, error);
}

inline bool primitive_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(commitments.begin(), commitments.end(),
                                    [&](const CommitmentWire& value) {
                                        return value.tensor_id == tensor_id;
                                    });
    if (found == commitments.end()) {
        if (error) *error = "zkMatMul canonical commitment is missing: " + tensor_id;
        return false;
    }
    index = static_cast<std::uint32_t>(std::distance(commitments.begin(), found));
    return true;
}

inline std::vector<FrWire> primitive_opening_point(
    const zkdl::EvaluationOpeningProof& opening) {
    std::vector<FrWire> point;
    point.reserve(opening.inner_point.size() + opening.outer_point.size());
    for (const auto& value : opening.inner_point) point.push_back(zkdl::fr_wire(value));
    for (const auto& value : opening.outer_point) point.push_back(zkdl::fr_wire(value));
    return point;
}

inline bool prove_primitive_binding(
    const PrimitiveDescriptor& descriptor, const std::vector<byte>& encoded,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<CommitmentWire>& commitments,
    OperationProof::PrimitiveBinding& binding, std::string* error) {
    zkdl::MatMulProof primitive;
    if (!zkdl::MatMulProof::deserialize(encoded, primitive, error) ||
        primitive.shape != descriptor.matmul_shape ||
        primitive.c_binding_mode != zkdl::MatMulProof::kCCommitment ||
        !primitive.has_c_opening) {
        if (error && error->empty()) *error = "zkMatMul primitive has no canonical C opening";
        return false;
    }
    const std::uint32_t a_domain = descriptor.matmul_shape.a_size();
    const std::uint32_t b_domain = descriptor.matmul_shape.b_size();
    const std::uint32_t c_domain = descriptor.matmul_shape.c_size();
    const auto a = tensor_map.find(descriptor.tensor_refs[0]);
    const auto b = tensor_map.find(descriptor.tensor_refs[1]);
    const auto c = tensor_map.find(descriptor.tensor_refs[2]);
    if (a == tensor_map.end() || b == tensor_map.end() || c == tensor_map.end() ||
        a->second == nullptr || b->second == nullptr || c->second == nullptr ||
        a->second->domain_size != a_domain || b->second->domain_size != b_domain ||
        c->second->domain_size != c_domain || generators.size() <
            std::max(a_domain, std::max(b_domain, c_domain))) {
        if (error) *error = "zkMatMul canonical binding tensor shape is invalid";
        return false;
    }
    if (!primitive_commitment_index(commitments, descriptor.tensor_refs[0],
                                    binding.a_commitment_index, error) ||
        !primitive_commitment_index(commitments, descriptor.tensor_refs[1],
                                    binding.b_commitment_index, error) ||
        !primitive_commitment_index(commitments, descriptor.tensor_refs[2],
                                    binding.c_commitment_index, error)) return false;
    const std::vector<FrWire> point_a = primitive_opening_point(primitive.opening_a);
    const std::vector<FrWire> point_b = primitive_opening_point(primitive.opening_b);
    const std::vector<FrWire> point_c = primitive_opening_point(primitive.opening_c);
    binding.opening_a = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + a_domain),
        a->second->values, point_a);
    binding.opening_b = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + b_domain),
        b->second->values, point_b);
    binding.opening_c = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + c_domain),
        c->second->values, point_c);
    binding.opening_a.commitment_index = binding.a_commitment_index;
    binding.opening_b.commitment_index = binding.b_commitment_index;
    binding.opening_c.commitment_index = binding.c_commitment_index;
    if (!same_wire(zkdl::fr_wire(binding.opening_a.value),
                   zkdl::fr_wire(primitive.opening_a.value)) ||
        !same_wire(zkdl::fr_wire(binding.opening_b.value),
                   zkdl::fr_wire(primitive.opening_b.value)) ||
        !same_wire(zkdl::fr_wire(binding.opening_c.value),
                   zkdl::fr_wire(primitive.opening_c.value))) {
        if (error) *error = "zkMatMul row commitment opening is not bound to canonical tensor opening";
        return false;
    }
    return true;
}

inline bool verify_primitive_binding(
    const PrimitiveDescriptor& descriptor, const std::vector<byte>& encoded,
    const OperationProof::PrimitiveBinding& binding,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators, std::string* error) {
    zkdl::MatMulProof primitive;
    if (!zkdl::MatMulProof::deserialize(encoded, primitive, error) ||
        primitive.shape != descriptor.matmul_shape ||
        primitive.c_binding_mode != zkdl::MatMulProof::kCCommitment ||
        !primitive.has_c_opening) {
        if (error && error->empty()) *error = "zkMatMul primitive binding mode is not canonical";
        return false;
    }
    const std::array<std::uint32_t, 3U> indices = {
        binding.a_commitment_index, binding.b_commitment_index, binding.c_commitment_index};
    const std::array<std::uint32_t, 3U> domains = {
        descriptor.matmul_shape.a_size(), descriptor.matmul_shape.b_size(), descriptor.matmul_shape.c_size()};
    const std::array<const zkdl::EvaluationOpeningProof*, 3U> openings = {
        &binding.opening_a, &binding.opening_b, &binding.opening_c};
    const std::array<const zkdl::EvaluationOpeningProof*, 3U> primitive_openings = {
        &primitive.opening_a, &primitive.opening_b, &primitive.opening_c};
    for (std::size_t i = 0U; i < indices.size(); ++i) {
        if (indices[i] >= commitments.size() ||
            commitments[indices[i]].tensor_id != descriptor.tensor_refs[i] ||
            commitments[indices[i]].domain_size != domains[i] ||
            openings[i]->commitment_index != indices[i] ||
            openings[i]->generator_size != domains[i] || !openings[i]->outer_point.empty() ||
            openings[i]->inner_point.size() != primitive_openings[i]->inner_point.size() +
                                               primitive_openings[i]->outer_point.size()) {
            if (error) *error = "zkMatMul canonical commitment binding metadata mismatch";
            return false;
        }
        const std::vector<FrWire> point = primitive_opening_point(*primitive_openings[i]);
        for (std::size_t j = 0U; j < point.size(); ++j)
            if (!same_wire(zkdl::fr_wire(openings[i]->inner_point[j]), point[j])) {
                if (error) *error = "zkMatMul canonical binding point differs from row opening";
                return false;
            }
        if (!zkdl::cpu_verify_opening(*openings[i],
                                      zkdl::cpu_g1_from_native(commitments[indices[i]].commitment),
                                      error) ||
            !same_wire(zkdl::fr_wire(openings[i]->value),
                       zkdl::fr_wire(primitive_openings[i]->value))) {
            if (error && error->empty()) *error = "zkMatMul canonical row reduction opening failed";
            return false;
        }
    }
    return true;
}

inline bool route_tensor_shapes_valid(
    const OperationDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const TensorWitness*& source, const TensorWitness*& output,
    std::string* error) {
    if ((descriptor.kind != RelationKind::SparseRoute &&
         descriptor.kind != RelationKind::Alpha) ||
        descriptor.selector_edge_domain == 0U || descriptor.selector_node_domain == 0U ||
        descriptor.selector_feature_domain == 0U ||
        descriptor.selector_source_feature_domain == 0U ||
        descriptor.selector_src.size() != descriptor.selector_edge_domain ||
        descriptor.selector_dst.size() != descriptor.selector_edge_domain ||
        descriptor.selector_active.size() != descriptor.selector_edge_domain ||
        descriptor.selector_feature_map.size() != descriptor.selector_feature_domain ||
        descriptor.selector_source_ref.empty() || descriptor.selector_output_ref.empty()) {
        if (error) *error = "sparse route descriptor is incomplete";
        return false;
    }
    const auto source_it = tensor_map.find(descriptor.selector_source_ref);
    const auto output_it = tensor_map.find(descriptor.selector_output_ref);
    if (source_it == tensor_map.end() || output_it == tensor_map.end() ||
        source_it->second == nullptr || output_it->second == nullptr) {
        if (error) *error = "sparse route canonical source/output tensor is missing";
        return false;
    }
    const std::uint64_t source_size = static_cast<std::uint64_t>(descriptor.selector_node_domain) *
                                      descriptor.selector_source_feature_domain;
    const std::uint64_t output_size = static_cast<std::uint64_t>(descriptor.selector_edge_domain) *
                                      descriptor.selector_feature_domain;
    if (source_size > std::numeric_limits<std::uint32_t>::max() ||
        output_size > std::numeric_limits<std::uint32_t>::max() ||
        source_it->second->domain_size != source_size ||
        output_it->second->domain_size != output_size) {
        if (error) *error = "sparse route source/output domains do not match public selector shape";
        return false;
    }
    for (std::uint32_t edge = 0U; edge < descriptor.selector_edge_domain; ++edge)
        if (descriptor.selector_active[edge] > 1U ||
            (!descriptor.selector_active[edge] &&
             (descriptor.selector_src[edge] != 0U || descriptor.selector_dst[edge] != 0U)) ||
            (descriptor.selector_active[edge] &&
             (descriptor.selector_src[edge] >= descriptor.selector_node_domain ||
              descriptor.selector_dst[edge] >= descriptor.selector_node_domain))) {
        if (error) *error = "sparse route public src/dst index is out of bounds";
            return false;
        }
    for (const std::uint32_t source : descriptor.selector_feature_map)
        if (source >= descriptor.selector_source_feature_domain) {
            if (error) *error = "sparse route public feature map is out of bounds";
            return false;
        }
    source = source_it->second;
    output = output_it->second;
    return true;
}

inline std::vector<FrWire> route_partial_feature(
    const std::vector<FrWire>& values, std::uint32_t outer_domain,
    std::uint32_t feature_domain, const std::vector<FrWire>& feature_point) {
    if (values.size() != static_cast<std::uint64_t>(outer_domain) * feature_domain ||
        feature_domain != (std::uint32_t(1U) << feature_point.size()))
        throw std::invalid_argument("sparse route feature MLE shape mismatch");
    std::vector<FrWire> partial(outer_domain);
    for (std::uint32_t outer = 0U; outer < outer_domain; ++outer) {
        std::vector<FrWire> block(
            values.begin() + static_cast<std::ptrdiff_t>(outer * feature_domain),
            values.begin() + static_cast<std::ptrdiff_t>((outer + 1U) * feature_domain));
        for (const auto& challenge : feature_point) fold_values(block, challenge);
        partial[outer] = block.front();
    }
    return partial;
}

inline std::vector<FrWire> public_route_selector_mle(
    const OperationDescriptor& descriptor, const std::vector<FrWire>& edge_point,
    bool destination = false) {
    const std::uint32_t edge_domain = descriptor.selector_edge_domain;
    const std::uint32_t node_domain = descriptor.selector_node_domain;
    if (edge_domain != (std::uint32_t(1U) << edge_point.size()) ||
        descriptor.selector_src.size() != edge_domain) {
        throw std::invalid_argument("public sparse selector MLE shape mismatch");
    }
    std::vector<FrWire> selector(node_domain);
    const FrWire one = zkdl::fr_from_u32(1U);
    const FrWire zero{};
    // This is the equality-MLE construction itself, not a dense one-hot
    // matrix multiplication: each public node coordinate is evaluated by
    // folding its edge indicator against the verifier's FS point.
    for (std::uint32_t node = 0U; node < node_domain; ++node) {
        std::vector<FrWire> indicator(edge_domain, zero);
        for (std::uint32_t edge = 0U; edge < edge_domain; ++edge)
            if (descriptor.selector_active[edge] &&
                (destination ? descriptor.selector_dst[edge] : descriptor.selector_src[edge]) == node)
                indicator[edge] = one;
        for (const auto& challenge : edge_point) fold_values(indicator, challenge);
        selector[node] = indicator.front();
    }
    return selector;
}

inline std::vector<FrWire> public_route_feature_selector_mle(
    const OperationDescriptor& descriptor,
    const std::vector<FrWire>& feature_point) {
    const std::uint32_t output_domain = descriptor.selector_feature_domain;
    const std::uint32_t source_domain = descriptor.selector_source_feature_domain;
    if (output_domain != (std::uint32_t(1U) << feature_point.size()) ||
        descriptor.selector_feature_map.size() != output_domain) {
        throw std::invalid_argument("public sparse feature selector MLE shape mismatch");
    }
    const FrWire one = zkdl::fr_from_u32(1U);
    const FrWire zero{};
    std::vector<FrWire> selector(source_domain, zero);
    // Build the public equality-MLE selector by folding output-feature
    // indicators.  This remains sparse/public and never materializes a dense
    // one-hot matrix or accepts a prover-selected feature map.
    for (std::uint32_t source = 0U; source < source_domain; ++source) {
        std::vector<FrWire> indicator(output_domain, zero);
        for (std::uint32_t output = 0U; output < output_domain; ++output)
            if (descriptor.selector_feature_map[output] == source) indicator[output] = one;
        for (const auto& challenge : feature_point) fold_values(indicator, challenge);
        selector[source] = indicator.front();
    }
    return selector;
}

inline bool route_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(commitments.begin(), commitments.end(),
                                    [&](const CommitmentWire& commitment) {
                                        return commitment.tensor_id == tensor_id;
                                    });
    if (found == commitments.end()) {
        if (error) *error = "sparse route commitment is missing: " + tensor_id;
        return false;
    }
    index = static_cast<std::uint32_t>(std::distance(commitments.begin(), found));
    return true;
}

inline bool prove_sparse_route(
    const OperationDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    const std::vector<CommitmentWire>& commitments,
    zkdl::Transcript& transcript, SparseRouteProof& route,
    std::string* error) {
    const TensorWitness* source = nullptr;
    const TensorWitness* output = nullptr;
    if (!route_tensor_shapes_valid(descriptor, tensor_map, source, output, error)) return false;
    const std::uint32_t edge_bits = log2_exact(descriptor.selector_edge_domain);
    const std::uint32_t node_bits = log2_exact(descriptor.selector_node_domain);
    const std::uint32_t feature_bits = log2_exact(descriptor.selector_feature_domain);
    const std::uint32_t source_feature_bits =
        log2_exact(descriptor.selector_source_feature_domain);
    if (edge_bits == std::numeric_limits<std::uint32_t>::max() ||
        node_bits == std::numeric_limits<std::uint32_t>::max() ||
        feature_bits == std::numeric_limits<std::uint32_t>::max() ||
        source_feature_bits == std::numeric_limits<std::uint32_t>::max() ||
        generators.size() < std::max(source->domain_size, output->domain_size)) {
        if (error) *error = "sparse route has no authenticated generator domain";
        return false;
    }
    route = SparseRouteProof{};
    route.edge_domain = descriptor.selector_edge_domain;
    route.node_domain = descriptor.selector_node_domain;
    route.feature_domain = descriptor.selector_feature_domain;
    route.source_feature_domain = descriptor.selector_source_feature_domain;
    route.edge_point.reserve(edge_bits);
    route.feature_point.reserve(feature_bits);
    for (std::uint32_t bit = 0U; bit < edge_bits; ++bit)
        route.edge_point.push_back(transcript.challenge(
            "gat.route." + descriptor.id + ".edge." + std::to_string(bit)));
    for (std::uint32_t bit = 0U; bit < feature_bits; ++bit)
        route.feature_point.push_back(transcript.challenge(
            "gat.route." + descriptor.id + ".feature." + std::to_string(bit)));
    const std::vector<FrWire> edge_point = [&]() {
        std::vector<FrWire> out;
        out.reserve(route.edge_point.size());
        for (const auto& value : route.edge_point) out.push_back(zkdl::fr_wire(value));
        return out;
    }();
    const std::vector<FrWire> feature_point = [&]() {
        std::vector<FrWire> out;
        out.reserve(route.feature_point.size());
        for (const auto& value : route.feature_point) out.push_back(zkdl::fr_wire(value));
        return out;
    }();
    const std::vector<FrWire> output_by_edge = route_partial_feature(
        output->values, descriptor.selector_edge_domain,
        descriptor.selector_feature_domain, feature_point);
    std::vector<FrWire> output_eval_values = output_by_edge;
    for (const auto& challenge : edge_point) fold_values(output_eval_values, challenge);
    route.output_claim = zkdl::fr_value(output_eval_values.front());
    const std::vector<FrWire> selector_by_node = public_route_selector_mle(
        descriptor, edge_point, descriptor.route_by_destination);
    const std::vector<FrWire> selector_by_feature = public_route_feature_selector_mle(
        descriptor, feature_point);
    std::vector<FrWire> selector_full(
        static_cast<std::size_t>(descriptor.selector_node_domain) *
            descriptor.selector_source_feature_domain);
    FrWire source_sum{};
    for (std::uint32_t node = 0U; node < descriptor.selector_node_domain; ++node) {
        for (std::uint32_t feature = 0U;
             feature < descriptor.selector_source_feature_domain; ++feature) {
            const std::size_t index = static_cast<std::size_t>(node) *
                                      descriptor.selector_source_feature_domain + feature;
            selector_full[index] = zkdl::fr_mul(selector_by_node[node], selector_by_feature[feature]);
            source_sum = zkdl::fr_add(source_sum,
                                     zkdl::fr_mul(selector_full[index], source->values[index]));
        }
    }
    route.source_sum_claim = zkdl::fr_value(source_sum);
    std::vector<FrWire> destination_by_node = public_route_selector_mle(
        descriptor, edge_point, !descriptor.route_by_destination);
    if (!same_wire(zkdl::fr_wire(route.output_claim), source_sum)) {
        if (error) *error = "sparse route witness does not satisfy public selector MLE claim";
        return false;
    }
    route.sumcheck.label = "gat.route." + descriptor.id;
    route.sumcheck.degree = 2U;
    std::vector<FrWire> current_selector = std::move(selector_full);
    std::vector<FrWire> current_source = source->values;
    const std::uint32_t sumcheck_rounds = source_feature_bits + node_bits;
    for (std::uint32_t round = 0U; round < sumcheck_rounds; ++round) {
        zkdl::SumcheckRound message;
        const std::size_t next_size = current_selector.size() / 2U;
        for (std::uint32_t evaluation = 0U; evaluation <= 2U; ++evaluation) {
            const FrWire x = zkdl::fr_from_u32(evaluation);
            FrWire total{};
            for (std::size_t row = 0U; row < next_size; ++row) {
                const FrWire selector_value = zkdl::fr_add(
                    current_selector[2U * row], zkdl::fr_mul(
                        x, zkdl::fr_sub(current_selector[2U * row + 1U],
                                       current_selector[2U * row])));
                const FrWire source_value = zkdl::fr_add(
                    current_source[2U * row], zkdl::fr_mul(
                        x, zkdl::fr_sub(current_source[2U * row + 1U],
                                       current_source[2U * row])));
                total = zkdl::fr_add(total, zkdl::fr_mul(selector_value, source_value));
            }
            message.evaluations.push_back(zkdl::fr_value(total));
        }
        route.sumcheck.rounds.push_back(message);
        transcript.absorb_fr(route.sumcheck.label + ".round", message.evaluations);
        const Fr_t challenge = transcript.challenge(route.sumcheck.label + ".challenge");
        if (round < source_feature_bits) route.source_feature_point.push_back(challenge);
        else route.node_point.push_back(challenge);
        fold_values(current_selector, zkdl::fr_wire(challenge));
        fold_values(current_source, zkdl::fr_wire(challenge));
    }
    route.sumcheck.terminal_claims = {
        zkdl::fr_value(current_selector.front()), zkdl::fr_value(current_source.front())};
    route.sumcheck.final_claim = zkdl::fr_value(zkdl::fr_mul(
        current_selector.front(), current_source.front()));
    for (const auto& challenge : route.node_point)
        fold_values(destination_by_node, zkdl::fr_wire(challenge));
    route.dst_selector_claim = zkdl::fr_value(destination_by_node.front());
    if (route.node_point.size() != node_bits ||
        route.source_feature_point.size() != source_feature_bits) {
        if (error) *error = "sparse route prover lost FS challenges";
        return false;
    }
    if (!route_commitment_index(commitments, descriptor.selector_source_ref,
                                route.source_commitment_index, error) ||
        !route_commitment_index(commitments, descriptor.selector_output_ref,
                                route.output_commitment_index, error)) return false;
    std::vector<FrWire> source_point;
    source_point.reserve(source_feature_bits + node_bits);
    for (const auto& challenge : route.source_feature_point)
        source_point.push_back(zkdl::fr_wire(challenge));
    for (const auto& challenge : route.node_point) source_point.push_back(zkdl::fr_wire(challenge));
    std::vector<FrWire> output_point = feature_point;
    for (const auto& challenge : route.edge_point) output_point.push_back(zkdl::fr_wire(challenge));
    route.source_opening = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + source->domain_size),
        source->values, source_point);
    route.source_opening.commitment_index = route.source_commitment_index;
    route.output_opening = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + output->domain_size),
        output->values, output_point);
    route.output_opening.commitment_index = route.output_commitment_index;
    std::vector<byte> source_opening_bytes, output_opening_bytes;
    zkdl::serialize_opening(route.source_opening, source_opening_bytes);
    zkdl::serialize_opening(route.output_opening, output_opening_bytes);
    transcript.absorb("gat.route." + descriptor.id + ".opening.source", source_opening_bytes);
    transcript.absorb("gat.route." + descriptor.id + ".opening.output", output_opening_bytes);
    route.rho = transcript.challenge("gat.fac4dnn.line8." + descriptor.id + ".rho");
    route.compressed_claim = zkdl::fr_value(zkdl::fr_add(
        zkdl::fr_wire(route.output_claim),
        zkdl::fr_add(
            zkdl::fr_mul(zkdl::fr_wire(route.rho), zkdl::fr_wire(route.source_sum_claim)),
            zkdl::fr_mul(zkdl::fr_mul(zkdl::fr_wire(route.rho), zkdl::fr_wire(route.rho)),
                          zkdl::fr_wire(route.dst_selector_claim)))));
    transcript.absorb_fr("gat.fac4dnn.line8." + descriptor.id + ".eq30",
                         {route.compressed_claim});
    return true;
}

inline bool verify_sparse_route(
    const OperationDescriptor& descriptor,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    zkdl::Transcript& transcript, const SparseRouteProof& route,
    std::string* error) {
    if ((descriptor.kind != RelationKind::SparseRoute &&
         descriptor.kind != RelationKind::Alpha) ||
        route.edge_domain != descriptor.selector_edge_domain ||
        route.node_domain != descriptor.selector_node_domain ||
        route.feature_domain != descriptor.selector_feature_domain ||
        route.source_feature_domain != descriptor.selector_source_feature_domain) {
        if (error) *error = "sparse route proof/descriptor domain mismatch";
        return false;
    }
    const std::uint32_t edge_bits = log2_exact(route.edge_domain);
    const std::uint32_t node_bits = log2_exact(route.node_domain);
    const std::uint32_t feature_bits = log2_exact(route.feature_domain);
    const std::uint32_t source_feature_bits = log2_exact(route.source_feature_domain);
    if (edge_bits == std::numeric_limits<std::uint32_t>::max() ||
        node_bits == std::numeric_limits<std::uint32_t>::max() ||
        feature_bits == std::numeric_limits<std::uint32_t>::max() ||
        source_feature_bits == std::numeric_limits<std::uint32_t>::max() ||
        route.edge_point.size() != edge_bits || route.node_point.size() != node_bits ||
        route.feature_point.size() != feature_bits ||
        route.sumcheck.label != "gat.route." + descriptor.id ||
        route.source_feature_point.size() != source_feature_bits ||
        route.sumcheck.degree != 2U ||
        route.sumcheck.rounds.size() != source_feature_bits + node_bits ||
        route.sumcheck.terminal_claims.size() != 2U ||
        route.source_commitment_index >= commitments.size() ||
        route.output_commitment_index >= commitments.size() ||
        commitments[route.source_commitment_index].tensor_id != descriptor.selector_source_ref ||
        commitments[route.output_commitment_index].tensor_id != descriptor.selector_output_ref ||
        commitments[route.source_commitment_index].domain_size !=
            static_cast<std::uint64_t>(route.node_domain) * route.source_feature_domain ||
        commitments[route.output_commitment_index].domain_size !=
            static_cast<std::uint64_t>(route.edge_domain) * route.feature_domain ||
        generators.size() < std::max(commitments[route.source_commitment_index].domain_size,
                                     commitments[route.output_commitment_index].domain_size)) {
        if (error) *error = "sparse route proof shape or commitment reference is invalid";
        return false;
    }
    for (std::uint32_t bit = 0U; bit < edge_bits; ++bit) {
        const Fr_t expected = transcript.challenge(
            "gat.route." + descriptor.id + ".edge." + std::to_string(bit));
        if (fr_compare_mixed(expected, route.edge_point[bit]) != 0) {
            if (error) *error = "sparse route edge challenge is not transcript-bound";
            return false;
        }
    }
    for (std::uint32_t bit = 0U; bit < feature_bits; ++bit) {
        const Fr_t expected = transcript.challenge(
            "gat.route." + descriptor.id + ".feature." + std::to_string(bit));
        if (fr_compare_mixed(expected, route.feature_point[bit]) != 0) {
            if (error) *error = "sparse route feature challenge is not transcript-bound";
            return false;
        }
    }
    std::vector<Fr_t> all_challenges;
    if (!zkdl::verify_sumcheck(route.sumcheck, route.output_claim, transcript,
                               &all_challenges, error) ||
        all_challenges.size() != route.source_feature_point.size() + route.node_point.size()) return false;
    for (std::size_t i = 0U; i < route.source_feature_point.size(); ++i)
        if (fr_compare_mixed(all_challenges[i], route.source_feature_point[i]) != 0) {
            if (error) *error = "sparse route source-feature challenge is not transcript-bound";
            return false;
        }
    for (std::size_t i = 0U; i < route.node_point.size(); ++i)
        if (fr_compare_mixed(all_challenges[route.source_feature_point.size() + i],
                             route.node_point[i]) != 0) {
            if (error) *error = "sparse route node challenge is not transcript-bound";
            return false;
        }
    std::vector<FrWire> expected_source_point;
    expected_source_point.reserve(source_feature_bits + node_bits);
    for (const auto& value : route.source_feature_point)
        expected_source_point.push_back(zkdl::fr_wire(value));
    for (const auto& value : route.node_point) expected_source_point.push_back(zkdl::fr_wire(value));
    std::vector<FrWire> expected_output_point;
    expected_output_point.reserve(feature_bits + edge_bits);
    for (const auto& value : route.feature_point) expected_output_point.push_back(zkdl::fr_wire(value));
    for (const auto& value : route.edge_point) expected_output_point.push_back(zkdl::fr_wire(value));
    const auto& source_commitment = commitments[route.source_commitment_index];
    const auto& output_commitment = commitments[route.output_commitment_index];
    if (route.source_opening.commitment_index != route.source_commitment_index ||
        route.output_opening.commitment_index != route.output_commitment_index ||
        route.source_opening.generator_size != source_commitment.domain_size ||
        route.output_opening.generator_size != output_commitment.domain_size ||
        route.source_opening.outer_point.size() != 0U ||
        route.output_opening.outer_point.size() != 0U ||
        route.source_opening.inner_point.size() != expected_source_point.size() ||
        route.output_opening.inner_point.size() != expected_output_point.size()) {
        if (error) *error = "sparse route opening metadata is not canonical";
        return false;
    }
    for (size_t i = 0U; i < expected_source_point.size(); ++i)
        if (fr_compare_mixed(route.source_opening.inner_point[i],
                             expected_source_point[i]) != 0) {
            if (error) *error = "sparse route source opening point mismatch";
            return false;
        }
    for (size_t i = 0U; i < expected_output_point.size(); ++i)
        if (fr_compare_mixed(route.output_opening.inner_point[i],
                             expected_output_point[i]) != 0) {
            if (error) *error = "sparse route output opening point mismatch";
            return false;
        }
    if (!zkdl::cpu_verify_opening(route.source_opening,
                                  zkdl::cpu_g1_from_native(source_commitment.commitment), error) ||
        !zkdl::cpu_verify_opening(route.output_opening,
                                  zkdl::cpu_g1_from_native(output_commitment.commitment), error)) return false;
    const std::vector<FrWire> edge_point = [&]() {
        std::vector<FrWire> point;
        point.reserve(route.edge_point.size());
        for (const auto& value : route.edge_point) point.push_back(zkdl::fr_wire(value));
        return point;
    }();
    std::vector<FrWire> selector_terminal_outer = public_route_selector_mle(
        descriptor, edge_point, descriptor.route_by_destination);
    const std::vector<FrWire> selector_terminal_feature = public_route_feature_selector_mle(
        descriptor, [&]() {
            std::vector<FrWire> point;
            point.reserve(route.feature_point.size());
            for (const auto& value : route.feature_point) point.push_back(zkdl::fr_wire(value));
            return point;
        }());
    std::vector<FrWire> selector_terminal(
        static_cast<std::size_t>(route.node_domain) * route.source_feature_domain);
    for (std::uint32_t node = 0U; node < route.node_domain; ++node)
        for (std::uint32_t feature = 0U; feature < route.source_feature_domain; ++feature)
            selector_terminal[static_cast<std::size_t>(node) * route.source_feature_domain + feature] =
                zkdl::fr_mul(selector_terminal_outer[node], selector_terminal_feature[feature]);
    for (const auto& challenge : route.source_feature_point)
        fold_values(selector_terminal, zkdl::fr_wire(challenge));
    for (const auto& challenge : route.node_point)
        fold_values(selector_terminal, zkdl::fr_wire(challenge));
    std::vector<FrWire> destination_terminal = public_route_selector_mle(
        descriptor, edge_point, !descriptor.route_by_destination);
    for (const auto& challenge : route.node_point)
        fold_values(destination_terminal, zkdl::fr_wire(challenge));
    if (fr_compare_mixed(route.sumcheck.terminal_claims[0],
                         selector_terminal.front()) != 0 ||
        fr_compare_mixed(route.sumcheck.terminal_claims[1],
                         route.source_opening.value) != 0 ||
        fr_compare_mixed(route.sumcheck.final_claim,
                         zkdl::fr_value(zkdl::fr_mul(
                             zkdl::fr_wire(route.sumcheck.terminal_claims[0]),
                             zkdl::fr_wire(route.sumcheck.terminal_claims[1])))) != 0 ||
        fr_compare_mixed(route.sumcheck.final_claim,
                         route.source_sum_claim) != 0 ||
        fr_compare_mixed(route.dst_selector_claim, destination_terminal.front()) != 0 ||
        fr_compare_mixed(route.output_opening.value, route.output_claim) != 0 ||
        fr_compare_mixed(route.source_sum_claim, route.output_claim) != 0) {
        if (error) *error = "sparse route terminal or Equation(30) claim failed";
        return false;
    }
    std::vector<byte> source_opening_bytes, output_opening_bytes;
    zkdl::serialize_opening(route.source_opening, source_opening_bytes);
    zkdl::serialize_opening(route.output_opening, output_opening_bytes);
    transcript.absorb("gat.route." + descriptor.id + ".opening.source", source_opening_bytes);
    transcript.absorb("gat.route." + descriptor.id + ".opening.output", output_opening_bytes);
    const Fr_t expected_rho = transcript.challenge(
        "gat.fac4dnn.line8." + descriptor.id + ".rho");
    if (fr_compare_mixed(expected_rho, route.rho) != 0) {
        if (error) *error = "FAC4DNN Line8 rho is not transcript-bound";
        return false;
    }
    const Fr_t expected_compressed = zkdl::fr_value(zkdl::fr_add(
        zkdl::fr_wire(route.output_claim),
        zkdl::fr_add(
            zkdl::fr_mul(zkdl::fr_wire(route.rho), zkdl::fr_wire(route.source_sum_claim)),
            zkdl::fr_mul(zkdl::fr_mul(zkdl::fr_wire(route.rho), zkdl::fr_wire(route.rho)),
                          zkdl::fr_wire(route.dst_selector_claim)))));
    if (fr_compare_mixed(expected_compressed, route.compressed_claim) != 0) {
        if (error) *error = "FAC4DNN Line8 Equation(30) compression failed";
        return false;
    }
    transcript.absorb_fr("gat.fac4dnn.line8." + descriptor.id + ".eq30",
                         {route.compressed_claim});
    return true;
}

inline bool prove(const ProverContext& context, Proof& proof, std::string* error = nullptr) {
    try {
        if (!descriptor_vector_valid(context.operations, error)) return false;
        if (context.tensors.empty() || context.generators.empty()) {
            if (error) *error = "relation prover requires canonical tensors and authenticated generators";
            return false;
        }
        std::map<std::string, const TensorWitness*> tensor_map;
        for (const auto& tensor : context.tensors) {
            if (!tensor.valid(error) || !tensor_map.emplace(tensor.id, &tensor).second) {
                if (error && error->empty()) *error = "duplicate relation tensor id";
                return false;
            }
        }
        proof = Proof{};
        proof.statement = context.statement;
        const std::array<byte, 32> expected_public_digest =
            zkdl::sha256(proof.statement.public_output);
        if (proof.statement.public_output.empty() ||
            (proof.statement.public_output_digest != std::array<byte, 32>{} &&
             proof.statement.public_output_digest != expected_public_digest)) {
            throw std::invalid_argument("public Y digest is not canonical");
        }
        proof.statement.public_output_digest = expected_public_digest;
        proof.hiding_commitment_records = context.hiding_commitment_records;
        for (const auto& tensor : context.tensors) {
            const std::uint32_t domain = tensor.domain_size;
            if (context.generators.size() < domain) {
                if (error) *error = "authenticated SRS generator domain is too small";
                return false;
            }
            CommitmentWire commitment;
            commitment.tensor_id = tensor.id;
            commitment.visibility = tensor.visibility;
            commitment.domain_size = domain;
            std::vector<zkdl::CpuG1Point> generators(context.generators.begin(),
                                                     context.generators.begin() + domain);
            commitment.commitment = zkdl::cpu_g1_to_native(zkdl::cpu_g1_sum(generators, tensor.values));
            proof.commitments.push_back(commitment);
        }
        if (!verify_model_registry(proof.commitments,
                                   context.registered_model_commitments, error) ||
            !verify_hiding_records(proof.commitments,
                                   context.hiding_commitment_records,
                                   context.generators, error)) return false;
        zkdl::Transcript transcript("zkdl-gat-relation-v1");
        absorb_statement(transcript, proof.statement, proof.commitments);
        absorb_hiding_records(transcript, proof.hiding_commitment_records);
        for (const auto& operation : context.operations) {
            absorb_descriptor(transcript, operation);
            OperationProof operation_proof;
            operation_proof.descriptor = operation;
            if (operation.primitives.size() > 8U) {
                throw std::invalid_argument("relation operation carries too many native primitives");
            }
            const bool has_route = operation.kind == RelationKind::SparseRoute ||
                                   !operation.selector_source_ref.empty();
            if (has_route) {
                SparseRouteProof route;
                if (!prove_sparse_route(operation, tensor_map, context.generators,
                                        proof.commitments, transcript, route, error)) return false;
                operation_proof.route_proof = route.serialize();
                transcript.absorb("gat.relation.route", operation_proof.route_proof);
            } else if (!operation_proof.route_proof.empty()) {
                throw std::invalid_argument("non-route operation carries a sparse route proof");
            }
            if (!operation.fac_bindings.empty()) {
                FacLine8Proof fac;
                if (!prove_fac_line8(operation, tensor_map, context.generators,
                                     proof.commitments, transcript, fac, error)) return false;
                operation_proof.fac_proof = fac.serialize();
                transcript.absorb("gat.relation.fac", operation_proof.fac_proof);
            } else if (!operation_proof.fac_proof.empty()) {
                throw std::invalid_argument("operation has an unexpected FAC4DNN proof");
            }
            if (!operation.batch_family.empty()) {
                const ConstraintDescriptor batched = batched_family_descriptor(
                    operation, transcript, error);
                if (batched.id.empty()) return false;
                ConstraintProof payload;
                if (!fill_constraint_openings(batched, operation.tensor_refs, tensor_map,
                                              context.generators, proof.commitments,
                                              transcript, payload, error)) return false;
                operation_proof.batch_proof = serialize_constraint_payload(payload);
                transcript.absorb("gat.relation.batch", operation_proof.batch_proof);
            } else if (!operation_proof.batch_proof.empty()) {
                throw std::invalid_argument("operation has an unexpected batched relation proof");
            }
            for (const auto& primitive : operation.primitives) {
                std::vector<byte> primitive_bytes;
                if (!prove_matmul_primitive(primitive, tensor_map, context.generators,
                                             primitive_bytes, error)) return false;
                operation_proof.primitive_proofs.push_back(std::move(primitive_bytes));
                transcript.absorb("gat.relation.primitive", operation_proof.primitive_proofs.back());
                OperationProof::PrimitiveBinding binding;
                if (!prove_primitive_binding(primitive,
                                              operation_proof.primitive_proofs.back(), tensor_map,
                                              context.generators, proof.commitments, binding, error)) return false;
                operation_proof.primitive_bindings.push_back(std::move(binding));
                std::vector<byte> binding_bytes;
                append_primitive_binding(binding_bytes, operation_proof.primitive_bindings.back());
                transcript.absorb("gat.relation.primitive.binding", binding_bytes);
            }
            if (!operation.batch_family.empty()) {
                if (!operation_proof.constraints.empty())
                    throw std::invalid_argument("batched relation carries repeated constraint proofs");
            }
            for (const auto& descriptor : operation.constraints) {
                if (!operation.batch_family.empty()) break;
                std::vector<const TensorWitness*> witnesses;
                for (const auto& ref : operation.tensor_refs) {
                    const auto found = tensor_map.find(ref);
                    if (found == tensor_map.end()) throw std::invalid_argument("relation tensor reference is missing");
                    witnesses.push_back(found->second);
                }
                ConstraintProof constraint;
                constraint.descriptor = descriptor;
                std::vector<Fr_t> challenges;
                std::vector<FrWire> terminal_values;
                constraint.sumcheck = prove_constraint_exact(descriptor, witnesses, transcript,
                                                             challenges, terminal_values);
                constraint.opening_tensor_indices = constraint_tensor_indices(
                    descriptor, operation.tensor_refs.size());
                for (std::size_t opening_index = 0U;
                     opening_index < constraint.opening_tensor_indices.size(); ++opening_index) {
                    const auto tensor_index = constraint.opening_tensor_indices[opening_index];
                    const auto& ref = operation.tensor_refs.at(tensor_index);
                    const auto found = std::find_if(proof.commitments.begin(), proof.commitments.end(),
                                                    [&](const CommitmentWire& value) { return value.tensor_id == ref; });
                    if (found == proof.commitments.end()) throw std::invalid_argument("relation commitment reference is missing");
                    const auto tensor = tensor_map.at(ref);
                    std::vector<FrWire> point;
                    point.reserve(challenges.size());
                    for (const auto& challenge : challenges) point.push_back(zkdl::fr_wire(challenge));
                    std::vector<zkdl::CpuG1Point> gens(context.generators.begin(), context.generators.begin() + tensor->domain_size);
                    constraint.opening_commitment_indices.push_back(static_cast<std::uint32_t>(
                        std::distance(proof.commitments.begin(), std::find_if(
                            proof.commitments.begin(), proof.commitments.end(),
                            [&](const CommitmentWire& value) { return value.tensor_id == ref; }))));
                    constraint.openings.push_back(zkdl::cpu_open(gens, tensor->values, point));
                }
                std::vector<Fr_t> reduced_terminal_values;
                reduced_terminal_values.reserve(constraint.opening_tensor_indices.size());
                for (const auto tensor_index : constraint.opening_tensor_indices)
                    reduced_terminal_values.push_back(
                        zkdl::fr_value(terminal_values.at(tensor_index)));
                constraint.sumcheck.terminal_claims = reduced_terminal_values;
                const std::uint32_t point_bits = log2_exact(
                    operation.tensor_refs.empty() ? 0U :
                    tensor_map.at(operation.tensor_refs.front())->domain_size);
                if (point_bits == std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("relation pointwise domain is not a power of two");
                for (std::uint32_t bit = 0U; bit < point_bits; ++bit)
                    constraint.pointwise_point.push_back(transcript.challenge(
                        "gat.relation." + descriptor.id + ".pointwise." + std::to_string(bit)));
                for (std::size_t opening_index = 0U;
                     opening_index < constraint.opening_tensor_indices.size(); ++opening_index) {
                    const auto tensor_index = constraint.opening_tensor_indices[opening_index];
                    const auto& ref = operation.tensor_refs.at(tensor_index);
                    const auto tensor = tensor_map.at(ref);
                    std::vector<FrWire> point;
                    point.reserve(constraint.pointwise_point.size());
                    for (const auto& challenge : constraint.pointwise_point)
                        point.push_back(zkdl::fr_wire(challenge));
                    std::vector<zkdl::CpuG1Point> gens(
                        context.generators.begin(), context.generators.begin() + tensor->domain_size);
                    auto opening = zkdl::cpu_open(gens, tensor->values, point);
                    opening.commitment_index = constraint.opening_commitment_indices.at(opening_index);
                    constraint.pointwise_openings.push_back(std::move(opening));
                }
                std::vector<byte> pointwise_bytes;
                for (const auto& opening : constraint.pointwise_openings)
                    zkdl::serialize_opening(opening, pointwise_bytes);
                transcript.absorb("gat.relation." + descriptor.id + ".pointwise.openings",
                                  pointwise_bytes);
                operation_proof.constraints.push_back(std::move(constraint));
            }
            proof.operations.push_back(std::move(operation_proof));
        }
        const auto public_it = std::find_if(context.tensors.begin(), context.tensors.end(),
                                            [](const TensorWitness& tensor) {
                                                return tensor.id == "public_Y" && tensor.visibility == 1U;
                                            });
        if (public_it == context.tensors.end()) throw std::invalid_argument("public output tensor is missing");
        if (context.statement.public_output.empty() ||
            context.statement.public_output != encode_public_field_vector(public_it->values)) {
            throw std::invalid_argument("public output witness is not the verifier-supplied canonical Y");
        }
        proof.public_output_commitment = static_cast<std::uint32_t>(
            std::distance(context.tensors.begin(), public_it));
        std::vector<FrWire> output_values = public_it->values;
        const std::uint32_t output_domain = public_it->domain_size;
        const std::uint32_t output_bits = log2_exact(output_domain);
        std::vector<FrWire> output_point;
        for (std::uint32_t i = 0U; i < output_bits; ++i)
            output_point.push_back(zkdl::fr_wire(transcript.challenge("gat.public_output.point." + std::to_string(i))));
        std::vector<zkdl::CpuG1Point> output_generators(context.generators.begin(),
                                                        context.generators.begin() + output_domain);
        proof.public_output_opening = zkdl::cpu_open(output_generators, output_values, output_point);
        proof.public_output_opening.commitment_index = proof.public_output_commitment;
        std::vector<byte> public_bytes;
        zkdl::serialize_opening(proof.public_output_opening, public_bytes);
        transcript.absorb("gat.public_output.opening", public_bytes);
        proof.transcript_digest = transcript.digest();
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool verify(const Proof& proof, const VerifierContext& context,
                   std::string* error = nullptr) {
    if (proof.version != Proof::kVersion || !same_bytes(proof.statement.statement_digest,
                                                         context.statement.statement_digest) ||
        !same_bytes(proof.statement.model_digest, context.statement.model_digest) ||
        !same_bytes(proof.statement.table_digest, context.statement.table_digest) ||
        !same_bytes(proof.statement.ledger_digest, context.statement.ledger_digest) ||
        !same_bytes(proof.statement.srs_digest, context.statement.srs_digest) ||
        !same_bytes(proof.statement.registry_digest, context.statement.registry_digest) ||
        proof.statement.public_output_digest != zkdl::sha256(proof.statement.public_output) ||
        (context.statement.public_output_digest != std::array<byte, 32>{} &&
         proof.statement.public_output_digest != context.statement.public_output_digest) ||
        proof.statement.public_output != context.statement.public_output) {
        if (error) *error = "relation proof statement/resource identity mismatch";
        return false;
    }
    if (!descriptor_vector_valid(context.operations, error) ||
        proof.operations.size() != context.operations.size() || proof.commitments.empty()) return false;
    if (!verify_tensor_contract(context, proof.commitments, error) ||
        !verify_model_registry(proof.commitments,
                               context.registered_model_commitments, error) ||
        !verify_hiding_records(proof.commitments,
                               context.hiding_commitment_records,
                               context.generators, error) ||
        !verify_public_tensor_values(context, proof.commitments, error)) return false;
    if (!hiding_records_equal(proof.hiding_commitment_records,
                              context.hiding_commitment_records)) {
        if (error) *error = "relation proof hiding records differ from independent verifier context";
        return false;
    }
    std::set<std::string> commitment_ids;
    for (const auto& commitment : proof.commitments) {
        if (commitment.tensor_id.empty() || !commitment_ids.insert(commitment.tensor_id).second ||
            commitment.domain_size == 0U || log2_exact(commitment.domain_size) == std::numeric_limits<std::uint32_t>::max() ||
            commitment.domain_size > context.generators.size()) {
            if (error) *error = "relation proof commitment section is invalid";
            return false;
        }
        const auto visibility = context.tensor_visibility.find(commitment.tensor_id);
        if (visibility != context.tensor_visibility.end() &&
            commitment.visibility != visibility->second) {
            if (error) *error = "relation proof commitment visibility does not match public context";
            return false;
        }
    }
    std::set<std::string> expected_commitment_ids;
    if (!context.tensor_ids.empty()) {
        expected_commitment_ids.insert(context.tensor_ids.begin(), context.tensor_ids.end());
    } else {
        for (const auto& operation : context.operations)
            expected_commitment_ids.insert(operation.tensor_refs.begin(), operation.tensor_refs.end());
        expected_commitment_ids.insert("public_Y");
    }
    if (commitment_ids != expected_commitment_ids) {
        if (error) *error = "relation proof commitment section contains an unplanned or missing tensor";
        return false;
    }
    zkdl::Transcript transcript("zkdl-gat-relation-v1");
    absorb_statement(transcript, proof.statement, proof.commitments);
    absorb_hiding_records(transcript, proof.hiding_commitment_records);
    for (size_t op_index = 0U; op_index < context.operations.size(); ++op_index) {
        const auto& expected_operation = context.operations[op_index];
        const auto& operation = proof.operations[op_index];
        std::vector<byte> expected_bytes, actual_bytes;
        append_operation(expected_bytes, expected_operation);
        append_operation(actual_bytes, operation.descriptor);
        if (expected_bytes != actual_bytes ||
            (!expected_operation.batch_family.empty()
                 ? !operation.constraints.empty()
                 : operation.constraints.size() != expected_operation.constraints.size())) {
            if (error) *error = "relation operation descriptor/count mismatch";
            return false;
        }
        absorb_descriptor(transcript, expected_operation);
        if (operation.primitive_proofs.size() != expected_operation.primitives.size()) {
            if (error) *error = "relation primitive proof count mismatch";
            return false;
        }
        if (operation.primitive_bindings.size() != expected_operation.primitives.size()) {
            if (error) *error = "relation primitive canonical binding count mismatch";
            return false;
        }
        const bool has_route = expected_operation.kind == RelationKind::SparseRoute ||
                               !expected_operation.selector_source_ref.empty();
        if (has_route) {
            if (operation.route_proof.empty()) {
                if (error) *error = "sparse route operation is missing its native relation proof";
                return false;
            }
            SparseRouteProof route;
            if (!SparseRouteProof::deserialize(operation.route_proof, route, error) ||
                !verify_sparse_route(expected_operation, proof.commitments, context.generators,
                                     transcript, route, error)) return false;
            transcript.absorb("gat.relation.route", operation.route_proof);
        } else if (!operation.route_proof.empty()) {
            if (error) *error = "non-route operation carries a sparse route proof";
            return false;
        }
        if (!expected_operation.fac_bindings.empty()) {
            if (operation.fac_proof.empty()) {
                if (error) *error = "FAC4DNN operation is missing its native view-binding proof";
                return false;
            }
            FacLine8Proof fac;
            if (!FacLine8Proof::deserialize(operation.fac_proof, fac, error) ||
                !verify_fac_line8(expected_operation, proof.commitments,
                                  context.generators, transcript, fac, error)) return false;
            transcript.absorb("gat.relation.fac", operation.fac_proof);
        } else if (!operation.fac_proof.empty()) {
            if (error) *error = "operation carries an unexpected FAC4DNN proof";
            return false;
        }
        if (!expected_operation.batch_family.empty()) {
            if (operation.batch_proof.empty()) {
                if (error) *error = "batched relation operation is missing its family proof";
                return false;
            }
            ConstraintProof payload;
            if (!deserialize_constraint_payload(operation.batch_proof, payload, error)) return false;
            const ConstraintDescriptor batched = batched_family_descriptor(
                expected_operation, transcript, error);
            if (batched.id.empty() ||
                !verify_constraint(payload, batched, expected_operation.tensor_refs,
                                   proof.commitments, context.generators, transcript, error)) return false;
            transcript.absorb("gat.relation.batch", operation.batch_proof);
        } else if (!operation.batch_proof.empty()) {
            if (error) *error = "operation carries an unexpected batched family proof";
            return false;
        }
        for (size_t primitive_index = 0U;
             primitive_index < expected_operation.primitives.size(); ++primitive_index) {
            if (!verify_matmul_primitive(expected_operation.primitives[primitive_index],
                                         operation.primitive_proofs[primitive_index],
                                         context.generators, error)) return false;
            transcript.absorb("gat.relation.primitive",
                              operation.primitive_proofs[primitive_index]);
            if (!verify_primitive_binding(expected_operation.primitives[primitive_index],
                                          operation.primitive_proofs[primitive_index],
                                          operation.primitive_bindings[primitive_index],
                                          proof.commitments, context.generators, error)) return false;
            std::vector<byte> binding_bytes;
            append_primitive_binding(binding_bytes, operation.primitive_bindings[primitive_index]);
            transcript.absorb("gat.relation.primitive.binding", binding_bytes);
        }
        if (expected_operation.batch_family.empty()) {
            for (size_t constraint_index = 0U;
                 constraint_index < expected_operation.constraints.size(); ++constraint_index)
                if (!verify_constraint(operation.constraints[constraint_index],
                                       expected_operation.constraints[constraint_index],
                                       expected_operation.tensor_refs,
                                       proof.commitments, context.generators,
                                       transcript, error)) return false;
        }
    }
    if (proof.public_output_commitment >= proof.commitments.size() ||
        proof.commitments[proof.public_output_commitment].tensor_id != "public_Y" ||
        proof.commitments[proof.public_output_commitment].visibility != 1U ||
        proof.public_output_opening.commitment_index != proof.public_output_commitment ||
        proof.public_output_opening.generator_size != proof.commitments[proof.public_output_commitment].domain_size ||
        proof.public_output_opening.generator_size > context.generators.size() ||
        !proof.public_output_opening.outer_point.empty()) {
        if (error) *error = "public output opening reference is invalid";
        return false;
    }
    const std::uint32_t domain = proof.public_output_opening.generator_size;
    const std::uint32_t bits = log2_exact(domain);
    std::vector<FrWire> point;
    for (std::uint32_t i = 0U; i < bits; ++i)
        point.push_back(zkdl::fr_wire(transcript.challenge("gat.public_output.point." + std::to_string(i))));
    if (proof.public_output_opening.inner_point.size() != point.size()) {
        if (error) *error = "public output opening point shape mismatch";
        return false;
    }
    for (size_t i = 0U; i < point.size(); ++i)
        if (!same_wire(point[i], zkdl::fr_wire(proof.public_output_opening.inner_point[i]))) {
            if (error) *error = "public output opening point is not transcript-bound";
            return false;
        }
    std::vector<FrWire> public_values;
    if (!decode_public_field_vector(context.statement.public_output, public_values) ||
        public_values.size() != domain) {
        if (error) *error = "public Y is not a canonical field vector matching its commitment domain";
        return false;
    }
    std::vector<zkdl::CpuG1Point> output_generators(context.generators.begin(), context.generators.begin() + domain);
    const G1Jacobian_t expected_public_commitment = zkdl::cpu_g1_to_native(
        zkdl::cpu_g1_sum(output_generators, public_values));
    if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(
                                proof.commitments[proof.public_output_commitment].commitment),
                            zkdl::cpu_g1_from_native(expected_public_commitment))) {
        if (error) *error = "public output commitment is not bound to verifier-supplied Y";
        return false;
    }
    if (!zkdl::cpu_verify_opening(proof.public_output_opening,
                                  zkdl::cpu_g1_from_native(proof.commitments[proof.public_output_commitment].commitment),
                                  error)) return false;
    std::vector<byte> opening_bytes;
    zkdl::serialize_opening(proof.public_output_opening, opening_bytes);
    transcript.absorb("gat.public_output.opening", opening_bytes);
    if (transcript.digest() != proof.transcript_digest) {
        if (error) *error = "GAT relation transcript digest mismatch";
        return false;
    }
    if (!same_wire(zkdl::fr_wire(proof.public_output_opening.value),
                   mle_value(public_values, point))) {
        if (error) *error = "public output opening value is not the verifier-supplied Y MLE";
        return false;
    }
    return true;
}

}  // namespace relation
}  // namespace zkdl_gat

#endif
