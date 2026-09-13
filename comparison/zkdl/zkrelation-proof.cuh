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
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
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
            (constraints.empty() && primitives.empty() &&
             kind != RelationKind::SparseRoute && kind != RelationKind::PublicOutput)) {
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
    // The terminal relation wire carries one hidden evaluation opening for
    // every canonical operation tensor.  Keep the explicit index maps on the
    // outer wire so the independent verifier can reject a reordered or
    // shortened terminal claim vector before it enters the transcript.
    std::vector<std::uint32_t> opening_tensor_indices;
    std::vector<std::uint32_t> opening_commitment_indices;
    // Formal private constraints use the native zero-knowledge Sumcheck and
    // the Hyrax v3/Pedersen terminal relation together.  There is deliberately
    // no ordinary Sumcheck or value-revealing IPA fallback in this wire.
    zkdl::ZkSumcheckProof zk_sumcheck;
    zkdl::ZkTerminalRelationProof terminal_relation;
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
    static const std::uint32_t kVersion = 3U;
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
    // The route is a zero-initial-claim Sumcheck.  Its terminal linear
    // combination is linked to hidden evaluations on the two canonical
    // tensor domains; no private evaluation scalar is present on the wire.
    zkdl::ZkSumcheckProof sumcheck;
    zkdl::HyraxOpeningProofV3 source_opening;
    zkdl::HyraxOpeningProofV3 output_opening;
    zkdl::PedersenScalarEqualityProof terminal_binding;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            SparseRouteProof& proof,
                            std::string* error = nullptr);
};

/* Prover-local output of the hidden sparse-route terminal relation.  These
 * scalars are needed only to aggregate several Line8 terminals after the FS
 * rho challenges have been sampled.  They are never serialized and never
 * enter VerifierContext. */
struct SparseRouteProverSecrets {
    FrWire terminal_value{};
    FrWire residual_blinder{};
    FrWire sumcheck_blinder{};
};

/* FAC4DNN Line8 proof for a fixed, public view-binding list.  Each binding is
 * reduced by a public selector-MLE Sumcheck to one opening of its canonical
 * source commitment.  The hidden terminal commitments are compressed in the
 * authenticated order with FS rho challenges and connected by one Pedersen
 * equality proof.  No terminal evaluation, residual, or compressed claim is
 * carried on the wire. */
struct FacLine8Proof {
    static const std::uint32_t kVersion = 3U;
    std::uint32_t version = kVersion;
    std::vector<SparseRouteProof> selector_proofs;
    zkdl::PedersenScalarEqualityProof aggregate_binding;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            FacLine8Proof& proof,
                            std::string* error = nullptr);
};

struct Proof {
    static const std::uint32_t kVersion = 3U;
    std::uint32_t version = kVersion;
    RelationStatement statement;
    std::vector<CommitmentWire> commitments;
    // Legacy hiding records are kept as an explicit zero-length wire field for
    // ABI compatibility and are rejected when non-empty.  Formal GATR binds
    // model/private tensors through the hidden IPA/Pedersen opening proofs.
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
    // IPA basis material is independent from the scalar-generator vector and
    // must be supplied by the authenticated resource loader.  Empty H/U is a
    // hard admission failure; no generator-derived fallback is permitted.
    std::vector<zkdl::CpuG1Point> ipa_generators;
    zkdl::CpuG1Point ipa_blinding_generator;
    zkdl::CpuG1Point ipa_u_generator;
    std::vector<CommitmentWire> registered_model_commitments;
    // Legacy hiding-record ABI material is retained only for strict rejection;
    // formal GATR uses the per-tensor Pedersen blinder carried by the native
    // hidden openings below and requires this vector to remain empty.
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    // One blinder is sampled per private/model tensor and reused for every
    // opening of that canonical commitment in this proof.
    std::map<std::string, FrWire> blinding_scalars;
    // Public graph-derived material may be attached to a non-public view
    // commitment (for example group-boundary bits).  The verifier accepts
    // only ids explicitly reconstructed from Graph/CSR, never arbitrary
    // caller-supplied private values.
    std::set<std::string> graph_public_tensor_ids;
    std::map<std::string, std::vector<FrWire>> public_tensor_values;
    // Optional native-core timing sink for the relation's commitment/opening
    // stage.  It is instrumentation only and never affects proof equations.
    double* commitment_open_time_seconds = nullptr;
    // Native core sets this for a requested CUDA proof.  The relation layer
    // currently fails closed until the complete GAT relation (not only a
    // primitive) is implemented on native CUDA cryptographic kernels.
    bool gpu_backend = false;
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
    std::vector<zkdl::CpuG1Point> ipa_generators;
    zkdl::CpuG1Point ipa_blinding_generator;
    zkdl::CpuG1Point ipa_u_generator;
    // Native core sets this for a requested CUDA verification.  The relation
    // layer currently fails closed until every proof family has a native CUDA
    // verifier path.
    bool gpu_backend = false;
};

inline bool same_wire(const FrWire& a, const FrWire& b) noexcept {
    return zkdl::fr_compare(a, b) == 0;
}

inline bool same_bytes(const std::array<byte, 32>& a, const std::array<byte, 32>& b) noexcept {
    return a == b;
}

inline bool same_g1_basis(const std::vector<zkdl::CpuG1Point>& left,
                          const std::vector<zkdl::CpuG1Point>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0U; i < left.size(); ++i)
        if (!zkdl::cpu_g1_equal(left[i], right[i])) return false;
    return true;
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
    zkdl::append_g1(out, record.blinding_generator);
    zkdl::append_g1(out, record.commitment);
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
    return zkdl::read_g1(in, offset, record.blinding_generator) &&
           zkdl::read_g1(in, offset, record.commitment);
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

inline zkdl::CpuIpaOpeningMetadata relation_opening_metadata(
    const OperationDescriptor& descriptor, std::uint32_t commitment_index,
    const std::vector<FrWire>& outer_point, const RelationStatement& statement) {
    if (statement.registry_digest == std::array<byte, 32U>{} ||
        statement.srs_digest == std::array<byte, 32U>{})
        throw std::invalid_argument("relation opening requires authenticated registry/SRS digests");
    std::vector<byte> encoded;
    append_operation(encoded, descriptor);
    zkdl::append_u32(encoded, commitment_index);
    for (const auto& coordinate : outer_point) zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.outer_point = outer_point;
    metadata.descriptor_digest = zkdl::sha256(encoded);
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    return metadata;
}

/* Constraint openings use the same authenticated IPA header as operation
 * openings, but their public descriptor is the canonical constraint wire.
 * Keep this as a distinct overload so a verifier cannot silently hash a
 * lossy OperationDescriptor projection (and so prover/verifier derive the
 * exact same commitment-opening identity). */
inline zkdl::CpuIpaOpeningMetadata relation_opening_metadata(
    const ConstraintDescriptor& descriptor, std::uint32_t commitment_index,
    const std::vector<FrWire>& outer_point, const RelationStatement& statement) {
    if (statement.registry_digest == std::array<byte, 32U>{} ||
        statement.srs_digest == std::array<byte, 32U>{})
        throw std::invalid_argument("relation opening requires authenticated registry/SRS digests");
    std::vector<byte> encoded;
    append_constraint(encoded, descriptor);
    zkdl::append_u32(encoded, commitment_index);
    for (const auto& coordinate : outer_point)
        zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.outer_point = outer_point;
    metadata.descriptor_digest = zkdl::sha256(encoded);
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    return metadata;
}

inline OperationDescriptor public_output_opening_descriptor() {
    OperationDescriptor descriptor;
    descriptor.id = "terminal_opening";
    descriptor.kind = RelationKind::PublicOutput;
    descriptor.degree = 1U;
    descriptor.source_refs = {"public_Y"};
    descriptor.tensor_refs = {"public_Y"};
    descriptor.mapping = "canonical-public-output";
    descriptor.padding = "authenticated-public-output";
    descriptor.terminal_claim = "public_Y";
    descriptor.terminal_point = "public_Y";
    return descriptor;
}

inline FrWire relation_blinding_scalar(
    const TensorWitness& tensor, const std::map<std::string, FrWire>& blinders) {
    if (tensor.visibility == 1U) return FrWire{};
    const auto found = blinders.find(tensor.id);
    if (found == blinders.end() || !zkdl::fr_canonical(found->second) ||
        zkdl::fr_compare(found->second, FrWire{}) == 0)
        throw std::invalid_argument("BLOCKED: missing nonzero per-tensor IPA blinder: " + tensor.id);
    return found->second;
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
    zkdl::serialize_zk_sumcheck(proof.zk_sumcheck, sumcheck);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck.size()));
    out.insert(out.end(), sumcheck.begin(), sumcheck.end());
    std::vector<byte> terminal;
    zkdl::serialize_zk_terminal_relation(proof.terminal_relation, terminal);
    zkdl::append_u64(out, static_cast<std::uint64_t>(terminal.size()));
    out.insert(out.end(), terminal.begin(), terminal.end());
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
    if (!zkdl::read_u64(in, offset, proof_bytes) || proof_bytes == 0U ||
        proof_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes ||
        proof_bytes > in.size() - offset) return false;
    std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                              in.begin() + static_cast<std::ptrdiff_t>(offset + proof_bytes));
    offset += static_cast<size_t>(proof_bytes);
    if (!zkdl::deserialize_zk_sumcheck(encoded, proof.zk_sumcheck, nullptr)) return false;
    if (!zkdl::read_u64(in, offset, proof_bytes) || proof_bytes == 0U ||
        proof_bytes > zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes ||
        proof_bytes > in.size() - offset) return false;
    encoded.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                   in.begin() + static_cast<std::ptrdiff_t>(offset + proof_bytes));
    offset += static_cast<size_t>(proof_bytes);
    if (!zkdl::deserialize_zk_terminal_relation(
            encoded, proof.terminal_relation, nullptr)) return false;
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
    out.push_back('F'); out.push_back('A'); out.push_back('C'); out.push_back('3');
    zkdl::append_u32(out, version);
    zkdl::append_u64(out, static_cast<std::uint64_t>(selector_proofs.size()));
    for (const auto& route : selector_proofs) {
        const std::vector<byte> encoded = route.serialize();
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }
    std::vector<byte> aggregate;
    zkdl::serialize_pedersen_scalar_equality(aggregate_binding, aggregate);
    zkdl::append_u64(out, static_cast<std::uint64_t>(aggregate.size()));
    out.insert(out.end(), aggregate.begin(), aggregate.end());
    return out;
}

inline bool FacLine8Proof::deserialize(const std::vector<byte>& bytes,
                                       FacLine8Proof& proof,
                                       std::string* error) {
    FacLine8Proof parsed;
    std::uint64_t count = 0U, nested_bytes = 0U;
    size_t offset = 0U;
    constexpr std::uint64_t kMaxWireItems = kMaxRelationOpenings;
    if (bytes.size() < 4U || bytes[0] != 'F' || bytes[1] != 'A' ||
        bytes[2] != 'C' || bytes[3] != '3') {
        if (error) *error = "invalid FAC4DNN Line8 proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, parsed.version) || parsed.version != kVersion ||
        !zkdl::read_u64(bytes, offset, count) || count == 0U || count > kMaxWireItems) {
        if (error) *error = "invalid FAC4DNN Line8 descriptor/header";
        return false;
    }
    parsed.selector_proofs.resize(static_cast<std::size_t>(count));
    for (auto& route : parsed.selector_proofs) {
        if (!zkdl::read_u64(bytes, offset, nested_bytes) || nested_bytes == 0U ||
            nested_bytes > bytes.size() - offset ||
            nested_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes) {
            if (error) *error = "invalid FAC4DNN selector route length";
            return false;
        }
        std::vector<byte> encoded(
            bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + nested_bytes));
        offset += static_cast<std::size_t>(nested_bytes);
        if (!SparseRouteProof::deserialize(encoded, route, error)) return false;
    }
    if (!zkdl::read_u64(bytes, offset, nested_bytes) || nested_bytes == 0U ||
        nested_bytes > bytes.size() - offset ||
        nested_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes) {
        if (error) *error = "invalid FAC4DNN aggregate binding length";
        return false;
    }
    std::vector<byte> aggregate(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(offset + nested_bytes));
    offset += static_cast<std::size_t>(nested_bytes);
    if (!zkdl::deserialize_pedersen_scalar_equality(
            aggregate, parsed.aggregate_binding, error)) return false;
    if (offset != bytes.size()) {
        if (error) *error = "trailing FAC4DNN Line8 proof bytes";
        return false;
    }
    proof = std::move(parsed);
    return true;
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
    std::vector<byte> sumcheck_bytes;
    zkdl::serialize_zk_sumcheck(sumcheck, sumcheck_bytes);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck_bytes.size()));
    out.insert(out.end(), sumcheck_bytes.begin(), sumcheck_bytes.end());
    zkdl::serialize_hyrax_opening_v3(source_opening, out);
    zkdl::serialize_hyrax_opening_v3(output_opening, out);
    std::vector<byte> terminal_bytes;
    zkdl::serialize_pedersen_scalar_equality(terminal_binding, terminal_bytes);
    zkdl::append_u64(out, static_cast<std::uint64_t>(terminal_bytes.size()));
    out.insert(out.end(), terminal_bytes.begin(), terminal_bytes.end());
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
        !zkdl::read_u64(bytes, offset, sumcheck_bytes) ||
        sumcheck_bytes == 0U ||
        sumcheck_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes ||
        sumcheck_bytes > bytes.size() - offset) {
        if (error) *error = "truncated sparse route proof header";
        return false;
    }
    std::vector<byte> sumcheck_bytes_vector(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(offset + sumcheck_bytes));
    offset += static_cast<size_t>(sumcheck_bytes);
    std::uint64_t terminal_bytes = 0U;
    if (!zkdl::deserialize_zk_sumcheck(sumcheck_bytes_vector, parsed.sumcheck, error) ||
        !zkdl::deserialize_hyrax_opening_v3(bytes, offset, parsed.source_opening) ||
        !zkdl::deserialize_hyrax_opening_v3(bytes, offset, parsed.output_opening) ||
        !zkdl::read_u64(bytes, offset, terminal_bytes) || terminal_bytes == 0U ||
        terminal_bytes > 4096U || terminal_bytes != bytes.size() - offset) {
        if (error) *error = "truncated or trailing sparse route proof";
        return false;
    }
    const std::vector<byte> terminal(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                     bytes.end());
    if (!zkdl::deserialize_pedersen_scalar_equality(terminal, parsed.terminal_binding, error))
        return false;
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
        parsed.source_feature_point.size() + parsed.node_point.size() > kMaxRelationPointBits ||
        parsed.feature_point.size() + parsed.edge_point.size() > kMaxRelationPointBits ||
        parsed.sumcheck.degree != 2U ||
        parsed.sumcheck.domain_size != std::max<std::uint64_t>(
            2U, static_cast<std::uint64_t>(parsed.node_domain) * parsed.source_feature_domain) ||
        parsed.source_opening.generator_size !=
            static_cast<std::uint64_t>(parsed.node_domain) * parsed.source_feature_domain ||
        parsed.output_opening.generator_size !=
            static_cast<std::uint64_t>(parsed.edge_domain) * parsed.feature_domain) {
        if (error) *error = "sparse route proof shape is not canonical";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline std::vector<byte> Proof::serialize() const {
    if (!hiding_commitment_records.empty())
        throw std::invalid_argument("legacy hiding commitment records are not part of formal GATR wire");
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
    if (!zkdl::read_u64(bytes, offset, count) || count != 0U) {
        if (error) *error = "invalid GAT hiding commitment section";
        return false;
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

/*
 * Hidden generic Sumcheck round engine.  This deliberately lives beside the
 * relation descriptor rather than in a second proof implementation: all
 * commitments, Sigma equations and Fiat--Shamir challenges are delegated to
 * proof.cuh/cpu-backend.cuh's authenticated Pedersen primitives.  The engine
 * is consumed by ConstraintProof together with the terminal DAG/opening
 * relation; legacy relation families below remain fail-closed until their
 * value-revealing wires are migrated to this native hidden interface.
 */
struct ZkSumcheckProverResult {
    zkdl::ZkSumcheckProof proof;
    std::vector<Fr_t> challenges;
    std::vector<FrWire> folded_terminal_values;
    FrWire final_q_blinder{};
};

struct ZkSumcheckVerifierResult {
    std::vector<Fr_t> challenges;
    G1Jacobian_t final_claim_commitment{};
};

inline bool zksc_digest_nonzero(const std::array<byte, 32U>& digest) {
    return digest != std::array<byte, 32U>{{}};
}

inline bool zksc_statement_resources_valid(const RelationStatement& statement,
                                           std::string* error = nullptr) {
    if (!zksc_digest_nonzero(statement.registry_digest) ||
        !zksc_digest_nonzero(statement.srs_digest)) {
        if (error) *error = "ZkSumcheck requires authenticated registry/SRS digests";
        return false;
    }
    return true;
}

inline std::array<byte, 32U> zksc_role_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round_index, std::uint32_t commitment_index,
    std::uint32_t related_index) {
    std::vector<byte> encoded;
    append_constraint(encoded, descriptor);
    zkdl::append_string(encoded, operation_context);
    zkdl::append_string(encoded, role);
    zkdl::append_u32(encoded, round_index);
    zkdl::append_u32(encoded, commitment_index);
    zkdl::append_u32(encoded, related_index);
    append_digest(encoded, statement.statement_digest);
    append_digest(encoded, statement.model_digest);
    append_digest(encoded, statement.table_digest);
    append_digest(encoded, statement.ledger_digest);
    append_digest(encoded, statement.registry_digest);
    append_digest(encoded, statement.srs_digest);
    return zkdl::sha256(encoded);
}

inline zkdl::PedersenProofContext zksc_make_context(
    std::uint32_t operation, std::uint32_t degree, std::uint32_t basis_size,
    std::uint32_t commitment_index, std::uint32_t related_index,
    const std::array<byte, 32U>& basis_digest,
    const std::array<byte, 32U>& descriptor_digest,
    const RelationStatement& statement) {
    zkdl::PedersenProofContext context;
    context.commitment_index = commitment_index;
    context.operation = operation;
    context.commitment_a_index = commitment_index;
    context.commitment_b_index = related_index;
    context.commitment_c_index = zkdl::PedersenProofContext::kUnsetIndex;
    context.evaluation_commitment_index = zkdl::PedersenProofContext::kUnsetIndex;
    if (operation == zkdl::PedersenProofContext::kOperationVectorLinear) {
        context.evaluation_commitment_index = related_index;
        context.commitment_b_index = zkdl::PedersenProofContext::kUnsetIndex;
    }
    context.degree = degree;
    context.basis_size = basis_size;
    context.basis_digest = basis_digest;
    context.descriptor_digest = descriptor_digest;
    context.registry_digest = statement.registry_digest;
    context.srs_digest = statement.srs_digest;
    return context;
}

inline std::uint32_t zksc_round_commitment_index(std::uint32_t round) {
    constexpr std::uint32_t kStride = 4U;
    if (round > (std::numeric_limits<std::uint32_t>::max() / kStride) - 2U)
        throw std::invalid_argument("ZkSumcheck round commitment index overflow");
    return round * kStride;
}

inline std::uint32_t zksc_endpoint_index(std::uint32_t round) {
    return zksc_round_commitment_index(round) + 1U;
}

inline std::uint32_t zksc_evaluation_index(std::uint32_t round) {
    return zksc_round_commitment_index(round) + 2U;
}

inline std::array<byte, 32U> zksc_proof_descriptor_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context) {
    return zksc_role_digest(descriptor, statement, operation_context,
                            "initial-public-claim", 0U, zksc_endpoint_index(0U),
                            zkdl::PedersenProofContext::kUnsetIndex);
}

inline std::array<byte, 32U> zksc_vector_role_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t related_index) {
    return zksc_role_digest(descriptor, statement, operation_context, role, round,
                            zksc_round_commitment_index(round), related_index);
}

inline std::array<byte, 32U> zksc_scalar_role_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t commitment_index,
    std::uint32_t related_index) {
    return zksc_role_digest(descriptor, statement, operation_context, role, round,
                            commitment_index, related_index);
}

inline FrWire zksc_eval_coefficients(const std::vector<FrWire>& coefficients,
                                     const FrWire& point) {
    if (coefficients.empty()) throw std::invalid_argument("empty ZkSumcheck coefficient vector");
    FrWire result{};
    for (std::size_t index = coefficients.size(); index-- > 0U;) {
        result = zkdl::fr_add(zkdl::fr_mul(result, point), coefficients[index]);
    }
    return result;
}

inline bool zksc_interpolate_coefficients(const std::vector<FrWire>& evaluations,
                                          std::vector<FrWire>& coefficients,
                                          std::string* error = nullptr) {
    if (evaluations.empty() || evaluations.size() > 9U) {
        if (error) *error = "ZkSumcheck interpolation degree is out of bounds";
        return false;
    }
    coefficients.assign(evaluations.size(), FrWire{});
    try {
        for (std::size_t i = 0U; i < evaluations.size(); ++i) {
            std::vector<FrWire> basis(1U, zkdl::fr_from_u32(1U));
            FrWire denominator = zkdl::fr_from_u32(1U);
            const FrWire xi = zkdl::fr_from_u32(static_cast<std::uint32_t>(i));
            for (std::size_t j = 0U; j < evaluations.size(); ++j) {
                if (i == j) continue;
                const FrWire xj = zkdl::fr_from_u32(static_cast<std::uint32_t>(j));
                std::vector<FrWire> next(basis.size() + 1U, FrWire{});
                const FrWire negative_xj = zkdl::fr_neg(xj);
                for (std::size_t k = 0U; k < basis.size(); ++k) {
                    next[k] = zkdl::fr_add(next[k], zkdl::fr_mul(basis[k], negative_xj));
                    next[k + 1U] = zkdl::fr_add(next[k + 1U], basis[k]);
                }
                basis.swap(next);
                denominator = zkdl::fr_mul(denominator, zkdl::fr_sub(xi, xj));
            }
            const FrWire scale = zkdl::fr_mul(evaluations[i], zkdl::cpu_fr_inv(denominator));
            for (std::size_t k = 0U; k < coefficients.size(); ++k)
                coefficients[k] = zkdl::fr_add(coefficients[k],
                                               zkdl::fr_mul(scale, basis[k]));
        }
        for (std::size_t i = 0U; i < evaluations.size(); ++i)
            if (!same_wire(zksc_eval_coefficients(coefficients,
                                                   zkdl::fr_from_u32(static_cast<std::uint32_t>(i))),
                           evaluations[i])) {
                if (error) *error = "ZkSumcheck interpolation failed its evaluation check";
                return false;
            }
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    return true;
}

inline void zksc_append_nested(std::vector<byte>& out,
                               const std::vector<byte>& nested) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
}

inline std::vector<byte> zksc_round_pre_bytes(
    const zkdl::ZkSumcheckRoundProof& round) {
    std::vector<byte> out;
    zkdl::append_u32(out, round.version);
    zkdl::append_u32(out, round.round_index);
    zkdl::append_g1(out, round.coefficient_commitment);
    zkdl::append_g1(out, round.endpoint_commitment);
    std::vector<byte> endpoint;
    zkdl::serialize_pedersen_vector_linear(round.endpoint_linear, endpoint);
    zksc_append_nested(out, endpoint);
    zkdl::append_u32(out, static_cast<std::uint32_t>(round.link_kind));
    if (round.link_kind == zkdl::ZkSumcheckLinkKind::InitialPublicClaim) {
        std::vector<byte> initial;
        zkdl::serialize_pedersen_scalar_value(round.initial_claim_binding, initial);
        zksc_append_nested(out, initial);
        zkdl::append_u64(out, 0U);
    } else if (round.link_kind == zkdl::ZkSumcheckLinkKind::PreviousEvaluation) {
        zkdl::append_u64(out, 0U);
        std::vector<byte> previous;
        zkdl::serialize_pedersen_scalar_equality(round.previous_claim_binding, previous);
        zksc_append_nested(out, previous);
    } else {
        throw std::invalid_argument("unknown ZkSumcheck link branch");
    }
    return out;
}

inline std::vector<byte> zksc_round_query_bytes(
    const zkdl::ZkSumcheckRoundProof& round) {
    std::vector<byte> out;
    zkdl::append_u32(out, round.round_index);
    zkdl::append_g1(out, round.evaluation_commitment);
    std::vector<byte> evaluation;
    zkdl::serialize_pedersen_vector_linear(round.evaluation_linear, evaluation);
    zksc_append_nested(out, evaluation);
    return out;
}

inline void zksc_absorb_header(zkdl::Transcript& transcript,
                               const zkdl::ZkSumcheckProof& proof) {
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, proof.version);
    zkdl::append_string(encoded, proof.label);
    zkdl::append_u32(encoded, proof.degree);
    zkdl::append_u32(encoded, proof.domain_size);
    zkdl::append_pedersen_context(encoded, proof.initial_claim_context);
    encoded.insert(encoded.end(), proof.descriptor_digest.begin(), proof.descriptor_digest.end());
    encoded.insert(encoded.end(), proof.registry_digest.begin(), proof.registry_digest.end());
    encoded.insert(encoded.end(), proof.srs_digest.begin(), proof.srs_digest.end());
    encoded.insert(encoded.end(), proof.basis_digest.begin(), proof.basis_digest.end());
    zkdl::append_g1(encoded, proof.blinding_generator);
    zkdl::append_g1(encoded, proof.u_generator);
    transcript.absorb("gat.zksumcheck.header", encoded);
}

inline zkdl::PedersenProofContext zksc_make_vector_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t related_index,
    const std::array<byte, 32U>& basis_digest, std::uint32_t degree) {
    return zksc_make_context(
        zkdl::PedersenProofContext::kOperationVectorLinear, degree, degree + 1U,
        zksc_round_commitment_index(round), related_index, basis_digest,
        zksc_vector_role_digest(descriptor, statement, operation_context, role,
                                round, related_index), statement);
}

inline zkdl::PedersenProofContext zksc_make_scalar_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t commitment_index,
    std::uint32_t related_index, std::uint32_t operation,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator) {
    return zksc_make_context(
        operation, 1U, 0U, commitment_index, related_index,
        zkdl::cpu_ipa_basis_digest({}, blinding_generator, u_generator, 0U),
        zksc_scalar_role_digest(descriptor, statement, operation_context, role,
                                round, commitment_index, related_index), statement);
}

inline bool zksc_validate_inputs(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, std::string* error) {
    if (!descriptor.valid(error) || descriptor.degree != degree ||
        degree == 0U || degree > 8U || tensor_values.empty() ||
        domain_size == 0U || (domain_size & (domain_size - 1U)) != 0U ||
        domain_size > (1U << 30U) || generators.size() < static_cast<std::size_t>(degree + 1U) ||
        !zkdl::fr_canonical(initial_claim) || operation_context.empty() ||
        !zksc_statement_resources_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid ZkSumcheck prover inputs";
        return false;
    }
    if (!zkdl::cpu_pedersen_basis_valid(
            std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + degree + 1U),
            blinding_generator, u_generator, degree + 1U, error)) return false;
    for (const auto& tensor : tensor_values) {
        if (tensor.size() != domain_size) {
            if (error) *error = "ZkSumcheck tensor/domain mismatch";
            return false;
        }
        for (const auto& value : tensor)
            if (!zkdl::fr_canonical(value)) {
                if (error) *error = "ZkSumcheck tensor value is not canonical";
                return false;
            }
    }
    return true;
}

inline bool prove_zk_sumcheck(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, zkdl::Transcript& parent_transcript,
    ZkSumcheckProverResult& result, std::string* error = nullptr) {
    try {
        if (!zksc_validate_inputs(descriptor, tensor_values, degree, domain_size,
                                  generators, blinding_generator, u_generator,
                                  initial_claim, statement, operation_context, error)) return false;
        const std::uint32_t bits = log2_exact(domain_size);
        if (bits == std::numeric_limits<std::uint32_t>::max() ||
            bits == 0U || bits > zkdl::ZkSumcheckProof::kMaxRounds) {
            if (error) *error = "ZkSumcheck domain has no supported round decomposition";
            return false;
        }
        result = ZkSumcheckProverResult{};
        auto& proof = result.proof;
        proof.label = operation_context;
        proof.degree = degree;
        proof.domain_size = domain_size;
        proof.descriptor_digest = zksc_proof_descriptor_digest(
            descriptor, statement, operation_context);
        proof.registry_digest = statement.registry_digest;
        proof.srs_digest = statement.srs_digest;
        const std::vector<zkdl::CpuG1Point> round_generators(
            generators.begin(), generators.begin() + degree + 1U);
        proof.basis_digest = zkdl::cpu_ipa_basis_digest(
            round_generators, blinding_generator, u_generator, degree + 1U);
        proof.blinding_generator = zkdl::cpu_g1_to_native(blinding_generator);
        proof.u_generator = zkdl::cpu_g1_to_native(u_generator);
        proof.initial_claim_context = zksc_make_scalar_context(
            descriptor, statement, operation_context, "initial-public-claim", 0U,
            zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
            zkdl::PedersenProofContext::kOperationScalarValue,
            blinding_generator, u_generator);
        proof.initial_claim_context.descriptor_digest = proof.descriptor_digest;
        proof.initial_claim_context.registry_digest = proof.registry_digest;
        proof.initial_claim_context.srs_digest = proof.srs_digest;

        zksc_absorb_header(parent_transcript, proof);
        std::vector<std::vector<FrWire>> current = tensor_values;
        FrWire previous_value = initial_claim;
        FrWire previous_blinder{};
        for (std::uint32_t round_index = 0U; round_index < bits; ++round_index) {
            if (current.empty() || current.front().size() < 2U ||
                (current.front().size() & 1U) != 0U) {
                if (error) *error = "ZkSumcheck current tensor domain cannot be folded";
                return false;
            }
            std::vector<FrWire> evaluations(degree + 1U, FrWire{});
            const std::size_t next_size = current.front().size() / 2U;
            for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
                const FrWire x = zkdl::fr_from_u32(evaluation);
                for (std::size_t row = 0U; row < next_size; ++row) {
                    std::vector<FrWire> values;
                    values.reserve(current.size());
                    for (const auto& tensor : current)
                        values.push_back(zkdl::fr_add(
                            tensor[2U * row],
                            zkdl::fr_mul(x, zkdl::fr_sub(
                                tensor[2U * row + 1U], tensor[2U * row]))));
                    evaluations[evaluation] = zkdl::fr_add(
                        evaluations[evaluation], eval_constraint(descriptor, values));
                }
            }
            std::vector<FrWire> coefficients;
            if (!zksc_interpolate_coefficients(evaluations, coefficients, error)) return false;
            const FrWire endpoint_value = zksc_eval_coefficients(
                coefficients, zkdl::fr_from_u32(0U));
            const FrWire endpoint_at_one = zksc_eval_coefficients(
                coefficients, zkdl::fr_from_u32(1U));
            const FrWire endpoint = zkdl::fr_add(endpoint_value, endpoint_at_one);
            const FrWire r_c = zkdl::cpu_fresh_csprng_scalar();
            const FrWire r_e = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point coefficient_point = zkdl::cpu_g1_add(
                zkdl::cpu_g1_sum(round_generators, coefficients),
                zkdl::cpu_g1_mul(blinding_generator, r_c));
            const zkdl::CpuG1Point endpoint_commitment = zkdl::cpu_pedersen_scalar_commit(
                endpoint, r_e, blinding_generator, u_generator);
            const std::array<byte, 32U> endpoint_digest = zksc_vector_role_digest(
                descriptor, statement, operation_context, "endpoint", round_index,
                zksc_endpoint_index(round_index));
            const auto endpoint_context = zksc_make_context(
                zkdl::PedersenProofContext::kOperationVectorLinear, degree, degree + 1U,
                zksc_round_commitment_index(round_index), zksc_endpoint_index(round_index),
                proof.basis_digest, endpoint_digest, statement);
            std::vector<FrWire> endpoint_weights(degree + 1U, zkdl::fr_from_u32(1U));
            endpoint_weights[0] = zkdl::fr_from_u32(2U);
            auto endpoint_linear = zkdl::cpu_open_pedersen_vector_linear(
                round_generators, coefficients, r_c, endpoint_weights, r_e,
                coefficient_point, endpoint_commitment, blinding_generator,
                u_generator, endpoint_context);

            zkdl::ZkSumcheckRoundProof round;
            round.round_index = round_index;
            round.coefficient_commitment = zkdl::cpu_g1_to_native(coefficient_point);
            round.endpoint_commitment = zkdl::cpu_g1_to_native(endpoint_commitment);
            round.endpoint_linear = endpoint_linear.proof;
            if (round_index == 0U) {
                round.link_kind = zkdl::ZkSumcheckLinkKind::InitialPublicClaim;
                const auto initial_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "initial-public-claim", 0U,
                    zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
                    zkdl::PedersenProofContext::kOperationScalarValue,
                    blinding_generator, u_generator);
                round.initial_claim_binding = zkdl::cpu_open_pedersen_scalar_value_public(
                    initial_claim, r_e, endpoint_commitment, blinding_generator,
                    u_generator, initial_context).proof;
            } else {
                round.link_kind = zkdl::ZkSumcheckLinkKind::PreviousEvaluation;
                const auto previous_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "previous-evaluation",
                    round_index, zksc_endpoint_index(round_index),
                    zksc_evaluation_index(round_index - 1U),
                    zkdl::PedersenProofContext::kOperationScalarEquality,
                    blinding_generator, u_generator);
                round.previous_claim_binding = zkdl::cpu_open_pedersen_scalar_equality(
                    previous_value, r_e, previous_blinder, endpoint_commitment,
                    zkdl::cpu_g1_from_native(proof.rounds.back().evaluation_commitment),
                    blinding_generator, u_generator, previous_context).proof;
            }
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.pre",
                zksc_round_pre_bytes(round));
            const Fr_t challenge = parent_transcript.challenge(
                "gat.zksumcheck." + operation_context + ".round." +
                std::to_string(round_index) + ".challenge");
            const FrWire challenge_wire = zkdl::fr_wire(challenge);
            std::vector<FrWire> query_weights(degree + 1U, zkdl::fr_from_u32(1U));
            for (std::uint32_t power = 1U; power <= degree; ++power)
                query_weights[power] = zkdl::fr_mul(query_weights[power - 1U], challenge_wire);
            const FrWire query_value = zksc_eval_coefficients(coefficients, challenge_wire);
            const FrWire r_q = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point query_commitment = zkdl::cpu_pedersen_scalar_commit(
                query_value, r_q, blinding_generator, u_generator);
            const auto query_context = zksc_make_vector_context(
                descriptor, statement, operation_context, "evaluation", round_index,
                zksc_evaluation_index(round_index), proof.basis_digest, degree);
            auto query_linear = zkdl::cpu_open_pedersen_vector_linear(
                round_generators, coefficients, r_c, query_weights, r_q,
                coefficient_point, query_commitment, blinding_generator,
                u_generator, query_context);
            round.evaluation_commitment = zkdl::cpu_g1_to_native(query_commitment);
            round.evaluation_linear = query_linear.proof;
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.query",
                zksc_round_query_bytes(round));
            proof.rounds.push_back(round);
            previous_value = query_value;
            previous_blinder = r_q;
            for (auto& tensor : current) fold_values(tensor, challenge_wire);
            result.challenges.push_back(challenge);
        }
        proof.final_claim_commitment = proof.rounds.back().evaluation_commitment;
        result.folded_terminal_values.reserve(current.size());
        for (const auto& tensor : current) result.folded_terminal_values.push_back(tensor.front());
        result.final_q_blinder = previous_blinder;
        std::vector<byte> final_bytes;
        zkdl::append_g1(final_bytes, proof.final_claim_commitment);
        parent_transcript.absorb("gat.zksumcheck." + operation_context + ".final", final_bytes);
        std::string shape_error;
        if (!zkdl::zk_sumcheck_shape_valid(proof, &shape_error)) {
            if (error) *error = shape_error.empty() ? "invalid generated ZkSumcheck proof" : shape_error;
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool verify_zk_sumcheck(
    const zkdl::ZkSumcheckProof& proof, const ConstraintDescriptor& descriptor,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, zkdl::Transcript& parent_transcript,
    ZkSumcheckVerifierResult& result, std::string* error = nullptr) {
    try {
        if (!descriptor.valid(error) || proof.label != operation_context ||
            proof.degree != degree || proof.domain_size != domain_size ||
            !zkdl::zk_sumcheck_shape_valid(proof, error) ||
            !zksc_statement_resources_valid(statement, error) ||
            !zkdl::fr_canonical(initial_claim) || generators.size() < degree + 1U) {
            if (error && error->empty()) *error = "invalid ZkSumcheck verifier inputs";
            return false;
        }
        const auto strict_external = [&](const G1Jacobian_t& wire,
                                         const zkdl::CpuG1Point& expected,
                                         const char* label) {
            std::array<byte, 48U> encoded{};
            std::string point_error;
            if (!zkdl_g1_wire::encode(wire, encoded, &point_error) ||
                !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(wire), expected)) {
                if (error) *error = std::string("ZkSumcheck ") + label + " basis mismatch";
                return false;
            }
            return true;
        };
        if (!strict_external(proof.blinding_generator, blinding_generator, "H") ||
            !strict_external(proof.u_generator, u_generator, "U")) return false;
        const std::vector<zkdl::CpuG1Point> round_generators(
            generators.begin(), generators.begin() + degree + 1U);
        if (!zkdl::cpu_pedersen_basis_valid(round_generators, blinding_generator,
                                            u_generator, degree + 1U, error) ||
            proof.basis_digest != zkdl::cpu_ipa_basis_digest(
                round_generators, blinding_generator, u_generator, degree + 1U) ||
            proof.descriptor_digest != zksc_proof_descriptor_digest(
                descriptor, statement, operation_context) ||
            proof.registry_digest != statement.registry_digest ||
            proof.srs_digest != statement.srs_digest) {
            if (error && error->empty()) *error = "ZkSumcheck descriptor/resource basis mismatch";
            return false;
        }
        const auto expected_initial_context = zksc_make_scalar_context(
            descriptor, statement, operation_context, "initial-public-claim", 0U,
            zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
            zkdl::PedersenProofContext::kOperationScalarValue,
            blinding_generator, u_generator);
        if (!zkdl::cpu_pedersen_context_equal(
                proof.initial_claim_context, expected_initial_context)) {
            if (error) *error = "ZkSumcheck initial claim context mismatch";
            return false;
        }
        result = ZkSumcheckVerifierResult{};
        zksc_absorb_header(parent_transcript, proof);
        for (std::size_t round_index = 0U; round_index < proof.rounds.size(); ++round_index) {
            const auto& round = proof.rounds[round_index];
            const auto endpoint_context = zksc_make_context(
                zkdl::PedersenProofContext::kOperationVectorLinear, degree, degree + 1U,
                zksc_round_commitment_index(static_cast<std::uint32_t>(round_index)),
                zksc_endpoint_index(static_cast<std::uint32_t>(round_index)),
                proof.basis_digest,
                zksc_vector_role_digest(descriptor, statement, operation_context,
                                        "endpoint", static_cast<std::uint32_t>(round_index),
                                        zksc_endpoint_index(static_cast<std::uint32_t>(round_index))),
                statement);
            std::vector<FrWire> endpoint_weights(degree + 1U, zkdl::fr_from_u32(1U));
            endpoint_weights[0] = zkdl::fr_from_u32(2U);
            if (!zkdl::cpu_verify_pedersen_vector_linear(
                    round.endpoint_linear,
                    zkdl::cpu_g1_from_native(round.coefficient_commitment),
                    zkdl::cpu_g1_from_native(round.endpoint_commitment), round_generators,
                    endpoint_weights, blinding_generator, u_generator, endpoint_context,
                    error)) return false;
            if (round_index == 0U) {
                const auto initial_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "initial-public-claim", 0U,
                    zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
                    zkdl::PedersenProofContext::kOperationScalarValue,
                    blinding_generator, u_generator);
                if (!zkdl::cpu_verify_pedersen_scalar_value(
                        round.initial_claim_binding, initial_claim,
                        zkdl::cpu_g1_from_native(round.endpoint_commitment),
                        blinding_generator, u_generator, initial_context, error)) return false;
            } else {
                const auto previous_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "previous-evaluation",
                    static_cast<std::uint32_t>(round_index),
                    zksc_endpoint_index(static_cast<std::uint32_t>(round_index)),
                    zksc_evaluation_index(static_cast<std::uint32_t>(round_index - 1U)),
                    zkdl::PedersenProofContext::kOperationScalarEquality,
                    blinding_generator, u_generator);
                if (!zkdl::cpu_verify_pedersen_scalar_equality(
                        round.previous_claim_binding,
                        zkdl::cpu_g1_from_native(round.endpoint_commitment),
                        zkdl::cpu_g1_from_native(
                            proof.rounds[round_index - 1U].evaluation_commitment),
                        blinding_generator, u_generator, previous_context, error)) return false;
            }
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.pre",
                zksc_round_pre_bytes(round));
            const Fr_t challenge = parent_transcript.challenge(
                "gat.zksumcheck." + operation_context + ".round." +
                std::to_string(round_index) + ".challenge");
            const FrWire challenge_wire = zkdl::fr_wire(challenge);
            std::vector<FrWire> query_weights(degree + 1U, zkdl::fr_from_u32(1U));
            for (std::uint32_t power = 1U; power <= degree; ++power)
                query_weights[power] = zkdl::fr_mul(query_weights[power - 1U], challenge_wire);
            const auto query_context = zksc_make_vector_context(
                descriptor, statement, operation_context, "evaluation",
                static_cast<std::uint32_t>(round_index),
                zksc_evaluation_index(static_cast<std::uint32_t>(round_index)),
                proof.basis_digest, degree);
            if (!zkdl::cpu_verify_pedersen_vector_linear(
                    round.evaluation_linear,
                    zkdl::cpu_g1_from_native(round.coefficient_commitment),
                    zkdl::cpu_g1_from_native(round.evaluation_commitment), round_generators,
                    query_weights, blinding_generator, u_generator, query_context,
                    error)) return false;
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.query",
                zksc_round_query_bytes(round));
            result.challenges.push_back(challenge);
        }
        result.final_claim_commitment = proof.final_claim_commitment;
        std::vector<byte> final_bytes;
        zkdl::append_g1(final_bytes, proof.final_claim_commitment);
        parent_transcript.absorb("gat.zksumcheck." + operation_context + ".final", final_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

/*
 * Hidden terminal relation proof.
 *
 * This seam is intentionally independent from ConstraintProof.  It commits
 * each descriptor tensor's MLE evaluation as T_i, opens the canonical tensor
 * commitment with Hyrax v4, and reduces every higher-degree monomial through
 * the deterministic Pedersen product DAG.  No terminal value or blinder is
 * serialized; the returned prover result keeps those scalars local for the
 * caller that owns the canonical witness.
 */
struct ZkTerminalRelationProverResult {
    zkdl::ZkTerminalRelationProof proof;
    std::vector<FrWire> terminal_values;
    std::vector<FrWire> terminal_blinders;
    std::vector<FrWire> product_blinders;
    FrWire residual_blinder{};
};

struct ZkTerminalRelationVerifierResult {
    zkdl::CpuG1Point residual_commitment;
    std::vector<zkdl::CpuG1Point> terminal_commitments;
};

struct ZkTerminalProductPlan {
    std::uint32_t monomial_index = 0U;
    std::uint32_t factor_step = 0U;
    std::uint32_t left_commitment_index = 0U;
    std::uint32_t right_commitment_index = 0U;
    std::uint32_t output_commitment_index = 0U;
};

constexpr std::uint32_t kZkTerminalEvaluationIndexBase = 0x40000000U;
constexpr std::uint32_t kZkTerminalProductIndexBase = 0x80000000U;
constexpr std::uint32_t kZkTerminalResidualIndex = 0xF0000000U;
constexpr std::uint32_t kZkTerminalExternalIndex = 0xF0000001U;

inline std::uint32_t zktr_terminal_evaluation_index(std::size_t tensor_index) {
    if (tensor_index >= 0x40000000U)
        throw std::invalid_argument("terminal evaluation commitment index overflow");
    return kZkTerminalEvaluationIndexBase + static_cast<std::uint32_t>(tensor_index);
}

inline std::uint32_t zktr_product_index(std::size_t node_index) {
    if (node_index >= 0x70000000U)
        throw std::invalid_argument("terminal product commitment index overflow");
    return kZkTerminalProductIndexBase + static_cast<std::uint32_t>(node_index);
}

inline bool zktr_all_statement_digests_valid(const RelationStatement& statement,
                                             std::string* error = nullptr) {
    const std::array<byte, 32U> zero{{}};
    if (statement.statement_digest == zero || statement.model_digest == zero ||
        statement.table_digest == zero || statement.ledger_digest == zero ||
        statement.registry_digest == zero || statement.srs_digest == zero) {
        if (error) *error = "terminal relation requires authenticated statement resources";
        return false;
    }
    return true;
}

inline bool zktr_validate_common_inputs(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<FrWire>& tensor_blinders,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    const std::string& operation_context,
    std::string* error = nullptr) {
    if (!descriptor.valid(error) || tensor_values.empty() ||
        tensor_values.size() != canonical_commitments.size() ||
        tensor_values.size() != tensor_blinders.size() ||
        tensor_values.size() != commitment_indices.size() ||
        tensor_values.size() > zkdl::ZkTerminalRelationProof::kMaxTerminalCommitments ||
        evaluation_point.size() >= 19U || evaluation_point.empty() ||
        operation_context.empty() || !zktr_all_statement_digests_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid terminal relation inputs";
        return false;
    }
    const std::uint64_t domain64 = std::uint64_t(1U) << evaluation_point.size();
    if (domain64 > (1U << 18U) || generators.size() != domain64 ||
        !zkdl::cpu_pedersen_basis_valid(generators, blinding_generator, u_generator,
                                        static_cast<std::uint32_t>(domain64), error))
        return false;
    std::set<std::uint32_t> seen_indices;
    for (std::size_t index = 0U; index < tensor_values.size(); ++index) {
        if (commitment_indices[index] >= kZkTerminalEvaluationIndexBase ||
            !seen_indices.insert(commitment_indices[index]).second ||
            tensor_values[index].size() != domain64 ||
            // Public tensors are canonically committed without H blinding;
            // private/model tensors are checked for a nonzero blinder by
            // relation_blinding_scalar before this terminal helper is called.
            !zkdl::fr_canonical(tensor_blinders[index]) ||
            !zkdl::cpu_pedersen_strict_g1(canonical_commitments[index], error)) {
            if (error && error->empty()) *error = "invalid canonical terminal tensor input";
            return false;
        }
        for (const auto& value : tensor_values[index])
            if (!zkdl::fr_canonical(value)) {
                if (error) *error = "terminal tensor value is not canonical";
                return false;
            }
    }
    for (const auto& coordinate : evaluation_point)
        if (!zkdl::fr_canonical(coordinate)) {
            if (error) *error = "terminal evaluation point is not canonical";
            return false;
        }
    for (const auto& term : descriptor.terms)
        for (const auto ref : term.refs)
            if (ref >= tensor_values.size()) {
                if (error) *error = "terminal monomial references an unknown tensor";
                return false;
            }
    return true;
}

inline bool zktr_validate_public_inputs(
    const ConstraintDescriptor& descriptor, std::size_t tensor_count,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement, const std::string& operation_context,
    std::string* error = nullptr) {
    if (!descriptor.valid(error) || tensor_count == 0U ||
        tensor_count != canonical_commitments.size() ||
        tensor_count != commitment_indices.size() ||
        tensor_count > zkdl::ZkTerminalRelationProof::kMaxTerminalCommitments ||
        evaluation_point.size() >= 19U || evaluation_point.empty() ||
        operation_context.empty() || !zktr_all_statement_digests_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid terminal relation verifier inputs";
        return false;
    }
    const std::uint64_t domain64 = std::uint64_t(1U) << evaluation_point.size();
    if (domain64 > (1U << 18U) || generators.size() != domain64 ||
        !zkdl::cpu_pedersen_basis_valid(generators, blinding_generator, u_generator,
                                        static_cast<std::uint32_t>(domain64), error))
        return false;
    std::set<std::uint32_t> seen_indices;
    for (std::size_t index = 0U; index < tensor_count; ++index)
        if (commitment_indices[index] >= kZkTerminalEvaluationIndexBase ||
            !seen_indices.insert(commitment_indices[index]).second ||
            !zkdl::cpu_pedersen_strict_g1(canonical_commitments[index], error)) {
            if (error && error->empty()) *error = "invalid canonical terminal commitment map";
            return false;
        }
    for (const auto& coordinate : evaluation_point)
        if (!zkdl::fr_canonical(coordinate)) {
            if (error) *error = "terminal evaluation point is not canonical";
            return false;
        }
    for (const auto& term : descriptor.terms)
        for (const auto ref : term.refs)
            if (ref >= tensor_count) {
                if (error) *error = "terminal monomial references an unknown tensor";
                return false;
            }
    return true;
}

inline bool zktr_build_product_plan(
    const ConstraintDescriptor& descriptor, std::size_t tensor_count,
    const std::vector<std::uint32_t>& commitment_indices,
    std::vector<ZkTerminalProductPlan>& plans,
    std::vector<std::uint32_t>& monomial_commitment_indices,
    std::string* error = nullptr) {
    if (commitment_indices.size() != tensor_count) {
        if (error) *error = "terminal product plan tensor/index shape mismatch";
        return false;
    }
    plans.clear();
    monomial_commitment_indices.clear();
    monomial_commitment_indices.reserve(descriptor.terms.size());
    for (std::size_t monomial = 0U; monomial < descriptor.terms.size(); ++monomial) {
        const auto& refs = descriptor.terms[monomial].refs;
        if (refs.empty()) {
            if (error) *error = "terminal product plan contains an empty monomial";
            return false;
        }
        for (const auto ref : refs)
            if (ref >= tensor_count) {
                if (error) *error = "terminal product plan reference is out of bounds";
                return false;
            }
        if (refs.size() == 1U) {
            monomial_commitment_indices.push_back(
                zktr_terminal_evaluation_index(refs.front()));
            continue;
        }
        std::uint32_t left = zktr_terminal_evaluation_index(refs[0U]);
        for (std::size_t factor = 1U; factor < refs.size(); ++factor) {
            const std::size_t node_index = plans.size();
            if (node_index >= zkdl::ZkTerminalRelationProof::kMaxProductNodes) {
                if (error) *error = "terminal product DAG exceeds its bounded node cap";
                return false;
            }
            const std::uint32_t right = zktr_terminal_evaluation_index(refs[factor]);
            const std::uint32_t output = zktr_product_index(node_index);
            plans.push_back(ZkTerminalProductPlan{
                static_cast<std::uint32_t>(monomial), static_cast<std::uint32_t>(factor),
                left, right, output});
            left = output;
        }
        monomial_commitment_indices.push_back(left);
    }
    return true;
}

inline zkdl::PedersenProofContext zktr_make_link_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const std::array<byte, 32U>& descriptor_digest) {
    const std::uint32_t operation =
        link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality
            ? zkdl::PedersenProofContext::kOperationScalarEquality
            : zkdl::PedersenProofContext::kOperationScalarValue;
    const std::uint32_t related =
        link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality
            ? kZkTerminalExternalIndex
            : zkdl::PedersenProofContext::kUnsetIndex;
    const auto basis_digest = zkdl::cpu_ipa_basis_digest(
        {}, zkdl::CpuG1Point::zero(), zkdl::CpuG1Point::zero(), 0U);
    // The basis is replaced by the caller after construction; this helper is
    // kept parameter-free only for role/index derivation.
    return zksc_make_context(
        operation, 1U, 0U, kZkTerminalResidualIndex, related, basis_digest,
        descriptor_digest, statement);
}

inline zkdl::PedersenProofContext zktr_make_link_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const std::array<byte, 32U>& descriptor_digest,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator) {
    (void)descriptor;
    (void)operation_context;
    auto context = zktr_make_link_context(
        descriptor, statement, operation_context, link_kind, descriptor_digest);
    context.basis_digest = zkdl::cpu_ipa_basis_digest(
        {}, blinding_generator, u_generator, 0U);
    return context;
}

inline zkdl::PedersenProofContext zktr_make_product_context(
    const RelationStatement& statement, const std::array<byte, 32U>& descriptor_digest,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    std::uint32_t left, std::uint32_t right, std::uint32_t output) {
    auto context = zksc_make_context(
        zkdl::PedersenProofContext::kOperationProduct, 2U, 0U, left, right,
        zkdl::cpu_ipa_basis_digest({}, blinding_generator, u_generator, 0U),
        descriptor_digest, statement);
    context.commitment_c_index = output;
    return context;
}

inline zkdl::CpuIpaOpeningMetadata zktr_make_opening_metadata(
    const std::array<byte, 32U>& descriptor_digest,
    const RelationStatement& statement, const std::vector<FrWire>& point,
    std::uint32_t commitment_index, std::uint32_t evaluation_commitment_index) {
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.evaluation_commitment_index = evaluation_commitment_index;
    metadata.outer_point = point;
    metadata.descriptor_digest = descriptor_digest;
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    return metadata;
}

inline void zktr_absorb_header(
    zkdl::Transcript& transcript, const zkdl::ZkTerminalRelationProof& proof,
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::vector<FrWire>& point,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& expected_public_value) {
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, proof.version);
    zkdl::append_string(encoded, operation_context);
    append_constraint(encoded, descriptor);
    zkdl::append_pedersen_context(encoded, proof.context);
    append_digest(encoded, statement.statement_digest);
    append_digest(encoded, statement.model_digest);
    append_digest(encoded, statement.table_digest);
    append_digest(encoded, statement.ledger_digest);
    append_digest(encoded, statement.registry_digest);
    append_digest(encoded, statement.srs_digest);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(point.size()));
    for (const auto& coordinate : point) zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(commitment_indices.size()));
    for (std::size_t index = 0U; index < commitment_indices.size(); ++index) {
        zkdl::append_u32(encoded, commitment_indices[index]);
        zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(canonical_commitments[index]));
    }
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(blinding_generator));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(u_generator));
    zkdl::append_u32(encoded, static_cast<std::uint32_t>(link_kind));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(external_claim_commitment));
    zkdl::append_fr(encoded, zkdl::fr_value(expected_public_value));
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(proof.terminal_commitments.size()));
    for (const auto& commitment : proof.terminal_commitments)
        zkdl::append_g1(encoded, commitment);
    transcript.absorb("gat.zkterminal.header", encoded);
}

inline void zktr_absorb_components(
    zkdl::Transcript& transcript, const zkdl::ZkTerminalRelationProof& proof,
    const std::string& operation_context) {
    for (std::size_t index = 0U; index < proof.terminal_openings.size(); ++index) {
        std::vector<byte> encoded;
        zkdl::serialize_hyrax_opening_v3(proof.terminal_openings[index], encoded);
        transcript.absorb("gat.zkterminal." + operation_context + ".opening." +
                          std::to_string(index), encoded);
    }
    for (std::size_t index = 0U; index < proof.product_nodes.size(); ++index) {
        std::vector<byte> encoded;
        zkdl::serialize_zk_product_node(proof.product_nodes[index], encoded);
        transcript.absorb("gat.zkterminal." + operation_context + ".product." +
                          std::to_string(index), encoded);
    }
    std::vector<byte> link;
    if (proof.claim_link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality)
        zkdl::serialize_pedersen_scalar_equality(proof.external_equality_binding, link);
    else
        zkdl::serialize_pedersen_scalar_value(proof.public_value_binding, link);
    transcript.absorb("gat.zkterminal." + operation_context + ".link", link);
}

inline bool zktr_link_kind_valid(zkdl::ZkTerminalClaimLinkKind link_kind) {
    return link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality ||
           link_kind == zkdl::ZkTerminalClaimLinkKind::PublicValue;
}

inline bool prove_zk_terminal_relation(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<FrWire>& tensor_blinders,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& external_claim_blinder,
    const FrWire& expected_public_value,
    const std::string& operation_context,
    zkdl::Transcript& parent_transcript,
    ZkTerminalRelationProverResult& result,
    std::string* error = nullptr) {
    try {
        if (!zktr_link_kind_valid(link_kind) ||
            !zktr_validate_common_inputs(
                descriptor, tensor_values, canonical_commitments, tensor_blinders,
                commitment_indices, evaluation_point, generators, blinding_generator,
                u_generator, statement, operation_context, error) ||
            !zkdl::fr_canonical(expected_public_value)) {
            if (error && error->empty()) *error = "invalid terminal relation prover inputs";
            return false;
        }
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            if (!zkdl::fr_canonical(external_claim_blinder) ||
                zkdl::fr_compare(external_claim_blinder, FrWire{}) == 0 ||
                !zkdl::cpu_pedersen_strict_g1(external_claim_commitment, error)) {
                if (error && error->empty()) *error = "invalid external terminal claim";
                return false;
            }
        } else if (!external_claim_commitment.is_zero() ||
                   zkdl::fr_compare(external_claim_blinder, FrWire{}) != 0) {
            if (error) *error = "public terminal link carries an external claim witness";
            return false;
        }

        std::vector<ZkTerminalProductPlan> plans;
        std::vector<std::uint32_t> monomial_indices;
        if (!zktr_build_product_plan(descriptor, tensor_values.size(), commitment_indices,
                                     plans, monomial_indices, error)) return false;

        result = ZkTerminalRelationProverResult{};
        auto& proof = result.proof;
        proof.claim_link_kind = link_kind;
        proof.context = zktr_make_link_context(
            descriptor, statement, operation_context, link_kind,
            zksc_proof_descriptor_digest(descriptor, statement, operation_context),
            blinding_generator, u_generator);
        const std::array<byte, 32U> descriptor_digest = proof.context.descriptor_digest;
        const std::vector<FrWire> weights = zkdl::cpu_ipa_mle_weights(evaluation_point);
        result.terminal_values.reserve(tensor_values.size());
        result.terminal_blinders.reserve(tensor_values.size());
        proof.terminal_commitments.reserve(tensor_values.size());
        proof.terminal_openings.reserve(tensor_values.size());

        struct ScalarState {
            FrWire value{};
            FrWire blinder{};
            zkdl::CpuG1Point commitment;
        };
        std::map<std::uint32_t, ScalarState> states;
        for (std::size_t tensor_index = 0U; tensor_index < tensor_values.size(); ++tensor_index) {
            const FrWire value = zkdl::cpu_ipa_inner_product(tensor_values[tensor_index], weights);
            const FrWire tau_blinder = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point terminal_commitment = zkdl::cpu_pedersen_scalar_commit(
                value, tau_blinder, blinding_generator, u_generator);
            if (!zkdl::cpu_pedersen_strict_g1(terminal_commitment, error) ||
                terminal_commitment.is_zero()) {
                if (error && error->empty()) *error = "terminal scalar commitment is invalid";
                return false;
            }
            const std::uint32_t terminal_index = zktr_terminal_evaluation_index(tensor_index);
            const zkdl::CpuIpaOpeningMetadata metadata = zktr_make_opening_metadata(
                descriptor_digest, statement, evaluation_point, commitment_indices[tensor_index],
                terminal_index);
            const auto opening = zkdl::cpu_open_hyrax_v3(
                generators, tensor_values[tensor_index], evaluation_point,
                blinding_generator, u_generator, tensor_blinders[tensor_index],
                tau_blinder, terminal_commitment, metadata);
            proof.terminal_commitments.push_back(zkdl::cpu_g1_to_native(terminal_commitment));
            proof.terminal_openings.push_back(opening.proof);
            result.terminal_values.push_back(value);
            result.terminal_blinders.push_back(tau_blinder);
            states.emplace(terminal_index, ScalarState{value, tau_blinder, terminal_commitment});
        }

        result.product_blinders.reserve(plans.size());
        for (const auto& plan : plans) {
            const auto left = states.find(plan.left_commitment_index);
            const auto right = states.find(plan.right_commitment_index);
            if (left == states.end() || right == states.end()) {
                if (error) *error = "terminal product plan references an unavailable claim";
                return false;
            }
            const FrWire output_value = zkdl::fr_mul(left->second.value, right->second.value);
            const FrWire output_blinder = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point output_commitment = zkdl::cpu_pedersen_scalar_commit(
                output_value, output_blinder, blinding_generator, u_generator);
            const auto product_context = zktr_make_product_context(
                statement, descriptor_digest, blinding_generator, u_generator,
                plan.left_commitment_index, plan.right_commitment_index,
                plan.output_commitment_index);
            const auto product = zkdl::cpu_open_pedersen_product(
                left->second.value, right->second.value, left->second.blinder,
                right->second.blinder, output_blinder, left->second.commitment,
                right->second.commitment, output_commitment, blinding_generator,
                u_generator, product_context);
            zkdl::ZkProductNodeProof node;
            node.monomial_index = plan.monomial_index;
            node.factor_step = plan.factor_step;
            node.left_commitment_index = plan.left_commitment_index;
            node.right_commitment_index = plan.right_commitment_index;
            node.output_commitment_index = plan.output_commitment_index;
            node.output_commitment = zkdl::cpu_g1_to_native(output_commitment);
            node.product_proof = product.proof;
            proof.product_nodes.push_back(node);
            result.product_blinders.push_back(output_blinder);
            states.emplace(plan.output_commitment_index,
                           ScalarState{output_value, output_blinder, output_commitment});
        }

        FrWire residual_value = descriptor.constant;
        FrWire residual_blinder{};
        zkdl::CpuG1Point residual_commitment = zkdl::cpu_g1_mul(
            u_generator, descriptor.constant);
        for (std::size_t monomial = 0U; monomial < descriptor.terms.size(); ++monomial) {
            const auto state = states.find(monomial_indices[monomial]);
            if (state == states.end()) {
                if (error) *error = "terminal monomial claim is unavailable";
                return false;
            }
            const FrWire coefficient = descriptor.terms[monomial].coefficient;
            residual_value = zkdl::fr_add(
                residual_value, zkdl::fr_mul(coefficient, state->second.value));
            residual_blinder = zkdl::fr_add(
                residual_blinder, zkdl::fr_mul(coefficient, state->second.blinder));
            residual_commitment = zkdl::cpu_g1_add(
                residual_commitment,
                zkdl::cpu_g1_mul(state->second.commitment, coefficient));
        }
        if (zkdl::fr_compare(residual_blinder, FrWire{}) == 0 ||
            !zkdl::fr_canonical(residual_blinder) ||
            !zkdl::cpu_g1_equal(
                residual_commitment,
                zkdl::cpu_pedersen_scalar_commit(
                    residual_value, residual_blinder, blinding_generator, u_generator))) {
            if (error) *error = "BLOCKED: terminal residual blinder/commitment is invalid";
            return false;
        }
        proof.residual_commitment = zkdl::cpu_g1_to_native(residual_commitment);
        result.residual_blinder = residual_blinder;
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            const zkdl::CpuG1Point expected_external = zkdl::cpu_pedersen_scalar_commit(
                residual_value, external_claim_blinder, blinding_generator, u_generator);
            if (!zkdl::cpu_g1_equal(expected_external, external_claim_commitment)) {
                if (error) *error = "external terminal claim commitment mismatch";
                return false;
            }
            proof.external_claim_commitment =
                zkdl::cpu_g1_to_native(external_claim_commitment);
            proof.external_equality_binding = zkdl::cpu_open_pedersen_scalar_equality(
                residual_value, residual_blinder, external_claim_blinder,
                residual_commitment, external_claim_commitment, blinding_generator,
                u_generator, proof.context).proof;
        } else {
            if (!same_wire(residual_value, expected_public_value)) {
                if (error) *error = "terminal residual does not match public value";
                return false;
            }
            proof.external_claim_commitment = proof.residual_commitment;
            proof.public_value_binding = zkdl::cpu_open_pedersen_scalar_value_public(
                residual_value, residual_blinder, residual_commitment,
                blinding_generator, u_generator, proof.context).proof;
        }
        std::string shape_error;
        if (!zkdl::zk_terminal_relation_shape_valid(proof, &shape_error)) {
            if (error) *error = shape_error.empty() ? "invalid terminal relation proof" : shape_error;
            return false;
        }
        zktr_absorb_header(parent_transcript, proof, descriptor, statement, operation_context,
                           evaluation_point, commitment_indices, canonical_commitments,
                           blinding_generator, u_generator, link_kind,
                           external_claim_commitment, expected_public_value);
        zktr_absorb_components(parent_transcript, proof, operation_context);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool verify_zk_terminal_relation(
    const zkdl::ZkTerminalRelationProof& proof,
    const ConstraintDescriptor& descriptor,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& expected_public_value,
    const std::string& operation_context,
    zkdl::Transcript& parent_transcript,
    ZkTerminalRelationVerifierResult& result,
    std::string* error = nullptr) {
    try {
        if (!zktr_link_kind_valid(link_kind) ||
            !zktr_validate_public_inputs(
                descriptor, canonical_commitments.size(), canonical_commitments,
                commitment_indices, evaluation_point, generators, blinding_generator,
                u_generator, statement, operation_context, error) ||
            !zkdl::fr_canonical(expected_public_value) ||
            proof.claim_link_kind != link_kind ||
            !zkdl::zk_terminal_relation_shape_valid(proof, error)) {
            if (error && error->empty()) *error = "invalid terminal relation verifier inputs";
            return false;
        }
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            if (!zkdl::cpu_pedersen_strict_g1(external_claim_commitment, error) ||
                external_claim_commitment.is_zero() ||
                !zkdl::cpu_g1_equal(
                    external_claim_commitment,
                    zkdl::cpu_g1_from_native(proof.external_claim_commitment))) {
                if (error && error->empty()) *error = "external terminal claim mismatch";
                return false;
            }
        } else if (!external_claim_commitment.is_zero()) {
            if (error) *error = "public terminal verifier carries an external claim";
            return false;
        }

        const std::array<byte, 32U> descriptor_digest =
            zksc_proof_descriptor_digest(descriptor, statement, operation_context);
        const auto expected_context = zktr_make_link_context(
            descriptor, statement, operation_context, link_kind, descriptor_digest,
            blinding_generator, u_generator);
        if (!zkdl::cpu_pedersen_context_equal(proof.context, expected_context) ||
            proof.context.descriptor_digest != descriptor_digest ||
            proof.context.registry_digest != statement.registry_digest ||
            proof.context.srs_digest != statement.srs_digest) {
            if (error) *error = "terminal relation link context mismatch";
            return false;
        }
        if (proof.terminal_commitments.size() != canonical_commitments.size() ||
            proof.terminal_openings.size() != canonical_commitments.size()) {
            if (error) *error = "terminal relation commitment/opening count mismatch";
            return false;
        }
        zktr_absorb_header(parent_transcript, proof, descriptor, statement, operation_context,
                           evaluation_point, commitment_indices, canonical_commitments,
                           blinding_generator, u_generator, link_kind,
                           external_claim_commitment, expected_public_value);

        std::map<std::uint32_t, zkdl::CpuG1Point> states;
        result = ZkTerminalRelationVerifierResult{};
        result.terminal_commitments.reserve(proof.terminal_commitments.size());
        for (std::size_t tensor_index = 0U;
             tensor_index < proof.terminal_commitments.size(); ++tensor_index) {
            const zkdl::CpuG1Point terminal_commitment = zkdl::cpu_g1_from_native(
                proof.terminal_commitments[tensor_index]);
            const auto& opening = proof.terminal_openings[tensor_index];
            const std::uint32_t terminal_index =
                zktr_terminal_evaluation_index(tensor_index);
            const auto metadata = zktr_make_opening_metadata(
                descriptor_digest, statement, evaluation_point,
                commitment_indices[tensor_index], terminal_index);
            if (opening.commitment_index != commitment_indices[tensor_index] ||
                opening.evaluation_commitment_index != terminal_index ||
                opening.generator_size != generators.size() ||
                opening.inner_point.size() != evaluation_point.size() ||
                opening.outer_point.size() != evaluation_point.size() ||
                !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(opening.xi),
                                    canonical_commitments[tensor_index]) ||
                !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(opening.tau),
                                    terminal_commitment)) {
                if (error) *error = "terminal Hyrax commitment/index map mismatch";
                return false;
            }
            for (std::size_t bit = 0U; bit < evaluation_point.size(); ++bit) {
                if (fr_compare_mixed(opening.inner_point[bit], evaluation_point[bit]) != 0 ||
                    fr_compare_mixed(opening.outer_point[bit], evaluation_point[bit]) != 0) {
                    if (error) *error = "terminal Hyrax evaluation point mismatch";
                    return false;
                }
            }
            if (!zkdl::cpu_verify_hyrax_v3(
                    opening, canonical_commitments[tensor_index], terminal_commitment,
                    generators, evaluation_point, blinding_generator, u_generator,
                    metadata, error)) return false;
            if (!states.emplace(terminal_index, terminal_commitment).second) {
                if (error) *error = "duplicate terminal evaluation commitment index";
                return false;
            }
            result.terminal_commitments.push_back(terminal_commitment);
        }

        std::vector<ZkTerminalProductPlan> plans;
        std::vector<std::uint32_t> monomial_indices;
        if (!zktr_build_product_plan(descriptor, canonical_commitments.size(),
                                     commitment_indices, plans, monomial_indices, error))
            return false;
        if (plans.size() != proof.product_nodes.size()) {
            if (error) *error = "terminal product DAG node count mismatch";
            return false;
        }
        for (std::size_t node_index = 0U; node_index < plans.size(); ++node_index) {
            const auto& plan = plans[node_index];
            const auto& node = proof.product_nodes[node_index];
            if (node.monomial_index != plan.monomial_index ||
                node.factor_step != plan.factor_step ||
                node.left_commitment_index != plan.left_commitment_index ||
                node.right_commitment_index != plan.right_commitment_index ||
                node.output_commitment_index != plan.output_commitment_index) {
                if (error) *error = "terminal product DAG order/ref mismatch";
                return false;
            }
            const auto left = states.find(plan.left_commitment_index);
            const auto right = states.find(plan.right_commitment_index);
            if (left == states.end() || right == states.end()) {
                if (error) *error = "terminal product DAG references an unknown claim";
                return false;
            }
            const zkdl::CpuG1Point output = zkdl::cpu_g1_from_native(node.output_commitment);
            const auto product_context = zktr_make_product_context(
                statement, descriptor_digest, blinding_generator, u_generator,
                plan.left_commitment_index, plan.right_commitment_index,
                plan.output_commitment_index);
            if (!zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(node.product_proof.commitment_a), left->second) ||
                !zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(node.product_proof.commitment_b), right->second) ||
                !zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(node.product_proof.commitment_c), output) ||
                !zkdl::cpu_verify_pedersen_product(
                    node.product_proof, left->second, right->second, output,
                    blinding_generator, u_generator, product_context, error))
                return false;
            if (!states.emplace(plan.output_commitment_index, output).second) {
                if (error) *error = "duplicate terminal product output index";
                return false;
            }
        }

        zkdl::CpuG1Point residual_commitment = zkdl::cpu_g1_mul(
            u_generator, descriptor.constant);
        for (std::size_t monomial = 0U; monomial < descriptor.terms.size(); ++monomial) {
            const auto state = states.find(monomial_indices[monomial]);
            if (state == states.end()) {
                if (error) *error = "terminal residual references an unknown monomial claim";
                return false;
            }
            residual_commitment = zkdl::cpu_g1_add(
                residual_commitment,
                zkdl::cpu_g1_mul(state->second, descriptor.terms[monomial].coefficient));
        }
        if (!zkdl::cpu_g1_equal(
                residual_commitment,
                zkdl::cpu_g1_from_native(proof.residual_commitment))) {
            if (error) *error = "terminal residual commitment equation mismatch";
            return false;
        }
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            if (!zkdl::cpu_verify_pedersen_scalar_equality(
                    proof.external_equality_binding, residual_commitment,
                    external_claim_commitment, blinding_generator, u_generator,
                    proof.context, error)) return false;
        } else if (!zkdl::cpu_verify_pedersen_scalar_value(
                       proof.public_value_binding, expected_public_value,
                       residual_commitment, blinding_generator, u_generator,
                       proof.context, error)) {
            return false;
        }
        zktr_absorb_components(parent_transcript, proof, operation_context);
        result.residual_commitment = residual_commitment;
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
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

inline bool verify_constraint(const ConstraintProof& proof,
                             const ConstraintDescriptor& expected,
                             const std::vector<std::string>& expected_tensor_refs,
                             const std::vector<CommitmentWire>& commitments,
                             const std::vector<zkdl::CpuG1Point>& generators,
                             const zkdl::CpuG1Point& ipa_blinding_generator,
                             const zkdl::CpuG1Point& ipa_u_generator,
                             const RelationStatement& statement,
                             zkdl::Transcript& transcript,
                             std::string* error = nullptr) {
    std::vector<byte> expected_bytes, actual_bytes;
    append_constraint(expected_bytes, expected);
    append_constraint(actual_bytes, proof.descriptor);
    if (expected_bytes != actual_bytes) {
        if (error) *error = "relation constraint descriptor mismatch";
        return false;
    }
    std::vector<std::uint32_t> expected_indices(expected_tensor_refs.size());
    for (std::size_t index = 0U; index < expected_indices.size(); ++index)
        expected_indices[index] = static_cast<std::uint32_t>(index);
    if (expected_indices.empty() || proof.opening_tensor_indices != expected_indices ||
        proof.opening_commitment_indices.size() != expected_indices.size()) {
        if (error) *error = "relation constraint terminal tensor map is not canonical";
        return false;
    }
    std::uint32_t domain = 0U;
    std::vector<zkdl::CpuG1Point> canonical_commitments;
    canonical_commitments.reserve(expected_indices.size());
    for (std::size_t i = 0U; i < expected_indices.size(); ++i) {
        const std::uint32_t commitment_index = proof.opening_commitment_indices[i];
        if (commitment_index >= commitments.size() ||
            commitments[commitment_index].tensor_id != expected_tensor_refs[i]) {
            if (error) *error = "relation terminal commitment reference mismatch";
            return false;
        }
        if (domain == 0U) domain = commitments[commitment_index].domain_size;
        if (commitments[commitment_index].domain_size != domain || domain == 0U ||
            log2_exact(domain) == std::numeric_limits<std::uint32_t>::max()) {
            if (error) *error = "relation terminal tensors do not share an authenticated MLE domain";
            return false;
        }
        canonical_commitments.push_back(
            zkdl::cpu_g1_from_native(commitments[commitment_index].commitment));
    }
    if (domain > (std::uint32_t(1U) << kMaxRelationPointBits) ||
        generators.size() < domain) {
        if (error) *error = "relation verifier has no authenticated terminal generator domain";
        return false;
    }
    const std::vector<zkdl::CpuG1Point> terminal_generators(
        generators.begin(), generators.begin() + domain);
    const std::string operation_context = "gat.relation." + expected.id;
    ZkSumcheckVerifierResult sumcheck_result;
    if (!verify_zk_sumcheck(proof.zk_sumcheck, expected, expected.degree, domain,
                            terminal_generators, ipa_blinding_generator,
                            ipa_u_generator, FrWire{}, statement, operation_context,
                            transcript, sumcheck_result, error)) return false;
    if (sumcheck_result.challenges.size() != log2_exact(domain)) {
        if (error) *error = "ZkSumcheck terminal point/domain mismatch";
        return false;
    }
    std::vector<FrWire> terminal_point;
    terminal_point.reserve(sumcheck_result.challenges.size());
    for (const auto& challenge : sumcheck_result.challenges)
        terminal_point.push_back(zkdl::fr_wire(challenge));
    std::reverse(terminal_point.begin(), terminal_point.end());
    ZkTerminalRelationVerifierResult terminal_result;
    if (!verify_zk_terminal_relation(
            proof.terminal_relation, expected, canonical_commitments,
            proof.opening_commitment_indices, terminal_point, terminal_generators,
            ipa_blinding_generator, ipa_u_generator, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            zkdl::cpu_g1_from_native(sumcheck_result.final_claim_commitment),
            FrWire{}, operation_context, transcript, terminal_result, error)) return false;
    return true;
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
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, ConstraintProof& proof,
    std::string* error = nullptr) {
    if (operation_refs.empty()) {
        if (error) *error = "relation batched proof has no operation tensor references";
        return false;
    }
    std::vector<std::vector<FrWire>> tensor_values;
    std::vector<zkdl::CpuG1Point> canonical_commitments;
    std::vector<FrWire> tensor_blinders;
    std::vector<std::uint32_t> commitment_indices;
    tensor_values.reserve(operation_refs.size());
    canonical_commitments.reserve(operation_refs.size());
    tensor_blinders.reserve(operation_refs.size());
    commitment_indices.reserve(operation_refs.size());
    std::uint32_t domain = 0U;
    for (const auto& ref : operation_refs) {
        const auto found = tensor_map.find(ref);
        if (found == tensor_map.end() || found->second == nullptr) {
            if (error) *error = "relation batched proof tensor reference is missing: " + ref;
            return false;
        }
        const TensorWitness& tensor = *found->second;
        if (domain == 0U) domain = tensor.domain_size;
        if (tensor.domain_size != domain || domain == 0U ||
            (domain & (domain - 1U)) != 0U || generators.size() < domain) {
            if (error) *error = "relation proof tensor references do not share an authenticated MLE domain";
            return false;
        }
        const auto commitment = std::find_if(
            commitments.begin(), commitments.end(),
            [&](const CommitmentWire& value) { return value.tensor_id == ref; });
        if (commitment == commitments.end() || commitment->domain_size != domain) {
            if (error) *error = "relation proof commitment reference is missing or has the wrong domain: " + ref;
            return false;
        }
        tensor_values.push_back(tensor.values);
        canonical_commitments.push_back(zkdl::cpu_g1_from_native(commitment->commitment));
        tensor_blinders.push_back(relation_blinding_scalar(tensor, blinders));
        commitment_indices.push_back(static_cast<std::uint32_t>(
            std::distance(commitments.begin(), commitment)));
    }
    try {
        if (domain == 0U || log2_exact(domain) == std::numeric_limits<std::uint32_t>::max() ||
            log2_exact(domain) > kMaxRelationPointBits) {
            if (error) *error = "relation proof domain exceeds the native MLE point cap";
            return false;
        }
        proof = ConstraintProof{};
        proof.descriptor = descriptor;
        proof.opening_tensor_indices.resize(operation_refs.size());
        std::iota(proof.opening_tensor_indices.begin(), proof.opening_tensor_indices.end(), 0U);
        proof.opening_commitment_indices = commitment_indices;
        const std::vector<zkdl::CpuG1Point> terminal_generators(
            generators.begin(), generators.begin() + domain);
        ZkSumcheckProverResult sumcheck_result;
        const std::string operation_context = "gat.relation." + descriptor.id;
        if (!prove_zk_sumcheck(
                descriptor, tensor_values, descriptor.degree, domain, terminal_generators,
                ipa_blinding_generator, ipa_u_generator, FrWire{}, statement,
                operation_context, transcript, sumcheck_result, error)) return false;
        proof.zk_sumcheck = std::move(sumcheck_result.proof);
        std::vector<FrWire> terminal_point;
        terminal_point.reserve(sumcheck_result.challenges.size());
        for (const auto& challenge : sumcheck_result.challenges)
            terminal_point.push_back(zkdl::fr_wire(challenge));
        std::reverse(terminal_point.begin(), terminal_point.end());
        ZkTerminalRelationProverResult terminal_result;
        if (!prove_zk_terminal_relation(
                descriptor, tensor_values, canonical_commitments, tensor_blinders,
                commitment_indices, terminal_point, terminal_generators,
                ipa_blinding_generator, ipa_u_generator, statement,
                zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
                zkdl::cpu_g1_from_native(sumcheck_result.proof.final_claim_commitment),
                sumcheck_result.final_q_blinder, FrWire{}, operation_context,
                transcript, terminal_result, error)) return false;
        proof.terminal_relation = std::move(terminal_result.proof);
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
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, SparseRouteProof& route,
    std::string* error, SparseRouteProverSecrets* secrets = nullptr);

inline bool verify_sparse_route(
    const OperationDescriptor& descriptor,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, const SparseRouteProof& route,
    std::string* error);

inline OperationDescriptor fac_line8_selector_descriptor(
    const OperationDescriptor& operation, const FacBindingDescriptor& binding,
    std::size_t index) {
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
    return selector;
}

inline bool fac_line8_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(
        commitments.begin(), commitments.end(), [&](const CommitmentWire& commitment) {
            return commitment.tensor_id == tensor_id;
        });
    if (found == commitments.end() || found->domain_size == 0U ||
        found->domain_size > (std::uint32_t(1U) << kMaxRelationPointBits)) {
        if (error) *error = "FAC4DNN canonical commitment is missing or out of bounds: " + tensor_id;
        return false;
    }
    index = static_cast<std::uint32_t>(std::distance(commitments.begin(), found));
    return true;
}

inline bool fac_line8_absorb_header(
    const OperationDescriptor& operation, const RelationStatement& statement,
    const std::vector<CommitmentWire>& commitments,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator, zkdl::Transcript& transcript,
    std::array<byte, 32U>& digest, std::string* error) {
    if (!operation.valid(error) || operation.fac_bindings.empty() ||
        operation.fac_bindings.size() > kMaxRelationOpenings ||
        !zktr_all_statement_digests_valid(statement, error)) return false;
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, FacLine8Proof::kVersion);
    append_statement(encoded, statement);
    append_operation(encoded, operation);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(operation.fac_bindings.size()));
    for (const auto& binding : operation.fac_bindings) {
        std::uint32_t canonical_index = 0U, view_index = 0U;
        if (!fac_line8_commitment_index(
                commitments, binding.canonical_ref, canonical_index, error) ||
            !fac_line8_commitment_index(
                commitments, binding.view_ref, view_index, error)) return false;
        zkdl::append_string(encoded, binding.canonical_ref);
        zkdl::append_string(encoded, binding.view_ref);
        zkdl::append_u32(encoded, canonical_index);
        zkdl::append_u32(encoded, view_index);
        append_commitment(encoded, commitments[canonical_index]);
        append_commitment(encoded, commitments[view_index]);
    }
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(ipa_blinding_generator));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(ipa_u_generator));
    transcript.absorb("gat.fac4dnn.line8.header", encoded);
    digest = zkdl::sha256(encoded);
    return digest != std::array<byte, 32U>{};
}

inline std::vector<FrWire> fac_line8_rho_challenges(
    const OperationDescriptor& operation, zkdl::Transcript& transcript) {
    std::vector<FrWire> rho;
    rho.reserve(operation.fac_bindings.size());
    for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index)
        rho.push_back(zkdl::cpu_ipa_nonzero_challenge(
            transcript, "gat.fac4dnn.line8." + operation.id + ".rho." +
                            std::to_string(index)));
    return rho;
}

inline std::array<byte, 32U> fac_line8_aggregate_digest(
    const std::array<byte, 32U>& header_digest,
    const std::vector<FrWire>& rho,
    const std::vector<SparseRouteProof>& routes) {
    std::vector<byte> encoded(header_digest.begin(), header_digest.end());
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(routes.size()));
    for (std::size_t index = 0U; index < routes.size(); ++index) {
        zkdl::append_fr(encoded, zkdl::fr_value(rho[index]));
        const std::vector<byte> route_bytes = routes[index].serialize();
        append_digest(encoded, zkdl::sha256(route_bytes));
        zkdl::append_g1(encoded, routes[index].terminal_binding.commitment_1);
        zkdl::append_g1(encoded, routes[index].terminal_binding.commitment_2);
        zkdl::append_g1(encoded, routes[index].sumcheck.final_claim_commitment);
    }
    return zkdl::sha256(encoded);
}

inline zkdl::PedersenProofContext fac_line8_aggregate_context(
    const std::array<byte, 32U>& digest, const RelationStatement& statement,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator) {
    constexpr std::uint32_t kAggregateResidualIndex = 0xF0000100U;
    constexpr std::uint32_t kAggregateSumcheckIndex = 0xF0000101U;
    return zksc_make_context(
        zkdl::PedersenProofContext::kOperationScalarEquality, 1U, 0U,
        kAggregateResidualIndex, kAggregateSumcheckIndex,
        zkdl::cpu_ipa_basis_digest({}, ipa_blinding_generator,
                                    ipa_u_generator, 0U),
        digest, statement);
}

inline bool fac_line8_aggregate_commitments(
    const std::vector<SparseRouteProof>& routes, const std::vector<FrWire>& rho,
    zkdl::CpuG1Point& residual_commitment,
    zkdl::CpuG1Point& sumcheck_commitment, std::string* error) {
    if (routes.empty() || routes.size() != rho.size()) {
        if (error) *error = "FAC4DNN aggregate terminal shape mismatch";
        return false;
    }
    residual_commitment = zkdl::CpuG1Point{};
    sumcheck_commitment = zkdl::CpuG1Point{};
    for (std::size_t index = 0U; index < routes.size(); ++index) {
        std::string point_error;
        if (!zkdl::pedersen_wire_g1_valid(
                routes[index].terminal_binding.commitment_1, &point_error) ||
            !zkdl::pedersen_wire_g1_valid(
                routes[index].terminal_binding.commitment_2, &point_error)) {
            if (error) *error = point_error.empty()
                ? "FAC4DNN aggregate contains an invalid terminal commitment"
                : point_error;
            return false;
        }
        residual_commitment = zkdl::cpu_g1_add(
            residual_commitment,
            zkdl::cpu_g1_mul(
                zkdl::cpu_g1_from_native(
                    routes[index].terminal_binding.commitment_1), rho[index]));
        sumcheck_commitment = zkdl::cpu_g1_add(
            sumcheck_commitment,
            zkdl::cpu_g1_mul(
                zkdl::cpu_g1_from_native(
                    routes[index].terminal_binding.commitment_2), rho[index]));
    }
    return true;
}

inline bool prove_fac_line8(
    const OperationDescriptor& operation,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, FacLine8Proof& proof,
    std::string* error = nullptr) {
    try {
        proof = FacLine8Proof{};
        std::array<byte, 32U> header_digest{};
        if (!fac_line8_absorb_header(
                operation, statement, commitments, ipa_blinding_generator,
                ipa_u_generator, transcript, header_digest, error)) return false;
        proof.selector_proofs.reserve(operation.fac_bindings.size());
        std::vector<SparseRouteProverSecrets> terminal_secrets;
        terminal_secrets.reserve(operation.fac_bindings.size());
        for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index) {
            const OperationDescriptor selector = fac_line8_selector_descriptor(
                operation, operation.fac_bindings[index], index);
            SparseRouteProof route;
            SparseRouteProverSecrets secrets;
            if (!prove_sparse_route(
                    selector, tensor_map, blinders, generators,
                    ipa_blinding_generator, ipa_u_generator, commitments,
                    statement, transcript, route, error, &secrets)) return false;
            const std::vector<byte> encoded = route.serialize();
            transcript.absorb(
                "gat.fac4dnn.line8." + operation.id + ".selector." +
                    std::to_string(index), encoded);
            proof.selector_proofs.push_back(std::move(route));
            terminal_secrets.push_back(secrets);
        }
        const std::vector<FrWire> rho = fac_line8_rho_challenges(operation, transcript);
        zkdl::CpuG1Point residual_commitment, sumcheck_commitment;
        if (!fac_line8_aggregate_commitments(
                proof.selector_proofs, rho, residual_commitment,
                sumcheck_commitment, error)) return false;
        FrWire aggregate_value{};
        FrWire residual_blinder{};
        FrWire sumcheck_blinder{};
        for (std::size_t index = 0U; index < terminal_secrets.size(); ++index) {
            aggregate_value = zkdl::fr_add(
                aggregate_value,
                zkdl::fr_mul(rho[index], terminal_secrets[index].terminal_value));
            residual_blinder = zkdl::fr_add(
                residual_blinder,
                zkdl::fr_mul(rho[index], terminal_secrets[index].residual_blinder));
            sumcheck_blinder = zkdl::fr_add(
                sumcheck_blinder,
                zkdl::fr_mul(rho[index], terminal_secrets[index].sumcheck_blinder));
        }
        if (residual_blinder == FrWire{} || sumcheck_blinder == FrWire{}) {
            if (error) *error =
                "FAC4DNN aggregate blinder is zero; retry with fresh proof randomness";
            return false;
        }
        const auto aggregate_digest = fac_line8_aggregate_digest(
            header_digest, rho, proof.selector_proofs);
        proof.aggregate_binding = zkdl::cpu_open_pedersen_scalar_equality(
            aggregate_value, residual_blinder, sumcheck_blinder,
            residual_commitment, sumcheck_commitment,
            ipa_blinding_generator, ipa_u_generator,
            fac_line8_aggregate_context(
                aggregate_digest, statement, ipa_blinding_generator,
                ipa_u_generator)).proof;
        std::vector<byte> aggregate_bytes;
        zkdl::serialize_pedersen_scalar_equality(
            proof.aggregate_binding, aggregate_bytes);
        transcript.absorb(
            "gat.fac4dnn.line8." + operation.id + ".aggregate",
            aggregate_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool verify_fac_line8(
    const OperationDescriptor& operation,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, const FacLine8Proof& proof,
    std::string* error = nullptr) {
    try {
        if (proof.version != FacLine8Proof::kVersion ||
            operation.fac_bindings.empty() ||
            proof.selector_proofs.size() != operation.fac_bindings.size()) {
            if (error) *error = "FAC4DNN selector proof count/version mismatch";
            return false;
        }
        std::array<byte, 32U> header_digest{};
        if (!fac_line8_absorb_header(
                operation, statement, commitments, ipa_blinding_generator,
                ipa_u_generator, transcript, header_digest, error)) return false;
        for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index) {
            const OperationDescriptor selector = fac_line8_selector_descriptor(
                operation, operation.fac_bindings[index], index);
            if (!verify_sparse_route(
                    selector, commitments, generators, ipa_blinding_generator,
                    ipa_u_generator, statement, transcript,
                    proof.selector_proofs[index], error)) return false;
            const std::vector<byte> encoded = proof.selector_proofs[index].serialize();
            transcript.absorb(
                "gat.fac4dnn.line8." + operation.id + ".selector." +
                    std::to_string(index), encoded);
        }
        const std::vector<FrWire> rho = fac_line8_rho_challenges(operation, transcript);
        zkdl::CpuG1Point residual_commitment, sumcheck_commitment;
        if (!fac_line8_aggregate_commitments(
                proof.selector_proofs, rho, residual_commitment,
                sumcheck_commitment, error)) return false;
        const auto aggregate_digest = fac_line8_aggregate_digest(
            header_digest, rho, proof.selector_proofs);
        if (!zkdl::cpu_verify_pedersen_scalar_equality(
                proof.aggregate_binding, residual_commitment,
                sumcheck_commitment, ipa_blinding_generator, ipa_u_generator,
                fac_line8_aggregate_context(
                    aggregate_digest, statement, ipa_blinding_generator,
                    ipa_u_generator), error)) return false;
        std::vector<byte> aggregate_bytes;
        zkdl::serialize_pedersen_scalar_equality(
            proof.aggregate_binding, aggregate_bytes);
        transcript.absorb(
            "gat.fac4dnn.line8." + operation.id + ".aggregate",
            aggregate_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
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

/* FAC4DNN Line8 has been migrated to hidden proof objects, but its canonical
 * fold-point/Hyrax evaluation identity has not yet passed the bounded native
 * roundtrip gate.  Keep it fail-closed rather than admitting an unvalidated
 * view binding.  The official zkMatMul primitive binding remains independently
 * blocked because it still carries historical value-revealing openings. */
inline bool formal_relation_components_supported(
    const std::vector<OperationDescriptor>& operations,
    std::string* error = nullptr) {
    for (const auto& operation : operations) {
        if (!operation.fac_bindings.empty()) {
            if (error) *error =
                "BLOCKED: formal zkDL FAC4DNN Line8 canonical MLE/opening identity is not validated";
            return false;
        }
        if (!operation.primitives.empty()) {
            if (error) *error =
                "BLOCKED: formal zkDL GAT relation still has value-revealing zkMatMul openings";
            return false;
        }
    }
    return true;
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
            !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(record.commitment),
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
        if (found->domain_size > context.ipa_generators.size()) {
            if (error) *error = "public tensor commitment exceeds authenticated generator domain";
            return false;
        }
        const std::vector<zkdl::CpuG1Point> generators(
            context.ipa_generators.begin(), context.ipa_generators.begin() + found->domain_size);
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
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
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
            a, b, c, shape, gen_a, gen_b, gen_c,
            nullptr, ipa_blinding_generator, ipa_u_generator);
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
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator, bool gpu_backend,
    std::string* error) {
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
    if (gpu_backend) {
        if (proof.commitment_a.size() != shape.num * shape.m ||
            proof.commitment_b.size() != shape.num * shape.n ||
            proof.commitment_c.size() != shape.num * shape.m) {
            if (error) *error = "zkMatMul GPU commitment tensor shape mismatch";
            return false;
        }
        std::vector<G1Jacobian_t> native_generators_a(
            generators.size() >= shape.n ? shape.n : 0U);
        std::vector<G1Jacobian_t> native_generators_b(
            generators.size() >= shape.k ? shape.k : 0U);
        for (std::uint32_t index = 0U; index < shape.n; ++index)
            native_generators_a[index] = zkdl::cpu_g1_to_native(generators[index]);
        for (std::uint32_t index = 0U; index < shape.k; ++index)
            native_generators_b[index] = zkdl::cpu_g1_to_native(generators[index]);
        Commitment gen_a(shape.n, native_generators_a.data());
        Commitment gen_b(shape.k, native_generators_b.data());
        G1TensorJacobian com_a(static_cast<std::uint32_t>(proof.commitment_a.size()),
                               proof.commitment_a.data());
        G1TensorJacobian com_b(static_cast<std::uint32_t>(proof.commitment_b.size()),
                               proof.commitment_b.data());
        G1TensorJacobian com_c(static_cast<std::uint32_t>(proof.commitment_c.size()),
                               proof.commitment_c.data());
        return zkdl::matmul_verify_gpu(proof, shape, gen_a, com_a, gen_b, com_b,
                                       &gen_b, &com_c, nullptr, error);
    }
    return zkdl::matmul_verify_cpu(proof, shape, a_view, b_view, &c_view, nullptr,
                                   ipa_blinding_generator, ipa_u_generator, error);
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
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
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
    auto metadata_for = [&](const std::string& suffix, std::uint32_t index) {
        OperationDescriptor opening_descriptor;
        opening_descriptor.id = descriptor.tensor_refs[0] + "/" + suffix;
        if (suffix == "B") opening_descriptor.id = descriptor.tensor_refs[1] + "/B";
        if (suffix == "C") opening_descriptor.id = descriptor.tensor_refs[2] + "/C";
        return relation_opening_metadata(opening_descriptor, index, {}, statement);
    };
    binding.opening_a = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + a_domain),
        a->second->values, point_a, ipa_blinding_generator, ipa_u_generator,
        relation_blinding_scalar(*a->second, blinders), metadata_for("A", binding.a_commitment_index));
    binding.opening_b = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + b_domain),
        b->second->values, point_b, ipa_blinding_generator, ipa_u_generator,
        relation_blinding_scalar(*b->second, blinders), metadata_for("B", binding.b_commitment_index));
    binding.opening_c = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + c_domain),
        c->second->values, point_c, ipa_blinding_generator, ipa_u_generator,
        relation_blinding_scalar(*c->second, blinders), metadata_for("C", binding.c_commitment_index));
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
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement, std::string* error) {
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
    auto metadata_for = [&](std::size_t slot) {
        OperationDescriptor opening_descriptor;
        opening_descriptor.id = descriptor.tensor_refs[slot] + "/" +
                                (slot == 0U ? "A" : (slot == 1U ? "B" : "C"));
        return relation_opening_metadata(opening_descriptor, indices[slot], {}, statement);
    };
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
                                      std::vector<zkdl::CpuG1Point>(generators.begin(),
                                                                    generators.begin() + domains[i]),
                                      ipa_blinding_generator, ipa_u_generator,
                                      metadata_for(i), error) ||
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

#include "gat_adapter/gat_sparse_route.cuh"

inline bool prove(const ProverContext& context, Proof& proof, std::string* error = nullptr) {
    try {
        if (!descriptor_vector_valid(context.operations, error)) return false;
        if (!formal_relation_components_supported(context.operations, error)) return false;
        if (context.gpu_backend) {
            if (error) *error =
                "BLOCKED: formal zkDL GPU GAT proving is unavailable until every relation family "
                "and terminal opening runs on native CUDA cryptographic kernels";
            return false;
        }
        if (context.tensors.empty() || context.generators.empty() ||
            context.ipa_generators.empty() || context.ipa_blinding_generator.is_zero() ||
            context.ipa_u_generator.is_zero() ||
            !same_g1_basis(context.generators, context.ipa_generators)) {
            if (error) *error = "BLOCKED: relation prover requires authenticated independent IPA G/H/U basis";
            return false;
        }
        if (!context.hiding_commitment_records.empty()) {
            if (error) *error = "BLOCKED: formal GATR uses IPA knowledge openings, not hiding records";
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
        proof.hiding_commitment_records.clear();
        const auto commitment_open_start = std::chrono::steady_clock::now();
        for (const auto& tensor : context.tensors) {
            const std::uint32_t domain = tensor.domain_size;
            if (context.ipa_generators.size() < domain) {
                if (error) *error = "authenticated SRS generator domain is too small";
                return false;
            }
            CommitmentWire commitment;
            commitment.tensor_id = tensor.id;
            commitment.visibility = tensor.visibility;
            commitment.domain_size = domain;
            std::vector<zkdl::CpuG1Point> generators(context.ipa_generators.begin(),
                                                     context.ipa_generators.begin() + domain);
            zkdl::CpuG1Point blinded = zkdl::cpu_g1_sum(generators, tensor.values);
            const FrWire randomness = relation_blinding_scalar(tensor, context.blinding_scalars);
            if (randomness != FrWire{})
                blinded = zkdl::cpu_g1_add(
                    blinded, zkdl::cpu_g1_mul(context.ipa_blinding_generator, randomness));
            commitment.commitment = zkdl::cpu_g1_to_native(blinded);
            proof.commitments.push_back(commitment);
        }
        if (!verify_model_registry(proof.commitments,
                                   context.registered_model_commitments, error)) return false;
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
                if (!prove_sparse_route(operation, tensor_map, context.blinding_scalars,
                                        context.ipa_generators,
                                        context.ipa_blinding_generator,
                                        context.ipa_u_generator,
                                        proof.commitments, context.statement,
                                        transcript, route, error)) return false;
                operation_proof.route_proof = route.serialize();
                transcript.absorb("gat.relation.route", operation_proof.route_proof);
            } else if (!operation_proof.route_proof.empty()) {
                throw std::invalid_argument("non-route operation carries a sparse route proof");
            }
            if (!operation.fac_bindings.empty()) {
                FacLine8Proof fac;
                if (!prove_fac_line8(operation, tensor_map, context.blinding_scalars,
                                     context.ipa_generators,
                                     context.ipa_blinding_generator,
                                     context.ipa_u_generator, proof.commitments,
                                     context.statement,
                                     transcript, fac, error)) return false;
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
                                              context.blinding_scalars,
                                              context.ipa_generators,
                                              context.ipa_blinding_generator,
                                              context.ipa_u_generator,
                                              proof.commitments,
                                              context.statement,
                                              transcript, payload, error)) return false;
                operation_proof.batch_proof = serialize_constraint_payload(payload);
                transcript.absorb("gat.relation.batch", operation_proof.batch_proof);
            } else if (!operation_proof.batch_proof.empty()) {
                throw std::invalid_argument("operation has an unexpected batched relation proof");
            }
            for (const auto& primitive : operation.primitives) {
                std::vector<byte> primitive_bytes;
                if (!prove_matmul_primitive(primitive, tensor_map, context.ipa_generators,
                                             context.ipa_blinding_generator,
                                             context.ipa_u_generator,
                                             primitive_bytes, error)) return false;
                operation_proof.primitive_proofs.push_back(std::move(primitive_bytes));
                transcript.absorb("gat.relation.primitive", operation_proof.primitive_proofs.back());
                OperationProof::PrimitiveBinding binding;
                if (!prove_primitive_binding(primitive,
                                              operation_proof.primitive_proofs.back(), tensor_map,
                                              context.blinding_scalars,
                                              context.ipa_generators,
                                              context.ipa_blinding_generator,
                                              context.ipa_u_generator,
                                              proof.commitments, context.statement,
                                              binding, error)) return false;
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
                ConstraintProof constraint;
                if (!fill_constraint_openings(
                        descriptor, operation.tensor_refs, tensor_map,
                        context.blinding_scalars, context.ipa_generators,
                        context.ipa_blinding_generator, context.ipa_u_generator,
                        proof.commitments, context.statement, transcript,
                        constraint, error)) return false;
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
        std::vector<zkdl::CpuG1Point> output_generators(context.ipa_generators.begin(),
                                                        context.ipa_generators.begin() + output_domain);
        const OperationDescriptor output_descriptor = public_output_opening_descriptor();
        const zkdl::CpuIpaOpeningMetadata output_metadata = relation_opening_metadata(
            output_descriptor, proof.public_output_commitment, {}, context.statement);
        proof.public_output_opening = zkdl::cpu_open(
            output_generators, output_values, output_point,
            context.ipa_blinding_generator, context.ipa_u_generator, FrWire{}, output_metadata);
        std::vector<byte> public_bytes;
        zkdl::serialize_opening(proof.public_output_opening, public_bytes);
        transcript.absorb("gat.public_output.opening", public_bytes);
        if (context.commitment_open_time_seconds != nullptr)
            *context.commitment_open_time_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - commitment_open_start).count();
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
    if (context.ipa_generators.empty() || context.ipa_blinding_generator.is_zero() ||
        context.ipa_u_generator.is_zero() ||
        !same_g1_basis(context.generators, context.ipa_generators)) {
        if (error) *error = "BLOCKED: independent verifier lacks authenticated IPA G/H/U basis";
        return false;
    }
    if (!descriptor_vector_valid(context.operations, error) ||
        !formal_relation_components_supported(context.operations, error)) return false;
    if (context.gpu_backend) {
        if (error) *error =
            "BLOCKED: formal zkDL GPU GAT verification is unavailable until every relation family "
            "and terminal opening runs on native CUDA cryptographic kernels";
        return false;
    }
    if (proof.operations.size() != context.operations.size() || proof.commitments.empty()) return false;
    if (!verify_tensor_contract(context, proof.commitments, error) ||
        !verify_model_registry(proof.commitments,
                               context.registered_model_commitments, error) ||
        !verify_public_tensor_values(context, proof.commitments, error)) return false;
    if (!proof.hiding_commitment_records.empty() ||
        !context.hiding_commitment_records.empty()) {
        if (error) *error = "legacy hiding commitment records are rejected by formal GATR";
        return false;
    }
    std::set<std::string> commitment_ids;
    for (const auto& commitment : proof.commitments) {
        if (commitment.tensor_id.empty() || !commitment_ids.insert(commitment.tensor_id).second ||
            commitment.domain_size == 0U || log2_exact(commitment.domain_size) == std::numeric_limits<std::uint32_t>::max() ||
            commitment.domain_size > context.ipa_generators.size()) {
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
                !verify_sparse_route(expected_operation, proof.commitments,
                                     context.ipa_generators,
                                     context.ipa_blinding_generator,
                                     context.ipa_u_generator, context.statement,
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
                                  context.ipa_generators,
                                  context.ipa_blinding_generator,
                                  context.ipa_u_generator, context.statement,
                                  transcript, fac, error)) return false;
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
                                   proof.commitments, context.ipa_generators,
                                   context.ipa_blinding_generator,
                                   context.ipa_u_generator, context.statement,
                                   transcript, error)) return false;
            transcript.absorb("gat.relation.batch", operation.batch_proof);
        } else if (!operation.batch_proof.empty()) {
            if (error) *error = "operation carries an unexpected batched family proof";
            return false;
        }
        for (size_t primitive_index = 0U;
             primitive_index < expected_operation.primitives.size(); ++primitive_index) {
            if (!verify_matmul_primitive(expected_operation.primitives[primitive_index],
                                         operation.primitive_proofs[primitive_index],
                                         context.ipa_generators,
                                         context.ipa_blinding_generator,
                                         context.ipa_u_generator, context.gpu_backend,
                                         error)) return false;
            transcript.absorb("gat.relation.primitive",
                              operation.primitive_proofs[primitive_index]);
            if (!verify_primitive_binding(expected_operation.primitives[primitive_index],
                                          operation.primitive_proofs[primitive_index],
                                          operation.primitive_bindings[primitive_index],
                                          proof.commitments, context.ipa_generators,
                                          context.ipa_blinding_generator,
                                          context.ipa_u_generator, context.statement,
                                          error)) return false;
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
                                       proof.commitments, context.ipa_generators,
                                       context.ipa_blinding_generator,
                                       context.ipa_u_generator,
                                       context.statement,
                                       transcript, error)) return false;
        }
    }
    if (proof.public_output_commitment >= proof.commitments.size() ||
        proof.commitments[proof.public_output_commitment].tensor_id != "public_Y" ||
        proof.commitments[proof.public_output_commitment].visibility != 1U ||
        proof.public_output_opening.commitment_index != proof.public_output_commitment ||
        proof.public_output_opening.generator_size != proof.commitments[proof.public_output_commitment].domain_size ||
        proof.public_output_opening.generator_size > context.ipa_generators.size() ||
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
    std::vector<zkdl::CpuG1Point> output_generators(context.ipa_generators.begin(), context.ipa_generators.begin() + domain);
    const G1Jacobian_t expected_public_commitment = zkdl::cpu_g1_to_native(
        zkdl::cpu_g1_sum(output_generators, public_values));
    if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(
                                proof.commitments[proof.public_output_commitment].commitment),
                            zkdl::cpu_g1_from_native(expected_public_commitment))) {
        if (error) *error = "public output commitment is not bound to verifier-supplied Y";
        return false;
    }
    const zkdl::CpuIpaOpeningMetadata output_metadata = relation_opening_metadata(
        public_output_opening_descriptor(), proof.public_output_commitment, {},
        context.statement);
    if (!zkdl::cpu_verify_opening(
            proof.public_output_opening,
            zkdl::cpu_g1_from_native(proof.commitments[proof.public_output_commitment].commitment),
            std::vector<zkdl::CpuG1Point>(context.ipa_generators.begin(),
                                          context.ipa_generators.begin() + domain),
            context.ipa_blinding_generator, context.ipa_u_generator,
            output_metadata, error)) return false;
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
