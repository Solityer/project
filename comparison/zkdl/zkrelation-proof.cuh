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

    bool valid(const std::vector<std::string>& operation_refs,
               std::string* error = nullptr) const {
        if (kind != 1U || !matmul_shape.valid(error) || tensor_refs.size() != 3U) {
            if (error && error->empty()) *error = "relation primitive descriptor is not canonical";
            return false;
        }
        std::set<std::string> refs;
        for (const auto& ref : tensor_refs) {
            if (ref.empty() || !refs.insert(ref).second ||
                std::find(operation_refs.begin(), operation_refs.end(), ref) == operation_refs.end()) {
                if (error) *error = "relation primitive tensor reference is not in operation";
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

    bool valid(std::string* error = nullptr) const {
        if (id.empty() || id.find('/') == std::string::npos ||
            !relation_kind_valid(static_cast<std::uint32_t>(kind)) || degree == 0U || degree > 8U) {
            if (error) *error = "relation operation descriptor is not canonical";
            return false;
        }
        if (mapping.empty() || padding.empty() || terminal_claim.empty() || terminal_point.empty()) {
            if (error) *error = "relation operation descriptor is missing mapping/padding/terminal identity";
            return false;
        }
        if (source_refs.empty() || tensor_refs.empty() || constraints.empty()) {
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
        for (const auto& primitive : primitives)
            if (!primitive.valid(tensor_refs, error)) return false;
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

struct RelationStatement {
    std::array<byte, 32> statement_digest{};
    std::array<byte, 32> model_digest{};
    std::array<byte, 32> table_digest{};
    std::array<byte, 32> ledger_digest{};
    std::array<byte, 32> srs_digest{};
    std::array<byte, 32> registry_digest{};
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
    std::vector<std::uint32_t> opening_commitment_indices;
    zkdl::SumcheckProof sumcheck;
    std::vector<zkdl::EvaluationOpeningProof> openings;
};

struct OperationProof {
    OperationDescriptor descriptor;
    std::vector<ConstraintProof> constraints;
    // One serialized proof per descriptor primitive, in descriptor order.
    std::vector<std::vector<byte>> primitive_proofs;
};

struct Proof {
    static const std::uint32_t kVersion = 1U;
    std::uint32_t version = kVersion;
    RelationStatement statement;
    std::vector<CommitmentWire> commitments;
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
    std::vector<OperationDescriptor> operations;
    std::vector<zkdl::CpuG1Point> generators;
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
        !zkdl::read_u64(in, offset, count) || count > 64U) return false;
    descriptor.terms.resize(static_cast<size_t>(count));
    for (auto& term : descriptor.terms) if (!read_monomial(in, offset, term)) return false;
    return true;
}

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
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.primitives.size()));
    for (const auto& primitive : descriptor.primitives) append_primitive(out, primitive);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.constraints.size()));
    for (const auto& constraint : descriptor.constraints) append_constraint(out, constraint);
}

/* The operation reader keeps its two boolean fields separate so their
 * canonical representation is unambiguous across languages. */
inline bool read_operation_strict(const std::vector<byte>& in, size_t& offset,
                                  OperationDescriptor& descriptor) {
    std::uint32_t kind = 0U, public_route = 0U, reindex = 0U;
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
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    descriptor.primitives.resize(static_cast<size_t>(count));
    for (auto& primitive : descriptor.primitives)
        if (!read_primitive(in, offset, primitive)) return false;
    if (!zkdl::read_u64(in, offset, count) || count == 0U || count > 64U) return false;
    descriptor.kind = static_cast<RelationKind>(kind);
    descriptor.public_route = public_route != 0U;
    descriptor.requires_reindex = reindex != 0U;
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
        !zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    statement.public_output.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                   in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    return true;
}

inline void append_constraint_proof(std::vector<byte>& out, const ConstraintProof& proof) {
    append_constraint(out, proof.descriptor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.opening_commitment_indices.size()));
    for (const std::uint32_t index : proof.opening_commitment_indices) zkdl::append_u32(out, index);
    std::vector<byte> sumcheck;
    zkdl::serialize_sumcheck(proof.sumcheck, sumcheck);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck.size()));
    out.insert(out.end(), sumcheck.begin(), sumcheck.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.openings.size()));
    for (const auto& opening : proof.openings) zkdl::serialize_opening(opening, out);
}

inline bool read_constraint_proof(const std::vector<byte>& in, size_t& offset,
                                  ConstraintProof& proof) {
    std::uint64_t count = 0U, bytes = 0U;
    if (!read_constraint(in, offset, proof.descriptor) ||
        !zkdl::read_u64(in, offset, count) || count == 0U || count > 64U) return false;
    proof.opening_commitment_indices.resize(static_cast<size_t>(count));
    for (auto& index : proof.opening_commitment_indices)
        if (!zkdl::read_u32(in, offset, index)) return false;
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                              in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    size_t inner = 0U;
    if (!zkdl::deserialize_sumcheck(encoded, inner, proof.sumcheck) || inner != encoded.size()) return false;
    if (!zkdl::read_u64(in, offset, count) || count == 0U || count > 64U) return false;
    proof.openings.resize(static_cast<size_t>(count));
    for (auto& opening : proof.openings)
        if (!zkdl::deserialize_opening(in, offset, opening)) return false;
    return true;
}

inline void append_operation_proof(std::vector<byte>& out, const OperationProof& proof) {
    append_operation(out, proof.descriptor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.primitive_proofs.size()));
    for (const auto& primitive : proof.primitive_proofs) {
        zkdl::append_u64(out, static_cast<std::uint64_t>(primitive.size()));
        out.insert(out.end(), primitive.begin(), primitive.end());
    }
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
    if (!zkdl::read_u64(in, offset, count) || count == 0U || count > 64U) return false;
    proof.constraints.resize(static_cast<size_t>(count));
    for (auto& constraint : proof.constraints)
        if (!read_constraint_proof(in, offset, constraint)) return false;
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
    if (proof.opening_commitment_indices.size() != proof.openings.size() ||
        proof.openings.size() != expected_tensor_refs.size() ||
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
        challenges.size() >= 32U ||
        domain != (std::uint32_t(1U) << challenges.size())) {
        if (error) *error = "relation verifier has no authenticated generator domain";
        return false;
    }
    std::vector<FrWire> terminal_values;
    terminal_values.reserve(proof.openings.size());
    for (size_t i = 0U; i < proof.openings.size(); ++i) {
        const std::uint32_t index = proof.opening_commitment_indices[i];
        if (index >= commitments.size() || index >= commitments.size() ||
            commitments[index].tensor_id != expected_tensor_refs[i] ||
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
    if (!same_wire(zkdl::fr_wire(proof.sumcheck.final_claim), eval_constraint(expected, terminal_values))) {
        if (error) *error = "relation terminal equation failed";
        return false;
    }
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
        zkdl::Transcript transcript("zkdl-gat-relation-v1");
        absorb_statement(transcript, proof.statement, proof.commitments);
        for (const auto& operation : context.operations) {
            absorb_descriptor(transcript, operation);
            OperationProof operation_proof;
            operation_proof.descriptor = operation;
            for (const auto& descriptor : operation.constraints) {
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
                for (const auto& ref : operation.tensor_refs) {
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
        proof.statement.public_output != context.statement.public_output) {
        if (error) *error = "relation proof statement/resource identity mismatch";
        return false;
    }
    if (!descriptor_vector_valid(context.operations, error) ||
        proof.operations.size() != context.operations.size() || proof.commitments.empty()) return false;
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
    for (size_t op_index = 0U; op_index < context.operations.size(); ++op_index) {
        const auto& expected_operation = context.operations[op_index];
        const auto& operation = proof.operations[op_index];
        std::vector<byte> expected_bytes, actual_bytes;
        append_operation(expected_bytes, expected_operation);
        append_operation(actual_bytes, operation.descriptor);
        if (expected_bytes != actual_bytes || operation.constraints.size() != expected_operation.constraints.size()) {
            if (error) *error = "relation operation descriptor/count mismatch";
            return false;
        }
        absorb_descriptor(transcript, expected_operation);
        for (size_t constraint_index = 0U; constraint_index < expected_operation.constraints.size(); ++constraint_index)
            if (!verify_constraint(operation.constraints[constraint_index], expected_operation.constraints[constraint_index],
                                   expected_operation.tensor_refs,
                                   proof.commitments, context.generators, transcript, error)) return false;
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
