#pragma once

// Native proof boundary for the zkGPT GAT adaptation.  This header only
// describes the adapter-side relation and wire format; the actual arithmetic
// primitives are the classes/functions exported by the official src tree.

#include "gat_types.hpp"
#include "hyrax.hpp"
#include "prover.hpp"
#include "range_prover.hpp"
#include "proof_session.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace zkgpt {
namespace gat {

struct InputSegment {
    std::string name;
    std::size_t offset = 0;
    std::size_t logical_length = 0;
    std::size_t padded_length = 0;
};

struct InputLayout {
    std::size_t n_raw = 0;
    std::size_t n_in = 0;
    std::vector<InputSegment> segments;
    std::string selector = "00:X_pub,01:W,10:Aux,11:L_qry";
};

struct MatMulDescriptor {
    std::string relation_id;
    std::string family_id;
    std::string lhs;
    std::string rhs;
    std::string output;
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t cols = 0;
    std::size_t output_coordinates = 0;
    std::size_t lhs_offset = 0;
    std::size_t rhs_offset = 0;
    std::size_t output_offset = 0;
    std::string coordinate_rule = "row_major_exact";
    std::string scale_rule = "canonical_quant_scale_chain";
    std::string native_path = "prover::FCONN / dedicated matrix Sumcheck";
};

struct SparseGkrDescriptor {
    std::string relation_id;
    std::string family_id;
    std::size_t edge_count = 0;
    std::size_t value_width = 0;
    std::size_t input_offset = 0;
    std::size_t source_offset = 0;
    std::size_t destination_offset = 0;
    std::size_t output_offset = 0;
    std::string coordinate_rule = "edge_gid_then_dst_stable";
    std::string group_rule = "Q_grp_new_at_group_start,Q_grp_end_at_group_end";
    std::vector<std::string> constraints;
};

struct LassoFamilyDescriptor {
    std::string relation_id;
    std::string family_id;
    std::string table_id;
    std::size_t real_queries = 0;
    std::size_t padded_queries = 0;
    std::size_t table_domain = 0;
    Int padding_value = 0;
    std::size_t query_offset = 0;
    std::string incidence_rule = "L_qry_row_major_canonical";
    std::string challenge_policy = "LC_before_challenge";
    std::string query_encoding = "lookup_tuple_encoding_id:packed_signed_pair_v1";
};

struct CheckpointDescriptor {
    std::string id;
    std::string relation_id;
    std::string segment;
    std::size_t coordinates = 0;
    std::size_t coordinate_offset = 0;
    std::string coordinate_rule = "canonical_row_major";
};

// This is the complete adapter description consumed by the official
// layered-circuit/prover path.  It intentionally carries no witness values.
struct GatCircuitPlan {
    layeredCircuit circuit;
    RelationRegistry registry;
    InputLayout input_layout;
    std::vector<MatMulDescriptor> matmuls;
    std::vector<SparseGkrDescriptor> sparse_gkr;
    std::vector<LassoFamilyDescriptor> lasso;
    std::vector<CheckpointDescriptor> checkpoints;
    std::vector<std::string> native_api;
    std::string circuit_id;
    std::string plan_digest;
    std::string opening_policy = "same_r_three_terminal_openings_v1";
    std::vector<std::string> terminal_openings;
    bool official_layered = false;
    bool has_placeholder = false;
};

GatCircuitPlan build_gat_circuit_plan(const Graph &graph, const Model &model,
                                      const QuantConfig &quant,
                                      const ForwardResult &forward);

// A plan without a private trace is useful during registration and for the
// independent verifier.  Its layout is a conservative canonical lower bound;
// a proof plan built with a trace must be used by the prover.
GatCircuitPlan build_gat_circuit_plan(const Graph &graph, const Model &model,
                                      const QuantConfig &quant);

// Structural and digest validation for the canonical, public plan.  This
// deliberately validates descriptors and gate coordinates only; it does not
// claim that the native masked transcript is available.
bool validate_gat_circuit_plan(const GatCircuitPlan &plan,
                               std::string *reason = nullptr);

struct CommitmentWire {
    std::string scheme = "hyrax-pedersen-bn254";
    std::string basis_id;
    std::size_t logical_size = 0;
    std::size_t padded_size = 0;
    std::string point_hex;

    bool valid() const;
};

struct ModelRegistration {
    CommitmentWire com_W;
    std::string dataset;
    std::string model_arch_id;
    std::string quant_cfg_id;
    std::string static_tables_id;
    std::string field_id = "BN254";
    std::string field_modulus;
    std::string manifest_digest;
    std::string model_digest;
    std::string bounds_digest;
    std::string quant_digest;
    std::string layout_digest;
    std::string selector = "00:X_pub,01:W,10:Aux,11:L_qry";
    std::size_t n_raw = 0;
    std::size_t n_in = 0;
    std::vector<InputSegment> input_segments;
    std::string opening_policy = "same_r_three_openings_v1";
    std::string binding_digest;
    std::string registration_digest;
    std::string status;
};

// Uses the official Hyrax Pedersen commitment primitive.  The blinding scalar
// is sampled internally and is never placed in the wire object.
ModelRegistration register_model(const Model &model, const QuantConfig &quant,
                                  const std::string &model_arch_id = "gat_l3_8x2");

// Register a model against the canonical plan layout.  The W commitment is
// padded to layout.n_in, the same domain shared by X_pub, Aux and L_qry.
ModelRegistration register_model(const Model &model, const QuantConfig &quant,
                                  const InputLayout &layout,
                                  const std::string &model_arch_id = "gat_l3_8x2");

// Recompute all public registration identities from the private model and
// quantization policy without recomputing a randomized Hyrax point.
bool registration_matches_model(const Model &model, const QuantConfig &quant,
                                const InputLayout &layout,
                                const ModelRegistration &registration);

CommitmentWire commit_private_segment(const std::vector<Int> &values,
                                      const std::string &basis_label);
CommitmentWire commit_private_segment(const std::vector<Int> &values,
                                      const std::string &basis_label,
                                      std::size_t padded_domain);

struct TranscriptMessage {
    std::string label;
    std::string payload;
};

class FiatShamirTranscript {
public:
    explicit FiatShamirTranscript(const std::string &domain = "zkgpt-gat-fs-v1");

    void absorb(const std::string &label, const std::string &payload);
    Field challenge(const std::string &label);
    std::string digest() const;
    const std::vector<TranscriptMessage> &messages() const { return messages_; }

private:
    std::string domain_;
    std::vector<TranscriptMessage> messages_;
};

struct ProofSection {
    std::string protocol;
    std::vector<TranscriptMessage> messages;
};

// Verifier-facing wire.  Deliberately no Y, W, Aux, L, trace, model tensor,
// or opening randomness is present: those belong to the public context or to
// the prover and are never serialized.
struct GatProofArtifact {
    std::uint32_t version = 1;
    std::string schema = "zkgpt-gat-proof-v1";
    std::string relation_id = "R_GAT-zkGPT";
    std::string privacy_profile;
    std::string statement_digest;
    CommitmentWire com_aux;
    CommitmentWire com_L;
    ProofSection Pi_Lasso{"Pi_Lasso", {}};
    ProofSection Pi_MatMul{"Pi_MatMul", {}};
    ProofSection Pi_GKR{"Pi_GKR", {}};
    ProofSection Pi_combine{"Pi_combine", {}};
    ProofSection Pi_open{"Pi_open", {}};
    std::string transcript_digest;

    bool has_private_payload() const;
    bool valid_wire(std::string *reason = nullptr) const;
    bool write(const std::string &path, std::string *reason = nullptr) const;
    static bool read(const std::string &path, GatProofArtifact &out,
                     std::string *reason = nullptr);
};

struct VerifierContext {
    std::string relation_id = "R_GAT-zkGPT";
    std::string model_arch_id = "gat_l3_8x2";
    std::string privacy_extension_profile_id;
    std::string quant_cfg_id;
    std::string static_tables_id;
    Graph graph;
    Tensor2D public_features;
    QuantConfig quant;
    StaticTables tables;
    Tensor2D public_logits;
    CommitmentWire com_W;
    ModelRegistration registration;
    GatCircuitPlan plan;
};

VerifierContext make_verifier_context(const Graph &graph,
                                      const Tensor2D &public_features,
                                      const QuantConfig &quant,
                                      const StaticTables &tables,
                                      const Tensor2D &public_logits,
                                      const CommitmentWire &com_W,
                                      const GatCircuitPlan &plan,
                                      const std::string &model_arch_id = "gat_l3_8x2",
                                      const std::string &privacy_profile = "masked_full");

VerifierContext make_verifier_context(const Graph &graph,
                                      const Tensor2D &public_features,
                                      const QuantConfig &quant,
                                      const StaticTables &tables,
                                      const Tensor2D &public_logits,
                                      const ModelRegistration &registration,
                                      const GatCircuitPlan &plan,
                                      const std::string &privacy_profile = "masked_full");

// Canonical digest used by both the producer and the independent verifier.
// The implementation remains in the native proof translation unit so the
// CLI cannot accidentally invent a second statement encoding.
std::string verifier_context_digest(const VerifierContext &context);

// Construct the native session statement from the same authenticated
// VerifierContext/resource identities and the canonical Hyrax basis setup.
// All digest fields are fixed-width lowercase values.  This factory is a
// foundation binding only; it does not make the formal masked_full prover
// available.
NativeProofStatement make_native_proof_statement(
    const VerifierContext &context, const CommitmentWire &com_aux,
    const CommitmentWire &com_L, const G1 &G,
    const std::vector<G1> &generators, const std::vector<Fr> &basis,
    std::size_t l_dimension);

enum class ProveStatus {
    GENERATED,
    UNSUPPORTED_PRIVACY_PROFILE,
    REJECTED
};

const char *prove_status_name(ProveStatus status);

struct ProveResult {
    ProveStatus status = ProveStatus::REJECTED;
    std::string diagnostic;
    bool has_artifact = false;
    GatProofArtifact artifact;
};

// The native zkGPT code exposes the interactive GKR/Lasso verifier only as an
// internal method and does not expose a serializable masking transcript.  The
// formal prover therefore refuses to emit a successful artifact until a
// complete masked_full implementation is available.
ProveResult prove_gat(const GatCircuitPlan &plan, const Graph &graph,
                      const Tensor2D &public_features, const Model &model,
                      const QuantConfig &quant, const ForwardResult &forward,
                      const CommitmentWire &com_W,
                      const std::string &privacy_profile = "masked_full");

ProveResult prove_gat(const GatCircuitPlan &plan, const Graph &graph,
                      const Tensor2D &public_features, const Model &model,
                      const QuantConfig &quant, const ForwardResult &forward,
                      const ModelRegistration &registration,
                      const std::string &privacy_profile = "masked_full");

bool verify_gat(const VerifierContext &context, const GatProofArtifact &artifact,
                std::string *reason = nullptr);

struct MaskingSession {
    std::string profile_id;
    std::string session_nonce;
    bool complete = false;
};

// Fresh masking/opening randomness is provisioned per proof.  It is exposed
// so a future native masking extension can be added without changing the
// proof wire, but complete masked_full support remains explicit and fail
// closed today.
MaskingSession begin_masking_session(const std::string &profile_id,
                                     const std::string &statement_digest);

struct NativePrimitiveEvidence {
    bool sumcheck_called = false;
    bool gkr_path_bound = false;
    bool lasso_range_called = false;
    bool hyrax_commit_called = false;
    std::vector<std::string> messages;
};

// Short, bounded bridge used by tests and diagnostics.  It executes the
// official prover/range/Hyrax primitives on a one-gate relation; it is not a
// formal GAT proof and cannot produce a production artifact.
NativePrimitiveEvidence exercise_official_primitives(const std::string &label);

} // namespace gat
} // namespace zkgpt
