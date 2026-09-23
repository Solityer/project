#pragma once

// Native proof boundary for the zkGPT GAT adaptation.  This header only
// describes the adapter-side relation and wire format; the actual arithmetic
// primitives are the classes/functions exported by the official src tree.

#include "gat_types.hpp"
#include "hyrax.hpp"
#include "prover.hpp"
#include "range_prover.hpp"
#include "proof_session.hpp"
#include "verifier.hpp"
#include "native_masked_sumcheck.hpp"
#include "native_lasso.hpp"
#include "gat_sparse_gkr.hpp"

#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace zkgpt {
namespace gat {

// Hashes canonical bytes to exactly 64 lowercase hexadecimal characters.
// Callers must pass bytes, not an already encoded digest; wire readers use
// strict validation and never normalize an attacker-supplied digest.
std::string canonical_digest64(const std::string &bytes);

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

// A bounded, canonical view of the four input-layer segments.  The native
// zkGPT input layer is logically X||W||Aux||L, but materialising four padded
// copies is both unnecessary and, for the large graph instances, unsafe.
// This oracle owns only logical segment values; coordinates beyond a segment
// logical length are the canonical implicit-zero padding.  All callers use
// the same low-bit-first MLE convention as the native GKR implementation.
class SegmentedInputOracle {
public:
    using ChunkCallback = std::function<void(std::size_t, const Fr &)>;
    using FoldChunkCallback =
        std::function<bool(std::uint64_t, const std::vector<Fr> &,
                           std::string *reason)>;

    SegmentedInputOracle() = default;

    SegmentedInputOracle(const InputLayout &layout,
                         std::array<std::vector<Fr>, 4> segments);

    static SegmentedInputOracle from_int_segments(
        const InputLayout &layout,
        const std::array<std::vector<Int>, 4> &segments);

    bool valid(std::string *reason = nullptr) const;
    const InputLayout &layout() const { return layout_; }

    // Global coordinates use the canonical X/W/Aux/L offsets.  This method
    // never allocates and returns zero for implicit padding.
    Fr value_at(std::size_t global_coordinate) const;
    Fr segment_value(std::size_t segment, std::size_t local_coordinate) const;

    // Stream a bounded range of one segment.  The callback receives logical
    // local coordinates; implicit padding is included as zero only when the
    // requested range reaches padded_length.
    void stream(std::size_t segment, std::size_t begin,
                std::size_t count, const ChunkCallback &callback) const;
    std::vector<Fr> chunk(std::size_t segment, std::size_t begin,
                          std::size_t count) const;

    // Evaluate one segment MLE without constructing a padded vector.  The
    // point uses the native low-bit-first variable order.
    Fr evaluate_mle(std::size_t segment,
                    const std::vector<Fr> &point) const;
    Fr evaluate_full_mle(const std::vector<Fr> &point) const;
    Fr mle_eval(std::size_t segment, const std::vector<Fr> &point) const {
        return evaluate_mle(segment, point);
    }

    // Fold a segment by a prefix of MLE challenges.  The returned vector is
    // bounded by the remaining domain and is intended for native Sumcheck
    // chunks, not for rebuilding the full four-segment input layer.
    std::vector<Fr> fold(std::size_t segment,
                         const std::vector<Fr> &prefix) const;

    // Stream a folded domain in fixed-size chunks.  This is the production
    // path for large inputs; it never allocates the remaining MLE domain and
    // uses an explicit u64 domain/offset contract.
    bool fold_chunks(std::size_t segment, const std::vector<Fr> &prefix,
                     std::uint64_t chunk_entries,
                     const FoldChunkCallback &callback,
                     std::string *reason = nullptr) const;

    // Weighted sum over one segment.  The weight callback is evaluated only
    // for logical coordinates, while padded coordinates contribute zero.
    Fr weighted_sum(std::size_t segment,
                    const std::function<Fr(std::size_t)> &weight) const;

private:
    InputLayout layout_;
    std::array<std::vector<Fr>, 4> segments_;
};

// Shared metadata-only sizing boundary.  Registration, the prover and the
// compact-context verifier all derive this exact four-segment layout here;
// the helper never evaluates the model or allocates a witness.
InputLayout build_gat_input_layout(const Graph &graph, const Model &model,
                                   const QuantConfig &quant);

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
    // v4 relation binding.  These fields are public recipe metadata; the
    // actual operands remain in the committed input segments.
    std::string relation_schema = "zkgpt-gat-relation-v1";
    std::string lhs_segment = "X_pub";
    std::string rhs_segment = "W";
    std::string output_segment = "Aux";
    std::string lhs_coordinate_rule = "segment_offset+row*inner+term";
    std::string rhs_coordinate_rule = "segment_offset+term*cols+col";
    std::string output_coordinate_rule = "segment_offset+row*cols+col";
    std::string terminal_claim_rule = "descriptor_index_ordered_mle_points_v1";
};

// The FCONN transition is a relation between two descriptor-specific input
// MLE claims and one output MLE claim.  It is kept separately from the
// materialised layeredCircuit so both producer and verifier can authenticate
// the exact handoff without serialising gate arrays.
struct FconnClaimLink {
    std::size_t descriptor_index = 0;
    std::string relation_id;
    std::string family_id;
    std::string lhs_segment;
    std::string rhs_segment;
    std::string output_segment;
    std::size_t lhs_offset = 0;
    std::size_t rhs_offset = 0;
    std::size_t output_offset = 0;
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t cols = 0;
    std::string lhs_point_rule = "fconn/lhs/row-major-mle-v1";
    std::string rhs_point_rule = "fconn/rhs/column-major-padded-mle-v1";
    std::string output_point_rule = "fconn/output/row-major-mle-v1";
    std::string source_rule = "canonical_input_oracle";
};

// Native relation metadata for sparse attention/aggregation and exact
// SignedRoundDiv.  A descriptor is not a hash-only promise: its fields are
// part of the canonical recipe and are consumed by the v4 relation walker.
struct NativeRelationDescriptor {
    std::string relation_id;
    std::string family_id;
    std::string relation_schema = "zkgpt-gat-relation-v1";
    std::string primitive;
    std::string domain_rule;
    std::string selector_rule;
    std::string coordinate_rule;
    std::string arithmetic_rule;
    std::string terminal_rule;
    std::size_t edge_count = 0;
    std::size_t value_width = 0;
    std::size_t chunk_size = 1ULL << 12;
    std::size_t input_offset = 0;
    std::size_t output_offset = 0;
    std::vector<std::string> equations;
};

// The canonical GAT/model input keeps matrix weights in row-major order
// [term,col].  Native FCONN MLE folding consumes the RHS with the inner
// coordinate as the low bit, i.e. column-major [col,term] with a
// next-power-of-two padded_inner stride for every column.  This adapter
// boundary performs that representation change explicitly; it does not alter
// the canonical input, plan encoding, model commitment, or registration.
std::vector<Fr> pack_native_gkr_matmul_rhs(
    const MatMulDescriptor &descriptor,
    const std::vector<Fr> &canonical_input_layer);

struct SparseGkrDescriptor {
    std::string relation_id;
    std::string family_id;
    SparseGkrRelationKind relation_kind = SparseGkrRelationKind::Residual;
    std::string relation_schema = kSparseGkrRelationSchema;
    std::size_t edge_count = 0;
    std::size_t value_width = 0;
    std::size_t chunk_size = kSparseGkrChunkEntries;
    std::size_t input_offset = 0;
    std::size_t source_offset = 0;
    std::size_t destination_offset = 0;
    std::size_t output_offset = 0;
    std::string coordinate_rule = "edge_gid_then_dst_stable";
    std::string group_rule = "Q_grp_new_at_group_start,Q_grp_end_at_group_end";
    std::string arithmetic_rule = kSparseGkrArithmeticRule;
    std::string overflow_rule = kSparseGkrOverflowRule;
    std::string terminal_rule = "terminal-claim-chain/input-output-binding-v1";
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

// Public model architecture metadata used by the compact verifier-context
// recipe.  It contains tensor names and shapes only; tensor values are never
// part of a verifier context and remain bound exclusively by com_W.
struct CanonicalTensorShape {
    std::string name;
    Shape shape;
};

struct CanonicalModelShape {
    std::string model_arch_id;
    std::size_t tensor_count = 0;
    std::size_t layers = 0;
    std::size_t output_heads = 0;
    std::size_t classes = 0;
    std::size_t input_dim = 0;
    std::size_t hidden_dim = 0;
    std::vector<CanonicalTensorShape> tensors;
};

struct GatCapacityReport {
    std::string schema = "zkgpt-gat-capacity-v1";
    std::size_t num_nodes = 0;
    std::size_t num_edges = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::size_t graph_count = 0;
    CanonicalModelShape model_shape;
    InputLayout input_layout;
    std::size_t fconn_count = 0;
    bool explicit_matmul_gate_expansion = false;
    std::size_t max_materialized_gate = 0;
    std::size_t max_materialized_vector = 0;
    std::size_t attention_relation_count = 0;
    std::size_t division_relation_count = 0;
    std::size_t context_recipe_bytes_bound = 0;
    std::size_t context_payload_bytes_bound = 0;
    std::size_t virtual_input_domain = 0;
    std::size_t materialized_fr = 0;
    std::size_t materialized_g1 = 0;
    std::size_t explicit_gate_count = 0;
    std::uint64_t estimated_peak_host_memory_bytes = 0;
    std::size_t native_sparse_relation_count = 0;
    std::size_t scratch_chunk_entries = 0;
    std::uint64_t sparse_domain_entries = 0;
    std::uint64_t sparse_working_set_entries = 0;
    std::uint64_t sparse_working_set_bytes = 0;
    std::size_t max_materialized_gate_bytes = 0;
    std::size_t max_materialized_vector_bytes = 0;
    bool ppi_node_ptr_boundary = false;
    bool ppi_edge_ptr_boundary = false;
    bool sparse_ready = false;
    bool ready = false;
    std::string reason;
};

// Metadata/capacity inspection only: no checkpoint values, forward pass,
// witness, materialized circuit, proof or verifier are constructed.
GatCapacityReport inspect_gat_capacity(const Graph &graph, const Model &model,
                                       const QuantConfig &quant);

// This is the complete adapter description consumed by the official
// layered-circuit/prover path.  It intentionally carries no witness values.
struct GatCircuitPlan {
    // canonical_plan_recipe_v1 is the sole public recipe representation.  A
    // materialized layeredCircuit is derived from it and is never serialized
    // in a verifier context.
    std::string recipe_schema = "canonical_plan_recipe_v2";
    std::string relation_schema = "zkgpt-gat-relation-v1";
    // Versioned sparse relation boundary.  A context carrying an older or
    // unknown value is rejected before materialization/proof consumption.
    std::string sparse_relation_schema = kSparseGkrRelationSchema;
    std::uint32_t sparse_relation_version = kSparseGkrRelationVersion;
    std::size_t sparse_chunk_size = kSparseGkrChunkEntries;
    std::string sparse_plan_digest;
    bool virtual_sparse = false;
    std::string model_arch_id;
    CanonicalModelShape model_shape;
    std::size_t graph_num_nodes = 0;
    std::size_t graph_num_features = 0;
    std::size_t graph_count = 0;
    std::size_t graph_num_classes = 0;
    std::size_t graph_edge_count = 0;
    std::string graph_topology_digest;
    std::string quant_cfg_id;
    std::string quant_digest;
    std::string static_tables_id;
    layeredCircuit circuit;
    RelationRegistry registry;
    InputLayout input_layout;
    std::vector<MatMulDescriptor> matmuls;
    std::vector<FconnClaimLink> fconn_claim_links;
    std::vector<SparseGkrDescriptor> sparse_gkr;
    std::vector<NativeRelationDescriptor> native_relations;
    std::vector<LassoFamilyDescriptor> lasso;
    std::vector<CheckpointDescriptor> checkpoints;
    std::vector<std::string> native_api;
    std::string circuit_id;
    std::string plan_digest;
    std::string opening_policy = "same_r_three_terminal_openings_v1";
    std::vector<std::string> terminal_openings;
    // The final official layer exposes the complete public Y prefix followed
    // by one public zero coordinate for every arithmetic residual.  Keeping
    // these counts in the plan makes the output statement unambiguous and
    // prevents an adapter caller from silently truncating the residual tail.
    std::size_t public_output_coordinates = 0;
    std::size_t residual_coordinates = 0;
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

// Prover-side values for the canonical layered relation.  `input_layer` is
// the padded X_pub || W || Aux || L_qry vector consumed by the official
// circuit; `values` follows layeredCircuit's native val[layer] convention.
// The canonical copy is retained only for bounded adapter-local relation
// checks and is never serialized into a proof artifact.
struct GatCircuitWitness {
    std::vector<Fr> input_layer;
    std::vector<Fr> canonical_input_layer;
    std::vector<std::vector<Fr>> values;
    std::vector<Fr> public_output;
    std::vector<Fr> residuals;
    // Logical segments used by v4 relation code.  Existing dense fields are
    // retained for bounded legacy callers; production scalable callers use
    // this oracle and do not construct a 4*n_in vector.
    std::shared_ptr<SegmentedInputOracle> input_oracle;
    bool relation_ok = false;
};

GatCircuitWitness build_gat_circuit_witness(
    const GatCircuitPlan &plan, const Graph &graph,
    const Tensor2D &public_features, const Model &model,
    const QuantConfig &quant, const ForwardResult &forward);

// Evaluate every native uni/bin gate using the official input/previous-layer
// routing rules and check the public Y prefix plus the zero residual tail.
// This is a fast prover-side correctness gate; it does not claim to replace
// the native GKR/Lasso/Hyrax proof verifier.
bool evaluate_gat_circuit_witness(const GatCircuitPlan &plan,
                                  const GatCircuitWitness &witness,
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

    // Prover-only state.  This value is deliberately absent from the JSON
    // registration wire and is never consulted by the independent verifier.
    // The long-lived W commitment blinds a canonical padding coordinate; the
    // prover must retain that fresh scalar in memory to open the same
    // commitment during this process.
    Fr com_W_padding_mask = Fr(0);
    bool has_private_padding_mask = false;
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

// Prover-side commitment material.  The scalar is intentionally returned only
// through this in-memory object; callers must keep it in a private process
// secret store and must never place it in model_registration.json or any
// verifier-facing artifact.
struct ProverCommitment {
    CommitmentWire wire;
    Fr blinder = Fr(0);
};

ProverCommitment commit_private_segment_for_prover(
    const std::vector<Int> &values, const std::string &basis_label,
    std::size_t padded_domain);

// Persist/recover only the prover-side blinder for a long-lived com_W.  The
// file is baseline-local, identity-bound, and created with owner-only
// permissions; it is never part of model_registration.json or a verifier
// context.  A mismatched registration, basis, point, or trailing payload is
// rejected fail-closed.
bool write_prover_commitment_secret(const ModelRegistration &registration,
                                    const std::string &path,
                                    std::string *reason = nullptr);
bool read_prover_commitment_secret(const ModelRegistration &registration,
                                   const std::string &path, Fr &blinder,
                                   std::string *reason = nullptr);

struct TranscriptMessage {
    std::string label;
    std::string payload;
    // NativeProofSession message type and original global order.  Keeping
    // these on the artifact wire allows the independent verifier to rebuild
    // the exact typed transcript instead of checking section digests only.
    std::uint8_t kind = 1;
    std::uint64_t sequence = 0;
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

// SparseGKR proof metadata is a first-class artifact section.  Native typed
// messages carry the actual masked sumcheck openings; these records bind the
// relation-local endpoints and every terminal/FCONN transition to the
// canonical plan order so a verifier cannot silently skip a scope.
struct SparseGkrRelationWire {
    std::string relation_id;
    std::string initial_claim;
    std::string final_claim;
    std::vector<std::string> terminal_values;
    std::string initial_commitment;
    std::vector<std::string> terminal_commitments;
    std::uint64_t transcript_begin = 0;
    std::uint64_t transcript_end = 0;
};

struct SparseGkrLinkWire {
    std::string from_relation;
    std::string to_relation;
    std::string claim_name;
    std::uint32_t from_terminal = 0;
    std::uint32_t to_initial = 0;
    std::string from_claim;
    std::string to_claim;
};

struct SparseGkrProofWire {
    std::string schema = "zkgpt-gat-sparse-proof-v2";
    std::uint32_t version = 2;
    std::string plan_digest;
    std::string schedule_digest;
    std::vector<SparseGkrRelationWire> relations;
    std::vector<SparseGkrLinkWire> terminal_links;
    std::vector<SparseGkrLinkWire> binding_links;
    std::string input_binding;
    std::string output_binding;
    NativeLassoChunkBundleWire lasso_chunks;
    std::vector<hyrax::ChunkedMleWire> input_segment_openings;
    hyrax::ChunkedMleWire output_opening;
    std::string input_terminal_commitment;
    std::string output_terminal_commitment;
    bool hyrax_lasso_complete = true;
};

// Verifier-facing wire.  Deliberately no Y, W, Aux, L, trace, model tensor,
// or opening randomness is present: those belong to the public context or to
// the prover and are never serialized.
struct GatProofArtifact {
    std::uint32_t version = 6;
    std::string schema = "zkgpt-gat-proof-v6";
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
    // The section summaries above are retained for collector compatibility,
    // but are not the proof boundary.  A production artifact carries the
    // complete typed native transcript and statement so an independent
    // verifier can replay Fiat--Shamir and consume every native proof object.
    std::string native_domain = "zkgpt-native-gat-fs-v4";
    NativeProofStatement native_statement;
    std::vector<NativeProofMessage> native_messages;
    std::string transcript_digest;
    SparseGkrProofWire sparse_gkr;

    bool has_private_payload() const;
    bool valid_wire(
        std::string *reason = nullptr,
        const NativeProofSession *authenticated_session = nullptr) const;
    bool native_session(NativeProofSession &out,
                       std::string *reason = nullptr) const;
    bool write(const std::string &path, std::string *reason = nullptr) const;
    static bool read(const std::string &path, GatProofArtifact &out,
                     std::string *reason = nullptr);
};

struct VerifierContext {
    std::string relation_id = "R_GAT-zkGPT";
    std::string backend;
    std::string model_arch_id = "gat_l3_8x2";
    std::string privacy_extension_profile_id;
    std::string quant_cfg_id;
    std::string static_tables_id;
    // Resource identities are public statement metadata.  They are carried
    // in the v3 context so an independent verifier report can be joined to a
    // result without consulting the checkpoint/cache or guessing from a
    // path.  They never contain private model material.
    std::string full_dataset_identity;
    std::string checkpoint_identity;
    std::string cache_identity;
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

// Bounded observability for adapter-local tests and result telemetry.  These
// report the compact recipe and v3 payload sizes; neither function serializes
// materialized gate arrays.
std::size_t compact_plan_recipe_bytes(const GatCircuitPlan &plan);
std::size_t verifier_context_payload_bytes(const VerifierContext &context);

// Strict baseline-local public verifier context wire.  The wire contains the
// complete public statement (graph/features/Y, quantization/static tables,
// model-registration commitment identity, and the structured compact
// canonical circuit recipe), but never model tensors, checkpoints, witness
// values, materialized gate arrays, or prover opening randomness.  Both
// functions reject non-canonical/truncated/
// trailing/oversized data and recompute the statement digest on read.
// This must be called before default-constructing a VerifierContext in a
// standalone verifier process because the public registration type contains
// an mcl field used only by the prover-side private state.
void initialize_gat_field();
bool write_verifier_context(const VerifierContext &context,
                            const std::string &path,
                            std::string *reason = nullptr);
bool read_verifier_context(const std::string &path,
                           VerifierContext &context,
                           std::string *reason = nullptr);

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

// Bounded adapter-only bridge for exercising the native GKR transcript.  It
// deliberately produces a diagnostic artifact, not a formal GAT proof:
// Lasso, MatMul aggregation, terminal Hyrax openings, and masked_full
// composition are not asserted by this helper.  The public context, output
// vector, and dynamic commitment wires are still authenticated in the native
// statement and in the typed session so serialization and tamper checks use
// the same wire that the production path will consume.
struct GkrOnlyArtifactResult {
    bool generated = false;
    std::string diagnostic;
    GatProofArtifact artifact;
};

GkrOnlyArtifactResult prove_gkr_only_artifact(
    prover &native_prover, const VerifierContext &context,
    const CommitmentWire &com_aux, const CommitmentWire &com_L,
    const G1 &G, const std::vector<G1> &generators,
    const std::vector<Fr> &basis, std::size_t l_dimension,
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls);

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

// A baseline-local, protocol-native core proof input.  It is deliberately
// generic so a bounded smoke can exercise the exact masked GKR/MatMul and
// Hyrax transcript grammar without loading a dataset.  The formal GAT
// adapter fills these vectors from its canonical circuit/trace; Lasso remains
// a separate relation family and is not silently replaced here.
struct MaskedCoreWitness {
    std::vector<Fr> public_input;
    std::vector<Fr> w;
    std::vector<Fr> aux;
    std::vector<Fr> l_qry;
    std::vector<Fr> public_output;
    std::vector<std::vector<Fr>> gkr_round_polys;
    std::vector<Fr> gkr_terminal_values;
    std::vector<Fr> gkr_coefficients;
    std::vector<std::vector<Fr>> matmul_round_polys;
    std::vector<Fr> matmul_terminal_values;
    // Optional pre-generated dynamic commitments.  Formal callers use these
    // to bind the statement digest before constructing the transcript; the
    // fallback is useful only for bounded diagnostic callers.
    ProverCommitment com_aux;
    ProverCommitment com_L;
    bool has_dynamic_commitments = false;
};

struct MaskedCoreResult {
    bool generated = false;
    std::string diagnostic;
    GatProofArtifact artifact;
    CommitmentWire com_aux;
    CommitmentWire com_L;
};

// Generate and authenticate the baseline-native masked core transcript.  All
// three private segments must use the same power-of-two MLE domain and the
// same transcript-derived evaluation point.  `com_W_blinder` is prover-only;
// no opening randomness or raw witness enters the returned artifact.
MaskedCoreResult prove_masked_core(
    const NativeProofStatement &statement, const CommitmentWire &com_W,
    const Fr &com_W_blinder, const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators, const MaskedCoreWitness &witness,
    const std::string &statement_digest);

// Same core bridge with the native masked Lasso transcript as the first
// canonical section.  The family matrices/query remain prover-side inputs;
// the resulting artifact contains only the native masked Lasso messages.
MaskedCoreResult prove_masked_core_with_lasso(
    const NativeProofStatement &statement, const CommitmentWire &com_W,
    const Fr &com_W_blinder, const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators, const MaskedCoreWitness &witness,
    std::vector<NativeLassoFamilyProver> &families,
    NativeLassoQueryProver &query, const std::string &statement_digest);

// Independent core verifier.  It consumes only public output/claim
// descriptors, commitments, the artifact and public generators.  It never
// receives W/Aux/L or a blinding scalar.  The Lasso family is intentionally
// left to native_lasso_verify_bundle by the formal adapter.
bool verify_masked_core(
    const GatProofArtifact &artifact, const CommitmentWire &com_W,
    const std::string &basis_id, const G1 &H,
    const std::vector<G1> &generators,
    const std::vector<Fr> &public_input,
    const std::vector<Fr> &public_output,
    const std::vector<Fr> &gkr_coefficients,
    std::size_t gkr_rounds, const Fr &matmul_initial_claim,
    std::size_t matmul_rounds, std::string *reason = nullptr);

// Short, bounded bridge used by tests and diagnostics.  It executes the
// official prover/range/Hyrax primitives on a one-gate relation; it is not a
// formal GAT proof and cannot produce a production artifact.
NativePrimitiveEvidence exercise_official_primitives(const std::string &label);

} // namespace gat
} // namespace zkgpt
