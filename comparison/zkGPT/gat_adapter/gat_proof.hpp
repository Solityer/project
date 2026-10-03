#pragma once

#include "gat_types.hpp"
#include "native_storage.hpp"
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

std::string canonical_digest64(const std::string& bytes);

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

class SegmentedInputOracle {
  public:
    using ChunkCallback = std::function<void(std::size_t, const Fr&)>;
    using FoldChunkCallback = std::function<bool(std::uint64_t, const std::vector<Fr>&, std::string* reason)>;
    using SegmentValueProvider = std::function<Fr(std::uint64_t)>;

    SegmentedInputOracle() = default;

    SegmentedInputOracle(const InputLayout& layout, std::array<std::vector<Fr>, 4> segments);
    SegmentedInputOracle(const InputLayout& layout, std::array<zkgpt::NativeFieldVector, 4> segments,
                         std::array<SegmentValueProvider, 4> providers = {});

    static SegmentedInputOracle from_int_segments(const InputLayout& layout, const std::array<std::vector<Int>, 4>& segments);

    bool valid(std::string* reason = nullptr) const;
    const InputLayout& layout() const { return layout_; }

    Fr value_at(std::size_t global_coordinate) const;
    Fr segment_value(std::size_t segment, std::size_t local_coordinate) const;

    void stream(std::size_t segment, std::size_t begin, std::size_t count, const ChunkCallback& callback) const;
    std::vector<Fr> chunk(std::size_t segment, std::size_t begin, std::size_t count) const;

    Fr evaluate_mle(std::size_t segment, const std::vector<Fr>& point) const;
    Fr evaluate_full_mle(const std::vector<Fr>& point) const;
    Fr mle_eval(std::size_t segment, const std::vector<Fr>& point) const { return evaluate_mle(segment, point); }

    std::vector<Fr> fold(std::size_t segment, const std::vector<Fr>& prefix) const;

    bool fold_chunks(std::size_t segment, const std::vector<Fr>& prefix, std::uint64_t chunk_entries, const FoldChunkCallback& callback,
                     std::string* reason = nullptr) const;

    Fr weighted_sum(std::size_t segment, const std::function<Fr(std::size_t)>& weight) const;

  private:
    InputLayout layout_;
    std::array<zkgpt::NativeFieldVector, 4> segments_;
    std::array<SegmentValueProvider, 4> providers_;
};

InputLayout build_gat_input_layout(const Graph& graph, const Model& model, const QuantConfig& quant);

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

    std::string relation_schema = "zkgpt-gat-relation-v1";
    std::string lhs_segment = "X_pub";
    std::string rhs_segment = "W";
    std::string output_segment = "Aux";
    std::string lhs_coordinate_rule = "segment_offset+row*inner+term";
    std::string rhs_coordinate_rule = "segment_offset+term*cols+col";
    std::string output_coordinate_rule = "segment_offset+row*cols+col";
    std::string terminal_claim_rule = "descriptor_index_ordered_mle_points_v1";
};

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

zkgpt::NativeFieldVector pack_native_gkr_matmul_rhs(const MatMulDescriptor& descriptor, const zkgpt::NativeFieldVector& canonical_input_layer);
zkgpt::NativeFieldVector pack_native_gkr_matmul_rhs(
    const MatMulDescriptor& descriptor, std::uint64_t canonical_input_size,
    const std::function<Fr(std::uint64_t)>& canonical_input_value_at);

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
    std::uint64_t masked_lasso_matrix_entries = 0;
    std::uint64_t masked_response_entries_lower_bound = 0;
    std::uint64_t masked_proof_bytes_lower_bound = 0;
    std::uint64_t maximum_masked_response_entries = 0;
    std::vector<std::array<std::uint64_t, 2>> lasso_family_domains;
    std::string proof_path = "unsupported";
    bool full_relation_bound = false;
    bool ready = false;
    std::string reason;
};

GatCapacityReport inspect_gat_capacity(const Graph& graph, const Model& model, const QuantConfig& quant);

struct GatCircuitPlan {
    std::string recipe_schema = "canonical_plan_recipe_v3";
    std::string relation_schema = "zkgpt-gat-relation-v1";

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

    std::size_t public_output_coordinates = 0;
    std::size_t residual_coordinates = 0;
    zkgpt::NativeVector<Int> residual_targets;
    std::string residual_target_digest;
    std::vector<std::array<unsigned, 5>> division_range_bits;
    bool official_layered = false;
    bool has_placeholder = false;
};

GatCircuitPlan build_gat_circuit_plan(const Graph& graph, const Model& model, const QuantConfig& quant, const ForwardResult& forward);

GatCircuitPlan build_gat_circuit_plan(const Graph& graph, const Model& model, const QuantConfig& quant);

bool validate_gat_circuit_plan(const GatCircuitPlan& plan, std::string* reason = nullptr);

struct GatCircuitWitness {
    zkgpt::NativeFieldVector input_layer;
    std::uint64_t input_layer_size = 0;
    std::vector<zkgpt::NativeFieldVector> values;
    zkgpt::NativeFieldVector public_output;
    zkgpt::NativeFieldVector residuals;

    std::shared_ptr<SegmentedInputOracle> input_oracle;
    bool relation_ok = false;
};

GatCircuitWitness build_gat_circuit_witness(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features, const Model& model,
                                            const QuantConfig& quant, const ForwardResult& forward);
GatCircuitWitness build_gat_circuit_witness(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features, const Model& model,
                                            const QuantConfig& quant, const ForwardResult& forward,
                                            const hyrax::HyraxScalarProvider& l_query_values);

bool evaluate_gat_circuit_witness(const GatCircuitPlan& plan, const GatCircuitWitness& witness, std::string* reason = nullptr);
bool evaluate_signed_round_div_gate_assignment(Int dividend, Int denominator, const DivisionResult& result, RoundingMode mode, std::string* reason = nullptr);

bool evaluate_gat_circuit_assignment(const GatCircuitPlan& plan, const zkgpt::NativeFieldVector& input, const Tensor2D& public_logits, std::string* reason = nullptr);

struct CommitmentWire {
    std::string scheme = "hyrax-pedersen-bn254";
    std::string basis_id;
    std::uint64_t logical_size = 0;
    std::uint64_t padded_size = 0;
    std::string point_hex;
    bool has_tensor_commitment = false;
    hyrax::HyraxTensorMleCommitment tensor_commitment;

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

    Fr com_W_padding_mask = Fr(0);
    bool has_private_padding_mask = false;
};

ModelRegistration register_model(const Model& model, const QuantConfig& quant, const std::string& model_arch_id = "gat_l3_8x2");

ModelRegistration register_model(const Model& model, const QuantConfig& quant, const InputLayout& layout, const std::string& model_arch_id = "gat_l3_8x2");

bool registration_matches_model(const Model& model, const QuantConfig& quant, const InputLayout& layout, const ModelRegistration& registration);

CommitmentWire commit_private_segment(const std::vector<Int>& values, const std::string& basis_label);
CommitmentWire commit_private_segment(const std::vector<Int>& values, const std::string& basis_label, std::size_t padded_domain);

struct ProverCommitment {
    CommitmentWire wire;
    Fr blinder = Fr(0);
};

ProverCommitment commit_private_segment_for_prover(const std::vector<Int>& values, const std::string& basis_label, std::size_t padded_domain);

bool write_prover_commitment_secret(const ModelRegistration& registration, const std::string& path, std::string* reason = nullptr);
bool read_prover_commitment_secret(const ModelRegistration& registration, const std::string& path, Fr& blinder, std::string* reason = nullptr);

struct TranscriptMessage {
    std::string label;
    std::string payload;

    std::uint8_t kind = 1;
    std::uint64_t sequence = 0;
};

class FiatShamirTranscript {
  public:
    explicit FiatShamirTranscript(const std::string& domain = "zkgpt-gat-fs-v1");

    void absorb(const std::string& label, const std::string& payload);
    Field challenge(const std::string& label);
    std::string digest() const;
    const std::vector<TranscriptMessage>& messages() const { return messages_; }

  private:
    std::string domain_;
    std::vector<TranscriptMessage> messages_;
};

struct ProofSection {
    std::string protocol;
    std::vector<TranscriptMessage> messages;
};

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

struct GatProofArtifact {
    std::uint32_t version = 8;
    std::string schema = "zkgpt-gat-proof-v8";
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

    std::string native_domain = "zkgpt-native-gat-fs-v6";
    NativeProofStatement native_statement;
    std::vector<NativeProofMessage> native_messages;
    std::string transcript_digest;
    SparseGkrProofWire sparse_gkr;

    bool has_private_payload() const;
    bool valid_wire(std::string* reason = nullptr, const NativeProofSession* authenticated_session = nullptr) const;
    bool native_session(NativeProofSession& out, std::string* reason = nullptr) const;
    bool write(const std::string& path, std::string* reason = nullptr) const;
    bool write(const std::string& path, std::uint64_t max_wire_bytes,
               std::string* reason = nullptr) const;
    static bool read(const std::string& path, GatProofArtifact& out, std::string* reason = nullptr);
    static bool read(const std::string& path, GatProofArtifact& out,
                     std::uint64_t max_wire_bytes,
                     std::string* reason = nullptr);
};

struct VerifierContext {
    std::string relation_id = "R_GAT-zkGPT";
    std::string backend;
    std::string model_arch_id = "gat_l3_8x2";
    std::string privacy_extension_profile_id;
    std::string quant_cfg_id;
    std::string static_tables_id;

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

VerifierContext make_verifier_context(const Graph& graph, const Tensor2D& public_features, const QuantConfig& quant, const StaticTables& tables,
                                      const Tensor2D& public_logits, const CommitmentWire& com_W, const GatCircuitPlan& plan,
                                      const std::string& model_arch_id = "gat_l3_8x2", const std::string& privacy_profile = "masked_full");

VerifierContext make_verifier_context(const Graph& graph, const Tensor2D& public_features, const QuantConfig& quant, const StaticTables& tables,
                                      const Tensor2D& public_logits, const ModelRegistration& registration, const GatCircuitPlan& plan,
                                      const std::string& privacy_profile = "masked_full");

std::string verifier_context_digest(const VerifierContext& context);
std::uint64_t gat_proof_artifact_wire_limit(const VerifierContext& context,
                                            std::string* reason = nullptr);

std::size_t compact_plan_recipe_bytes(const GatCircuitPlan& plan);
std::size_t verifier_context_payload_bytes(const VerifierContext& context);

void initialize_gat_field();
bool write_verifier_context(const VerifierContext& context, const std::string& path, std::string* reason = nullptr);
bool read_verifier_context(const std::string& path, VerifierContext& context, std::string* reason = nullptr);

NativeProofStatement make_native_proof_statement(const VerifierContext& context, const CommitmentWire& com_aux, const CommitmentWire& com_L, const G1& G,
                                                 const std::vector<G1>& generators, const std::vector<Fr>& basis, std::size_t l_dimension);
NativeProofStatement make_native_proof_statement(const VerifierContext& context, const CommitmentWire& com_aux, const CommitmentWire& com_L, const G1& H,
                                                 const std::string& canonical_basis_label, const std::string& canonical_basis_id, std::uint64_t full_domain);

enum class ProveStatus { GENERATED, UNSUPPORTED_PRIVACY_PROFILE, REJECTED };

const char* prove_status_name(ProveStatus status);

struct ProveResult {
    ProveStatus status = ProveStatus::REJECTED;
    std::string diagnostic;
    bool has_artifact = false;
    double commitment_ms = 0.0;
    double self_verify_ms = 0.0;
    GatProofArtifact artifact;
};

struct GkrOnlyArtifactResult {
    bool generated = false;
    std::string diagnostic;
    GatProofArtifact artifact;
};

GkrOnlyArtifactResult prove_gkr_only_artifact(prover& native_prover, const VerifierContext& context, const CommitmentWire& com_aux, const CommitmentWire& com_L,
                                              const G1& G, const std::vector<G1>& generators, const std::vector<Fr>& basis, std::size_t l_dimension,
                                              const std::vector<F>& public_output, const std::vector<NativeGkrMatmulShape>& matmuls);

ProveResult prove_gat(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features, const Model& model, const QuantConfig& quant,
                      const ForwardResult& forward, const CommitmentWire& com_W, const std::string& privacy_profile = "masked_full");

ProveResult prove_gat(const GatCircuitPlan& plan, const Graph& graph, const Tensor2D& public_features, const Model& model, const QuantConfig& quant,
                      const ForwardResult& forward, const ModelRegistration& registration, const std::string& privacy_profile = "masked_full");

bool verify_gat(const VerifierContext& context, const GatProofArtifact& artifact, std::string* reason = nullptr);

struct MaskingSession {
    std::string profile_id;
    std::string session_nonce;
    bool complete = false;
};

MaskingSession begin_masking_session(const std::string& profile_id, const std::string& statement_digest);

struct NativePrimitiveEvidence {
    bool sumcheck_called = false;
    bool gkr_path_bound = false;
    bool lasso_range_called = false;
    bool hyrax_commit_called = false;
    std::vector<std::string> messages;
};

struct MaskedCoreWitness {
    std::vector<Fr> public_input;
    std::vector<Fr> w;
    std::vector<Fr> aux;
    std::vector<Fr> l_qry;
    zkgpt::NativeFieldVector public_output;
    std::vector<std::vector<Fr>> gkr_round_polys;
    std::vector<Fr> gkr_terminal_values;
    std::vector<Fr> gkr_coefficients;
    std::vector<std::vector<Fr>> matmul_round_polys;
    std::vector<Fr> matmul_terminal_values;

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

MaskedCoreResult prove_masked_core(const NativeProofStatement& statement, const CommitmentWire& com_W, const Fr& com_W_blinder, const std::string& basis_id,
                                   const G1& H, const std::vector<G1>& generators, const MaskedCoreWitness& witness, const std::string& statement_digest);

MaskedCoreResult prove_masked_core_with_lasso(const NativeProofStatement& statement, const CommitmentWire& com_W, const Fr& com_W_blinder,
                                              const std::string& basis_id, const G1& H, const std::vector<G1>& generators, const MaskedCoreWitness& witness,
                                              std::vector<NativeLassoFamilyProver>& families, NativeLassoQueryProver& query,
                                              const std::string& statement_digest);

bool verify_masked_core(const GatProofArtifact& artifact, const CommitmentWire& com_W, const std::string& basis_id, const G1& H,
                        const std::vector<G1>& generators, const std::vector<Fr>& public_input, const std::vector<Fr>& public_output,
                        const std::vector<Fr>& gkr_coefficients, std::size_t gkr_rounds, const Fr& matmul_initial_claim, std::size_t matmul_rounds,
                        std::string* reason = nullptr);

NativePrimitiveEvidence exercise_official_primitives(const std::string& label);

}
}
