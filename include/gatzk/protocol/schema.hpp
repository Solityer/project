#pragma once
// 本文件维护正式 proof schema 使用的标签命名规则与 block 顺序，
// 是 trace、prove、verify 与 artifact 共享的命名真源。

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

struct ProtocolContext;
struct BindingRelationMeta;
struct KZGShiftRelationMeta;
struct MetadataReference;

enum class PolynomialVisibility {
    Public,
    PublicOutput,
    PrivateWitness,
};

enum class EvaluationKind {
    PublicValue,
    MaskedPrivateValue,
};

struct ProofSchemaEntry {
    std::string tag;
    std::string domain;
    std::string owner_block;
    std::string field_type;
    std::string padding_mode;
    std::string owner_policy;
    PolynomialVisibility visibility = PolynomialVisibility::PrivateWitness;
    std::size_t layer_id = 0;
    std::size_t head_id = 0;
    std::string opening_policy;
    std::string binding_mode;
    std::string blinding_basis;
    bool enters_commitment = false;
    bool enters_opening = false;
    bool enters_quotient = false;
    bool operator==(const ProofSchemaEntry&) const = default;
};

struct ProofSchema {
    std::vector<ProofSchemaEntry> entries;
    std::size_t private_plain_value_count = 0;
    bool operator==(const ProofSchema&) const = default;
};

struct CompilerLedgerEntry {
    std::string id;
    std::string object_or_challenge;
    std::string domain;
    std::string transcript_stage;
    std::string expression_or_certificate;
    std::size_t degree_bound = 0;
    std::size_t query_bound = 0;
    std::size_t exclusion_degree = 0;
    std::size_t multiplicity = 1;
    bool operator==(const CompilerLedgerEntry&) const = default;
};

// A derived private observation never owns fresh randomness merely because it
// is serialized.  These edges are the canonical expression-DAG links used by
// the ObservationLedger compiler to trace every derived field back to the
// independent base objects that actually own blinder coordinates.
struct ObservationDependencyEdge {
    std::string id;
    std::string derived_object;
    std::vector<std::string> source_objects;
    std::vector<std::string> serialized_observations;
    std::string transcript_stage;
    std::string propagation_rule;
    bool introduces_independent_randomness = false;
    bool operator==(const ObservationDependencyEdge&) const = default;
};

struct CoefficientChunkLayoutEntry {
    std::size_t chunk_index = 0;
    std::size_t offset = 0;
    std::size_t length = 0;
    bool operator==(const CoefficientChunkLayoutEntry&) const = default;
};

struct CoefficientRegionLayout {
    std::size_t low_degree_length = 0;
    std::size_t tail_length = 0;
    std::size_t total_length = 0;
    std::vector<std::size_t> semantic_positions;
    std::vector<std::uint8_t> q_sem;
    std::vector<std::uint8_t> q_gap;
    std::vector<std::uint8_t> q_tail;
    std::vector<CoefficientChunkLayoutEntry> chunks;
    bool operator==(const CoefficientRegionLayout&) const = default;
};

struct ProtocolCompilerArtifact {
    ProofSchema proof_schema;
    // Canonical, length-delimited RelationMeta records for the exact
    // instance.  This is the authenticated source used by the proof codec and
    // transcript; labels in the executable alone are not an artifact.
    std::vector<std::string> binding_relation_registry;
    std::vector<std::string> proof_blocks;
    std::vector<std::string> serializer_tag_table;
    std::vector<std::string> object_canonical_order;
    std::vector<std::string> transcript_stages;
    std::vector<std::string> verifier_equation_registry;
    std::vector<std::string> ledger_templates;
    std::vector<std::string> constraint_registry;
    std::vector<std::string> point_group_registry_template;
    std::vector<std::string> degree_registry;
    std::vector<std::string> padding_registry;
    std::vector<ObservationDependencyEdge> observation_dependency_dag;
    std::vector<CompilerLedgerEntry> observation_ledger;
    std::vector<CompilerLedgerEntry> random_identity_ledger;
    std::vector<CompilerLedgerEntry> security_budget_ledger;
    std::string kzg_assumption_profile;
    std::string shift_algorithm_id;
    std::string shift_algorithm_spec;
    std::string protocol_canonical_bytes;
    std::string instance_canonical_bytes;
};

// 最终 proof 的 block 顺序，任何改变都会影响 serializer / parser 与 verifier。
// 它也决定哪些标签会进入 proof 的正式传输形状，因此间接影响 proof_size_bytes。
const std::vector<std::string>& proof_block_order();
// Exact protocol-template bytes consumed by the pre-ceremony SRS compiler.
// This object is independent of any dataset, witness, SRS trapdoor or exact
// instance dimensions.
std::string protocol_template_canonical_bytes();
std::string protocol_proof_schema_template_canonical_bytes();
std::string protocol_observation_ledger_template_canonical_bytes();
std::string protocol_random_identity_ledger_template_canonical_bytes();
std::string protocol_security_budget_ledger_template_canonical_bytes();
ProofSchema build_proof_schema(const ProtocolContext& context);
ProtocolCompilerArtifact build_protocol_compiler_artifact(const ProtocolContext& context);
std::vector<BindingRelationMeta> build_binding_relation_metadata(const ProtocolContext& context);
// Canonical S_route payload.  Every entry is a reference to the unique
// standard point-group owner in Open_dom/W_ext; no route terminal value is
// ever materialized in the proof object.
std::vector<MetadataReference> route_opening_references(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges);

std::string opening_group_reference(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const algebra::FieldElement& point);
const KZGShiftRelationMeta* find_unique_shift_relation_by_kind(
    const BindingRelationMeta& metadata,
    std::string_view relation_kind);
bool validate_protocol_compiler_artifact(
    const ProtocolContext& context,
    const ProtocolCompilerArtifact& artifact,
    std::string* reason = nullptr);
bool instance_within_authenticated_max_envelope(
    const ProtocolContext& context,
    std::string* reason = nullptr);
// Canonical coefficient carrier layout from section 0.7: every chunk lives on
// H_edge, has capacity n_edge-1, and reserves the final row for its terminal.
std::vector<CoefficientChunkLayoutEntry> coefficient_chunk_layout(
    std::size_t total_coefficient_length,
    std::size_t edge_domain_size);

// The edge work domain also carries row-major hidden/output quantization
// traces, so topology edge count alone is not a sufficient lower bound.
std::size_t minimum_edge_trace_length(
    std::size_t node_count,
    std::size_t edge_count,
    std::size_t hidden_width,
    std::size_t class_count);
// Canonical public partition for a coefficient-tail object.  semantic
// positions are strict increasing low-degree offsets; all remaining low
// positions are gaps and the unique suffix is the random tail.
CoefficientRegionLayout coefficient_region_layout(
    std::size_t low_degree_length,
    const std::vector<std::size_t>& semantic_positions,
    std::size_t tail_length,
    std::size_t edge_domain_size);
std::vector<algebra::FieldElement> materialize_coefficient_tail(
    const CoefficientRegionLayout& layout,
    const std::vector<algebra::FieldElement>& low_degree_coefficients,
    const std::vector<algebra::FieldElement>& tail_randomness);
// Returns the compiler-registered prefix length q_O^max for the independent
// blinder owned by object_id.  Formal prover code must not choose this length
// from the witness or from a local constant.
std::size_t registered_blinder_coefficient_count(
    const ProtocolContext& context,
    const std::string& object_id);
// Exact finite-field image-inclusion gate used by compiler certificate tests.
// Rows are serialized observations and columns are secret/blinder variables.
bool observation_image_is_covered(
    const std::vector<std::vector<algebra::FieldElement>>& blind,
    const std::vector<std::vector<algebra::FieldElement>>& secret);
// Single source of truth for payload visibility. In particular, per-head output
// objects (for example `P_Y_o0`) remain private even when final `P_Y` is public.
PolynomialVisibility polynomial_visibility(const ProtocolContext& context, const std::string& label);
bool schema_accepts_evaluation(PolynomialVisibility visibility, EvaluationKind kind);
// 下面这些函数负责生成隐藏层和输出层使用的正式标签。
// 它们共同决定 dynamic commitments、opening bundle 与 external evaluation 的命名真源。
std::string hidden_head_prefix(std::size_t global_head_index);
std::string hidden_layer_concat_label(std::size_t layer_index, bool is_final_layer);
std::string hidden_layer_concat_star_label(std::size_t layer_index, bool is_final_layer);
std::string hidden_layer_cat_prefix(std::size_t layer_index, bool is_final_layer);
std::string hidden_weight_label(std::size_t global_head_index);
std::string hidden_weight_column_label(std::size_t global_head_index, std::size_t column_index);
std::string hidden_src_label(std::size_t global_head_index);
std::string hidden_dst_label(std::size_t global_head_index);
std::string output_head_prefix(std::size_t head_index);
std::string output_y_lin_label(std::size_t head_index);
std::string output_y_label(std::size_t head_index);
std::string canonical_matrix_column_label(
    const std::string& matrix_object,
    std::size_t column_index);
std::string hidden_aggregation_binding_label(
    std::size_t global_head_index,
    std::string_view suffix);
std::string output_aggregation_binding_label(
    std::size_t head_index,
    std::string_view suffix);
std::string output_weight_label(std::size_t head_index);
std::string output_weight_column_label(std::size_t head_index, std::size_t column_index);
std::string output_src_label(std::size_t head_index);
std::string output_dst_label(std::size_t head_index);
std::string output_bias_label(std::size_t head_index);
std::string output_challenge_name(std::string_view base, std::size_t head_index);
std::string hidden_model_binding_blinder_label(std::size_t head_index, std::string_view role);
std::string output_model_binding_blinder_label(std::size_t head_index, std::string_view role);
std::string hidden_score_mask_label(std::size_t head_index, std::string_view role);
std::string output_score_mask_label(std::size_t head_index, std::string_view role);
std::string hidden_score_mask_eval_name(std::size_t head_index, std::string_view role);
std::string output_score_mask_eval_name(std::size_t head_index, std::string_view role);
// A private evaluation-form object is committed as P_sem + Z_D R.  Whenever
// an off-domain opening is used as a semantic accumulator target, R is exposed
// through this derived commitment and opened at the same challenge point.
std::string external_mask_label(std::string_view source_label);
std::string external_mask_eval_name(std::string_view source_eval_name);
std::string model_proof_local_label(std::string_view vk_model_anchor_label);
std::string model_reblind_witness_label(std::string_view vk_model_anchor_label);
std::string output_bias_view_prefix(std::size_t head_index);
std::string output_bias_view_label(std::size_t head_index, std::string_view suffix);
std::string output_bias_view_eta_label(std::size_t head_index);
std::string output_bias_view_beta_label(std::size_t head_index);
std::string output_bias_view_relation_challenge_label(std::size_t head_index);
std::string output_bias_basis_prefix(std::size_t head_index);
std::string output_bias_basis_label(
    std::size_t head_index, std::string_view suffix);
std::string output_bias_basis_chunk_label(
    std::size_t head_index, std::size_t chunk_index);
std::string output_bias_basis_chunk_accumulator_label(
    std::size_t head_index, std::size_t chunk_index);
std::string output_bias_basis_chunk_shifted_accumulator_label(
    std::size_t head_index, std::size_t chunk_index);
std::string output_bias_basis_region_selector_label(
    std::size_t head_index, std::size_t chunk_index, std::string_view region);
std::string output_bias_basis_gamma_label(std::size_t head_index);
std::string output_bias_basis_relation_challenge_label(std::size_t head_index);

// Canonical SignedRoundDiv limb/range family naming.  These helpers are shared
// by schema expansion, trace construction and transcript replay.
std::size_t signed_division_limb_count(std::size_t range_bits);
std::string signed_division_limb_label(
    std::string_view division_prefix,
    std::string_view value_name,
    std::size_t limb_index);
std::string signed_division_range_label(
    std::string_view division_prefix,
    std::string_view value_name,
    std::size_t limb_index,
    std::string_view column_name);
std::string signed_division_beta_name(
    std::string_view division_prefix,
    std::string_view value_name,
    std::size_t limb_index);
std::string output_external_eval_name(std::string_view base, std::size_t head_index);
std::string hidden_concat_xi_name(std::size_t layer_index, bool is_final_layer);
std::string hidden_concat_y_name(std::size_t layer_index, bool is_final_layer);

}  // namespace gatzk::protocol
