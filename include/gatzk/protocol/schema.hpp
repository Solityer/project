#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::protocol
{

    struct ProtocolContext;
    struct VerifierContext;
    struct StaticTables;
    struct ProtocolCompilerArtifact;
    struct BindingRelationMeta;
    struct KZGShiftRelationMeta;
    struct MetadataReference;

    struct CanonicalStaticTableShape
    {
        std::size_t range_table_size = 0;
        std::int64_t lrelu_min_raw = 0;
        std::int64_t lrelu_max_raw = 0;
        std::size_t lrelu_table_size = 0;
        std::size_t elu_bound = 0;
        std::size_t elu_table_size = 0;
        std::uint64_t lrelu_output_bound = 0;
        std::uint64_t exp_input_bound = 0;
        std::size_t exp_table_size = 0;
        std::size_t table_edge_domain_size = 0;
        bool large_whole_graph = false;
        bool operator==(const CanonicalStaticTableShape &) const = default;
    };

    struct CanonicalStaticTableShapeInputs
    {
        std::size_t range_bits = 0;
        util::QuantizationScaleChain quantization;
        std::string batching_rule;
        std::size_t dataset_num_nodes = 0;
        std::size_t dataset_num_features = 0;
        std::size_t local_num_nodes = 0;
        std::size_t local_edge_count = 0;
    };

    enum class PolynomialVisibility
    {
        Public,
        PublicOutput,
        PrivateWitness,
    };

    enum class EvaluationKind
    {
        PublicValue,
        MaskedPrivateValue,
    };

    struct ProofSchemaEntry
    {
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
        bool operator==(const ProofSchemaEntry &) const = default;
    };

    struct ProofSchema
    {
        std::vector<ProofSchemaEntry> entries;
        std::size_t private_plain_value_count = 0;
        bool operator==(const ProofSchema &) const = default;
    };

    struct CompilerLedgerEntry
    {
        std::string id;
        std::string object_or_challenge;
        std::string domain;
        std::string transcript_stage;
        std::string expression_or_certificate;
        std::size_t degree_bound = 0;
        std::size_t query_bound = 0;
        std::size_t exclusion_degree = 0;
        std::size_t multiplicity = 1;
        bool operator==(const CompilerLedgerEntry &) const = default;
    };

    struct SecurityBudgetEvaluation
    {
        bool passed = false;
        std::string ledger_digest;
        std::string exact_budget;
        bool operator==(const SecurityBudgetEvaluation &) const = default;
    };

    struct IntegerBoundLedgerEntry
    {
        std::string id;
        std::string stage;
        std::string source;
        std::uint64_t absolute_bound = 0;
        std::size_t limb_count = 0;
        bool operator==(const IntegerBoundLedgerEntry &) const = default;
    };

    struct ObservationDependencyEdge
    {
        std::string id;
        std::string derived_object;
        std::vector<std::string> source_objects;
        std::vector<std::string> serialized_observations;
        std::string transcript_stage;
        std::string propagation_rule;
        bool introduces_independent_randomness = false;
        bool operator==(const ObservationDependencyEdge &) const = default;
    };

    struct CoefficientChunkLayoutEntry
    {
        std::size_t chunk_index = 0;
        std::size_t offset = 0;
        std::size_t length = 0;
        bool operator==(const CoefficientChunkLayoutEntry &) const = default;
    };

    struct CoefficientRegionLayout
    {
        std::size_t low_degree_length = 0;
        std::size_t tail_length = 0;
        std::size_t total_length = 0;
        std::vector<std::size_t> semantic_positions;
        std::vector<std::uint8_t> q_sem;
        std::vector<std::uint8_t> q_gap;
        std::vector<std::uint8_t> q_tail;
        std::vector<CoefficientChunkLayoutEntry> chunks;
        bool operator==(const CoefficientRegionLayout &) const = default;
    };
    struct MatrixFoldBindingSpec
    {
        std::string relation_id;
        std::string owner_relation_id;
        std::string source_matrix_object;
        std::string fold_object;
        std::string source_fold_object;
        std::string source_accumulator_object;
        std::string fold_accumulator_object;
        std::string source_point_challenge;
        std::string lambda_challenge;
        std::string fold_domain;
        std::string fold_selector_label;
        std::size_t source_rows = 0;
        std::size_t width = 0;
        bool public_source = false;
        bool operator==(const MatrixFoldBindingSpec &) const = default;
    };

    struct FlatMatrixBindingSpec
    {
        std::string relation_id;
        std::string owner_relation_id;
        std::string matrix_object;
        std::string flat_object;
        std::string row_fold_object;
        std::string row_accumulator_object;
        std::string flat_accumulator_object;
        std::string lambda_challenge;
        std::string flat_selector_label;
        std::size_t rows = 0;
        std::size_t columns = 0;
        bool operator==(const FlatMatrixBindingSpec &) const = default;
    };

    struct BroadcastFlatBindingSpec
    {
        std::string relation_id;
        std::string owner_relation_id;
        std::string node_object;
        std::string flat_object;
        std::string node_accumulator_object;
        std::string flat_accumulator_object;
        std::string lambda_challenge;
        std::string flat_selector_label;
        std::size_t rows = 0;
        std::size_t width = 0;
        bool operator==(const BroadcastFlatBindingSpec &) const = default;
    };

    struct ProtocolCompilerArtifact
    {
        ProofSchema proof_schema;
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
        std::vector<IntegerBoundLedgerEntry> integer_bound_ledger;
        std::string kzg_assumption_profile;
        std::string shift_algorithm_id;
        std::string shift_algorithm_spec;
        std::string protocol_canonical_bytes;
        std::string instance_canonical_bytes;
    };

    const std::vector<std::string> &proof_block_order();
    std::string protocol_template_canonical_bytes();
    std::string protocol_proof_schema_template_canonical_bytes();
    std::string protocol_observation_ledger_template_canonical_bytes();
    std::string protocol_random_identity_ledger_template_canonical_bytes();
    std::string protocol_security_budget_ledger_template_canonical_bytes();
    std::string integer_bound_ledger_canonical_bytes(const std::vector<IntegerBoundLedgerEntry> &ledger);
    std::vector<IntegerBoundLedgerEntry> rebuild_integer_bound_ledger(const VerifierContext &context);
    CanonicalStaticTableShape plan_canonical_static_table_shape(const ProtocolContext &context);
    CanonicalStaticTableShape plan_canonical_static_table_shape(const VerifierContext &context);
    CanonicalStaticTableShape plan_canonical_static_table_shape(const CanonicalStaticTableShapeInputs &inputs);
    std::size_t canonical_exp_table_size_from_delta_bound(std::uint64_t delta_bound);
    std::string security_budget_ledger_canonical_bytes(const std::vector<CompilerLedgerEntry> &ledger);
    SecurityBudgetEvaluation evaluate_security_budget(const ProtocolContext &context, const ProtocolCompilerArtifact &artifact);
    bool validate_static_tables(const StaticTables &tables, std::size_t range_bits, std::string *reason = nullptr);
    StaticTables build_canonical_static_tables(const CanonicalStaticTableShapeInputs &inputs);
    StaticTables build_canonical_static_tables(const ProtocolContext &context);
    bool validate_static_tables(const ProtocolContext &context, std::string *reason = nullptr);
    std::string protocol_instance_canonical_bytes(const ProtocolCompilerArtifact &artifact);
    ProofSchema build_proof_schema(const ProtocolContext &context);
    ProtocolCompilerArtifact build_protocol_compiler_artifact(const ProtocolContext &context);
    std::vector<BindingRelationMeta> build_binding_relation_metadata(const ProtocolContext &context);
    std::vector<BindingRelationMeta> build_binding_relation_metadata(const ProtocolContext &context, const ProtocolCompilerArtifact &artifact);
    std::vector<MatrixFoldBindingSpec> matrix_fold_binding_specs(const ProtocolContext &context);

    std::vector<FlatMatrixBindingSpec> flat_matrix_binding_specs(const ProtocolContext &context);

    std::vector<BroadcastFlatBindingSpec> broadcast_flat_binding_specs(const ProtocolContext &context);

    std::vector<MetadataReference> route_opening_references(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges);

    std::string opening_group_reference(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                        const algebra::FieldElement &point);
    const KZGShiftRelationMeta *find_unique_shift_relation_by_kind(const BindingRelationMeta &metadata, std::string_view relation_kind);
    bool validate_protocol_compiler_artifact(const ProtocolContext &context, const ProtocolCompilerArtifact &artifact, std::string *reason = nullptr);
    bool instance_within_authenticated_max_envelope(const ProtocolContext &context, std::string *reason = nullptr);
    std::vector<CoefficientChunkLayoutEntry> coefficient_chunk_layout(std::size_t total_coefficient_length, std::size_t edge_domain_size);

    std::size_t minimum_edge_trace_length(std::size_t node_count, std::size_t edge_count, std::size_t hidden_width, std::size_t class_count);
    CoefficientRegionLayout coefficient_region_layout(std::size_t low_degree_length, const std::vector<std::size_t> &semantic_positions,
                                                      std::size_t tail_length, std::size_t edge_domain_size);
    std::vector<algebra::FieldElement> materialize_coefficient_tail(const CoefficientRegionLayout &layout,
                                                                    const std::vector<algebra::FieldElement> &low_degree_coefficients,
                                                                    const std::vector<algebra::FieldElement> &tail_randomness);
    std::size_t registered_blinder_coefficient_count(const ProtocolContext &context, const std::string &object_id);
    PolynomialVisibility polynomial_visibility(const ProtocolContext &context, const std::string &label);
    bool schema_accepts_evaluation(PolynomialVisibility visibility, EvaluationKind kind);
    std::string hidden_head_prefix(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_layer_concat_label(std::size_t layer_index);
    std::string hidden_layer_concat_star_label(std::size_t layer_index);
    std::string hidden_layer_cat_prefix(std::size_t layer_index);
    std::string hidden_weight_label(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_weight_column_label(std::size_t layer_index, std::size_t local_head_index, std::size_t column_index);
    std::string hidden_src_label(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_dst_label(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_challenge_name(std::string_view base, std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_q_in_selector_label(std::size_t layer_index);
    std::string hidden_q_d_h_selector_label(std::size_t layer_index);
    std::string hidden_q_cat_selector_label(std::size_t layer_index);
    std::string hidden_q_nd_h_selector_label(std::size_t layer_index);
    std::string output_q_cat_selector_label();
    std::string output_head_prefix(std::size_t head_index);
    std::string output_y_lin_label(std::size_t head_index);
    std::string output_y_label(std::size_t head_index);
    std::string canonical_matrix_column_label(const std::string &matrix_object, std::size_t column_index);
    std::string hidden_aggregation_binding_label(std::size_t layer_index, std::size_t local_head_index, std::string_view suffix);
    std::string output_aggregation_binding_label(std::size_t head_index, std::string_view suffix);
    std::string output_weight_label(std::size_t head_index);
    std::string output_weight_column_label(std::size_t head_index, std::size_t column_index);
    std::string output_src_label(std::size_t head_index);
    std::string output_dst_label(std::size_t head_index);
    std::string output_bias_label(std::size_t head_index);
    std::string output_challenge_name(std::string_view base, std::size_t head_index);
    std::string hidden_model_binding_blinder_label(std::size_t layer_index, std::size_t local_head_index, std::string_view role);
    std::string output_model_binding_blinder_label(std::size_t head_index, std::string_view role);
    std::string hidden_score_mask_label(std::size_t layer_index, std::size_t local_head_index, std::string_view role);
    std::string output_score_mask_label(std::size_t head_index, std::string_view role);
    std::string hidden_score_mask_eval_name(std::size_t layer_index, std::size_t local_head_index, std::string_view role);
    std::string output_score_mask_eval_name(std::size_t head_index, std::string_view role);
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
    std::string output_bias_basis_label(std::size_t head_index, std::string_view suffix);
    std::string output_bias_basis_chunk_label(std::size_t head_index, std::size_t chunk_index);
    std::string output_bias_basis_chunk_accumulator_label(std::size_t head_index, std::size_t chunk_index);
    std::string output_bias_basis_chunk_shifted_accumulator_label(std::size_t head_index, std::size_t chunk_index);
    std::string output_bias_basis_region_selector_label(std::size_t head_index, std::size_t chunk_index, std::string_view region);
    std::string output_bias_basis_gamma_label(std::size_t head_index);
    std::string output_bias_basis_relation_challenge_label(std::size_t head_index);

    std::size_t signed_division_limb_count(std::size_t range_bits);
    std::string signed_division_limb_label(std::string_view division_prefix, std::string_view value_name, std::size_t limb_index);
    std::string signed_division_range_label(std::string_view division_prefix, std::string_view value_name, std::size_t limb_index,
                                            std::string_view column_name);
    std::string signed_division_beta_name(std::string_view division_prefix, std::string_view value_name, std::size_t limb_index);
    std::uint64_t canonical_signed_round_div_bound(std::uint64_t numerator_bound, std::uint64_t positive_denominator, std::string_view label);
    std::string output_external_eval_name(std::string_view base, std::size_t head_index);
    std::string hidden_concat_xi_name(std::size_t layer_index);
    std::string hidden_concat_y_name(std::size_t layer_index);

}
