#pragma once
// 本文件维护正式 proof schema 使用的标签命名规则与 block 顺序，
// 是 trace、prove、verify 与 artifact 共享的命名真源。

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

    // Deterministic table-shape plan shared by the canonical producer, domain
    // planner and verifier registration.  It contains only public dimensions,
    // the registered fixed-point scales and the canonical large-graph shape
    // rule; it never materializes table rows.  In particular, `T_range` stays
    // exactly 2^B rows while `T_exp` is derived from the certified LReLU output
    // bound and therefore covers every delta in [0, delta_bound].
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

    // Public, context-neutral inputs for the canonical static-table planner.
    // The planner consumes only dimensions and the authenticated quantization
    // chain; it never needs model tensors, a proving key, or opening material.
    // Both the prover and the point-only verifier feed this same value object,
    // so neither side has to project or fabricate a ProtocolContext.
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

    // Result of reconstructing the concrete SecurityBudgetLedger.  The
    // verifier and the run acceptance gate consume this result rather than a
    // boolean copied from public metadata.  `ledger_digest` authenticates the
    // complete typed ledger, while `exact_budget` is the canonical reduced
    // rational epsilon_total string.
    struct SecurityBudgetEvaluation
    {
        bool passed = false;
        std::string ledger_digest;
        std::string exact_budget;
        bool operator==(const SecurityBudgetEvaluation &) const = default;
    };

    // Canonical, typed integer no-wrap certificate for the fixed-point
    // execution chain.  `absolute_bound` is the conservative signed-integer
    // magnitude (never a field element or a hash), and `limb_count` is the
    // minimum registered base-2^B representation width required by the
    // corresponding range object.  The verifier rebuilds these rows from the
    // public dimensions, quantization chain, static tables and authenticated
    // model bounds before accepting an instance.
    struct IntegerBoundLedgerEntry
    {
        std::string id;
        std::string stage;
        std::string source;
        std::uint64_t absolute_bound = 0;
        std::size_t limb_count = 0;
        bool operator==(const IntegerBoundLedgerEntry &) const = default;
    };

    // A derived private observation never owns fresh randomness merely because it
    // is serialized.  These edges are the canonical expression-DAG links used by
    // the ObservationLedger compiler to trace every derived field back to the
    // independent base objects that actually own blinder coordinates.
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
        std::vector<IntegerBoundLedgerEntry> integer_bound_ledger;
        std::string kzg_assumption_profile;
        std::string shift_algorithm_id;
        std::string shift_algorithm_spec;
        std::string protocol_canonical_bytes;
        std::string instance_canonical_bytes;
    };

    // 最终 proof 的 block 顺序，任何改变都会影响 serializer / parser 与 verifier。
    // 它也决定哪些标签会进入 proof 的正式传输形状，因此间接影响 proof_size_bytes。
    const std::vector<std::string> &proof_block_order();
    // Exact protocol-template bytes consumed by the pre-ceremony SRS compiler.
    // This object is independent of any dataset, witness, SRS trapdoor or exact
    // instance dimensions.
    std::string protocol_template_canonical_bytes();
    std::string protocol_proof_schema_template_canonical_bytes();
    std::string protocol_observation_ledger_template_canonical_bytes();
    std::string protocol_random_identity_ledger_template_canonical_bytes();
    std::string protocol_security_budget_ledger_template_canonical_bytes();
    std::string integer_bound_ledger_canonical_bytes(
        const std::vector<IntegerBoundLedgerEntry> &ledger);
    // Rebuild the complete fixed-point no-wrap ledger from public inputs,
    // authenticated static tables/quantization, shape and point-only model
    // bound certificates.  This path deliberately does not consume a
    // ProtocolContext or any raw model tensor.
    std::vector<IntegerBoundLedgerEntry> rebuild_integer_bound_ledger(
        const VerifierContext &context);
    // Compute the canonical static-table row/domain requirements without
    // allocating or generating any table row.  This is intentionally exposed
    // for bounded shape/range tests and for the pre-SRS domain planner.
    CanonicalStaticTableShape plan_canonical_static_table_shape(
        const ProtocolContext &context);
    CanonicalStaticTableShape plan_canonical_static_table_shape(
        const VerifierContext &context);
    CanonicalStaticTableShape plan_canonical_static_table_shape(
        const CanonicalStaticTableShapeInputs &inputs);
    std::size_t canonical_exp_table_size_from_delta_bound(
        std::uint64_t delta_bound);
    std::string security_budget_ledger_canonical_bytes(
        const std::vector<CompilerLedgerEntry> &ledger);
    SecurityBudgetEvaluation evaluate_security_budget(
        const ProtocolContext &context,
        const ProtocolCompilerArtifact &artifact);
    bool validate_static_tables(
        const StaticTables &tables,
        std::size_t range_bits,
        std::string *reason = nullptr);
    // Canonical static-table producer shared by the prover and the independent
    // verifier registration loader.  The table dimensions are derived from the
    // normalized public graph/configuration, while every row is rebuilt from
    // the registered fixed-point scales.  Keeping this producer in the schema
    // module prevents a loader from accepting a table that merely has the
    // right cardinality or monotonicity.
    StaticTables build_canonical_static_tables(
        const CanonicalStaticTableShapeInputs &inputs);
    StaticTables build_canonical_static_tables(const ProtocolContext &context);
    bool validate_static_tables(
        const ProtocolContext &context,
        std::string *reason = nullptr);
    // Canonical re-encoding of a fully populated concrete compiler artifact;
    // this is usable by a point-only verifier without rebuilding from model
    // tensors.
    std::string protocol_instance_canonical_bytes(
        const ProtocolCompilerArtifact &artifact);
    ProofSchema build_proof_schema(const ProtocolContext &context);
    ProtocolCompilerArtifact build_protocol_compiler_artifact(const ProtocolContext &context);

    std::vector<MatrixFoldBindingSpec> matrix_fold_binding_specs(
        const ProtocolContext &context)
    {
        std::vector<MatrixFoldBindingSpec> out;

        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
        {
            const auto &hidden_layer =
                context.model.hidden_layers.at(layer);

            for (std::size_t head = 0;
                 head < hidden_layer.heads.size();
                 ++head)
            {
                const auto prefix =
                    hidden_head_prefix(layer, head);

                const auto owner =
                    "bind:hidden:" +
                    std::to_string(layer + 1) +
                    ":" +
                    std::to_string(head);

                const auto base =
                    "P_SB_h_l" +
                    std::to_string(layer) +
                    "_h" +
                    std::to_string(head);

                out.push_back(MatrixFoldBindingSpec{
                    .relation_id = base + "_proj",
                    .owner_relation_id = owner,
                    .source_matrix_object =
                        layer == 0
                            ? std::string{}
                            : hidden_layer_concat_label(layer - 1),
                    .fold_object = prefix + "_a_proj",
                    .source_fold_object = base + "_proj_SrcFold",
                    .source_accumulator_object = base + "_proj_SrcAcc",
                    .fold_accumulator_object = base + "_proj_FoldAcc",
                    .source_point_challenge =
                        hidden_challenge_name(
                            "rho_proj_h",
                            layer,
                            head),
                    .lambda_challenge =
                        "lambda_sb:" +
                        base +
                        ":proj",
                    .fold_domain = context.domains.in->name,
                    .fold_selector_label =
                        hidden_q_in_selector_label(layer),
                    .source_rows = context.local.num_nodes,
                    .width = hidden_layer.input_dim,
                    .public_source = layer == 0,
                });

                for (const auto *kind : {"src", "dst"})
                {
                    out.push_back(MatrixFoldBindingSpec{
                        .relation_id =
                            base + "_" + kind,
                        .owner_relation_id = owner,
                        .source_matrix_object =
                            prefix + "_H_prime",
                        .fold_object =
                            prefix + "_a_" + kind,
                        .source_fold_object =
                            base + "_" + kind + "_SrcFold",
                        .source_accumulator_object =
                            base + "_" + kind + "_SrcAcc",
                        .fold_accumulator_object =
                            base + "_" + kind + "_FoldAcc",
                        .source_point_challenge =
                            hidden_challenge_name(
                                std::string("y_") +
                                    kind +
                                    "_h",
                                layer,
                                head),
                        .lambda_challenge =
                            "lambda_sb:" +
                            base +
                            ":" +
                            kind,
                        .fold_domain = context.domains.d->name,
                        .fold_selector_label =
                            hidden_q_d_h_selector_label(layer),
                        .source_rows =
                            context.local.num_nodes,
                        .width =
                            hidden_layer.shape.head_dim,
                        .public_source = false,
                    });
                }
            }
        }

        if (context.model.hidden_layers.empty())
        {
            throw std::runtime_error(
                "soundness binding requires at least one hidden layer");
        }

        const auto output_input =
            hidden_layer_concat_label(
                context.model.hidden_layers.size() - 1);

        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size();
             ++head)
        {
            const auto prefix =
                output_head_prefix(head);

            const auto owner =
                "bind:output:" +
                std::to_string(head);

            const auto base =
                "P_SB_out" +
                std::to_string(head);

            out.push_back(MatrixFoldBindingSpec{
                .relation_id = base + "_proj",
                .owner_relation_id = owner,
                .source_matrix_object = output_input,
                .fold_object = prefix + "_a_proj",
                .source_fold_object =
                    base + "_proj_SrcFold",
                .source_accumulator_object =
                    base + "_proj_SrcAcc",
                .fold_accumulator_object =
                    base + "_proj_FoldAcc",
                .source_point_challenge =
                    output_challenge_name(
                        "rho_proj_out",
                        head),
                .lambda_challenge =
                    "lambda_sb:" +
                    base +
                    ":proj",
                .fold_domain = context.domains.cat->name,
                .fold_selector_label =
                    output_q_cat_selector_label(),
                .source_rows =
                    context.local.num_nodes,
                .width =
                    context.model.output_layer.input_dim,
                .public_source = false,
            });

            for (const auto *kind : {"src", "dst"})
            {
                out.push_back(MatrixFoldBindingSpec{
                    .relation_id =
                        base + "_" + kind,
                    .owner_relation_id = owner,
                    .source_matrix_object =
                        prefix + "_Y_prime",
                    .fold_object =
                        prefix + "_a_" + kind,
                    .source_fold_object =
                        base + "_" + kind + "_SrcFold",
                    .source_accumulator_object =
                        base + "_" + kind + "_SrcAcc",
                    .fold_accumulator_object =
                        base + "_" + kind + "_FoldAcc",
                    .source_point_challenge =
                        output_challenge_name(
                            std::string("y_") +
                                kind +
                                "_out",
                            head),
                    .lambda_challenge =
                        "lambda_sb:" +
                        base +
                        ":" +
                        kind,
                    .fold_domain = context.domains.c->name,
                    .fold_selector_label =
                        "P_Q_C_valid",
                    .source_rows =
                        context.local.num_nodes,
                    .width =
                        context.local.num_classes,
                    .public_source = false,
                });
            }
        }

        return out;
    }

    std::vector<FlatMatrixBindingSpec> flat_matrix_binding_specs(
        const ProtocolContext &context)
    {
        std::vector<FlatMatrixBindingSpec> out;

        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
        {
            for (std::size_t head = 0;
                 head <
                 context.model.hidden_layers[layer]
                     .heads.size();
                 ++head)
            {
                const auto prefix =
                    hidden_head_prefix(layer, head);

                const auto owner =
                    "bind:hidden:" +
                    std::to_string(layer + 1) +
                    ":" +
                    std::to_string(head);

                const auto base =
                    "P_SB_h_l" +
                    std::to_string(layer) +
                    "_h" +
                    std::to_string(head);

                const auto width =
                    context.model.hidden_layers[layer]
                        .shape.head_dim;

                for (const auto &[kind, matrix, flat] :
                     std::vector<
                         std::tuple<
                             std::string,
                             std::string,
                             std::string>>{
                         {
                             "aggpre",
                             prefix + "_H_agg_pre",
                             prefix + "_H_agg_pre_flat",
                         },
                         {
                             "agg",
                             prefix + "_H_agg",
                             prefix + "_H_agg_flat",
                         },
                     })
                {
                    out.push_back(FlatMatrixBindingSpec{
                        .relation_id =
                            base + "_flat_" + kind,
                        .owner_relation_id = owner,
                        .matrix_object = matrix,
                        .flat_object = flat,
                        .row_fold_object =
                            base + "_flat_" +
                            kind + "_RowFold",
                        .row_accumulator_object =
                            base + "_flat_" +
                            kind + "_RowAcc",
                        .flat_accumulator_object =
                            base + "_flat_" +
                            kind + "_FlatAcc",
                        .lambda_challenge =
                            "lambda_flat:" +
                            base +
                            ":" +
                            kind,
                        .flat_selector_label =
                            hidden_q_nd_h_selector_label(
                                layer),
                        .rows =
                            context.local.num_nodes,
                        .columns = width,
                    });
                }
            }
        }

        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size();
             ++head)
        {
            const auto prefix =
                output_head_prefix(head);

            const auto owner =
                "bind:output:" +
                std::to_string(head);

            const auto base =
                "P_SB_out" +
                std::to_string(head);

            for (const auto &[kind, matrix, flat] :
                 std::vector<
                     std::tuple<
                         std::string,
                         std::string,
                         std::string>>{
                     {
                         "ylin",
                         output_y_lin_label(head),
                         prefix + "_Y_pre_field_flat",
                     },
                     {
                         "y",
                         output_y_label(head),
                         prefix + "_Y_flat",
                     },
                 })
            {
                out.push_back(FlatMatrixBindingSpec{
                    .relation_id =
                        base + "_flat_" + kind,
                    .owner_relation_id = owner,
                    .matrix_object = matrix,
                    .flat_object = flat,
                    .row_fold_object =
                        base + "_flat_" +
                        kind + "_RowFold",
                    .row_accumulator_object =
                        base + "_flat_" +
                        kind + "_RowAcc",
                    .flat_accumulator_object =
                        base + "_flat_" +
                        kind + "_FlatAcc",
                    .lambda_challenge =
                        "lambda_flat:" +
                        base +
                        ":" +
                        kind,
                    .flat_selector_label =
                        "P_Q_div_out" +
                        std::to_string(head),
                    .rows =
                        context.local.num_nodes,
                    .columns =
                        context.local.num_classes,
                });
            }
        }

        return out;
    }

    std::vector<BroadcastFlatBindingSpec>
    broadcast_flat_binding_specs(
        const ProtocolContext &context)
    {
        std::vector<BroadcastFlatBindingSpec> out;

        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
        {
            for (std::size_t head = 0;
                 head <
                 context.model.hidden_layers[layer]
                     .heads.size();
                 ++head)
            {
                const auto prefix =
                    hidden_head_prefix(layer, head);

                const auto base =
                    "P_SB_h_l" +
                    std::to_string(layer) +
                    "_h" +
                    std::to_string(head) +
                    "_sum";

                out.push_back(BroadcastFlatBindingSpec{
                    .relation_id = base,
                    .owner_relation_id =
                        "bind:hidden:" +
                        std::to_string(layer + 1) +
                        ":" +
                        std::to_string(head),
                    .node_object = prefix + "_Sum",
                    .flat_object = prefix + "_Sum_flat",
                    .node_accumulator_object =
                        base + "_NodeAcc",
                    .flat_accumulator_object =
                        base + "_FlatAcc",
                    .lambda_challenge =
                        "lambda_broadcast:" +
                        base,
                    .flat_selector_label =
                        hidden_q_nd_h_selector_label(
                            layer),
                    .rows =
                        context.local.num_nodes,
                    .width =
                        context.model.hidden_layers[layer]
                            .shape.head_dim,
                });
            }
        }

        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size();
             ++head)
        {
            const auto prefix =
                output_head_prefix(head);

            const auto base =
                "P_SB_out" +
                std::to_string(head) +
                "_sum";

            out.push_back(BroadcastFlatBindingSpec{
                .relation_id = base,
                .owner_relation_id =
                    "bind:output:" +
                    std::to_string(head),
                .node_object = prefix + "_Sum",
                .flat_object = prefix + "_Sum_flat",
                .node_accumulator_object =
                    base + "_NodeAcc",
                .flat_accumulator_object =
                    base + "_FlatAcc",
                .lambda_challenge =
                    "lambda_broadcast:" +
                    base,
                .flat_selector_label =
                    "P_Q_div_out" +
                    std::to_string(head),
                .rows =
                    context.local.num_nodes,
                .width =
                    context.local.num_classes,
            });
        }

        return out;
    }

    std::vector<BindingRelationMeta> build_binding_relation_metadata(const ProtocolContext &context);
    std::vector<MatrixFoldBindingSpec> matrix_fold_binding_specs(
        const ProtocolContext &context);

    std::vector<FlatMatrixBindingSpec> flat_matrix_binding_specs(
        const ProtocolContext &context);

    std::vector<BroadcastFlatBindingSpec> broadcast_flat_binding_specs(
        const ProtocolContext &context);
    // Canonical S_route payload.  Every entry is a reference to the unique
    // standard point-group owner in Open_dom/W_ext; no route terminal value is
    // ever materialized in the proof object.
    std::vector<MetadataReference> route_opening_references(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges);

    std::string opening_group_reference(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const algebra::FieldElement &point);
    const KZGShiftRelationMeta *find_unique_shift_relation_by_kind(
        const BindingRelationMeta &metadata,
        std::string_view relation_kind);
    bool validate_protocol_compiler_artifact(
        const ProtocolContext &context,
        const ProtocolCompilerArtifact &artifact,
        std::string *reason = nullptr);
    bool instance_within_authenticated_max_envelope(
        const ProtocolContext &context,
        std::string *reason = nullptr);
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
        const std::vector<std::size_t> &semantic_positions,
        std::size_t tail_length,
        std::size_t edge_domain_size);
    std::vector<algebra::FieldElement> materialize_coefficient_tail(
        const CoefficientRegionLayout &layout,
        const std::vector<algebra::FieldElement> &low_degree_coefficients,
        const std::vector<algebra::FieldElement> &tail_randomness);
    // Returns the compiler-registered prefix length q_O^max for the independent
    // blinder owned by object_id.  Formal prover code must not choose this length
    // from the witness or from a local constant.
    std::size_t registered_blinder_coefficient_count(
        const ProtocolContext &context,
        const std::string &object_id);
    // Single source of truth for payload visibility. In particular, per-head output
    // objects (for example `P_Y_o0`) remain private even when final `P_Y` is public.
    PolynomialVisibility polynomial_visibility(const ProtocolContext &context, const std::string &label);
    bool schema_accepts_evaluation(PolynomialVisibility visibility, EvaluationKind kind);
    // 下面这些函数负责生成隐藏层和输出层使用的正式标签。
    // 它们共同决定 dynamic commitments、opening bundle 与 external evaluation 的命名真源。
    // Formal hidden objects are identified by the pair (layer, local head).
    // A flattened head index is intentionally not part of the proof schema.
    std::string hidden_head_prefix(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_layer_concat_label(std::size_t layer_index);
    std::string hidden_layer_concat_star_label(std::size_t layer_index);
    std::string hidden_layer_cat_prefix(std::size_t layer_index);
    std::string hidden_weight_label(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_weight_column_label(
        std::size_t layer_index,
        std::size_t local_head_index,
        std::size_t column_index);
    std::string hidden_src_label(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_dst_label(std::size_t layer_index, std::size_t local_head_index);
    std::string hidden_challenge_name(
        std::string_view base,
        std::size_t layer_index,
        std::size_t local_head_index);
    std::string hidden_q_in_selector_label(std::size_t layer_index);
    std::string hidden_q_d_h_selector_label(std::size_t layer_index);
    std::string hidden_q_cat_selector_label(std::size_t layer_index);
    std::string hidden_q_nd_h_selector_label(std::size_t layer_index);
    std::string output_q_cat_selector_label();
    std::string output_head_prefix(std::size_t head_index);
    std::string output_y_lin_label(std::size_t head_index);
    std::string output_y_label(std::size_t head_index);
    std::string canonical_matrix_column_label(
        const std::string &matrix_object,
        std::size_t column_index);
    std::string hidden_aggregation_binding_label(
        std::size_t layer_index,
        std::size_t local_head_index,
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
    std::string hidden_model_binding_blinder_label(
        std::size_t layer_index,
        std::size_t local_head_index,
        std::string_view role);
    std::string output_model_binding_blinder_label(std::size_t head_index, std::string_view role);
    std::string hidden_score_mask_label(
        std::size_t layer_index,
        std::size_t local_head_index,
        std::string_view role);
    std::string output_score_mask_label(std::size_t head_index, std::string_view role);
    std::string hidden_score_mask_eval_name(
        std::size_t layer_index,
        std::size_t local_head_index,
        std::string_view role);
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
    // Conservative quotient bound for the canonical signed nearest-integer
    // division used by the fixed-point trace:
    // q = (abs(numerator) + denominator / 2) / denominator.
    // The numerator argument is an absolute-magnitude bound and the
    // denominator is the positive, common factor after any sound cancellation
    // from the aggregate relation.  Overflow and a non-positive denominator
    // fail closed so producer and verifier cannot diverge silently.
    std::uint64_t canonical_signed_round_div_bound(
        std::uint64_t numerator_bound,
        std::uint64_t positive_denominator,
        std::string_view label);
    std::string output_external_eval_name(std::string_view base, std::size_t head_index);
    std::string hidden_concat_xi_name(std::size_t layer_index);
    std::string hidden_concat_y_name(std::size_t layer_index);

} // namespace gatzk::protocol
