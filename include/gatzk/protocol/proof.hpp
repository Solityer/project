#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/data/dataset.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/trace_spill.hpp"

namespace gatzk::protocol
{

    struct ProtocolCompilerArtifact;

    namespace detail
    {
        struct ValidatedVerifierExecutionState;
    }

    inline constexpr const char *kFormalTimingSchema = "online_trace_commit_prove_protocol_verify_v2";

    struct StaticTables
    {
        std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> lrelu;
        std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> elu;
        std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> exp;
        std::vector<algebra::FieldElement> range;
    };

    struct AttentionHeadDomains
    {
        std::shared_ptr<algebra::RootOfUnityDomain> in;
        std::shared_ptr<algebra::RootOfUnityDomain> d;
    };

    struct WorkDomains
    {
        std::shared_ptr<algebra::RootOfUnityDomain> fh;
        std::shared_ptr<algebra::RootOfUnityDomain> edge;
        std::shared_ptr<algebra::RootOfUnityDomain> in;
        std::shared_ptr<algebra::RootOfUnityDomain> d;
        std::shared_ptr<algebra::RootOfUnityDomain> cat;
        std::shared_ptr<algebra::RootOfUnityDomain> c;
        std::shared_ptr<algebra::RootOfUnityDomain> n;
        std::vector<std::vector<AttentionHeadDomains>> hidden_layers;
        std::optional<AttentionHeadDomains> output_head;
    };

    struct VerifierPublicPolynomialDescriptor
    {
        std::string name;
        algebra::PolynomialBasis basis = algebra::PolynomialBasis::Coefficient;
        std::string domain;
        std::size_t size = 0;
        bool operator==(const VerifierPublicPolynomialDescriptor &) const = default;
    };

    struct PublicMetadata
    {
        std::string protocol_name;
        std::string proof_nonce;
        std::string proof_task;
        std::string srs_id;
        std::string protocol_schema_id;
        std::string instance_schema_hash;
        std::string srs_type;
        std::string srs_degree_bound;
        std::string kzg_assumption_profile;
        std::string privacy_policy;
        std::string input_privacy;
        std::string output_mode;
        std::string opening_mode;
        std::string binding_mode;
        std::string model_arch;
        std::string model_params;
        std::string input_preprocess_id;
        std::string static_tables;
        std::string quant_config;
        std::string domain_cfg;
        std::string dim_cfg;
        std::string encoding;
        std::string padding_rule;
        std::string degree_bound;
        std::string security_budget;
        std::string public_output;
    };

    struct ProtocolContext
    {
        util::AppConfig config;
        data::GraphDataset dataset;
        data::LocalGraph local;
        model::ModelParameters model;
        StaticTables tables;
        WorkDomains domains;
        crypto::KZGKeyPair kzg;
        std::unordered_map<std::string, algebra::Polynomial> public_polynomials;
        std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> public_polynomial_descriptors;
        std::unordered_map<std::string, crypto::Commitment> public_commitments;
        std::unordered_map<std::string, std::shared_ptr<const algebra::Polynomial>> public_fh_coefficient_materials;
        std::unordered_map<std::string, crypto::Commitment> static_commitments;
        std::shared_ptr<const ProtocolCompilerArtifact> compiler_artifact;
        const ProtocolCompilerArtifact *validated_compiler_artifact_identity = nullptr;
        bool verifier_shape_only = false;
    };

    struct VerifierConfig
    {
        std::string dataset;
        std::string task_type = "transductive_node_classification";
        std::string report_unit = "node";
        std::string batching_rule = "whole_graph_single";
        std::string subgraph_rule = "whole_graph";
        std::string self_loop_rule = "per_node";
        std::string edge_sort_rule = "edge_gid_then_dst_stable";
        std::string chunking_rule = "none";
        std::string quant_cfg_id;
        util::QuantizationScaleChain quantization;
        std::string model_arch_id;
        std::string model_param_id;
        std::string input_preprocess_id;
        std::string static_table_id;
        std::string degree_bound_id;
        std::string srs_type = "trusted_srs";
        std::string srs_id;
        std::string privacy_policy = "model_and_trace_hidden";
        std::string input_privacy = "public_graph_public_features";
        std::string output_mode = "public_logits";
        std::string opening_mode = "point_grouped_batch_opening";
        std::string binding_mode = "zk_subproof_for_crpc";
        std::string proof_task = "node_classification_full_gat";
        std::string security_profile_id = "zkgat_secure_profile";
        std::size_t num_classes = 2;
        std::size_t range_bits = 10;
        std::size_t layer_count = 0;
        std::size_t K_out = 1;
        std::size_t batch_graphs = 1;
        std::vector<util::HiddenLayerProfile> hidden_profile;
        std::vector<std::size_t> d_in_profile;
    };

    struct VerifierModelHeadShape
    {
    };

    struct VerifierHiddenLayerShape
    {
        std::size_t layer_index = 0;
        std::size_t input_dim = 0;
        model::HiddenLayerShape shape;
        std::vector<VerifierModelHeadShape> heads;
    };

    struct VerifierOutputLayerShape
    {
        std::size_t input_dim = 0;
        std::size_t output_dim = 0;
        std::size_t head_count = 0;
        std::vector<VerifierModelHeadShape> heads;
    };

    struct VerifierModelShape
    {
        std::string model_arch_id;
        std::string model_param_id;
        std::string input_preprocess_id;
        std::string quant_cfg_id;
        std::string static_table_id;
        std::string degree_bound_id;
        std::size_t L = 0;
        std::vector<std::size_t> d_in_profile;
        std::vector<model::HiddenLayerShape> hidden_profile;
        std::size_t K_out = 1;
        std::size_t C = 0;
        std::vector<VerifierHiddenLayerShape> hidden_layers;
        VerifierOutputLayerShape output_layer;
    };

    struct VerifierSRSMetadata
    {
        std::string manifest_format;
        std::string srs_id;
        std::string curve_id;
        std::string field_modulus_id;
        std::string srs_type;
        std::string ceremony_transcript_hash;
        std::size_t degree_bound = 0;
        std::unordered_map<std::string, std::size_t> domain_sizes;
        std::unordered_map<std::string, std::string> domain_generators;
        std::string root_derivation_id;
        std::string row_order;
        std::string protocol_compiler_artifact_hash;
        std::string max_instance_envelope_id;
        std::string max_instance_envelope_hash;
        std::unordered_map<std::string, std::size_t> max_instance_envelope;
        std::string proof_schema_template_hash;
        std::string observation_ledger_hash;
        std::string random_identity_ledger_hash;
        std::string security_budget_ledger_hash;
        std::string vk_hash;
        std::string vanishing_g2_hash;
        std::string srs_audit_report_hash;
        std::string srs_consistency_hash;
        std::string manifest_hash;
        std::string generator_id;
        std::string encoding;
        std::string protocol_schema_id;
        std::string kzg_assumption_profile_id;
        std::size_t kzg_assumption_degree_bound = 0;
        std::size_t kzg_assumption_security_bits = 0;
        std::string kzg_assumption_domain_catalog_hash;
        std::string domain_generator_catalog_hash;
        std::string security_level;
    };

    struct VerifierPointCommitment
    {
        std::string name;
        crypto::G1Point point;
    };

    struct ModelBoundCertificate
    {
        bool present = false;
        std::string label;
        std::size_t layer_index = 0;
        std::size_t local_head_index = 0;
        std::string domain;
        std::size_t degree_bound = 0;
        std::uint64_t scale = 0;
        std::uint64_t absolute_bound = 0;
        bool operator==(const ModelBoundCertificate &) const = default;
    };

    struct VerifierRegistryEntry
    {
        std::string label;
        std::string domain;
        std::size_t degree_bound = 0;
        std::uint64_t scale = 0;
        std::int64_t range_min = 0;
        std::int64_t range_max = 0;
        crypto::G1Point point;
        ModelBoundCertificate bound_certificate;
        bool operator==(const VerifierRegistryEntry &) const = default;
    };

    struct AuthenticatedVerifierRegistry
    {
        std::string format = "zkgat_verifier_registry";
        std::string registry_id;
        std::string model_arch_id;
        std::string model_param_id;
        std::string input_preprocess_id;
        std::string quant_cfg_id;
        std::string static_table_id;
        std::string srs_id;
        std::string protocol_schema_id;
        std::string instance_schema_hash;
        std::string srs_manifest_hash;
        std::size_t srs_degree_bound = 0;
        crypto::G1Point srs_g1_generator;
        std::string srs_generator_id;
        std::string srs_encoding;
        std::string srs_spec_id;
        std::string srs_protocol_compiler_artifact_hash;
        std::string compiler_artifact_digest;
        std::string model_layout_digest;
        std::string quant_chain_digest;
        std::string static_table_digest;
        std::string static_table_generator_id;
        std::string static_table_spec_hash;
        std::string range_and_scale_certificate;
        std::string integer_bound_ledger_digest;
        std::string public_input_digest;
        std::string topology_digest;
        std::vector<IntegerBoundLedgerEntry> integer_bound_ledger;
        std::string security_budget_ledger_digest;
        std::string security_budget_exact;
        std::vector<VerifierRegistryEntry> model_entries;
        std::vector<VerifierRegistryEntry> static_entries;
        std::string canonical_serialization() const;
        std::string canonical_digest() const;
        bool operator==(const AuthenticatedVerifierRegistry &) const = default;
    };

    struct VerifierContext
    {
        VerifierConfig config;
        data::GraphDataset dataset;
        data::LocalGraph local;
        VerifierModelShape model;
        StaticTables tables;
        WorkDomains domains;
        VerifierSRSMetadata srs;
        crypto::KZGVerifierKey kzg_vk;
        std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> public_polynomial_descriptors;
        std::unordered_map<std::string, VerifierPointCommitment> public_commitments;
        std::unordered_map<std::string, VerifierPointCommitment> static_commitments;
        std::shared_ptr<const ProtocolCompilerArtifact> compiler_artifact;
        AuthenticatedVerifierRegistry registry;
        std::string authenticated_registry_digest;
        std::string trusted_registry_digest;

        std::shared_ptr<const detail::ValidatedVerifierExecutionState> validated_execution_state;
    };

    struct VerifierRegistrationArtifact
    {
        VerifierConfig config;
        data::GraphDataset dataset;
        data::LocalGraph local;
        VerifierModelShape model;
        StaticTables tables;
        WorkDomains domains;
        VerifierSRSMetadata srs;
        crypto::KZGVerifierKey kzg_vk;
        std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> public_polynomial_descriptors;
        std::unordered_map<std::string, VerifierPointCommitment> public_commitments;
        std::unordered_map<std::string, VerifierPointCommitment> static_commitments;
        ProtocolCompilerArtifact compiler_artifact;
        AuthenticatedVerifierRegistry registry;

        std::vector<std::uint8_t> canonical_bytes() const;
        std::string canonical_digest() const;
    };

    struct PiBindProof;

    struct TraceArtifacts
    {
        std::shared_ptr<const ProtocolCompilerArtifact> compiler_artifact;
        std::shared_ptr<crypto::EvaluationCommitmentCache> evaluation_commitment_cache;
        PublicMetadata public_metadata;
        struct SpilledEvaluationPolynomial
        {
            std::string path;
            std::string domain_name;
            std::size_t size = 0;
            std::size_t physical_size = 0;
            bool zero_tail_compacted = false;
            std::uintptr_t semantic_source_id = 0;
            bool compact_semantic_source_all_zero = false;
            std::vector<algebra::FieldElement> zero_domain_blinder;
            std::shared_ptr<const util::TraceSpillLease> lease;
        };

        struct CompactSemanticSourceZeroCacheEntry
        {
            std::string domain_name;
            std::size_t size = 0;
            bool all_zero = false;
        };
        struct SemanticSpillSourceCacheEntry
        {
            std::weak_ptr<const std::vector<algebra::FieldElement>> values;

            std::string path;
            std::string domain_name;
            std::size_t size = 0;
            std::size_t physical_size = 0;
            bool zero_tail_compacted = false;
            bool compact_semantic_source_all_zero = false;
        };

        struct SemanticSourceIdentityCacheEntry
        {
            std::weak_ptr<const std::vector<algebra::FieldElement>> values;
            std::string domain_name;
            std::size_t size = 0;
            std::uintptr_t token = 0;
        };

        std::unordered_map<std::string, std::vector<algebra::FieldElement>> columns;
        std::unordered_map<std::string, model::Matrix> matrices;
        std::unordered_map<std::string, algebra::Polynomial> polynomials;
        std::unordered_map<std::string, SpilledEvaluationPolynomial> spilled_polynomials;

        struct SpilledCoefficientPolynomial
        {
            std::string path;
            std::size_t size = 0;
            std::shared_ptr<const util::TraceSpillLease> lease;
        };
        std::unordered_map<std::string, SpilledCoefficientPolynomial> spilled_coefficient_polynomials;
        bool large_execution_working_set = false;
        std::uintptr_t next_spill_semantic_source_token = 1;
        std::vector<SemanticSourceIdentityCacheEntry> semantic_source_identity_cache;
        std::unordered_map<std::uintptr_t, CompactSemanticSourceZeroCacheEntry> compact_semantic_source_zero_cache;
        std::unordered_map<std::string, SemanticSpillSourceCacheEntry> semantic_spill_source_cache;
        std::shared_ptr<const util::TraceSpillLease> spill_directory_lease;
        std::unordered_map<std::string, std::string> polynomial_domains;
        std::unordered_map<std::string, crypto::Commitment> commitments;
        std::vector<std::string> commitment_order;
        std::map<std::string, algebra::FieldElement> challenges;
        std::map<std::string, algebra::FieldElement> external_evaluations;
        std::vector<algebra::FieldElement> public_logits;
        std::string reference_fp32_logits_hash;
        std::string reference_quant_logits_hash;
        std::string secure_trace_logits_hash;
        bool reference_consistency_passed = false;
        std::string proof_public_output_hash;
        std::shared_ptr<PiBindProof> pi_bind;
    };

    inline std::size_t spilled_file_value_count(const TraceArtifacts::SpilledEvaluationPolynomial &spilled)
    {
        if (!spilled.zero_tail_compacted)
        {
            return spilled.size;
        }
        if (spilled.physical_size > spilled.size)
        {
            throw std::runtime_error("spilled polynomial physical size exceeds logical size");
        }
        return spilled.physical_size;
    }

    struct DynamicCommitmentObjectProfile
    {
        std::string label;
        std::string dataset;
        std::string domain;
        std::string basis;
        std::string classification;
        std::string ownership;
        std::string category;
        std::string optimization_opportunity;
        std::string duplicate_of;
        std::string commitment_fingerprint;
        std::size_t vector_length = 0;
        std::size_t domain_size = 0;
        std::size_t nonzero_scalar_count = 0;
        std::size_t effective_scalar_count = 0;
        double zero_scalar_ratio = 0.0;
        double padding_ratio = 0.0;
        double materialization_ms = 0.0;
        double msm_ms = 0.0;
        double total_commitment_ms = 0.0;
        bool compact = false;
        bool shared_or_borrowed = false;
        bool reused_shared = false;
        bool duplicates_another = false;
        bool zero_scalar_skipping = false;
        bool sparse_compaction = false;
        bool constant_tail_folding = false;
        bool multithreaded_msm = false;
    };

    enum class PayloadValueKind
    {
        PublicValue,
        MaskedPrivateValue,
    };

    struct DomainOpeningBundle
    {
        std::vector<algebra::FieldElement> points;
        std::vector<std::pair<std::string, std::vector<algebra::FieldElement>>> values;
        std::vector<PayloadValueKind> value_kinds;
    };

    struct PointGroupOpening
    {
        std::string group_id;
        std::string owner_block;
        algebra::FieldElement point = algebra::FieldElement::zero();
        crypto::G1Point witness;
    };

    struct MetadataReference
    {
        std::string object_id;
        bool operator==(const MetadataReference &) const = default;
    };

    struct ModelReblindProof
    {
        std::string object_id;
        std::string domain;
        MetadataReference vk_model_anchor;
        crypto::Commitment proof_local_commitment;
        crypto::Commitment blinder_commitment;
    };

    struct IndexedViewRelationMeta
    {
        std::string relation_id;
        std::string canonical_object;
        std::string view_object;
        std::string canonical_domain;
        std::string view_domain;
        std::size_t canonical_length = 0;
        std::size_t view_length = 0;
        std::string index_map_id;
        std::string canonical_selector_id;
        std::string query_selector_id;
        std::string multiplicity_id;
        std::string eta_challenge_label;
        std::string beta_challenge_label;
        std::vector<std::string> commitment_order;
        bool operator==(const IndexedViewRelationMeta &) const = default;
    };

    struct KZGShiftRelationMeta
    {
        std::string relation_id;
        std::vector<std::string> source_linear_maps;
        std::vector<std::string> source_domains;
        std::vector<algebra::FieldElement> canonical_points;
        algebra::FieldElement lambda0 = algebra::FieldElement::zero();
        std::vector<algebra::FieldElement> deltas;
        std::vector<std::string> source_value_kinds;
        std::string challenge_label;
        std::size_t exclusion_degree = 0;
        std::size_t max_difference_degree = 0;
        std::vector<std::size_t> configured_degrees;
        std::vector<std::size_t> shift_domain_sizes;
        std::vector<std::string> opening_claim_order;
        bool operator==(const KZGShiftRelationMeta &) const = default;
    };

    struct CoefficientBasisChunkMeta
    {
        std::string object_id;
        std::string accumulator_object;
        std::string shifted_accumulator_object;
        std::size_t chunk_index = 0;
        std::size_t offset = 0;
        std::size_t length = 0;
        algebra::FieldElement terminal_point = algebra::FieldElement::zero();
        std::size_t configured_shift_degree = 0;
        std::size_t shift_domain_size = 0;
        bool operator==(const CoefficientBasisChunkMeta &) const = default;
    };

    struct CoefficientBasisRelationMeta
    {
        std::string relation_id;
        std::string canonical_object;
        std::string coefficient_object;
        std::string carrier_domain;
        std::size_t low_degree_length = 0;
        std::size_t tail_length = 0;
        std::size_t total_length = 0;
        std::vector<std::size_t> semantic_positions;
        std::vector<CoefficientBasisChunkMeta> chunks;
        std::string power_object;
        std::string gamma_challenge_label;
        std::string relation_challenge_label;
        std::vector<std::string> commitment_order;
        std::vector<std::string> opening_claim_order;
        bool operator==(const CoefficientBasisRelationMeta &) const = default;
    };

    struct CanonicalMatrixRelationMeta
    {
        std::string relation_id;
        std::string matrix_object;
        std::string canonical_domain;
        std::size_t row_count = 0;
        std::size_t column_count = 0;
        std::vector<std::string> column_objects;
        std::string compressed_object;
        std::string compression_challenge_label;
        bool operator==(const CanonicalMatrixRelationMeta &) const = default;
    };

    struct AggregationRouteRelationMeta
    {
        std::string relation_id;
        std::string index_object;
        std::string node_value_object;
        std::string edge_value_object;
        std::string table_object;
        std::string query_object;
        std::string multiplicity_object;
        std::string node_inverse_object;
        std::string edge_inverse_object;
        std::string node_accumulator_object;
        std::string edge_accumulator_object;
        std::string shifted_accumulator_object;
        std::string relation_virtual_object;
        std::string eta_challenge_label;
        std::string beta_challenge_label;
        std::string terminal_challenge_label;
        algebra::FieldElement node_terminal_point = algebra::FieldElement::zero();
        algebra::FieldElement edge_terminal_point = algebra::FieldElement::zero();
        algebra::FieldElement terminal_delta = algebra::FieldElement::zero();
        std::size_t configured_shift_degree = 0;
        std::size_t shift_domain_size = 0;
        bool operator==(const AggregationRouteRelationMeta &) const = default;
    };

    struct AggregationRelationMeta
    {
        std::string relation_id;
        std::string source_matrix_object;
        std::string target_matrix_object;
        std::string alpha_object;
        std::string node_domain;
        std::string edge_domain;
        std::size_t row_count = 0;
        std::size_t column_count = 0;
        std::string compression_challenge_label;
        std::string source_node_object;
        std::string target_node_object;
        std::string source_edge_object;
        std::string target_edge_object;
        std::string product_object;
        std::string psq_object;
        AggregationRouteRelationMeta source_route;
        AggregationRouteRelationMeta target_route;
        std::vector<std::string> commitment_order;
        bool operator==(const AggregationRelationMeta &) const = default;
    };

    struct RouteTerminalRelationMeta
    {
        std::string relation_id;
        std::string node_accumulator_object;
        std::string edge_accumulator_object;
        std::string shifted_accumulator_object;
        std::string relation_virtual_object;
        std::string node_domain;
        std::string edge_domain;
        std::string challenge_label;
        algebra::FieldElement node_terminal_point = algebra::FieldElement::zero();
        algebra::FieldElement edge_terminal_point = algebra::FieldElement::zero();
        algebra::FieldElement terminal_delta = algebra::FieldElement::zero();
        std::size_t configured_shift_degree = 0;
        std::size_t shift_domain_size = 0;
        bool operator==(const RouteTerminalRelationMeta &) const = default;
    };

    struct BindingRelationMeta
    {
        std::string relation_id;
        std::string object_kind;
        std::optional<std::size_t> layer_id;
        std::optional<std::size_t> head_id;
        std::vector<std::string> source_objects;
        std::vector<std::string> target_objects;
        std::vector<std::string> work_domains;
        std::vector<std::size_t> real_lengths;
        std::vector<std::string> quotient_relations;
        std::vector<CanonicalMatrixRelationMeta> canonical_matrices;
        std::vector<AggregationRelationMeta> aggregation_relations;
        std::vector<RouteTerminalRelationMeta> route_terminal_relations;
        std::vector<IndexedViewRelationMeta> indexed_views;
        std::vector<CoefficientBasisRelationMeta> coefficient_basis_relations;
        std::vector<KZGShiftRelationMeta> shift_relations;
        bool operator==(const BindingRelationMeta &) const = default;
    };

    struct BindingEvaluation
    {
        std::string domain;
        std::string object_id;
        std::vector<algebra::FieldElement> values;
        PayloadValueKind kind = PayloadValueKind::MaskedPrivateValue;
    };

    struct BindingSubproof
    {
        BindingRelationMeta relation_meta;
        std::vector<ModelReblindProof> model_reblinds;
        std::vector<MetadataReference> view_references;
        std::vector<std::pair<std::string, crypto::Commitment>> view_commitments;
        std::vector<std::pair<std::string, crypto::Commitment>> view_logup_commitments;
        std::vector<std::pair<std::string, crypto::Commitment>> aggregation_commitments;
        std::vector<std::pair<std::string, crypto::Commitment>> basis_commitments;
        std::vector<std::pair<std::string, crypto::Commitment>> crpc_commitments;
        std::vector<std::pair<std::string, crypto::Commitment>> relation_commitments;
        std::vector<BindingEvaluation> evaluations;
        std::vector<MetadataReference> opening_references;
    };

    struct PiBindProof
    {
        std::vector<BindingSubproof> subproofs;
    };

    struct Proof
    {
        PublicMetadata public_metadata;
        std::vector<std::string> block_order;
        std::vector<std::pair<std::string, crypto::Commitment>> dynamic_commitments;
        std::vector<std::pair<std::string, crypto::Commitment>> quotient_commitments;
        std::vector<std::pair<std::string, DomainOpeningBundle>> domain_openings;
        std::vector<std::pair<std::string, algebra::FieldElement>> external_evaluations;
        std::vector<PayloadValueKind> external_evaluation_kinds;
        std::vector<MetadataReference> route_references;
        std::vector<algebra::FieldElement> public_logits;
        std::vector<PointGroupOpening> opening_groups;
        PiBindProof pi_bind;
        std::map<std::string, algebra::FieldElement> challenges;
    };

    struct RunMetrics
    {
        std::string backend_name;
        std::string crypto_backend_name;
        std::string algebra_backend_name;
        std::string compute_backend_name;
        std::string config;
        std::string dataset;
        std::size_t node_count = 0;
        std::size_t edge_count = 0;
        double context_build_ms = 0.0;
        double total_wall_clock_ms = 0.0;
        double formal_time_sum_ms = 0.0;
        double non_formal_overhead_ms = 0.0;
        double reference_forward_ms = 0.0;
        double payload_audit_ms = 0.0;
        double reference_consistency_ms = 0.0;
        bool reference_consistency_passed = false;
        bool security_budget_passed = false;
        std::string security_budget_ledger_digest;
        std::string security_budget_exact;
        bool tamper_tests_passed = false;
        std::string tamper_test_case_map;
        double tamper_tests_ms = 0.0;
        double export_artifact_ms = 0.0;
        double export_manifest_validation_ms = 0.0;
        double profile_logging_overhead_ms = 0.0;
        double forward_ms = 0.0;
        double feature_projection_ms = 0.0;
        double hidden_forward_projection_ms = 0.0;
        double hidden_forward_attention_ms = 0.0;
        double hidden_forward_activation_ms = 0.0;
        double hidden_concat_ms = 0.0;
        double output_forward_projection_ms = 0.0;
        double output_forward_attention_ms = 0.0;
        double output_forward_activation_ms = 0.0;
        double trace_generation_ms = 0.0;
        double trace_misc_ms = 0.0;
        double witness_materialization_ms = 0.0;
        double lookup_trace_ms = 0.0;
        double lookup_table_pack_ms = 0.0;
        double lookup_query_pack_ms = 0.0;
        double lookup_key_build_ms = 0.0;
        double lookup_multiplicity_ms = 0.0;
        double lookup_inverse_ms = 0.0;
        double lookup_accumulator_ms = 0.0;
        double lookup_state_machine_ms = 0.0;
        double lookup_selector_mask_ms = 0.0;
        double lookup_public_helper_ms = 0.0;
        double lookup_copy_convert_ms = 0.0;
        double route_trace_ms = 0.0;
        double psq_trace_ms = 0.0;
        double zkmap_trace_ms = 0.0;
        double state_machine_trace_ms = 0.0;
        double padding_selector_trace_ms = 0.0;
        double public_poly_trace_ms = 0.0;
        double hidden_head_trace_ms = 0.0;
        double hidden_projection_trace_ms = 0.0;
        double hidden_src_attention_trace_ms = 0.0;
        double hidden_dst_attention_trace_ms = 0.0;
        double hidden_edge_score_trace_ms = 0.0;
        double hidden_softmax_chain_trace_ms = 0.0;
        double hidden_h_star_trace_ms = 0.0;
        double hidden_h_agg_pre_star_trace_ms = 0.0;
        double hidden_h_agg_star_trace_ms = 0.0;
        double hidden_route_trace_ms = 0.0;
        double hidden_copy_convert_ms = 0.0;
        double output_head_trace_ms = 0.0;
        double route_pack_residual_ms = 0.0;
        double selector_padding_residual_ms = 0.0;
        double public_poly_residual_ms = 0.0;
        double hidden_output_object_residual_ms = 0.0;
        double shared_helper_build_ms = 0.0;
        double field_conversion_residual_ms = 0.0;
        double copy_move_residual_ms = 0.0;
        double trace_finalize_ms = 0.0;
        double fh_table_materialization_ms = 0.0;
        double fh_query_materialization_ms = 0.0;
        double fh_multiplicity_build_ms = 0.0;
        double fh_accumulator_build_ms = 0.0;
        double fh_interpolation_ms = 0.0;
        double fh_lagrange_eval_ms = 0.0;
        double fh_barycentric_weight_fetch_ms = 0.0;
        double fh_point_powers_ms = 0.0;
        double fh_public_poly_interp_ms = 0.0;
        double fh_feature_poly_interp_ms = 0.0;
        double fh_fold_prep_ms = 0.0;
        double fh_opening_eval_prep_ms = 0.0;
        double fh_copy_convert_ms = 0.0;
        double fh_eval_prep_ms = 0.0;
        double fh_public_eval_reuse_ms = 0.0;
        double fh_quotient_assembly_ms = 0.0;
        double fh_open_gather_ms = 0.0;
        double fh_open_witness_ms = 0.0;
        double fh_open_fold_prepare_ms = 0.0;
        double load_static_ms = 0.0;
        double fft_plan_ms = 0.0;
        double srs_prepare_ms = 0.0;
        double commit_dynamic_ms = 0.0;
        double dynamic_commit_input_ms = 0.0;
        double dynamic_polynomial_materialization_ms = 0.0;
        double dynamic_commit_pack_ms = 0.0;
        double dynamic_fft_ms = 0.0;
        double dynamic_domain_convert_ms = 0.0;
        double dynamic_copy_convert_ms = 0.0;
        double dynamic_commit_msm_ms = 0.0;
        double derived_commitment_point_ms = 0.0;
        double dynamic_commit_finalize_ms = 0.0;
        double dynamic_bundle_finalize_ms = 0.0;
        double commitment_time_ms = 0.0;
        double quotient_build_ms = 0.0;
        double quotient_t_fh_ms = 0.0;
        double quotient_t_edge_ms = 0.0;
        double quotient_t_in_ms = 0.0;
        double quotient_t_d_h_ms = 0.0;
        double quotient_t_cat_ms = 0.0;
        double quotient_t_c_ms = 0.0;
        double quotient_t_n_ms = 0.0;
        double quotient_public_eval_ms = 0.0;
        double quotient_commitment_ms = 0.0;
        double quotient_fold_prepare_ms = 0.0;
        double quotient_copy_convert_ms = 0.0;
        double domain_opening_ms = 0.0;
        double point_group_registry_ms = 0.0;
        double domain_eval_gather_ms = 0.0;
        double domain_open_witness_ms = 0.0;
        double domain_open_denominator_inverse_ms = 0.0;
        double domain_open_denominator_inverse_reuse_ms = 0.0;
        std::size_t domain_open_denominator_inverse_reuse_hits = 0;
        std::size_t domain_open_denominator_inverse_cache_misses = 0;
        std::size_t domain_open_point_parallel_batches = 0;
        std::size_t domain_open_point_parallel_workers = 0;
        std::size_t domain_open_point_msm_worker_cap = 0;
        double domain_open_fh_ms = 0.0;
        double domain_open_edge_ms = 0.0;
        double domain_open_in_ms = 0.0;
        double domain_open_d_h_ms = 0.0;
        double domain_open_cat_ms = 0.0;
        double domain_open_c_ms = 0.0;
        double domain_open_n_ms = 0.0;
        double external_opening_ms = 0.0;
        double proof_serialization_ms = 0.0;
        double prove_finalize_ms = 0.0;
        double prove_accounted_ms = 0.0;
        double prove_accounting_gap_ms = 0.0;
        double prove_time_ms = 0.0;
        double prover_total_ms = 0.0;
        double prover_wall_including_audit_ms = 0.0;
        double prover_reference_audit_ms = 0.0;
        double verify_time_ms = 0.0;
        double proof_parse_ms = 0.0;
        double verify_metadata_ms = 0.0;
        double verify_transcript_ms = 0.0;
        double verify_domain_opening_ms = 0.0;
        double verify_point_group_registry_ms = 0.0;
        double verify_quotient_ms = 0.0;
        double verify_external_opening_ms = 0.0;
        double verify_misc_ms = 0.0;
        double verify_accounted_ms = 0.0;
        double verify_accounting_gap_ms = 0.0;
        double verify_fh_ms = 0.0;
        double verify_edge_ms = 0.0;
        double verify_in_ms = 0.0;
        double verify_d_h_ms = 0.0;
        double verify_cat_ms = 0.0;
        double verify_c_ms = 0.0;
        double verify_n_ms = 0.0;
        double verify_public_eval_ms = 0.0;
        double verify_bundle_lookup_ms = 0.0;
        double verify_fold_ms = 0.0;
        double verify_copy_convert_ms = 0.0;
        std::size_t proof_size_bytes = 0;
        bool enabled_fast_msm = false;
        bool enabled_parallel_fft = false;
        bool enabled_packed_fft_backend = false;
        bool enabled_rotated_fft_kernel = false;
        bool enabled_compact_trace_layout = false;
        bool enabled_fast_verify_pairing = false;
        bool enabled_cuda_trace_hotspots = false;
        bool enabled_cuda_field_hotspots = false;
        bool is_full_dataset = false;
        bool gpu_runtime_present = false;
        bool gpu_runtime_loaded = false;
        bool gpu_backend_requested = false;
        std::size_t gpu_kernel_invocation_count = 0;
        std::uint64_t gpu_h2d_bytes = 0;
        std::uint64_t gpu_d2h_bytes = 0;
        std::uint64_t gpu_h2d_operations = 0;
        std::uint64_t gpu_d2h_operations = 0;
        std::uint64_t gpu_stream_synchronizations = 0;
        std::uint64_t gpu_ntt_calls = 0;
        std::uint64_t gpu_intt_calls = 0;
        std::uint64_t gpu_quotient_calls = 0;
        std::uint64_t gpu_t_edge_calls = 0;
        std::uint64_t gpu_t_fh_calls = 0;
        std::uint64_t gpu_opening_fold_calls = 0;
        std::uint64_t gpu_msm_single_calls = 0;
        std::uint64_t gpu_msm_batch_calls = 0;
        std::uint64_t gpu_resident_hits = 0;
        std::uint64_t gpu_resident_misses = 0;
        std::string gpu_device;
        std::string gpu_used_for;
        std::string gpu_fallback_stages;
        std::string gpu_fallback_reason;
        std::vector<DynamicCommitmentObjectProfile> dynamic_commitment_profiles;
        std::unordered_map<std::string, std::string> dynamic_commitment_first_label_by_fingerprint;
        std::string benchmark_mode = "single";
        std::string fft_backend_route = "plain";
        std::string route2_label = "baseline";
        std::string optimization_profile_id;
        std::string optimization_flags;
        std::string hardware_and_device;
        std::string gpu_acceleration_status;
        std::string notes;

        bool timing_state_valid = true;
        std::string timing_state_error;
        bool prove_timer_started = false;
        bool prove_timer_running = false;
        bool prove_timer_finished = false;
        bool reference_audit_timer_running = false;
        std::size_t commitment_timer_depth = 0;
        double prove_timer_commitment_baseline_ms = 0.0;
        std::chrono::steady_clock::time_point prove_timer_wall_start{};
        std::chrono::steady_clock::time_point prove_timer_segment_start{};
        std::chrono::steady_clock::time_point commitment_timer_start{};
    };

    namespace timing
    {
        using Clock = std::chrono::steady_clock;

        inline double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        inline void mark_invalid(RunMetrics *metrics, const char *reason)
        {
            if (metrics == nullptr)
                return;
            metrics->timing_state_valid = false;
            metrics->timing_state_error = reason != nullptr ? reason : "formal timing state is invalid";
        }

        inline void require_valid(const RunMetrics &metrics)
        {
            if (!metrics.timing_state_valid)
                throw std::runtime_error(metrics.timing_state_error.empty() ? "formal timing state is invalid" : metrics.timing_state_error);
        }

        inline void start_prove_timer(RunMetrics *metrics)
        {
            if (metrics == nullptr)
                return;
            require_valid(*metrics);
            if (metrics->prove_timer_started || metrics->prove_timer_finished || metrics->commitment_timer_depth != 0 || metrics->reference_audit_timer_running)
            {
                mark_invalid(metrics, "formal prove timer started in an invalid timing state");
                throw std::runtime_error("formal prove timer started in an invalid timing state");
            }
            const auto started = Clock::now();

            metrics->prove_time_ms = 0.0;
            metrics->prover_total_ms = 0.0;
            metrics->prover_wall_including_audit_ms = 0.0;
            metrics->prover_reference_audit_ms = 0.0;
            metrics->prove_timer_commitment_baseline_ms = metrics->commitment_time_ms;
            metrics->prove_timer_started = true;
            metrics->prove_timer_running = true;
            metrics->prove_timer_finished = false;
            metrics->prove_timer_wall_start = started;
            metrics->prove_timer_segment_start = started;
        }

        inline void finish_prove_timer(RunMetrics *metrics)
        {
            if (metrics == nullptr)
                return;
            require_valid(*metrics);
            if (!metrics->prove_timer_started || !metrics->prove_timer_running || metrics->commitment_timer_depth != 0 ||
                metrics->reference_audit_timer_running)
            {
                mark_invalid(metrics, "formal prove timer finished in an invalid timing state");
                throw std::runtime_error("formal prove timer finished in an invalid timing state");
            }
            const auto finished = Clock::now();
            const auto duration_ms = elapsed_ms(metrics->prove_timer_segment_start, finished);
            if (duration_ms < 0.0)
            {
                mark_invalid(metrics, "formal prove timer measured a negative duration");
                throw std::runtime_error("formal prove timer measured a negative duration");
            }
            metrics->prove_time_ms += duration_ms;
            if (metrics->prove_time_ms < 0.0)
            {
                mark_invalid(metrics, "formal prove timer measured a negative duration");
                throw std::runtime_error("formal prove timer measured a negative duration");
            }
            metrics->prove_timer_started = false;
            metrics->prove_timer_running = false;
            metrics->prove_timer_finished = true;
        }

        inline void begin_commitment(RunMetrics *metrics)
        {
            if (metrics == nullptr)
                return;
            require_valid(*metrics);
            if (metrics->prove_timer_finished || metrics->reference_audit_timer_running)
            {
                mark_invalid(metrics, "commitment started outside a valid proof-production interval");
                throw std::runtime_error("commitment started outside a valid proof-production interval");
            }
            if (metrics->commitment_timer_depth == 0)
            {
                metrics->commitment_timer_start = Clock::now();
                if (metrics->prove_timer_started && metrics->prove_timer_running)
                {
                    metrics->prove_time_ms += elapsed_ms(metrics->prove_timer_segment_start, metrics->commitment_timer_start);
                    metrics->prove_timer_running = false;
                }
            }
            ++metrics->commitment_timer_depth;
        }

        inline void end_commitment(RunMetrics *metrics)
        {
            if (metrics == nullptr)
                return;
            require_valid(*metrics);
            if (metrics->commitment_timer_depth == 0)
            {
                mark_invalid(metrics, "formal commitment timer ended without a matching scope");
                throw std::runtime_error("formal commitment timer ended without a matching scope");
            }
            --metrics->commitment_timer_depth;
            if (metrics->commitment_timer_depth != 0)
                return;

            const auto finished = Clock::now();
            const auto duration_ms = elapsed_ms(metrics->commitment_timer_start, finished);
            if (duration_ms < 0.0)
            {
                mark_invalid(metrics, "formal commitment timer measured a negative duration");
                throw std::runtime_error("formal commitment timer measured a negative duration");
            }
            metrics->commitment_time_ms += duration_ms;
            if (metrics->commitment_time_ms < 0.0)
            {
                mark_invalid(metrics, "formal commitment timer measured a negative duration");
                throw std::runtime_error("formal commitment timer measured a negative duration");
            }
            if (metrics->prove_timer_started)
            {
                metrics->prove_timer_segment_start = finished;
                metrics->prove_timer_running = true;
            }
        }

        inline void assert_quiescent(RunMetrics &metrics)
        {
            require_valid(metrics);
            if (metrics.commitment_timer_depth != 0 || metrics.prove_timer_started || metrics.prove_timer_running || metrics.reference_audit_timer_running)
            {
                mark_invalid(&metrics, "formal timing state is not quiescent");
                throw std::runtime_error("formal timing state is not quiescent");
            }
        }

        inline void assert_prove_finished(RunMetrics &metrics)
        {
            require_valid(metrics);
            if (!metrics.prove_timer_finished || metrics.prove_timer_started || metrics.prove_timer_running || metrics.commitment_timer_depth != 0 ||
                metrics.reference_audit_timer_running)
            {
                mark_invalid(&metrics, "formal prove timer has not completed canonical serialization");
                throw std::runtime_error("formal prove timer has not completed canonical serialization");
            }
        }

        class ScopedReferenceAudit
        {
        public:
            explicit ScopedReferenceAudit(RunMetrics *metrics) : metrics_(metrics)
            {
                if (metrics_ == nullptr)
                    return;
                require_valid(*metrics_);
                if (metrics_->reference_audit_timer_running || metrics_->commitment_timer_depth != 0 || metrics_->prove_timer_finished)
                {
                    mark_invalid(metrics_, "reference audit started in an invalid timing state");
                    throw std::runtime_error("reference audit started in an invalid timing state");
                }
                started_ = Clock::now();
                resume_prove_ = metrics_->prove_timer_running;
                if (resume_prove_)
                {
                    metrics_->prove_time_ms += elapsed_ms(metrics_->prove_timer_segment_start, started_);
                    metrics_->prove_timer_running = false;
                }
                metrics_->reference_audit_timer_running = true;
            }
            ScopedReferenceAudit(const ScopedReferenceAudit &) = delete;
            ScopedReferenceAudit &operator=(const ScopedReferenceAudit &) = delete;

            ~ScopedReferenceAudit() noexcept
            {
                if (metrics_ == nullptr)
                    return;
                const auto finished = Clock::now();
                const auto duration = elapsed_ms(started_, finished);
                if (duration < 0.0 || !metrics_->reference_audit_timer_running || metrics_->commitment_timer_depth != 0)
                {
                    mark_invalid(metrics_, "reference audit failed to close");
                    return;
                }
                metrics_->prover_reference_audit_ms += duration;
                metrics_->reference_audit_timer_running = false;
                if (resume_prove_)
                {
                    if (!metrics_->prove_timer_started || metrics_->prove_timer_finished)
                    {
                        mark_invalid(metrics_, "reference audit cannot resume the prover clock");
                        return;
                    }
                    metrics_->prove_timer_segment_start = finished;
                    metrics_->prove_timer_running = true;
                }
            }

        private:
            RunMetrics *metrics_ = nullptr;
            Clock::time_point started_{};
            bool resume_prove_ = false;
        };

        class ScopedCommitment
        {
        public:
            explicit ScopedCommitment(RunMetrics *metrics) : metrics_(metrics)
            {
                begin_commitment(metrics_);
                active_ = metrics_ != nullptr;
            }

            ScopedCommitment(const ScopedCommitment &) = delete;
            ScopedCommitment &operator=(const ScopedCommitment &) = delete;

            ~ScopedCommitment() noexcept
            {
                if (!active_)
                    return;
                try
                {
                    end_commitment(metrics_);
                }
                catch (...)
                {
                    mark_invalid(metrics_, "formal commitment scope failed to close");
                }
            }

        private:
            RunMetrics *metrics_ = nullptr;
            bool active_ = false;
        };
    }

}
