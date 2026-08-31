#pragma once
// 定义协议层共享的数据契约，
// 包括 context、trace、proof 与正式运行指标，是 artifact contract 的核心来源。

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

namespace gatzk::protocol
{

    struct ProtocolCompilerArtifact;

    inline constexpr const char *kFormalTimingSchema =
        "exclusive_commit_prove_verify_v2";

    // 固定静态查表集合，对应 formal note 中的激活、指数与范围表。
    struct StaticTables
    {
        std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> lrelu;
        std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> elu;
        std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> exp;
        std::vector<algebra::FieldElement> range;
    };

    // 单个注意力头使用的局部工作域集合。
    struct AttentionHeadDomains
    {
        std::shared_ptr<algebra::RootOfUnityDomain> in;
        std::shared_ptr<algebra::RootOfUnityDomain> d;
    };

    // 一次 formal run 中所有 proving / verifying 工作域的统一持有者。
    struct WorkDomains
    {
        std::shared_ptr<algebra::RootOfUnityDomain> fh;
        std::shared_ptr<algebra::RootOfUnityDomain> edge;
        std::shared_ptr<algebra::RootOfUnityDomain> in;
        std::shared_ptr<algebra::RootOfUnityDomain> d;
        std::shared_ptr<algebra::RootOfUnityDomain> cat;
        std::shared_ptr<algebra::RootOfUnityDomain> c;
        std::shared_ptr<algebra::RootOfUnityDomain> n;
        // Formal domains retain the layer/head coordinates.  The checkpoint
        // compatibility view is kept on ModelParameters, not here.
        std::vector<std::vector<AttentionHeadDomains>> hidden_layers;
        std::optional<AttentionHeadDomains> output_head;
    };

    // 对外公开且会进入 transcript / manifest / verifier 的元数据契约。
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

    // 构建一次 formal 证明所需的静态上下文。
    // 这里不保存动态 witness，只保存数据集、模型、SRS、公开列和静态承诺。
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
        std::unordered_map<std::string, crypto::Commitment> public_commitments;
        // Prover-side opening material for public FH polynomials. This does not change public
        // polynomial semantics or transcript-visible commitments.
        std::unordered_map<std::string, std::shared_ptr<const algebra::Polynomial>> public_fh_coefficient_materials;
        std::unordered_map<std::string, crypto::Commitment> static_commitments;
        std::shared_ptr<const ProtocolCompilerArtifact> compiler_artifact;
        // Exact immutable artifact object that passed the full setup-time
        // canonical rebuild.  Proof verification rejects a swapped pointer but
        // does not repeat the expensive compiler pass inside verify_time.
        const ProtocolCompilerArtifact *validated_compiler_artifact_identity = nullptr;
        // Internal verifier compatibility projection marker.  A context with
        // this bit set carries shape-only model shells and must never be used
        // by the prover or by an artifact compiler that needs model values.
        bool verifier_shape_only = false;
    };

    // The verifier must not receive the proving context.  This deliberately
    // repeats only public configuration fields; in particular it has no
    // project/checkpoint/SRS paths, seed, model tensors, proving key, or
    // opening-polynomial storage.
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
        std::string security_profile_id = "zkgat_secure_profile_v1";
        std::size_t hidden_dim = 4;
        std::size_t num_classes = 2;
        std::size_t range_bits = 10;
        std::size_t layer_count = 0;
        std::size_t K_out = 1;
        std::size_t batch_graphs = 1;
        std::vector<util::HiddenLayerProfile> hidden_profile;
        std::vector<std::size_t> d_in_profile;
    };

    // Shape-only model view.  Head entries intentionally carry no tensor or
    // floating-point parameter fields; they exist solely to preserve the
    // canonical layer/local-head iteration order used by the verifier.
    struct VerifierModelHeadShape {};

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
        bool has_real_multihead = false;
        std::string model_arch_id;
        std::string model_param_id;
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
        // Public SRS-manifest identity and verifier-side catalogue metadata.
        // Path fields and all prover material are deliberately absent.
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
        std::string generator_version;
        std::string encoding;
        std::string protocol_schema_id;
        std::string kzg_assumption_profile_id;
        std::size_t kzg_assumption_degree_bound = 0;
        std::size_t kzg_assumption_security_bits = 0;
        std::string kzg_assumption_domain_catalog_hash;
        std::string domain_generator_catalog_hash;
        std::string security_level;
    };

    // A point-only commitment.  It is intentionally not crypto::Commitment:
    // the latter owns an optional opening_polynomial for the prover.
    struct VerifierPointCommitment
    {
        std::string name;
        crypto::G1Point point;
    };

    // Setup-time, point-only certificate for one model object.  It exposes
    // only the metadata needed to independently rebuild the fixed-point
    // no-wrap ledger; raw tensor values and checkpoint provenance never cross
    // the registration boundary.
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

    // Authenticated VK_model/VK_static material.  Canonical serialization is
    // length-delimited and SHA-256 hashed; entry order is part of the digest.
    struct AuthenticatedVerifierRegistry
    {
        std::string format = "zkgat_verifier_registry_v1";
        std::string registry_id;
        std::string model_arch_id;
        std::string model_param_id;
        std::string quant_cfg_id;
        std::string static_table_id;
        std::string srs_id;
        std::string protocol_schema_id;
        std::string instance_schema_hash;
        std::string srs_manifest_hash;
        std::size_t srs_degree_bound = 0;
        crypto::G1Point srs_g1_generator;
        std::string srs_generator_id;
        std::string srs_generator_version;
        std::string srs_encoding;
        std::string srs_spec_id;
        // Exact hash of the compiler artifact material copied into the SRS
        // directory.  This is distinct from compiler_artifact_digest, which
        // authenticates the typed instance artifact in this registry.
        std::string srs_protocol_compiler_artifact_hash;
        std::string compiler_artifact_digest;
        std::string model_layout_digest;
        std::string quant_chain_digest;
        std::string static_table_digest;
        // These are produced from the complete registered table bytes and
        // scale/bound certificate, rather than fixed marker strings.
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

    // Independent verifier-side context.  The public API below accepts only
    // this type.  It owns point-only registries and a verifier key, never a
    // proving key or raw model/checkpoint material.
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
        std::unordered_map<std::string, algebra::Polynomial> public_polynomials;
        std::unordered_map<std::string, VerifierPointCommitment> public_commitments;
        std::unordered_map<std::string, VerifierPointCommitment> static_commitments;
        std::shared_ptr<const ProtocolCompilerArtifact> compiler_artifact;
        AuthenticatedVerifierRegistry registry;
        // Digest captured from the authenticated registry file.  Verifier
        // entry points reject any post-load mutation of the registry or its
        // canonical metadata.
        std::string authenticated_registry_digest;
        // Out-of-band trust anchor supplied by the run configuration.  The
        // registry file's own digest is never accepted as its sole root.
        std::string trusted_registry_digest;
        bool in_process_transition = false;
    };

    // Self-contained, canonical verifier registration artifact.  This is the
    // only wire object accepted by the formal verifier loader.  It owns every
    // public value needed to reconstruct VerifierContext (including the full
    // typed compiler artifact); it deliberately contains no model tensors,
    // checkpoint paths, proving key, or opening polynomial.
    struct VerifierRegistrationArtifact
    {
        std::uint32_t format_version = 1;
        VerifierConfig config;
        data::GraphDataset dataset;
        data::LocalGraph local;
        VerifierModelShape model;
        StaticTables tables;
        WorkDomains domains;
        VerifierSRSMetadata srs;
        crypto::KZGVerifierKey kzg_vk;
        std::unordered_map<std::string, algebra::Polynomial> public_polynomials;
        std::unordered_map<std::string, VerifierPointCommitment> public_commitments;
        std::unordered_map<std::string, VerifierPointCommitment> static_commitments;
        ProtocolCompilerArtifact compiler_artifact;
        AuthenticatedVerifierRegistry registry;

        std::vector<std::uint8_t> canonical_bytes() const;
        std::string canonical_digest() const;
    };

    struct PiBindProof;

    // `build_trace` 输出的全部动态对象。
    // 这里的标签、承诺顺序与挑战映射会直接进入 proof 生成阶段。
    struct TraceArtifacts
    {
        // Read-only instance compiler contract used while materializing blinded
        // objects. It is execution context and is never serialized into proof.
        std::shared_ptr<const ProtocolCompilerArtifact> compiler_artifact;
        // Per-proof execution cache for identical semantic evaluation columns.
        // It never crosses proof boundaries and is never serialized.
        std::shared_ptr<crypto::EvaluationCommitmentCache>
            evaluation_commitment_cache;
        PublicMetadata public_metadata;
        // 对超大 evaluation 列的临时落盘描述，只是执行层优化，不改变多项式语义。
        struct SpilledEvaluationPolynomial
        {
            std::string path;
            std::string domain_name;
            std::size_t size = 0;
            // Exact per-proof identity of an immutable shared evaluation buffer.
            // Execution-only: it is used to share semantic cosets across
            // independently blinded columns and is never serialized.
            std::uintptr_t semantic_source_id = 0;
            // Execution-only result of a proof-local scan of the compact shared
            // semantic source.  This metadata is never serialized into proof.
            bool compact_semantic_source_all_zero = false;
            std::vector<algebra::FieldElement> zero_domain_blinder;
        };

        struct CompactSemanticSourceZeroCacheEntry
        {
            std::string domain_name;
            std::size_t size = 0;
            bool all_zero = false;
        };
        struct SemanticSpillSourceCacheEntry
        {
            // Keep only a weak execution owner so this cache never extends the
            // natural lifetime of a large trace aggregate.
            std::weak_ptr<
                const std::vector<algebra::FieldElement>>
                values;

            std::string path;
            std::string domain_name;
            std::size_t size = 0;
            bool compact_semantic_source_all_zero = false;
        };

        // 原始列与矩阵，主要服务 artifact 导出与后续按标签取值。
        std::unordered_map<std::string, std::vector<algebra::FieldElement>> columns;
        std::unordered_map<std::string, model::Matrix> matrices;
        // 多项式视图与可选落盘信息。
        std::unordered_map<std::string, algebra::Polynomial> polynomials;
        std::unordered_map<std::string, SpilledEvaluationPolynomial> spilled_polynomials;
        // Proof-local execution cache keyed by exact shared-buffer identity.
        // It prevents Limb/Query aliases from rescanning the same compact source
        // and is never serialized.
        std::unordered_map<
            std::uintptr_t,
            CompactSemanticSourceZeroCacheEntry>
            compact_semantic_source_zero_cache;
        // One physical canonical semantic spill per exact shared source/domain.
        // Independently blinded formal objects reuse only the value file; every
        // SpilledEvaluationPolynomial still owns its own zero-domain blinder.
        // Execution-only and never serialized.
        std::unordered_map<
            std::string,
            SemanticSpillSourceCacheEntry>
            semantic_spill_source_cache;
        // Owns the per-proof spill directory.  The custom lease deleter removes
        // the directory after the last TraceArtifacts copy is released.
        std::shared_ptr<const std::string> spill_directory_lease;
        std::unordered_map<std::string, std::string> polynomial_domains;
        // 动态承诺、正式承诺顺序以及挑战重放结果。
        std::unordered_map<std::string, crypto::Commitment> commitments;
        std::vector<std::string> commitment_order;
        std::map<std::string, algebra::FieldElement> challenges;
        // verifier 直接消费的外点值。route 终值不是 proof 标量；
        // 它们只能通过 S_route 中的全局 point-group 引用进入正式关系。
        std::map<std::string, algebra::FieldElement> external_evaluations;
        // 主协议的公开 logits，按 N x C 行优先展平。这一份数据必须由
        // verifier 独立重建多项式求值，不得用 trace 内部 hash 代替。
        std::vector<algebra::FieldElement> public_logits;
        std::string public_logits_hash;
        std::string reference_fp32_logits_hash;
        std::string reference_quant_logits_hash;
        std::string secure_trace_logits_hash;
        bool reference_consistency_passed = false;
        std::string proof_public_output_hash;
        std::shared_ptr<PiBindProof> pi_bind;
    };

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

    // 单个工作域 opening bundle 的 proof 载体。
    // 标签顺序由 schema 固定，proof_size_bytes 只统计真正随 proof 发送的点值与 witness。
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

    // One physical KZG witness is owned by exactly one canonical point group.
    // Claims and their values remain in Eval_dom/Eval_ext/Eval_bind and are
    // reconstructed from the compiler schema; they are never duplicated here.
    struct PointGroupOpening
    {
        std::string group_id;
        std::string owner_block;
        algebra::FieldElement point = algebra::FieldElement::zero();
        crypto::G1Point witness;
    };

    // Pi_bind carries commitments and metadata references only.  KZG witnesses remain
    // owned by Open_dom/W_ext; a binding subproof can only refer to those owner groups.
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

    // Public, verifier-reconstructible metadata for one indexed-view relation.
    // None of these fields is prover-selectable witness data: the instance compiler
    // fixes the index map, selectors, multiplicity rule, challenge labels and the
    // exact Com_viewLogUp order.
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

    // Canonical metadata for the KZG same-point normalization used by a binding
    // relation.  source_linear_maps describes the public linear reconstruction of
    // F_j from already-owned commitments; points and deltas are in canonical point
    // order, with points.front() the public pivot.
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

    // Canonical instance metadata for a coefficient-tail/chunk basis link.  The
    // transcript derives gamma after the coefficient and chunk commitments; the
    // edge-domain accumulator is then Taylor-shifted back to gamma so the verifier
    // can check the link with ordinary KZG point groups without receiving either
    // private value in the clear.
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

    // Canonical node-domain representation of one private matrix.  The column
    // family is the only H_N trace for the matrix.  compressed_object is committed
    // only after the column commitments have fixed compression_challenge_label;
    // its equality to the challenge-weighted canonical columns is enforced in the
    // global N-domain quotient.
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

    // One N-domain table / edge-domain query broadcast used by an aggregation
    // binding.  The two accumulators are related by the registered KZG Taylor
    // shift; no route terminal is serialized as a scalar.
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

    // Independent post-commit compression and PSQ proof for one attention
    // aggregation matrix.  commitment_order is the exact Pi_bind.Com_agg order.
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

    // Cross-domain terminal equality for one of the protocol's native LogUp
    // routes.  The edge accumulator is Taylor-shifted onto a verifier-chosen
    // relation point and the node/edge terminal difference is opened as a
    // verifier-reconstructed virtual commitment.  No route terminal scalar is
    // carried by the proof.
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

    // 最终 proof 对象，顺序与字段语义必须与 verifier 和 artifact 导出保持一致。
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

    // 运行期与 benchmark 统一使用的指标表。
    // 这里记录三项正式时间、canonical proof bytes 大小及内部热路径拆分。
    // 正式展示面只能读取最小指标；细粒度字段只允许停留在内部 manifest。
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
        // Point generation and its required batch synchronization only.  Quotient
        // materialization and proof-object packing remain in prove_time.
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
        // formal proof 的传输尺寸，不是对象内存占用，也不是目录大小。
        std::size_t proof_size_bytes = 0;
        bool enabled_fast_msm = false;
        bool enabled_parallel_fft = false;
        bool enabled_fft_backend_upgrade = false;
        bool enabled_fft_kernel_upgrade = false;
        bool enabled_trace_layout_upgrade = false;
        bool enabled_fast_verify_pairing = false;
        bool enabled_cuda_trace_hotspots = false;
        bool enabled_cuda_field_hotspots = false;
        bool is_full_dataset = false;
        bool gpu_runtime_present = false;
        bool gpu_backend_requested = false;
        std::size_t gpu_kernel_invocation_count = 0;
        std::string gpu_device;
        std::string gpu_used_for;
        std::string gpu_fallback_stages;
        std::string gpu_fallback_reason;
        std::vector<DynamicCommitmentObjectProfile> dynamic_commitment_profiles;
        std::string benchmark_mode = "single";
        std::string fft_backend_route = "plain";
        std::string route2_label = "baseline";
        std::string optimization_profile_id;
        std::string optimization_flags;
        std::string hardware_and_device;
        std::string gpu_acceleration_status;
        std::string notes;

        // Execution-only state for the exclusive prover timer.  These fields
        // are never serialized into a proof or an artifact; they let the
        // commitment scopes pause the prove clock without deriving prove_time
        // as a residual from a larger wall-clock interval.
        bool timing_state_valid = true;
        std::string timing_state_error;
        bool prove_timer_started = false;
        bool prove_timer_running = false;
        bool prove_timer_finished = false;
        std::size_t commitment_timer_depth = 0;
        std::chrono::steady_clock::time_point prove_timer_segment_start{};
        std::chrono::steady_clock::time_point commitment_timer_start{};
    };

    namespace timing
    {
        using Clock = std::chrono::steady_clock;

        inline double elapsed_ms(
            const Clock::time_point &start,
            const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        inline void mark_invalid(
            RunMetrics *metrics,
            const char *reason)
        {
            if (metrics == nullptr)
                return;
            metrics->timing_state_valid = false;
            metrics->timing_state_error = reason != nullptr ? reason : "formal timing state is invalid";
        }

        inline void require_valid(const RunMetrics &metrics)
        {
            if (!metrics.timing_state_valid)
                throw std::runtime_error(
                    metrics.timing_state_error.empty()
                        ? "formal timing state is invalid"
                        : metrics.timing_state_error);
        }

        inline void start_prove_timer(RunMetrics *metrics)
        {
            if (metrics == nullptr)
                return;
            require_valid(*metrics);
            if (metrics->prove_timer_started
                || metrics->prove_timer_finished
                || metrics->commitment_timer_depth != 0)
            {
                mark_invalid(
                    metrics,
                    "formal prove timer started in an invalid timing state");
                throw std::runtime_error(
                    "formal prove timer started in an invalid timing state");
            }
            metrics->prove_time_ms = 0.0;
            metrics->prove_timer_started = true;
            metrics->prove_timer_running = true;
            metrics->prove_timer_finished = false;
            metrics->prove_timer_segment_start = Clock::now();
        }

        inline void finish_prove_timer(RunMetrics *metrics)
        {
            if (metrics == nullptr)
                return;
            require_valid(*metrics);
            if (!metrics->prove_timer_started
                || !metrics->prove_timer_running
                || metrics->commitment_timer_depth != 0)
            {
                mark_invalid(
                    metrics,
                    "formal prove timer finished in an invalid timing state");
                throw std::runtime_error(
                    "formal prove timer finished in an invalid timing state");
            }
            const auto finished = Clock::now();
            const auto duration_ms = elapsed_ms(
                metrics->prove_timer_segment_start,
                finished);
            if (duration_ms < 0.0)
            {
                mark_invalid(
                    metrics,
                    "formal prove timer measured a negative duration");
                throw std::runtime_error(
                    "formal prove timer measured a negative duration");
            }
            metrics->prove_time_ms += duration_ms;
            if (metrics->prove_time_ms < 0.0)
            {
                mark_invalid(
                    metrics,
                    "formal prove timer measured a negative duration");
                throw std::runtime_error(
                    "formal prove timer measured a negative duration");
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
            if (metrics->prove_timer_finished)
            {
                mark_invalid(
                    metrics,
                    "commitment started after formal prove serialization");
                throw std::runtime_error(
                    "commitment started after formal prove serialization");
            }
            if (metrics->commitment_timer_depth == 0)
            {
                metrics->commitment_timer_start = Clock::now();
                if (metrics->prove_timer_started
                    && metrics->prove_timer_running)
                {
                    metrics->prove_time_ms += elapsed_ms(
                        metrics->prove_timer_segment_start,
                        metrics->commitment_timer_start);
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
                mark_invalid(
                    metrics,
                    "formal commitment timer ended without a matching scope");
                throw std::runtime_error(
                    "formal commitment timer ended without a matching scope");
            }
            --metrics->commitment_timer_depth;
            if (metrics->commitment_timer_depth != 0)
                return;

            const auto finished = Clock::now();
            const auto duration_ms = elapsed_ms(
                metrics->commitment_timer_start,
                finished);
            if (duration_ms < 0.0)
            {
                mark_invalid(
                    metrics,
                    "formal commitment timer measured a negative duration");
                throw std::runtime_error(
                    "formal commitment timer measured a negative duration");
            }
            metrics->commitment_time_ms += duration_ms;
            if (metrics->commitment_time_ms < 0.0)
            {
                mark_invalid(
                    metrics,
                    "formal commitment timer measured a negative duration");
                throw std::runtime_error(
                    "formal commitment timer measured a negative duration");
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
            if (metrics.commitment_timer_depth != 0
                || metrics.prove_timer_started
                || metrics.prove_timer_running)
            {
                mark_invalid(
                    &metrics,
                    "formal timing state is not quiescent");
                throw std::runtime_error(
                    "formal timing state is not quiescent");
            }
        }

        inline void assert_prove_finished(RunMetrics &metrics)
        {
            require_valid(metrics);
            if (!metrics.prove_timer_finished
                || metrics.prove_timer_started
                || metrics.prove_timer_running
                || metrics.commitment_timer_depth != 0)
            {
                mark_invalid(
                    &metrics,
                    "formal prove timer has not completed canonical serialization");
                throw std::runtime_error(
                    "formal prove timer has not completed canonical serialization");
            }
        }

        class ScopedCommitment
        {
        public:
            explicit ScopedCommitment(RunMetrics *metrics)
                : metrics_(metrics)
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
                    // A commitment scope must not hide the original proving
                    // failure.  The normal path is checked by the timer
                    // state transitions above.
                    mark_invalid(
                        metrics_,
                        "formal commitment scope failed to close");
                }
            }

        private:
            RunMetrics *metrics_ = nullptr;
            bool active_ = false;
        };
    } // namespace timing

} // namespace gatzk::protocol
