#pragma once
// 本文件定义 protocol 层共享的数据契约，
// 包括 context、trace、proof 与 benchmark 指标，是 artifact contract 的核心真源。

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/data/dataset.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::protocol {

// 固定静态查表集合，对应 formal note 中的激活、指数与范围表。
struct StaticTables {
    std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> lrelu;
    std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> elu;
    std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> exp;
    std::vector<algebra::FieldElement> range;
};

// 单个注意力头使用的局部工作域集合。
struct AttentionHeadDomains {
    std::shared_ptr<algebra::RootOfUnityDomain> in;
    std::shared_ptr<algebra::RootOfUnityDomain> d;
};

// 一次 formal run 中所有 proving / verifying 工作域的统一持有者。
struct WorkDomains {
    std::shared_ptr<algebra::RootOfUnityDomain> fh;
    std::shared_ptr<algebra::RootOfUnityDomain> edge;
    std::shared_ptr<algebra::RootOfUnityDomain> in;
    std::shared_ptr<algebra::RootOfUnityDomain> d;
    std::shared_ptr<algebra::RootOfUnityDomain> cat;
    std::shared_ptr<algebra::RootOfUnityDomain> c;
    std::shared_ptr<algebra::RootOfUnityDomain> n;
    std::vector<AttentionHeadDomains> hidden_heads;
    std::optional<AttentionHeadDomains> output_head;
};

// 对外公开且会进入 transcript / manifest / verifier 的元数据契约。
struct PublicMetadata {
    std::string protocol_id;
    std::string dataset_name;
    std::string task_type;
    std::string report_unit;
    std::string graph_count;
    std::string L;
    std::string hidden_profile;
    std::string d_in_profile;
    std::string K_out;
    std::string C;
    std::string batching_rule;
    std::string subgraph_rule;
    std::string self_loop_rule;
    std::string edge_sort_rule;
    std::string chunking_rule;
    std::string model_arch_id;
    std::string model_param_id;
    std::string static_table_id;
    std::string quant_cfg_id;
    std::string domain_cfg;
    std::string dim_cfg;
    std::string encoding_id;
    std::string padding_rule_id;
    std::string degree_bound_id;
};

// 构建一次 formal 证明所需的静态上下文。
// 这里不保存动态 witness，只保存数据集、模型、SRS、公开列和静态承诺。
struct ProtocolContext {
    util::AppConfig config;
    data::GraphDataset dataset;
    data::LocalGraph local;
    model::ModelParameters model;
    StaticTables tables;
    WorkDomains domains;
    crypto::KZGKeyPair kzg;
    std::unordered_map<std::string, algebra::Polynomial> public_polynomials;
    std::unordered_map<std::string, crypto::Commitment> public_commitments;
    std::unordered_map<std::string, crypto::Commitment> static_commitments;
};

// `build_trace` 输出的全部动态对象。
// 这里的标签、承诺顺序与挑战映射会直接进入 proof 生成阶段。
struct TraceArtifacts {
    // 对超大 evaluation 列的临时落盘描述，只是执行层优化，不改变多项式语义。
    struct SpilledEvaluationPolynomial {
        std::string path;
        std::string domain_name;
        std::size_t size = 0;
    };

    // 原始列与矩阵，主要服务 artifact 导出与后续按标签取值。
    std::unordered_map<std::string, std::vector<algebra::FieldElement>> columns;
    std::unordered_map<std::string, model::Matrix> matrices;
    // 多项式视图与可选落盘信息。
    std::unordered_map<std::string, algebra::Polynomial> polynomials;
    std::unordered_map<std::string, SpilledEvaluationPolynomial> spilled_polynomials;
    std::unordered_map<std::string, std::string> polynomial_domains;
    // 动态承诺、正式承诺顺序以及挑战重放结果。
    std::unordered_map<std::string, crypto::Commitment> commitments;
    std::vector<std::string> commitment_order;
    std::map<std::string, algebra::FieldElement> challenges;
    // verifier 直接消费的外点值与 witness 标量。
    std::map<std::string, algebra::FieldElement> external_evaluations;
    std::map<std::string, algebra::FieldElement> witness_scalars;
};

// 单个工作域 opening bundle 的 proof 载体。
// 标签顺序由 schema 固定，proof_size_bytes 只统计真正随 proof 发送的点值与 witness。
struct DomainOpeningBundle {
    std::vector<algebra::FieldElement> points;
    std::vector<std::pair<std::string, std::vector<algebra::FieldElement>>> values;
    crypto::G1Point witness;
};

// 最终 proof 对象，顺序与字段语义必须与 verifier 和 artifact 导出保持一致。
struct Proof {
    PublicMetadata public_metadata;
    std::vector<std::string> block_order;
    std::vector<std::pair<std::string, crypto::Commitment>> dynamic_commitments;
    std::vector<std::pair<std::string, crypto::Commitment>> quotient_commitments;
    std::vector<std::pair<std::string, DomainOpeningBundle>> domain_openings;
    std::vector<std::pair<std::string, algebra::FieldElement>> external_evaluations;
    std::vector<std::pair<std::string, algebra::FieldElement>> witness_scalars;
    crypto::G1Point external_witness;
    std::map<std::string, algebra::FieldElement> challenges;
};

// 运行期与 benchmark 统一使用的指标表。
// 这里既记录正式四项指标，也记录热路径拆分，方便 tests / manifests / README 对齐。
// 正式展示面只能读取最小指标；细粒度字段只允许停留在内部 manifest。
struct RunMetrics {
    std::string backend_name;
    std::string crypto_backend_name;
    std::string algebra_backend_name;
    std::string compute_backend_name;
    std::string config;
    std::string dataset;
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    double context_build_ms = 0.0;
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
    double quotient_bundle_pack_ms = 0.0;
    double quotient_fold_prepare_ms = 0.0;
    double quotient_copy_convert_ms = 0.0;
    double domain_opening_ms = 0.0;
    double domain_eval_gather_ms = 0.0;
    double domain_open_witness_ms = 0.0;
    double domain_open_fh_ms = 0.0;
    double domain_open_edge_ms = 0.0;
    double domain_open_in_ms = 0.0;
    double domain_open_d_h_ms = 0.0;
    double domain_open_cat_ms = 0.0;
    double domain_open_c_ms = 0.0;
    double domain_open_n_ms = 0.0;
    double external_opening_ms = 0.0;
    double prove_finalize_ms = 0.0;
    double prove_accounted_ms = 0.0;
    double prove_accounting_gap_ms = 0.0;
    double prove_time_ms = 0.0;
    double verify_time_ms = 0.0;
    double verify_metadata_ms = 0.0;
    double verify_transcript_ms = 0.0;
    double verify_domain_opening_ms = 0.0;
    double verify_quotient_ms = 0.0;
    double verify_external_fold_ms = 0.0;
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
    bool is_cold_run = true;
    bool is_full_dataset = false;
    bool gpu_runtime_present = false;
    std::string benchmark_mode = "single";
    std::string fft_backend_route = "plain";
    std::string route2_label = "baseline";
    std::string notes;
};

}  // namespace gatzk::protocol
