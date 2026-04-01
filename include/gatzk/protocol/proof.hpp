#pragma once

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

struct StaticTables {
    std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> lrelu;
    std::vector<std::pair<algebra::FieldElement, algebra::FieldElement>> exp;
    std::vector<algebra::FieldElement> range;
};

struct AttentionHeadDomains {
    std::shared_ptr<algebra::RootOfUnityDomain> in;
    std::shared_ptr<algebra::RootOfUnityDomain> d;
};

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

struct PublicMetadata {
    std::string protocol_id;
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

struct TraceArtifacts {
    std::unordered_map<std::string, std::vector<algebra::FieldElement>> columns;
    std::unordered_map<std::string, model::Matrix> matrices;
    std::unordered_map<std::string, algebra::Polynomial> polynomials;
    std::unordered_map<std::string, std::string> polynomial_domains;
    std::unordered_map<std::string, crypto::Commitment> commitments;
    std::vector<std::string> commitment_order;
    std::map<std::string, algebra::FieldElement> challenges;
    std::map<std::string, algebra::FieldElement> external_evaluations;
    std::map<std::string, algebra::FieldElement> witness_scalars;
};

struct DomainOpeningBundle {
    std::vector<algebra::FieldElement> points;
    std::vector<std::pair<std::string, std::vector<algebra::FieldElement>>> values;
    crypto::G1Point witness;
};

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

struct RunMetrics {
    std::string backend_name;
    std::string config;
    std::string dataset;
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
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
    double witness_materialization_ms = 0.0;
    double lookup_trace_ms = 0.0;
    double route_trace_ms = 0.0;
    double psq_trace_ms = 0.0;
    double zkmap_trace_ms = 0.0;
    double load_static_ms = 0.0;
    double fft_plan_ms = 0.0;
    double srs_prepare_ms = 0.0;
    double commit_dynamic_ms = 0.0;
    double dynamic_commit_input_ms = 0.0;
    double dynamic_polynomial_materialization_ms = 0.0;
    double dynamic_commit_msm_ms = 0.0;
    double dynamic_commit_finalize_ms = 0.0;
    double quotient_build_ms = 0.0;
    double quotient_t_fh_ms = 0.0;
    double quotient_t_edge_ms = 0.0;
    double quotient_t_in_ms = 0.0;
    double quotient_t_d_h_ms = 0.0;
    double quotient_t_cat_ms = 0.0;
    double quotient_t_c_ms = 0.0;
    double quotient_t_n_ms = 0.0;
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
    double prove_time_ms = 0.0;
    double verify_time_ms = 0.0;
    double verify_metadata_ms = 0.0;
    double verify_transcript_ms = 0.0;
    double verify_domain_opening_ms = 0.0;
    double verify_quotient_ms = 0.0;
    double verify_external_fold_ms = 0.0;
    double verify_fh_ms = 0.0;
    double verify_edge_ms = 0.0;
    double verify_in_ms = 0.0;
    double verify_d_h_ms = 0.0;
    double verify_cat_ms = 0.0;
    double verify_c_ms = 0.0;
    double verify_n_ms = 0.0;
    std::size_t proof_size_bytes = 0;
    bool enabled_fast_msm = false;
    bool enabled_parallel_fft = false;
    bool enabled_fft_backend_upgrade = false;
    bool enabled_fft_kernel_upgrade = false;
    bool enabled_trace_layout_upgrade = false;
    bool enabled_fast_verify_pairing = false;
    bool is_cold_run = true;
    bool is_full_dataset = false;
    std::string fft_backend_route = "legacy";
    std::string notes;
};

}  // namespace gatzk::protocol
