#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/data/dataset.hpp"

namespace gatzk::model {

using Matrix = std::vector<std::vector<algebra::FieldElement>>;
using FloatMatrix = std::vector<std::vector<double>>;

struct DenseTensor {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

struct AttentionHeadParameters {
    FloatMatrix seq_kernel_fp;
    std::vector<double> attn_dst_kernel_fp;
    double attn_dst_bias_fp = 0.0;
    std::vector<double> attn_src_kernel_fp;
    double attn_src_bias_fp = 0.0;
    std::vector<double> output_bias_fp;
};

struct HeadForwardTrace {
    FloatMatrix H_prime;
    std::vector<double> E_src;
    std::vector<double> E_dst;
    std::vector<double> S;
    std::vector<double> Z;
    std::vector<double> M;
    std::vector<double> Delta;
    std::vector<double> U;
    std::vector<double> Sum;
    std::vector<double> inv;
    std::vector<double> alpha;
    FloatMatrix H_agg_pre_bias;
    FloatMatrix H_agg;
};

struct HeadForwardProfile {
    double projection_ms = 0.0;
    double attention_ms = 0.0;
    double activation_ms = 0.0;
};

struct ForwardProfile {
    double hidden_projection_ms = 0.0;
    double hidden_attention_ms = 0.0;
    double hidden_activation_ms = 0.0;
    double hidden_concat_ms = 0.0;
    double output_projection_ms = 0.0;
    double output_attention_ms = 0.0;
    double output_activation_ms = 0.0;
};

struct HiddenLayerShape {
    std::size_t head_count = 0;
    std::size_t head_dim = 0;
};

struct HiddenLayerParameters {
    std::size_t layer_index = 0;
    std::size_t input_dim = 0;
    HiddenLayerShape shape;
    std::vector<AttentionHeadParameters> heads;
};

struct OutputLayerParameters {
    std::size_t input_dim = 0;
    std::size_t output_dim = 0;
    std::size_t head_count = 0;
    std::vector<AttentionHeadParameters> heads;
};

struct ModelParameters {
    Matrix W;
    std::vector<algebra::FieldElement> a_src;
    std::vector<algebra::FieldElement> a_dst;
    Matrix W_out;
    std::vector<algebra::FieldElement> b;
    bool has_real_multihead = false;
    std::size_t L = 0;
    std::vector<std::size_t> d_in_profile;
    std::vector<HiddenLayerShape> hidden_profile;
    std::size_t K_out = 1;
    std::size_t C = 0;
    std::vector<HiddenLayerParameters> hidden_layers;
    OutputLayerParameters output_layer;
    std::vector<AttentionHeadParameters> hidden_heads;
    AttentionHeadParameters output_head;
};

struct CheckpointLayerInfo {
    std::size_t layer_index = 0;
    std::size_t input_dim = 0;
    HiddenLayerShape shape;
};

struct CheckpointHeadSpec {
    std::size_t layer_index = 0;
    std::size_t local_head_index = 0;
    std::size_t global_head_index = 0;
    std::string seq_kernel;
    std::string attn_dst_kernel;
    std::string attn_dst_bias;
    std::string attn_src_kernel;
    std::string attn_src_bias;
    std::string output_bias;
};

struct CheckpointOutputHeadSpec {
    std::size_t head_index = 0;
    std::string seq_kernel;
    std::string attn_dst_kernel;
    std::string attn_dst_bias;
    std::string attn_src_kernel;
    std::string attn_src_bias;
    std::string output_bias;
};

struct CheckpointBundleInfo {
    std::string bundle_root;
    std::string family_schema_version;
    std::string output_average_rule;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string degree_bound_id;
    std::size_t layer_count = 0;
    std::vector<std::size_t> d_in_profile;
    std::vector<HiddenLayerShape> hidden_profile;
    std::vector<CheckpointLayerInfo> hidden_layers;
    std::vector<CheckpointHeadSpec> hidden_head_specs;
    std::vector<CheckpointOutputHeadSpec> output_head_specs;
    std::size_t output_head_count = 0;
    std::size_t class_count = 0;
    bool has_output_attention_head = false;
};

struct HiddenLayerForwardTrace {
    FloatMatrix input;
    std::vector<HeadForwardTrace> head_traces;
    FloatMatrix concat;
};

struct MultiHeadForwardTrace {
    FloatMatrix H;
    FloatMatrix bias;
    std::vector<HiddenLayerForwardTrace> hidden_layer_traces;
    std::vector<HeadForwardTrace> hidden_head_traces;
    FloatMatrix hidden_concat;
    std::vector<HeadForwardTrace> output_head_traces;
    std::vector<FloatMatrix> output_head_values;
    HeadForwardTrace output_head_trace;
    FloatMatrix Y_lin;
    FloatMatrix Y;
};

ModelParameters build_model_parameters(
    std::size_t input_dim,
    std::size_t hidden_dim,
    std::size_t num_classes,
    std::uint64_t seed);
ModelParameters build_family_model_parameters(
    const std::vector<std::size_t>& d_in_profile,
    const std::vector<HiddenLayerShape>& hidden_profile,
    std::size_t k_out,
    std::size_t num_classes,
    std::uint64_t seed);
CheckpointBundleInfo inspect_checkpoint_bundle(const std::string& bundle_root);
bool checkpoint_bundle_matches_formal_proof_shape(const CheckpointBundleInfo& info, std::string* reason = nullptr);
ModelParameters load_checkpoint_bundle_parameters(const std::string& bundle_root);
std::size_t attention_head_output_width(const AttentionHeadParameters& parameters);
FloatMatrix build_attention_bias_matrix(std::size_t num_nodes, const std::vector<data::Edge>& edges);
HeadForwardTrace attention_head_forward(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const AttentionHeadParameters& parameters,
    HeadForwardProfile* profile = nullptr);
MultiHeadForwardTrace forward_reference_style(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const ModelParameters& parameters,
    ForwardProfile* profile = nullptr);
MultiHeadForwardTrace forward_note_style(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const ModelParameters& parameters,
    ForwardProfile* profile = nullptr);

Matrix project_features(const Matrix& left, const Matrix& right);
std::vector<algebra::FieldElement> matvec_projection(const Matrix& matrix, const std::vector<algebra::FieldElement>& vector);
std::vector<algebra::FieldElement> compress_rows(const Matrix& matrix, const algebra::FieldElement& challenge);
Matrix aggregate_by_edges(
    const Matrix& h_prime,
    const std::vector<algebra::FieldElement>& alpha,
    const std::vector<data::Edge>& edges,
    std::size_t num_nodes);
Matrix output_projection(
    const Matrix& h_agg,
    const Matrix& w_out,
    const std::vector<algebra::FieldElement>& bias,
    Matrix* linear_part);
bool hidden_family_dimension_chain_is_valid(const ModelParameters& parameters, std::string* reason = nullptr);
std::size_t flattened_hidden_head_count(const ModelParameters& parameters);
std::size_t max_hidden_input_dim(const ModelParameters& parameters);
std::size_t max_hidden_head_dim(const ModelParameters& parameters);
std::size_t max_hidden_concat_width(const ModelParameters& parameters);
bool supports_current_formal_proof_shape(const ModelParameters& parameters);

}  // namespace gatzk::model
