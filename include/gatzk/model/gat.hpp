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

struct ModelParameters {
    Matrix W;
    std::vector<algebra::FieldElement> a_src;
    std::vector<algebra::FieldElement> a_dst;
    Matrix W_out;
    std::vector<algebra::FieldElement> b;
    bool has_real_multihead = false;
    std::vector<AttentionHeadParameters> hidden_heads;
    AttentionHeadParameters output_head;
};

struct CheckpointBundleInfo {
    std::string bundle_root;
    std::size_t hidden_head_count = 0;
    bool has_output_attention_head = false;
};

struct MultiHeadForwardTrace {
    FloatMatrix H;
    FloatMatrix bias;
    std::vector<HeadForwardTrace> hidden_head_traces;
    FloatMatrix hidden_concat;
    HeadForwardTrace output_head_trace;
    FloatMatrix Y_lin;
    FloatMatrix Y;
};

ModelParameters build_model_parameters(
    std::size_t input_dim,
    std::size_t hidden_dim,
    std::size_t num_classes,
    std::uint64_t seed);
CheckpointBundleInfo inspect_checkpoint_bundle(const std::string& bundle_root);
bool checkpoint_bundle_matches_single_head_protocol(
    const CheckpointBundleInfo& info,
    std::string* reason = nullptr);
ModelParameters load_checkpoint_bundle_parameters(const std::string& bundle_root);
FloatMatrix build_attention_bias_matrix(std::size_t num_nodes, const std::vector<data::Edge>& edges);
HeadForwardTrace attention_head_forward(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const AttentionHeadParameters& parameters);
MultiHeadForwardTrace forward_reference_style(
    const FloatMatrix& features,
    const std::vector<data::Edge>& edges,
    const ModelParameters& parameters);

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

}  // namespace gatzk::model
