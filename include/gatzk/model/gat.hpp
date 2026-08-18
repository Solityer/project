#pragma once
// 本文件定义 GAT 模型在本仓库中的参数、前向 trace 与 checkpoint 束接口，
// 负责把上游 bundle 形状与 formal proving 主线需要的对象对齐。

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/data/dataset.hpp"

namespace gatzk::model
{
    // 量化后的矩阵与浮点矩阵分别服务 formal witness 与参考前向。
    using Matrix = std::vector<std::vector<algebra::FieldElement>>;
    using FloatMatrix = std::vector<std::vector<double>>;

    // 文本导出的稠密张量，用于从 checkpoint bundle 读取真实参数。
    struct DenseTensor
    {
        std::vector<std::size_t> shape;
        std::vector<double> values;
    };

    // 单个注意力头的原始浮点参数，后续会在 trace / context 阶段量化到域元素。
    struct AttentionHeadParameters
    {
        FloatMatrix seq_kernel_fp;
        std::vector<double> attn_dst_kernel_fp;
        double attn_dst_bias_fp = 0.0;
        std::vector<double> attn_src_kernel_fp;
        double attn_src_bias_fp = 0.0;
        std::vector<double> output_bias_fp;
    };

    // 单头前向过程中需要被 formal 约束的中间量集合。
    struct HeadForwardTrace
    {
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

    // 单头前向阶段的执行耗时拆分，最终会汇总到 RunMetrics。
    struct HeadForwardProfile
    {
        double projection_ms = 0.0;
        double attention_ms = 0.0;
        double activation_ms = 0.0;
    };

    // 完整模型前向的阶段级耗时拆分。
    struct ForwardProfile
    {
        double hidden_projection_ms = 0.0;
        double hidden_attention_ms = 0.0;
        double hidden_activation_ms = 0.0;
        double hidden_concat_ms = 0.0;
        double output_projection_ms = 0.0;
        double output_attention_ms = 0.0;
        double output_activation_ms = 0.0;
    };

    // 隐层家族形状，直接对应 formal note 中每层 `head_count × head_dim` 配置。
    struct HiddenLayerShape
    {
        std::size_t head_count = 0;
        std::size_t head_dim = 0;
    };

    // 单个隐藏层的参数集合，由 checkpoint manifest 或 synthetic tests 构造。
    struct HiddenLayerParameters
    {
        std::size_t layer_index = 0;
        std::size_t input_dim = 0;
        HiddenLayerShape shape;
        std::vector<AttentionHeadParameters> heads;
    };

    // 输出层多头参数集合。
    struct OutputLayerParameters
    {
        std::size_t input_dim = 0;
        std::size_t output_dim = 0;
        std::size_t head_count = 0;
        std::vector<AttentionHeadParameters> heads;
    };

    // formal 主线消费的统一模型对象，同时兼容正式 checkpoint 与测试用 synthetic 模式。
    struct ModelParameters
    {
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

    // checkpoint manifest 中记录的层级元数据。
    struct CheckpointLayerInfo
    {
        std::size_t layer_index = 0;
        std::size_t input_dim = 0;
        HiddenLayerShape shape;
    };

    // checkpoint manifest 中记录的隐藏头张量名。
    struct CheckpointHeadSpec
    {
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

    // checkpoint manifest 中记录的输出头张量名。
    struct CheckpointOutputHeadSpec
    {
        std::size_t head_index = 0;
        std::string seq_kernel;
        std::string attn_dst_kernel;
        std::string attn_dst_bias;
        std::string attn_src_kernel;
        std::string attn_src_bias;
        std::string output_bias;
    };

    // 上游 bundle 的结构化摘要，用来校验 formal proof shape 与 artifact contract。
    struct CheckpointBundleInfo
    {
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

    // 按层聚合的隐藏层前向结果。
    struct HiddenLayerForwardTrace
    {
        FloatMatrix input;
        std::vector<HeadForwardTrace> head_traces;
        FloatMatrix concat;
    };

    // 参考前向输出，trace 构建会从这里提取 witness、lookup 与 route 所需对象。
    struct MultiHeadForwardTrace
    {
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
    // 读取 bundle 的 manifest 元数据，不直接加载权重张量。
    CheckpointBundleInfo inspect_checkpoint_bundle(const std::string &bundle_root);
    // 检查 bundle 是否满足当前 formal 主线要求的层数、hidden_profile、K_out 与输出平均规则。
    bool checkpoint_bundle_matches_formal_proof_shape(const CheckpointBundleInfo &info, std::string *reason = nullptr);
    // 从真实 checkpoint bundle 载入模型参数，是正式 proving 主线的唯一模型来源。
    ModelParameters load_checkpoint_bundle_parameters(const std::string &bundle_root);
    // 推出一个注意力头的输出宽度，供 domain 规划与 trace 物化使用。
    std::size_t attention_head_output_width(const AttentionHeadParameters &parameters);
    // 当前 formal 主线对应的前向实现，trace 构建直接依赖它的输出对象。
    MultiHeadForwardTrace forward_note_style(
        const FloatMatrix &features,
        const std::vector<data::Edge> &edges,
        const ModelParameters &parameters,
        ForwardProfile *profile = nullptr);

    // 下面这些辅助函数将矩阵、向量与边聚合结果转换成 trace / lookup 可消费的对象。
    Matrix project_features(const Matrix &left, const Matrix &right);
    std::vector<algebra::FieldElement> matvec_projection(const Matrix &matrix, const std::vector<algebra::FieldElement> &vector);
    std::vector<algebra::FieldElement> compress_rows(const Matrix &matrix, const algebra::FieldElement &challenge);
    Matrix aggregate_by_edges(
        const Matrix &h_prime,
        const std::vector<algebra::FieldElement> &alpha,
        const std::vector<data::Edge> &edges,
        std::size_t num_nodes);
    Matrix output_projection(
        const Matrix &h_agg,
        const Matrix &w_out,
        const std::vector<algebra::FieldElement> &bias,
        Matrix *linear_part);
    // 验证多层多头形状链是否自洽，是 context 构建阶段的重要前置约束。
    bool hidden_family_dimension_chain_is_valid(const ModelParameters &parameters, std::string *reason = nullptr);
    // 下面这些查询函数为 domain 规划、metadata 导出与 tests 提供统一形状信息。
    std::size_t flattened_hidden_head_count(const ModelParameters &parameters);
    std::size_t max_hidden_input_dim(const ModelParameters &parameters);
    std::size_t max_hidden_head_dim(const ModelParameters &parameters);
    std::size_t max_hidden_concat_width(const ModelParameters &parameters);
    bool supports_current_formal_proof_shape(const ModelParameters &parameters);
}
