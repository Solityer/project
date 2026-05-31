#pragma once
// 本文件维护正式 proof schema 使用的标签命名规则与 block 顺序，
// 是 trace、prove、verify 与 artifact 共享的命名真源。

#include <string>
#include <vector>

namespace gatzk::protocol {

struct ProtocolContext;

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
    PolynomialVisibility visibility = PolynomialVisibility::PrivateWitness;
    std::size_t layer_id = 0;
    std::size_t head_id = 0;
    std::string opening_policy;
    std::string binding_mode;
    bool enters_commitment = false;
    bool enters_opening = false;
    bool enters_quotient = false;
};

struct ProofSchema {
    std::vector<ProofSchemaEntry> entries;
    std::size_t private_plain_value_count = 0;
};

// 最终 proof 的 block 顺序，任何改变都会影响 serializer / parser 与 verifier。
// 它也决定哪些标签会进入 proof 的正式传输形状，因此间接影响 proof_size_bytes。
const std::vector<std::string>& proof_block_order();
ProofSchema build_proof_schema(const ProtocolContext& context);
bool schema_accepts_evaluation(PolynomialVisibility visibility, EvaluationKind kind);
// 下面这些函数负责生成隐藏层、输出层与兼容单头路径使用的正式标签。
// 它们共同决定 dynamic commitments、opening bundle 与 external evaluation 的命名真源。
std::string hidden_head_prefix(std::size_t global_head_index);
std::string hidden_layer_concat_label(std::size_t layer_index, bool is_final_layer);
std::string hidden_layer_concat_star_label(std::size_t layer_index, bool is_final_layer);
std::string hidden_layer_cat_prefix(std::size_t layer_index, bool is_final_layer);
std::string hidden_weight_label(std::size_t global_head_index);
std::string hidden_src_label(std::size_t global_head_index);
std::string hidden_dst_label(std::size_t global_head_index);
std::string output_head_prefix(std::size_t head_index, bool use_single_output_compat_labels);
std::string output_y_lin_label(std::size_t head_index, bool use_single_output_compat_labels);
std::string output_y_label(std::size_t head_index, bool use_single_output_compat_labels);
std::string output_weight_label(std::size_t head_index, bool use_single_output_compat_labels);
std::string output_src_label(std::size_t head_index, bool use_single_output_compat_labels);
std::string output_dst_label(std::size_t head_index, bool use_single_output_compat_labels);
std::string output_challenge_name(std::string_view base, std::size_t head_index, bool use_single_output_compat_labels);
std::string output_external_eval_name(std::string_view base, std::size_t head_index, bool use_single_output_compat_labels);
std::string output_witness_scalar_name(std::string_view base, std::size_t head_index, bool use_single_output_compat_labels);
std::string hidden_concat_xi_name(std::size_t layer_index, bool is_final_layer);
std::string hidden_concat_y_name(std::size_t layer_index, bool is_final_layer);

}  // namespace gatzk::protocol
