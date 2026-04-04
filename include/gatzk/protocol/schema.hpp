#pragma once

#include <string>
#include <vector>

namespace gatzk::protocol {

const std::vector<std::string>& proof_block_order();
std::string hidden_head_prefix(std::size_t global_head_index);
std::string hidden_layer_concat_label(std::size_t layer_index, bool is_final_layer);
std::string hidden_layer_concat_star_label(std::size_t layer_index, bool is_final_layer);
std::string hidden_layer_cat_prefix(std::size_t layer_index, bool is_final_layer);
std::string hidden_weight_label(std::size_t global_head_index);
std::string hidden_src_label(std::size_t global_head_index);
std::string hidden_dst_label(std::size_t global_head_index);
std::string output_head_prefix(std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_y_lin_label(std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_y_label(std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_weight_label(std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_src_label(std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_dst_label(std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_challenge_name(std::string_view base, std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_external_eval_name(std::string_view base, std::size_t head_index, bool use_legacy_single_head_labels);
std::string output_witness_scalar_name(std::string_view base, std::size_t head_index, bool use_legacy_single_head_labels);
std::string hidden_concat_xi_name(std::size_t layer_index, bool is_final_layer);
std::string hidden_concat_y_name(std::size_t layer_index, bool is_final_layer);

}  // namespace gatzk::protocol
