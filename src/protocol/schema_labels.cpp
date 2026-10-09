#include "gatzk/protocol/schema.hpp"

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/proof.hpp"
#include "gatzk/protocol/quotients.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>
#include <memory>
#include <openssl/bn.h>

namespace gatzk::protocol
{
    std::string hidden_head_prefix(std::size_t layer_index, std::size_t local_head_index)
    {
        return "P_h_l" + std::to_string(layer_index) + "_h" + std::to_string(local_head_index);
    }

    std::string hidden_layer_concat_label(std::size_t layer_index) { return "P_H_cat_l" + std::to_string(layer_index); }

    std::string hidden_layer_concat_star_label(std::size_t layer_index) { return "P_H_cat_star_l" + std::to_string(layer_index); }

    std::string hidden_layer_cat_prefix(std::size_t layer_index) { return "P_cat_l" + std::to_string(layer_index); }

    std::string hidden_weight_label(std::size_t layer_index, std::size_t local_head_index)
    {
        return "V_h_l" + std::to_string(layer_index) + "_h" + std::to_string(local_head_index) + "_W";
    }

    std::string hidden_weight_column_label(std::size_t layer_index, std::size_t local_head_index, std::size_t column_index)
    {
        return hidden_weight_label(layer_index, local_head_index) + "_col" + std::to_string(column_index);
    }

    std::string hidden_src_label(std::size_t layer_index, std::size_t local_head_index)
    {
        return "V_h_l" + std::to_string(layer_index) + "_h" + std::to_string(local_head_index) + "_a_src";
    }

    std::string hidden_dst_label(std::size_t layer_index, std::size_t local_head_index)
    {
        return "V_h_l" + std::to_string(layer_index) + "_h" + std::to_string(local_head_index) + "_a_dst";
    }

    std::string hidden_challenge_name(std::string_view base, std::size_t layer_index, std::size_t local_head_index)
    {
        return std::string(base) + "_l" + std::to_string(layer_index) + "_h" + std::to_string(local_head_index);
    }

    std::string hidden_q_in_selector_label(std::size_t layer_index) { return "P_Q_in_valid_l" + std::to_string(layer_index); }

    std::string hidden_q_d_h_selector_label(std::size_t layer_index) { return "P_Q_d_h_valid_l" + std::to_string(layer_index); }

    std::string hidden_q_cat_selector_label(std::size_t layer_index) { return "P_Q_cat_valid_l" + std::to_string(layer_index); }

    std::string hidden_q_nd_h_selector_label(std::size_t layer_index) { return "P_Q_Nd_h_valid_l" + std::to_string(layer_index); }

    std::string output_q_cat_selector_label() { return "P_Q_cat_valid_out"; }

    std::string output_head_prefix(std::size_t head_index) { return "P_out" + std::to_string(head_index); }

    std::string output_y_lin_label(std::size_t head_index) { return "P_Y_lin_o" + std::to_string(head_index); }

    std::string output_y_label(std::size_t head_index) { return "P_Y_o" + std::to_string(head_index); }

    std::string canonical_matrix_column_label(const std::string &matrix_object, std::size_t column_index)
    {
        return matrix_object + "_CanCol" + std::to_string(column_index);
    }

    std::string hidden_aggregation_binding_label(std::size_t layer_index, std::size_t local_head_index, std::string_view suffix)
    {
        return hidden_head_prefix(layer_index, local_head_index) + "_AggBind_" + std::string(suffix);
    }

    std::string output_aggregation_binding_label(std::size_t head_index, std::string_view suffix)
    {
        return output_head_prefix(head_index) + "_AggBind_" + std::string(suffix);
    }

    std::string output_weight_label(std::size_t head_index) { return "V_out" + std::to_string(head_index) + "_W"; }

    std::string output_weight_column_label(std::size_t head_index, std::size_t column_index)
    {
        return output_weight_label(head_index) + "_col" + std::to_string(column_index);
    }

    std::string output_src_label(std::size_t head_index) { return "V_out" + std::to_string(head_index) + "_a_src"; }

    std::string output_dst_label(std::size_t head_index) { return "V_out" + std::to_string(head_index) + "_a_dst"; }

    std::string output_bias_label(std::size_t head_index) { return "V_o" + std::to_string(head_index) + "_b"; }

    std::string output_challenge_name(std::string_view base, std::size_t head_index) { return std::string(base) + "_o" + std::to_string(head_index); }

    std::string hidden_model_binding_blinder_label(std::size_t layer_index, std::size_t local_head_index, std::string_view role)
    {
        return hidden_head_prefix(layer_index, local_head_index) + "_ModelBind_" + std::string(role) + "_R";
    }

    std::string output_model_binding_blinder_label(std::size_t head_index, std::string_view role)
    {
        return output_head_prefix(head_index) + "_ModelBind_" + std::string(role) + "_R";
    }

    std::string hidden_score_mask_label(std::size_t layer_index, std::size_t local_head_index, std::string_view role)
    {
        return hidden_head_prefix(layer_index, local_head_index) + "_ScoreMask_" + std::string(role);
    }

    std::string output_score_mask_label(std::size_t head_index, std::string_view role)
    {
        return output_head_prefix(head_index) + "_ScoreMask_" + std::string(role);
    }

    std::string hidden_score_mask_eval_name(std::size_t layer_index, std::size_t local_head_index, std::string_view role)
    {
        return "rho_h_l" + std::to_string(layer_index) + "_h" + std::to_string(local_head_index) + "_" + std::string(role);
    }

    std::string output_score_mask_eval_name(std::size_t head_index, std::string_view role)
    {
        return "rho_out" + std::to_string(head_index) + "_" + std::string(role);
    }

    std::string external_mask_label(std::string_view source_label) { return std::string(source_label) + "_ExternalMask"; }

    std::string external_mask_eval_name(std::string_view source_eval_name) { return std::string(source_eval_name) + "_mask"; }

    std::string model_proof_local_label(std::string_view vk_model_anchor_label) { return std::string(vk_model_anchor_label) + "_proof"; }

    std::string model_reblind_witness_label(std::string_view vk_model_anchor_label) { return std::string(vk_model_anchor_label) + "_S_model"; }

    std::string output_bias_view_prefix(std::size_t head_index) { return output_head_prefix(head_index) + "_BiasView_"; }

    std::string output_bias_view_label(std::size_t head_index, std::string_view suffix) { return output_bias_view_prefix(head_index) + std::string(suffix); }

    std::string output_bias_view_eta_label(std::size_t head_index) { return "eta_view:output-bias:" + std::to_string(head_index); }

    std::string output_bias_view_beta_label(std::size_t head_index) { return "beta_view:output-bias:" + std::to_string(head_index); }

    std::string output_bias_view_relation_challenge_label(std::size_t head_index) { return "y_rel:output-bias-terminal:" + std::to_string(head_index); }

    std::string output_bias_basis_prefix(std::size_t head_index) { return output_head_prefix(head_index) + "_BiasBasis_"; }

    std::string output_bias_basis_label(std::size_t head_index, std::string_view suffix) { return output_bias_basis_prefix(head_index) + std::string(suffix); }

    std::string output_bias_basis_chunk_label(std::size_t head_index, std::size_t chunk_index)
    {
        return output_bias_basis_prefix(head_index) + "Chunk_" + std::to_string(chunk_index);
    }

    std::string output_bias_basis_chunk_accumulator_label(std::size_t head_index, std::size_t chunk_index)
    {
        return output_bias_basis_chunk_label(head_index, chunk_index) + "_Acc";
    }

    std::string output_bias_basis_chunk_shifted_accumulator_label(std::size_t head_index, std::size_t chunk_index)
    {
        return output_bias_basis_chunk_label(head_index, chunk_index) + "_Acc_shifted";
    }

    std::string output_bias_basis_region_selector_label(std::size_t head_index, std::size_t chunk_index, std::string_view region)
    {
        return output_bias_basis_prefix(head_index) + "Q_" + std::string(region) + "_" + std::to_string(chunk_index);
    }

    std::string output_bias_basis_gamma_label(std::size_t head_index) { return "gamma_basis:output-bias:" + std::to_string(head_index); }

    std::string output_bias_basis_relation_challenge_label(std::size_t head_index)
    {
        return "y_rel:coefficient-basis:output-bias:" + std::to_string(head_index);
    }

    std::size_t signed_division_limb_count(std::size_t range_bits)
    {
        if (range_bits == 0 || range_bits >= 64)
        {
            throw std::runtime_error("SignedRoundDiv range_bits must be in [1,63]");
        }
        return (64U + range_bits - 1U) / range_bits;
    }

    std::string signed_division_limb_label(std::string_view division_prefix, std::string_view value_name, std::size_t limb_index)
    {
        return std::string(division_prefix) + "Limb_" + std::string(value_name) + "_" + std::to_string(limb_index);
    }

    std::string signed_division_range_label(std::string_view division_prefix, std::string_view value_name, std::size_t limb_index, std::string_view column_name)
    {
        return std::string(division_prefix) + "Range_" + std::string(value_name) + "_" + std::to_string(limb_index) + "_" + std::string(column_name);
    }

    std::string signed_division_beta_name(std::string_view division_prefix, std::string_view value_name, std::size_t limb_index)
    {
        std::string prefix(division_prefix);
        if (!prefix.empty() && prefix.back() == '_')
            prefix.pop_back();
        return "beta_div_range:" + prefix + ":" + std::string(value_name) + ":" + std::to_string(limb_index);
    }

    std::string output_external_eval_name(std::string_view base, std::size_t head_index)
    {
        return "mu_out" + std::to_string(head_index) + "_" + std::string(base);
    }

    std::string hidden_concat_xi_name(std::size_t layer_index) { return "xi_cat_l" + std::to_string(layer_index); }

    std::string hidden_concat_y_name(std::size_t layer_index) { return "y_cat_l" + std::to_string(layer_index); }

}
