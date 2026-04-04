#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol {

const std::vector<std::string>& proof_block_order() {
    static const std::vector<std::string> order = {
        "M_pub",
        "Com_dyn",
        "S_route",
        "Eval_ext",
        "Eval_dom",
        "Com_quot",
        "Open_dom",
        "W_ext",
        "Pi_bind",
    };
    return order;
}

std::string hidden_head_prefix(std::size_t global_head_index) {
    return "P_h" + std::to_string(global_head_index);
}

std::string hidden_layer_concat_label(std::size_t layer_index, bool is_final_layer) {
    if (is_final_layer) {
        return "P_H_cat";
    }
    return "P_H_cat_l" + std::to_string(layer_index);
}

std::string hidden_layer_concat_star_label(std::size_t layer_index, bool is_final_layer) {
    if (is_final_layer) {
        return "P_H_cat_star";
    }
    return "P_H_cat_star_l" + std::to_string(layer_index);
}

std::string hidden_layer_cat_prefix(std::size_t layer_index, bool is_final_layer) {
    if (is_final_layer) {
        return "P_cat";
    }
    return "P_cat_l" + std::to_string(layer_index);
}

std::string hidden_weight_label(std::size_t global_head_index) {
    return "V_h" + std::to_string(global_head_index) + "_W";
}

std::string hidden_src_label(std::size_t global_head_index) {
    return "V_h" + std::to_string(global_head_index) + "_a_src";
}

std::string hidden_dst_label(std::size_t global_head_index) {
    return "V_h" + std::to_string(global_head_index) + "_a_dst";
}

std::string output_head_prefix(std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "P_out";
    }
    return "P_out" + std::to_string(head_index);
}

std::string output_y_lin_label(std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "P_Y_lin";
    }
    return "P_Y_lin_o" + std::to_string(head_index);
}

std::string output_y_label(std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "P_Y";
    }
    return "P_Y_o" + std::to_string(head_index);
}

std::string output_weight_label(std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "V_out_W";
    }
    return "V_out" + std::to_string(head_index) + "_W";
}

std::string output_src_label(std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "V_out_a_src";
    }
    return "V_out" + std::to_string(head_index) + "_a_src";
}

std::string output_dst_label(std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "V_out_a_dst";
    }
    return "V_out" + std::to_string(head_index) + "_a_dst";
}

std::string output_challenge_name(std::string_view base, std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return std::string(base);
    }
    return std::string(base) + "_o" + std::to_string(head_index);
}

std::string output_external_eval_name(std::string_view base, std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "mu_" + std::string(base);
    }
    return "mu_out" + std::to_string(head_index) + "_" + std::string(base);
}

std::string output_witness_scalar_name(std::string_view base, std::size_t head_index, bool use_legacy_single_head_labels) {
    if (use_legacy_single_head_labels) {
        return "S_" + std::string(base) + "_out";
    }
    return "S_" + std::string(base) + "_out" + std::to_string(head_index);
}

std::string hidden_concat_xi_name(std::size_t layer_index, bool is_final_layer) {
    if (is_final_layer) {
        return "xi_cat";
    }
    return "xi_cat_l" + std::to_string(layer_index);
}

std::string hidden_concat_y_name(std::size_t layer_index, bool is_final_layer) {
    if (is_final_layer) {
        return "y_cat";
    }
    return "y_cat_l" + std::to_string(layer_index);
}

}  // namespace gatzk::protocol
