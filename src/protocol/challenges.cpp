#include "gatzk/protocol/challenges.hpp"

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "gatzk/crypto/transcript.hpp"

namespace gatzk::protocol {
namespace {

std::uint64_t fnv1a(std::string_view data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : data) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void absorb_text_scalar(crypto::Transcript& transcript, const std::string& label, const std::string& value) {
    transcript.absorb_scalar(label, algebra::FieldElement(fnv1a(label + "=" + value)));
}

void absorb(
    crypto::Transcript& transcript,
    const std::string& label,
    const std::unordered_map<std::string, crypto::Commitment>& commitments) {
    transcript.absorb_commitment(label, commitments.at(label).point);
}

void absorb_static(crypto::Transcript& transcript, const ProtocolContext& context, const std::string& label) {
    transcript.absorb_commitment(label, context.static_commitments.at(label).point);
}

void absorb_public(crypto::Transcript& transcript, const ProtocolContext& context, const std::string& label) {
    transcript.absorb_commitment(label, context.public_commitments.at(label).point);
}

void absorb_static_if_present(crypto::Transcript& transcript, const ProtocolContext& context, const std::string& label) {
    if (const auto it = context.static_commitments.find(label); it != context.static_commitments.end()) {
        transcript.absorb_commitment(label, it->second.point);
    }
}

std::string head_prefix(std::size_t head_index) {
    return "P_h" + std::to_string(head_index);
}

std::string hidden_weight_label(std::size_t head_index) {
    return "V_h" + std::to_string(head_index) + "_W";
}

std::string hidden_src_label(std::size_t head_index) {
    return "V_h" + std::to_string(head_index) + "_a_src";
}

std::string hidden_dst_label(std::size_t head_index) {
    return "V_h" + std::to_string(head_index) + "_a_dst";
}

std::string output_weight_label() {
    return "V_out_W";
}

std::string output_src_label() {
    return "V_out_a_src";
}

std::string output_dst_label() {
    return "V_out_a_dst";
}

std::size_t hidden_head_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_heads.empty()) {
        return model::attention_head_output_width(parameters.hidden_heads.front());
    }
    return config.hidden_dim;
}

std::size_t concat_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_heads.empty()) {
        return hidden_head_width(parameters, config) * parameters.hidden_heads.size();
    }
    return config.hidden_dim;
}

void append_attention_head_dynamic_labels(std::vector<std::string>& labels, const std::string& prefix) {
    auto push = [&](const std::string& suffix) { labels.push_back(prefix + "_" + suffix); };
    push("H_prime");
    push("a_proj");
    push("b_proj");
    push("Acc_proj");
    push("E_src");
    push("E_dst");
    push("a_src");
    push("b_src");
    push("Acc_src");
    push("a_dst");
    push("b_dst");
    push("Acc_dst");
    push("H_star");
    push("a_star");
    push("b_star");
    push("Acc_star");
    push("E_src_edge");
    push("E_dst_edge");
    push("H_src_star_edge");
    push("Table_src");
    push("Query_src");
    push("m_src");
    push("R_src_node");
    push("R_src");
    push("S");
    push("Z");
    push("M");
    push("M_edge");
    push("Delta");
    push("U");
    push("Sum");
    push("Sum_edge");
    push("inv");
    push("inv_edge");
    push("alpha");
    push("H_agg_pre");
    push("H_agg_pre_star");
    push("H_agg_pre_star_edge");
    push("widehat_v_pre_star");
    push("w_psq");
    push("T_psq");
    push("T_psq_edge");
    push("PSQ");
    push("H_agg");
    push("H_agg_star");
    push("H_agg_star_edge");
    push("a_agg");
    push("b_agg");
    push("Acc_agg");
    push("Table_dst");
    push("Query_dst");
    push("m_dst");
    push("R_dst_node");
    push("R_dst");
}

}  // namespace

PublicMetadata canonical_public_metadata(const ProtocolContext& context) {
    PublicMetadata metadata;
    metadata.protocol_id = "gatzkml";
    metadata.model_arch_id = context.model.has_real_multihead
        ? "single_layer_gat_hidden8_output1"
        : "legacy_single_head_debug";
    metadata.model_param_id = context.model.has_real_multihead
        ? ("checkpoint_bundle:" + context.config.checkpoint_bundle)
        : ("synthetic_seed:" + std::to_string(context.config.seed));
    metadata.static_table_id = "tables:lrelu+exp+range";
    metadata.quant_cfg_id = "range_bits=" + std::to_string(context.config.range_bits);
    metadata.domain_cfg =
        "FH=" + context.domains.fh->name + ":" + std::to_string(context.domains.fh->size)
        + ",edge=" + context.domains.edge->name + ":" + std::to_string(context.domains.edge->size)
        + ",in=" + context.domains.in->name + ":" + std::to_string(context.domains.in->size)
        + ",d_h=" + context.domains.d->name + ":" + std::to_string(context.domains.d->size)
        + ",cat=" + context.domains.cat->name + ":" + std::to_string(context.domains.cat->size)
        + ",C=" + context.domains.c->name + ":" + std::to_string(context.domains.c->size)
        + ",N=" + context.domains.n->name + ":" + std::to_string(context.domains.n->size);
    metadata.dim_cfg =
        "N=" + std::to_string(context.local.num_nodes)
        + ",E=" + std::to_string(context.local.edges.size())
        + ",d_in=" + std::to_string(context.local.num_features)
        + ",d_h=" + std::to_string(hidden_head_width(context.model, context.config))
        + ",d_cat=" + std::to_string(concat_width(context.model, context.config))
        + ",C=" + std::to_string(context.local.num_classes);
    metadata.encoding_id = "project-fixed-order-v1";
    metadata.padding_rule_id = "zero-pad+selector-mask";
    metadata.degree_bound_id = context.model.has_real_multihead
        ? "note-target:FH,edge,in,d_h,cat,C,N"
        : "legacy:FH,edge,in,d,N";
    return metadata;
}

void absorb_public_metadata(crypto::Transcript& transcript, const PublicMetadata& metadata) {
    absorb_text_scalar(transcript, "M_pub.protocol_id", metadata.protocol_id);
    absorb_text_scalar(transcript, "M_pub.model_arch_id", metadata.model_arch_id);
    absorb_text_scalar(transcript, "M_pub.model_param_id", metadata.model_param_id);
    absorb_text_scalar(transcript, "M_pub.static_table_id", metadata.static_table_id);
    absorb_text_scalar(transcript, "M_pub.quant_cfg_id", metadata.quant_cfg_id);
    absorb_text_scalar(transcript, "M_pub.domain_cfg", metadata.domain_cfg);
    absorb_text_scalar(transcript, "M_pub.dim_cfg", metadata.dim_cfg);
    absorb_text_scalar(transcript, "M_pub.encoding_id", metadata.encoding_id);
    absorb_text_scalar(transcript, "M_pub.padding_rule_id", metadata.padding_rule_id);
    absorb_text_scalar(transcript, "M_pub.degree_bound_id", metadata.degree_bound_id);
}

std::vector<std::string> dynamic_commitment_labels(const ProtocolContext& context) {
    if (!context.model.has_real_multihead) {
        return {
            "P_H",
            "P_Table_feat",
            "P_Query_feat",
            "P_m_feat",
            "P_R_feat",
            "P_H_prime",
            "P_a_proj",
            "P_b_proj",
            "P_Acc_proj",
            "P_E_src",
            "P_E_dst",
            "P_H_star",
            "P_a_src",
            "P_b_src",
            "P_Acc_src",
            "P_a_dst",
            "P_b_dst",
            "P_Acc_dst",
            "P_a_star",
            "P_b_star",
            "P_Acc_star",
            "P_E_src_edge",
            "P_H_src_star_edge",
            "P_Table_src",
            "P_Query_src",
            "P_m_src",
            "P_R_src_node",
            "P_R_src",
            "P_S",
            "P_Z",
            "P_Table_L",
            "P_Query_L",
            "P_m_L",
            "P_R_L",
            "P_M",
            "P_M_edge",
            "P_Delta",
            "P_s_max",
            "P_C_max",
            "P_Table_R",
            "P_Query_R",
            "P_m_R",
            "P_R_R",
            "P_U",
            "P_Sum",
            "P_inv",
            "P_alpha",
            "P_Table_exp",
            "P_Query_exp",
            "P_m_exp",
            "P_R_exp",
            "P_H_agg",
            "P_H_agg_star",
            "P_a_agg",
            "P_b_agg",
            "P_Acc_agg",
            "P_E_dst_edge",
            "P_Sum_edge",
            "P_inv_edge",
            "P_H_agg_star_edge",
            "P_Table_dst",
            "P_Query_dst",
            "P_m_dst",
            "P_R_dst_node",
            "P_R_dst",
            "P_v_hat",
            "P_w_psq",
            "P_T_psq_edge",
            "P_PSQ",
            "P_Y_lin",
            "P_Y",
            "P_a_out",
            "P_b_out",
            "P_Acc_out",
        };
    }

    std::vector<std::string> labels = {
        "P_H",
        "P_Table_feat",
        "P_Query_feat",
        "P_m_feat",
        "P_R_feat",
    };
    for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
        append_attention_head_dynamic_labels(labels, head_prefix(head_index));
    }
    labels.push_back("P_H_cat");
    labels.push_back("P_H_cat_star");
    labels.push_back("P_cat_a");
    labels.push_back("P_cat_b");
    labels.push_back("P_cat_Acc");
    labels.push_back("P_out_Y_prime");
    labels.push_back("P_out_a_proj");
    labels.push_back("P_out_b_proj");
    labels.push_back("P_out_Acc_proj");
    labels.push_back("P_out_E_src");
    labels.push_back("P_out_E_dst");
    labels.push_back("P_out_a_src");
    labels.push_back("P_out_b_src");
    labels.push_back("P_out_Acc_src");
    labels.push_back("P_out_a_dst");
    labels.push_back("P_out_b_dst");
    labels.push_back("P_out_Acc_dst");
    labels.push_back("P_out_E_src_edge");
    labels.push_back("P_out_E_dst_edge");
    labels.push_back("P_out_Y_prime_star");
    labels.push_back("P_out_Y_prime_star_edge");
    labels.push_back("P_out_Table_src");
    labels.push_back("P_out_Query_src");
    labels.push_back("P_out_m_src");
    labels.push_back("P_out_R_src_node");
    labels.push_back("P_out_R_src");
    labels.push_back("P_out_S");
    labels.push_back("P_out_Z");
    labels.push_back("P_out_M");
    labels.push_back("P_out_M_edge");
    labels.push_back("P_out_Delta");
    labels.push_back("P_out_U");
    labels.push_back("P_out_Sum");
    labels.push_back("P_out_Sum_edge");
    labels.push_back("P_out_inv");
    labels.push_back("P_out_inv_edge");
    labels.push_back("P_out_alpha");
    labels.push_back("P_out_widehat_y_star");
    labels.push_back("P_out_w");
    labels.push_back("P_out_T");
    labels.push_back("P_out_T_edge");
    labels.push_back("P_out_PSQ");
    labels.push_back("P_Y_lin");
    labels.push_back("P_Y");
    labels.push_back("P_out_Y_star");
    labels.push_back("P_out_Y_star_edge");
    labels.push_back("P_out_a_y");
    labels.push_back("P_out_b_y");
    labels.push_back("P_out_Acc_y");
    labels.push_back("P_out_Table_dst");
    labels.push_back("P_out_Query_dst");
    labels.push_back("P_out_m_dst");
    labels.push_back("P_out_R_dst_node");
    labels.push_back("P_out_R_dst");
    return labels;
}

std::vector<std::string> quotient_commitment_labels(const ProtocolContext& context) {
    if (context.model.has_real_multihead) {
        return {"t_FH", "t_edge", "t_in", "t_d_h", "t_cat", "t_C", "t_N"};
    }
    return {"t_FH", "t_edge", "t_in", "t_d", "t_N"};
}

std::map<std::string, algebra::FieldElement> replay_challenges(
    const ProtocolContext& context,
    const std::unordered_map<std::string, crypto::Commitment>& dynamic_commitments,
    const std::unordered_map<std::string, crypto::Commitment>& quotient_commitments) {
    if (context.model.has_real_multihead) {
        crypto::Transcript transcript("gatzkml");
        std::map<std::string, algebra::FieldElement> out;

        const std::size_t d_h = model::attention_head_output_width(context.model.hidden_heads.front());
        const std::size_t d_cat = d_h * context.model.hidden_heads.size();

        absorb_public_metadata(transcript, canonical_public_metadata(context));
        transcript.absorb_scalar("N", algebra::FieldElement(context.local.num_nodes));
        transcript.absorb_scalar("E", algebra::FieldElement(context.local.edges.size()));
        transcript.absorb_scalar("d_in", algebra::FieldElement(context.local.num_features));
        transcript.absorb_scalar("d_h", algebra::FieldElement(d_h));
        transcript.absorb_scalar("d_cat", algebra::FieldElement(d_cat));
        transcript.absorb_scalar("C", algebra::FieldElement(context.local.num_classes));
        transcript.absorb_scalar("B", algebra::FieldElement(context.config.range_bits));
        absorb_public(transcript, context, "P_I");
        absorb_public(transcript, context, "P_src");
        absorb_public(transcript, context, "P_dst");
        absorb_public(transcript, context, "P_Q_new_edge");
        absorb_public(transcript, context, "P_Q_end_edge");
        absorb_public(transcript, context, "P_Q_edge_valid");
        absorb_public(transcript, context, "P_Q_N");
        absorb_public(transcript, context, "P_Q_proj_valid");
        absorb_public(transcript, context, "P_Q_d_valid");
        absorb_public(transcript, context, "P_Q_cat_valid");
        absorb_public(transcript, context, "P_Q_C_valid");
        absorb(transcript, "P_H", dynamic_commitments);
        absorb_static(transcript, context, "V_T_H");
        out["eta_feat"] = transcript.challenge("eta_feat");
        out["beta_feat"] = transcript.challenge("beta_feat");

        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            const auto prefix = head_prefix(head_index);
            absorb(transcript, "P_H", dynamic_commitments);
            absorb(transcript, prefix + "_H_prime", dynamic_commitments);
            absorb_static(transcript, context, hidden_weight_label(head_index));
            out["y_proj_h" + std::to_string(head_index)] = transcript.challenge("y_proj_h" + std::to_string(head_index));
            out["xi_h" + std::to_string(head_index)] = transcript.challenge("xi_h" + std::to_string(head_index));

            absorb(transcript, prefix + "_H_prime", dynamic_commitments);
            absorb(transcript, prefix + "_E_src", dynamic_commitments);
            absorb_static(transcript, context, hidden_src_label(head_index));
            out["y_src_h" + std::to_string(head_index)] = transcript.challenge("y_src_h" + std::to_string(head_index));

            absorb(transcript, prefix + "_H_prime", dynamic_commitments);
            absorb(transcript, prefix + "_E_dst", dynamic_commitments);
            absorb_static(transcript, context, hidden_dst_label(head_index));
            out["y_dst_h" + std::to_string(head_index)] = transcript.challenge("y_dst_h" + std::to_string(head_index));

            absorb(transcript, prefix + "_H_star", dynamic_commitments);
            out["y_star_h" + std::to_string(head_index)] = transcript.challenge("y_star_h" + std::to_string(head_index));

            out["eta_src_h" + std::to_string(head_index)] = transcript.challenge("eta_src_h" + std::to_string(head_index));
            out["beta_src_h" + std::to_string(head_index)] = transcript.challenge("beta_src_h" + std::to_string(head_index));
            out["lambda_psq_h" + std::to_string(head_index)] = transcript.challenge("lambda_psq_h" + std::to_string(head_index));

            absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
            absorb(transcript, prefix + "_H_agg_pre_star", dynamic_commitments);
            out["y_agg_pre_h" + std::to_string(head_index)] = transcript.challenge("y_agg_pre_h" + std::to_string(head_index));

            absorb(transcript, prefix + "_H_agg", dynamic_commitments);
            absorb(transcript, prefix + "_H_agg_star", dynamic_commitments);
            out["y_agg_h" + std::to_string(head_index)] = transcript.challenge("y_agg_h" + std::to_string(head_index));

            out["eta_dst_h" + std::to_string(head_index)] = transcript.challenge("eta_dst_h" + std::to_string(head_index));
            out["beta_dst_h" + std::to_string(head_index)] = transcript.challenge("beta_dst_h" + std::to_string(head_index));
        }

        absorb(transcript, "P_H_cat", dynamic_commitments);
        out["xi_cat"] = transcript.challenge("xi_cat");
        absorb(transcript, "P_H_cat_star", dynamic_commitments);
        out["y_cat"] = transcript.challenge("y_cat");

        absorb(transcript, "P_H_cat", dynamic_commitments);
        absorb(transcript, "P_out_Y_prime", dynamic_commitments);
        absorb_static(transcript, context, output_weight_label());
        out["y_proj_out"] = transcript.challenge("y_proj_out");
        out["xi_out"] = transcript.challenge("xi_out");
        absorb(transcript, "P_out_Y_prime", dynamic_commitments);
        absorb(transcript, "P_out_E_src", dynamic_commitments);
        absorb_static(transcript, context, output_src_label());
        out["y_src_out"] = transcript.challenge("y_src_out");
        absorb(transcript, "P_out_Y_prime", dynamic_commitments);
        absorb(transcript, "P_out_E_dst", dynamic_commitments);
        absorb_static(transcript, context, output_dst_label());
        out["y_dst_out"] = transcript.challenge("y_dst_out");
        out["eta_src_out"] = transcript.challenge("eta_src_out");
        out["beta_src_out"] = transcript.challenge("beta_src_out");
        out["lambda_out"] = transcript.challenge("lambda_out");
        absorb(transcript, "P_Y", dynamic_commitments);
        absorb(transcript, "P_out_Y_star", dynamic_commitments);
        out["y_out_star"] = transcript.challenge("y_out_star");
        out["eta_dst_out"] = transcript.challenge("eta_dst_out");
        out["beta_dst_out"] = transcript.challenge("beta_dst_out");
        absorb(transcript, "P_out_Y_prime", dynamic_commitments);
        absorb(transcript, "P_Y", dynamic_commitments);
        absorb(transcript, "P_out_Y_star", dynamic_commitments);
        absorb(transcript, "P_out_Table_dst", dynamic_commitments);
        absorb(transcript, "P_out_Query_dst", dynamic_commitments);
        out["y_out"] = transcript.challenge("y_out");

        for (const auto& label : dynamic_commitment_labels(context)) {
            absorb(transcript, label, dynamic_commitments);
        }
        for (const auto& label : {
                 std::string("V_T_H"),
                 std::string("V_T_L_x"),
                 std::string("V_T_L_y"),
                 std::string("V_T_exp_x"),
                 std::string("V_T_exp_y"),
                 std::string("V_T_range"),
             }) {
            absorb_static_if_present(transcript, context, label);
        }
        out["alpha_quot"] = transcript.challenge("alpha_quot");

        if (quotient_commitments.empty()) {
            return out;
        }

        for (const auto& label : quotient_commitment_labels(context)) {
            absorb(transcript, label, quotient_commitments);
        }
        out["z_FH"] = transcript.challenge("z_FH");
        out["z_edge"] = transcript.challenge("z_edge");
        out["z_in"] = transcript.challenge("z_in");
        out["z_d_h"] = transcript.challenge("z_d_h");
        out["z_cat"] = transcript.challenge("z_cat");
        out["z_C"] = transcript.challenge("z_C");
        out["z_N"] = transcript.challenge("z_N");
        out["v_FH"] = transcript.challenge("v_FH");
        out["v_edge"] = transcript.challenge("v_edge");
        out["v_in"] = transcript.challenge("v_in");
        out["v_d_h"] = transcript.challenge("v_d_h");
        out["v_cat"] = transcript.challenge("v_cat");
        out["v_C"] = transcript.challenge("v_C");
        out["v_N"] = transcript.challenge("v_N");
        out["rho_ext"] = transcript.challenge("rho_ext");
        return out;
    }

    crypto::Transcript transcript("gatzkml");
    std::map<std::string, algebra::FieldElement> out;

    transcript.absorb_scalar("N", algebra::FieldElement(context.local.num_nodes));
    transcript.absorb_scalar("E", algebra::FieldElement(context.local.edges.size()));
    transcript.absorb_scalar("d_in", algebra::FieldElement(context.local.num_features));
    transcript.absorb_scalar("d", algebra::FieldElement(context.config.hidden_dim));
    transcript.absorb_scalar("C", algebra::FieldElement(context.local.num_classes));
    transcript.absorb_scalar("B", algebra::FieldElement(context.config.range_bits));
    absorb_public(transcript, context, "P_I");
    absorb_public(transcript, context, "P_src");
    absorb_public(transcript, context, "P_dst");
    absorb_public(transcript, context, "P_Q_new_edge");
    absorb_public(transcript, context, "P_Q_end_edge");
    absorb_public(transcript, context, "P_Q_edge_valid");
    absorb_public(transcript, context, "P_Q_N");
    absorb_public(transcript, context, "P_Q_proj_valid");
    absorb_public(transcript, context, "P_Q_d_valid");
    absorb(transcript, "P_H", dynamic_commitments);
    absorb_static(transcript, context, "V_T_H");
    out["eta_feat"] = transcript.challenge("eta_feat");
    out["beta_feat"] = transcript.challenge("beta_feat");

    absorb(transcript, "P_H", dynamic_commitments);
    absorb(transcript, "P_H_prime", dynamic_commitments);
    absorb_static(transcript, context, "V_W");
    out["y_proj"] = transcript.challenge("y_proj");

    absorb(transcript, "P_H_prime", dynamic_commitments);
    out["xi"] = transcript.challenge("xi");

    absorb(transcript, "P_H_prime", dynamic_commitments);
    absorb(transcript, "P_E_src", dynamic_commitments);
    absorb_static(transcript, context, "V_a_src");
    out["y_src"] = transcript.challenge("y_src");

    absorb(transcript, "P_H_prime", dynamic_commitments);
    absorb(transcript, "P_E_dst", dynamic_commitments);
    absorb_static(transcript, context, "V_a_dst");
    out["y_dst"] = transcript.challenge("y_dst");

    absorb(transcript, "P_H_prime", dynamic_commitments);
    absorb(transcript, "P_H_star", dynamic_commitments);
    out["y_star"] = transcript.challenge("y_star");

    absorb(transcript, "P_E_src", dynamic_commitments);
    absorb(transcript, "P_H_star", dynamic_commitments);
    out["eta_src"] = transcript.challenge("eta_src");
    out["beta_src"] = transcript.challenge("beta_src");

    absorb(transcript, "P_S", dynamic_commitments);
    absorb(transcript, "P_Z", dynamic_commitments);
    absorb_static(transcript, context, "V_T_L_x");
    absorb_static(transcript, context, "V_T_L_y");
    out["eta_L"] = transcript.challenge("eta_L");
    out["beta_L"] = transcript.challenge("beta_L");

    absorb(transcript, "P_M", dynamic_commitments);
    absorb(transcript, "P_M_edge", dynamic_commitments);
    absorb(transcript, "P_Delta", dynamic_commitments);
    absorb_static(transcript, context, "V_T_range");
    out["beta_R"] = transcript.challenge("beta_R");

    absorb(transcript, "P_Delta", dynamic_commitments);
    absorb(transcript, "P_U", dynamic_commitments);
    absorb_static(transcript, context, "V_T_exp_x");
    absorb_static(transcript, context, "V_T_exp_y");
    out["eta_exp"] = transcript.challenge("eta_exp");
    out["beta_exp"] = transcript.challenge("beta_exp");

    absorb(transcript, "P_H_agg", dynamic_commitments);
    absorb(transcript, "P_H_agg_star", dynamic_commitments);
    out["y_agg"] = transcript.challenge("y_agg");

    absorb(transcript, "P_E_dst", dynamic_commitments);
    absorb(transcript, "P_M", dynamic_commitments);
    absorb(transcript, "P_Sum", dynamic_commitments);
    absorb(transcript, "P_inv", dynamic_commitments);
    absorb(transcript, "P_H_agg_star", dynamic_commitments);
    absorb(transcript, "P_E_dst_edge", dynamic_commitments);
    absorb(transcript, "P_M_edge", dynamic_commitments);
    absorb(transcript, "P_Sum_edge", dynamic_commitments);
    absorb(transcript, "P_inv_edge", dynamic_commitments);
    absorb(transcript, "P_H_agg_star_edge", dynamic_commitments);
    out["eta_dst"] = transcript.challenge("eta_dst");
    out["beta_dst"] = transcript.challenge("beta_dst");

    absorb(transcript, "P_U", dynamic_commitments);
    absorb(transcript, "P_alpha", dynamic_commitments);
    absorb(transcript, "P_H_src_star_edge", dynamic_commitments);
    absorb(transcript, "P_Sum", dynamic_commitments);
    absorb(transcript, "P_H_agg_star", dynamic_commitments);
    absorb(transcript, "P_H_agg_star_edge", dynamic_commitments);
    absorb(transcript, "P_v_hat", dynamic_commitments);
    out["lambda_psq"] = transcript.challenge("lambda_psq");

    absorb(transcript, "P_H_agg", dynamic_commitments);
    absorb(transcript, "P_Y_lin", dynamic_commitments);
    absorb(transcript, "P_Y", dynamic_commitments);
    absorb_static(transcript, context, "V_W_out");
    absorb_static(transcript, context, "V_b");
    out["y_out"] = transcript.challenge("y_out");

    for (const auto& label : dynamic_commitment_labels(context)) {
        absorb(transcript, label, dynamic_commitments);
    }
    for (const auto& label : {
             std::string("V_T_H"),
             std::string("V_T_L_x"),
             std::string("V_T_L_y"),
             std::string("V_T_exp_x"),
             std::string("V_T_exp_y"),
             std::string("V_T_range"),
             std::string("V_W"),
             std::string("V_a_src"),
             std::string("V_a_dst"),
             std::string("V_W_out"),
             std::string("V_b"),
         }) {
        absorb_static(transcript, context, label);
    }
    out["alpha_quot"] = transcript.challenge("alpha_quot");

    if (quotient_commitments.empty()) {
        return out;
    }

    
    for (const auto& label : quotient_commitment_labels(context)) {
        absorb(transcript, label, quotient_commitments);
    }
    out["z_FH"] = transcript.challenge("z_FH");
    out["z_edge"] = transcript.challenge("z_edge");
    out["z_in"] = transcript.challenge("z_in");
    out["z_d"] = transcript.challenge("z_d");
    out["z_N"] = transcript.challenge("z_N");
    out["v_FH"] = transcript.challenge("v_FH");
    out["v_edge"] = transcript.challenge("v_edge");
    out["v_in"] = transcript.challenge("v_in");
    out["v_d"] = transcript.challenge("v_d");
    out["v_N"] = transcript.challenge("v_N");
    out["rho_ext"] = transcript.challenge("rho_ext");
    return out;
}
}
