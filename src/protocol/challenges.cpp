#include "gatzk/protocol/challenges.hpp"

#include <stdexcept>

#include "gatzk/crypto/transcript.hpp"

namespace gatzk::protocol {
namespace {

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

}  // namespace

std::vector<std::string> dynamic_commitment_labels() {
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

std::vector<std::string> quotient_commitment_labels() {
    return {"t_FH", "t_edge", "t_in", "t_d", "t_N"};
}

std::map<std::string, algebra::FieldElement> replay_challenges(
    const ProtocolContext& context,
    const std::unordered_map<std::string, crypto::Commitment>& dynamic_commitments,
    const std::unordered_map<std::string, crypto::Commitment>& quotient_commitments) {
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

    for (const auto& label : dynamic_commitment_labels()) {
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

    
    for (const auto& label : quotient_commitment_labels()) {
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