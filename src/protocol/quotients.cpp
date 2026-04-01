#include "gatzk/protocol/quotients.hpp"

#include <vector>

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;

std::vector<FieldElement> powers(const FieldElement& base, std::size_t max_exponent) {
    std::vector<FieldElement> out(max_exponent + 1, FieldElement::one());
    for (std::size_t i = 1; i <= max_exponent; ++i) {
        out[i] = out[i - 1] * base;
    }
    return out;
}

FieldElement lagrange_first(const algebra::RootOfUnityDomain& domain, const FieldElement& z) {
    return domain.lagrange_basis_eval(0, z);
}

FieldElement lagrange_last(const algebra::RootOfUnityDomain& domain, const FieldElement& z) {
    return domain.lagrange_basis_eval(domain.size - 1, z);
}

FieldElement lagrange_at(const algebra::RootOfUnityDomain& domain, std::size_t index, const FieldElement& z) {
    return domain.lagrange_basis_eval(index, z);
}

FieldElement c_lookup_1(
    const EvalFn& eval,
    const std::string& prefix,
    const FieldElement& beta,
    const FieldElement& z,
    const FieldElement& omega_z) {
    return (eval("P_R_" + prefix, omega_z) - eval("P_R_" + prefix, z))
            * (eval("P_Table_" + prefix, z) + beta)
            * (eval("P_Query_" + prefix, z) + beta)
        - eval("P_Q_tbl_" + prefix, z) * eval("P_m_" + prefix, z) * (eval("P_Query_" + prefix, z) + beta)
        + eval("P_Q_qry_" + prefix, z) * (eval("P_Table_" + prefix, z) + beta);
}

}

std::vector<std::pair<std::string, algebra::Polynomial>> build_multihead_zero_quotients(const ProtocolContext& context) {
    auto zero_eval_poly = [](const std::string& name, const std::shared_ptr<algebra::RootOfUnityDomain>& domain) {
        return algebra::Polynomial::from_evaluations(
            name,
            std::vector<FieldElement>(domain->size, FieldElement::zero()),
            domain);
    };
    return {
        {"t_FH", zero_eval_poly("t_FH", context.domains.fh)},
        {"t_edge", zero_eval_poly("t_edge", context.domains.edge)},
        {"t_in", zero_eval_poly("t_in", context.domains.in)},
        {"t_d_h", zero_eval_poly("t_d_h", context.domains.d)},
        {"t_cat", zero_eval_poly("t_cat", context.domains.cat)},
        {"t_C", zero_eval_poly("t_C", context.domains.c)},
        {"t_N", zero_eval_poly("t_N", context.domains.n)},
    };
}

FieldElement evaluate_t_fh(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto alpha = challenges.at("alpha_quot");
    const auto alpha_powers = powers(alpha, 2);
    const auto beta_feat = challenges.at("beta_feat");
    const auto omega_z = z * context.domains.fh->omega;
    const auto l0 = lagrange_first(*context.domains.fh, z);
    const auto l_last = lagrange_last(*context.domains.fh, z);
    const auto zero_eval = context.domains.fh->zero_polynomial_eval(z);
    const auto numerator =
        l0 * eval("P_R_feat", z)
        + alpha_powers[1] * c_lookup_1(eval, "feat", beta_feat, z, omega_z)
        + alpha_powers[2] * l_last * eval("P_R_feat", z);
    return numerator / zero_eval;
}

FieldElement evaluate_t_edge(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto alpha = challenges.at("alpha_quot");
    const auto alpha_powers = powers(alpha, 73);
    const auto omega_z = z * context.domains.edge->omega;
    const auto beta_src = challenges.at("beta_src");
    const auto beta_dst = challenges.at("beta_dst");
    const auto eta_src = challenges.at("eta_src");
    const auto eta_dst = challenges.at("eta_dst");
    const auto beta_l = challenges.at("beta_L");
    const auto eta_l = challenges.at("eta_L");
    const auto beta_r = challenges.at("beta_R");
    const auto beta_exp = challenges.at("beta_exp");
    const auto eta_exp = challenges.at("eta_exp");
    const auto l0 = lagrange_first(*context.domains.edge, z);
    const auto l_last = lagrange_last(*context.domains.edge, z);
    const auto zero_eval = context.domains.edge->zero_polynomial_eval(z);

    const auto c_src_0 = l0 * eval("P_R_src", z);
    const auto c_src_1 = eval("P_Q_qry_src", z)
        * ((eval("P_R_src", omega_z) - eval("P_R_src", z)) * (eval("P_Query_src", z) + beta_src) - FieldElement::one());
    const auto c_src_2 = (FieldElement::one() - eval("P_Q_qry_src", z)) * (eval("P_R_src", omega_z) - eval("P_R_src", z));
    const auto c_src_3 = l_last * (eval("P_R_src", z) - witness_scalars.at("S_src"));
    const auto c_src_bind = eval("P_Query_src", z)
        - (eval("P_src", z) + eta_src * eval("P_E_src_edge", z) + eta_src.pow(2) * eval("P_H_src_star_edge", z));

    const auto c_dst_0 = l0 * eval("P_R_dst", z);
    const auto c_dst_1 = eval("P_Q_qry_dst", z)
        * ((eval("P_R_dst", omega_z) - eval("P_R_dst", z)) * (eval("P_Query_dst", z) + beta_dst) - FieldElement::one());
    const auto c_dst_2 = (FieldElement::one() - eval("P_Q_qry_dst", z)) * (eval("P_R_dst", omega_z) - eval("P_R_dst", z));
    const auto c_dst_3 = l_last * (eval("P_R_dst", z) - witness_scalars.at("S_dst"));
    const auto c_dst_bind = eval("P_Query_dst", z)
        - (eval("P_dst", z)
            + eta_dst * eval("P_E_dst_edge", z)
            + eta_dst.pow(2) * eval("P_M_edge", z)
            + eta_dst.pow(3) * eval("P_Sum_edge", z)
            + eta_dst.pow(4) * eval("P_inv_edge", z)
            + eta_dst.pow(5) * eval("P_H_agg_star_edge", z));

    const auto c_l_0 = l0 * eval("P_R_L", z);
    const auto c_l_1 = c_lookup_1(eval, "L", beta_l, z, omega_z);
    const auto c_l_2 = l_last * eval("P_R_L", z);
    const auto c_l_bind_tbl = eval("P_Table_L", z) - (eval("P_T_L_x", z) + eta_l * eval("P_T_L_y", z));
    const auto c_l_bind_qry = eval("P_Query_L", z) - (eval("P_S", z) + eta_l * eval("P_Z", z));

    const auto c_r_0 = l0 * eval("P_R_R", z);
    const auto c_r_1 = c_lookup_1(eval, "R", beta_r, z, omega_z);
    const auto c_r_2 = l_last * eval("P_R_R", z);
    const auto c_r_bind_tbl = eval("P_Table_R", z) - eval("P_T_range", z);
    const auto c_r_bind_qry = eval("P_Query_R", z) - eval("P_Delta", z);

    const auto c_exp_0 = l0 * eval("P_R_exp", z);
    const auto c_exp_1 = c_lookup_1(eval, "exp", beta_exp, z, omega_z);
    const auto c_exp_2 = l_last * eval("P_R_exp", z);
    const auto c_exp_bind_tbl = eval("P_Table_exp", z) - (eval("P_T_exp_x", z) + eta_exp * eval("P_T_exp_y", z));
    const auto c_exp_bind_qry = eval("P_Query_exp", z) - (eval("P_Delta", z) + eta_exp * eval("P_U", z));

    const auto c_max_0 = eval("P_Delta", z) - eval("P_M_edge", z) + eval("P_Z", z);
    const auto c_max_1 = eval("P_s_max", z) * (eval("P_s_max", z) - FieldElement::one());
    const auto c_max_2 = eval("P_s_max", z) * eval("P_Delta", z);
    const auto c_max_3 = l0 * (eval("P_C_max", z) - eval("P_s_max", z));
    const auto c_max_4 =
        eval("P_Q_edge_valid", omega_z)
            * (eval("P_C_max", omega_z)
                - eval("P_Q_new_edge", omega_z) * eval("P_s_max", omega_z)
                - (FieldElement::one() - eval("P_Q_new_edge", omega_z)) * (eval("P_C_max", z) + eval("P_s_max", omega_z)))
        + (FieldElement::one() - eval("P_Q_edge_valid", omega_z)) * (eval("P_C_max", omega_z) - eval("P_C_max", z));
    const auto c_max_5 = eval("P_Q_end_edge", z) * (eval("P_C_max", z) - FieldElement::one());

    const auto c_inv_1 = eval("P_alpha", z) - eval("P_U", z) * eval("P_inv_edge", z);
    const auto c_vhat_0 = eval("P_v_hat", z) - eval("P_alpha", z) * eval("P_H_src_star_edge", z);
    const auto c_psq_0 = l0 * (eval("P_PSQ", z) - eval("P_w_psq", z));
    const auto c_psq_1 =
        eval("P_PSQ", omega_z)
        - eval("P_Q_new_edge", omega_z) * eval("P_w_psq", omega_z)
        - (FieldElement::one() - eval("P_Q_new_edge", omega_z)) * (eval("P_PSQ", z) + eval("P_w_psq", omega_z));
    const auto c_psq_2 = eval("P_Q_end_edge", z) * (eval("P_PSQ", z) - eval("P_T_psq_edge", z));

    FieldElement n_edge = FieldElement::zero();
    n_edge += alpha_powers[3] * c_src_0;
    n_edge += alpha_powers[4] * c_src_1;
    n_edge += alpha_powers[5] * c_src_2;
    n_edge += alpha_powers[6] * c_dst_0;
    n_edge += alpha_powers[7] * c_dst_1;
    n_edge += alpha_powers[8] * c_dst_2;
    n_edge += alpha_powers[9] * c_l_0;
    n_edge += alpha_powers[10] * c_l_1;
    n_edge += alpha_powers[11] * c_l_2;
    n_edge += alpha_powers[12] * c_r_0;
    n_edge += alpha_powers[13] * c_r_1;
    n_edge += alpha_powers[14] * c_r_2;
    n_edge += alpha_powers[15] * c_exp_0;
    n_edge += alpha_powers[16] * c_exp_1;
    n_edge += alpha_powers[17] * c_exp_2;
    n_edge += alpha_powers[18] * c_max_0;
    n_edge += alpha_powers[19] * c_max_1;
    n_edge += alpha_powers[20] * c_max_2;
    n_edge += alpha_powers[21] * c_max_3;
    n_edge += alpha_powers[22] * c_max_4;
    n_edge += alpha_powers[23] * c_max_5;
    n_edge += alpha_powers[24] * c_inv_1;
    n_edge += alpha_powers[25] * c_vhat_0;
    n_edge += alpha_powers[26] * c_psq_0;
    n_edge += alpha_powers[27] * c_psq_1;
    n_edge += alpha_powers[28] * c_psq_2;
    n_edge += alpha_powers[54] * c_l_bind_tbl;
    n_edge += alpha_powers[55] * c_l_bind_qry;
    n_edge += alpha_powers[56] * c_r_bind_tbl;
    n_edge += alpha_powers[57] * c_r_bind_qry;
    n_edge += alpha_powers[58] * c_exp_bind_tbl;
    n_edge += alpha_powers[59] * c_exp_bind_qry;
    n_edge += alpha_powers[70] * c_src_3;
    n_edge += alpha_powers[71] * c_src_bind;
    n_edge += alpha_powers[72] * c_dst_3;
    n_edge += alpha_powers[73] * c_dst_bind;
    return n_edge / zero_eval;
}

FieldElement evaluate_t_n(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto c_inv_0 = eval("P_Q_N", z) * (eval("P_Sum", z) * eval("P_inv", z) - FieldElement::one());
    const auto alpha_powers = powers(challenges.at("alpha_quot"), 69);
    const auto q_n = eval("P_Q_N", z);
    const auto q_invalid = FieldElement::one() - q_n;
    const auto omega_z = z * context.domains.n->omega;
    const auto l0 = lagrange_first(*context.domains.n, z);
    const auto l_last = lagrange_last(*context.domains.n, z);
    const auto beta_src = challenges.at("beta_src");
    const auto eta_src = challenges.at("eta_src");
    const auto beta_dst = challenges.at("beta_dst");
    const auto eta_dst = challenges.at("eta_dst");

    const auto c_src_node_0 = l0 * eval("P_R_src_node", z);
    const auto c_src_node_1 = q_n
        * ((eval("P_R_src_node", omega_z) - eval("P_R_src_node", z)) * (eval("P_Table_src", z) + beta_src) - eval("P_m_src", z));
    const auto c_src_node_2 = q_invalid * (eval("P_R_src_node", omega_z) - eval("P_R_src_node", z));
    const auto c_src_node_3 = l_last * (eval("P_R_src_node", z) - witness_scalars.at("S_src"));
    const auto c_src_node_bind = eval("P_Table_src", z)
        - (eval("P_I", z) + eta_src * eval("P_E_src", z) + eta_src.pow(2) * eval("P_H_star", z));

    const auto c_dst_node_0 = l0 * eval("P_R_dst_node", z);
    const auto c_dst_node_1 = q_n
        * ((eval("P_R_dst_node", omega_z) - eval("P_R_dst_node", z)) * (eval("P_Table_dst", z) + beta_dst) - eval("P_m_dst", z));
    const auto c_dst_node_2 = q_invalid * (eval("P_R_dst_node", omega_z) - eval("P_R_dst_node", z));
    const auto c_dst_node_3 = l_last * (eval("P_R_dst_node", z) - witness_scalars.at("S_dst"));
    const auto c_dst_node_bind = eval("P_Table_dst", z)
        - (eval("P_I", z)
            + eta_dst * eval("P_E_dst", z)
            + eta_dst.pow(2) * eval("P_M", z)
            + eta_dst.pow(3) * eval("P_Sum", z)
            + eta_dst.pow(4) * eval("P_inv", z)
            + eta_dst.pow(5) * eval("P_H_agg_star", z));

    FieldElement numerator = alpha_powers[29] * c_inv_0;
    numerator += alpha_powers[60] * c_src_node_0;
    numerator += alpha_powers[61] * c_src_node_1;
    numerator += alpha_powers[62] * c_src_node_2;
    numerator += alpha_powers[63] * c_src_node_3;
    numerator += alpha_powers[64] * c_src_node_bind;
    numerator += alpha_powers[65] * c_dst_node_0;
    numerator += alpha_powers[66] * c_dst_node_1;
    numerator += alpha_powers[67] * c_dst_node_2;
    numerator += alpha_powers[68] * c_dst_node_3;
    numerator += alpha_powers[69] * c_dst_node_bind;
    return numerator / context.domains.n->zero_polynomial_eval(z);
}

FieldElement evaluate_t_in(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto omega_z = z * context.domains.in->omega;
    const auto l0 = lagrange_first(*context.domains.in, z);
    const auto ld = lagrange_at(*context.domains.in, context.local.num_features, z);
    const auto zero_eval = context.domains.in->zero_polynomial_eval(z);
    const auto c_proj_0 = l0 * eval("P_Acc_proj", z);
    const auto c_proj_1 = eval("P_Q_proj_valid", z)
        * (eval("P_Acc_proj", omega_z) - eval("P_Acc_proj", z) - eval("P_a_proj", z) * eval("P_b_proj", z));
    const auto c_proj_2 = (FieldElement::one() - eval("P_Q_proj_valid", z))
        * (eval("P_Acc_proj", omega_z) - eval("P_Acc_proj", z));
    const auto c_proj_3 = ld * (eval("P_Acc_proj", z) - external_evaluations.at("mu_proj"));
    const auto alpha = challenges.at("alpha_quot");
    const auto alpha_powers = powers(alpha, 33);
    const auto numerator =
        alpha_powers[30] * c_proj_0
        + alpha_powers[31] * c_proj_1
        + alpha_powers[32] * c_proj_2
        + alpha_powers[33] * c_proj_3;
    return numerator / zero_eval;
}

FieldElement evaluate_t_d(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto omega_z = z * context.domains.d->omega;
    const auto q_valid = eval("P_Q_d_valid", z);
    const auto q_invalid = FieldElement::one() - q_valid;
    const auto l0 = lagrange_first(*context.domains.d, z);
    const auto ld = lagrange_at(*context.domains.d, context.model.a_src.size(), z);
    const auto zero_eval = context.domains.d->zero_polynomial_eval(z);

    auto c0 = [&](const std::string& acc) { return l0 * eval(acc, z); };
    auto c1 = [&](const std::string& a, const std::string& b, const std::string& acc) {
        return q_valid * (eval(acc, omega_z) - eval(acc, z) - eval(a, z) * eval(b, z));
    };
    auto c2 = [&](const std::string& acc) { return q_invalid * (eval(acc, omega_z) - eval(acc, z)); };
    auto c3 = [&](const std::string& acc, const std::string& mu_name) {
        return ld * (eval(acc, z) - external_evaluations.at(mu_name));
    };

    const auto alpha = challenges.at("alpha_quot");
    const auto alpha_powers = powers(alpha, 53);
    FieldElement numerator = FieldElement::zero();
    numerator += alpha_powers[34] * c0("P_Acc_src");
    numerator += alpha_powers[35] * c1("P_a_src", "P_b_src", "P_Acc_src");
    numerator += alpha_powers[36] * c2("P_Acc_src");
    numerator += alpha_powers[37] * c3("P_Acc_src", "mu_src");
    numerator += alpha_powers[38] * c0("P_Acc_dst");
    numerator += alpha_powers[39] * c1("P_a_dst", "P_b_dst", "P_Acc_dst");
    numerator += alpha_powers[40] * c2("P_Acc_dst");
    numerator += alpha_powers[41] * c3("P_Acc_dst", "mu_dst");
    numerator += alpha_powers[42] * c0("P_Acc_star");
    numerator += alpha_powers[43] * c1("P_a_star", "P_b_star", "P_Acc_star");
    numerator += alpha_powers[44] * c2("P_Acc_star");
    numerator += alpha_powers[45] * c3("P_Acc_star", "mu_star");
    numerator += alpha_powers[46] * c0("P_Acc_agg");
    numerator += alpha_powers[47] * c1("P_a_agg", "P_b_agg", "P_Acc_agg");
    numerator += alpha_powers[48] * c2("P_Acc_agg");
    numerator += alpha_powers[49] * c3("P_Acc_agg", "mu_agg");
    numerator += alpha_powers[50] * c0("P_Acc_out");
    numerator += alpha_powers[51] * c1("P_a_out", "P_b_out", "P_Acc_out");
    numerator += alpha_powers[52] * c2("P_Acc_out");
    numerator += alpha_powers[53] * c3("P_Acc_out", "mu_Y_lin");
    return numerator / zero_eval;
}

}  // namespace gatzk::protocol
