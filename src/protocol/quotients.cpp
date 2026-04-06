#include "gatzk/protocol/quotients.hpp"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <string_view>
#include <vector>

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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

bool starts_with(const std::string& value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::optional<std::string> hidden_head_suffix(const std::string& label) {
    if (!starts_with(label, "P_h")) {
        return std::nullopt;
    }
    std::size_t pos = 3;
    while (pos < label.size() && std::isdigit(static_cast<unsigned char>(label[pos]))) {
        ++pos;
    }
    if (pos >= label.size() || label[pos] != '_') {
        return std::nullopt;
    }
    return label.substr(pos + 1);
}

std::optional<std::string> output_head_suffix(const std::string& label) {
    if (!starts_with(label, "P_out")) {
        return std::nullopt;
    }
    std::size_t pos = 5;
    while (pos < label.size() && std::isdigit(static_cast<unsigned char>(label[pos]))) {
        ++pos;
    }
    if (pos >= label.size() || label[pos] != '_') {
        return std::nullopt;
    }
    return label.substr(pos + 1);
}

bool hidden_concat_binding_label(const ProtocolContext& context, const std::string& label, std::string_view suffix) {
    for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index) {
        const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
        if (label == hidden_layer_cat_prefix(layer_index, is_final_layer) + std::string(suffix)) {
            return true;
        }
    }
    return false;
}

FieldElement quantize_bias_value(double value) {
    const auto scaled = value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5;
    return FieldElement::from_signed(static_cast<std::int64_t>(scaled));
}

FieldElement bias_fold_vector(
    const std::vector<double>& bias,
    std::size_t node_count,
    const FieldElement& y_out) {
    if (bias.empty() || node_count == 0) {
        return FieldElement::zero();
    }

    FieldElement column_fold = FieldElement::zero();
    FieldElement column_power = FieldElement::one();
    for (std::size_t col = 0; col < bias.size(); ++col) {
        column_fold += quantize_bias_value(bias[col]) * column_power;
        column_power *= y_out;
    }
    const auto row_stride = y_out.pow(static_cast<std::uint64_t>(bias.size()));
    FieldElement row_sum = FieldElement::zero();
    if (row_stride == FieldElement::one()) {
        row_sum = FieldElement(static_cast<std::uint64_t>(node_count));
    } else {
        row_sum = (row_stride.pow(static_cast<std::uint64_t>(node_count)) - FieldElement::one())
            / (row_stride - FieldElement::one());
    }
    return column_fold * row_sum;
}

std::optional<std::string> dynamic_label_domain_name(const ProtocolContext& context, const std::string& label) {
    if (!context.model.has_real_multihead) {
        return std::nullopt;
    }
    if (label == "P_H" || label == "P_Table_feat" || label == "P_Query_feat" || label == "P_m_feat" || label == "P_R_feat") {
        return std::string("FH");
    }
    for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index) {
        const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
        if (label == hidden_layer_concat_star_label(layer_index, is_final_layer)) {
            return std::string("N");
        }
    }
    if (hidden_concat_binding_label(context, label, "_a")
        || hidden_concat_binding_label(context, label, "_b")
        || hidden_concat_binding_label(context, label, "_Acc")) {
        return std::string("cat");
    }
    if (const auto suffix = output_head_suffix(label); suffix.has_value()) {
        const auto& name = *suffix;
        if (name == "a_proj" || name == "b_proj" || name == "Acc_proj") {
            return std::string("cat");
        }
        if (name == "a_src" || name == "b_src" || name == "Acc_src"
            || name == "a_dst" || name == "b_dst" || name == "Acc_dst"
            || name == "a_y" || name == "b_y" || name == "Acc_y") {
            return std::string("C");
        }
        if (name == "E_src" || name == "E_dst" || name == "M"
            || name == "Sum" || name == "inv" || name == "Y_prime_star"
            || name == "Table_src" || name == "m_src" || name == "R_src_node"
            || name == "Y_star" || name == "T" || name == "Table_dst"
            || name == "m_dst" || name == "R_dst_node"
            || name == "Table_t" || name == "m_t" || name == "R_t_node") {
            return std::string("N");
        }
        if (name == "E_src_edge" || name == "E_dst_edge" || name == "Query_src"
            || name == "R_src" || name == "S" || name == "Z"
            || name == "Table_L" || name == "Query_L" || name == "m_L" || name == "R_L"
            || name == "M_edge" || name == "Delta" || name == "U"
            || name == "s_max" || name == "C_max"
            || name == "Table_R" || name == "Query_R" || name == "m_R" || name == "R_R"
            || name == "Sum_edge" || name == "inv_edge" || name == "alpha"
            || name == "Table_exp" || name == "Query_exp" || name == "m_exp" || name == "R_exp"
            || name == "Y_prime_star_edge" || name == "widehat_y_star"
            || name == "w" || name == "T_edge" || name == "PSQ"
            || name == "Y_star_edge" || name == "Query_dst" || name == "R_dst"
            || name == "Query_t" || name == "R_t") {
            return std::string("edge");
        }
    }
    if (label == "P_H_cat_star") {
        return std::string("N");
    }
    if (label == "P_cat_a" || label == "P_cat_b" || label == "P_cat_Acc") {
        return std::string("cat");
    }
    if (const auto suffix = hidden_head_suffix(label); suffix.has_value()) {
        const auto& name = *suffix;
        if (name == "a_proj" || name == "b_proj" || name == "Acc_proj") {
            return std::string("in");
        }
        if (name == "a_src" || name == "b_src" || name == "Acc_src"
            || name == "a_dst" || name == "b_dst" || name == "Acc_dst"
            || name == "a_star" || name == "b_star" || name == "Acc_star"
            || name == "a_agg_pre" || name == "b_agg_pre" || name == "Acc_agg_pre"
            || name == "a_agg" || name == "b_agg" || name == "Acc_agg") {
            return std::string("d");
        }
        if (name == "E_src" || name == "E_dst" || name == "M" || name == "Sum" || name == "inv"
            || name == "H_star" || name == "Table_src" || name == "m_src" || name == "R_src_node"
            || name == "H_agg_pre_star" || name == "T_psq" || name == "H_agg_star"
            || name == "Table_dst" || name == "m_dst" || name == "R_dst_node"
            || name == "Table_t" || name == "m_t" || name == "R_t_node") {
            return std::string("N");
        }
        if (name == "E_src_edge" || name == "E_dst_edge" || name == "H_src_star_edge"
            || name == "Query_src" || name == "R_src" || name == "S" || name == "Z"
            || name == "Table_L" || name == "Query_L" || name == "m_L" || name == "R_L"
            || name == "M_edge" || name == "Delta" || name == "U" || name == "Sum_edge"
            || name == "s_max" || name == "C_max"
            || name == "Table_R" || name == "Query_R" || name == "m_R" || name == "R_R"
            || name == "inv_edge" || name == "alpha" || name == "H_agg_pre_star_edge"
            || name == "H_agg_pre_flat" || name == "H_agg_flat"
            || name == "Table_exp" || name == "Query_exp" || name == "m_exp" || name == "R_exp"
            || name == "Table_ELU" || name == "Query_ELU" || name == "m_ELU" || name == "R_ELU"
            || name == "widehat_v_pre_star" || name == "w_psq" || name == "T_psq_edge"
            || name == "PSQ" || name == "H_agg_star_edge" || name == "Query_dst"
            || name == "R_dst" || name == "Query_t" || name == "R_t") {
            return std::string("edge");
        }
    }
    return std::nullopt;
}

class QuotientAccumulator {
  public:
    explicit QuotientAccumulator(const FieldElement& alpha) : alpha_(alpha) {}

    void add(const FieldElement& term) {
        sum_ += power_ * term;
        power_ *= alpha_;
    }

    FieldElement value() const {
        return sum_;
    }

  private:
    FieldElement alpha_;
    FieldElement power_ = FieldElement::one();
    FieldElement sum_ = FieldElement::zero();
};

FieldElement route_node_step(
    const FieldElement& next_value,
    const FieldElement& current_value,
    const FieldElement& table_value,
    const FieldElement& beta,
    const FieldElement& q_valid,
    const FieldElement& multiplicity) {
    return (next_value - current_value) * (table_value + beta) - q_valid * multiplicity;
}

FieldElement route_edge_step(
    const FieldElement& next_value,
    const FieldElement& current_value,
    const FieldElement& query_value,
    const FieldElement& beta,
    const FieldElement& q_edge_valid) {
    return (next_value - current_value) * (query_value + beta) - q_edge_valid;
}

FieldElement acc_step(
    const FieldElement& next_value,
    const FieldElement& current_value,
    const FieldElement& a_value,
    const FieldElement& b_value) {
    return next_value - current_value - a_value * b_value;
}

FieldElement psq_step(
    const FieldElement& psq_next,
    const FieldElement& psq_current,
    const FieldElement& w_next,
    const FieldElement& q_valid_next,
    const FieldElement& q_new_next) {
    return psq_next
        - ((FieldElement::one() - q_valid_next) * psq_current
            + q_valid_next * (q_new_next * w_next + (FieldElement::one() - q_new_next) * (psq_current + w_next)));
}

FieldElement lookup_step(
    const FieldElement& r_next,
    const FieldElement& r_current,
    const FieldElement& table_value,
    const FieldElement& query_value,
    const FieldElement& multiplicity,
    const FieldElement& q_tbl,
    const FieldElement& q_qry,
    const FieldElement& beta) {
    return (r_next - r_current) * (table_value + beta) * (query_value + beta)
        - q_tbl * multiplicity * (query_value + beta)
        + q_qry * (table_value + beta);
}

FieldElement zero_on_unselected(const FieldElement& selector, const FieldElement& value) {
    return (FieldElement::one() - selector) * value;
}

}

std::vector<std::string> domain_opening_labels(const ProtocolContext& context, const std::string& domain_name) {
    std::vector<std::string> labels;
    for (const auto& label : dynamic_commitment_labels(context)) {
        if (const auto mapped = dynamic_label_domain_name(context, label); mapped.has_value() && *mapped == domain_name) {
            labels.push_back(label);
        }
    }
    return labels;
}

FieldElement evaluate_t_fh(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const EvalFn& eval,
    const FieldElement& z,
    FHQuotientProfile* profile) {
    const auto total_start = Clock::now();
    const auto dependency_before = profile != nullptr ? profile->dependency_eval_ms : 0.0;
    auto profiled_eval = [&](const std::string& name, const FieldElement& point) {
        const auto eval_start = Clock::now();
        const auto value = eval(name, point);
        if (profile != nullptr) {
            profile->dependency_eval_ms += elapsed_ms(eval_start, Clock::now());
        }
        return value;
    };
    const auto beta_feat = challenges.at("beta_feat");
    const auto eta_feat = challenges.at("eta_feat");
    const auto omega_z = z * context.domains.fh->omega;
    const auto l0 = lagrange_first(*context.domains.fh, z);
    const auto l_last = lagrange_last(*context.domains.fh, z);
    const auto zero_eval = context.domains.fh->zero_polynomial_eval(z);
    QuotientAccumulator acc(challenges.at("alpha_quot"));
    acc.add(l0 * profiled_eval("P_R_feat", z));
    acc.add(c_lookup_1(profiled_eval, "feat", beta_feat, z, omega_z));
    acc.add(l_last * profiled_eval("P_R_feat", z));
    acc.add(
        profiled_eval("P_Table_feat", z)
        - (profiled_eval("P_Row_feat_tbl", z)
            + eta_feat * profiled_eval("P_Col_feat_tbl", z)
            + eta_feat.pow(2) * profiled_eval("P_T_H", z)));
    acc.add(
        profiled_eval("P_Query_feat", z)
        - (profiled_eval("P_I_feat_qry", z)
            + eta_feat * profiled_eval("P_Col_feat_qry", z)
            + eta_feat.pow(2) * profiled_eval("P_H", z)));
    const auto result = acc.value() / zero_eval;
    if (profile != nullptr) {
        const auto total_ms = elapsed_ms(total_start, Clock::now());
        profile->assembly_ms += total_ms - (profile->dependency_eval_ms - dependency_before);
    }
    return result;
}

FieldElement evaluate_t_edge(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const EvalFn& eval,
    const FieldElement& z) {
    if (context.model.has_real_multihead) {
        const auto omega_z = z * context.domains.edge->omega;
        const auto first = lagrange_first(*context.domains.edge, z);
        const auto last = lagrange_last(*context.domains.edge, z);
        const auto zero_eval = context.domains.edge->zero_polynomial_eval(z);
        const auto one = FieldElement::one();
        std::unordered_map<std::string, FieldElement> z_cache;
        std::unordered_map<std::string, FieldElement> omega_cache;
        z_cache.reserve(256);
        omega_cache.reserve(128);
        auto cached_eval = [&](std::unordered_map<std::string, FieldElement>& cache,
                               const FieldElement& point,
                               const std::string& label) {
            if (const auto it = cache.find(label); it != cache.end()) {
                return it->second;
            }
            const auto value = eval(label, point);
            cache.emplace(label, value);
            return value;
        };
        auto eval_z = [&](const std::string& label) {
            return cached_eval(z_cache, z, label);
        };
        auto eval_omega = [&](const std::string& label) {
            return cached_eval(omega_cache, omega_z, label);
        };

        const auto q_edge = eval_z("P_Q_edge_valid");
        const auto q_edge_next = eval_omega("P_Q_edge_valid");
        const auto q_new_next = eval_omega("P_Q_new_edge");
        const auto q_end = eval_z("P_Q_end_edge");
        const auto src_z = eval_z("P_src");
        const auto dst_z = eval_z("P_dst");
        const auto q_tbl_l_z = eval_z("P_Q_tbl_L");
        const auto q_qry_l_z = eval_z("P_Q_qry_L");
        const auto q_tbl_r_z = eval_z("P_Q_tbl_R");
        const auto q_qry_r_z = eval_z("P_Q_qry_R");
        const auto q_tbl_exp_z = eval_z("P_Q_tbl_exp");
        const auto q_qry_exp_z = eval_z("P_Q_qry_exp");
        const auto q_tbl_elu_z = eval_z("P_Q_tbl_ELU");
        const auto q_qry_elu_z = eval_z("P_Q_qry_ELU");
        const auto t_l_x_z = eval_z("P_T_L_x");
        const auto t_l_y_z = eval_z("P_T_L_y");
        const auto t_range_z = eval_z("P_T_range");
        const auto t_exp_x_z = eval_z("P_T_exp_x");
        const auto t_exp_y_z = eval_z("P_T_exp_y");
        const auto t_elu_x_z = eval_z("P_T_ELU_x");
        const auto t_elu_y_z = eval_z("P_T_ELU_y");
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            const auto prefix = "P_h" + std::to_string(head_index) + "_";
            const auto beta_src = challenges.at("beta_src_h" + std::to_string(head_index));
            const auto beta_dst = challenges.at("beta_dst_h" + std::to_string(head_index));
            const auto eta_src = challenges.at("eta_src_h" + std::to_string(head_index));
            const auto eta_dst = challenges.at("eta_dst_h" + std::to_string(head_index));
            const auto beta_l = challenges.at("beta_L_h" + std::to_string(head_index));
            const auto eta_l = challenges.at("eta_L_h" + std::to_string(head_index));
            const auto beta_r = challenges.at("beta_R_h" + std::to_string(head_index));
            const auto beta_exp = challenges.at("beta_exp_h" + std::to_string(head_index));
            const auto eta_exp = challenges.at("eta_exp_h" + std::to_string(head_index));
            const auto beta_elu = challenges.at("beta_ELU_h" + std::to_string(head_index));
            const auto eta_elu = challenges.at("eta_ELU_h" + std::to_string(head_index));
            const auto beta_t = challenges.at("beta_t_h" + std::to_string(head_index));
            const auto eta_t = challenges.at("eta_t_h" + std::to_string(head_index));
            const auto eta_src_sq = eta_src * eta_src;
            const auto eta_dst_sq = eta_dst * eta_dst;
            const auto eta_dst_cu = eta_dst_sq * eta_dst;
            const auto eta_dst_4 = eta_dst_cu * eta_dst;
            const auto eta_dst_5 = eta_dst_4 * eta_dst;

            auto z_head = [&](std::string_view suffix) {
                return eval_z(prefix + std::string(suffix));
            };
            auto omega_head = [&](std::string_view suffix) {
                return eval_omega(prefix + std::string(suffix));
            };

            const auto r_src_z = z_head("R_src");
            const auto r_src_omega = omega_head("R_src");
            const auto query_src_z = z_head("Query_src");
            const auto e_src_edge_z = z_head("E_src_edge");
            const auto e_dst_edge_z = z_head("E_dst_edge");
            const auto h_src_star_edge_z = z_head("H_src_star_edge");
            const auto s_z = z_head("S");
            const auto r_l_z = z_head("R_L");
            const auto r_l_omega = omega_head("R_L");
            const auto table_l_z = z_head("Table_L");
            const auto query_l_z = z_head("Query_L");
            const auto m_l_z = z_head("m_L");
            const auto z_z = z_head("Z");
            const auto delta_z = z_head("Delta");
            const auto m_edge_z = z_head("M_edge");
            const auto s_max_z = z_head("s_max");
            const auto s_max_omega = omega_head("s_max");
            const auto c_max_z = z_head("C_max");
            const auto c_max_omega = omega_head("C_max");
            const auto r_r_z = z_head("R_R");
            const auto r_r_omega = omega_head("R_R");
            const auto table_r_z = z_head("Table_R");
            const auto query_r_z = z_head("Query_R");
            const auto m_r_z = z_head("m_R");
            const auto alpha_z = z_head("alpha");
            const auto u_z = z_head("U");
            const auto inv_edge_z = z_head("inv_edge");
            const auto r_exp_z = z_head("R_exp");
            const auto r_exp_omega = omega_head("R_exp");
            const auto table_exp_z = z_head("Table_exp");
            const auto query_exp_z = z_head("Query_exp");
            const auto m_exp_z = z_head("m_exp");
            const auto psq_z = z_head("PSQ");
            const auto psq_omega = omega_head("PSQ");
            const auto w_psq_z = z_head("w_psq");
            const auto w_psq_omega = omega_head("w_psq");
            const auto t_psq_edge_z = z_head("T_psq_edge");
            const auto r_t_z = z_head("R_t");
            const auto r_t_omega = omega_head("R_t");
            const auto query_t_z = z_head("Query_t");
            const auto r_elu_z = z_head("R_ELU");
            const auto r_elu_omega = omega_head("R_ELU");
            const auto table_elu_z = z_head("Table_ELU");
            const auto query_elu_z = z_head("Query_ELU");
            const auto m_elu_z = z_head("m_ELU");
            const auto h_agg_pre_flat_z = z_head("H_agg_pre_flat");
            const auto h_agg_flat_z = z_head("H_agg_flat");
            const auto r_dst_z = z_head("R_dst");
            const auto r_dst_omega = omega_head("R_dst");
            const auto query_dst_z = z_head("Query_dst");
            const auto sum_edge_z = z_head("Sum_edge");
            const auto h_agg_star_edge_z = z_head("H_agg_star_edge");

            acc.add(first * r_src_z);
            acc.add(route_edge_step(r_src_omega, r_src_z, query_src_z, beta_src, q_edge));
            acc.add(last * (r_src_z - witness_scalars.at("S_src_h" + std::to_string(head_index))));
            acc.add(query_src_z - (src_z + eta_src * e_src_edge_z + eta_src_sq * h_src_star_edge_z));
            acc.add(zero_on_unselected(q_edge, query_src_z));

            acc.add(q_edge * (s_z - e_src_edge_z - e_dst_edge_z));
            acc.add(first * r_l_z);
            acc.add(lookup_step(
                r_l_omega,
                r_l_z,
                table_l_z,
                query_l_z,
                m_l_z,
                q_tbl_l_z,
                q_qry_l_z,
                beta_l));
            acc.add(last * r_l_z);
            acc.add(table_l_z - (t_l_x_z + eta_l * t_l_y_z));
            acc.add(query_l_z - (s_z + eta_l * z_z));
            acc.add(zero_on_unselected(q_tbl_l_z, table_l_z));
            acc.add(zero_on_unselected(q_tbl_l_z, m_l_z));
            acc.add(zero_on_unselected(q_qry_l_z, query_l_z));

            acc.add(q_edge * (delta_z - m_edge_z + z_z));
            acc.add(s_max_z * (s_max_z - one));
            acc.add(s_max_z * delta_z);
            acc.add(zero_on_unselected(q_edge, s_max_z));
            acc.add(first * (c_max_z - s_max_z));
            acc.add(
                q_edge_next
                * (c_max_omega
                    - q_new_next * s_max_omega
                    - (one - q_new_next) * (c_max_z + s_max_omega))
                + (one - q_edge_next) * (c_max_omega - c_max_z));
            acc.add(q_end * (c_max_z - one));

            acc.add(first * r_r_z);
            acc.add(lookup_step(
                r_r_omega,
                r_r_z,
                table_r_z,
                query_r_z,
                m_r_z,
                q_tbl_r_z,
                q_qry_r_z,
                beta_r));
            acc.add(last * r_r_z);
            acc.add(table_r_z - t_range_z);
            acc.add(query_r_z - delta_z);
            acc.add(zero_on_unselected(q_tbl_r_z, table_r_z));
            acc.add(zero_on_unselected(q_tbl_r_z, m_r_z));
            acc.add(zero_on_unselected(q_qry_r_z, query_r_z));

            acc.add(q_edge * (alpha_z - u_z * inv_edge_z));
            acc.add(first * r_exp_z);
            acc.add(lookup_step(
                r_exp_omega,
                r_exp_z,
                table_exp_z,
                query_exp_z,
                m_exp_z,
                q_tbl_exp_z,
                q_qry_exp_z,
                beta_exp));
            acc.add(last * r_exp_z);
            acc.add(table_exp_z - (t_exp_x_z + eta_exp * t_exp_y_z));
            acc.add(query_exp_z - (delta_z + eta_exp * u_z));
            acc.add(zero_on_unselected(q_tbl_exp_z, table_exp_z));
            acc.add(zero_on_unselected(q_tbl_exp_z, m_exp_z));
            acc.add(zero_on_unselected(q_qry_exp_z, query_exp_z));

            acc.add(first * (psq_z - q_edge * w_psq_z));
            acc.add(psq_step(psq_omega, psq_z, w_psq_omega, q_edge_next, q_new_next));
            acc.add(q_end * (psq_z - t_psq_edge_z));
            acc.add(first * r_t_z);
            acc.add(route_edge_step(r_t_omega, r_t_z, query_t_z, beta_t, q_edge));
            acc.add(last * (r_t_z - witness_scalars.at("S_t_h" + std::to_string(head_index))));
            acc.add(query_t_z - (dst_z + eta_t * t_psq_edge_z));
            acc.add(zero_on_unselected(q_edge, query_t_z));

            acc.add(first * r_elu_z);
            acc.add(lookup_step(
                r_elu_omega,
                r_elu_z,
                table_elu_z,
                query_elu_z,
                m_elu_z,
                q_tbl_elu_z,
                q_qry_elu_z,
                beta_elu));
            acc.add(last * r_elu_z);
            acc.add(table_elu_z - (t_elu_x_z + eta_elu * t_elu_y_z));
            acc.add(query_elu_z - (h_agg_pre_flat_z + eta_elu * h_agg_flat_z));
            acc.add(zero_on_unselected(q_tbl_elu_z, table_elu_z));
            acc.add(zero_on_unselected(q_tbl_elu_z, m_elu_z));
            acc.add(zero_on_unselected(q_qry_elu_z, query_elu_z));

            acc.add(first * r_dst_z);
            acc.add(route_edge_step(r_dst_omega, r_dst_z, query_dst_z, beta_dst, q_edge));
            acc.add(last * (r_dst_z - witness_scalars.at("S_dst_h" + std::to_string(head_index))));
            acc.add(query_dst_z
                - (dst_z
                    + eta_dst * e_dst_edge_z
                    + eta_dst_sq * m_edge_z
                    + eta_dst_cu * sum_edge_z
                    + eta_dst_4 * inv_edge_z
                    + eta_dst_5 * h_agg_star_edge_z));
            acc.add(zero_on_unselected(q_edge, query_dst_z));
        }

        const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
            const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
            const auto beta_src_out = challenges.at(output_challenge_name("beta_src_out", head_index, legacy_single_output));
            const auto beta_dst_out = challenges.at(output_challenge_name("beta_dst_out", head_index, legacy_single_output));
            const auto eta_src_out = challenges.at(output_challenge_name("eta_src_out", head_index, legacy_single_output));
            const auto eta_dst_out = challenges.at(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
            const auto beta_l_out = challenges.at(output_challenge_name("beta_L_out", head_index, legacy_single_output));
            const auto eta_l_out = challenges.at(output_challenge_name("eta_L_out", head_index, legacy_single_output));
            const auto beta_r_out = challenges.at(output_challenge_name("beta_R_out", head_index, legacy_single_output));
            const auto beta_exp_out = challenges.at(output_challenge_name("beta_exp_out", head_index, legacy_single_output));
            const auto eta_exp_out = challenges.at(output_challenge_name("eta_exp_out", head_index, legacy_single_output));
            const auto beta_t_out = challenges.at(output_challenge_name("beta_t_out", head_index, legacy_single_output));
            const auto eta_t_out = challenges.at(output_challenge_name("eta_t_out", head_index, legacy_single_output));
            const auto eta_dst_out_sq = eta_dst_out * eta_dst_out;
            const auto eta_dst_out_cu = eta_dst_out_sq * eta_dst_out;
            const auto eta_dst_out_4 = eta_dst_out_cu * eta_dst_out;
            const auto eta_dst_out_5 = eta_dst_out_4 * eta_dst_out;

            auto z_out = [&](std::string_view suffix) {
                return eval_z(prefix + std::string(suffix));
            };
            auto omega_out = [&](std::string_view suffix) {
                return eval_omega(prefix + std::string(suffix));
            };

            const auto r_src_z = z_out("R_src");
            const auto r_src_omega = omega_out("R_src");
            const auto query_src_z = z_out("Query_src");
            const auto e_src_edge_z = z_out("E_src_edge");
            const auto e_dst_edge_z = z_out("E_dst_edge");
            const auto s_z = z_out("S");
            const auto r_l_z = z_out("R_L");
            const auto r_l_omega = omega_out("R_L");
            const auto table_l_z = z_out("Table_L");
            const auto query_l_z = z_out("Query_L");
            const auto m_l_z = z_out("m_L");
            const auto z_z = z_out("Z");
            const auto delta_z = z_out("Delta");
            const auto m_edge_z = z_out("M_edge");
            const auto s_max_z = z_out("s_max");
            const auto s_max_omega = omega_out("s_max");
            const auto c_max_z = z_out("C_max");
            const auto c_max_omega = omega_out("C_max");
            const auto r_r_z = z_out("R_R");
            const auto r_r_omega = omega_out("R_R");
            const auto table_r_z = z_out("Table_R");
            const auto query_r_z = z_out("Query_R");
            const auto m_r_z = z_out("m_R");
            const auto alpha_z = z_out("alpha");
            const auto u_z = z_out("U");
            const auto inv_edge_z = z_out("inv_edge");
            const auto r_exp_z = z_out("R_exp");
            const auto r_exp_omega = omega_out("R_exp");
            const auto table_exp_z = z_out("Table_exp");
            const auto query_exp_z = z_out("Query_exp");
            const auto m_exp_z = z_out("m_exp");
            const auto psq_z = z_out("PSQ");
            const auto psq_omega = omega_out("PSQ");
            const auto w_z = z_out("w");
            const auto w_omega = omega_out("w");
            const auto t_edge_z = z_out("T_edge");
            const auto r_t_z = z_out("R_t");
            const auto r_t_omega = omega_out("R_t");
            const auto query_t_z = z_out("Query_t");
            const auto r_dst_z = z_out("R_dst");
            const auto r_dst_omega = omega_out("R_dst");
            const auto query_dst_z = z_out("Query_dst");
            const auto sum_edge_z = z_out("Sum_edge");
            const auto y_star_edge_z = z_out("Y_star_edge");

            acc.add(first * r_src_z);
            acc.add(route_edge_step(r_src_omega, r_src_z, query_src_z, beta_src_out, q_edge));
            acc.add(last * (r_src_z - witness_scalars.at(output_witness_scalar_name("src", head_index, legacy_single_output))));
            acc.add(query_src_z - (src_z + eta_src_out * e_src_edge_z));
            acc.add(zero_on_unselected(q_edge, query_src_z));

            acc.add(q_edge * (s_z - e_src_edge_z - e_dst_edge_z));
            acc.add(first * r_l_z);
            acc.add(lookup_step(
                r_l_omega,
                r_l_z,
                table_l_z,
                query_l_z,
                m_l_z,
                q_tbl_l_z,
                q_qry_l_z,
                beta_l_out));
            acc.add(last * r_l_z);
            acc.add(table_l_z - (t_l_x_z + eta_l_out * t_l_y_z));
            acc.add(query_l_z - (s_z + eta_l_out * z_z));
            acc.add(zero_on_unselected(q_tbl_l_z, table_l_z));
            acc.add(zero_on_unselected(q_tbl_l_z, m_l_z));
            acc.add(zero_on_unselected(q_qry_l_z, query_l_z));

            acc.add(q_edge * (delta_z - m_edge_z + z_z));
            acc.add(s_max_z * (s_max_z - one));
            acc.add(s_max_z * delta_z);
            acc.add(zero_on_unselected(q_edge, s_max_z));
            acc.add(first * (c_max_z - s_max_z));
            acc.add(
                q_edge_next
                * (c_max_omega
                    - q_new_next * s_max_omega
                    - (one - q_new_next) * (c_max_z + s_max_omega))
                + (one - q_edge_next) * (c_max_omega - c_max_z));
            acc.add(q_end * (c_max_z - one));

            acc.add(first * r_r_z);
            acc.add(lookup_step(
                r_r_omega,
                r_r_z,
                table_r_z,
                query_r_z,
                m_r_z,
                q_tbl_r_z,
                q_qry_r_z,
                beta_r_out));
            acc.add(last * r_r_z);
            acc.add(table_r_z - t_range_z);
            acc.add(query_r_z - delta_z);
            acc.add(zero_on_unselected(q_tbl_r_z, table_r_z));
            acc.add(zero_on_unselected(q_tbl_r_z, m_r_z));
            acc.add(zero_on_unselected(q_qry_r_z, query_r_z));

            acc.add(q_edge * (alpha_z - u_z * inv_edge_z));
            acc.add(first * r_exp_z);
            acc.add(lookup_step(
                r_exp_omega,
                r_exp_z,
                table_exp_z,
                query_exp_z,
                m_exp_z,
                q_tbl_exp_z,
                q_qry_exp_z,
                beta_exp_out));
            acc.add(last * r_exp_z);
            acc.add(table_exp_z - (t_exp_x_z + eta_exp_out * t_exp_y_z));
            acc.add(query_exp_z - (delta_z + eta_exp_out * u_z));
            acc.add(zero_on_unselected(q_tbl_exp_z, table_exp_z));
            acc.add(zero_on_unselected(q_tbl_exp_z, m_exp_z));
            acc.add(zero_on_unselected(q_qry_exp_z, query_exp_z));

            acc.add(first * (psq_z - q_edge * w_z));
            acc.add(psq_step(psq_omega, psq_z, w_omega, q_edge_next, q_new_next));
            acc.add(q_end * (psq_z - t_edge_z));
            acc.add(first * r_t_z);
            acc.add(route_edge_step(r_t_omega, r_t_z, query_t_z, beta_t_out, q_edge));
            acc.add(last * (r_t_z - witness_scalars.at(output_witness_scalar_name("t", head_index, legacy_single_output))));
            acc.add(query_t_z - (dst_z + eta_t_out * t_edge_z));
            acc.add(zero_on_unselected(q_edge, query_t_z));

            acc.add(first * r_dst_z);
            acc.add(route_edge_step(r_dst_omega, r_dst_z, query_dst_z, beta_dst_out, q_edge));
            acc.add(last * (r_dst_z - witness_scalars.at(output_witness_scalar_name("dst", head_index, legacy_single_output))));
            acc.add(query_dst_z
                - (dst_z
                    + eta_dst_out * e_dst_edge_z
                    + eta_dst_out_sq * m_edge_z
                    + eta_dst_out_cu * sum_edge_z
                    + eta_dst_out_4 * inv_edge_z
                    + eta_dst_out_5 * y_star_edge_z));
            acc.add(zero_on_unselected(q_edge, query_dst_z));
        }

        return acc.value() / zero_eval;
    }

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
    if (context.model.has_real_multihead) {
        const auto omega_z = z * context.domains.n->omega;
        const auto first = lagrange_first(*context.domains.n, z);
        const auto last = lagrange_last(*context.domains.n, z);
        const auto zero_eval = context.domains.n->zero_polynomial_eval(z);
        const auto q_n = eval("P_Q_N", z);
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            const auto prefix = "P_h" + std::to_string(head_index) + "_";
            const auto beta_src = challenges.at("beta_src_h" + std::to_string(head_index));
            const auto beta_dst = challenges.at("beta_dst_h" + std::to_string(head_index));
            const auto eta_src = challenges.at("eta_src_h" + std::to_string(head_index));
            const auto eta_dst = challenges.at("eta_dst_h" + std::to_string(head_index));
            const auto beta_t = challenges.at("beta_t_h" + std::to_string(head_index));
            const auto eta_t = challenges.at("eta_t_h" + std::to_string(head_index));
            const auto lambda_psq = challenges.at("lambda_psq_h" + std::to_string(head_index));

            acc.add(first * eval(prefix + "R_src_node", z));
            acc.add(route_node_step(eval(prefix + "R_src_node", omega_z), eval(prefix + "R_src_node", z), eval(prefix + "Table_src", z), beta_src, q_n, eval(prefix + "m_src", z)));
            acc.add(last * (eval(prefix + "R_src_node", z) - witness_scalars.at("S_src_h" + std::to_string(head_index))));
            acc.add(first * eval(prefix + "R_dst_node", z));
            acc.add(route_node_step(eval(prefix + "R_dst_node", omega_z), eval(prefix + "R_dst_node", z), eval(prefix + "Table_dst", z), beta_dst, q_n, eval(prefix + "m_dst", z)));
            acc.add(last * (eval(prefix + "R_dst_node", z) - witness_scalars.at("S_dst_h" + std::to_string(head_index))));
            acc.add(first * eval(prefix + "R_t_node", z));
            acc.add(route_node_step(eval(prefix + "R_t_node", omega_z), eval(prefix + "R_t_node", z), eval(prefix + "Table_t", z), beta_t, q_n, eval(prefix + "m_t", z)));
            acc.add(last * (eval(prefix + "R_t_node", z) - witness_scalars.at("S_t_h" + std::to_string(head_index))));
            acc.add(q_n * (eval(prefix + "Sum", z) * eval(prefix + "inv", z) - FieldElement::one()));
            acc.add(q_n * (eval(prefix + "T_psq", z) - eval(prefix + "Sum", z) - lambda_psq * eval(prefix + "H_agg_pre_star", z)));
            acc.add(eval(prefix + "Table_src", z)
                - (eval("P_I", z) + eta_src * eval(prefix + "E_src", z) + eta_src.pow(2) * eval(prefix + "H_star", z)));
            acc.add(eval(prefix + "Table_dst", z)
                - (eval("P_I", z)
                    + eta_dst * eval(prefix + "E_dst", z)
                    + eta_dst.pow(2) * eval(prefix + "M", z)
                    + eta_dst.pow(3) * eval(prefix + "Sum", z)
                    + eta_dst.pow(4) * eval(prefix + "inv", z)
                    + eta_dst.pow(5) * eval(prefix + "H_agg_star", z)));
            acc.add(eval(prefix + "Table_t", z) - (eval("P_I", z) + eta_t * eval(prefix + "T_psq", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "E_src", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "E_dst", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "M", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Sum", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "inv", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "H_star", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "H_agg_pre_star", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "H_agg_star", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Table_src", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "m_src", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Table_dst", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "m_dst", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "T_psq", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Table_t", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "m_t", z)));
        }

        const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
            const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
            const auto beta_src_out = challenges.at(output_challenge_name("beta_src_out", head_index, legacy_single_output));
            const auto beta_dst_out = challenges.at(output_challenge_name("beta_dst_out", head_index, legacy_single_output));
            const auto eta_src_out = challenges.at(output_challenge_name("eta_src_out", head_index, legacy_single_output));
            const auto eta_dst_out = challenges.at(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
            const auto beta_t_out = challenges.at(output_challenge_name("beta_t_out", head_index, legacy_single_output));
            const auto eta_t_out = challenges.at(output_challenge_name("eta_t_out", head_index, legacy_single_output));
            const auto lambda_out = challenges.at(output_challenge_name("lambda_out", head_index, legacy_single_output));

            acc.add(first * eval(prefix + "R_src_node", z));
            acc.add(route_node_step(eval(prefix + "R_src_node", omega_z), eval(prefix + "R_src_node", z), eval(prefix + "Table_src", z), beta_src_out, q_n, eval(prefix + "m_src", z)));
            acc.add(last * (eval(prefix + "R_src_node", z) - witness_scalars.at(output_witness_scalar_name("src", head_index, legacy_single_output))));
            acc.add(first * eval(prefix + "R_dst_node", z));
            acc.add(route_node_step(eval(prefix + "R_dst_node", omega_z), eval(prefix + "R_dst_node", z), eval(prefix + "Table_dst", z), beta_dst_out, q_n, eval(prefix + "m_dst", z)));
            acc.add(last * (eval(prefix + "R_dst_node", z) - witness_scalars.at(output_witness_scalar_name("dst", head_index, legacy_single_output))));
            acc.add(first * eval(prefix + "R_t_node", z));
            acc.add(route_node_step(eval(prefix + "R_t_node", omega_z), eval(prefix + "R_t_node", z), eval(prefix + "Table_t", z), beta_t_out, q_n, eval(prefix + "m_t", z)));
            acc.add(last * (eval(prefix + "R_t_node", z) - witness_scalars.at(output_witness_scalar_name("t", head_index, legacy_single_output))));
            acc.add(q_n * (eval(prefix + "Sum", z) * eval(prefix + "inv", z) - FieldElement::one()));
            acc.add(q_n * (eval(prefix + "T", z) - eval(prefix + "Sum", z) - lambda_out * eval(prefix + "Y_star", z)));
            acc.add(eval(prefix + "Table_src", z) - (eval("P_I", z) + eta_src_out * eval(prefix + "E_src", z)));
            acc.add(eval(prefix + "Table_dst", z)
                - (eval("P_I", z)
                    + eta_dst_out * eval(prefix + "E_dst", z)
                    + eta_dst_out.pow(2) * eval(prefix + "M", z)
                    + eta_dst_out.pow(3) * eval(prefix + "Sum", z)
                    + eta_dst_out.pow(4) * eval(prefix + "inv", z)
                    + eta_dst_out.pow(5) * eval(prefix + "Y_star", z)));
            acc.add(eval(prefix + "Table_t", z) - (eval("P_I", z) + eta_t_out * eval(prefix + "T", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "E_src", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "E_dst", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "M", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Sum", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "inv", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Y_prime_star", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Y_star", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "T", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Table_src", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "m_src", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Table_dst", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "m_dst", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "Table_t", z)));
            acc.add(zero_on_unselected(q_n, eval(prefix + "m_t", z)));
        }

        return acc.value() / zero_eval;
    }

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
    if (context.model.has_real_multihead) {
        const auto omega_z = z * context.domains.in->omega;
        const auto first = lagrange_first(*context.domains.in, z);
        const auto last = lagrange_last(*context.domains.in, z);
        const auto zero_eval = context.domains.in->zero_polynomial_eval(z);
        const auto q_valid = eval("P_Q_proj_valid", z);
        const auto q_invalid = FieldElement::one() - q_valid;
        QuotientAccumulator acc(challenges.at("alpha_quot"));
        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            const auto prefix = "P_h" + std::to_string(head_index) + "_";
            acc.add(first * eval(prefix + "Acc_proj", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_proj", omega_z), eval(prefix + "Acc_proj", z), eval(prefix + "a_proj", z), eval(prefix + "b_proj", z)));
            acc.add(q_invalid * (eval(prefix + "Acc_proj", omega_z) - eval(prefix + "Acc_proj", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_proj", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_proj", z)));
            acc.add(last * (eval(prefix + "Acc_proj", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_proj")));
        }
        return acc.value() / zero_eval;
    }

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
    if (context.model.has_real_multihead) {
        const auto omega_z = z * context.domains.d->omega;
        const auto first = lagrange_first(*context.domains.d, z);
        const auto last = lagrange_last(*context.domains.d, z);
        const auto zero_eval = context.domains.d->zero_polynomial_eval(z);
        const auto q_valid = eval("P_Q_d_valid", z);
        const auto q_invalid = FieldElement::one() - q_valid;
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            const auto prefix = "P_h" + std::to_string(head_index) + "_";
            acc.add(first * eval(prefix + "Acc_src", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_src", omega_z), eval(prefix + "Acc_src", z), eval(prefix + "a_src", z), eval(prefix + "b_src", z)));
            acc.add(q_invalid * (eval(prefix + "Acc_src", omega_z) - eval(prefix + "Acc_src", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_src", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_src", z)));
            acc.add(last * (eval(prefix + "Acc_src", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_src")));

            acc.add(first * eval(prefix + "Acc_dst", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_dst", omega_z), eval(prefix + "Acc_dst", z), eval(prefix + "a_dst", z), eval(prefix + "b_dst", z)));
            acc.add(q_invalid * (eval(prefix + "Acc_dst", omega_z) - eval(prefix + "Acc_dst", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_dst", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_dst", z)));
            acc.add(last * (eval(prefix + "Acc_dst", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_dst")));

            acc.add(first * eval(prefix + "Acc_star", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_star", omega_z), eval(prefix + "Acc_star", z), eval(prefix + "a_star", z), eval(prefix + "b_star", z)));
            acc.add(q_invalid * (eval(prefix + "Acc_star", omega_z) - eval(prefix + "Acc_star", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_star", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_star", z)));
            acc.add(last * (eval(prefix + "Acc_star", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_star")));

            acc.add(first * eval(prefix + "Acc_agg_pre", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_agg_pre", omega_z), eval(prefix + "Acc_agg_pre", z), eval(prefix + "a_agg_pre", z), eval(prefix + "b_agg_pre", z)));
            acc.add(q_invalid * (eval(prefix + "Acc_agg_pre", omega_z) - eval(prefix + "Acc_agg_pre", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_agg_pre", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_agg_pre", z)));
            acc.add(last * (eval(prefix + "Acc_agg_pre", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_agg_pre")));

            acc.add(first * eval(prefix + "Acc_agg", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_agg", omega_z), eval(prefix + "Acc_agg", z), eval(prefix + "a_agg", z), eval(prefix + "b_agg", z)));
            acc.add(q_invalid * (eval(prefix + "Acc_agg", omega_z) - eval(prefix + "Acc_agg", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_agg", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_agg", z)));
            acc.add(last * (eval(prefix + "Acc_agg", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_agg")));
        }

        return acc.value() / zero_eval;
    }

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

FieldElement evaluate_t_cat(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto omega_z = z * context.domains.cat->omega;
    const auto first = lagrange_first(*context.domains.cat, z);
    const auto last = lagrange_last(*context.domains.cat, z);
    const auto zero_eval = context.domains.cat->zero_polynomial_eval(z);
    const auto q_valid = eval("P_Q_cat_valid", z);
    const auto q_invalid = FieldElement::one() - q_valid;
    QuotientAccumulator acc(challenges.at("alpha_quot"));

    for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index) {
        const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
        const auto prefix = hidden_layer_cat_prefix(layer_index, is_final_layer);
        const auto mu_name = is_final_layer ? std::string("mu_cat") : "mu_cat_l" + std::to_string(layer_index);
        acc.add(first * eval(prefix + "_Acc", z));
        acc.add(q_valid * acc_step(eval(prefix + "_Acc", omega_z), eval(prefix + "_Acc", z), eval(prefix + "_a", z), eval(prefix + "_b", z)));
        acc.add(q_invalid * (eval(prefix + "_Acc", omega_z) - eval(prefix + "_Acc", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "_a", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "_b", z)));
        acc.add(last * (eval(prefix + "_Acc", z) - external_evaluations.at(mu_name)));
    }

    const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
    for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
        const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
        const auto mu_name =
            legacy_single_output ? std::string("mu_out_proj") : output_external_eval_name("proj", head_index, false);
        acc.add(first * eval(prefix + "Acc_proj", z));
        acc.add(q_valid * acc_step(eval(prefix + "Acc_proj", omega_z), eval(prefix + "Acc_proj", z), eval(prefix + "a_proj", z), eval(prefix + "b_proj", z)));
        acc.add(q_invalid * (eval(prefix + "Acc_proj", omega_z) - eval(prefix + "Acc_proj", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "a_proj", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "b_proj", z)));
        acc.add(last * (eval(prefix + "Acc_proj", z) - external_evaluations.at(mu_name)));
    }

    return acc.value() / zero_eval;
}

FieldElement evaluate_t_c(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const EvalFn& eval,
    const FieldElement& z) {
    const auto omega_z = z * context.domains.c->omega;
    const auto first = lagrange_first(*context.domains.c, z);
    const auto last = lagrange_last(*context.domains.c, z);
    const auto zero_eval = context.domains.c->zero_polynomial_eval(z);
    const auto q_valid = eval("P_Q_C_valid", z);
    const auto q_invalid = FieldElement::one() - q_valid;
    QuotientAccumulator acc(challenges.at("alpha_quot"));

    const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
    for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
        const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
        const auto mu_src =
            legacy_single_output ? std::string("mu_out_src") : output_external_eval_name("src", head_index, false);
        const auto mu_dst =
            legacy_single_output ? std::string("mu_out_dst") : output_external_eval_name("dst", head_index, false);
        const auto mu_star =
            legacy_single_output ? std::string("mu_out_star") : output_external_eval_name("star", head_index, false);
        const auto mu_y_lin =
            legacy_single_output ? std::string("mu_Y_lin") : output_external_eval_name("y_lin", head_index, false);
        const auto mu_y =
            legacy_single_output ? std::string("mu_out") : output_external_eval_name("y", head_index, false);

        acc.add(first * eval(prefix + "Acc_src", z));
        acc.add(q_valid * acc_step(eval(prefix + "Acc_src", omega_z), eval(prefix + "Acc_src", z), eval(prefix + "a_src", z), eval(prefix + "b_src", z)));
        acc.add(q_invalid * (eval(prefix + "Acc_src", omega_z) - eval(prefix + "Acc_src", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "a_src", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "b_src", z)));
        acc.add(last * (eval(prefix + "Acc_src", z) - external_evaluations.at(mu_src)));

        acc.add(first * eval(prefix + "Acc_dst", z));
        acc.add(q_valid * acc_step(eval(prefix + "Acc_dst", omega_z), eval(prefix + "Acc_dst", z), eval(prefix + "a_dst", z), eval(prefix + "b_dst", z)));
        acc.add(q_invalid * (eval(prefix + "Acc_dst", omega_z) - eval(prefix + "Acc_dst", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "a_dst", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "b_dst", z)));
        acc.add(last * (eval(prefix + "Acc_dst", z) - external_evaluations.at(mu_dst)));

        acc.add(first * eval(prefix + "Acc_y", z));
        acc.add(q_valid * acc_step(eval(prefix + "Acc_y", omega_z), eval(prefix + "Acc_y", z), eval(prefix + "a_y", z), eval(prefix + "b_y", z)));
        acc.add(q_invalid * (eval(prefix + "Acc_y", omega_z) - eval(prefix + "Acc_y", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "a_y", z)));
        acc.add(zero_on_unselected(q_valid, eval(prefix + "b_y", z)));
        acc.add(last * (eval(prefix + "Acc_y", z) - external_evaluations.at(mu_star)));

        acc.add(first * (external_evaluations.at(mu_y) - external_evaluations.at(mu_y_lin)
            - bias_fold_vector(
                context.model.output_layer.heads[head_index].output_bias_fp,
                context.local.num_nodes,
                challenges.at("y_out"))));
    }

    if (!legacy_single_output) {
        FieldElement summed_y_lin = FieldElement::zero();
        FieldElement summed_y = FieldElement::zero();
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
            summed_y_lin += external_evaluations.at(output_external_eval_name("y_lin", head_index, false));
            summed_y += external_evaluations.at(output_external_eval_name("y", head_index, false));
        }
        const auto k_out = FieldElement(context.model.output_layer.heads.size());
        acc.add(first * (k_out * external_evaluations.at("mu_Y_lin") - summed_y_lin));
        acc.add(first * (k_out * external_evaluations.at("mu_out") - summed_y));
    }

    return acc.value() / zero_eval;
}

}  // namespace gatzk::protocol
