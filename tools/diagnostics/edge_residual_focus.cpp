#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/util/config.hpp"

using gatzk::algebra::FieldElement;

namespace {

std::filesystem::path repo_root() {
    auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / "configs" / "cora_secure.cfg")) {
        return cwd;
    }
    if (std::filesystem::exists(cwd.parent_path() / "configs" / "cora_secure.cfg")) {
        return cwd.parent_path();
    }
    return cwd;
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

FieldElement route_edge_step(
    const FieldElement& next_value,
    const FieldElement& current_value,
    const FieldElement& query_value,
    const FieldElement& beta,
    const FieldElement& q_edge_valid) {
    return (next_value - current_value) * (query_value + beta) - q_edge_valid;
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

FieldElement route_node_step(
    const FieldElement& next_value,
    const FieldElement& current_value,
    const FieldElement& table_value,
    const FieldElement& beta,
    const FieldElement& q_valid,
    const FieldElement& multiplicity) {
    return (next_value - current_value) * (table_value + beta) - q_valid * multiplicity;
}

FieldElement zero_on_unselected(const FieldElement& selector, const FieldElement& value) {
    return (FieldElement::one() - selector) * value;
}

FieldElement acc_step(
    const FieldElement& next_value,
    const FieldElement& current_value,
    const FieldElement& a_value,
    const FieldElement& b_value) {
    return next_value - current_value - a_value * b_value;
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
    for (const auto value : bias) {
        column_fold += quantize_bias_value(value) * column_power;
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

[[noreturn]] void fail(const std::string& domain, std::size_t row, const std::string& term, const FieldElement& value) {
    std::cerr << "first nonzero " << domain << " row=" << row << " term=" << term
              << " value=" << value.to_string() << "\n";
    throw std::runtime_error("residual failure");
}

void check(const std::string& domain, std::size_t row, const std::string& term, const FieldElement& value) {
    if (value != FieldElement::zero()) {
        fail(domain, row, term, value);
    }
}

}  // namespace

int main() {
    try {
        auto config = gatzk::util::load_config((repo_root() / "configs/cora_secure.cfg").string());
        auto context = gatzk::protocol::build_context(config, nullptr);
        auto trace = gatzk::protocol::build_trace(context, nullptr);

        auto at = [&](const std::string& label, std::size_t row) -> FieldElement {
            if (auto it = trace.polynomials.find(label); it != trace.polynomials.end()) {
                return row < it->second.size() ? it->second.value_at(row) : FieldElement::zero();
            }
            if (auto it = context.public_polynomials.find(label); it != context.public_polynomials.end()) {
                return row < it->second.size() ? it->second.value_at(row) : FieldElement::zero();
            }
            throw std::runtime_error("missing label " + label);
        };
        auto eval_trace_poly = [&](const std::string& label, const FieldElement& point) -> FieldElement {
            if (auto it = trace.polynomials.find(label); it != trace.polynomials.end()) {
                return it->second.evaluate(point);
            }
            throw std::runtime_error("missing trace polynomial " + label);
        };

        for (std::size_t row = 0; row < context.domains.edge->size; ++row) {
            const auto next = (row + 1) % context.domains.edge->size;
            const auto first = row == 0 ? FieldElement::one() : FieldElement::zero();
            const auto last = row + 1 == context.domains.edge->size ? FieldElement::one() : FieldElement::zero();
            const auto not_last = FieldElement::one() - last;
            const auto q_edge = at("P_Q_edge_valid", row);
            const auto q_edge_next = at("P_Q_edge_valid", next);
            const auto q_new_next = at("P_Q_new_edge", next);
            const auto q_end = at("P_Q_end_edge", row);
            const auto dst = at("P_dst", row);
            const auto q_tbl_elu = at("P_Q_tbl_ELU", row);
            const auto q_qry_elu = at("P_Q_qry_ELU", row);
            const auto q_tbl_l = at("P_Q_tbl_L", row);
            const auto q_qry_l = at("P_Q_qry_L", row);
            const auto q_tbl_r = at("P_Q_tbl_R", row);
            const auto q_qry_r = at("P_Q_qry_R", row);
            const auto q_tbl_exp = at("P_Q_tbl_exp", row);
            const auto q_qry_exp = at("P_Q_qry_exp", row);
            const auto t_elu_x = at("P_T_ELU_x", row);
            const auto t_elu_y = at("P_T_ELU_y", row);
            const auto src = at("P_src", row);
            const auto t_l_x = at("P_T_L_x", row);
            const auto t_l_y = at("P_T_L_y", row);
            const auto t_range = at("P_T_range", row);
            const auto t_exp_x = at("P_T_exp_x", row);
            const auto t_exp_y = at("P_T_exp_y", row);

            for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head) {
                const auto prefix = "P_h" + std::to_string(head) + "_";
                const auto beta_elu = trace.challenges.at("beta_ELU_h" + std::to_string(head));
                const auto eta_elu = trace.challenges.at("eta_ELU_h" + std::to_string(head));
                const auto beta_t = trace.challenges.at("beta_t_h" + std::to_string(head));
                const auto eta_t = trace.challenges.at("eta_t_h" + std::to_string(head));
                const auto beta_dst = trace.challenges.at("beta_dst_h" + std::to_string(head));
                const auto eta_dst = trace.challenges.at("eta_dst_h" + std::to_string(head));
                const auto eta_dst_2 = eta_dst * eta_dst;
                const auto eta_dst_3 = eta_dst_2 * eta_dst;
                const auto eta_dst_4 = eta_dst_3 * eta_dst;
                const auto eta_dst_5 = eta_dst_4 * eta_dst;
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };

                check("edge", row, prefix + "psq first", first * (v("PSQ") - q_edge * v("w_psq")));
                check("edge", row, prefix + "psq step", psq_step(vn("PSQ"), v("PSQ"), vn("w_psq"), q_edge_next, q_new_next));
                check("edge", row, prefix + "psq end", q_end * (v("PSQ") - v("T_psq_edge")));
                check("edge", row, prefix + "t first", first * v("R_t"));
                check("edge", row, prefix + "t step", not_last * route_edge_step(vn("R_t"), v("R_t"), v("Query_t"), beta_t, q_edge));
                check("edge", row, prefix + "t last", last * (v("R_t") - trace.witness_scalars.at("S_t_h" + std::to_string(head))));
                check("edge", row, prefix + "t query", v("Query_t") - (dst + eta_t * v("T_psq_edge")));
                check("edge", row, prefix + "t zero", zero_on_unselected(q_edge, v("Query_t")));
                check("edge", row, prefix + "ELU first", first * v("R_ELU"));
                check("edge", row, prefix + "ELU step", lookup_step(vn("R_ELU"), v("R_ELU"), v("Table_ELU"), v("Query_ELU"), v("m_ELU"), q_tbl_elu, q_qry_elu, beta_elu));
                check("edge", row, prefix + "ELU last", last * v("R_ELU"));
                check("edge", row, prefix + "ELU table", v("Table_ELU") - (t_elu_x + eta_elu * t_elu_y));
                check("edge", row, prefix + "ELU query", v("Query_ELU") - (v("H_agg_pre_flat") + eta_elu * v("H_agg_flat")));
                check("edge", row, prefix + "ELU table zero", zero_on_unselected(q_tbl_elu, v("Table_ELU")));
                check("edge", row, prefix + "ELU m zero", zero_on_unselected(q_tbl_elu, v("m_ELU")));
                check("edge", row, prefix + "ELU query zero", zero_on_unselected(q_qry_elu, v("Query_ELU")));
                check("edge", row, prefix + "dst first", first * v("R_dst"));
                check("edge", row, prefix + "dst step", not_last * route_edge_step(vn("R_dst"), v("R_dst"), v("Query_dst"), beta_dst, q_edge));
                check("edge", row, prefix + "dst last", last * (v("R_dst") - trace.witness_scalars.at("S_dst_h" + std::to_string(head))));
                check("edge", row, prefix + "dst bind",
                    v("Query_dst")
                        - (dst + eta_dst * v("E_dst_edge") + eta_dst_2 * v("M_edge") + eta_dst_3 * v("Sum_edge")
                            + eta_dst_4 * v("inv_edge") + eta_dst_5 * v("H_agg_star_edge")));
                check("edge", row, prefix + "dst zero", zero_on_unselected(q_edge, v("Query_dst")));
            }

            const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head) {
                const auto prefix = gatzk::protocol::output_head_prefix(head, legacy_single_output) + "_";
                const auto beta_src = trace.challenges.at(gatzk::protocol::output_challenge_name("beta_src_out", head, legacy_single_output));
                const auto beta_dst = trace.challenges.at(gatzk::protocol::output_challenge_name("beta_dst_out", head, legacy_single_output));
                const auto eta_src = trace.challenges.at(gatzk::protocol::output_challenge_name("eta_src_out", head, legacy_single_output));
                const auto eta_dst = trace.challenges.at(gatzk::protocol::output_challenge_name("eta_dst_out", head, legacy_single_output));
                const auto beta_l = trace.challenges.at(gatzk::protocol::output_challenge_name("beta_L_out", head, legacy_single_output));
                const auto eta_l = trace.challenges.at(gatzk::protocol::output_challenge_name("eta_L_out", head, legacy_single_output));
                const auto beta_r = trace.challenges.at(gatzk::protocol::output_challenge_name("beta_R_out", head, legacy_single_output));
                const auto beta_exp = trace.challenges.at(gatzk::protocol::output_challenge_name("beta_exp_out", head, legacy_single_output));
                const auto eta_exp = trace.challenges.at(gatzk::protocol::output_challenge_name("eta_exp_out", head, legacy_single_output));
                const auto beta_t = trace.challenges.at(gatzk::protocol::output_challenge_name("beta_t_out", head, legacy_single_output));
                const auto eta_t = trace.challenges.at(gatzk::protocol::output_challenge_name("eta_t_out", head, legacy_single_output));
                const auto eta_dst_2 = eta_dst * eta_dst;
                const auto eta_dst_3 = eta_dst_2 * eta_dst;
                const auto eta_dst_4 = eta_dst_3 * eta_dst;
                const auto eta_dst_5 = eta_dst_4 * eta_dst;
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };

                check("edge", row, prefix + "src first", first * v("R_src"));
                check("edge", row, prefix + "src step", not_last * route_edge_step(vn("R_src"), v("R_src"), v("Query_src"), beta_src, q_edge));
                check("edge", row, prefix + "src last", last * (v("R_src") - trace.witness_scalars.at(gatzk::protocol::output_witness_scalar_name("src", head, legacy_single_output))));
                check("edge", row, prefix + "src bind", v("Query_src") - (src + eta_src * v("E_src_edge")));
                check("edge", row, prefix + "src zero", zero_on_unselected(q_edge, v("Query_src")));
                check("edge", row, prefix + "s bind", q_edge * (v("S") - v("E_src_edge") - v("E_dst_edge")));
                check("edge", row, prefix + "L first", first * v("R_L"));
                check("edge", row, prefix + "L step", lookup_step(vn("R_L"), v("R_L"), v("Table_L"), v("Query_L"), v("m_L"), q_tbl_l, q_qry_l, beta_l));
                check("edge", row, prefix + "L last", last * v("R_L"));
                check("edge", row, prefix + "L table", v("Table_L") - (t_l_x + eta_l * t_l_y));
                check("edge", row, prefix + "L query", v("Query_L") - (v("S") + eta_l * v("Z")));
                check("edge", row, prefix + "L table zero", zero_on_unselected(q_tbl_l, v("Table_L")));
                check("edge", row, prefix + "L m zero", zero_on_unselected(q_tbl_l, v("m_L")));
                check("edge", row, prefix + "L query zero", zero_on_unselected(q_qry_l, v("Query_L")));
                check("edge", row, prefix + "delta bind", q_edge * (v("Delta") - v("M_edge") + v("Z")));
                check("edge", row, prefix + "smax bool", v("s_max") * (v("s_max") - FieldElement::one()));
                check("edge", row, prefix + "smax delta", v("s_max") * v("Delta"));
                check("edge", row, prefix + "smax zero", zero_on_unselected(q_edge, v("s_max")));
                check("edge", row, prefix + "cmax first", first * (v("C_max") - v("s_max")));
                check("edge", row, prefix + "cmax step",
                    q_edge_next
                            * (vn("C_max") - q_new_next * vn("s_max")
                                - (FieldElement::one() - q_new_next) * (v("C_max") + vn("s_max")))
                        + (FieldElement::one() - q_edge_next) * (vn("C_max") - v("C_max")));
                check("edge", row, prefix + "cmax end", q_end * (v("C_max") - FieldElement::one()));
                check("edge", row, prefix + "R first", first * v("R_R"));
                check("edge", row, prefix + "R step", lookup_step(vn("R_R"), v("R_R"), v("Table_R"), v("Query_R"), v("m_R"), q_tbl_r, q_qry_r, beta_r));
                check("edge", row, prefix + "R last", last * v("R_R"));
                check("edge", row, prefix + "R table", v("Table_R") - t_range);
                check("edge", row, prefix + "R query", v("Query_R") - v("Delta"));
                check("edge", row, prefix + "R table zero", zero_on_unselected(q_tbl_r, v("Table_R")));
                check("edge", row, prefix + "R m zero", zero_on_unselected(q_tbl_r, v("m_R")));
                check("edge", row, prefix + "R query zero", zero_on_unselected(q_qry_r, v("Query_R")));
                check("edge", row, prefix + "alpha bind", q_edge * (v("alpha") - v("U") * v("inv_edge")));
                check("edge", row, prefix + "exp first", first * v("R_exp"));
                check("edge", row, prefix + "exp step", lookup_step(vn("R_exp"), v("R_exp"), v("Table_exp"), v("Query_exp"), v("m_exp"), q_tbl_exp, q_qry_exp, beta_exp));
                check("edge", row, prefix + "exp last", last * v("R_exp"));
                check("edge", row, prefix + "exp table", v("Table_exp") - (t_exp_x + eta_exp * t_exp_y));
                check("edge", row, prefix + "exp query", v("Query_exp") - (v("Delta") + eta_exp * v("U")));
                check("edge", row, prefix + "exp table zero", zero_on_unselected(q_tbl_exp, v("Table_exp")));
                check("edge", row, prefix + "exp m zero", zero_on_unselected(q_tbl_exp, v("m_exp")));
                check("edge", row, prefix + "exp query zero", zero_on_unselected(q_qry_exp, v("Query_exp")));
                check("edge", row, prefix + "psq first", first * (v("PSQ") - q_edge * v("w")));
                check("edge", row, prefix + "psq step", psq_step(vn("PSQ"), v("PSQ"), vn("w"), q_edge_next, q_new_next));
                check("edge", row, prefix + "psq end", q_end * (v("PSQ") - v("T_edge")));
                check("edge", row, prefix + "t first", first * v("R_t"));
                check("edge", row, prefix + "t step", not_last * route_edge_step(vn("R_t"), v("R_t"), v("Query_t"), beta_t, q_edge));
                check("edge", row, prefix + "t last", last * (v("R_t") - trace.witness_scalars.at(gatzk::protocol::output_witness_scalar_name("t", head, legacy_single_output))));
                check("edge", row, prefix + "t query", v("Query_t") - (dst + eta_t * v("T_edge")));
                check("edge", row, prefix + "t zero", zero_on_unselected(q_edge, v("Query_t")));
                check("edge", row, prefix + "dst first", first * v("R_dst"));
                check("edge", row, prefix + "dst step", not_last * route_edge_step(vn("R_dst"), v("R_dst"), v("Query_dst"), beta_dst, q_edge));
                check("edge", row, prefix + "dst last", last * (v("R_dst") - trace.witness_scalars.at(gatzk::protocol::output_witness_scalar_name("dst", head, legacy_single_output))));
                check("edge", row, prefix + "dst bind",
                    v("Query_dst")
                        - (dst + eta_dst * v("E_dst_edge") + eta_dst_2 * v("M_edge") + eta_dst_3 * v("Sum_edge")
                            + eta_dst_4 * v("inv_edge") + eta_dst_5 * v("Y_star_edge")));
                check("edge", row, prefix + "dst zero", zero_on_unselected(q_edge, v("Query_dst")));
            }
        }

        for (std::size_t row = 0; row < context.domains.n->size; ++row) {
            const auto next = (row + 1) % context.domains.n->size;
            const auto first = row == 0 ? FieldElement::one() : FieldElement::zero();
            const auto last = row + 1 == context.domains.n->size ? FieldElement::one() : FieldElement::zero();
            const auto not_last = FieldElement::one() - last;
            const auto q_n = at("P_Q_N", row);
            const auto node_id = at("P_I", row);
            for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head) {
                const auto prefix = "P_h" + std::to_string(head) + "_";
                const auto beta_src = trace.challenges.at("beta_src_h" + std::to_string(head));
                const auto beta_dst = trace.challenges.at("beta_dst_h" + std::to_string(head));
                const auto beta_t = trace.challenges.at("beta_t_h" + std::to_string(head));
                const auto eta_src = trace.challenges.at("eta_src_h" + std::to_string(head));
                const auto eta_dst = trace.challenges.at("eta_dst_h" + std::to_string(head));
                const auto eta_t = trace.challenges.at("eta_t_h" + std::to_string(head));
                const auto lambda_psq = trace.challenges.at("lambda_psq_h" + std::to_string(head));
                const auto eta_dst_2 = eta_dst * eta_dst;
                const auto eta_dst_3 = eta_dst_2 * eta_dst;
                const auto eta_dst_4 = eta_dst_3 * eta_dst;
                const auto eta_dst_5 = eta_dst_4 * eta_dst;
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };

                check("N", row, prefix + "src node first", first * v("R_src_node"));
                check("N", row, prefix + "src node step", not_last * route_node_step(vn("R_src_node"), v("R_src_node"), v("Table_src"), beta_src, q_n, v("m_src")));
                check("N", row, prefix + "src node last", last * (v("R_src_node") - trace.witness_scalars.at("S_src_h" + std::to_string(head))));
                check("N", row, prefix + "dst node first", first * v("R_dst_node"));
                check("N", row, prefix + "dst node step", not_last * route_node_step(vn("R_dst_node"), v("R_dst_node"), v("Table_dst"), beta_dst, q_n, v("m_dst")));
                check("N", row, prefix + "dst node last", last * (v("R_dst_node") - trace.witness_scalars.at("S_dst_h" + std::to_string(head))));
                check("N", row, prefix + "t node first", first * v("R_t_node"));
                check("N", row, prefix + "t node step", not_last * route_node_step(vn("R_t_node"), v("R_t_node"), v("Table_t"), beta_t, q_n, v("m_t")));
                check("N", row, prefix + "t node last", last * (v("R_t_node") - trace.witness_scalars.at("S_t_h" + std::to_string(head))));
                check("N", row, prefix + "sum inv", q_n * (v("Sum") * v("inv") - FieldElement::one()));
                check("N", row, prefix + "T psq", q_n * (v("T_psq") - v("Sum") - lambda_psq * v("H_agg_pre_star")));
                check("N", row, prefix + "Table src", v("Table_src") - (node_id + eta_src * v("E_src") + eta_src * eta_src * v("H_star")));
                check("N", row, prefix + "Table dst",
                    v("Table_dst")
                        - (node_id + eta_dst * v("E_dst") + eta_dst_2 * v("M") + eta_dst_3 * v("Sum")
                            + eta_dst_4 * v("inv") + eta_dst_5 * v("H_agg_star")));
                check("N", row, prefix + "Table t", v("Table_t") - (node_id + eta_t * v("T_psq")));
                check("N", row, prefix + "Sum zero", zero_on_unselected(q_n, v("Sum")));
                check("N", row, prefix + "inv zero", zero_on_unselected(q_n, v("inv")));
                check("N", row, prefix + "H_agg_pre_star zero", zero_on_unselected(q_n, v("H_agg_pre_star")));
                check("N", row, prefix + "H_agg_star zero", zero_on_unselected(q_n, v("H_agg_star")));
                check("N", row, prefix + "T_psq zero", zero_on_unselected(q_n, v("T_psq")));
            }
        }

        for (std::size_t row = 0; row < context.domains.in->size; ++row) {
            const auto next = (row + 1) % context.domains.in->size;
            const auto first = row == 0 ? FieldElement::one() : FieldElement::zero();
            const auto last = row + 1 == context.domains.in->size ? FieldElement::one() : FieldElement::zero();
            const auto not_last = FieldElement::one() - last;
            const auto q_valid = at("P_Q_proj_valid", row);
            const auto q_invalid = FieldElement::one() - q_valid;
            for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head) {
                const auto prefix = "P_h" + std::to_string(head) + "_";
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };
                check("in", row, prefix + "proj first", first * v("Acc_proj"));
                check("in", row, prefix + "proj step",
                    q_valid * (vn("Acc_proj") - v("Acc_proj") - v("a_proj") * v("b_proj")));
                check("in", row, prefix + "proj pad", not_last * q_invalid * (vn("Acc_proj") - v("Acc_proj")));
                check("in", row, prefix + "a zero", zero_on_unselected(q_valid, v("a_proj")));
                check("in", row, prefix + "b zero", zero_on_unselected(q_valid, v("b_proj")));
                check("in", row, prefix + "proj last",
                    last * (v("Acc_proj") - trace.external_evaluations.at("mu_h" + std::to_string(head) + "_proj")));
            }
        }

        for (std::size_t row = 0; row < context.domains.d->size; ++row) {
            const auto next = (row + 1) % context.domains.d->size;
            const auto first = row == 0 ? FieldElement::one() : FieldElement::zero();
            const auto last = row + 1 == context.domains.d->size ? FieldElement::one() : FieldElement::zero();
            const auto not_last = FieldElement::one() - last;
            const auto q_valid = at("P_Q_d_valid", row);
            const auto q_invalid = FieldElement::one() - q_valid;
            for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head) {
                const auto prefix = "P_h" + std::to_string(head) + "_";
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };
                auto check_binding = [&](const std::string& name, const std::string& mu_name) {
                    check("d_h", row, prefix + name + " first", first * v("Acc_" + name));
                    check("d_h", row, prefix + name + " step",
                        q_valid * acc_step(vn("Acc_" + name), v("Acc_" + name), v("a_" + name), v("b_" + name)));
                    check("d_h", row, prefix + name + " pad",
                        not_last * q_invalid * (vn("Acc_" + name) - v("Acc_" + name)));
                    check("d_h", row, prefix + name + " a zero", zero_on_unselected(q_valid, v("a_" + name)));
                    check("d_h", row, prefix + name + " b zero", zero_on_unselected(q_valid, v("b_" + name)));
                    check("d_h", row, prefix + name + " last",
                        last * (v("Acc_" + name) - trace.external_evaluations.at(mu_name)));
                };
                check_binding("src", "mu_h" + std::to_string(head) + "_src");
                check_binding("dst", "mu_h" + std::to_string(head) + "_dst");
                check_binding("star", "mu_h" + std::to_string(head) + "_star");
                check_binding("agg_pre", "mu_h" + std::to_string(head) + "_agg_pre");
                check_binding("agg", "mu_h" + std::to_string(head) + "_agg");
            }
        }

        for (std::size_t row = 0; row < context.domains.cat->size; ++row) {
            const auto next = (row + 1) % context.domains.cat->size;
            const auto first = row == 0 ? FieldElement::one() : FieldElement::zero();
            const auto last = row + 1 == context.domains.cat->size ? FieldElement::one() : FieldElement::zero();
            const auto not_last = FieldElement::one() - last;
            const auto q_valid = at("P_Q_cat_valid", row);
            const auto q_invalid = FieldElement::one() - q_valid;
            for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer) {
                const bool is_final_layer = layer + 1 == context.model.hidden_layers.size();
                const auto prefix = gatzk::protocol::hidden_layer_cat_prefix(layer, is_final_layer);
                const auto mu_name = is_final_layer ? std::string("mu_cat") : "mu_cat_l" + std::to_string(layer);
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };
                check("cat", row, prefix + " first", first * v("_Acc"));
                check("cat", row, prefix + " step",
                    q_valid * acc_step(vn("_Acc"), v("_Acc"), v("_a"), v("_b")));
                check("cat", row, prefix + " pad", not_last * q_invalid * (vn("_Acc") - v("_Acc")));
                check("cat", row, prefix + " a zero", zero_on_unselected(q_valid, v("_a")));
                check("cat", row, prefix + " b zero", zero_on_unselected(q_valid, v("_b")));
                check("cat", row, prefix + " last", last * (v("_Acc") - trace.external_evaluations.at(mu_name)));
            }
            const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head) {
                const auto prefix = gatzk::protocol::output_head_prefix(head, legacy_single_output) + "_";
                const auto mu_name = legacy_single_output
                    ? std::string("mu_out_proj")
                    : gatzk::protocol::output_external_eval_name("proj", head, false);
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };
                check("cat", row, prefix + "proj first", first * v("Acc_proj"));
                check("cat", row, prefix + "proj step",
                    q_valid * acc_step(vn("Acc_proj"), v("Acc_proj"), v("a_proj"), v("b_proj")));
                check("cat", row, prefix + "proj pad",
                    not_last * q_invalid * (vn("Acc_proj") - v("Acc_proj")));
                check("cat", row, prefix + "proj a zero", zero_on_unselected(q_valid, v("a_proj")));
                check("cat", row, prefix + "proj b zero", zero_on_unselected(q_valid, v("b_proj")));
                check("cat", row, prefix + "proj last",
                    last * (v("Acc_proj") - trace.external_evaluations.at(mu_name)));
            }
        }

        for (std::size_t row = 0; row < context.domains.c->size; ++row) {
            const auto next = (row + 1) % context.domains.c->size;
            const auto first = row == 0 ? FieldElement::one() : FieldElement::zero();
            const auto last = row + 1 == context.domains.c->size ? FieldElement::one() : FieldElement::zero();
            const auto not_last = FieldElement::one() - last;
            const auto q_valid = at("P_Q_C_valid", row);
            const auto q_invalid = FieldElement::one() - q_valid;
            const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
            FieldElement summed_y_lin = FieldElement::zero();
            FieldElement summed_y = FieldElement::zero();
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head) {
                const auto prefix = gatzk::protocol::output_head_prefix(head, legacy_single_output) + "_";
                const auto mu_src = legacy_single_output
                    ? std::string("mu_out_src")
                    : gatzk::protocol::output_external_eval_name("src", head, false);
                const auto mu_dst = legacy_single_output
                    ? std::string("mu_out_dst")
                    : gatzk::protocol::output_external_eval_name("dst", head, false);
                const auto mu_star = legacy_single_output
                    ? std::string("mu_out_star")
                    : gatzk::protocol::output_external_eval_name("star", head, false);
                const auto mu_y_lin = legacy_single_output
                    ? std::string("mu_Y_lin")
                    : gatzk::protocol::output_external_eval_name("y_lin", head, false);
                const auto mu_y = legacy_single_output
                    ? std::string("mu_out")
                    : gatzk::protocol::output_external_eval_name("y", head, false);
                auto v = [&](const std::string& suffix) { return at(prefix + suffix, row); };
                auto vn = [&](const std::string& suffix) { return at(prefix + suffix, next); };
                auto check_binding = [&](const std::string& name, const std::string& mu_name) {
                    check("C", row, prefix + name + " first", first * v("Acc_" + name));
                    check("C", row, prefix + name + " step",
                        q_valid * acc_step(vn("Acc_" + name), v("Acc_" + name), v("a_" + name), v("b_" + name)));
                    check("C", row, prefix + name + " pad",
                        not_last * q_invalid * (vn("Acc_" + name) - v("Acc_" + name)));
                    check("C", row, prefix + name + " a zero", zero_on_unselected(q_valid, v("a_" + name)));
                    check("C", row, prefix + name + " b zero", zero_on_unselected(q_valid, v("b_" + name)));
                    check("C", row, prefix + name + " last",
                        last * (v("Acc_" + name) - trace.external_evaluations.at(mu_name)));
                };
                check_binding("src", mu_src);
                check_binding("dst", mu_dst);
                check_binding("y", mu_star);
                check("C", row, prefix + "bias relation",
                    first
                        * (trace.external_evaluations.at(mu_y)
                            - trace.external_evaluations.at(mu_y_lin)
                            - bias_fold_vector(
                                context.model.output_layer.heads[head].output_bias_fp,
                                context.local.num_nodes,
                                trace.challenges.at("y_out"))));
                if (!legacy_single_output) {
                    summed_y_lin += trace.external_evaluations.at(mu_y_lin);
                    summed_y += trace.external_evaluations.at(mu_y);
                }
            }
            if (!legacy_single_output) {
                const auto k_out = FieldElement(context.model.output_layer.heads.size());
                check("C", row, "output average pre bias",
                    first * (k_out * trace.external_evaluations.at("mu_Y_lin") - summed_y_lin));
                check("C", row, "output average",
                    first * (k_out * trace.external_evaluations.at("mu_out") - summed_y));
            }
        }

        for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head) {
            const auto suffix = "h" + std::to_string(head);
            const auto prefix = "P_h" + std::to_string(head) + "_";
            check("external", 0, "mu_" + suffix + "_proj",
                trace.external_evaluations.at("mu_" + suffix + "_proj")
                    - eval_trace_poly(prefix + "H_prime", trace.challenges.at("y_proj_h" + std::to_string(head))));
            check("external", 0, "mu_" + suffix + "_src",
                trace.external_evaluations.at("mu_" + suffix + "_src")
                    - eval_trace_poly(prefix + "E_src", trace.challenges.at("y_src_h" + std::to_string(head))));
            check("external", 0, "mu_" + suffix + "_dst",
                trace.external_evaluations.at("mu_" + suffix + "_dst")
                    - eval_trace_poly(prefix + "E_dst", trace.challenges.at("y_dst_h" + std::to_string(head))));
            check("external", 0, "mu_" + suffix + "_star",
                trace.external_evaluations.at("mu_" + suffix + "_star")
                    - eval_trace_poly(prefix + "H_star", trace.challenges.at("y_star_h" + std::to_string(head))));
            check("external", 0, "mu_" + suffix + "_agg_pre",
                trace.external_evaluations.at("mu_" + suffix + "_agg_pre")
                    - eval_trace_poly(prefix + "H_agg_pre_star", trace.challenges.at("y_agg_pre_h" + std::to_string(head))));
            check("external", 0, "mu_" + suffix + "_agg",
                trace.external_evaluations.at("mu_" + suffix + "_agg")
                    - eval_trace_poly(prefix + "H_agg_star", trace.challenges.at("y_agg_h" + std::to_string(head))));
        }
        for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer) {
            const bool is_final_layer = layer + 1 == context.model.hidden_layers.size();
            const auto mu_name = is_final_layer ? std::string("mu_cat") : "mu_cat_l" + std::to_string(layer);
            check("external", 0, mu_name,
                trace.external_evaluations.at(mu_name)
                    - eval_trace_poly(
                        gatzk::protocol::hidden_layer_concat_star_label(layer, is_final_layer),
                        trace.challenges.at(gatzk::protocol::hidden_concat_y_name(layer, is_final_layer))));
        }
        const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head) {
            const auto prefix = gatzk::protocol::output_head_prefix(head, legacy_single_output);
            auto output_check = [&](const std::string& base, const std::string& label_suffix, const std::string& challenge_name) {
                const auto mu_name = legacy_single_output
                    ? (base == "proj" ? std::string("mu_out_proj")
                        : base == "src" ? std::string("mu_out_src")
                        : base == "dst" ? std::string("mu_out_dst")
                        : std::string("mu_out_star"))
                    : gatzk::protocol::output_external_eval_name(base, head, false);
                check("external", 0, mu_name,
                    trace.external_evaluations.at(mu_name)
                        - eval_trace_poly(prefix + label_suffix, trace.challenges.at(challenge_name)));
            };
            output_check(
                "proj",
                "_Y_prime",
                gatzk::protocol::output_challenge_name("y_proj_out", head, legacy_single_output));
            output_check(
                "src",
                "_E_src",
                gatzk::protocol::output_challenge_name("y_src_out", head, legacy_single_output));
            output_check(
                "dst",
                "_E_dst",
                gatzk::protocol::output_challenge_name("y_dst_out", head, legacy_single_output));
            output_check(
                "star",
                "_Y_star",
                gatzk::protocol::output_challenge_name("y_out_star", head, legacy_single_output));
            if (!legacy_single_output) {
                check("external", 0, gatzk::protocol::output_external_eval_name("y_lin", head, false),
                    trace.external_evaluations.at(gatzk::protocol::output_external_eval_name("y_lin", head, false))
                        - eval_trace_poly(gatzk::protocol::output_y_lin_label(head, false), trace.challenges.at("y_out")));
                check("external", 0, gatzk::protocol::output_external_eval_name("y", head, false),
                    trace.external_evaluations.at(gatzk::protocol::output_external_eval_name("y", head, false))
                        - eval_trace_poly(gatzk::protocol::output_y_label(head, false), trace.challenges.at("y_out")));
            }
        }
        check("external", 0, "mu_Y_lin",
            trace.external_evaluations.at("mu_Y_lin") - eval_trace_poly("P_Y_lin", trace.challenges.at("y_out")));
        check("external", 0, "mu_out",
            trace.external_evaluations.at("mu_out") - eval_trace_poly("P_Y", trace.challenges.at("y_out")));

        std::cout << "focus passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
